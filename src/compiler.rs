use std::{cell::RefCell, cmp::max, fmt::Display, mem, rc::Rc};

use crate::{
    ast::{
        ArrayLiteral, BlockStm, BoolLiteral, CallExp, Exp, ExpStm, FuncLiteral, HashLiteral,
        IdentExp, IfExp, IndexExp, InfixExp, InfixOperator, IntLiteral, LetStm, PrefixExp,
        PrefixOperator, Program, ReturnStm, Stm, StringLiteral,
    },
    builtin::BUILTIN_NAMES,
    code::{self, make, Instructions, Opcode},
    object::{self, Object},
    symbol_table::{Symbol, SymbolScope, SymbolTable},
};

pub struct Compiler {
    pub constants: Rc<RefCell<Vec<Rc<Object>>>>,
    pub symbol_table: Rc<RefCell<SymbolTable>>,
    scopes: Vec<CompilationScope>,
    scope_index: usize,
}
impl Compiler {
    pub fn new() -> Self {
        let main_scope = CompilationScope {
            instructions: vec![],
            last_instruction: EmittedInstruction {
                opcode: Opcode::OpConstant,
                position: 0,
            },
            previous_instruction: EmittedInstruction {
                opcode: Opcode::OpConstant,
                position: 0,
            },
        };
        Self {
            constants: Rc::new(RefCell::new(vec![])),
            symbol_table: {
                let st = SymbolTable::new();
                BUILTIN_NAMES.iter().enumerate().for_each(|(index, name)| {
                    st.borrow_mut().define_builtin(index, name);
                });
                st
            },
            scopes: vec![main_scope],
            scope_index: 0,
        }
    }
    pub fn new_with_state(
        symbol_table: Rc<RefCell<SymbolTable>>,
        constants: Rc<RefCell<Vec<Rc<Object>>>>,
    ) -> Self {
        let mut compiler = Self::new();
        compiler.symbol_table = symbol_table;
        compiler.constants = constants;
        compiler
    }
    pub fn compile(&mut self, program: Program) -> Result<(), CompilerError> {
        program
            .stms
            .iter()
            .try_for_each(|stm| self.compile_stm(stm))
    }
    pub fn bytecode(mut self) -> Bytecode {
        Bytecode {
            instructions: mem::replace(&mut self.scopes[self.scope_index].instructions, vec![]),
            constants: self.constants,
        }
    }
    fn compile_stm(&mut self, stm: &Stm) -> Result<(), CompilerError> {
        match stm {
            Stm::ExpStm(ExpStm { exp }) => {
                self.compile_exp(&exp)?;
                self.emit(Opcode::OpPop, vec![]);
            }
            Stm::LetStm(LetStm { name, value }) => {
                let symbol = self.symbol_table.borrow_mut().define(name.as_str());
                self.compile_exp(&value)?;
                match symbol.scope {
                    SymbolScope::Global => self.emit(Opcode::OpSetGlobal, vec![symbol.index]),
                    _ => self.emit(Opcode::OpSetLocal, vec![symbol.index]),
                };
            }
            Stm::ReturnStm(ReturnStm { return_value }) => {
                self.compile_exp(return_value)?;
                self.emit(Opcode::OpReturnValue, vec![]);
            }
        }
        Ok(())
    }
    fn compile_exp(&mut self, exp: &Exp) -> Result<(), CompilerError> {
        match exp {
            Exp::IntLiteral(IntLiteral { value }) => {
                let integer = object::new_int(*value);
                let pos = self.add_constant(integer);
                self.emit(Opcode::OpConstant, vec![pos]);
            }
            Exp::InfixExp(InfixExp { left, op, right }) => {
                if op == &InfixOperator::Lt {
                    self.compile_exp(right)?;
                    self.compile_exp(left)?;
                } else {
                    self.compile_exp(left)?;
                    self.compile_exp(right)?;
                }
                match op {
                    InfixOperator::Add => self.emit(Opcode::OpAdd, vec![]),
                    InfixOperator::Sub => self.emit(Opcode::OpSub, vec![]),
                    InfixOperator::Mul => self.emit(Opcode::OpMul, vec![]),
                    InfixOperator::Div => self.emit(Opcode::OpDiv, vec![]),
                    InfixOperator::Gt => self.emit(Opcode::OpGreaterThan, vec![]),
                    InfixOperator::Eq => self.emit(Opcode::OpEqual, vec![]),
                    InfixOperator::NotEq => self.emit(Opcode::OpNotEqual, vec![]),
                    InfixOperator::Lt => self.emit(Opcode::OpGreaterThan, vec![]),
                };
            }
            Exp::BoolLiteral(BoolLiteral { value }) => {
                if *value {
                    self.emit(Opcode::OpTrue, vec![]);
                } else {
                    self.emit(Opcode::OpFalse, vec![]);
                }
            }
            Exp::PrefixExp(PrefixExp { op, right }) => {
                self.compile_exp(right)?;
                match op {
                    PrefixOperator::Bang => {
                        self.emit(Opcode::OpBang, vec![]);
                    }
                    PrefixOperator::Minus => {
                        self.emit(Opcode::OpMinus, vec![]);
                    }
                }
            }
            Exp::IfExp(IfExp {
                cond,
                conseq,
                alter,
            }) => {
                self.compile_exp(cond)?;
                let jump_not_truthy_pos = self.emit(Opcode::OpJumpNotTruthy, vec![9999]);
                self.compile_block_stm(conseq)?;
                if self.last_instruction_is(Opcode::OpPop) {
                    self.remove_last_pop();
                }
                let jump_pos = self.emit(Opcode::OpJump, vec![9999]);
                let after_consequence_pos = self.scopes[self.scope_index].instructions.len() as u16;
                self.change_operand(jump_not_truthy_pos, after_consequence_pos)?;
                match alter {
                    Some(alter) => {
                        self.compile_block_stm(alter)?;
                        if self.last_instruction_is(Opcode::OpPop) {
                            self.remove_last_pop();
                        }
                    }
                    _ => {
                        self.emit(Opcode::OpNull, vec![]);
                    }
                }
                let after_alternative_pos = self.scopes[self.scope_index].instructions.len() as u16;
                self.change_operand(jump_pos, after_alternative_pos)?;
            }
            Exp::IdentExp(IdentExp { value }) => {
                let symbol = match self.symbol_table.borrow_mut().resolve(value.as_str()) {
                    Some(symbol) => symbol,
                    _ => return Err(CompilerError::UndefinedVariable(value.to_owned())),
                };
                self.load_symbol(symbol);
            }
            Exp::StringLiteral(StringLiteral { value }) => {
                let pos = self.add_constant(object::new_string(value.as_str()));
                self.emit(Opcode::OpConstant, vec![pos]);
            }
            Exp::ArrayLiteral(ArrayLiteral { elems }) => {
                elems.iter().try_for_each(|elem| self.compile_exp(elem))?;
                self.emit(Opcode::OpArray, vec![elems.len() as u16]);
            }
            Exp::HashLiteral(HashLiteral { pairs }) => {
                let mut pairs = pairs
                    .iter()
                    .map(|(key, value)| (key, value))
                    .collect::<Vec<(&Exp, &Exp)>>();

                #[cfg(test)]
                {
                    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                }

                let len = (pairs.len() * 2) as u16;
                for pair in pairs {
                    self.compile_exp(pair.0)?;
                    self.compile_exp(pair.1)?;
                }
                self.emit(Opcode::OpHash, vec![len]);
            }
            Exp::IndexExp(IndexExp { left, index }) => {
                self.compile_exp(&*left)?;
                self.compile_exp(&*index)?;
                self.emit(Opcode::OpIndex, vec![]);
            }
            Exp::FuncLiteral(FuncLiteral { name, params, body }) => {
                self.enter_scope();
                match name {
                    Some(name) => {
                        self.symbol_table
                            .borrow_mut()
                            .define_function_name(name.as_str());
                    }
                    _ => {}
                }
                params.iter().for_each(|param| {
                    self.symbol_table.borrow_mut().define(param.as_str());
                });
                self.compile_block_stm(body)?;
                if self.last_instruction_is(Opcode::OpPop) {
                    self.replace_last_pop_with_return()
                }
                if !self.last_instruction_is(Opcode::OpReturnValue) {
                    self.emit(Opcode::OpReturn, vec![]);
                }

                let free_symbols = self.symbol_table.borrow().free_symbols.clone();
                let num_locals = self.symbol_table.borrow().num_defintions;
                let instructions = self.leave_scope();
                for s in &free_symbols {
                    self.load_symbol(Rc::clone(s));
                }

                let compiled_fn =
                    object::new_func(vec![instructions], num_locals, params.len() as u16);
                let func_index = self.add_constant(compiled_fn);
                self.emit(
                    Opcode::OpClosure,
                    vec![func_index, free_symbols.len() as u16],
                );
            }
            Exp::CallExp(CallExp { func, args }) => {
                self.compile_exp(func)?;
                args.iter().try_for_each(|arg| self.compile_exp(arg))?;
                self.emit(Opcode::OpCall, vec![args.len() as u16]);
            }
        }
        Ok(())
    }

    fn load_symbol(&mut self, symbol: Rc<Symbol>) {
        match symbol.scope {
            SymbolScope::Global => self.emit(Opcode::OpGetGlobal, vec![symbol.index]),
            SymbolScope::Local => self.emit(Opcode::OpGetLocal, vec![symbol.index]),
            SymbolScope::Builtin => self.emit(Opcode::OpGetBuiltin, vec![symbol.index]),
            SymbolScope::Free => self.emit(Opcode::OpGetFree, vec![symbol.index]),
            SymbolScope::Function => self.emit(Opcode::OpCurrentClosure, vec![]),
        };
    }
    fn add_constant(&mut self, obj: Rc<Object>) -> u16 {
        self.constants.borrow_mut().push(obj);
        (self.constants.borrow().len() - 1) as u16
    }
    fn emit(&mut self, opcode: Opcode, operands: Vec<u16>) -> u16 {
        let ins = make(opcode, operands);
        let pos = self.add_instruction(ins);
        self.set_last_instruction(opcode, pos);
        pos
    }
    fn add_instruction(&mut self, ins: Instructions) -> u16 {
        let pos_new_instruction = self.scopes[self.scope_index].instructions.len();
        self.scopes[self.scope_index]
            .instructions
            .append(&mut ins.clone());
        pos_new_instruction as u16
    }
    fn compile_block_stm(&mut self, block_stm: &BlockStm) -> Result<(), CompilerError> {
        block_stm
            .stms
            .iter()
            .try_for_each(|stm| self.compile_stm(stm))
    }
    fn set_last_instruction(&mut self, op: Opcode, pos: u16) {
        self.scopes[self.scope_index].previous_instruction = mem::replace(
            &mut self.scopes[self.scope_index].last_instruction,
            EmittedInstruction {
                opcode: op,
                position: pos,
            },
        );
    }
    fn last_instruction_is(&mut self, op: Opcode) -> bool {
        if self.scopes[self.scope_index].instructions.len() == 0 {
            return false;
        }
        self.scopes[self.scope_index].last_instruction.opcode == op
    }
    fn remove_last_pop(&mut self) {
        let len = self.scopes[self.scope_index].last_instruction.position as usize;
        self.scopes[self.scope_index].instructions.truncate(len);
        self.scopes[self.scope_index].last_instruction = mem::replace(
            &mut self.scopes[self.scope_index].previous_instruction,
            EmittedInstruction {
                opcode: Opcode::OpConstant,
                position: 0,
            },
        );
    }
    fn replace_instruction(&mut self, pos: u16, new_instruction: Vec<u8>) {
        let start = pos as usize;
        let end = start + new_instruction.len();
        let max_len = self.scopes[self.scope_index].instructions.len();
        self.scopes[self.scope_index]
            .instructions
            .resize(max(end, max_len), 0);

        let slice = &mut self.scopes[self.scope_index].instructions[start..end];
        slice.copy_from_slice(new_instruction.as_slice())
    }
    fn change_operand(&mut self, op_pos: u16, operand: u16) -> Result<(), CompilerError> {
        let opcode = self.scopes[self.scope_index].instructions[op_pos as usize].try_into()?;
        let new_instruction = make(opcode, vec![operand]);
        self.replace_instruction(op_pos, new_instruction);
        Ok(())
    }
    fn enter_scope(&mut self) {
        let scope = CompilationScope {
            instructions: vec![],
            last_instruction: EmittedInstruction {
                opcode: Opcode::OpConstant,
                position: 0,
            },
            previous_instruction: EmittedInstruction {
                opcode: Opcode::OpConstant,
                position: 0,
            },
        };
        self.scopes.push(scope);
        self.scope_index += 1;
        self.symbol_table = SymbolTable::new_enclosed(Rc::clone(&self.symbol_table));
    }
    fn leave_scope(&mut self) -> Instructions {
        let instructions = mem::replace(&mut self.scopes[self.scope_index].instructions, vec![]);

        self.scopes.truncate(self.scopes.len() - 1);
        self.scope_index -= 1;
        let outer = Rc::clone(&self.symbol_table.borrow().outer.as_ref().unwrap());
        self.symbol_table = outer;
        instructions
    }
    fn replace_last_pop_with_return(&mut self) {
        let last_pos = self.scopes[self.scope_index].last_instruction.position;
        self.replace_instruction(last_pos, make(Opcode::OpReturnValue, vec![]));

        self.scopes[self.scope_index].last_instruction.opcode = Opcode::OpReturnValue
    }
}
pub struct Bytecode {
    pub instructions: Instructions,
    pub constants: Rc<RefCell<Vec<Rc<Object>>>>,
}

struct EmittedInstruction {
    opcode: Opcode,
    position: u16,
}

#[derive(Debug)]
pub enum CompilerError {
    CodeError(code::CodeError),
    UndefinedVariable(String),
}
impl Display for CompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CodeError(e) => write!(f, "{}", e),
            Self::UndefinedVariable(var) => write!(f, "undefined variable {}", var),
        }
    }
}
impl From<code::CodeError> for CompilerError {
    fn from(e: code::CodeError) -> Self {
        Self::CodeError(e)
    }
}

pub struct CompilationScope {
    instructions: Instructions,
    last_instruction: EmittedInstruction,
    previous_instruction: EmittedInstruction,
}

#[cfg(test)]
mod tests {

    use std::vec;

    use crate::{
        code::{make, Opcode},
        object::new_int,
        test_util::parse,
    };

    use super::*;

    struct CompilerTestCase {
        input: &'static str,
        expected_constants: Rc<RefCell<Vec<Rc<Object>>>>,
        expected_instructions: Vec<Instructions>,
    }
    impl CompilerTestCase {
        fn new(
            input: &'static str,
            expected_constants: Vec<Rc<Object>>,
            expected_instructions: Vec<Instructions>,
        ) -> Self {
            Self {
                input,
                expected_constants: Rc::new(RefCell::new(expected_constants)),
                expected_instructions,
            }
        }
    }

    fn run_compiler_tests(tests: &[CompilerTestCase]) {
        for tt in tests {
            let program = parse(tt.input);
            let mut compiler = Compiler::new();
            compiler.compile(program).unwrap();
            assert_eq!(
                compiler.scopes[compiler.scope_index].instructions,
                tt.expected_instructions
                    .iter()
                    .flatten()
                    .cloned()
                    .collect::<Instructions>()
            );
            assert_eq!(compiler.constants, tt.expected_constants);
        }
    }

    #[test]
    fn test_integer_arithmetic() {
        let tests = [
            CompilerTestCase::new(
                "1 + 2",
                vec![object::new_int(1), object::new_int(2)],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpAdd, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "1; 2",
                vec![object::new_int(1), object::new_int(2)],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpPop, vec![]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "1 - 2",
                vec![object::new_int(1), object::new_int(2)],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpSub, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "1 * 2",
                vec![object::new_int(1), object::new_int(2)],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpMul, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "2 / 1",
                vec![object::new_int(2), object::new_int(1)],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpDiv, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "-1",
                vec![object::new_int(1)],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpMinus, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
        ];

        run_compiler_tests(&tests)
    }
    #[test]
    fn test_boolean_exp() {
        let tests = [
            CompilerTestCase::new(
                "true",
                vec![],
                vec![make(Opcode::OpTrue, vec![]), make(Opcode::OpPop, vec![])],
            ),
            CompilerTestCase::new(
                "false",
                vec![],
                vec![make(Opcode::OpFalse, vec![]), make(Opcode::OpPop, vec![])],
            ),
            CompilerTestCase::new(
                "1 > 2",
                vec![object::new_int(1), object::new_int(2)],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpGreaterThan, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "1 < 2",
                vec![object::new_int(2), object::new_int(1)],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpGreaterThan, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "1 == 2",
                vec![object::new_int(1), object::new_int(2)],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpEqual, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "1 != 2",
                vec![object::new_int(1), object::new_int(2)],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpNotEqual, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "true == false",
                vec![],
                vec![
                    make(Opcode::OpTrue, vec![]),
                    make(Opcode::OpFalse, vec![]),
                    make(Opcode::OpEqual, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "true != false",
                vec![],
                vec![
                    make(Opcode::OpTrue, vec![]),
                    make(Opcode::OpFalse, vec![]),
                    make(Opcode::OpNotEqual, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "!true",
                vec![],
                vec![
                    make(Opcode::OpTrue, vec![]),
                    make(Opcode::OpBang, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
        ];

        run_compiler_tests(&tests)
    }
    #[test]
    fn test_conditionals() {
        let tests = [
            CompilerTestCase::new(
                "if (true) { 10 }; 3333;",
                vec![object::new_int(10), object::new_int(3333)],
                vec![
                    make(Opcode::OpTrue, vec![]),
                    make(Opcode::OpJumpNotTruthy, vec![10]),
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpJump, vec![11]),
                    make(Opcode::OpNull, vec![]),
                    make(Opcode::OpPop, vec![]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "if (true) { 10 } else { 20 }; 3333;",
                vec![
                    object::new_int(10),
                    object::new_int(20),
                    object::new_int(3333),
                ],
                vec![
                    make(Opcode::OpTrue, vec![]),
                    make(Opcode::OpJumpNotTruthy, vec![10]),
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpJump, vec![13]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpPop, vec![]),
                    make(Opcode::OpConstant, vec![2]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
        ];
        run_compiler_tests(&tests)
    }
    #[test]
    fn test_global_let_stm() {
        let tests = [
            CompilerTestCase::new(
                r#"
let one = 1;
let two = 2;
"#,
                vec![new_int(1), new_int(2)],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpSetGlobal, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpSetGlobal, vec![1]),
                ],
            ),
            CompilerTestCase::new(
                r#"
let one = 1;
one;
"#,
                vec![new_int(1)],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpSetGlobal, vec![0]),
                    make(Opcode::OpGetGlobal, vec![0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                r#"
let one = 1;
let two = one;
two;
"#,
                vec![new_int(1)],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpSetGlobal, vec![0]),
                    make(Opcode::OpGetGlobal, vec![0]),
                    make(Opcode::OpSetGlobal, vec![1]),
                    make(Opcode::OpGetGlobal, vec![1]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
        ];
        run_compiler_tests(&tests)
    }
    #[test]
    fn test_string_exp() {
        let tests = [
            CompilerTestCase::new(
                r#""monkey""#,
                vec![object::new_string("monkey")],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                r#""mon" + "key""#,
                vec![object::new_string("mon"), object::new_string("key")],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpAdd, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
        ];
        run_compiler_tests(&tests)
    }
    #[test]
    fn test_array_literals() {
        let tests = [
            CompilerTestCase::new(
                "[]",
                vec![],
                vec![make(Opcode::OpArray, vec![0]), make(Opcode::OpPop, vec![])],
            ),
            CompilerTestCase::new(
                "[1, 2, 3]",
                vec![object::new_int(1), object::new_int(2), object::new_int(3)],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpConstant, vec![2]),
                    make(Opcode::OpArray, vec![3]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "[1 + 2, 3 - 4, 5 * 6]",
                vec![
                    object::new_int(1),
                    object::new_int(2),
                    object::new_int(3),
                    object::new_int(4),
                    object::new_int(5),
                    object::new_int(6),
                ],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpAdd, vec![]),
                    make(Opcode::OpConstant, vec![2]),
                    make(Opcode::OpConstant, vec![3]),
                    make(Opcode::OpSub, vec![]),
                    make(Opcode::OpConstant, vec![4]),
                    make(Opcode::OpConstant, vec![5]),
                    make(Opcode::OpMul, vec![]),
                    make(Opcode::OpArray, vec![3]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
        ];
        run_compiler_tests(&tests)
    }
    #[test]
    fn test_hash_literals() {
        let tests = [
            CompilerTestCase::new(
                "{}",
                vec![],
                vec![make(Opcode::OpHash, vec![0]), make(Opcode::OpPop, vec![])],
            ),
            CompilerTestCase::new(
                "{1:2, 3:4, 5:6}",
                vec![
                    object::new_int(1),
                    object::new_int(2),
                    object::new_int(3),
                    object::new_int(4),
                    object::new_int(5),
                    object::new_int(6),
                ],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpConstant, vec![2]),
                    make(Opcode::OpConstant, vec![3]),
                    make(Opcode::OpConstant, vec![4]),
                    make(Opcode::OpConstant, vec![5]),
                    make(Opcode::OpHash, vec![6]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "{1:2+ 3,4: 5*6}",
                vec![
                    object::new_int(1),
                    object::new_int(2),
                    object::new_int(3),
                    object::new_int(4),
                    object::new_int(5),
                    object::new_int(6),
                ],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpConstant, vec![2]),
                    make(Opcode::OpAdd, vec![]),
                    make(Opcode::OpConstant, vec![3]),
                    make(Opcode::OpConstant, vec![4]),
                    make(Opcode::OpConstant, vec![5]),
                    make(Opcode::OpMul, vec![]),
                    make(Opcode::OpHash, vec![4]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
        ];
        run_compiler_tests(&tests)
    }
    #[test]
    fn test_index_exp() {
        let tests = [
            CompilerTestCase::new(
                "[1, 2, 3][1 + 1]",
                vec![
                    object::new_int(1),
                    object::new_int(2),
                    object::new_int(3),
                    object::new_int(1),
                    object::new_int(1),
                ],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpConstant, vec![2]),
                    make(Opcode::OpArray, vec![3]),
                    make(Opcode::OpConstant, vec![3]),
                    make(Opcode::OpConstant, vec![4]),
                    make(Opcode::OpAdd, vec![]),
                    make(Opcode::OpIndex, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "{1: 2}[2 - 1]",
                vec![
                    object::new_int(1),
                    object::new_int(2),
                    object::new_int(2),
                    object::new_int(1),
                ],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpHash, vec![2]),
                    make(Opcode::OpConstant, vec![2]),
                    make(Opcode::OpConstant, vec![3]),
                    make(Opcode::OpSub, vec![]),
                    make(Opcode::OpIndex, vec![]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
        ];
        run_compiler_tests(&tests)
    }
    #[test]
    fn test_functions() {
        let tests = [
            CompilerTestCase::new(
                "fn() { return 5 + 10 }",
                vec![
                    object::new_int(5),
                    object::new_int(10),
                    object::new_func(
                        vec![
                            make(Opcode::OpConstant, vec![0]),
                            make(Opcode::OpConstant, vec![1]),
                            make(Opcode::OpAdd, vec![]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        0,
                        0,
                    ),
                ],
                vec![
                    make(Opcode::OpClosure, vec![2, 0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "fn() { 5 + 10 }",
                vec![
                    object::new_int(5),
                    object::new_int(10),
                    object::new_func(
                        vec![
                            make(Opcode::OpConstant, vec![0]),
                            make(Opcode::OpConstant, vec![1]),
                            make(Opcode::OpAdd, vec![]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        0,
                        0,
                    ),
                ],
                vec![
                    make(Opcode::OpClosure, vec![2, 0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                "fn() { 1; 2 }",
                vec![
                    object::new_int(1),
                    object::new_int(2),
                    object::new_func(
                        vec![
                            make(Opcode::OpConstant, vec![0]),
                            make(Opcode::OpPop, vec![]),
                            make(Opcode::OpConstant, vec![1]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        0,
                        0,
                    ),
                ],
                vec![
                    make(Opcode::OpClosure, vec![2, 0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
        ];
        run_compiler_tests(&tests)
    }
    #[test]
    fn test_compiler_scopes() {
        let mut compiler = Compiler::new();
        assert_eq!(compiler.scope_index, 0);

        let global_symbol_table = Rc::clone(&compiler.symbol_table);

        compiler.emit(Opcode::OpMul, vec![]);
        compiler.enter_scope();
        assert_eq!(compiler.scope_index, 1);

        compiler.emit(Opcode::OpSub, vec![]);
        assert_eq!(compiler.scopes[compiler.scope_index].instructions.len(), 1);

        let last = &compiler.scopes[compiler.scope_index].last_instruction;
        assert_eq!(last.opcode, Opcode::OpSub);

        assert_eq!(
            compiler.symbol_table.borrow().outer,
            Some(Rc::clone(&global_symbol_table))
        );

        compiler.leave_scope();
        assert_eq!(compiler.scope_index, 0);

        assert_eq!(compiler.symbol_table, global_symbol_table);
        assert_eq!(compiler.symbol_table.borrow().outer, None);

        compiler.emit(Opcode::OpAdd, vec![]);
        assert_eq!(compiler.scopes[compiler.scope_index].instructions.len(), 2);

        let last = &compiler.scopes[compiler.scope_index].last_instruction;
        assert_eq!(last.opcode, Opcode::OpAdd);

        let previous = &compiler.scopes[compiler.scope_index].previous_instruction;
        assert_eq!(previous.opcode, Opcode::OpMul);
    }
    #[test]
    fn test_functions_without_return_value() {
        let tests = [CompilerTestCase::new(
            "fn() {}",
            vec![object::new_func(vec![make(Opcode::OpReturn, vec![])], 0, 0)],
            vec![
                make(Opcode::OpClosure, vec![0, 0]),
                make(Opcode::OpPop, vec![]),
            ],
        )];
        run_compiler_tests(&tests)
    }
    #[test]
    fn test_function_calls() {
        let tests = [
            CompilerTestCase::new(
                "fn() {24}();",
                vec![
                    object::new_int(24),
                    object::new_func(
                        vec![
                            make(Opcode::OpConstant, vec![0]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        0,
                        0,
                    ),
                ],
                vec![
                    make(Opcode::OpClosure, vec![1, 0]),
                    make(Opcode::OpCall, vec![0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                r#"
            let noArg = fn() {24};
            noArg();
            "#,
                vec![
                    object::new_int(24),
                    object::new_func(
                        vec![
                            make(Opcode::OpConstant, vec![0]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        0,
                        0,
                    ),
                ],
                vec![
                    make(Opcode::OpClosure, vec![1, 0]),
                    make(Opcode::OpSetGlobal, vec![0]),
                    make(Opcode::OpGetGlobal, vec![0]),
                    make(Opcode::OpCall, vec![0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                r#"
                let oneArg = fn(a) { };
                oneArg(24);
            "#,
                vec![
                    object::new_func(vec![make(Opcode::OpReturn, vec![])], 1, 1),
                    object::new_int(24),
                ],
                vec![
                    make(Opcode::OpClosure, vec![0, 0]),
                    make(Opcode::OpSetGlobal, vec![0]),
                    make(Opcode::OpGetGlobal, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpCall, vec![1]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                r#"
                let manyArg = fn(a, b, c) { };
                manyArg(24, 25, 26);
            "#,
                vec![
                    object::new_func(vec![make(Opcode::OpReturn, vec![])], 3, 3),
                    object::new_int(24),
                    object::new_int(25),
                    object::new_int(26),
                ],
                vec![
                    make(Opcode::OpClosure, vec![0, 0]),
                    make(Opcode::OpSetGlobal, vec![0]),
                    make(Opcode::OpGetGlobal, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpConstant, vec![2]),
                    make(Opcode::OpConstant, vec![3]),
                    make(Opcode::OpCall, vec![3]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                r#"
                let oneArg = fn(a) { a };
                oneArg(24);
            "#,
                vec![
                    object::new_func(
                        vec![
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        1,
                        1,
                    ),
                    object::new_int(24),
                ],
                vec![
                    make(Opcode::OpClosure, vec![0, 0]),
                    make(Opcode::OpSetGlobal, vec![0]),
                    make(Opcode::OpGetGlobal, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpCall, vec![1]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                r#"
                let manyArg = fn(a, b, c) { a; b; c };
                manyArg(24, 25, 26);
            "#,
                vec![
                    object::new_func(
                        vec![
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpPop, vec![]),
                            make(Opcode::OpGetLocal, vec![1]),
                            make(Opcode::OpPop, vec![]),
                            make(Opcode::OpGetLocal, vec![2]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        3,
                        3,
                    ),
                    object::new_int(24),
                    object::new_int(25),
                    object::new_int(26),
                ],
                vec![
                    make(Opcode::OpClosure, vec![0, 0]),
                    make(Opcode::OpSetGlobal, vec![0]),
                    make(Opcode::OpGetGlobal, vec![0]),
                    make(Opcode::OpConstant, vec![1]),
                    make(Opcode::OpConstant, vec![2]),
                    make(Opcode::OpConstant, vec![3]),
                    make(Opcode::OpCall, vec![3]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
        ];
        run_compiler_tests(&tests)
    }
    #[test]
    fn test_let_stm_scopes() {
        let tests = [
            CompilerTestCase::new(
                r#"
            let num = 55;
            fn() { num }
            "#,
                vec![
                    object::new_int(55),
                    object::new_func(
                        vec![
                            make(Opcode::OpGetGlobal, vec![0]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        0,
                        0,
                    ),
                ],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpSetGlobal, vec![0]),
                    make(Opcode::OpClosure, vec![1, 0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                r#"
            fn() {
                let num = 55; 
                num 
            }
            "#,
                vec![
                    object::new_int(55),
                    object::new_func(
                        vec![
                            make(Opcode::OpConstant, vec![0]),
                            make(Opcode::OpSetLocal, vec![0]),
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        1,
                        0,
                    ),
                ],
                vec![
                    make(Opcode::OpClosure, vec![1, 0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                r#"
            fn() { 
                let a = 55; 
                let b = 77; 
                a + b 
            }
            "#,
                vec![
                    object::new_int(55),
                    object::new_int(77),
                    object::new_func(
                        vec![
                            make(Opcode::OpConstant, vec![0]),
                            make(Opcode::OpSetLocal, vec![0]),
                            make(Opcode::OpConstant, vec![1]),
                            make(Opcode::OpSetLocal, vec![1]),
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpGetLocal, vec![1]),
                            make(Opcode::OpAdd, vec![]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        2,
                        0,
                    ),
                ],
                vec![
                    make(Opcode::OpClosure, vec![2, 0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
        ];
        run_compiler_tests(&tests)
    }
    #[test]
    fn test_builtins() {
        let tests = [
            CompilerTestCase::new(
                r#"
            len([]);
            push([], 1);
            "#,
                vec![object::new_int(1)],
                vec![
                    make(Opcode::OpGetBuiltin, vec![0]),
                    make(Opcode::OpArray, vec![0]),
                    make(Opcode::OpCall, vec![1]),
                    make(Opcode::OpPop, vec![]),
                    make(Opcode::OpGetBuiltin, vec![5]),
                    make(Opcode::OpArray, vec![0]),
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpCall, vec![2]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                r#"
            fn() { len([]) }
            "#,
                vec![object::new_func(
                    vec![
                        make(Opcode::OpGetBuiltin, vec![0]),
                        make(Opcode::OpArray, vec![0]),
                        make(Opcode::OpCall, vec![1]),
                        make(Opcode::OpReturnValue, vec![]),
                    ],
                    0,
                    0,
                )],
                vec![
                    make(Opcode::OpClosure, vec![0, 0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
        ];
        run_compiler_tests(&tests)
    }
    #[test]
    fn test_closures() {
        let tests = [
            CompilerTestCase::new(
                r#"
            fn(a) {
                fn(b) {
                    a + b
                }
            }
            "#,
                vec![
                    object::new_func(
                        vec![
                            make(Opcode::OpGetFree, vec![0]),
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpAdd, vec![]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        1,
                        1,
                    ),
                    object::new_func(
                        vec![
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpClosure, vec![0, 1]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        1,
                        1,
                    ),
                ],
                vec![
                    make(Opcode::OpClosure, vec![1, 0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                r#"
        fn(a) {
            fn(b) {
                fn(c) {
                    a + b + c
                }
            }
        };
        "#,
                vec![
                    object::new_func(
                        vec![
                            make(Opcode::OpGetFree, vec![0]),
                            make(Opcode::OpGetFree, vec![1]),
                            make(Opcode::OpAdd, vec![]),
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpAdd, vec![]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        1,
                        1,
                    ),
                    object::new_func(
                        vec![
                            make(Opcode::OpGetFree, vec![0]),
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpClosure, vec![0, 2]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        1,
                        1,
                    ),
                    object::new_func(
                        vec![
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpClosure, vec![1, 1]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        1,
                        1,
                    ),
                ],
                vec![
                    make(Opcode::OpClosure, vec![2, 0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                r#"
            let global = 55;
            fn() {
                let a = 66;
                fn() {
                    let b = 77;
                    fn() {
                        let c = 88;
                        global + a + b + c;
                    }
                }
            }
            "#,
                vec![
                    object::new_int(55),
                    object::new_int(66),
                    object::new_int(77),
                    object::new_int(88),
                    object::new_func(
                        vec![
                            make(Opcode::OpConstant, vec![3]),
                            make(Opcode::OpSetLocal, vec![0]),
                            make(Opcode::OpGetGlobal, vec![0]),
                            make(Opcode::OpGetFree, vec![0]),
                            make(Opcode::OpAdd, vec![]),
                            make(Opcode::OpGetFree, vec![1]),
                            make(Opcode::OpAdd, vec![]),
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpAdd, vec![]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        1,
                        0,
                    ),
                    object::new_func(
                        vec![
                            make(Opcode::OpConstant, vec![2]),
                            make(Opcode::OpSetLocal, vec![0]),
                            make(Opcode::OpGetFree, vec![0]),
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpClosure, vec![4, 2]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        1,
                        0,
                    ),
                    object::new_func(
                        vec![
                            make(Opcode::OpConstant, vec![1]),
                            make(Opcode::OpSetLocal, vec![0]),
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpClosure, vec![5, 1]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        1,
                        0,
                    ),
                ],
                vec![
                    make(Opcode::OpConstant, vec![0]),
                    make(Opcode::OpSetGlobal, vec![0]),
                    make(Opcode::OpClosure, vec![6, 0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
        ];
        run_compiler_tests(&tests)
    }
    #[test]
    fn test_recursive_functions() {
        let tests = [
            CompilerTestCase::new(
                r#"
            let countDown = fn(x) { countDown(x - 1); };
            countDown(1);
            "#,
                vec![
                    object::new_int(1),
                    object::new_func(
                        vec![
                            make(Opcode::OpCurrentClosure, vec![]),
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpConstant, vec![0]),
                            make(Opcode::OpSub, vec![]),
                            make(Opcode::OpCall, vec![1]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        1,
                        1,
                    ),
                    object::new_int(1),
                ],
                vec![
                    make(Opcode::OpClosure, vec![1, 0]),
                    make(Opcode::OpSetGlobal, vec![0]),
                    make(Opcode::OpGetGlobal, vec![0]),
                    make(Opcode::OpConstant, vec![2]),
                    make(Opcode::OpCall, vec![1]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
            CompilerTestCase::new(
                r#"
            let wrapper = fn() {
                let countDown = fn(x) { countDown(x - 1); };
                countDown(1);
            };
            wrapper();
            "#,
                vec![
                    object::new_int(1),
                    object::new_func(
                        vec![
                            make(Opcode::OpCurrentClosure, vec![]),
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpConstant, vec![0]),
                            make(Opcode::OpSub, vec![]),
                            make(Opcode::OpCall, vec![1]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        1,
                        1,
                    ),
                    object::new_int(1),
                    object::new_func(
                        vec![
                            make(Opcode::OpClosure, vec![1, 0]),
                            make(Opcode::OpSetLocal, vec![0]),
                            make(Opcode::OpGetLocal, vec![0]),
                            make(Opcode::OpConstant, vec![2]),
                            make(Opcode::OpCall, vec![1]),
                            make(Opcode::OpReturnValue, vec![]),
                        ],
                        1,
                        0,
                    ),
                ],
                vec![
                    make(Opcode::OpClosure, vec![3, 0]),
                    make(Opcode::OpSetGlobal, vec![0]),
                    make(Opcode::OpGetGlobal, vec![0]),
                    make(Opcode::OpCall, vec![0]),
                    make(Opcode::OpPop, vec![]),
                ],
            ),
        ];
        run_compiler_tests(&tests)
    }
}
