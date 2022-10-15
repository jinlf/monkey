use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::Display,
    mem::{self, MaybeUninit},
    rc::Rc,
};

use crate::{
    builtin::{builtins, Builtin},
    code::{self, read_u16, Opcode},
    compiler::Bytecode,
    frame::Frame,
    object::{
        self, Array, Boolean, Closure, HashKey, HashObj, HashPair, Integer, MayHasHashKey, Object,
        StringObj,
    },
};

const STACK_SIZE: usize = 2048;
const GLOBAL_SIZE: usize = 65536;
const MAX_FRAMES: usize = 1024;

pub struct Vm {
    constants: Rc<RefCell<Vec<Rc<Object>>>>,
    stack: [Rc<Object>; STACK_SIZE],
    sp: u16,
    globals: Rc<RefCell<Vec<Rc<Object>>>>,
    frames: [Frame; MAX_FRAMES],
    frame_index: usize,
}
impl Vm {
    pub fn new(bytecode: Bytecode) -> Self {
        let main_fn = object::CompiledFunction {
            instructions: bytecode.instructions,
            num_locals: 0,
            num_params: 0,
        };
        let main_closure = Closure {
            func: main_fn,
            free: vec![],
        };
        let main_frame = Frame::new(main_closure, 0);
        Self {
            constants: bytecode.constants,
            stack: {
                let mut data: [MaybeUninit<Rc<Object>>; STACK_SIZE] = MaybeUninit::uninit_array();
                for i in 0..STACK_SIZE {
                    data[i].write(Rc::new(Object::Null));
                }
                unsafe { MaybeUninit::array_assume_init(data) }
            },
            sp: 0,
            globals: Rc::new(RefCell::new(Vec::with_capacity(GLOBAL_SIZE))),
            frames: {
                let mut data: [MaybeUninit<Frame>; MAX_FRAMES] = MaybeUninit::uninit_array();
                data[0].write(main_frame);
                for i in 1..MAX_FRAMES {
                    data[i].write(Frame::default());
                }
                unsafe { MaybeUninit::array_assume_init(data) }
            },
            frame_index: 0,
        }
    }
    pub fn new_with_global_store(
        bytecode: Bytecode,
        globals: Rc<RefCell<Vec<Rc<Object>>>>,
    ) -> Self {
        let mut vm = Self::new(bytecode);
        vm.globals = globals;
        vm
    }
    pub fn last_popped_stack_elem(&self) -> Rc<Object> {
        Rc::clone(&self.stack[self.sp as usize])
    }
    fn current_frame(&mut self) -> &mut Frame {
        &mut self.frames[self.frame_index]
    }
    fn push_frame(&mut self, f: Frame) {
        self.frame_index += 1;
        self.frames[self.frame_index] = f;
    }
    fn pop_frame(&mut self) -> Frame {
        let f = mem::replace(&mut self.frames[self.frame_index], Frame::default());
        self.frame_index -= 1;
        f
    }
    pub fn run(&mut self) -> Result<(), VmError> {
        while self.current_frame().ip < self.current_frame().instructions().len() {
            let ip = self.current_frame().ip;
            let ins = self.current_frame().instructions();
            let op: Opcode = ins[ip].try_into()?;
            match op {
                Opcode::OpConstant => {
                    let const_index = read_u16(&ins[ip + 1..])?;
                    self.current_frame().ip += 2;
                    let obj = Rc::clone(&self.constants.borrow()[const_index as usize]);
                    self.push(obj)?;
                }
                Opcode::OpAdd | Opcode::OpSub | Opcode::OpMul | Opcode::OpDiv => {
                    self.execute_binary_operation(op)?;
                }
                Opcode::OpPop => {
                    self.pop();
                }
                Opcode::OpTrue => {
                    self.push(object::new_bool(true))?;
                }
                Opcode::OpFalse => {
                    self.push(object::new_bool(false))?;
                }
                Opcode::OpEqual | Opcode::OpNotEqual | Opcode::OpGreaterThan => {
                    self.execute_comparison(op)?;
                }
                Opcode::OpBang => {
                    self.execute_bang_operator()?;
                }
                Opcode::OpMinus => {
                    self.execute_minus_operator()?;
                }
                Opcode::OpJump => {
                    let pos = code::read_u16(&ins[ip + 1..])?;
                    self.current_frame().ip = (pos - 1) as usize;
                }
                Opcode::OpJumpNotTruthy => {
                    let pos = code::read_u16(&ins[ip + 1..])?;
                    self.current_frame().ip += 2;
                    let condition = self.pop();
                    if !is_truthy(condition) {
                        self.current_frame().ip = (pos - 1) as usize;
                    }
                }
                Opcode::OpNull => {
                    self.push(object::new_null())?;
                }
                Opcode::OpSetGlobal => {
                    let global_index = read_u16(&ins[ip + 1..])?;
                    self.current_frame().ip += 2;
                    let obj = self.pop();
                    set_object(&self.globals, global_index, obj);
                }
                Opcode::OpGetGlobal => {
                    let global_index = read_u16(&ins[ip + 1..])?;
                    self.current_frame().ip += 2;
                    self.push(get_object(&self.globals, global_index))?;
                }
                Opcode::OpArray => {
                    let num_elements = read_u16(&ins[ip + 1..])?;
                    self.current_frame().ip += 2;
                    let array = self.build_array(self.sp - num_elements, self.sp);
                    self.sp -= num_elements;
                    self.push(array)?;
                }
                Opcode::OpHash => {
                    let num_elements = read_u16(&ins[ip + 1..])?;
                    self.current_frame().ip += 2;
                    let hash = self.build_hash(self.sp - num_elements, self.sp)?;
                    self.sp -= num_elements;
                    self.push(hash)?;
                }
                Opcode::OpIndex => {
                    let index = self.pop();
                    let left = self.pop();
                    self.execute_index_exp(left, index)?;
                }
                Opcode::OpCall => {
                    let num_args = ins[ip + 1] as u16;
                    self.current_frame().ip += 1;

                    if self.execute_call(num_args)? {
                        continue; // change frame, keep ip zero
                    }
                }
                Opcode::OpReturnValue => {
                    let return_value = self.pop();
                    let frame = self.pop_frame();
                    self.sp = frame.base_pointer - 1;
                    self.push(return_value)?;
                }
                Opcode::OpReturn => {
                    let frame = self.pop_frame();
                    self.sp = frame.base_pointer - 1;
                    self.push(object::new_null())?;
                }
                Opcode::OpSetLocal => {
                    let local_index = ins[ip + 1];
                    self.current_frame().ip += 1;
                    let base_pointer = self.current_frame().base_pointer;
                    self.stack[base_pointer as usize + local_index as usize] = self.pop();
                }
                Opcode::OpGetLocal => {
                    let local_index = ins[ip + 1];
                    self.current_frame().ip += 1;
                    let base_pointer = self.current_frame().base_pointer;
                    self.push(Rc::clone(
                        &self.stack[base_pointer as usize + local_index as usize],
                    ))?;
                }
                Opcode::OpGetBuiltin => {
                    let builtin_index = ins[ip + 1];
                    self.current_frame().ip += 1;
                    let definition =
                        builtins(builtin_index).ok_or(VmError::CallingNonFunctionAndNonBuiltin)?;
                    self.push(Rc::new(Object::Builtin(definition)))?;
                }
                Opcode::OpClosure => {
                    let const_index = read_u16(&ins[ip + 1..])?;
                    let num_free = ins[ip + 3];
                    self.current_frame().ip += 3;

                    self.push_closure(const_index, num_free)?;
                }
                Opcode::OpGetFree => {
                    let free_index = ins[ip + 1];
                    self.current_frame().ip += 1;

                    let current_closure = &self.current_frame().cl;
                    let obj = Rc::clone(&current_closure.free[free_index as usize]);
                    self.push(obj)?;
                }
                Opcode::OpCurrentClosure => {
                    let current_closure = &self.current_frame().cl;
                    let obj = object::new_closure(
                        current_closure.func.clone(),
                        current_closure.free.iter().map(|o| Rc::clone(o)).collect(),
                    );
                    self.push(obj)?;
                }
            }
            self.current_frame().ip += 1
        }
        Ok(())
    }
    fn call_closure(&mut self, cl: &Closure, num_args: u16) -> Result<(), VmError> {
        if num_args != cl.func.num_params {
            return Err(VmError::WrongNumberOfArguments(
                cl.func.num_params,
                num_args,
            ));
        }
        let num_locals = cl.func.num_locals;
        let frame = Frame::new(cl.clone(), self.sp - num_args);
        let base_pointer = frame.base_pointer;
        self.push_frame(frame);
        self.sp = base_pointer + num_locals;
        Ok(())
    }
    fn push(&mut self, o: Rc<Object>) -> Result<(), VmError> {
        if self.sp >= STACK_SIZE as u16 {
            return Err(VmError::StackOverflow);
        }
        self.stack[self.sp as usize] = o;
        self.sp += 1;
        Ok(())
    }
    fn pop(&mut self) -> Rc<Object> {
        let o = Rc::clone(&self.stack[(self.sp - 1) as usize]);
        self.sp -= 1;
        o
    }
    fn execute_binary_operation(&mut self, op: Opcode) -> Result<(), VmError> {
        let right = self.pop();
        let left = self.pop();
        match &*left {
            Object::Integer(Integer { value: left }) => match &*right {
                Object::Integer(Integer { value: right }) => {
                    return self.execute_binary_integer_operation(op, *left, *right);
                }
                _ => {}
            },
            Object::String(StringObj { value: left }) => match &*right {
                Object::String(StringObj { value: right }) => {
                    return self.execute_binary_string_operation(op, left.as_str(), right.as_str());
                }
                _ => {}
            },
            _ => {}
        }
        Err(VmError::UnsupportedTypesForBinaryOperation(
            left.get_type(),
            right.get_type(),
        ))
    }
    fn execute_binary_integer_operation(
        &mut self,
        op: Opcode,
        left: i64,
        right: i64,
    ) -> Result<(), VmError> {
        let result = match op {
            Opcode::OpAdd => left + right,
            Opcode::OpSub => left - right,
            Opcode::OpMul => left * right,
            Opcode::OpDiv => left / right,
            _ => todo!(),
        };
        self.push(object::new_int(result))?;
        Ok(())
    }
    fn execute_comparison(&mut self, op: Opcode) -> Result<(), VmError> {
        let right = self.pop();
        let left = self.pop();

        match &*left {
            Object::Integer(Integer { value: left }) => match &*right {
                Object::Integer(Integer { value: right }) => {
                    return self.execute_integer_comparison(op, *left, *right);
                }
                _ => {}
            },
            _ => {}
        }
        match op {
            Opcode::OpEqual => return self.push(object::new_bool(right == left)),
            Opcode::OpNotEqual => return self.push(object::new_bool(right != left)),
            _ => {}
        }
        Err(VmError::UnknownOperator(
            op,
            left.get_type(),
            right.get_type(),
        ))
    }
    fn execute_integer_comparison(
        &mut self,
        op: Opcode,
        left: i64,
        right: i64,
    ) -> Result<(), VmError> {
        match op {
            Opcode::OpEqual => self.push(object::new_bool(left == right)),
            Opcode::OpNotEqual => self.push(object::new_bool(left != right)),
            Opcode::OpGreaterThan => self.push(object::new_bool(left > right)),
            _ => Err(VmError::UnknownOperator(op, "INTEGER", "INTEGER")),
        }
    }
    fn execute_bang_operator(&mut self) -> Result<(), VmError> {
        let operand = self.pop();
        match *operand {
            Object::Boolean(Boolean { value }) => match value {
                true => self.push(object::new_bool(false)),
                false => self.push(object::new_bool(true)),
            },
            Object::Null => self.push(object::new_bool(true)),
            _ => self.push(object::new_bool(false)),
        }
    }
    fn execute_minus_operator(&mut self) -> Result<(), VmError> {
        let operand = self.pop();
        match *operand {
            Object::Integer(Integer { value }) => self.push(object::new_int(-value)),
            _ => Err(VmError::UnsupportedTypeForNegation(operand.get_type())),
        }
    }
    fn execute_binary_string_operation(
        &mut self,
        op: Opcode,
        left: &str,
        right: &str,
    ) -> Result<(), VmError> {
        match op {
            Opcode::OpAdd => {
                let result = format!("{}{}", left, right);
                self.push(object::new_string(result.as_str()))
            }
            _ => Err(VmError::UnknowStringOperator(op)),
        }
    }
    fn build_array(&mut self, start_index: u16, end_index: u16) -> Rc<Object> {
        let start_index = start_index as usize;
        let end_index = end_index as usize;
        object::new_array(self.stack[start_index..end_index].to_vec())
    }
    fn build_hash(&mut self, start_index: u16, end_index: u16) -> Result<Rc<Object>, VmError> {
        let mut hashed_pairs = HashMap::new();

        let start_index = start_index as usize;
        let end_index = end_index as usize;
        let mut i = start_index;
        while i < end_index {
            let key = Rc::clone(&self.stack[i]);
            let value = Rc::clone(&self.stack[i + 1]);

            let pair = HashPair {
                key: Rc::clone(&key),
                value,
            };
            let hash_key = key.hash_key()?;
            hashed_pairs.insert(hash_key, pair);

            i += 2;
        }
        Ok(object::new_hash(hashed_pairs))
    }
    fn execute_index_exp(&mut self, left: Rc<Object>, index: Rc<Object>) -> Result<(), VmError> {
        match &*left {
            Object::Array(Array { elems }) => match &*index {
                Object::Integer(Integer { value }) => {
                    return Ok(self.execute_array_index(elems, *value)?)
                }
                _ => {}
            },
            Object::Hash(HashObj { pairs }) => return Ok(self.execute_hash_index(pairs, index)?),
            _ => {}
        }
        Err(VmError::IndexOperatorNotSupported(left.get_type()))
    }
    fn execute_array_index(&mut self, array: &Vec<Rc<Object>>, index: i64) -> Result<(), VmError> {
        if index < 0 || index as usize >= array.len() {
            self.push(object::new_null())
        } else {
            self.push(Rc::clone(&array[index as usize]))
        }
    }
    fn execute_hash_index(
        &mut self,
        pairs: &HashMap<HashKey, HashPair>,
        index: Rc<Object>,
    ) -> Result<(), VmError> {
        let key = index.hash_key()?;
        pairs
            .get(&key)
            .map_or(self.push(object::new_null()), |pair| {
                self.push(Rc::clone(&pair.value))
            })
    }
    fn execute_call(&mut self, num_args: u16) -> Result<bool, VmError> {
        let callee = Rc::clone(&self.stack[(self.sp - 1 - num_args) as usize]);
        match &*callee {
            Object::Closure(cl) => {
                self.call_closure(cl, num_args)?;
                Ok(true)
            }
            Object::Builtin(bi) => {
                self.call_builtin(bi, num_args)?;
                Ok(false)
            }
            _ => Err(VmError::CallingNonFunctionAndNonBuiltin),
        }
    }
    fn call_builtin(&mut self, bi: &Builtin, num_args: u16) -> Result<(), VmError> {
        let args = self.stack[(self.sp - num_args) as usize..self.sp as usize].to_vec();
        let result = (*bi.func)(args);
        self.sp -= num_args + 1;
        self.push(result)
    }
    fn push_closure(&mut self, const_index: u16, num_free: u8) -> Result<(), VmError> {
        let constant = Rc::clone(&self.constants.borrow()[const_index as usize]);
        match &*constant {
            Object::CompiledFunction(cf) => {
                let free =
                    self.stack[(self.sp - num_free as u16) as usize..self.sp as usize].to_vec();
                let closure = object::new_closure(cf.clone(), free);
                self.push(closure)
            }
            _ => Err(VmError::NotAFunction(format!("{}", constant))),
        }
    }
}

fn is_truthy(obj: Rc<Object>) -> bool {
    match *obj {
        Object::Boolean(Boolean { value }) => value,
        Object::Null => false,
        _ => true,
    }
}
fn set_object(v: &Rc<RefCell<Vec<Rc<Object>>>>, pos: u16, obj: Rc<Object>) {
    let pos = pos as usize;
    if v.borrow().len() <= pos {
        v.borrow_mut().resize_with(pos + 1, || object::new_null());
    }
    v.borrow_mut()[pos] = obj
}
fn get_object(v: &Rc<RefCell<Vec<Rc<Object>>>>, pos: u16) -> Rc<Object> {
    Rc::clone(&v.borrow()[pos as usize])
}

#[derive(Debug)]
pub enum VmError {
    CodeError(code::CodeError),
    StackOverflow,
    UnsupportedTypesForBinaryOperation(&'static str, &'static str),
    UnknownOperator(Opcode, &'static str, &'static str),
    UnsupportedTypeForNegation(&'static str),
    UnknowStringOperator(Opcode),
    ObjectError(object::ObjectError),
    IndexOperatorNotSupported(&'static str),
    CallingNonFunctionAndNonBuiltin,
    WrongNumberOfArguments(u16, u16),
    NotAFunction(String),
}
impl Display for VmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CodeError(e) => write!(f, "{}", e),
            Self::StackOverflow => write!(f, "stack overflow"),
            Self::UnsupportedTypesForBinaryOperation(left_type, right_type) => write!(
                f,
                "unsupported types for binary operation: {} {}",
                left_type, right_type
            ),
            Self::UnknownOperator(op, left_type, right_type) => {
                write!(f, "unknown operator: {} ({} {})", op, left_type, right_type)
            }
            Self::UnsupportedTypeForNegation(t) => {
                write!(f, "unsupported type for negation: {}", t)
            }
            Self::UnknowStringOperator(op) => write!(f, "unknown string operator: {}", op),
            Self::ObjectError(e) => write!(f, "{}", e),
            Self::IndexOperatorNotSupported(t) => write!(f, "index operator not supported: {}", t),
            Self::CallingNonFunctionAndNonBuiltin => {
                write!(f, "calling non-function and non-built-in")
            }
            Self::WrongNumberOfArguments(want, got) => {
                write!(f, "wrong number of arguments: want={}, got={}", want, got)
            }
            Self::NotAFunction(func) => write!(f, "not a function: {}", func),
        }
    }
}
impl From<code::CodeError> for VmError {
    fn from(e: code::CodeError) -> Self {
        Self::CodeError(e)
    }
}
impl From<object::ObjectError> for VmError {
    fn from(e: object::ObjectError) -> Self {
        Self::ObjectError(e)
    }
}

#[cfg(test)]
mod tests {
    use crate::object::{HashPair, MayHasHashKey};
    use std::{collections::HashMap, rc::Rc};

    use crate::{
        compiler::Compiler,
        object::{self, Object},
        test_util::parse,
    };

    use super::*;

    struct VmTestCase {
        input: &'static str,
        expected: Rc<Object>,
    }
    impl VmTestCase {
        fn new(input: &'static str, expected: Rc<Object>) -> Self {
            Self { input, expected }
        }
    }

    fn run_vm_tests(tests: &[VmTestCase]) {
        for tt in tests {
            let program = parse(tt.input);
            let mut comp = Compiler::new();
            comp.compile(program).unwrap();
            let mut vm = Vm::new(comp.bytecode());
            match vm.run() {
                Ok(_) => {
                    let stack_elem = vm.last_popped_stack_elem();
                    assert_eq!(tt.expected, stack_elem);
                }
                Err(err) => assert_eq!(object::new_error(format!("{}", err)), tt.expected),
            }
        }
    }
    #[test]
    fn test_integer_arithmetic() {
        let tests = [
            VmTestCase::new("1 + 2", object::new_int(3)),
            VmTestCase::new("1 - 2", object::new_int(-1)),
            VmTestCase::new("1 * 2", object::new_int(2)),
            VmTestCase::new("4 / 2", object::new_int(2)),
            VmTestCase::new("50 / 2 * 2 + 10 - 5", object::new_int(55)),
            VmTestCase::new("5 + 5 + 5 + 5 - 10", object::new_int(10)),
            VmTestCase::new("2 * 2 * 2 * 2 * 2", object::new_int(32)),
            VmTestCase::new("5 * 2 + 10", object::new_int(20)),
            VmTestCase::new("5 + 2 * 10", object::new_int(25)),
            VmTestCase::new("5 * (2 + 10)", object::new_int(60)),
            VmTestCase::new("-5", object::new_int(-5)),
            VmTestCase::new("-10", object::new_int(-10)),
            VmTestCase::new("-50 + 100 + -50", object::new_int(0)),
            VmTestCase::new("(5 + 10 * 2 + 15 / 3) * 2 + -10", object::new_int(50)),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_boolean_exp() {
        let tests = [
            VmTestCase::new("true", object::new_bool(true)),
            VmTestCase::new("false", object::new_bool(false)),
            VmTestCase::new("1 < 2", object::new_bool(true)),
            VmTestCase::new("1 > 2", object::new_bool(false)),
            VmTestCase::new("1 < 1", object::new_bool(false)),
            VmTestCase::new("1 > 1", object::new_bool(false)),
            VmTestCase::new("1 == 1", object::new_bool(true)),
            VmTestCase::new("1 != 1", object::new_bool(false)),
            VmTestCase::new("1 == 2", object::new_bool(false)),
            VmTestCase::new("1 != 2", object::new_bool(true)),
            VmTestCase::new("true == true", object::new_bool(true)),
            VmTestCase::new("false == false", object::new_bool(true)),
            VmTestCase::new("true == false", object::new_bool(false)),
            VmTestCase::new("true != false", object::new_bool(true)),
            VmTestCase::new("false != true", object::new_bool(true)),
            VmTestCase::new("(1 < 2) == true", object::new_bool(true)),
            VmTestCase::new("(1 < 2) == false", object::new_bool(false)),
            VmTestCase::new("(1 > 2) == true", object::new_bool(false)),
            VmTestCase::new("(1 > 2) == false", object::new_bool(true)),
            VmTestCase::new("!true", object::new_bool(false)),
            VmTestCase::new("!false", object::new_bool(true)),
            VmTestCase::new("!5", object::new_bool(false)),
            VmTestCase::new("!!true", object::new_bool(true)),
            VmTestCase::new("!!false", object::new_bool(false)),
            VmTestCase::new("!!5", object::new_bool(true)),
            VmTestCase::new("!(if (false) { 5; })", object::new_bool(true)),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_conditionals() {
        let tests = [
            VmTestCase::new("if (true) { 10 }", object::new_int(10)),
            VmTestCase::new("if (true) { 10 } else { 20 }", object::new_int(10)),
            VmTestCase::new("if (false) { 10 } else { 20 } ", object::new_int(20)),
            VmTestCase::new("if (1) { 10 }", object::new_int(10)),
            VmTestCase::new("if (1 < 2) { 10 }", object::new_int(10)),
            VmTestCase::new("if (1 < 2) { 10 } else { 20 }", object::new_int(10)),
            VmTestCase::new("if (1 > 2) { 10 } else { 20 }", object::new_int(20)),
            VmTestCase::new("if (1 > 2) { 10 }", object::new_null()),
            VmTestCase::new("if (false) { 10 }", object::new_null()),
            VmTestCase::new(
                "if ((if (false) { 10 })) { 10 } else { 20 }",
                object::new_int(20),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_global_let_stm() {
        let tests = [
            VmTestCase::new("let one = 1; one", object::new_int(1)),
            VmTestCase::new("let one = 1; let two = 2; one + two", object::new_int(3)),
            VmTestCase::new(
                "let one = 1; let two = one + one; one + two",
                object::new_int(3),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_string_exp() {
        let tests = [
            VmTestCase::new(r#""monkey""#, object::new_string("monkey")),
            VmTestCase::new(r#""mon" + "key""#, object::new_string("monkey")),
            VmTestCase::new(
                r#""mon" + "key" + "banana""#,
                object::new_string("monkeybanana"),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_array_literals() {
        let tests = [
            VmTestCase::new("[]", object::new_array(vec![])),
            VmTestCase::new(
                "[1, 2, 3]",
                object::new_array(vec![
                    object::new_int(1),
                    object::new_int(2),
                    object::new_int(3),
                ]),
            ),
            VmTestCase::new(
                "[1 + 2, 3 * 4, 5 + 6]",
                object::new_array(vec![
                    object::new_int(3),
                    object::new_int(12),
                    object::new_int(11),
                ]),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_hash_literals() {
        let tests = [
            VmTestCase::new("{}", object::new_hash(HashMap::new())),
            VmTestCase::new(
                "{1:2, 2:3}",
                object::new_hash({
                    let mut hm = HashMap::new();
                    hm.insert(
                        Integer { value: 1 }.hash_key().unwrap(),
                        HashPair {
                            key: object::new_int(1),
                            value: object::new_int(2),
                        },
                    );
                    hm.insert(
                        Integer { value: 2 }.hash_key().unwrap(),
                        HashPair {
                            key: object::new_int(2),
                            value: object::new_int(3),
                        },
                    );
                    hm
                }),
            ),
            VmTestCase::new(
                "{1+1:2*2, 3+3:4*4}",
                object::new_hash({
                    let mut hm = HashMap::new();
                    hm.insert(
                        Integer { value: 2 }.hash_key().unwrap(),
                        HashPair {
                            key: object::new_int(2),
                            value: object::new_int(4),
                        },
                    );
                    hm.insert(
                        Integer { value: 6 }.hash_key().unwrap(),
                        HashPair {
                            key: object::new_int(6),
                            value: object::new_int(16),
                        },
                    );
                    hm
                }),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_index_exp() {
        let tests = [
            VmTestCase::new("[1, 2, 3][1]", object::new_int(2)),
            VmTestCase::new("[1, 2, 3][0 + 2]", object::new_int(3)),
            VmTestCase::new("[[1, 1, 1]][0][0]", object::new_int(1)),
            VmTestCase::new("[][0]", object::new_null()),
            VmTestCase::new("[1, 2, 3][99]", object::new_null()),
            VmTestCase::new("[1][-1]", object::new_null()),
            VmTestCase::new("{1: 1, 2: 2}[1]", object::new_int(1)),
            VmTestCase::new("{1: 1, 2: 2}[2]", object::new_int(2)),
            VmTestCase::new("{1: 1}[0]", object::new_null()),
            VmTestCase::new("{}[0]", object::new_null()),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_calling_functions_without_arguments() {
        let tests = [
            VmTestCase::new(
                r#"
            let fivePlusTen = fn() { 5 + 10; };
            fivePlusTen();
            "#,
                object::new_int(15),
            ),
            VmTestCase::new(
                r#"
            let one = fn() { 1; };
            let two = fn() { 2; };
            one() + two()
            "#,
                object::new_int(3),
            ),
            VmTestCase::new(
                r#"
            let a = fn() { 1 };
            let b = fn() { a() + 1 };
            let c = fn() { b() + 1 };
            c();
            "#,
                object::new_int(3),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_functions_with_return_stm() {
        let tests = [
            VmTestCase::new(
                r#"
            let earlyExit = fn() { return 99; 100; };
            earlyExit();
            "#,
                object::new_int(99),
            ),
            VmTestCase::new(
                r#"
            let earlyExit = fn() { return 99; return 100; };
            earlyExit();
            "#,
                object::new_int(99),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_function_without_return_value() {
        let tests = [
            VmTestCase::new(
                r#"
            let noReturn = fn() { };
            noReturn();
            "#,
                object::new_null(),
            ),
            VmTestCase::new(
                r#"
            let noReturn = fn() { };
            let noReturnTwo = fn() { noReturn(); };
            noReturn();
            noReturnTwo();
            "#,
                object::new_null(),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_first_class_functions() {
        let tests = [
            VmTestCase::new(
                r#"
            let returnsOne = fn() { 1; };
            let returnsOneReturner = fn() { returnsOne; };
            returnsOneReturner()();
            "#,
                object::new_int(1),
            ),
            VmTestCase::new(
                r#"
            let returnsOneReturner = fn() { 
                let returnsOne = fn() { 1; };
                return returnsOne;
             };
            returnsOneReturner()();
            "#,
                object::new_int(1),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_calling_functions_with_bindings() {
        let tests = [
            VmTestCase::new(
                r#"
            let one = fn() { let one = 1; one };
            one();
            "#,
                object::new_int(1),
            ),
            VmTestCase::new(
                r#"
            let oneAndTwo = fn() { let one = 1; let two = 2; one + two; };
            oneAndTwo();
            "#,
                object::new_int(3),
            ),
            VmTestCase::new(
                r#"
            let oneAndTwo = fn() { let one = 1; let two = 2; one + two; };
            let threeAndFour = fn() { let three = 3; let four = 4; three + four; };
            oneAndTwo() + threeAndFour();
            "#,
                object::new_int(10),
            ),
            VmTestCase::new(
                r#"
            let firstFoobar = fn() { let foobar = 50; foobar; };
            let secondFoobar = fn() { let foobar = 100; foobar; };
            firstFoobar() + secondFoobar();
            "#,
                object::new_int(150),
            ),
            VmTestCase::new(
                r#"
            let globalSeed = 50;
            let minusOne = fn() {
                let num = 1;
                globalSeed - num;
            }
            let minusTwo = fn() {
                let num = 2;
                globalSeed - num;
            }
            minusOne() + minusTwo();
            "#,
                object::new_int(97),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_calling_functions_with_arguments_and_bindings() {
        let tests = [
            VmTestCase::new(
                r#"
            let identity = fn(a) { a; };
            identity(4);
            "#,
                object::new_int(4),
            ),
            VmTestCase::new(
                r#"
            let sum = fn(a, b) { a + b; };
            sum(1, 2);
            "#,
                object::new_int(3),
            ),
            VmTestCase::new(
                r#"
            let sum = fn(a, b) {
                let c = a + b;
                c;
            };
            sum(1, 2);
            "#,
                object::new_int(3),
            ),
            VmTestCase::new(
                r#"
            let sum = fn(a, b) {
                let c = a + b;
                c;
            };
            sum(1, 2) + sum(3, 4);
            "#,
                object::new_int(10),
            ),
            VmTestCase::new(
                r#"
            let sum = fn(a, b) {
                let c = a + b;
                c;
            };
            let outer = fn() {
                sum(1, 2) + sum(3, 4);
            };
            outer();
            "#,
                object::new_int(10),
            ),
            VmTestCase::new(
                r#"
            let globalNum = 10;
            let sum = fn(a, b) {
                let c = a + b;
                c + globalNum;
            };
            let outer = fn() {
                sum(1, 2) + sum(3, 4) + globalNum;
            };
            outer() + globalNum;
            "#,
                object::new_int(50),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_calling_functions_with_wrong_arguments() {
        let tests = [
            VmTestCase::new(
                r#"
            fn() { 1; }(1);
            "#,
                object::new_error("wrong number of arguments: want=0, got=1".to_owned()),
            ),
            VmTestCase::new(
                r#"
            fn(a) { a; }();
            "#,
                object::new_error("wrong number of arguments: want=1, got=0".to_owned()),
            ),
            VmTestCase::new(
                r#"
            fn(a,b) { a + b; }(1);
            "#,
                object::new_error("wrong number of arguments: want=2, got=1".to_owned()),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_builtin_functions() {
        let tests = [
            VmTestCase::new(r#"len("")"#, object::new_int(0)),
            VmTestCase::new(r#"len("four")"#, object::new_int(4)),
            VmTestCase::new(r#"len("hello world")"#, object::new_int(11)),
            VmTestCase::new(
                r#"len(1)"#,
                object::new_error("argument to `len` not supported, got INTEGER".to_owned()),
            ),
            VmTestCase::new(
                r#"len("one", "two")"#,
                object::new_error("wrong number of arguments. got=2, want=1".to_owned()),
            ),
            VmTestCase::new("len([1, 2, 3])", object::new_int(3)),
            VmTestCase::new("len([])", object::new_int(0)),
            VmTestCase::new(r#"puts("hello", "world!")"#, object::new_null()),
            VmTestCase::new(r#"first([1, 2, 3])"#, object::new_int(1)),
            VmTestCase::new("first([])", object::new_null()),
            VmTestCase::new(
                "first(1)",
                object::new_error("argument to `first` must be ARRAY, got INTEGER".to_owned()),
            ),
            VmTestCase::new("last([1,2,3])", object::new_int(3)),
            VmTestCase::new("last([])", object::new_null()),
            VmTestCase::new(
                "last(1)",
                object::new_error("argument to `last` must be ARRAY, got INTEGER".to_owned()),
            ),
            VmTestCase::new(
                "rest([1,2,3])",
                object::new_array(vec![object::new_int(2), object::new_int(3)]),
            ),
            VmTestCase::new("rest([])", object::new_null()),
            VmTestCase::new("push([],1)", object::new_array(vec![object::new_int(1)])),
            VmTestCase::new(
                "push(1,1)",
                object::new_error("argument to `push` must be ARRAY, got INTEGER".to_owned()),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_closures() {
        let tests = [
            VmTestCase::new(
                r#"
        let newClosure = fn(a) {
            fn() { a; };
        };
        let closure = newClosure(99);
        closure();
        "#,
                object::new_int(99),
            ),
            VmTestCase::new(
                r#"
        let newAdder = fn(a, b) {
            fn(c) { a + b + c };
        };
        let adder = newAdder(1, 2);
        adder(8);
        "#,
                object::new_int(11),
            ),
            VmTestCase::new(
                r#"
            let newAdder = fn(a, b) {
                let c = a + b;
                fn(d) { c + d };
            };
            let adder = newAdder(1, 2);
            adder(8);
            "#,
                object::new_int(11),
            ),
            VmTestCase::new(
                r#"
            let newAdderOuter = fn(a, b) {
                let c = a + b;
                fn(d) {
                    let e = d + c;
                    fn(f) { e + f; };
                };
            };
            let newAdderInner = newAdderOuter(1, 2)
            let adder = newAdderInner(3);
            adder(8);
            "#,
                object::new_int(14),
            ),
            VmTestCase::new(
                r#"
            let a = 1;
            let newAdderOuter = fn(b) {
                fn(c) {
                    fn(d) { a + b + c + d };
                };
            };
            let newAdderInner = newAdderOuter(2)
            let adder = newAdderInner(3);
            adder(8);
            "#,
                object::new_int(14),
            ),
            VmTestCase::new(
                r#"
            let newClosure = fn(a, b) {
                let one = fn() { a; };
                let two = fn() { b; };
                fn() { one() + two(); };
            };
            let closure = newClosure(9, 90);
            closure();
            "#,
                object::new_int(99),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_recursive_functions() {
        let tests = [
            VmTestCase::new(
                r#"
            let countDown = fn(x) {
                if (x == 0) {
                    return 0;
                } else {
                    countDown(x - 1);
                }
            };
            countDown(1);
            "#,
                object::new_int(0),
            ),
            VmTestCase::new(
                r#"
        let countDown = fn(x) {
            if (x == 0) {
                return 0;
            } else {
                countDown(x - 1);
            }
        };
        let wrapper = fn() {
            countDown(1);
        };
        wrapper();
        "#,
                object::new_int(0),
            ),
            VmTestCase::new(
                r#"
            let wrapper = fn() {
                let countDown = fn(x) {
                    if (x == 0) {
                        return 0;
                    } else {
                        countDown(x - 1);
                    }
                };
                countDown(1);
            };
            wrapper();
            "#,
                object::new_int(0),
            ),
        ];
        run_vm_tests(&tests)
    }
    #[test]
    fn test_recursive_fibonacci() {
        let tests = [VmTestCase::new(
            r#"
            let fibonacci = fn(x) {
                if (x == 0) {
                    return 0;
                } else {
                    if (x == 1) {
                        return 1; 
                    } else { 
                        fibonacci(x - 1) + fibonacci(x - 2); 
                    } 
                } 
            };
            fibonacci(15);
            "#,
            object::new_int(610),
        )];
        run_vm_tests(&tests)
    }
}
