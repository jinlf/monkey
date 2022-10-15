use crate::{
    builtin::BUILTIN_NAMES, compiler::Compiler, lexer::Lexer, parser::Parser,
    symbol_table::SymbolTable, token::Token, vm::Vm,
};
use std::{
    cell::RefCell,
    io::{BufRead, BufReader, Read, Write},
    rc::Rc,
};

const PROMPT: &'static str = ">> ";

pub fn start_tokens(input: &mut dyn Read, output: &mut dyn Write) -> std::io::Result<()> {
    let mut scanner = BufReader::new(input);
    loop {
        write!(output, "{}", PROMPT)?;
        output.flush()?;

        let mut line = String::new();
        scanner.read_line(&mut line)?;

        let mut l = Lexer::new(line.as_str());
        loop {
            match l.next_token() {
                Ok(Token::Eof) => break,
                Ok(tok) => {
                    writeln!(output, "{:?}", tok)?;
                    output.flush()?;
                }
                Err(err) => {
                    writeln!(output, "{}", err)?;
                    output.flush()?;
                }
            }
        }
    }
}

pub fn start_ast(input: &mut dyn Read, output: &mut dyn Write) -> std::io::Result<()> {
    let mut scanner = BufReader::new(input);
    loop {
        write!(output, "{}", PROMPT)?;
        output.flush()?;

        let mut line = String::new();
        scanner.read_line(&mut line)?;

        let l = Lexer::new(line.as_str());
        let mut p = Parser::new(l).unwrap();
        match p.parse_program() {
            Ok(program) => {
                writeln!(output, "{}", program)?;
                output.flush()?;
            }
            Err(err) => {
                writeln!(output, "{}", err)?;
                output.flush()?;
            }
        }
    }
}
pub fn start(input: &mut dyn Read, output: &mut dyn Write) -> std::io::Result<()> {
    let mut scanner = BufReader::new(input);
    let constants = Rc::new(RefCell::new(vec![]));
    let globals = Rc::new(RefCell::new(vec![]));
    let symbol_table = SymbolTable::new();
    BUILTIN_NAMES.iter().enumerate().for_each(|(index, name)| {
        symbol_table.borrow_mut().define_builtin(index, name);
    });
    loop {
        write!(output, "{}", PROMPT).and_then(|_| output.flush())?;

        let mut line = String::new();
        scanner.read_line(&mut line)?;

        let l = Lexer::new(line.as_str());
        let mut p = Parser::new(l).unwrap();
        p.parse_program()
            .map_err(|e| format!("{}", e))
            .and_then(|program| {
                let mut comp =
                    Compiler::new_with_state(Rc::clone(&symbol_table), Rc::clone(&constants));
                comp.compile(program).map_err(|e| format!("{}", e))?;
                Ok(comp)
            })
            .and_then(|comp| {
                let mut machine = Vm::new_with_global_store(comp.bytecode(), Rc::clone(&globals));
                machine.run().map_err(|e| format!("{}", e))?;
                Ok(machine)
            })
            .and_then(|machine| {
                let stack_top = machine.last_popped_stack_elem();
                writeln!(output, "{}", stack_top)
                    .and_then(|_| output.flush())
                    .map_err(|e| format!("{}", e))?;
                Ok(())
            })
            .or_else(|e| writeln!(output, "{}", e).and_then(|_| output.flush()))?;
    }
}
