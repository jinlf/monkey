use crate::{ast::Program, lexer::Lexer, parser::Parser};

pub fn parse(input: &str) -> Program {
    let l = Lexer::new(input);
    let mut p = Parser::new(l).unwrap();
    p.parse_program().unwrap()
}
