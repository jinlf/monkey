use crate::{
    ast::{
        ArrayLiteral, BlockStm, BoolLiteral, CallExp, Exp, ExpStm, FuncLiteral, HashLiteral, Ident,
        IdentExp, IfExp, IndexExp, InfixExp, InfixOperator, IntLiteral, LetStm, PrefixExp,
        PrefixOperator, Program, ReturnStm, Stm, StringLiteral,
    },
    lexer::{self, Lexer},
    token::Token,
};
use std::{collections::HashMap, fmt::Display, mem};

fn prefix_fns(
    cur_token: &Token,
) -> Option<Box<dyn FnOnce(&mut Parser) -> Result<Exp, ParserError>>> {
    Some(Box::new(match &cur_token {
        Token::Ident(_) => |parser| Ok(Exp::IdentExp(parser.parse_ident_exp()?)),
        Token::Int(_) => |parser| Ok(Exp::IntLiteral(parser.parse_int_literal()?)),
        Token::Bang | Token::Minus => |parser| Ok(Exp::PrefixExp(parser.parse_prefix_exp()?)),
        Token::True | Token::False => |parser| Ok(Exp::BoolLiteral(parser.parse_bool_literal()?)),
        Token::LParen => |parser| parser.parse_grouped_exp(),
        Token::If => |parser| Ok(Exp::IfExp(parser.parse_if_exp()?)),
        Token::Function => |parser| Ok(Exp::FuncLiteral(parser.parse_func_literal()?)),
        Token::String(_) => |parser| Ok(Exp::StringLiteral(parser.parse_string_literal()?)),
        Token::LBracket => |parser| Ok(Exp::ArrayLiteral(parser.parse_array_literal()?)),
        Token::LBrace => |parser| Ok(Exp::HashLiteral(parser.parse_hash_literal()?)),
        _ => return None,
    }))
}
fn infix_fns(
    cur_token: &Token,
) -> Option<Box<dyn FnOnce(&mut Parser, Exp) -> Result<Exp, ParserError>>> {
    Some(Box::new(match &cur_token {
        Token::Plus
        | Token::Minus
        | Token::Asterisk
        | Token::Slash
        | Token::Eq
        | Token::NotEq
        | Token::Lt
        | Token::Gt => |parser, left| Ok(Exp::InfixExp(parser.parse_infix_exp(left)?)),
        Token::LParen => |parser, left| Ok(Exp::CallExp(parser.parse_call_exp(left)?)),
        Token::LBracket => |parser, left| Ok(Exp::IndexExp(parser.parse_index_exp(left)?)),
        _ => return None,
    }))
}

#[derive(PartialEq, PartialOrd)]
enum Precedence {
    Lowest,
    Equals,
    LessGreater,
    Sum,
    Product,
    Prefix,
    Call,
    Index,
}
fn get_precedence(token: &Token) -> Precedence {
    match token {
        Token::Eq | Token::NotEq => Precedence::Equals,
        Token::Lt | Token::Gt => Precedence::LessGreater,
        Token::Plus | Token::Minus => Precedence::Sum,
        Token::Asterisk | Token::Slash => Precedence::Product,
        Token::LParen => Precedence::Call,
        Token::LBracket => Precedence::Index,
        _ => Precedence::Lowest,
    }
}

pub struct Parser<'input> {
    l: Lexer<'input>,
    cur_tok: Token,
    peek_tok: Token,
}
impl<'input> Parser<'input> {
    pub fn new(l: Lexer<'input>) -> Result<Self, ParserError> {
        let mut p = Parser {
            l,
            cur_tok: Token::Eof,
            peek_tok: Token::Eof,
        };
        p.next_token()?;
        p.next_token()?;
        Ok(p)
    }
    fn next_token(&mut self) -> Result<(), ParserError> {
        self.cur_tok = mem::replace(&mut self.peek_tok, self.l.next_token()?);
        Ok(())
    }
    pub fn parse_program(&mut self) -> Result<Program, ParserError> {
        let mut stms = vec![];
        while self.cur_tok != Token::Eof {
            stms.push(self.parse_stm()?);
        }
        Ok(Program { stms })
    }
    fn parse_stm(&mut self) -> Result<Stm, ParserError> {
        Ok(match self.cur_tok {
            Token::Let => Stm::LetStm(self.parse_let_stm()?),
            Token::Return => Stm::ReturnStm(self.parse_return_stm()?),
            _ => Stm::ExpStm(self.parse_exp_stm()?),
        })
    }
    fn parse_let_stm(&mut self) -> Result<LetStm, ParserError> {
        self.expect_token(Token::Let)?;
        let name = self.parse_ident()?;
        self.expect_token(Token::Assign)?;
        let value = self.parse_exp(Precedence::Lowest)?;
        let value = match value {
            Exp::FuncLiteral(FuncLiteral {
                name: _,
                body,
                params,
            }) => Exp::FuncLiteral(FuncLiteral {
                name: Some(name.to_owned()),
                body,
                params,
            }),
            _ => value,
        };

        if self.cur_tok == Token::Semicolon {
            self.next_token()?;
        }
        Ok(LetStm { name, value })
    }
    fn expect_token(&mut self, expected: Token) -> Result<(), ParserError> {
        if self.cur_tok == expected {
            self.next_token()
        } else {
            Err(ParserError::UnexpectedToken {
                actual: self.cur_tok.clone(),
                expected: vec![expected],
            })
        }
    }
    fn parse_ident(&mut self) -> Result<Ident, ParserError> {
        let ident = match &self.cur_tok {
            Token::Ident(ident) => ident.to_owned(),
            _ => {
                return Err(ParserError::UnexpectedToken {
                    actual: self.cur_tok.clone(),
                    expected: vec![Token::Ident(String::new())],
                })
            }
        };
        self.next_token()?;
        Ok(ident)
    }
    fn parse_exp(&mut self, precedence: Precedence) -> Result<Exp, ParserError> {
        match prefix_fns(&self.cur_tok) {
            Some(prefix) => {
                let mut left = prefix(self)?;
                while self.cur_tok != Token::Semicolon && precedence < get_precedence(&self.cur_tok)
                {
                    left = match infix_fns(&self.cur_tok) {
                        Some(infix) => infix(self, left)?,
                        _ => return Ok(left),
                    }
                }
                return Ok(left);
            }
            _ => Err(ParserError::NoPrefixParseFn(self.cur_tok.clone())),
        }
    }
    fn parse_return_stm(&mut self) -> Result<ReturnStm, ParserError> {
        self.expect_token(Token::Return)?;
        let return_value = self.parse_exp(Precedence::Lowest)?;
        if self.cur_tok == Token::Semicolon {
            self.next_token()?;
        }
        Ok(ReturnStm { return_value })
    }
    fn parse_exp_stm(&mut self) -> Result<ExpStm, ParserError> {
        let exp = self.parse_exp(Precedence::Lowest)?;
        if self.cur_tok == Token::Semicolon {
            self.next_token()?;
        }
        Ok(ExpStm { exp })
    }
    fn parse_ident_exp(&mut self) -> Result<IdentExp, ParserError> {
        let ident_exp = match &self.cur_tok {
            Token::Ident(ident) => IdentExp {
                value: ident.to_owned(),
            },
            _ => {
                return Err(ParserError::UnexpectedToken {
                    actual: self.cur_tok.clone(),
                    expected: vec![Token::Ident(String::new())],
                })
            }
        };
        self.next_token()?;
        Ok(ident_exp)
    }
    fn parse_int_literal(&mut self) -> Result<IntLiteral, ParserError> {
        let int_literal = match &self.cur_tok {
            Token::Int(value) => IntLiteral { value: *value },
            _ => {
                return Err(ParserError::UnexpectedToken {
                    actual: self.cur_tok.clone(),
                    expected: vec![Token::Int(0)],
                })
            }
        };
        self.next_token()?;
        Ok(int_literal)
    }
    fn parse_prefix_exp(&mut self) -> Result<PrefixExp, ParserError> {
        let op = (&self.cur_tok).try_into()?;
        self.next_token()?;
        Ok(PrefixExp {
            op,
            right: Box::new(self.parse_exp(Precedence::Prefix)?),
        })
    }
    fn parse_infix_exp(&mut self, left: Exp) -> Result<InfixExp, ParserError> {
        let precedence = get_precedence(&self.cur_tok);
        let op = (&self.cur_tok).try_into()?;
        self.next_token()?;
        let right = self.parse_exp(precedence)?;
        Ok(InfixExp {
            left: Box::new(left),
            op,
            right: Box::new(right),
        })
    }
    fn parse_bool_literal(&mut self) -> Result<BoolLiteral, ParserError> {
        let bool_literal = match &self.cur_tok {
            Token::True => BoolLiteral { value: true },
            Token::False => BoolLiteral { value: false },
            _ => {
                return Err(ParserError::UnexpectedToken {
                    actual: self.cur_tok.clone(),
                    expected: vec![Token::True, Token::False],
                })
            }
        };
        self.next_token()?;
        Ok(bool_literal)
    }
    fn parse_grouped_exp(&mut self) -> Result<Exp, ParserError> {
        self.expect_token(Token::LParen)?;
        let exp = self.parse_exp(Precedence::Lowest)?;
        self.expect_token(Token::RParen)?;
        Ok(exp)
    }
    fn parse_if_exp(&mut self) -> Result<IfExp, ParserError> {
        self.expect_token(Token::If)?;
        self.expect_token(Token::LParen)?;
        let cond = self.parse_exp(Precedence::Lowest)?;
        self.expect_token(Token::RParen)?;
        let conseq = self.parse_block_stm()?;
        let alter = if self.cur_tok == Token::Else {
            self.next_token()?;
            Some(self.parse_block_stm()?)
        } else {
            None
        };
        Ok(IfExp {
            cond: Box::new(cond),
            conseq,
            alter,
        })
    }
    fn parse_block_stm(&mut self) -> Result<BlockStm, ParserError> {
        let mut stms = vec![];
        self.expect_token(Token::LBrace)?;
        while self.cur_tok != Token::RBrace {
            stms.push(self.parse_stm()?);
        }
        self.expect_token(Token::RBrace)?;
        Ok(BlockStm { stms })
    }
    fn parse_func_literal(&mut self) -> Result<FuncLiteral, ParserError> {
        self.expect_token(Token::Function)?;
        self.expect_token(Token::LParen)?;
        let params = self.parse_ident_list(Token::RParen)?;
        self.expect_token(Token::RParen)?;
        let body = self.parse_block_stm()?;
        Ok(FuncLiteral {
            name: None,
            params,
            body,
        })
    }
    fn parse_ident_list(&mut self, end_token: Token) -> Result<Vec<Ident>, ParserError> {
        let mut ident_list = vec![];
        if self.cur_tok == end_token {
            return Ok(ident_list);
        }
        ident_list.push(self.parse_ident()?);
        while self.cur_tok == Token::Comma {
            self.next_token()?;
            ident_list.push(self.parse_ident()?);
        }
        Ok(ident_list)
    }
    fn parse_call_exp(&mut self, left: Exp) -> Result<CallExp, ParserError> {
        self.expect_token(Token::LParen)?;
        let args = self.parse_exp_list(Token::RParen)?;
        self.expect_token(Token::RParen)?;
        Ok(CallExp {
            func: Box::new(left),
            args,
        })
    }
    fn parse_exp_list(&mut self, end_token: Token) -> Result<Vec<Exp>, ParserError> {
        let mut exp_list = vec![];
        if self.cur_tok == end_token {
            return Ok(exp_list);
        }
        exp_list.push(self.parse_exp(Precedence::Lowest)?);
        while self.cur_tok == Token::Comma {
            self.next_token()?;
            exp_list.push(self.parse_exp(Precedence::Lowest)?);
        }
        Ok(exp_list)
    }
    fn parse_string_literal(&mut self) -> Result<StringLiteral, ParserError> {
        let string_literal = match &self.cur_tok {
            Token::String(value) => StringLiteral {
                value: value.to_owned(),
            },
            _ => {
                return Err(ParserError::UnexpectedToken {
                    actual: self.cur_tok.clone(),
                    expected: vec![Token::String(String::new())],
                })
            }
        };
        self.next_token()?;
        Ok(string_literal)
    }
    fn parse_array_literal(&mut self) -> Result<ArrayLiteral, ParserError> {
        self.expect_token(Token::LBracket)?;
        let elems = self.parse_exp_list(Token::RBracket)?;
        self.expect_token(Token::RBracket)?;
        Ok(ArrayLiteral { elems })
    }
    fn parse_index_exp(&mut self, left: Exp) -> Result<IndexExp, ParserError> {
        self.expect_token(Token::LBracket)?;
        let index = self.parse_exp(Precedence::Lowest)?;
        self.expect_token(Token::RBracket)?;
        Ok(IndexExp {
            left: Box::new(left),
            index: Box::new(index),
        })
    }
    fn parse_hash_literal(&mut self) -> Result<HashLiteral, ParserError> {
        self.expect_token(Token::LBrace)?;
        let pairs = self.parse_pair_list(Token::RBrace)?;
        self.expect_token(Token::RBrace)?;
        Ok(HashLiteral { pairs })
    }
    fn parse_pair_list(&mut self, end_token: Token) -> Result<HashMap<Exp, Exp>, ParserError> {
        let mut pairs = HashMap::new();
        if self.cur_tok == end_token {
            return Ok(pairs);
        }
        let key = self.parse_exp(Precedence::Lowest)?;
        self.expect_token(Token::Colon)?;
        let value = self.parse_exp(Precedence::Lowest)?;
        pairs.insert(key, value);

        while self.cur_tok == Token::Comma {
            self.next_token()?;
            let key = self.parse_exp(Precedence::Lowest)?;
            self.expect_token(Token::Colon)?;
            let value = self.parse_exp(Precedence::Lowest)?;
            pairs.insert(key, value);
        }
        Ok(pairs)
    }
}

impl TryFrom<&Token> for PrefixOperator {
    type Error = ParserError;

    fn try_from(value: &Token) -> Result<Self, Self::Error> {
        Ok(match value {
            Token::Bang => Self::Bang,
            Token::Minus => Self::Minus,
            _ => {
                return Err(Self::Error::UnexpectedToken {
                    actual: value.clone(),
                    expected: vec![Token::Bang, Token::Minus],
                })
            }
        })
    }
}
impl TryFrom<&Token> for InfixOperator {
    type Error = ParserError;

    fn try_from(value: &Token) -> Result<Self, Self::Error> {
        Ok(match value {
            Token::Plus => Self::Add,
            Token::Minus => Self::Sub,
            Token::Asterisk => Self::Mul,
            Token::Slash => Self::Div,
            Token::Lt => Self::Lt,
            Token::Gt => Self::Gt,
            Token::Eq => Self::Eq,
            Token::NotEq => Self::NotEq,
            _ => {
                return Err(Self::Error::UnexpectedToken {
                    actual: value.clone(),
                    expected: vec![
                        Token::Plus,
                        Token::Minus,
                        Token::Asterisk,
                        Token::Slash,
                        Token::Lt,
                        Token::Gt,
                        Token::Eq,
                        Token::NotEq,
                    ],
                })
            }
        })
    }
}

#[derive(Debug)]
pub enum ParserError {
    LexerError(lexer::LexerError),
    UnexpectedToken { actual: Token, expected: Vec<Token> },
    NoPrefixParseFn(Token),
}
impl Display for ParserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LexerError(e) => write!(f, "{}", e),
            Self::UnexpectedToken { actual, expected } => {
                write!(f, "unexpected token {:?}, want {:?}", actual, expected)
            }
            Self::NoPrefixParseFn(tok) => write!(f, "no prefix parse function for {:?}", tok),
        }
    }
}
impl From<lexer::LexerError> for ParserError {
    fn from(e: lexer::LexerError) -> Self {
        Self::LexerError(e)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        ast::{
            ArrayLiteral, BlockStm, CallExp, FuncLiteral, HashLiteral, IdentExp, IfExp, IndexExp,
            LetStm, StringLiteral,
        },
        test_util::parse,
    };

    use super::*;

    #[test]
    fn test_let_stms() {
        let input = r#"
let x = 5;
let y = 10;
let foobar = 838383;
"#;
        let program = parse(input);
        assert_eq!(program.stms.len(), 3);

        let tests = [("x", 5), ("y", 10), ("foobar", 838383)];
        for (i, tt) in tests.iter().enumerate() {
            assert_eq!(
                program.stms[i],
                Stm::LetStm(LetStm {
                    name: tt.0.to_owned(),
                    value: Exp::IntLiteral(IntLiteral { value: tt.1 })
                })
            )
        }
    }

    #[test]
    fn test_return_stms() {
        let input = r#"
return 5;
return 10;
return 993322;
        "#;
        let program = parse(input);
        assert_eq!(program.stms.len(), 3);
        program.stms.iter().for_each(|stm| match stm {
            Stm::ReturnStm(_) => {}
            _ => assert!(false),
        })
    }
    #[test]
    fn test_ident_exp() {
        let input = "foobar;";
        let program = parse(input);
        assert_eq!(program.stms.len(), 1);
        assert_eq!(
            program.stms[0],
            Stm::ExpStm(ExpStm {
                exp: Exp::IdentExp(IdentExp {
                    value: "foobar".to_owned()
                })
            })
        );
    }
    #[test]
    fn test_int_literal_exp() {
        let input = "5;";
        let program = parse(input);
        assert_eq!(program.stms.len(), 1);
        assert_eq!(
            program.stms[0],
            Stm::ExpStm(ExpStm {
                exp: Exp::IntLiteral(IntLiteral { value: 5 })
            })
        )
    }
    #[test]
    fn test_prefix_exp() {
        let tests = [
            ("!5;", PrefixOperator::Bang, 5),
            ("-15;", PrefixOperator::Minus, 15),
        ];
        for tt in tests {
            let program = parse(tt.0);
            assert_eq!(program.stms.len(), 1);
            assert_eq!(
                program.stms[0],
                Stm::ExpStm(ExpStm {
                    exp: Exp::PrefixExp(PrefixExp {
                        op: tt.1,
                        right: Box::new(Exp::IntLiteral(IntLiteral { value: tt.2 }))
                    })
                })
            )
        }
    }
    #[test]
    fn test_infix_exp() {
        let tests = [
            ("5 + 5;", 5, InfixOperator::Add, 5),
            ("5 - 5;", 5, InfixOperator::Sub, 5),
            ("5 * 5;", 5, InfixOperator::Mul, 5),
            ("5 / 5;", 5, InfixOperator::Div, 5),
            ("5 > 5;", 5, InfixOperator::Gt, 5),
            ("5 < 5;", 5, InfixOperator::Lt, 5),
            ("5 == 5;", 5, InfixOperator::Eq, 5),
            ("5 != 5;", 5, InfixOperator::NotEq, 5),
        ];
        for tt in tests {
            let program = parse(tt.0);
            assert_eq!(program.stms.len(), 1);
            assert_eq!(
                program.stms[0],
                Stm::ExpStm(ExpStm {
                    exp: Exp::InfixExp(InfixExp {
                        left: Box::new(Exp::IntLiteral(IntLiteral { value: tt.1 })),
                        op: tt.2,
                        right: Box::new(Exp::IntLiteral(IntLiteral { value: tt.3 })),
                    })
                })
            )
        }
    }
    #[test]
    fn parse_operator_precedence() {
        let tests = [
            ("-a * b", "((-a) * b)"),
            ("!-a", "(!(-a))"),
            ("a + b + c", "((a + b) + c)"),
            ("a + b - c", "((a + b) - c)"),
            ("a * b * c", "((a * b) * c)"),
            ("a * b / c", "((a * b) / c)"),
            ("a + b / c", "(a + (b / c))"),
            ("a + b * c + d / e - f", "(((a + (b * c)) + (d / e)) - f)"),
            ("3 + 4; -5 * 5", "(3 + 4)((-5) * 5)"),
            ("5 > 4 == 3 < 4", "((5 > 4) == (3 < 4))"),
            ("5 < 4 != 3 > 4", "((5 < 4) != (3 > 4))"),
            (
                "3 + 4 * 5 == 3 * 1 + 4 * 5",
                "((3 + (4 * 5)) == ((3 * 1) + (4 * 5)))",
            ),
            (
                "3 + 4 * 5 == 3 * 1 + 4 * 5",
                "((3 + (4 * 5)) == ((3 * 1) + (4 * 5)))",
            ),
            ("true", "true"),
            ("false", "false"),
            ("3 > 5 == false", "((3 > 5) == false)"),
            ("3 < 5 == true", "((3 < 5) == true)"),
            ("1 + (2 + 3) + 4", "((1 + (2 + 3)) + 4)"),
            ("(5 + 5) * 2", "((5 + 5) * 2)"),
            ("2 / (5 + 5)", "(2 / (5 + 5))"),
            ("-(5 + 5)", "(-(5 + 5))"),
            ("!(true == true)", "(!(true == true))"),
            ("a + add(b * c) + d", "((a + add((b * c))) + d)"),
            (
                "add(a, b, 1, 2 * 3, 4 + 5, add(6, 7 * 8))",
                "add(a, b, 1, (2 * 3), (4 + 5), add(6, (7 * 8)))",
            ),
            (
                "add(a + b + c * d / f + g)",
                "add((((a + b) + ((c * d) / f)) + g))",
            ),
            (
                "a * [1, 2, 3, 4][b * c] * d",
                "((a * ([1, 2, 3, 4][(b * c)])) * d)",
            ),
            (
                "add(a * b[2], b[1], 2 * [1, 2][1])",
                "add((a * (b[2])), (b[1]), (2 * ([1, 2][1])))",
            ),
        ];
        for tt in tests {
            let program = parse(tt.0);
            assert_eq!(format!("{}", program), tt.1);
        }
    }
    #[test]
    fn test_infix_exp_bool() {
        let tests = [
            ("true == true", true, InfixOperator::Eq, true),
            ("true != false", true, InfixOperator::NotEq, false),
            ("false == false", false, InfixOperator::Eq, false),
        ];
        for tt in tests {
            let program = parse(tt.0);
            assert_eq!(program.stms.len(), 1);
            assert_eq!(
                program.stms[0],
                Stm::ExpStm(ExpStm {
                    exp: Exp::InfixExp(InfixExp {
                        left: Box::new(Exp::BoolLiteral(BoolLiteral { value: tt.1 })),
                        op: tt.2,
                        right: Box::new(Exp::BoolLiteral(BoolLiteral { value: tt.3 }))
                    })
                })
            );
        }
    }
    #[test]
    fn test_if_exp() {
        let input = "if (x < y) { x }";
        let program = parse(input);
        assert_eq!(program.stms.len(), 1);
        assert_eq!(
            program.stms[0],
            Stm::ExpStm(ExpStm {
                exp: Exp::IfExp(IfExp {
                    cond: Box::new(Exp::InfixExp(InfixExp {
                        left: Box::new(Exp::IdentExp(IdentExp {
                            value: "x".to_owned()
                        })),
                        op: InfixOperator::Lt,
                        right: Box::new(Exp::IdentExp(IdentExp {
                            value: "y".to_owned()
                        }))
                    })),
                    conseq: BlockStm {
                        stms: vec![Stm::ExpStm(ExpStm {
                            exp: Exp::IdentExp(IdentExp {
                                value: "x".to_owned()
                            })
                        })]
                    },
                    alter: None
                })
            })
        )
    }
    #[test]
    fn test_func_literal() {
        let input = "fn(x, y) { x + y; }";
        let program = parse(input);
        assert_eq!(program.stms.len(), 1);
        assert_eq!(
            program.stms[0],
            Stm::ExpStm(ExpStm {
                exp: Exp::FuncLiteral(FuncLiteral {
                    name: None,
                    params: vec!["x".to_owned(), "y".to_owned()],
                    body: BlockStm {
                        stms: vec![Stm::ExpStm(ExpStm {
                            exp: Exp::InfixExp(InfixExp {
                                left: Box::new(Exp::IdentExp(IdentExp {
                                    value: "x".to_owned()
                                })),
                                op: InfixOperator::Add,
                                right: Box::new(Exp::IdentExp(IdentExp {
                                    value: "y".to_owned()
                                })),
                            })
                        })]
                    }
                })
            })
        )
    }
    #[test]
    fn test_func_params() {
        let tests = [
            ("fn() {};", vec![]),
            ("fn(x) {};", vec!["x"]),
            ("fn(x,y,z) {};", vec!["x", "y", "z"]),
        ];
        for tt in tests {
            let program = parse(tt.0);
            assert_eq!(
                program.stms[0],
                Stm::ExpStm(ExpStm {
                    exp: Exp::FuncLiteral(FuncLiteral {
                        name: None,
                        params: tt
                            .1
                            .into_iter()
                            .map(|p| p.to_owned())
                            .collect::<Vec<Ident>>(),
                        body: BlockStm { stms: vec![] }
                    })
                })
            )
        }
    }
    #[test]
    fn test_call_exp() {
        let input = "add(1, 2 * 3, 4 + 5);";
        let program = parse(input);
        assert_eq!(program.stms.len(), 1);
        assert_eq!(
            program.stms[0],
            Stm::ExpStm(ExpStm {
                exp: Exp::CallExp(CallExp {
                    func: Box::new(Exp::IdentExp(IdentExp {
                        value: "add".to_owned()
                    })),
                    args: vec![
                        Exp::IntLiteral(IntLiteral { value: 1 }),
                        Exp::InfixExp(InfixExp {
                            left: Box::new(Exp::IntLiteral(IntLiteral { value: 2 })),
                            op: InfixOperator::Mul,
                            right: Box::new(Exp::IntLiteral(IntLiteral { value: 3 })),
                        }),
                        Exp::InfixExp(InfixExp {
                            left: Box::new(Exp::IntLiteral(IntLiteral { value: 4 })),
                            op: InfixOperator::Add,
                            right: Box::new(Exp::IntLiteral(IntLiteral { value: 5 })),
                        })
                    ]
                })
            })
        )
    }
    #[test]
    fn test_string_literal() {
        let input = r#""hello world""#;
        let program = parse(input);
        assert_eq!(
            program.stms[0],
            Stm::ExpStm(ExpStm {
                exp: Exp::StringLiteral(StringLiteral {
                    value: "hello world".to_owned()
                })
            })
        )
    }
    #[test]
    fn test_array_literal() {
        let input = "[1, 2 * 2, 3 + 3]";
        let program = parse(input);
        assert_eq!(
            program.stms[0],
            Stm::ExpStm(ExpStm {
                exp: Exp::ArrayLiteral(ArrayLiteral {
                    elems: vec![
                        Exp::IntLiteral(IntLiteral { value: 1 }),
                        Exp::InfixExp(InfixExp {
                            left: Box::new(Exp::IntLiteral(IntLiteral { value: 2 })),
                            op: InfixOperator::Mul,
                            right: Box::new(Exp::IntLiteral(IntLiteral { value: 2 })),
                        }),
                        Exp::InfixExp(InfixExp {
                            left: Box::new(Exp::IntLiteral(IntLiteral { value: 3 })),
                            op: InfixOperator::Add,
                            right: Box::new(Exp::IntLiteral(IntLiteral { value: 3 })),
                        }),
                    ]
                })
            })
        )
    }
    #[test]
    fn test_index_exp() {
        let input = "myArray[1 + 1];";
        let program = parse(input);
        assert_eq!(
            program.stms[0],
            Stm::ExpStm(ExpStm {
                exp: Exp::IndexExp(IndexExp {
                    left: Box::new(Exp::IdentExp(IdentExp {
                        value: "myArray".to_owned()
                    })),
                    index: Box::new(Exp::InfixExp(InfixExp {
                        left: Box::new(Exp::IntLiteral(IntLiteral { value: 1 })),
                        op: InfixOperator::Add,
                        right: Box::new(Exp::IntLiteral(IntLiteral { value: 1 })),
                    }))
                })
            })
        )
    }
    #[test]
    fn test_hash_literal() {
        let input = r#"{"one": 1, "two": 2, "three": 3}"#;
        let program = parse(input);
        assert_eq!(
            program.stms[0],
            Stm::ExpStm(ExpStm {
                exp: Exp::HashLiteral(HashLiteral {
                    pairs: {
                        let mut m = HashMap::new();
                        m.insert("one", 1);
                        m.insert("two", 2);
                        m.insert("three", 3);

                        m.iter()
                            .map(|(&key, &value)| {
                                (
                                    Exp::StringLiteral(StringLiteral {
                                        value: key.to_owned(),
                                    }),
                                    Exp::IntLiteral(IntLiteral { value }),
                                )
                            })
                            .collect()
                    }
                })
            })
        )
    }
    #[test]
    fn test_empty_hash_literal() {
        let input = "{}";
        let program = parse(input);
        assert_eq!(
            program.stms[0],
            Stm::ExpStm(ExpStm {
                exp: Exp::HashLiteral(HashLiteral {
                    pairs: HashMap::new()
                })
            })
        )
    }
    #[test]
    fn test_hash_literals_with_exp() {
        let input = r#"{"one": 0 + 1, "two": 10 - 8, "three": 15 / 5}"#;
        let program = parse(input);
        assert_eq!(
            program.stms[0],
            Stm::ExpStm(ExpStm {
                exp: Exp::HashLiteral(HashLiteral {
                    pairs: {
                        let mut m = HashMap::new();
                        m.insert("one", (0, InfixOperator::Add, 1));
                        m.insert("two", (10, InfixOperator::Sub, 8));
                        m.insert("three", (15, InfixOperator::Div, 5));
                        m.iter()
                            .map(|(&key, value)| {
                                (
                                    Exp::StringLiteral(StringLiteral {
                                        value: key.to_owned(),
                                    }),
                                    Exp::InfixExp(InfixExp {
                                        left: Box::new(Exp::IntLiteral(IntLiteral {
                                            value: value.0,
                                        })),
                                        op: value.1.clone(),
                                        right: Box::new(Exp::IntLiteral(IntLiteral {
                                            value: value.2,
                                        })),
                                    }),
                                )
                            })
                            .collect()
                    }
                })
            })
        )
    }
    #[test]
    fn test_function_literal_with_name() {
        let input = "let myFunction = fn() { };";
        let program = parse(input);
        assert_eq!(program.stms.len(), 1);
        assert_eq!(
            program.stms[0],
            Stm::LetStm(LetStm {
                name: "myFunction".to_owned(),
                value: Exp::FuncLiteral(FuncLiteral {
                    name: Some("myFunction".to_owned()),
                    params: vec![],
                    body: BlockStm { stms: vec![] }
                })
            })
        );
    }
}
