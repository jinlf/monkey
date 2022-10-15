use crate::token::{lookup_ident, Token};
use std::{fmt::Display, str};

pub struct Lexer<'input> {
    input: &'input [u8],
    read_pos: usize,
    pos: usize,
    ch: u8,
}
impl<'input> Lexer<'input> {
    pub fn new(input: &'input str) -> Self {
        let mut l = Self {
            input: input.as_bytes(),
            read_pos: 0,
            pos: 0,
            ch: 0,
        };
        l.read_char();
        l
    }
    pub fn next_token(&mut self) -> Result<Token, LexerError> {
        self.skip_whitespaces();
        let tok = match self.ch {
            b'=' => {
                if self.peek_char() == b'=' {
                    self.read_char();
                    Token::Eq
                } else {
                    Token::Assign
                }
            }
            b'+' => Token::Plus,
            b'-' => Token::Minus,
            b'!' => {
                if self.peek_char() == b'=' {
                    self.read_char();
                    Token::NotEq
                } else {
                    Token::Bang
                }
            }
            b'*' => Token::Asterisk,
            b'/' => Token::Slash,
            b'<' => Token::Lt,
            b'>' => Token::Gt,
            b'(' => Token::LParen,
            b')' => Token::RParen,
            b'{' => Token::LBrace,
            b'}' => Token::RBrace,
            b',' => Token::Comma,
            b';' => Token::Semicolon,
            b'[' => Token::LBracket,
            b']' => Token::RBracket,
            b':' => Token::Colon,
            b'"' => self.read_string()?,
            0 => Token::Eof,
            _ => {
                if self.ch.is_ascii_alphabetic() {
                    return self.read_identifier();
                } else if self.ch.is_ascii_digit() {
                    return self.read_number();
                } else {
                    Token::Illegal
                }
            }
        };
        self.read_char();
        Ok(tok)
    }
    fn read_char(&mut self) {
        self.ch = self.peek_char();
        self.pos = self.read_pos;
        self.read_pos += 1;
    }
    fn peek_char(&self) -> u8 {
        if self.read_pos >= self.input.len() {
            0
        } else {
            self.input[self.read_pos]
        }
    }
    fn read_identifier(&mut self) -> Result<Token, LexerError> {
        let pos = self.pos;
        while self.ch.is_ascii_alphabetic() || self.ch == b'_' {
            self.read_char();
        }
        let ident = str::from_utf8(&self.input[pos..self.pos])?;
        Ok(lookup_ident(ident))
    }
    fn skip_whitespaces(&mut self) {
        while [b' ', b'\t', b'\n', b'\r'].contains(&self.ch) {
            self.read_char();
        }
    }
    fn read_number(&mut self) -> Result<Token, LexerError> {
        let pos = self.pos;
        while self.ch.is_ascii_digit() {
            self.read_char();
        }
        let number = i64::from_str_radix(str::from_utf8(&self.input[pos..self.pos])?, 10)?;
        Ok(Token::Int(number))
    }
    fn read_string(&mut self) -> Result<Token, LexerError> {
        let pos = self.pos + 1;
        self.read_char();
        while self.ch != b'"' && self.ch != 0 {
            self.read_char()
        }
        let s = str::from_utf8(&self.input[pos..self.pos])?;
        Ok(Token::String(s.to_owned()))
    }
}

#[derive(Debug)]
pub enum LexerError {
    Utf8Error(std::str::Utf8Error),
    ParseIntError(std::num::ParseIntError),
}
impl Display for LexerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Utf8Error(e) => write!(f, "{}", e),
            Self::ParseIntError(e) => write!(f, "{}", e),
        }
    }
}
impl From<std::str::Utf8Error> for LexerError {
    fn from(e: std::str::Utf8Error) -> Self {
        Self::Utf8Error(e)
    }
}
impl From<std::num::ParseIntError> for LexerError {
    fn from(e: std::num::ParseIntError) -> Self {
        Self::ParseIntError(e)
    }
}

#[cfg(test)]
mod tests {
    use crate::token::Token;

    use super::*;

    #[test]
    fn test_next_token() {
        let input = r#"=+(){},;
let five = 5;
let ten = 10;
let add = fn(x, y) {
    x + y;
};
let result = add(five, ten);
!-/*5;
5 < 10 > 5;

if (5 < 10) {
    return true;
} else {
    return false;
}

10 == 10;
10 != 9;
"foobar"
"foo bar"
[1, 2];
{"foo": "bar"}
"#;
        let tests = [
            Token::Assign,
            Token::Plus,
            Token::LParen,
            Token::RParen,
            Token::LBrace,
            Token::RBrace,
            Token::Comma,
            Token::Semicolon,
            Token::Let,
            Token::Ident("five".to_owned()),
            Token::Assign,
            Token::Int(5),
            Token::Semicolon,
            Token::Let,
            Token::Ident("ten".to_owned()),
            Token::Assign,
            Token::Int(10),
            Token::Semicolon,
            Token::Let,
            Token::Ident("add".to_owned()),
            Token::Assign,
            Token::Function,
            Token::LParen,
            Token::Ident("x".to_owned()),
            Token::Comma,
            Token::Ident("y".to_owned()),
            Token::RParen,
            Token::LBrace,
            Token::Ident("x".to_owned()),
            Token::Plus,
            Token::Ident("y".to_owned()),
            Token::Semicolon,
            Token::RBrace,
            Token::Semicolon,
            Token::Let,
            Token::Ident("result".to_owned()),
            Token::Assign,
            Token::Ident("add".to_owned()),
            Token::LParen,
            Token::Ident("five".to_owned()),
            Token::Comma,
            Token::Ident("ten".to_owned()),
            Token::RParen,
            Token::Semicolon,
            Token::Bang,
            Token::Minus,
            Token::Slash,
            Token::Asterisk,
            Token::Int(5),
            Token::Semicolon,
            Token::Int(5),
            Token::Lt,
            Token::Int(10),
            Token::Gt,
            Token::Int(5),
            Token::Semicolon,
            Token::If,
            Token::LParen,
            Token::Int(5),
            Token::Lt,
            Token::Int(10),
            Token::RParen,
            Token::LBrace,
            Token::Return,
            Token::True,
            Token::Semicolon,
            Token::RBrace,
            Token::Else,
            Token::LBrace,
            Token::Return,
            Token::False,
            Token::Semicolon,
            Token::RBrace,
            Token::Int(10),
            Token::Eq,
            Token::Int(10),
            Token::Semicolon,
            Token::Int(10),
            Token::NotEq,
            Token::Int(9),
            Token::Semicolon,
            Token::String("foobar".to_owned()),
            Token::String("foo bar".to_owned()),
            Token::LBracket,
            Token::Int(1),
            Token::Comma,
            Token::Int(2),
            Token::RBracket,
            Token::Semicolon,
            Token::LBrace,
            Token::String("foo".to_owned()),
            Token::Colon,
            Token::String("bar".to_owned()),
            Token::RBrace,
            Token::Eof,
        ];

        let mut l = Lexer::new(input);
        for tt in tests {
            assert_eq!(l.next_token().unwrap(), tt);
        }
    }
}
