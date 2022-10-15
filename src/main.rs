#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]

use crate::repl::start;

mod ast;
mod builtin;
mod code;
mod compiler;
mod frame;
mod lexer;
mod object;
mod parser;
mod repl;
mod symbol_table;
mod token;
mod vm;

#[cfg(test)]
mod test_util;

fn main() -> std::io::Result<()> {
    println!("Hello, This is the Monkey programming language!");
    //start_tokens(&mut std::io::stdin(), &mut std::io::stdout())
    //start_ast(&mut std::io::stdin(), &mut std::io::stdout())
    start(&mut std::io::stdin(), &mut std::io::stdout())
}
