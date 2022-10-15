use std::{collections::HashMap, fmt::Display, hash::Hash};

pub struct Program {
    pub stms: Vec<Stm>,
}

#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub enum Stm {
    LetStm(LetStm),
    ReturnStm(ReturnStm),
    ExpStm(ExpStm),
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub enum Exp {
    IdentExp(IdentExp),
    IntLiteral(IntLiteral),
    BoolLiteral(BoolLiteral),
    PrefixExp(PrefixExp),
    InfixExp(InfixExp),
    IfExp(IfExp),
    FuncLiteral(FuncLiteral),
    CallExp(CallExp),
    StringLiteral(StringLiteral),
    ArrayLiteral(ArrayLiteral),
    IndexExp(IndexExp),
    HashLiteral(HashLiteral),
}
pub type Ident = String;
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct LetStm {
    pub name: Ident,
    pub value: Exp,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct IdentExp {
    pub value: String,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct ReturnStm {
    pub return_value: Exp,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct ExpStm {
    pub exp: Exp,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct IntLiteral {
    pub value: i64,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct BoolLiteral {
    pub value: bool,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct PrefixExp {
    pub op: PrefixOperator,
    pub right: Box<Exp>,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct InfixExp {
    pub left: Box<Exp>,
    pub op: InfixOperator,
    pub right: Box<Exp>,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub enum PrefixOperator {
    Minus,
    Bang,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq, Clone)]
pub enum InfixOperator {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Gt,
    Eq,
    NotEq,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct IfExp {
    pub cond: Box<Exp>,
    pub conseq: BlockStm,
    pub alter: Option<BlockStm>,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct BlockStm {
    pub stms: Vec<Stm>,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct FuncLiteral {
    pub name: Option<Ident>,
    pub params: Vec<Ident>,
    pub body: BlockStm,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct CallExp {
    pub func: Box<Exp>,
    pub args: Vec<Exp>,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct StringLiteral {
    pub value: String,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct ArrayLiteral {
    pub elems: Vec<Exp>,
}
#[cfg_attr(test, derive(PartialOrd))]
#[derive(PartialEq, Debug, Hash, Eq)]
pub struct IndexExp {
    pub left: Box<Exp>,
    pub index: Box<Exp>,
}
#[derive(PartialEq, Debug, Eq)]
pub struct HashLiteral {
    pub pairs: HashMap<Exp, Exp>,
}
impl Hash for HashLiteral {
    fn hash<H: std::hash::Hasher>(&self, _: &mut H) {
        todo!()
    }
}
#[cfg(test)]
impl PartialOrd for HashLiteral {
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        todo!()
    }
}

impl Exp {
    pub fn get_type(&self) -> &'static str {
        match self {
            Self::IdentExp(_) => "IDENTIFIER",
            Self::IntLiteral(_) => "INTEGER",
            Self::BoolLiteral(_) => "BOOLEAN",
            Self::PrefixExp(_) => "PREFIX",
            Self::InfixExp(_) => "INFIX",
            Self::IfExp(_) => "IF",
            Self::FuncLiteral(_) => "FUNCTION",
            Self::CallExp(_) => "CALL",
            Self::StringLiteral(_) => "STRING",
            Self::ArrayLiteral(_) => "ARRAY",
            Self::IndexExp(_) => "INDEX",
            Self::HashLiteral(_) => "HASH",
        }
    }
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.stms.iter().try_for_each(|stm| write!(f, "{}", stm))
    }
}
impl Display for Stm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LetStm(let_stm) => write!(f, "{}", let_stm),
            Self::ReturnStm(return_stm) => write!(f, "{}", return_stm),
            Self::ExpStm(exp_stm) => write!(f, "{}", exp_stm),
        }
    }
}
impl Display for Exp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exp::IdentExp(ident_exp) => write!(f, "{}", ident_exp),
            Exp::IntLiteral(int_literal) => write!(f, "{}", int_literal),
            Exp::BoolLiteral(bool_literal) => write!(f, "{}", bool_literal),
            Exp::PrefixExp(prefix_exp) => write!(f, "{}", prefix_exp),
            Exp::InfixExp(infix_exp) => write!(f, "{}", infix_exp),
            Exp::IfExp(if_exp) => write!(f, "{}", if_exp),
            Exp::FuncLiteral(func_literal) => write!(f, "{}", func_literal),
            Exp::CallExp(call_exp) => write!(f, "{}", call_exp),
            Exp::StringLiteral(string_literal) => write!(f, "{}", string_literal),
            Exp::ArrayLiteral(array_literal) => write!(f, "{}", array_literal),
            Exp::IndexExp(index_exp) => write!(f, "{}", index_exp),
            Exp::HashLiteral(hash_literal) => write!(f, "{}", hash_literal),
        }
    }
}
impl Display for LetStm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "let {} = {};", self.name, self.value)
    }
}
impl Display for ReturnStm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "return {};", self.return_value)
    }
}
impl Display for ExpStm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.exp)
    }
}
impl Display for IdentExp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}
impl Display for IntLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}
impl Display for BoolLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}
impl Display for PrefixExp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}{})", self.op, self.right)
    }
}
impl Display for PrefixOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Minus => write!(f, "-"),
            Self::Bang => write!(f, "!"),
        }
    }
}
impl Display for InfixExp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {} {})", self.left, self.op, self.right)
    }
}
impl Display for InfixOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(f, "+"),
            Self::Sub => write!(f, "-"),
            Self::Mul => write!(f, "*"),
            Self::Div => write!(f, "/"),
            Self::Lt => write!(f, "<"),
            Self::Gt => write!(f, ">"),
            Self::Eq => write!(f, "=="),
            Self::NotEq => write!(f, "!="),
        }
    }
}
impl Display for IfExp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "if ({}) {} {}",
            self.cond,
            self.conseq,
            self.alter
                .as_ref()
                .map_or(String::new(), |stm| format!("else {}", stm))
        )
    }
}
impl Display for BlockStm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{{}}}",
            self.stms
                .iter()
                .fold(String::new(), |acc, x| acc + format!("{}", x).as_str())
        )
    }
}
impl Display for FuncLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fn({}){}", self.params.join(", "), self.body)
    }
}
impl Display for CallExp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}({})",
            self.func,
            self.args
                .iter()
                .map(|arg| format!("{}", arg))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}
impl Display for StringLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, r#""{}""#, self.value)
    }
}
impl Display for ArrayLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]",
            self.elems
                .iter()
                .map(|elem| format!("{}", elem))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}
impl Display for IndexExp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}[{}])", self.left, self.index)
    }
}
impl Display for HashLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{{}}}",
            self.pairs
                .iter()
                .map(|(key, value)| format!("{}:{}", key, value))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string() {
        let program = Program {
            stms: vec![Stm::LetStm(LetStm {
                name: "myVar".to_owned(),
                value: Exp::IdentExp(IdentExp {
                    value: "anotherVar".to_owned(),
                }),
            })],
        };
        assert_eq!(format!("{}", program), "let myVar = anotherVar;");
    }
}
