use crate::{builtin::Builtin, code::Instructions};
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    fmt::Display,
    hash::{Hash, Hasher},
    rc::Rc,
};

#[derive(Debug, PartialEq, Hash, Eq)]
pub enum Object {
    Integer(Integer),
    Boolean(Boolean),
    Null,
    Error(Error),
    String(StringObj),
    Array(Array),
    Hash(HashObj),
    CompiledFunction(CompiledFunction),
    Builtin(Builtin),
    Closure(Closure),
}

#[derive(Debug, PartialEq, Hash, Eq)]
pub struct Integer {
    pub value: i64,
}
#[derive(Debug, PartialEq, Hash, Eq)]
pub struct Boolean {
    pub value: bool,
}
#[derive(Debug, PartialEq, Hash, Eq)]
pub struct ReturnValue {
    pub value: Box<Object>,
}
#[derive(Debug, PartialEq, Hash, Eq)]
pub struct Error {
    pub msg: String,
}
#[derive(Debug, PartialEq, Hash, Eq)]
pub struct StringObj {
    pub value: String,
}
#[derive(Debug, PartialEq, Hash, Eq)]
pub struct Array {
    pub elems: Vec<Rc<Object>>,
}
#[derive(Debug, PartialEq, Eq)]
pub struct HashObj {
    pub pairs: HashMap<HashKey, HashPair>,
}
impl Hash for HashObj {
    fn hash<H: std::hash::Hasher>(&self, _: &mut H) {
        todo!()
    }
}

#[derive(Debug, PartialEq, Hash, Eq)]
pub enum ObjectType {
    Integer,
    Boolean,
    String,
}
#[derive(Debug, PartialEq, Hash, Eq)]
pub struct HashKey {
    object_type: ObjectType,
    value: u64,
}

#[derive(Debug, PartialEq, Hash, Eq)]
pub struct HashPair {
    pub key: Rc<Object>,
    pub value: Rc<Object>,
}

#[derive(Debug, PartialEq, Hash, Eq, Default, Clone)]
pub struct CompiledFunction {
    pub instructions: Instructions,
    pub num_locals: u16,
    pub num_params: u16,
}

#[derive(Debug, PartialEq, Hash, Eq, Default, Clone)]
pub struct Closure {
    pub func: CompiledFunction,
    pub free: Vec<Rc<Object>>,
}

impl Object {
    pub fn get_type(&self) -> &'static str {
        match self {
            Self::Integer(_) => "INTEGER",
            Self::Boolean(_) => "BOOLEAN",
            Self::Null => "NULL",
            // Self::ReturnValue(_) => "RETURN_VALUE",
            Self::Error(_) => "ERROR",
            //Self::Function(_) => "FUNCTION",
            Self::String(_) => "STRING",
            Self::Array(_) => "ARRAY",
            Self::Hash(_) => "HASH",
            Self::CompiledFunction(_) => "COMPILED_FUNCTION",
            Self::Builtin(_) => "BUILTIN",
            Self::Closure(_) => "CLOSURE",
        }
    }
}

pub fn new_int(value: i64) -> Rc<Object> {
    Rc::new(Object::Integer(Integer { value }))
}
pub fn new_bool(value: bool) -> Rc<Object> {
    Rc::new(Object::Boolean(Boolean { value }))
}
pub fn new_null() -> Rc<Object> {
    Rc::new(Object::Null)
}
pub fn new_string(value: &str) -> Rc<Object> {
    Rc::new(Object::String(StringObj {
        value: value.to_owned(),
    }))
}
pub fn new_array(elems: Vec<Rc<Object>>) -> Rc<Object> {
    Rc::new(Object::Array(Array { elems }))
}
pub fn new_hash(pairs: HashMap<HashKey, HashPair>) -> Rc<Object> {
    Rc::new(Object::Hash(HashObj { pairs }))
}
pub fn new_func(ins: Vec<Instructions>, num_locals: u16, num_params: u16) -> Rc<Object> {
    Rc::new(Object::CompiledFunction(CompiledFunction {
        instructions: ins.into_iter().flatten().collect(),
        num_locals,
        num_params,
    }))
}
pub fn new_error(msg: String) -> Rc<Object> {
    Rc::new(Object::Error(Error { msg }))
}
pub fn new_closure(func: CompiledFunction, free: Vec<Rc<Object>>) -> Rc<Object> {
    Rc::new(Object::Closure(Closure { func, free }))
}

impl Display for Object {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Integer(Integer { value }) => write!(f, "{}", value),
            Self::Boolean(Boolean { value }) => write!(f, "{}", value),
            Self::Null => write!(f, "null"),
            // Self::ReturnValue(return_value) => write!(f, "{}", return_value),
            Self::Error(err) => write!(f, "{}", err),
            //Self::Function(_) => write!(f, "FUNCTION"),
            Self::String(StringObj { value }) => write!(f, r#""{}""#, value),
            Self::Array(array) => write!(f, "{}", array),
            Self::Hash(hash) => write!(f, "{}", hash),
            Self::CompiledFunction(cf) => write!(f, "{}", cf),
            Self::Builtin(_) => write!(f, "BUILTIN"),
            Self::Closure(_) => write!(f, "CLOSURE"),
        }
    }
}
impl Display for ReturnValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}
impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}
impl Display for Array {
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
impl Display for HashObj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{{}}}",
            self.pairs
                .iter()
                .map(|(_, hash_pair)| { format!("{}:{}", hash_pair.key, hash_pair.value) })
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}
impl Display for CompiledFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let addr = std::ptr::addr_of!(self);
        write!(f, "CompiledFunction:{:p}", addr)
    }
}

#[derive(Debug)]
pub enum ObjectError {
    UnusableAsHashKey(&'static str),
}
impl Display for ObjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnusableAsHashKey(t) => write!(f, "unusable as hash key: {}", t),
        }
    }
}

pub trait MayHasHashKey {
    fn hash_key(&self) -> Result<HashKey, ObjectError>;
}
impl MayHasHashKey for Object {
    fn hash_key(&self) -> Result<HashKey, ObjectError> {
        match self {
            Object::Integer(i) => i.hash_key(),
            Object::String(s) => s.hash_key(),
            Object::Boolean(b) => b.hash_key(),
            _ => Err(ObjectError::UnusableAsHashKey(self.get_type())),
        }
    }
}

impl MayHasHashKey for Integer {
    fn hash_key(&self) -> Result<HashKey, ObjectError> {
        Ok(HashKey {
            object_type: ObjectType::Integer,
            value: self.value as u64,
        })
    }
}
impl MayHasHashKey for StringObj {
    fn hash_key(&self) -> Result<HashKey, ObjectError> {
        Ok(HashKey {
            object_type: ObjectType::String,
            value: {
                let mut hasher = DefaultHasher::new();
                self.value.hash(&mut hasher);
                hasher.finish()
            },
        })
    }
}
impl MayHasHashKey for Boolean {
    fn hash_key(&self) -> Result<HashKey, ObjectError> {
        Ok(HashKey {
            object_type: ObjectType::Boolean,
            value: self.value as u64,
        })
    }
}
