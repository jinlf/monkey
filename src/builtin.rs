use crate::object::{self, Object};
use std::{fmt::Debug, hash::Hash, rc::Rc};

type BuiltinFunc = Box<dyn Fn(Vec<Rc<Object>>) -> Rc<Object>>;

pub struct Builtin {
    pub func: BuiltinFunc,
}
impl Debug for Builtin {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}
impl Hash for Builtin {
    fn hash<H: std::hash::Hasher>(&self, _: &mut H) {
        todo!()
    }
}
impl PartialEq for Builtin {
    fn eq(&self, _: &Self) -> bool {
        todo!()
    }
}
impl Eq for Builtin {}
#[cfg(test)]
impl PartialOrd for Builtin {
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        todo!()
    }
}

pub const BUILTIN_NAMES: &[&'static str] = &["len", "puts", "first", "last", "rest", "push"];

pub fn builtins(index: u8) -> Option<Builtin> {
    let index = index as usize;
    if index >= BUILTIN_NAMES.len() {
        return None;
    }
    Some(match BUILTIN_NAMES[index] {
        "len" => Builtin {
            func: Box::new(|args| {
                if args.len() != 1 {
                    return object::new_error(format!(
                        "wrong number of arguments. got={}, want=1",
                        args.len()
                    ));
                }
                match &*args[0] {
                    Object::Array(a) => object::new_int(a.elems.len() as i64),
                    Object::String(s) => object::new_int(s.value.len() as i64),
                    _ => object::new_error(format!(
                        "argument to `len` not supported, got {}",
                        args[0].get_type()
                    )),
                }
            }),
        },
        "puts" => Builtin {
            func: Box::new(|args| {
                args.iter().for_each(|arg| println!("{}", arg));
                object::new_null()
            }),
        },
        "first" => Builtin {
            func: Box::new(|args| {
                if args.len() != 1 {
                    return object::new_error(format!(
                        "wrong number of arguments. got={}, want=1",
                        args.len()
                    ));
                }
                match &*args[0] {
                    Object::Array(a) => {
                        if a.elems.len() > 0 {
                            Rc::clone(&a.elems[0])
                        } else {
                            object::new_null()
                        }
                    }
                    _ => object::new_error(format!(
                        "argument to `first` must be ARRAY, got {}",
                        args[0].get_type()
                    )),
                }
            }),
        },
        "last" => Builtin {
            func: Box::new(|args| {
                if args.len() != 1 {
                    return object::new_error(format!(
                        "wrong number of arguments. got={}, want=1",
                        args.len()
                    ));
                }
                match &*args[0] {
                    Object::Array(a) => {
                        if a.elems.len() > 0 {
                            Rc::clone(a.elems.last().unwrap())
                        } else {
                            object::new_null()
                        }
                    }
                    _ => object::new_error(format!(
                        "argument to `last` must be ARRAY, got {}",
                        args[0].get_type()
                    )),
                }
            }),
        },
        "rest" => Builtin {
            func: Box::new(|args| {
                if args.len() != 1 {
                    return object::new_error(format!(
                        "wrong number of arguments. got={}, want=1",
                        args.len()
                    ));
                }
                match &*args[0] {
                    Object::Array(a) => {
                        if a.elems.len() > 0 {
                            object::new_array(a.elems[1..].to_vec())
                        } else {
                            object::new_null()
                        }
                    }
                    _ => object::new_error(format!(
                        "argument to `rest` must be ARRAY, got {}",
                        args[0].get_type()
                    )),
                }
            }),
        },
        "push" => Builtin {
            func: Box::new(|args| {
                if args.len() != 2 {
                    return object::new_error(format!(
                        "wrong number of arguments. got={}, want=2",
                        args.len()
                    ));
                }
                match &*args[0] {
                    Object::Array(a) => {
                        let mut new_elems = a.elems.clone();
                        new_elems.push(Rc::clone(&args[1]));
                        object::new_array(new_elems)
                    }
                    _ => object::new_error(format!(
                        "argument to `push` must be ARRAY, got {}",
                        args[0].get_type()
                    )),
                }
            }),
        },
        _ => return None,
    })
}
