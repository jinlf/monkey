use std::{cell::RefCell, collections::HashMap, rc::Rc};

#[derive(Debug, PartialEq)]
pub enum SymbolScope {
    Global,
    Local,
    Builtin,
    Free,
    Function,
}

#[derive(Debug, PartialEq)]
pub struct Symbol {
    name: String,
    pub scope: SymbolScope,
    pub index: u16,
}
impl Symbol {
    fn new(name: &str, scope: SymbolScope, index: u16) -> Rc<Self> {
        Rc::new(Self {
            name: name.to_owned(),
            scope,
            index,
        })
    }
}

#[derive(PartialEq, Debug)]
pub struct SymbolTable {
    store: HashMap<String, Rc<Symbol>>,
    pub num_defintions: u16,
    pub outer: Option<Rc<RefCell<SymbolTable>>>,
    pub free_symbols: Vec<Rc<Symbol>>,
}
impl SymbolTable {
    pub fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            store: HashMap::new(),
            num_defintions: 0,
            outer: None,
            free_symbols: vec![],
        }))
    }
    pub fn new_enclosed(outer: Rc<RefCell<SymbolTable>>) -> Rc<RefCell<Self>> {
        let s = SymbolTable::new();
        s.borrow_mut().outer = Some(outer);
        s
    }
    pub fn define(&mut self, name: &str) -> Rc<Symbol> {
        let scope = match self.outer {
            Some(_) => SymbolScope::Local,
            None => SymbolScope::Global,
        };
        let symbol = Symbol::new(name, scope, self.num_defintions);
        self.store.insert(name.to_owned(), Rc::clone(&symbol));
        self.num_defintions += 1;
        symbol
    }
    pub fn resolve(&mut self, name: &str) -> Option<Rc<Symbol>> {
        match self.store.get(name) {
            Some(s) => Some(Rc::clone(s)),
            None => match &self.outer {
                Some(outer) => {
                    let obj = match outer.borrow_mut().resolve(name) {
                        Some(obj) => match obj.scope {
                            SymbolScope::Global | SymbolScope::Builtin => {
                                return Some(Rc::clone(&obj))
                            }
                            _ => obj,
                        },
                        _ => return None,
                    };
                    let free = self.define_free(obj);
                    Some(free)
                }
                None => None,
            },
        }
    }
    pub fn define_builtin(&mut self, index: usize, name: &str) -> Rc<Symbol> {
        let symbol = Symbol::new(name, SymbolScope::Builtin, index as u16);
        self.store.insert(name.to_owned(), Rc::clone(&symbol));
        symbol
    }
    pub fn define_free(&mut self, original: Rc<Symbol>) -> Rc<Symbol> {
        self.free_symbols.push(Rc::clone(&original));

        let symbol = Symbol::new(
            original.name.as_str(),
            SymbolScope::Free,
            (self.free_symbols.len() - 1) as u16,
        );
        self.store
            .insert(original.name.to_owned(), Rc::clone(&symbol));
        symbol
    }
    pub fn define_function_name(&mut self, name: &str) -> Rc<Symbol> {
        let symbol = Symbol::new(name, SymbolScope::Function, 0);
        self.store.insert(name.to_owned(), Rc::clone(&symbol));
        symbol
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_define() {
        let mut expected = HashMap::new();
        expected.insert("a", Symbol::new("a", SymbolScope::Global, 0));
        expected.insert("b", Symbol::new("b", SymbolScope::Global, 1));
        expected.insert("c", Symbol::new("c", SymbolScope::Local, 0));
        expected.insert("d", Symbol::new("d", SymbolScope::Local, 1));
        expected.insert("e", Symbol::new("e", SymbolScope::Local, 0));
        expected.insert("f", Symbol::new("f", SymbolScope::Local, 1));

        let global = SymbolTable::new();
        let a = global.borrow_mut().define("a");
        assert_eq!(a, expected["a"]);
        let b = global.borrow_mut().define("b");
        assert_eq!(b, expected["b"]);
        let first_local = SymbolTable::new_enclosed(global);
        let c = first_local.borrow_mut().define("c");
        assert_eq!(c, expected["c"]);
        let d = first_local.borrow_mut().define("d");
        assert_eq!(d, expected["d"]);
        let second_local = SymbolTable::new_enclosed(first_local);
        let e = second_local.borrow_mut().define("e");
        assert_eq!(e, expected["e"]);
        let f = second_local.borrow_mut().define("f");
        assert_eq!(f, expected["f"]);
    }
    #[test]
    fn test_resolve_global() {
        let global = SymbolTable::new();
        global.borrow_mut().define("a");
        global.borrow_mut().define("b");

        let expected = [
            Symbol::new("a", SymbolScope::Global, 0),
            Symbol::new("b", SymbolScope::Global, 1),
        ];
        for sym in expected {
            assert_eq!(sym, global.borrow_mut().resolve(sym.name.as_str()).unwrap())
        }
    }
    #[test]
    fn test_resolve_local() {
        let global = SymbolTable::new();
        global.borrow_mut().define("a");
        global.borrow_mut().define("b");

        let local = SymbolTable::new_enclosed(global);
        local.borrow_mut().define("c");
        local.borrow_mut().define("d");

        let expected = [
            Symbol::new("a", SymbolScope::Global, 0),
            Symbol::new("b", SymbolScope::Global, 1),
            Symbol::new("c", SymbolScope::Local, 0),
            Symbol::new("d", SymbolScope::Local, 1),
        ];
        for sym in expected {
            assert_eq!(sym, local.borrow_mut().resolve(sym.name.as_str()).unwrap());
        }
    }
    #[test]
    fn test_resolve_nested_local() {
        let global = SymbolTable::new();
        global.borrow_mut().define("a");
        global.borrow_mut().define("b");

        let first_local = SymbolTable::new_enclosed(global);
        first_local.borrow_mut().define("c");
        first_local.borrow_mut().define("d");

        let second_local = SymbolTable::new_enclosed(Rc::clone(&first_local));
        second_local.borrow_mut().define("e");
        second_local.borrow_mut().define("f");

        let tests = [
            (
                Rc::clone(&first_local),
                vec![
                    Symbol::new("a", SymbolScope::Global, 0),
                    Symbol::new("b", SymbolScope::Global, 1),
                    Symbol::new("c", SymbolScope::Local, 0),
                    Symbol::new("d", SymbolScope::Local, 1),
                ],
            ),
            (
                Rc::clone(&second_local),
                vec![
                    Symbol::new("a", SymbolScope::Global, 0),
                    Symbol::new("b", SymbolScope::Global, 1),
                    Symbol::new("e", SymbolScope::Local, 0),
                    Symbol::new("f", SymbolScope::Local, 1),
                ],
            ),
        ];
        for tt in tests {
            for sym in tt.1 {
                assert_eq!(tt.0.borrow_mut().resolve(sym.name.as_str()).unwrap(), sym);
            }
        }
    }
    #[test]
    fn test_define_resolve_builtins() {
        let global = SymbolTable::new();
        let first_local = SymbolTable::new_enclosed(Rc::clone(&global));
        let second_local = SymbolTable::new_enclosed(Rc::clone(&first_local));

        let expected = [
            Symbol::new("a", SymbolScope::Builtin, 0),
            Symbol::new("c", SymbolScope::Builtin, 1),
            Symbol::new("e", SymbolScope::Builtin, 2),
            Symbol::new("f", SymbolScope::Builtin, 3),
        ];

        expected.iter().enumerate().for_each(|(index, e)| {
            global.borrow_mut().define_builtin(index, e.name.as_str());
        });
        for table in [global, first_local, second_local] {
            for sym in &expected {
                assert_eq!(table.borrow_mut().resolve(sym.name.as_str()).unwrap(), *sym);
            }
        }
    }
    #[test]
    fn test_resolve_free() {
        let global = SymbolTable::new();
        global.borrow_mut().define("a");
        global.borrow_mut().define("b");
        let first_local = SymbolTable::new_enclosed(Rc::clone(&global));
        first_local.borrow_mut().define("c");
        first_local.borrow_mut().define("d");
        let second_local = SymbolTable::new_enclosed(Rc::clone(&first_local));
        second_local.borrow_mut().define("e");
        second_local.borrow_mut().define("f");

        let tests = [
            (
                first_local,
                vec![
                    Symbol::new("a", SymbolScope::Global, 0),
                    Symbol::new("b", SymbolScope::Global, 1),
                    Symbol::new("c", SymbolScope::Local, 0),
                    Symbol::new("d", SymbolScope::Local, 1),
                ],
                vec![],
            ),
            (
                second_local,
                vec![
                    Symbol::new("a", SymbolScope::Global, 0),
                    Symbol::new("b", SymbolScope::Global, 1),
                    Symbol::new("c", SymbolScope::Free, 0),
                    Symbol::new("d", SymbolScope::Free, 1),
                    Symbol::new("e", SymbolScope::Local, 0),
                    Symbol::new("f", SymbolScope::Local, 1),
                ],
                vec![
                    Symbol::new("c", SymbolScope::Local, 0),
                    Symbol::new("d", SymbolScope::Local, 1),
                ],
            ),
        ];
        for tt in tests {
            for sym in &tt.1 {
                let result = tt.0.borrow_mut().resolve(&sym.name).unwrap();
                assert_eq!(result, *sym);
            }
            assert_eq!(tt.0.borrow().free_symbols.len(), tt.2.len());
            for (i, sym) in tt.2.iter().enumerate() {
                let result = &tt.0.borrow().free_symbols[i];
                assert_eq!(result, sym)
            }
        }
    }
    #[test]
    fn test_resolve_unresolvable_free() {
        let global = SymbolTable::new();
        global.borrow_mut().define("a");
        let first_local = SymbolTable::new_enclosed(Rc::clone(&global));
        first_local.borrow_mut().define("c");
        let second_local = SymbolTable::new_enclosed(Rc::clone(&first_local));
        second_local.borrow_mut().define("e");
        second_local.borrow_mut().define("f");

        let expected = vec![
            Symbol::new("a", SymbolScope::Global, 0),
            Symbol::new("c", SymbolScope::Free, 0),
            Symbol::new("e", SymbolScope::Local, 0),
            Symbol::new("f", SymbolScope::Local, 1),
        ];

        for sym in expected {
            let result = second_local.borrow_mut().resolve(&sym.name).unwrap();
            assert_eq!(result, sym);
        }
        let expected_unresolvable = ["b", "d"];
        for name in expected_unresolvable {
            assert!(second_local.borrow_mut().resolve(name).is_none())
        }
    }
    #[test]
    fn test_define_and_resolve_function_name() {
        let global = SymbolTable::new();
        global.borrow_mut().define_function_name("a");

        let expected = Symbol::new("a", SymbolScope::Function, 0);

        let result = global.borrow_mut().resolve(expected.name.as_str()).unwrap();
        assert_eq!(result, expected);
    }
    #[test]
    fn test_shadowing_function_name() {
        let global = SymbolTable::new();
        global.borrow_mut().define_function_name("a");
        global.borrow_mut().define("a");

        let expected = Symbol::new("a", SymbolScope::Global, 0);
        let result = global.borrow_mut().resolve(expected.name.as_str()).unwrap();
        assert_eq!(result, expected)
    }
}
