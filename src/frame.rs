use crate::{code::Instructions, object::Closure};

#[derive(Default, Clone)]
pub struct Frame {
    pub cl: Closure,
    pub ip: usize,
    pub base_pointer: u16,
}
impl Frame {
    pub fn new(cl: Closure, base_pointer: u16) -> Self {
        Self {
            cl,
            ip: 0,
            base_pointer,
        }
    }
    pub fn instructions(&self) -> &Instructions {
        &self.cl.func.instructions
    }
}
