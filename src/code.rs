use std::fmt::Display;

pub type Instructions = Vec<u8>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Opcode {
    OpConstant,
    OpAdd,
    OpPop,
    OpSub,
    OpMul,
    OpDiv,
    OpTrue,
    OpFalse,
    OpEqual,
    OpNotEqual,
    OpGreaterThan,
    OpMinus,
    OpBang,
    OpJumpNotTruthy,
    OpJump,
    OpNull,
    OpGetGlobal,
    OpSetGlobal,
    OpArray,
    OpHash,
    OpIndex,
    OpCall,
    OpReturnValue,
    OpReturn,
    OpGetLocal,
    OpSetLocal,
    OpGetBuiltin,
    OpClosure,
    OpGetFree,
    OpCurrentClosure,
}
impl Into<u8> for Opcode {
    fn into(self) -> u8 {
        match self {
            Opcode::OpConstant => 0,
            Opcode::OpAdd => 1,
            Opcode::OpPop => 2,
            Opcode::OpSub => 3,
            Opcode::OpMul => 4,
            Opcode::OpDiv => 5,
            Opcode::OpTrue => 6,
            Opcode::OpFalse => 7,
            Opcode::OpEqual => 8,
            Opcode::OpNotEqual => 9,
            Opcode::OpGreaterThan => 10,
            Opcode::OpMinus => 11,
            Opcode::OpBang => 12,
            Opcode::OpJumpNotTruthy => 13,
            Opcode::OpJump => 14,
            Opcode::OpNull => 15,
            Opcode::OpGetGlobal => 16,
            Opcode::OpSetGlobal => 17,
            Opcode::OpArray => 18,
            Opcode::OpHash => 19,
            Opcode::OpIndex => 20,
            Opcode::OpCall => 21,
            Opcode::OpReturnValue => 22,
            Opcode::OpReturn => 23,
            Opcode::OpGetLocal => 24,
            Opcode::OpSetLocal => 25,
            Opcode::OpGetBuiltin => 26,
            Opcode::OpClosure => 27,
            Opcode::OpGetFree => 28,
            Opcode::OpCurrentClosure => 29,
        }
    }
}
impl TryFrom<u8> for Opcode {
    type Error = CodeError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => Self::OpConstant,
            1 => Self::OpAdd,
            2 => Self::OpPop,
            3 => Self::OpSub,
            4 => Self::OpMul,
            5 => Self::OpDiv,
            6 => Self::OpTrue,
            7 => Self::OpFalse,
            8 => Self::OpEqual,
            9 => Self::OpNotEqual,
            10 => Self::OpGreaterThan,
            11 => Self::OpMinus,
            12 => Self::OpBang,
            13 => Self::OpJumpNotTruthy,
            14 => Self::OpJump,
            15 => Self::OpNull,
            16 => Self::OpGetGlobal,
            17 => Self::OpSetGlobal,
            18 => Self::OpArray,
            19 => Self::OpHash,
            20 => Self::OpIndex,
            21 => Self::OpCall,
            22 => Self::OpReturnValue,
            23 => Self::OpReturn,
            24 => Self::OpGetLocal,
            25 => Self::OpSetLocal,
            26 => Self::OpGetBuiltin,
            27 => Self::OpClosure,
            28 => Self::OpGetFree,
            29 => Self::OpCurrentClosure,
            _ => return Err(CodeError::OpcodeUndefined(value)),
        })
    }
}
pub struct Definition {
    pub name: &'static str,
    pub operand_widths: Vec<u16>,
}
fn definitions(opcode: Opcode) -> Definition {
    match opcode {
        Opcode::OpConstant => Definition {
            name: "OpConstant",
            operand_widths: vec![2],
        },
        Opcode::OpAdd => Definition {
            name: "OpAdd",
            operand_widths: vec![],
        },
        Opcode::OpPop => Definition {
            name: "OpPop",
            operand_widths: vec![],
        },
        Opcode::OpSub => Definition {
            name: "OpSub",
            operand_widths: vec![],
        },
        Opcode::OpMul => Definition {
            name: "OpMul",
            operand_widths: vec![],
        },
        Opcode::OpDiv => Definition {
            name: "OpDiv",
            operand_widths: vec![],
        },
        Opcode::OpTrue => Definition {
            name: "OpTrue",
            operand_widths: vec![],
        },
        Opcode::OpFalse => Definition {
            name: "OpFalse",
            operand_widths: vec![],
        },
        Opcode::OpEqual => Definition {
            name: "OpEqual",
            operand_widths: vec![],
        },
        Opcode::OpNotEqual => Definition {
            name: "OpNotEqual",
            operand_widths: vec![],
        },
        Opcode::OpGreaterThan => Definition {
            name: "OpGreaterThan",
            operand_widths: vec![],
        },
        Opcode::OpMinus => Definition {
            name: "OpMinus",
            operand_widths: vec![],
        },
        Opcode::OpBang => Definition {
            name: "OpBang",
            operand_widths: vec![],
        },
        Opcode::OpJumpNotTruthy => Definition {
            name: "OpJumpNotTruthy",
            operand_widths: vec![2],
        },
        Opcode::OpJump => Definition {
            name: "OpJump",
            operand_widths: vec![2],
        },
        Opcode::OpNull => Definition {
            name: "OpNull",
            operand_widths: vec![],
        },
        Opcode::OpGetGlobal => Definition {
            name: "OpGetGlobal",
            operand_widths: vec![2],
        },
        Opcode::OpSetGlobal => Definition {
            name: "OpSetGlobal",
            operand_widths: vec![2],
        },
        Opcode::OpArray => Definition {
            name: "OpArray",
            operand_widths: vec![2],
        },
        Opcode::OpHash => Definition {
            name: "OpHash",
            operand_widths: vec![2],
        },
        Opcode::OpIndex => Definition {
            name: "OpIndex",
            operand_widths: vec![],
        },
        Opcode::OpCall => Definition {
            name: "OpCall",
            operand_widths: vec![1],
        },
        Opcode::OpReturnValue => Definition {
            name: "OpReturnValue",
            operand_widths: vec![],
        },
        Opcode::OpReturn => Definition {
            name: "OpReturn",
            operand_widths: vec![],
        },
        Opcode::OpGetLocal => Definition {
            name: "OpGetLocal",
            operand_widths: vec![1],
        },
        Opcode::OpSetLocal => Definition {
            name: "OpSetLocal",
            operand_widths: vec![1],
        },
        Opcode::OpGetBuiltin => Definition {
            name: "OpGetBuiltin",
            operand_widths: vec![1],
        },
        Opcode::OpClosure => Definition {
            name: "OpClosure",
            operand_widths: vec![2, 1],
        },
        Opcode::OpGetFree => Definition {
            name: "OpGetFree",
            operand_widths: vec![1],
        },
        Opcode::OpCurrentClosure => Definition {
            name: "OpCurrentClosure",
            operand_widths: vec![],
        },
    }
}
fn lookup(op: u8) -> Result<Definition, CodeError> {
    Ok(definitions(op.try_into()?))
}

impl Display for Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", definitions(*self).name)
    }
}

pub fn make(opcode: Opcode, operands: Vec<u16>) -> Instructions {
    let def = definitions(opcode);
    let instruction_len = def.operand_widths.iter().fold(1, |acc, x| acc + x);
    let mut instruction = vec![0; instruction_len as usize];
    instruction[0] = opcode.into();
    let mut offset = 1;
    for (i, o) in operands.iter().enumerate() {
        let width = def.operand_widths[i];
        match width {
            1 => instruction[offset] = (*o) as u8,
            2 => put_u16(&mut instruction[offset..], *o),
            _ => todo!(),
        }
        offset += width as usize
    }
    instruction
}
pub fn disassemble(ins: Instructions) -> Result<String, CodeError> {
    let mut out = String::new();
    let mut i = 0;
    while i < ins.len() {
        let def = lookup(ins[i])?;
        let (operands, read) = read_operands(&def, &ins[i + 1..])?;
        out += format!("{:04} {}\n", i, fmt_instruction(&def, operands)?).as_str();
        i += (1 + read) as usize;
    }
    Ok(out)
}
fn fmt_instruction(def: &Definition, operands: Vec<u16>) -> Result<String, CodeError> {
    let operand_count = def.operand_widths.len();
    if operands.len() != operand_count {
        return Err(CodeError::OperandLenDoesNotMatchDefined(
            operands.len(),
            operand_count,
        ));
    }
    Ok(match operand_count {
        0 => format!("{}", def.name),
        1 => format!("{} {}", def.name, operands[0]),
        2 => format!("{} {} {}", def.name, operands[0], operands[1]),
        _ => return Err(CodeError::UnhandledOperandCountFor(def.name)),
    })
}

fn read_operands(def: &Definition, ins: &[u8]) -> Result<(Vec<u16>, u16), CodeError> {
    let mut operands = vec![0_u16; def.operand_widths.len()];
    let mut offset = 0;
    for (i, width) in def.operand_widths.iter().enumerate() {
        match width {
            1 => operands[i] = ins[offset] as u16,
            2 => operands[i] = read_u16(&ins[offset..])?,
            _ => todo!(),
        }
        offset += (*width) as usize;
    }
    Ok((operands, offset as u16))
}

pub fn put_u16(target: &mut [u8], val: u16) {
    let target = &mut target[..2];
    target.copy_from_slice(&val.to_be_bytes());
}
pub fn read_u16(src: &[u8]) -> Result<u16, CodeError> {
    let src = &src[..2];
    Ok(u16::from_be_bytes(src.try_into()?))
}

#[derive(Debug)]
pub enum CodeError {
    OpcodeUndefined(u8),
    TryFromSliceError(std::array::TryFromSliceError),
    OperandLenDoesNotMatchDefined(usize, usize),
    UnhandledOperandCountFor(&'static str),
}
impl Display for CodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpcodeUndefined(op) => write!(f, "opcode {} undefined", op),
            Self::TryFromSliceError(e) => write!(f, "{}", e),
            Self::OperandLenDoesNotMatchDefined(actual, defined) => write!(
                f,
                "operand len {} does not match defined {}",
                actual, defined
            ),
            Self::UnhandledOperandCountFor(name) => {
                write!(f, "unhandled operand count for {}", name)
            }
        }
    }
}
impl From<std::array::TryFromSliceError> for CodeError {
    fn from(e: std::array::TryFromSliceError) -> Self {
        Self::TryFromSliceError(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make() {
        let tests = [
            (
                Opcode::OpConstant,
                vec![65534_u16],
                vec![Opcode::OpConstant.into(), 255_u8, 254_u8],
            ),
            (Opcode::OpAdd, vec![], vec![Opcode::OpAdd.into()]),
            (
                Opcode::OpGetLocal,
                vec![255],
                vec![Opcode::OpGetLocal.into(), 255],
            ),
            (
                Opcode::OpClosure,
                vec![65534, 255],
                vec![Opcode::OpClosure.into(), 255, 254, 255],
            ),
        ];
        for tt in tests {
            let instruction = make(tt.0, tt.1);
            assert_eq!(instruction, tt.2)
        }
    }
    #[test]
    fn test_instruction_string() {
        let instructions = [
            make(Opcode::OpAdd, vec![]),
            make(Opcode::OpGetLocal, vec![1]),
            make(Opcode::OpConstant, vec![2]),
            make(Opcode::OpConstant, vec![65535]),
            make(Opcode::OpClosure, vec![65535, 255]),
        ];
        let expected = r#"0000 OpAdd
0001 OpGetLocal 1
0003 OpConstant 2
0006 OpConstant 65535
0009 OpClosure 65535 255
"#;
        let concated = instructions.into_iter().flatten().collect::<Instructions>();
        assert_eq!(disassemble(concated).unwrap(), expected);
    }
    #[test]
    fn test_read_operands() {
        let tests = [
            (Opcode::OpConstant, vec![65535_u16], 2_u16),
            (Opcode::OpGetLocal, vec![255_u16], 1_u16),
            (Opcode::OpClosure, vec![65535_u16, 255_u16], 3_u16),
        ];
        for tt in tests {
            let instruction = make(tt.0, tt.1.clone());
            let def = lookup(tt.0.into()).unwrap();
            let (operands_read, n) = read_operands(&def, &instruction[1..]).unwrap();
            assert_eq!(n, tt.2);
            assert_eq!(operands_read, tt.1);
        }
    }
}
