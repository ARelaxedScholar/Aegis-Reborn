// src/vm/engine.rs

//! The core execution engine for Aegis strategies.
//! This module defines the Virtual Machine (VM) that runs the bytecode.

use crate::vm::op::{IndicatorType, Op, PriceType}; // Correctly import dependencies

const STACK_CAPACITY: usize = 256; // Max depth of the stack
const MEMORY_SIZE: usize = 16; // Number of f64 slots for strategy state

/// Defines errors that can occur during VM execution.
#[derive(Debug, PartialEq)]
pub enum VmError {
    StackOverflow,
    StackUnderflow,
    MemoryOutOfBounds(u8),
    InvalidProgram,
}

/// The Aegis Virtual Machine.
/// It is a simple, stack-based machine designed for performance and safety.
pub struct VirtualMachine {
    stack: Vec<f64>,
    memory: [f64; MEMORY_SIZE],
}

impl VirtualMachine {
    /// Creates a new, clean instance of the VM.
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(STACK_CAPACITY),
            memory: [0.0; MEMORY_SIZE],
        }
    }

    /// Executes a bytecode program.
    pub fn execute(&mut self, program: &[Op]) -> Result<f64, VmError> {
        self.stack.clear();
        let mut pc = 0;

        while pc < program.len() {
            let op = &program[pc];
            pc += 1;

            match op {
                Op::PushConstant(val) => self.push(*val)?,

                Op::Store(idx) => {
                    let val = self.pop()?;
                    if let Some(mem_slot) = self.memory.get_mut(*idx as usize) {
                        *mem_slot = val;
                    } else {
                        return Err(VmError::MemoryOutOfBounds(*idx));
                    }
                }
                Op::Load(idx) => {
                    if let Some(val) = self.memory.get(*idx as usize) {
                        self.push(*val)?;
                    } else {
                        return Err(VmError::MemoryOutOfBounds(*idx));
                    }
                }

                Op::Add => self.apply_binary_op(|a, b| a + b)?,
                Op::Subtract => self.apply_binary_op(|a, b| a - b)?,
                Op::Multiply => self.apply_binary_op(|a, b| a * b)?,
                Op::Divide => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    if b.abs() < 1e-9 {
                        self.push(0.0)?;
                    } else {
                        self.push(a / b)?;
                    }
                }

                Op::GreaterThan => self.apply_binary_op(|a, b| (a > b) as i32 as f64)?,
                Op::LessThan => self.apply_binary_op(|a, b| (a < b) as i32 as f64)?,
                Op::GreaterThanOrEqual => self.apply_binary_op(|a, b| (a >= b) as i32 as f64)?,
                Op::LessThanOrEqual => self.apply_binary_op(|a, b| (a <= b) as i32 as f64)?,
                Op::Equal => self.apply_binary_op(|a, b| (a == b) as i32 as f64)?,
                Op::And => self.apply_binary_op(|a, b| ((a > 0.0) && (b > 0.0)) as i32 as f64)?,
                Op::Or => self.apply_binary_op(|a, b| ((a > 0.0) || (b > 0.0)) as i32 as f64)?,
                Op::Not => {
                    let val = self.pop()?;
                    self.push((val == 0.0) as i32 as f64)?;
                }

                Op::PushPrice(_) | Op::PushIndicator(_) => {
                    // Placeholder: The backtester is responsible for loading context.
                    self.push(0.0)?;
                }

                Op::JumpIfFalse(_) | Op::Jump(_) => {
                    unimplemented!(
                        "Control flow opcodes are not yet implemented in the 'Crawl' phase VM."
                    )
                }

                Op::Return => break,
            }
        }

        self.pop().or(Err(VmError::InvalidProgram))
    }

    #[inline]
    fn push(&mut self, val: f64) -> Result<(), VmError> {
        if self.stack.len() < STACK_CAPACITY {
            self.stack.push(val);
            Ok(())
        } else {
            Err(VmError::StackOverflow)
        }
    }

    #[inline]
    fn pop(&mut self) -> Result<f64, VmError> {
        self.stack.pop().ok_or(VmError::StackUnderflow)
    }

    #[inline]
    fn apply_binary_op<F>(&mut self, op: F) -> Result<(), VmError>
    where
        F: Fn(f64, f64) -> f64,
    {
        let b = self.pop()?;
        let a = self.pop()?;
        self.push(op(a, b))
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::op::Op::*;

    #[test]
    fn test_basic_arithmetic() {
        let mut vm = VirtualMachine::new();
        let program = vec![PushConstant(5.0), PushConstant(10.0), Add];
        assert_eq!(vm.execute(&program).unwrap(), 15.0);
    }

    #[test]
    fn test_rpn_expression() {
        let mut vm = VirtualMachine::new();
        let program = vec![
            PushConstant(5.0),
            PushConstant(10.0),
            Add,
            PushConstant(2.0),
            Multiply,
        ];
        assert_eq!(vm.execute(&program).unwrap(), 30.0);
    }

    #[test]
    fn test_division_by_zero_safety() {
        let mut vm = VirtualMachine::new();
        let program = vec![PushConstant(10.0), PushConstant(0.0), Divide];
        assert_eq!(vm.execute(&program).unwrap(), 0.0);
    }

    #[test]
    fn test_comparison_operators() {
        let mut vm = VirtualMachine::new();
        let program_gt_false = vec![PushConstant(5.0), PushConstant(10.0), GreaterThan];
        assert_eq!(vm.execute(&program_gt_false).unwrap(), 0.0);
        let program_gt_true = vec![PushConstant(10.0), PushConstant(5.0), GreaterThan];
        assert_eq!(vm.execute(&program_gt_true).unwrap(), 1.0);
        let program_eq_true = vec![PushConstant(10.0), PushConstant(10.0), Equal];
        assert_eq!(vm.execute(&program_eq_true).unwrap(), 1.0);
    }

    #[test]
    fn test_logical_operators() {
        let mut vm = VirtualMachine::new();
        let program = vec![PushConstant(1.0), PushConstant(0.0), And];
        assert_eq!(vm.execute(&program).unwrap(), 0.0);
        let program_not = vec![PushConstant(1.0), Not];
        assert_eq!(vm.execute(&program_not).unwrap(), 0.0);
    }

    #[test]
    fn test_state_management_store_and_load() {
        let mut vm = VirtualMachine::new();
        let program = vec![
            PushConstant(99.0),
            Store(5),
            Load(5),
            PushConstant(1.0),
            Add,
        ];
        assert_eq!(vm.execute(&program).unwrap(), 100.0);
        assert_eq!(vm.memory[5], 99.0);
    }

    #[test]
    fn test_stack_underflow() {
        let mut vm = VirtualMachine::new();
        let program = vec![Add];
        assert_eq!(vm.execute(&program), Err(VmError::StackUnderflow));
    }

    #[test]
    fn test_memory_out_of_bounds() {
        let mut vm = VirtualMachine::new();
        let program = vec![PushConstant(1.0), Store(MEMORY_SIZE as u8)];
        assert_eq!(
            vm.execute(&program),
            Err(VmError::MemoryOutOfBounds(MEMORY_SIZE as u8))
        );
    }

    #[test]
    fn test_stack_overflow() {
        let mut vm = VirtualMachine::new();
        let mut program = vec![];
        for i in 0..257 {
            // Exceeds STACK_CAPACITY
            program.push(PushConstant(i as f64));
        }
        assert_eq!(vm.execute(&program), Err(VmError::StackOverflow));
    }

    #[test]
    fn test_invalid_program_empty() {
        let mut vm = VirtualMachine::new();
        let program = vec![];
        assert_eq!(vm.execute(&program), Err(VmError::InvalidProgram));
    }

    #[test]
    fn test_stack_isolation_between_executions() {
        let mut vm = VirtualMachine::new();
        vm.execute(&vec![PushConstant(99.0)]).unwrap();
        let result = vm.execute(&vec![PushConstant(1.0)]).unwrap();
        assert_eq!(result, 1.0); // Should not see the 99.0
    }
}
