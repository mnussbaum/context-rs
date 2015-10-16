#![feature(asm, repr_simd)]

#[macro_use]
extern crate log;
extern crate libc;
extern crate memmap;

pub use context::Context;
pub use error::ContextError;
pub use stack::Stack;

use std::result;
pub type Result<T> = result::Result<T, ContextError>;

pub mod context;
pub mod error;
pub mod stack;
mod sys;
#[cfg(target_arch = "x86_64")]
mod simd;
