use std::error::Error;
use std::fmt;

#[derive(Debug, PartialEq)]
pub enum ContextError {
    EmptyStack,
    InvalidSetArgIndex,
}

impl ContextError {
    pub fn description(&self) -> &str {
        match *self {
            ContextError::EmptyStack => "Attempted to use context without a stack",
            ContextError::InvalidSetArgIndex => {
                "Attempted to set context argument to an index greater then 1"
            },
        }
    }
}

impl Error for ContextError {
    fn description(&self) -> &str {
        self.description()
    }

    fn cause(&self) -> Option<&Error> {
        match *self {
            ContextError::EmptyStack => None,
            ContextError::InvalidSetArgIndex => None,
        }
    }
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.description())
    }
}
