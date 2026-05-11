//! Synthetic answer methods for temporal queries (temporal choice, intervals, event counts).

pub mod count_events;
pub mod time_choice;
pub mod title_events;

pub use self::count_events::*;
pub use self::time_choice::*;
pub use self::title_events::*;
