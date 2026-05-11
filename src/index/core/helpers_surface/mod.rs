//! Helper functions for answer surface scoring, projection, and named entity extraction.

pub mod scoring_classify;
pub mod projection;
pub mod named_extraction;

pub use self::scoring_classify::*;
pub use self::projection::*;
pub use self::named_extraction::*;
