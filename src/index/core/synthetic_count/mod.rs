//! Synthetic answer methods for counting queries (count, financial, education, collections).

pub mod collections_activity;
pub mod education_career;
pub mod financial_medical;

pub use self::collections_activity::*;
pub use self::education_career::*;
pub use self::financial_medical::*;
