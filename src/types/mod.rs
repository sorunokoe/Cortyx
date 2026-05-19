//! Core domain types for Cortyx — the foundation of Type-Driven Design.
//!
//! Importing `cortyx::types::*` gives access to all typed primitives.
//! These types eliminate entire classes of bugs by making invalid states
//! unrepresentable at compile time:
//!
//! | Type | Replaces | Invalid state eliminated |
//! |------|----------|------------------------|
//! | `NeuronUuid` | `String` | Non-hex or wrong-length UUIDs |
//! | `NeuronId` | `PathBuf` | Absolute paths used as neuron identity (must be relative) |
//! | `NeuronRelPath` | `PathBuf` | Absolute or `..`-escaped paths in synapse targets |
//! | `EditId` | `Option<String>` | Empty edit IDs in provenance chains |
//! | `AuthorId` | `String` | Empty author identifiers |
//! | `TokenCount` | `usize` | Confusing counts with other usizes |
//! | `TokenBudget` | `usize` | Unreasonable budgets (>100k) |
//! | `QueryText` | `String` | Empty or unvalidated queries |
//! | `TermFrequency` | `f32` | Negative term frequencies |
//! | `SynapseWeight` | `f32` | Out-of-range weights; wrong score type |
//! | `ConfidenceScore` | `f32` | Scores outside `[0, 1]` |
//! | `QualityScore` | `f32` | Scores outside `[0, 1]` |
//! | `BM25Score` | `f32` | Negative raw scores; confusion with normalised scores |
//! | `StalenessMultiplier` | `f32` | Zero multiplier that permanently suppresses a neuron |
//! | `IsoDate` | `String` | Invalid date formats; unsafe string comparison |
//! | `ModuleScope` | `Option<String>` | `"@alice"` vs `"alice"` normalisation |

pub mod date;
pub mod evidence;
pub mod ids;
pub mod primitives;
pub mod scope;
pub mod scores;
pub mod state_machines;

pub use date::IsoDate;
pub use evidence::{EvidenceFact, EvidenceFamily};
pub use ids::{AuthorId, EditId, NeuronId, NeuronRelPath, NeuronUuid};
pub use primitives::{QueryText, TermFrequency, TokenBudget, TokenCount};
pub use scope::{ModuleScope, PersonSlug};
pub use scores::{BM25Score, ConfidenceScore, QualityScore, StalenessMultiplier, SynapseWeight};
pub use state_machines::*;
