#!/usr/bin/env python3
"""Automatically split temporal.rs into manageable submodules."""

import re
from pathlib import Path

# Read the backup of the original temporal.rs
original_file = Path("src/answer_plane/temporal_backup.rs")
output_dir = Path("src/answer_plane/temporal")

if not original_file.exists():
    # If no backup, read from current location
    original_file = Path("src/answer_plane/temporal.rs")

with open(original_file, 'r') as f:
    content = f.read()

# Extract the imports section (lines 1-51)
lines = content.split('\n')
imports = '\n'.join(lines[0:51])

# Define sections to extract
sections = {
    'selectors.rs': (53, 849, 'All select_* answer functions'),
    'kg_temporal.rs': (850, 1046, 'KG-backed temporal helpers'),
    'gap_parser.rs': (1047, 1213, 'Gap/elapsed query parsing'),
    'candidates.rs': (1214, 2507, 'Candidate collection and ranking'),
    'duration_render.rs': (2508, 3416, 'Duration arithmetic and rendering'),
}

# Create submodule files
for filename, (start, end, doc) in sections.items():
    module_imports = f"""//! {doc}

use super::*;
"""

    # Extract lines (converting to 0-indexed)
    body = '\n'.join(lines[start-1:end])

    # Remove the top-level imports from body if they exist
    if body.startswith('use super:'):
        # Find the end of imports block
        match = re.search(r'};?\s*\n', body)
        if match:
            body = body[match.end():]

    content = module_imports + body + '\n'

    output_file = output_dir / filename
    with open(output_file, 'w') as f:
        f.write(content)

    print(f"Created {filename} ({len(content)} bytes, lines {start}-{end})")

print("\nNow create temporal/mod.rs:")
mod_content = """//! Temporal reasoning: query processing, gap/duration answering, and calendar-grounded ranking.

mod selectors;
mod kg_temporal;
mod gap_parser;
mod candidates;
mod duration_render;

// Re-export all public items from submodules
pub(super) use self::selectors::*;
pub(super) use self::kg_temporal::*;
pub(super) use self::gap_parser::*;
pub(super) use self::candidates::*;
pub(super) use self::duration_render::*;
"""

mod_file = output_dir / "mod.rs"
with open(mod_file, 'w') as f:
    f.write(mod_content)

print(f"Created temporal/mod.rs")
print("\nRefactoring complete. Now run:")
print("  cargo check")
print("  cargo clippy")
