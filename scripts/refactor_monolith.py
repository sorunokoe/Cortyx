#!/usr/bin/env python3
"""
Automated refactoring script for Cortyx monolith extraction.

This script systematically extracts logical modules from the 33,870-line
src/index/core/mod.rs monolith following the established pattern.

Usage:
    python3 scripts/refactor_monolith.py --dry-run  # Preview changes
    python3 scripts/refactor_monolith.py            # Execute refactoring
"""

import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ModuleExtraction:
    """Represents a module to be extracted."""
    name: str
    target_dir: str
    start_marker: str
    end_marker: str
    description: str

# Define extraction plan
EXTRACTIONS = [
    ModuleExtraction(
        name="config",
        target_dir="src/index/core/config",
        start_marker="mod config;",
        end_marker="use config::*;",
        description="BM25 configuration constants"
    ),
    ModuleExtraction(
        name="schema_migrations",
        target_dir="src/index/core/migrations",
        start_marker="// ─── Schema migrations",
        end_marker="// ─── NeuronIndex",
        description="Index schema version migrations"
    ),
    # Add more extractions here following the pattern
]

def find_section(content: str, start: str, end: str) -> Tuple[int, int]:
    """Find start and end positions of a section."""
    start_pos = content.find(start)
    if start_pos == -1:
        return (-1, -1)
    
    end_pos = content.find(end, start_pos)
    if end_pos == -1:
        return (start_pos, len(content))
    
    return (start_pos, end_pos)

def extract_module(source_file: Path, extraction: ModuleExtraction, dry_run: bool = True):
    """Extract a module from the monolith."""
    with open(source_file, 'r') as f:
        content = f.read()
    
    start_pos, end_pos = find_section(content, extraction.start_marker, extraction.end_marker)
    
    if start_pos == -1:
        print(f"⚠️  Could not find section: {extraction.name}")
        return False
    
    section_content = content[start_pos:end_pos]
    
    # Create target directory
    target_dir = Path(extraction.target_dir)
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Write extracted content
        with open(target_dir / "mod.rs", 'w') as f:
            f.write(f"//! {extraction.description}\n\n")
            f.write(section_content)
        
        print(f"✅ Extracted: {extraction.name} ({len(section_content)} chars)")
    else:
        print(f"🔍 Would extract: {extraction.name} ({len(section_content)} chars)")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Refactor Cortyx monolith")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    args = parser.parse_args()
    
    source_file = Path("src/index/core/mod.rs")
    
    if not source_file.exists():
        print(f"❌ Source file not found: {source_file}")
        return 1
    
    print(f"{'🔍 DRY RUN MODE' if args.dry_run else '🚀 EXECUTING REFACTORING'}")
    print(f"Source: {source_file} ({source_file.stat().st_size} bytes)")
    print()
    
    for extraction in EXTRACTIONS:
        extract_module(source_file, extraction, args.dry_run)
    
    print()
    print("Next steps:")
    print("1. Review extracted modules")
    print("2. Update src/index/core/mod.rs to re-export")
    print("3. Run: cargo test --lib")
    print("4. Commit: git add . && git commit -m 'refactor: extract modules from monolith'")
    
    return 0

if __name__ == "__main__":
    exit(main())
