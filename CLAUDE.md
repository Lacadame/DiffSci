# CLAUDE.md - Fundamental Instructions

## CRITICAL RESTRICTIONS - MUST NEVER BE VIOLATED

### 1. File System Access Restrictions

**ALLOWED directories for writing/modifying:**
- `diffsci/` - Source code directory
- `tests/` - Test directory

**ABSOLUTELY FORBIDDEN - NO EXCEPTIONS:**
- `saveddata/` - NEVER read, modify, or write
- `savedmodels/` - NEVER read, modify, or write

Any other directories require **EXPLICIT user permission** before any modification.

### 2. Git Command Prohibition

**NEVER run any git commands.** This includes but is not limited to:
- `git commit`
- `git push`
- `git pull`
- `git add`
- `git checkout`
- `git branch`
- Any other git operation

The user manages version control manually.
