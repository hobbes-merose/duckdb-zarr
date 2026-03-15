# Project Instructions

## Pre-commit Build Check

Before committing any code changes, Claude MUST:

1. Run `make` to build the extension
2. Fix any build errors that arise
3. Only commit after the build passes

This ensures that CI doesn't fail due to build issues.
