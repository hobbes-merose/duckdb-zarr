# Ralph Loop Prompt for duckdb-zarr

## Task Prompt

You are working on duckdb-zarr, a DuckDB extension for querying Zarr arrays with SQL. This is part of the xarray-sql vision for making n-dimensional scientific data queryable like a database.

### Core Architecture Insights (from DESIGN.md)
- **Pivot algorithm**: Transform multidimensional arrays into tabular form using strided memory access for efficient out-of-cache processing
- **Native C++ implementation**: No external crates (like zarr-datafusion) - build minimal, focused implementation using nlohmann/json, c-blosc2, Arrow C++
- **Table functions**: Use DuckDB's table function API for `read_zarr(path)` and `read_zarr_metadata(path)`

### Your Mission
Use crosslink to track issues and work through them systematically. Check the crosslink database for high-priority items.

**Crosslink commands:**
- `crosslink issues` - list all issues with status and priority
- `crosslink issue <id>` - get details of a specific issue
- `crosslink --help` - see all available commands

### Version Control Workflow
- Use **git worktrees** to manage stacked PRs - create a new worktree for each feature branch so you can work on multiple PRs simultaneously
- Rebase as necessary to keep history clean
- Commit early and often, create a PR when ready for alxmrs's review (do not merge directly to main)
- Feature branches should be named descriptively (e.g., `feature/zarr-metadata-parser`, `fix/ci-format`)

### Code Review Model
- All code changes require alxmrs's review, but his review does NOT block development
- Push feature branches freely and address review feedback in follow-up PRs or commits
- Use PRs to track review status but don't wait for approval before continuing
- Use stacked PRs (via git worktrees) to continue developing even while waiting for review

### Stop Conditions
Continue working until you hit:
- A **blocker** that requires human input (ambiguous requirements, design decisions, external dependencies)
- An **explicit instruction to pause** from alxmrs
- When there are no more issues to work on

**Blocker Convention:**
When you encounter a blocker:
1. Create a new issue with title prefixed with "BLOCKED: <brief reason>" 
2. Add details about what's blocked and what input you need
3. Set the issue priority to high
4. Continue working on other high-priority issues if available

Do NOT stop after completing each PR or issue - keep going until you hit a stop condition.

---

## Completion Promise

```
DONE when:
- alxmrs explicitly tells you to stop
- You encounter a blocker that requires human input to proceed
- There are no more high-priority crosslink issues remaining

NOT DONE when:
- You've completed a single issue or PR
- You've hit a minor obstacle you can work around
- You're waiting for review feedback (continue with other work)
```

---

## Initial Context

Start by:
1. Reading DESIGN.md to understand the project architecture
2. Checking crosslink issues to find the highest priority item
3. Begin working on that issue

---

<promise>COMPLETE</promise>
