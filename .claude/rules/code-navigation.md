# Code Navigation

CodeGraph is indexed for this repo. Use it instead of naive `grep`/`find`/`cat` for anything that
is really a "where is X" or "what calls X" question — those tools burn tokens rereading whole
files and re-deriving structure that CodeGraph already has indexed.

## Default to CodeGraph / serena, not raw search

- **"Where is X defined", "what calls X", "how is X structured"** → `codegraph explore <query>`,
  `codegraph query <symbol>`, `codegraph node <name>`, `codegraph callers`/`callees`/`impact`. These
  return source + call graph in one shot instead of a grep hit list you then have to open and read.
- **Renaming or moving a symbol repo-wide** → serena's LSP tools (`find_symbol`,
  `find_referencing_symbols`, `rename_symbol`), not grep-and-replace. Grep-and-replace does not
  know about scoping, shadowing, or string/comment false positives; serena does.
- **Reserve Bash `grep`/`find`/`cat`** for narrow, already-scoped lookups where you already know the
  file or exact string (e.g., "does this specific config file contain this exact key") — not for
  open-ended discovery. If a query would require reading multiple files to answer, it belongs to
  CodeGraph or serena, not a grep fan-out.
- Prefer a single scoped query over broad ones: `codegraph explore "src/fluxmem entrypoint"` beats
  `codegraph explore "entrypoint"` — an unscoped query returns noise from anywhere in the index,
  not just the package you meant.

## Keep the index current

- Run `codegraph sync` after any multi-file change: a rename, a deletion, a subsystem removal, a
  large refactor. An unsynced index gives confidently wrong answers rather than an error, which is
  worse than no index.
- Run a full `codegraph index` (not just `sync`) after structural surgery — deleting or adding a
  top-level package, moving directories — rather than trusting incremental sync to reconcile it.
- If `codegraph explore` returns results that don't match what you expect (e.g. citing a file you
  know was deleted), sync or reindex before trusting it further; don't silently fall back to grep
  without noting the index was stale.

## Why this is a rule, not just a routing-table row

The routing table (`CLAUDE.md`) only surfaces when a task matches one of its rows. This is a
standing default: any exploration task should reach for CodeGraph/serena first, not just the ones
explicitly listed as "navigating unfamiliar code."
