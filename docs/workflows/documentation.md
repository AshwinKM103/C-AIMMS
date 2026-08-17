# Documentation Workflow

## What to document

- Public API functions: parameters, return types, error conditions, and at least one example.
- Architecture decisions: why an approach was chosen — write an ADR (`docs/adr/`), not a comment.
- Setup and installation: prerequisites, steps, common failure modes.
- Configuration: every option, its default, and its environment variable.
- Non-obvious behavior: edge cases, gotchas, workarounds.

## What not to document

- Obvious code (getters, setters, thin wrappers).
- Implementation details that change frequently — they will rot faster than they help.
- Anything the type system already expresses.
- Temporary workarounds that don't have a tracking issue.

## Docstring bar

Every module gets a one-line module docstring stating its purpose. Every public function or class
gets a docstring with a one-line summary, an `Args` section, a `Returns` section, and a `Raises`
section (only if it raises). Non-obvious logic gets a one-line inline comment explaining _why_, not
_what_. `HyperMem/tests/` is the acceptance bar for this — e.g.
`HyperMem/tests/test_checkpoint_manager.py` shows the expected shape: module docstring naming the
key test cases up front, per-function docstring with a summary, `Args`, and `Raises`.

## Formats

- Inline comments: explain why, not what. One line, placed above the code.
- Docstrings: required for all public APIs. Parameters, returns, exceptions raised, one usage
  example.
- README: installation, quick start, usage (most common case first), configuration reference,
  contributing guidelines, license — in that order.
- CLAUDE.md: project context, conventions, build commands, key architecture decisions. Kept under
  ~150 lines; it loads in full every session, so depth belongs elsewhere.
- ADRs: date, status, context, decision, consequences. Nygard format, sequential, `docs/adr/`.
  Supersede an ADR rather than editing it.

## Style

- Concrete examples, not abstract descriptions.
- Write for the person maintaining this code in six months, not for the person who just wrote it.
- Consistent terminology — don't introduce a synonym for a term the codebase already uses.
- Short sentences, one idea per paragraph.
- Code blocks carry a language tag.
- Tables for option lists, comparisons, and configuration references.

## Keep docs current

- Update documentation in the same PR that changes the code it describes — a docs update that
  lands later is a docs update that doesn't land.
- Stale docs are worse than no docs; review them as part of code review, not as an afterthought.
- Keep CLAUDE.md current with the project's actual layout and conventions, not its aspirational
  ones.
