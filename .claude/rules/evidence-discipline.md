# Evidence Discipline

Research code fails differently from product code: it does not crash, it reports a number that is
wrong. These rules target that failure mode. Sources are named so the reasoning can be re-checked.

## Before writing code that depends on data

**Verify the schema against real records, not assumptions.** Load an actual sample and look at it
before writing the code that consumes it. Field present? Nullable? Nested? What does an empty case
look like?

*Why:* an A/B test of a hook that forced this step found the ungated agent "assumed a schema" and
invented keys that tests did not catch (GateGuard, zunoworks). Their evidence is weak on its own —
N=3, self-scored, authors acknowledge bias — but the failure it describes is exactly how a silent
metric bug is born, and checking costs one `head`.

## While investigating

- **Two hypotheses minimum** before acting on a diagnosis. One hypothesis is a guess wearing a
  conclusion's clothes.
- **Number your observations and cite the source of each** — file:line, a command's output, a paper.
  Three lines maximum per observation.
- Distinguish *evidence* from *pattern-match*. If you recognized the shape of the problem but did
  not confirm it, say so explicitly. (paul-graham-skills: `mark-what-you-don't-know`,
  `distrust-the-surface` — a clean answer is a hypothesis, not a result.)
- **A failed method is the problem, not the goal.** When something is not working, change the
  approach rather than quietly narrowing what counts as success.
  (paul-graham-skills: `change-method-not-goal`.)

## Two workflows, not one

Development and research are different loops. Naming which one you are in prevents applying the
wrong standard. (claude-agentic-coding-playbook, john-wilmes.)

| Loop | Shape |
|---|---|
| **Development** | Explore → Plan → Code → Verify → Commit |
| **Research** | Question → Collect → Synthesize → Close |

A research question closes with a written finding and its sources, not with merged code. Do not
let a research thread quietly become an implementation thread without a decision in between —
if it does, that decision is an ADR (`/adr`).

## Before declaring done

Run the checklist. Not "it looks right":

1. Did the tests actually run, and did they pass? Paste the output, do not summarize it.
2. If a number changed, is the change explained, or merely observed?
3. Is anything half-finished, TODO'd without context, or worked around silently?
   (`LightMem/CLAUDE.md`: Codex reviews everything — it will flag these.)
4. Was anything skipped? Say so plainly.

*Source:* the completion-checklist pattern from `obey` (Lexxes-Projects), ported as a rule rather
than as a Stop hook — this project keeps its hook layer minimal.

## Reporting

State what was verified and what was assumed. "Tests pass" without having run them, or a
confident summary of an unverified claim, is worse than saying nothing: it spends the reader's
trust on something you did not check.
