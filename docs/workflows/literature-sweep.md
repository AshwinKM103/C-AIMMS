# Workflow: Literature sweep

For related-work sections, baseline hunting, and "has someone already done this?".

## 1. Frame one question
Not a topic. "Which memory systems report LongMemEval R@5, and under what retrieval budget?"
beats "agent memory".

## 2. Fan out
| Tool | Use for |
|---|---|
| `/deep-dive` | DAG-based decomposition, parallel sub-questions, gap-driven iteration, sourced report. No external API |
| AutoSearch | arXiv, PubMed, OpenAlex, DBLP — deduped, ranked, source URLs, keyless |
| `bgpt` MCP | paper search with full-text experimental data |
| `academic-researcher` agent | literature review structure, citation analysis, methodology critique |
| `fetch` MCP | pull a specific paper or repo README as markdown |

## 3. Capture as you go
Findings go to the `memory` MCP knowledge graph during the sweep, then get written down.
An uncaptured sweep gets repeated next month by someone else.

## 4. Record baselines separately
Anything that publishes numbers on **LongMemEval, LoCoMo, or EgoLife** goes into
`docs/related-work-agent-memory.md` with the number, the source, and whether *we* reproduced it.
Two are already tracked there (Cortex, axme-code) — both self-reported, neither reproduced by us.

## 5. Close the question
A sweep ends with a written finding and its sources, not with an open browser. State what the
evidence supports and what it does not. If it changes a design choice, that is an `/adr`.

## Related
`deep-dive` skill · `research-analyst`, `search-specialist`, `technology-scout` agents
