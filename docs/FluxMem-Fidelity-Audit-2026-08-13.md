# FluxMem Paper-to-Code Fidelity Audit — 2026-08-13

**Target:** `/home/durgesh/aditya/C-AIMMS/fluxmem/` vs. FluxMem (Lu et al., arXiv:2602.14038v1, 15 Feb 2026),
_"Choosing How to Remember: Adaptive Memory Structures for LLM Agents."_
**Secondary source:** COLM submission (`docs/cited-papers/COLM_Cognitive_AI_Memory_Architecture_Project.pdf`), §1.3.2–1.3.3, Eq. 5–7.
**Audit protocol:** `.claude/prompts/fluxmem-paper-code-fidelity-audit.md`
**Prior baseline:** `docs/FluxMem-Closure-Report.md` (2026-08-10; 146 tests, 99% coverage).

---

## 1. Executive Summary

All 16 in-scope FluxMem equations (Eq. 5–10, 17–26) and all three algorithms (Alg. 1–3) are
located and implemented in `fluxmem/`; **14 of 16 match the paper verbatim, 2 are documented
deviations**, and none are missing or stubbed. The test suite has grown since closure and is
green: **161 passed, 99% coverage** (`pytest tests/ --cov=fluxmem`, run 2026-08-13), up from the
closure report's 146. Equation-level fidelity is high and, unusually, the code's own docstrings
cite equation numbers that are **correct against the actual PDF** — a stronger signal than the
prior closure report captured.

**Risk level: Low**, with one Medium-severity defect.

Top three findings, in order of concern:

1. **[Medium — real defect, D-1] `ltsm.is_eligible` silently ignores MTEM's configured
   `reference_length` and `half_life`.** `fluxmem/ltsm.py:82` calls `utility_score(episode,
weights, now)` without forwarding the two scale parameters, so it always uses
   `mtem.DEFAULT_REFERENCE_LENGTH=50.0` / `DEFAULT_HALF_LIFE=100.0`. Any `MtemConfig` that
   overrides them makes the promotion gate score episodes on a **different `U` than MTEM itself
   uses**. Verified numerically: for one episode with `reference_length=5.0, half_life=1.0`,
   `MidTermEpisodicMemory.utility` returns **1.2998** while `ltsm`'s path returns **1.6204**.
   This is the classic research-code failure mode — no crash, a wrong number. Undocumented and
   untested; `MtemConfig` exposes both fields (`config.py:97–98`), so the config surface invites
   the bug.

2. **[Meta — High impact on this audit's inputs] The audit prompt's own reference equation table
   is substantially incorrect.** `.claude/prompts/fluxmem-paper-code-fidelity-audit.md` lines
   114–129 mislabel Eq. 6, 7, 8, 18, 19, 20, 21, 22, 23 and 24, invent an Eq. 22/24 that do not
   exist as described ("variance stabilization", "convergence criterion ΔL < ε"), omit Eq. 25
   entirely, and misattribute all three algorithms. **The code is right and the prompt is wrong.**
   An auditor who trusted the prompt's table would have raised 8–10 false divergences against
   correct code. Details in §5, D-0.

3. **[Low — documented] Eq. 6's promotion gate is deliberately non-verbatim.** The code
   substitutes COLM's composite `U(e)` for FluxMem's usage-only `u(m)`, and drops the optional
   third conjunct `c(m) ≥ τ_c`. Both are argued at length in `fluxmem/ltsm.py:1–55` and are
   consistent with ADR 0008 (COLM wins on conflicts). Not an oversight — but it does mean
   FluxMem's tuned `τ_u` values are not portable, which the docstring states explicitly.

---

## 2. Component Status Table

| Paper Section           | Component                        | Code Location            | Status         | Confidence | Notes                                                                                                                                                                                                       |
| ----------------------- | -------------------------------- | ------------------------ | -------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| §3.2, Eq. 5             | STIM / LRU tier                  | `stim.py:15–81`          | ⚠ Partial      | High       | Correct LRU by last-access; `touch()` exists but is never called on the write path, so it runs as FIFO today. Documented at `stim.py:18–25`. Capacity 4 matches paper (line 372: "STIM (capacity 4, LRU)"). |
| §3.2                    | MTEM tier + capacity             | `mtem.py:84–162`         | ✓              | High       | Capacity 2000 matches paper line 373 ("MTEM (up to 2000 episodic sessions"). Eviction by lowest `U`.                                                                                                        |
| §3.2                    | Utility score `U(s)=w₁c+w₂ℓ+w₃d` | `mtem.py:69–81`          | ✓              | High       | Structure verbatim. `c`, `ℓ`, `d` concretizations are modeling choices — neither paper defines them; flagged as such at `mtem.py:1–14`.                                                                     |
| §3.2, Eq. 6             | LTSM eligibility gate            | `ltsm.py:75–84`          | ⚠ Partial      | High       | 2 of 3 conjuncts; `u(m)`→composite `U`. Deliberate (D-2, D-3).                                                                                                                                              |
| §3.2                    | LTSM store                       | `ltsm.py:93–129`         | ✓              | High       | FAISS `IndexFlatIP` + L2 normalization ⇒ cosine. Paper does not specify the index; not a divergence.                                                                                                        |
| App. C, Table 5         | Conversation features            | `features.py:29–163`     | ⚠ Partial      | High       | 7 features, not FluxMem Table 5's 12. Matches **COLM §1.3.3 lines 122–127 exactly** (5 FluxMem-class + hyperedge density + visual salience). Reading-rule choice, not drift (D-4).                          |
| §3.4, Eq. 7–8           | Structure selector               | `selector.py:34–159`     | ✓              | High       | `Linear→ReLU→Linear`, softmax (Eq. 7), `argmax` (Eq. 8). Paper says "shallow MLP" — satisfied.                                                                                                              |
| §3.4, Eq. 9             | Reward combination               | `supervision.py:102–117` | ⚠ Partial      | High       | Linear form and λ=0.7/0.3 correct; `Judge`/`MemUtil` internals are ADR 0002 concretizations (D-5).                                                                                                          |
| §3.4, Eq. 10            | Selector loss                    | `selector.py:119`        | ✓              | High       | `F.cross_entropy`, literal.                                                                                                                                                                                 |
| §3.5 / App. B, Eq. 17   | Score normalization              | `fusion.py:24–35`        | ✓              | High       | Both branches of the piecewise definition present, including the `s_max=s_min → 0.5` case.                                                                                                                  |
| App. B, Eq. 18–19       | Two-component Beta mixture       | `fusion.py:48–119`       | ✓              | High       | `_N_COMPONENTS = 2`; Beta pdf via `scipy.stats.beta`.                                                                                                                                                       |
| App. B, Eq. 20–23       | EM fit                           | `fusion.py:121–160`      | ✓              | High       | Log-space E-step; π, μ, σ², κ moment matching, all with the clamps the paper itself calls for.                                                                                                              |
| App. B, Eq. 24–25       | High component + gate            | `fusion.py:115, 162–173` | ✓              | High       | `argmax α/(α+β)`; posterior in log-space.                                                                                                                                                                   |
| App. B, Eq. 26 / Alg. 1 | Threshold + min-keep             | `fusion.py:186–236`      | ✓              | High       | Threshold, TopK fallback by `x_i`, `NEW` branch, `argmax_{i∈I} x_i`.                                                                                                                                        |
| App. D, Alg. 2          | Offline labeling                 | `supervision.py:132–154` | ✓              | High       | Per-format reward loop + `argmax`, deterministic tie-break.                                                                                                                                                 |
| App. D, Alg. 3          | Selector training                | `selector.py:82–143`     | ✓              | High       | StandardScaler (train-only), minibatch, Adam, **early stopping present** with best-state restore.                                                                                                           |
| §3.1, Eq. 1             | Episodic unit                    | `interfaces.py:50–69`    | ✓              | High       | `EpisodicUnit.turns` is `e_j = {p_{j,1..m}}`.                                                                                                                                                               |
| §3.1, Eq. 2–4           | Retrieval / generation           | —                        | ○ Out-of-scope | High       | ITERRET/WORKMEM are declared non-goals (`ltsm.py:48–54`, `supervision.py:42–46`).                                                                                                                           |
| App. A, Eq. 11–16       | Eval metrics (F1/BLEU)           | —                        | ○ Out-of-scope | High       | Benchmark metrics, not architecture. `_token_f1` at `supervision.py:57` is the reward judge, not Eq. 13.                                                                                                    |
| COLM §1.2               | `EpisodeEncoder` seam            | `interfaces.py:90–114`   | ✓              | High       | Not a FluxMem equation. HetRep boundary (ADR 0005); correctly documented as encoding-only, distinct from `EpisodeProducer`.                                                                                 |
| COLM §1.3.2             | MemOCR producer                  | `memocr_episodes.py`     | ○ Out-of-scope | High       | Adapter, no equations. Excluded per audit scope (§"Exclude HyperMem/MemOCR integration").                                                                                                                   |

---

## 3. Equation Compliance Matrix

Equation numbers below are from the **actual PDF**, verified by `pdftotext -layout` extraction
(equation tags at flux.txt lines 229, 302, 306, 319, 849–948). Where the audit prompt's table
disagrees, the prompt is wrong — see D-0.

| Eq. | Paper (verified)                                      | Code File:Line                            | Formula Match                               | Edge Cases Guarded                                                                                                                                                                                                | Status                                                              |
| --- | ----------------------------------------------------- | ----------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 5   | `E_t = argmin_{P⊂M_t,                                 | P                                         | =                                           | M_t                                                                                                                                                                                                               | −C} Σ τ(p)`                                                         | `stim.py:54–61`                                      | ✓ (subset of size 1; equivalent under one-at-a-time push) | ✓ capacity≥0 validated; capacity-0 short-circuit | ⚠ (FIFO in practice) |
| 6   | `m ∈ M_LTSM iff u(m)≥τ_u ∧ r(m)≥τ_r (∧ c(m)≥τ_c)`     | `ltsm.py:81–84`                           | ⚠ `u`→`U`; 3rd conjunct dropped             | ✓ `max_age>0` raises; `max(0,dt)`                                                                                                                                                                                 | ⚠                                                                   |
| 7   | `f_θ(x_t) = Softmax(g_θ(x_t))`                        | `selector.py:153`                         | ✓                                           | ✓ `RuntimeError` if unfitted                                                                                                                                                                                      | ✓                                                                   |
| 8   | `ŝ_t = argmax_s f_θ(x_t)[s]`                          | `selector.py:156–159`                     | ✓                                           | ✓ `np.argmax` total on non-empty                                                                                                                                                                                  | ✓                                                                   |
| 9   | `r_t(s) = λ_q r^judge + λ_m r^mem`                    | `supervision.py:116`                      | ✓ (λ from `RewardConfig`, defaults 0.7/0.3) | ✓ `_token_f1` guards empty + zero-overlap; `TokenBudgetMemUtil` guards empty gold and `1+budget/ref` never 0                                                                                                      | ⚠ (Judge/MemUtil = ADR 0002)                                        |
| 10  | `L_sel(θ) = Σ ℓ(f_θ(x_t), s*_t)`                      | `selector.py:119, 127`                    | ✓                                           | ✓                                                                                                                                                                                                                 | ✓                                                                   |
| 17  | `x_i = ε+(1−2ε)(s_i−s_min)/(s_max−s_min)`, else `0.5` | `fusion.py:31–35`                         | ✓ both branches                             | ✓ **div-by-zero explicitly handled** (`s_max==s_min`)                                                                                                                                                             | ✓                                                                   |
| 18  | `p(x)=π₀Beta+π₁Beta`, `Σπ_k=1`                        | `fusion.py:106, 156`                      | ✓                                           | ✓ `Σπ=1` holds since responsibilities are normalized per-point                                                                                                                                                    | ✓                                                                   |
| 19  | `Beta(x;α,β)` density                                 | `fusion.py:142, 168` (`beta_dist.logpdf`) | ✓                                           | ✓ log-space avoids overflow at `x→0⁺/1⁻`                                                                                                                                                                          | ✓                                                                   |
| 20  | E-step, log-space `r_ik`                              | `fusion.py:136–147`                       | ✓                                           | ✓ `logsumexp`; **log(0) impossible** — `π_k=N_k/n` with `N_k` floored via `n_k_safe` only in μ/σ², but π is re-derived each M-step and a truly-zero π would only arise from an exactly-zero responsibility column | ✓                                                                   |
| 21  | `π_k ← N_k/n`                                         | `fusion.py:154–156`                       | ✓ (uses unfloored `n_k`, correct)           | ✓ `n = x.shape[0] ≥ 1` enforced at `fusion.py:95–96`                                                                                                                                                              | ✓                                                                   |
| 22  | `μ_k, σ²_k` responsibility-weighted                   | `fusion.py:157–158`                       | ✓                                           | ✓ `n_k_safe = max(n_k, 1e-6)` prevents 0/0                                                                                                                                                                        | ✓                                                                   |
| 23  | `κ_k ← μ(1−μ)/σ² − 1; α←μκ; β←(1−μ)κ`                 | `fusion.py:121–134`                       | ✓                                           | ✓ **three** floors: `mu_floor`, `variance_floor`, `kappa_floor`. Paper explicitly sanctions this ("we clamp σ² away from zero and clip κ, α, β")                                                                  | ✓                                                                   |
| 24  | `k* = argmax_k α_k/(α_k+β_k)`                         | `fusion.py:115`                           | ✓                                           | ✓ `α+β = κ ≥ kappa_floor > 0`                                                                                                                                                                                     | ✓                                                                   |
| 25  | `g(x) = π_{k*}Beta/Σ_k π_k Beta`                      | `fusion.py:162–173`                       | ✓                                           | ✓ log-space + `logsumexp`                                                                                                                                                                                         | ✓                                                                   |
| 26  | `I = {i : g(x_i) ≥ τ}`; if `                          | I                                         | <m_min`, `I←TopK({x_i},m_min)`              | `fusion.py:218–220`                                                                                                                                                                                               | ✓ (TopK by `x_i`, not by posterior — matches paper's parenthetical) | ✓ empty-`candidates` early return at `fusion.py:204` | ✓                                                         |

**Tally:** 16/16 located, 0 missing, 14 ✓, 2 ⚠. Equation-presence 100%; verbatim fidelity 87.5%.

---

## 4. Algorithm Deep-Dives

### Alg. 1 — Beta-Mixture-Gated Memory Fusion (`fusion.py:186`)

**Locate.** PDF line 916. 21 lines of pseudocode; the highest-complexity item in the paper.

**Survey → search.** `fusion.py` docstring (line 1) cites "FluxMem Eqs. 17-26 + Alg. 1" and calls
itself "highest logic risk in this package." `select_merge_target` is the entry point.

**Read → compare**, line by line against the pseudocode:

| Alg. 1 line | Pseudocode                            | Code                                                               | Match                        |
| ----------- | ------------------------------------- | ------------------------------------------------------------------ | ---------------------------- |
| 1           | `s_i ← score(m, c_i)`                 | `fusion.py:207` `raw_scores = np.array([scorer(incoming, c) ...])` | ✓                            |
| 2–5         | minmax; `x_i←0.5` if `max=min`        | `fusion.py:208` → `normalize_scores` `:33–35`                      | ✓                            |
| 6           | init 2 components from `q_0.3, q_0.7` | `fusion.py:98` `np.quantile(x,[0.3,0.7])`                          | ✓                            |
| 7–10        | `T` iterations of E-step / M-step     | `fusion.py:111–113`                                                | ✓                            |
| 11          | `k* ← argmax α/(α+β)`                 | `fusion.py:115`                                                    | ✓                            |
| 12          | `g_i ← p(z=k*                         | x_i)`                                                              | `fusion.py:212`              | ✓                   |
| 13          | `I ← {i : g_i ≥ τ}`                   | `fusion.py:218`                                                    | ✓                            |
| 14–16       | `if                                   | I                                                                  | <m_min: I←TopK({x_i},m_min)` | `fusion.py:219–220` | ✓ (replaces `I`, as the paper does) |
| 17–19       | `if I=∅: return NewSession(m)`        | `fusion.py:222–228` (+ early return `:204`)                        | ✓                            |
| 20–21       | `s*←argmax_{i∈I} x_i; Merge`          | `fusion.py:230–236`                                                | ✓                            |

```python
# fusion.py:218–236
eligible = [i for i, g in enumerate(posteriors) if g >= cfg.tau]
if len(eligible) < cfg.m_min:
    eligible = list(np.argsort(-x)[: cfg.m_min])
if not eligible:
    return MergeDecision(action="NEW", target_id=None, ...)
best_idx = max(eligible, key=lambda i: x[i])
return MergeDecision(action="MERGE", target_id=ids[best_idx], ...)
```

**Assess.** ✓ **Correct, line-for-line.** Two points of craft worth recording: (a) the code
notices that with `m_min ≥ 1` the `I = ∅` branch (Alg. 1 line 17) is unreachable except when
`candidates` is empty, and handles that case with an explicit early return rather than leaving
dead code — documented at `fusion.py:198–202`; (b) the TopK fallback ranks by `x_i`, not by
posterior, which is what the paper's parenthetical ("equivalently, by the original matching
scores", PDF line ~950) requires and is easy to get wrong.

**On EM convergence:** the code runs a fixed `n_iter=50` with no `ΔL < ε` check. This is
**compliant** — the paper says "EM for a fixed number of iterations (or until convergence)"
(App. B). The audit prompt's claimed "Eq. 24: convergence criterion" does not exist. **Confidence: High.**

### Alg. 2 — Offline Structure Labeling (`supervision.py:132`)

**Locate.** PDF line 1023.

**Compare.** Line 3 `x ← Feat(ξ)` → `supervision.py:145` `extract_features(...).to_array()`.
Lines 4–9 (per-structure run, judge, memutil, weighted sum) → `per_format_rewards`
`supervision.py:109–117`, iterating `FORMAT_ORDER`. Line 10 `y ← argmax_s r(s)` →
`_argmax_format_index` `:120–129`. Line 12 accumulation → the list comprehension `:147–153`.
Reward weights: paper `Require: λ_judge, λ_mem (e.g., 0.7/0.3)`; code default
`RewardConfig(0.7, 0.3)` (`config.py:45–46`).

**Assess.** ✓ **Structurally exact.** Two observations:

- The code adds a **deterministic tie-break** the paper does not specify: `FORMAT_ORDER = (HG,
VC, VS)` with strict `>` keeps the first maximum (`supervision.py:22–23, 121`). This is a
  necessary addition (`argmax` is ambiguous on ties) and is documented, not silent. Note it
  encodes a mild prior toward HG on exact ties.
- `A(ξ; s)` — "run the agent under structure s" — is behind the `FormatRunner` Protocol
  (`supervision.py:41–54`) with the real implementation declared a non-goal. The **algorithm** is
  faithful; the **experiment** it enables is not yet runnable end-to-end. This is a scope
  boundary, correctly labeled, but it means Alg. 2 has never been exercised against a real agent.

**Confidence: High** on structure; **Medium** on whether reward magnitudes will behave as the
paper's do, since `Judge`/`MemUtil` are ADR 0002 substitutes.

### Alg. 3 — Supervised Training of Structure Selector (`selector.py:82`)

**Locate.** PDF line 1045.

**Compare.**

| Alg. 3 line | Pseudocode                    | Code                                                    | Match                                         |
| ----------- | ----------------------------- | ------------------------------------------------------- | --------------------------------------------- |
| 1           | fit StandardScaler on `{x_i}` | `selector.py:91–92`                                     | ✓ + fit on **train only**, `transform` on val |
| 2–3         | epochs × minibatches          | `selector.py:111, 115–116` (`randperm`)                 | ✓                                             |
| 4           | normalize                     | `:91–92` (hoisted out of the loop; equivalent, cheaper) | ✓                                             |
| 5           | `p ← Softmax(f_θ(x̃))`         | `:119` `F.cross_entropy(logits, ...)` (fuses softmax)   | ✓                                             |
| 6           | Adam SGD step                 | `:99, 117–121`                                          | ✓                                             |
| 8           | **early stop on validation**  | `:130–138`                                              | ✓                                             |
| 10          | return θ                      | `:140–143` best-state restore                           | ✓ **exceeds spec**                            |

```python
# selector.py:130–141
if val_loss < best_val_loss - 1e-6:
    best_val_loss = val_loss
    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1
    if epochs_without_improvement >= self._config.patience:
        stopped_epoch = epoch + 1
        break
if best_state is not None:
    self._net.load_state_dict(best_state)
```

**Assess.** ✓ **Fully implemented.** This directly contradicts the audit prompt's Few-Shot
Example 3, which presents early stopping as missing with a `# TODO`. **No such TODO exists in the
repository** (`grep -rn "TODO" fluxmem/` returns nothing relevant). The prompt's Example 3 is
illustrative fiction and must not be read as a finding.

Two non-divergences worth recording so they are not re-litigated: the scaler is fit on train only
(`:61–65` argues why — leakage), which the paper's line 1 leaves ambiguous; and `learning_rate`
defaults to `1e-2` (`config.py:53`), where the prompt asserted `1e-3` — **the paper specifies no
learning rate at all**, so neither value is a divergence. **Confidence: High.**

---

## 5. Divergence Log

### D-0 — The audit prompt's reference table contradicts the paper _(Meta; Intent: oversight in the prompt, not the code)_

**Paper spec (verified via `pdftotext -layout`, equation tags at flux.txt lines 229–948):**

| Eq.    | Actual paper content         | Prompt's claim (lines 114–129) | Verdict              |
| ------ | ---------------------------- | ------------------------------ | -------------------- |
| 6      | LTSM eligibility `u∧r∧(c)`   | "Promotion condition"          | ✓ roughly right      |
| 7      | `f_θ(x_t)=Softmax(g_θ(x_t))` | "`U = w₁c+w₂ℓ+w₃d`"            | ✗ **wrong**          |
| 8      | `ŝ_t = argmax f_θ(x_t)[s]`   | "[0,1] normalization"          | ✗ **wrong**          |
| 19     | Beta pdf definition          | "E-step responsibility"        | ✗ **wrong**          |
| 20     | E-step (log-space)           | "M-step `π_k←Σr/N`"            | ✗ **off by one**     |
| 21     | `π_k ← N_k/n`                | "moment matching κ_floor"      | ✗ **off by one**     |
| 22     | `μ_k, σ²_k`                  | "variance stabilization"       | ✗ **wrong**          |
| 23     | κ moment matching            | "log-space EM"                 | ✗ **wrong**          |
| 24     | `k*=argmax α/(α+β)`          | "convergence criterion ΔL<ε"   | ✗ **does not exist** |
| 25     | `g(x)` gate posterior        | _omitted entirely_             | ✗ **missing**        |
| Alg. 1 | BMM-gated fusion             | "STIM push/evict"              | ✗ **wrong**          |
| Alg. 2 | Offline labeling             | "referenced in Eq. 5"          | ✗ **wrong**          |
| Alg. 3 | Selector training            | ✓                              | ✓                    |

**Code implementation:** `fluxmem/`'s docstrings cite `F Eq. 17`, `F Eq. 20`, `F Eq. 21`,
`F Eq. 23`, `F Eq. 25`, `F Eq. 26` (`fusion.py:25, 122, 139, 152, 163, 192`) — **every one of
these matches the real PDF.** The code is the accurate artifact here.

**Intent:** Oversight in the prompt (plausibly drafted from memory or an earlier preprint
numbering). Not a code defect.

**Risk:** **High to the audit process, zero to the system.** The prompt instructs the auditor to
"flag silent divergences ... even if tests pass" and to treat absence as ✗ — an auditor obeying it
literally against this table would file ~10 false defects against correct code, and specifically
would have reported early stopping as missing (Few-Shot Example 3) when it is present and tested.

**Recommendation:** Correct lines 114–129 and Example 3 of
`.claude/prompts/fluxmem-paper-code-fidelity-audit.md` against the PDF before the prompt is reused,
and add Eq. 25. Consider deleting the reference table entirely — the code's own docstrings are a
more reliable index, and a second, drifting copy of the equation numbering is exactly the kind of
duplicated source of truth that produced this problem.

---

### D-1 — `ltsm.is_eligible` ignores MTEM's configured `U` scale parameters _(Intent: oversight — genuine defect)_

**Paper spec:** Eq. 6 gates on the same utility quantity MTEM maintains. There is only one `U`.

**Code implementation:**

```python
# ltsm.py:81–84
def is_eligible(episode, weights, thresholds, now) -> bool:
    u = utility_score(episode, weights, now)          # <-- no reference_length, no half_life
    r = recency_gate(episode, now, thresholds.max_age)
    return u >= thresholds.tau_u and r >= thresholds.tau_r
```

`utility_score` (`mtem.py:69–76`) takes `reference_length` and `half_life` as keyword-only
arguments **defaulting to 50.0 and 100.0**. `MidTermEpisodicMemory.utility` (`mtem.py:117–124`)
forwards its instance's configured values; `is_eligible` forwards neither. `MtemConfig`
(`config.py:96–98`) exposes both as tunable fields, so a config that changes them silently
desynchronizes the two.

**Verification (executed, not assumed):** one `EpisodicUnit` (1 turn, 10+10 tokens,
`access_count=1`, `last_access=0.0`), `w1=w2=w3=1`, `now=3.0`,
`MidTermEpisodicMemory(reference_length=5.0, half_life=1.0)`:

```
MTEM-configured U : 1.2997870683678638
ltsm is_eligible U: 1.6204455335485082
```

A 25% discrepancy on the same episode. `mtem.promotion_candidates` (`mtem.py:153–159`, which uses
the configured values) and `ltsm.promote` (`ltsm.py:132–155`, which does not) will therefore
disagree about which episodes are promotable — the two functions the codebase offers for the same
job give different answers.

**Why the tests miss it:** `tests/test_ltsm.py` constructs episodes and calls `is_eligible`
directly with default scales; no test builds an MTEM with non-default `reference_length` or
`half_life` and then asserts agreement with the LTSM gate. Coverage is 98% on `ltsm.py` — this is
a precise illustration of the prompt's own anti-pattern, _"tests verify behavior, not fidelity to
spec."_

**Risk:** **Medium.** Latent today (nothing in-repo overrides the defaults — there is no assembled
pipeline yet), but `AdaStoreConfig` is designed to be the assembly point and both fields are
user-facing. If promotion rate is ever reported in an experiment run under a tuned `MtemConfig`,
the number will be wrong with no error raised. This also violates
`.claude/rules/evidence-discipline.md`'s stated target failure mode.

**Recommendation:** Thread the parameters through — either pass them explicitly from the caller,
or (preferable, and it removes the whole class of bug) make `is_eligible` accept the
`MidTermEpisodicMemory` instance and call `mtem.utility(episode, now)`, so exactly one object owns
the `U` definition. Add a regression test asserting
`mtem.utility(e, now) == U used by is_eligible` under non-default scales.

---

### D-2 — Eq. 6 conjunct 1: `u(m)` (usage) implemented as composite `U(e)` _(Intent: design choice — documented)_

**Paper spec:** FluxMem Eq. 6 — `u(m)` is "usage" alone. COLM §1.3.2 writes `U(e_j) ≥ τ_u`, where
`U` is COLM Eq. 5's composite.

**Code:** `ltsm.py:82` uses the composite.

**Intent:** Design choice, argued at length in `ltsm.py:3–18` and consistent with the project
Reading rule (COLM wins on conflicts) formalized in ADR 0008. The docstring correctly identifies
the substantive consequence: the composite is **compensatory** (a rarely-accessed but long and
recent episode can promote via `w₂ℓ + w₃d`) where FluxMem's usage-only reading is not.

**Risk:** **Low, but with a live experimental hazard the docstring already names:** `τ_u` now lives
on scale `[0, w1+w2+w3]` rather than a raw count, so promotion is coupled to the weights and
FluxMem's published `τ_u` is not portable. Recommendation: ensure the experiment record logs
`w1,w2,w3` alongside `τ_u` — without both, promotion rate is not reproducible. This is asserted in
the docstring but I found no code or test that enforces it.

---

### D-3 — Eq. 6 conjunct 3 `c(m) ≥ τ_c` not implemented _(Intent: design choice — documented)_

**Paper spec:** parenthesized and explicitly labeled optional in the paper's own notation:
`(∧ c(m) ≥ τ_c)`. COLM omits it.

**Code:** two conjuncts only (`ltsm.py:84`).

**Intent:** Documented at `ltsm.py:20–32` with a correct monotonicity argument — dropping a pure
AND-conjunct makes the gate strictly **more permissive**, so it cannot reject anything the
remaining gates accept. Confidence would require a fact extractor rating its own output, which no
component in this scope emits. Notably, the absence is **enforced**: `tests/test_config.py:35–46`
asserts no `tau_c` field and no `tau_c` YAML key exist, per `config.py:9–11` ("config is for
tuning, not for hedging").

**Risk:** **Low.** Correctly classified as a known, non-silent gap rather than an oversight. First
knob to revisit if LTSM precision underperforms.

---

### D-4 — 7 features, not FluxMem Table 5's 12 _(Intent: design choice — documented, and COLM-faithful)_

**Paper spec:** FluxMem App. C, Table 5 lists **12** features (page count, avg page length, entity
density, relation indicators, topic diversity, topic transitions, is_qna_pattern,
is_decision_tree, is_entity_centric, time_span, temporal_density, semantic_complexity).

**Code:** `features.py:29–38` defines exactly 7: `entity_count`, `cooccurrence_density`,
`temporal_ordering`, `topic_diversity`, `token_length`, `hyperedge_density`, `visual_salience`.

**Verification:** COLM §1.3.3 lines 122–127 specify precisely this set — "the number of distinct
named entities and their co-occurrence density ..., the degree of temporal ordering ..., the topic
diversity ... and the total token length. These are the same classes of features used by FluxMem
(Lu et al., 2026), **extended with two additional dimensions for hyperedge density and visual
salience score**." The code is a **verbatim** implementation of COLM's list.

**Intent:** Design choice under the Reading rule (ADR 0008). Not drift.

**Risk:** **Low for COLM fidelity; Medium for cross-paper comparability.** Because FluxMem's
selector consumes a 12-d input and this one consumes 7-d, selector accuracy is **not** directly
comparable to FluxMem's reported numbers, and any claim of reproducing FluxMem's selector results
would be unsupported. Recommendation: state the feature-set difference explicitly in any results
table that sits beside FluxMem baselines. Note also that `hyperedge_density` and `visual_salience`
are read straight off the episode and default to `0.0` (`interfaces.py:68–69`) until HetRep
populates them — so **2 of 7 input dimensions are currently constant zero**, which will degrade
selector discrimination in any experiment run before HetRep lands. That consequence is implied by
`features.py:143–146` but not stated as an experimental caveat anywhere I found.

---

### D-5 — `r_judge` / `r_mem` concretizations _(Intent: design choice — ADR 0002)_

**Paper spec:** Eq. 9 gives the linear form; `Judge(·)` and `MemUtil(·)` are named in Alg. 2 lines
6–7 but never defined. This is a genuine paper silence, not an omission by the code.

**Code:** `ExactMatchF1Judge` = token-overlap F1 (`supervision.py:57–76`); `TokenBudgetMemUtil` =
`hit_rate / (1 + token_budget/reference_tokens)` (`supervision.py:79–99`).

**Intent:** ADR 0002 (`docs/adr/0002-format-selector-reward-design.md`) explicitly decides both —
"(a), exact-match/F1" (line 50) and "(b), hit-rate normalized by retrieved token budget" (line
75), with consequences recorded at lines 105–118 including "approximation, not an oversight" and a
swap path once ITERRET exists. Both are injected behind Protocols, so the swap is real.

**Risk:** **Low.** Correctly documented at both the ADR and docstring level.

**Two corrections to the audit prompt's framing of this item.** (i) The prompt's Example 2 shows
`r_mem = hit_rate / (token_budget + 1e-8)`, which would be an unstable division; the **actual**
code uses `hit_rate / (1.0 + token_budget/reference_tokens)`, whose denominator is `≥ 1` by
construction — **no division-by-zero is reachable**, and `reference_tokens > 0` is validated at
`supervision.py:88–89`. The prompt's example is fiction; the real code is safer than the example
it warns about. (ii) `config.py:40` describes 0.7/0.3 as "FluxMem's own tuned values — cited, not
derived," but the paper writes `Require: Reward weights λ_judge, λ_mem (e.g., 0.7/0.3)` (PDF line
1026). "e.g." is an illustrative default, not a tuned result. **Minor citation overstatement**;
recommend softening the comment to "FluxMem's stated example values."

---

### D-6 — STIM LRU currently degenerates to FIFO _(Intent: incomplete — documented, blocked on out-of-scope work)_

**Paper spec:** Eq. 5's `τ(p)` is "the last access time of page p" — LRU proper.

**Code:** `stim.py` implements true LRU via `OrderedDict` + `touch()` (`:63–71`), but **nothing in
the repository calls `touch()`.** `grep -rn "\.touch(" fluxmem/` returns no call site outside
tests. With no reads, last-access ≡ insertion time and eviction order is FIFO.

**Intent:** Explicitly documented at `stim.py:18–25`: "Today nothing calls `touch()` (COLM Alg. 1's
write phase never re-reads STIM), so this is currently FIFO in practice — it starts to matter the
moment a read path (ITERRET, out of scope here) touches STIM." Incomplete rather than wrong: the
mechanism exists and is correct; its activator is out of scope.

**Risk:** **Low.** The behavior is identical to LRU until a read path exists, and the write-path
sequencing means no information is lost today. Flagged here only so it is not mistaken for a ✓ —
per the prompt's rule, an approximated behavior is ⚠, not ✓.

---

### D-7 — Eq. 5 evicts one page, not an arbitrary subset _(Intent: design choice — benign; noted for completeness)_

**Paper spec:** `E_t = argmin_{P⊂M_t, |P|=|M_t|−C} Σ_{p∈P} τ(p)` — a **set** of `|M_t|−C` pages.

**Code:** `stim.py:58–61` evicts exactly one page per `push`.

**Analysis:** Since `push` admits one turn at a time, `|M_t|` never exceeds `C+1`, so
`|M_t|−C = 1` and the subset argmin reduces to the single-element argmin the code computes. **The
implementations are equivalent under the code's call pattern.** They would diverge only under a
bulk-insert API, which does not exist. Recording it because it is a latent constraint: if a
`push_many` is ever added, `stim.py:54–61` must be revisited.

**Risk:** **Very Low.** No action required now.

---

## 6. Known Assumptions

**Verified in this audit (command output or direct file read):**

1. **Test suite is green and has grown.** `pytest tests/ -q --cov=fluxmem` →
   `161 passed in 4.12s`, `TOTAL 650 stmts, 5 miss, 99%`. The closure report's 146 has become 161.
   The 5 uncovered lines are `fusion.py:85`, `ltsm.py:103`, `mtem.py:115`, `selector.py:80`,
   `stim.py:46` — all trivial property getters / guard clauses, none equation-bearing.
2. **Paper equation numbering**, extracted from the PDF (`pdftotext -layout`), tags at flux.txt
   lines 229 (Eq. 5), 302 (7), 306 (8), 319 (9), 849 (17), 857 (18), 862 (19), 875 (20), 883 (21),
   891 (22), 898 (23), 905 (24), 909 (25), 948 (26); Alg. 1/2/3 at lines 916/1023/1045.
3. **Paper capacity constants:** "STIM (capacity 4, LRU), MTEM (up to 2000 episodic sessions" —
   PDF lines 372–373. Both match `config.py:90, 96`.
4. **Paper reward weights:** `Require: Reward weights λ_judge, λ_mem (e.g., 0.7/0.3)` — PDF line 1026. Matches `config.py:45–46` in value; see D-5(ii) on the "tuned" characterization.
5. **D-1 is real**, demonstrated by executing both code paths (1.2998 vs 1.6204).
6. **COLM §1.3.3 specifies exactly the 7 implemented features** — COLM lines 122–127, quoted in D-4.
7. **Alg. 3 early stopping exists** (`selector.py:130–138`), contradicting the audit prompt's
   Example 3. No relevant `TODO` markers exist in `fluxmem/`.
8. **ADR 0002 exists and covers Eq. 9's concretization** — decisions at lines 50 and 75,
   consequences at 105–118.
9. **`tau_c`'s absence is test-enforced** — `tests/test_config.py:35–46`.

**Assumed / not verified:**

1. **I did not re-verify the closure report's 146/99% claim**; I verified today's 161/99% directly.
   The delta is not explained by anything I read — if the closure report is meant to be the
   standing record, it is now stale by 15 tests.
2. **I did not execute the BMM against adversarial score distributions.** The floors are present
   and argued, and log-space is used throughout, but I verified guard _presence_ by reading, not
   guard _sufficiency_ by fuzzing. A property test over degenerate inputs (all-identical scores,
   n=1, scores at 0/1 boundaries) would convert this from reasoned to demonstrated.
3. **Eq. 2–4 and Eq. 11–16 were classified out-of-scope by reading the audit prompt's scope
   statement plus the non-goal declarations in `ltsm.py:48–54` and `supervision.py:42–46`.** I did
   not search for an ADR that formally scopes out ITERRET/WORKMEM; the non-goal is asserted in
   docstrings only. If a reviewer wants that to be a documented decision rather than a comment, it
   needs an ADR.
4. **Selector accuracy vs. FluxMem's reported numbers is untested** — Alg. 2 has never run against
   a real agent (`FormatRunner` has only test fakes), so no empirical claim about reward-derived
   labels is currently supportable.
5. **`selector.save/load` use `pickle`** (`selector.py:161–181`). This is a write to disk, not a
   committed artifact, so `.claude/rules/storage-invariants.md`'s "never commit `*.pkl`" is not
   violated by the code itself — but the artifact carries the same torch/sklearn build coupling
   that rule exists to prevent, and `load` will unpickle arbitrary input. Out of this audit's
   equation scope; flagged for the storage/security reviewer.

---

## 7. Closure Checklist

- [x] **All Eq. 5–26 located and verified** — 16/16 in-scope equations mapped to `file:line`
      (§3). Eq. 2–4, 11–16 classified out-of-scope with reasons. **Note:** verification was
      performed against the PDF, not against the audit prompt's reference table, which is
      incorrect (D-0).
- [x] **All Alg. 1–3 traced in code** — line-by-line tables in §4. Alg. 1 ✓ (21/21 lines), Alg. 2
      ✓, Alg. 3 ✓ (including early stopping, which the prompt wrongly presents as missing).
- [x] **Edge cases reviewed** — division by zero: Eq. 17 `s_max=s_min` handled (`fusion.py:33`),
      Eq. 22 `n_k_safe` floor (`:155`), `r_mem` denominator `≥1` (`supervision.py:99`),
      `_l2_normalize` zero-norm guard (`ltsm.py:89`). log(0): avoided by `logsumexp` +
      `kappa_floor`/`mu_floor` keeping `α,β > 0`. Empty sets: `fusion.py:95, 204`,
      `mtem.py:47`, `features.py:92, 111, 128`, `supervision.py:61, 93`, `ltsm.py:121`. NaN: not
      reachable given the above floors, by reading — **not fuzz-tested** (see Assumption 2).
- [x] **Divergences classified by intent** — 8 logged: 1 oversight in the prompt (D-0), 1
      oversight in code (D-1), 4 documented design choices (D-2, D-3, D-4, D-5), 1 incomplete
      pending out-of-scope work (D-6), 1 benign equivalence (D-7).
- [x] **Test suite coverage confirmed** — **161 passed, 99%** (re-run today), superseding the
      closure report's 146/99%. Note the caveat: D-1 is a real defect at 98% coverage of
      `ltsm.py`, so coverage is not evidence of fidelity.
- [x] **Assumptions explicitly listed** — §6, split into 9 verified and 5 assumed.

---

## Overall Assessment

**Fidelity score: 87.5% verbatim (14/16), 100% present (16/16), 0 missing.**
**Risk: Low**, with one Medium defect (D-1) and one process risk (D-0).

This codebase is unusually well-behaved against its sources: every deviation from FluxMem except
one is argued in a docstring or an ADR, the equation citations in the code check out against the
actual PDF, and the two hardest items in the paper (Alg. 1's fusion gate and the log-space EM
loop) are implemented correctly including guards the paper only gestures at. The prevailing
pattern of divergence is _deliberate substitution of COLM's reading for FluxMem's_, which is a
project rule (ADR 0008) rather than drift.

The single actionable code finding is **D-1**, and it is worth taking seriously precisely because
it survived a 99%-coverage test suite and a prior closure report: it produces no exception, only a
wrong `U`. Recommended actions, in priority order:

1. **Fix D-1** — collapse ownership of `U` onto one object; add a regression test under
   non-default `MtemConfig` scales.
2. **Fix D-0** — correct or delete the reference table and Example 3 in
   `.claude/prompts/fluxmem-paper-code-fidelity-audit.md` before it is used again.
3. **Record the D-4 caveat** — 2 of 7 selector features are constant zero until HetRep lands;
   selector results are not comparable to FluxMem's 12-feature baseline.
4. **Soften the D-5(ii) citation** — 0.7/0.3 is the paper's "e.g.", not a tuned value.
5. **Refresh or supersede** `docs/FluxMem-Closure-Report.md`, now stale at 146 tests.

---

_Audit performed 2026-08-13 against `fluxmem/` at branch `feat/adastore-fluxmem`. Paper text
extracted with `pdftotext -layout`. Test results from a live run in the `caimms` environment
(Python 3.11.15), not quoted from prior documents._
