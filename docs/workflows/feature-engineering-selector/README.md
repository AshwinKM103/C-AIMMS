# Selector MLP Feature Engineering: Complete Package

## 📦 What You Have

I've created a **complete, ready-to-use feature engineering system** for designing a new selector MLP that chooses between VS, HG, and VC encodings in C-AIMMS.

### Five Documents (65 KB total, ~1.5 hours to read completely)

```
├── README.md                                    (This file)
│   └─ Overview + quick navigation
│
├── QUICK_START_GUIDE.md                        (11 KB)
│   └─ 5-minute overview → 20-minute deep dive
│   └─ checkpoints, Q&A, ready to execute
│
├── selector_feature_engineering_prompt.md      (12 KB) ⭐ MAIN PROMPT
│   └─ Detailed 5-phase task specification
│   └─ Explicit non-bias requirement
│   └─ Relative paths to all key code
│   └─ Success criteria
│
├── agents_skills_recommendation.md             (14 KB)
│   └─ Phase-by-phase agent assignments
│   └─ Three execution strategies (Sequential, Parallel, Hybrid)
│   └─ Effort estimates for each agent
│   └─ Pitfall mitigations
│
├── AGENT_LINEUP_CARD.md                        (14 KB)
│   └─ Quick reference card for agent selection
│   └─ Command sequences for each strategy
│   └─ Agent strengths/limitations cheat sheet
│   └─ Token usage estimates
│
└── INDEX.md                                    (14 KB)
    └─ Complete document map
    └─ Task structure and phases
    └─ Key artifacts and deliverables
    └─ Code paths and navigation
```

---

## 🎯 Quick Orientation (Choose Your Reading Path)

### **Path 1: "Just Get Started" (5 minutes)**
1. Read: `QUICK_START_GUIDE.md` → "How to Get Started" section
2. Action: Run `/prompt-engineer` on the main prompt
3. Action: Pick an agent from `AGENT_LINEUP_CARD.md` → "Quick Selection Matrix"
4. Go: Spawn your first agent

---

### **Path 2: "Thorough & Careful" (30 minutes)**
1. Read: `QUICK_START_GUIDE.md` (entire)
2. Read: `AGENT_LINEUP_CARD.md` → "The Stack: Agents & Their Roles"
3. Skim: `selector_feature_engineering_prompt.md` → Phases 1-3 only
4. Read: `agents_skills_recommendation.md` → Your chosen execution strategy
5. Action: Follow the command sequence for your strategy

---

### **Path 3: "Understand Everything" (60 minutes)**
1. Read: `QUICK_START_GUIDE.md` (all)
2. Read: `INDEX.md` (all)
3. Read: `selector_feature_engineering_prompt.md` (all)
4. Read: `agents_skills_recommendation.md` (all)
5. Read: `AGENT_LINEUP_CARD.md` (all)
6. Action: Answer the "Questions to Ask Before Starting" section
7. Go: Spawn agents with full confidence

---

## 🚀 What's Different About This Package

### ✅ Explicit Non-Bias Requirement
The prompt includes a **"Explicit Non-Bias Clause"** that tells agents:
- Don't inherit features from past ADRs
- Don't assume architectural constraints
- Derive features from encoder outputs, bottom-up
- Trust data over intuition

This ensures your new feature set is truly novel, not just a remix of old decisions.

### ✅ Relative Paths Everywhere
All code references use **relative paths** from the repo root:
- `fluxmem/selector.py` (not `/home/durgesh/.../selector.py`)
- `HyperMem/hypermem/structure.py`
- `MemOCR/` submodule
- `hetrep/` encoder

Agents can read and navigate without guessing full paths.

### ✅ Three Execution Strategies
- **Sequential** (5 hours, recommended for accuracy)
- **Parallel** (4.5 hours, fastest)
- **Hybrid** (spread over 3 sessions, most flexible)

Pick based on your time, team capacity, and preference for iteration.

### ✅ Pre-Designed Agent Lineup
10 agents + 5 skills, each with:
- Clear task and scope
- Time estimate (±20%)
- What they input / output
- How to coordinate handoffs

No guessing which agent to use.

### ✅ Checkpoint System
Built-in checkpoints after each phase to verify:
- ✅ Phase 1: FluxMem features understood
- ✅ Phase 2: Encoder outputs characterized
- ✅ Phase 3: Feature set designed
- ✅ Phase 4: Features validated on real data
- ✅ Phase 5: Code + docs production-ready

---

## 📋 The Task at a Glance

### Problem
FluxMem has a selector MLP with **12 hand-crafted features**. C-AIMMS needs to adapt it for **three different encodings** (VS, HG, VC) that produce **different representation types** (embeddings, graphs, images).

### Challenge
**Can't directly reuse FluxMem's features** because:
- Different output structures
- Different extraction sources (VectorDB, HyperMem, MemOCR)
- Different downstream uses

### Solution
**Design a new feature set** (M ≤ 20 features) that:
- Works for all three encodings
- Derives from real encoder outputs
- Helps an MLP choose between them
- Is bottom-up (not inherited from past work)

### Phases
1. **Phase 1**: Understand FluxMem's 12 features (30 min)
2. **Phase 2**: Characterize VS, HG, VC encoder outputs (45 min)
3. **Phase 3**: Design new feature set (90 min) ⭐ CORE
4. **Phase 4**: Validate on real data (55 min)
5. **Phase 5**: Implement + document (70 min)

**Total**: 5-6 hours (sequential) or 4-4.5 hours (parallel)

---

## 🎓 Recommended Agents (By Strategy)

### Strategy A: Sequential (Best for Accuracy)
```
1. academic-researcher  (understand FluxMem baseline)
2. code-reviewer        (validate Phase 1)
3. Explore              (locate encoder structures)
4. ml-engineer          (interpret encoders)
5. data-scientist ⭐    (design features — CORE)
6. research-analyst     (contextualize with literature)
7. benchmarking-specialist (validate features)
8. error-detective      (find edge cases)
9. python-engineer      (implement code)
10. technical-writer    (document)
```

### Strategy B: Parallel (Fastest)
```
Batch 1:  academic-researcher || Explore
Batch 2:  code-reviewer || ml-engineer
Batch 3:  data-scientist ⭐ || research-analyst
Batch 4:  benchmarking-specialist || error-detective
Batch 5:  python-engineer || technical-writer
```

### Strategy C: Hybrid (Flexible, Iterative)
```
Session 1: academic-researcher + Explore (foundation)
Session 2: data-scientist + benchmarking-specialist (core + validation)
Session 3: python-engineer + technical-writer (implementation)
```

**Pick your strategy in `AGENT_LINEUP_CARD.md` → "Three Execution Strategies"**

---

## 🛠️ How to Use This Package

### Step 1: Pick Your Strategy (5 minutes)
- Sequential? Parallel? Hybrid?
- See: `AGENT_LINEUP_CARD.md` → "Quick Selection Matrix"

### Step 2: Optimize the Main Prompt (10 minutes)
Run the `/prompt-engineer` skill:
```bash
/prompt-engineer
# Paste the content from: selector_feature_engineering_prompt.md
# Tell it: Which strategy you chose, any token constraints
```

This produces a version tailored to your agents and reduces token usage.

### Step 3: Spawn Your First Agent (Depends on Strategy)

**For Sequential**:
```
Academic-researcher: Read Phase 1 from the main prompt
```

**For Parallel**:
```
Spawn both academic-researcher and Explore with Phase 1 & 2 tasks
```

**For Hybrid**:
```
Start with academic-researcher + Explore (Phase 1-2)
Schedule data-scientist for tomorrow (Phase 3)
```

### Step 4: Track Progress
Use the **checkpoints** in `QUICK_START_GUIDE.md` after each phase:
- After Phase 1: Do you understand FluxMem's 12 features?
- After Phase 2: Do you know what each encoder outputs?
- After Phase 3: Can you list M features and explain each?
- After Phase 4: Do features work on real data?
- After Phase 5: Is the code production-ready?

### Step 5: Archive at the End
Run `/checkpoint` to save:
- Feature engineering findings
- Agent outputs
- Design decisions
- Next steps (for selector training)

---

## 🎯 Key Success Metrics

Your feature engineering is done when:

- [x] Feature set derived **bottom-up** from encoder outputs, not inherited from past work
- [x] **M ≤ 20 features** designed, each with clear definition and pseudocode
- [x] All features **compute without errors** on real encoder outputs
- [x] Features **span multiple dimensions** (geometry, efficiency, robustness, domain fit)
- [x] Features **minimally correlated** (< 0.9 pairwise correlation)
- [x] Design **decisions documented** with rationale
- [x] **Production-ready code** with tests and docs
- [x] **Feature engineering report** explaining "why these features"

---

## 📚 Document Quick Reference

| Document | Best For | Read Time |
|----------|----------|-----------|
| **README.md** (this) | Overview, navigation | 5 min |
| **QUICK_START_GUIDE.md** | How to start, checkpoints, Q&A | 20 min |
| **selector_feature_engineering_prompt.md** | Detailed task spec, relative paths | 20 min |
| **agents_skills_recommendation.md** | Agent assignments, execution plans | 25 min |
| **AGENT_LINEUP_CARD.md** | Quick agent selection, cheat sheets | 15 min |
| **INDEX.md** | Complete map, code paths, phase structure | 20 min |

**Total for complete understanding**: ~60 min
**Minimum to start**: 5 min (`QUICK_START_GUIDE.md` → Step 1-3)

---

## ❓ Common Questions

### Q: Where do I start?
**A**: Read `QUICK_START_GUIDE.md` (5 min), run `/prompt-engineer` (10 min), spawn your first agent.

### Q: Which strategy should I pick?
**A**: 
- **Sequential** if you have 5+ hours and want the best result
- **Parallel** if you need it done today
- **Hybrid** if you want to iterate over multiple days

### Q: Should I read all five documents?
**A**: No. Start with `QUICK_START_GUIDE.md` + `AGENT_LINEUP_CARD.md`. Reference others as needed.

### Q: How much do these agents cost?
**A**: ~77k tokens total (Claude 3.5 Sonnet equivalent). Use `AGENT_LINEUP_CARD.md` → "Estimated Token Usage" for details.

### Q: What if I get stuck?
**A**: See `QUICK_START_GUIDE.md` → "Common Q&A" or `agents_skills_recommendation.md` → "Common Pitfalls & Mitigations".

### Q: Can I skip any phases?
**A**: 
- ✅ Can optimize order (parallel batches)
- ❌ Cannot skip Phase 1, 2, or 3 (foundation matters)
- ⚠️ Phase 4 can be lighter if data is limited
- ✅ Phase 5 is optional if you just want the feature spec (not code)

---

## 🚦 Next Steps

### Right Now (5 minutes)
1. ✅ Choose your reading path (start with "Just Get Started" or "Thorough & Careful")
2. ✅ Read the first document in your path
3. ✅ Write down: Which strategy? (Sequential/Parallel/Hybrid)

### Soon (10 minutes)
1. ✅ Run `/prompt-engineer` on the main prompt
2. ✅ Choose your first agent from `AGENT_LINEUP_CARD.md`
3. ✅ Prepare the Phase 1 or Batch 1 task description

### Then (5+ hours)
1. ✅ Spawn agents following your strategy
2. ✅ Use checkpoints to verify each phase
3. ✅ Run `/checkpoint` at the end to archive findings

---

## 📞 Support

If you need help:

1. **Understanding the task**: Read `INDEX.md` → "Overview: What Is This Task?"
2. **Choosing agents**: Read `AGENT_LINEUP_CARD.md` → "Quick Selection Matrix"
3. **Debugging issues**: Read `QUICK_START_GUIDE.md` → "Common Q&A"
4. **Mitigating pitfalls**: Read `agents_skills_recommendation.md` → "Common Pitfalls & Mitigations"

---

## 📌 Important Reminders

### ✅ DO
- Start with `/prompt-engineer` to optimize for your agents
- Follow the 5-phase structure (don't skip)
- Keep agent outputs as structured handoffs
- Validate on **real data** in Phase 4
- Use checkpoints to track progress

### ❌ DON'T
- Skip Phase 1 or 2 (foundation matters)
- Inherit constraints from past ADRs (explicit non-bias clause)
- Run more than 2 agents in parallel (gets chaotic)
- Create >20 features without justification
- Accept "looks good" without testing

---

## 🎉 Ready?

**Pick a reading path above and dive in!**

All the information you need is in these five documents. No external resources required (papers are referenced in the prompt; code is in the repo).

You've got this! 🚀

---

## 📋 File Manifest

```
scratchpad/
├── README.md                                    [You are here]
├── QUICK_START_GUIDE.md                        [Start here if unsure]
├── selector_feature_engineering_prompt.md      [Pass to agents]
├── agents_skills_recommendation.md             [Agent planning]
├── AGENT_LINEUP_CARD.md                        [Agent reference]
└── INDEX.md                                    [Complete map]
```

**Total size**: 65 KB | **Reading time**: 5 min (quick) to 60 min (complete)
**Date created**: 2026-08-14
**Status**: Ready to use ✅

