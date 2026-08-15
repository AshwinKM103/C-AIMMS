# Contributing to C-AIMMS

Thank you for your interest in contributing to the Cognitive AI Memory Architecture project! This guide covers general contribution practices for all components (FluxMem, HetRep, HyperMem, MemOCR, IterRet).

## General Contribution Guidelines

### Git Workflow

**Branching**: Create feature branches from main using descriptive names:

```bash
git checkout -b feature/your-feature-name
```

Use prefixes: `feature/`, `fix/`, `chore/`, `refactor/`, `docs/`.

**Commit Style**: Follow [conventional commits](https://www.conventionalcommits.org/):

```bash
git commit -m "type(scope): description"
# Examples:
# feat(hypermem): add adaptive retrieval config
# fix(memocr): correct tensor shape in stage 3
# docs(iterret): clarify setup instructions
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `style`, `ci`, `exp`
(see `.claude/rules/git-workflow.md` for details).

**Pull Requests**:

- One logical change per PR (don't mix features, refactoring, and docs)
- Keep PRs under 400 lines of diff when possible
- Write a clear description: what changed, why, how to test
- Link related issues: `Closes #123`
- Include a test plan describing verification steps

### Environment Setup

All four active components (FluxMem/HetRep, HyperMem, MemOCR, IterRet) share a unified conda environment:

```bash
git clone --recursive https://github.com/AshwinKM103/C-AIMMS.git
cd C-AIMMS
conda activate caimms  # or: conda create -n caimms python=3.11.15 && conda activate caimms
pip install -e '.[hypermem,memocr,iterret]'  # install all components
# Or: pip install -e '.[hypermem]' to install only HyperMem
```

**Python version**: 3.11.15 (canonical per ADR 0009), 3.10–3.11 supported.

Each component's dependencies are declared as optional-dependency groups in `pyproject.toml`:

- `hypermem`: Hydra, OmegaConf, LLM clients, retrieval utils
- `memocr`: Full RL training stack (transformers, ray, vllm, flash-attn, etc.)
- `iterret`: LangGraph, LLM integration

### Code Quality

**Linting & Formatting**: The project uses **ruff** (see `pyproject.toml`):

```bash
ruff check .           # lint
ruff format .          # auto-format
```

**Type hints**: Use type annotations throughout (especially on public APIs):

```python
def retrieve_facts(query: str, top_k: int = 30) -> List[Fact]:
    """Retrieve top-k facts matching the query."""
```

**Naming conventions** (from `.claude/rules/naming.md`):

- Variables and functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Booleans: prefix with `is`, `has`, `can`, `should`

**Docstrings**: Document public APIs:

```python
def extract_episodes(dialogue: List[str]) -> List[Episode]:
    """Extract episodes from a dialogue stream.

    Args:
        dialogue: List of dialogue turns

    Returns:
        List of extracted episodes
    """
```

### Testing

**Run tests** before committing:

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_config.py -v

# With coverage (target >80% on new code)
pytest --cov=hypermem tests/
```

**Write tests** that mirror source structure:

- Place in `tests/` directory
- Use descriptive names: `test_should_extract_episodes_from_dialogue()`
- One logical behavior per test; avoid unnecessary mocks

---

## HyperMem Development

HyperMem is a hypergraph-based long-term conversation memory system. This section covers HyperMem-specific workflows.

### Quick Start

```bash
# From C-AIMMS root, assuming caimms env with hypermem extras is activated
cd HyperMem

# One-time setup (creates .env.local and project dirs)
bash scripts/setup_local.sh

# Run tests
make test

# Format code
make format
```

### Project Structure

```
HyperMem/
├── hypermem/
│   ├── config/                 # Config, constants, environment defaults
│   ├── types.py               # Episode, Topic, Fact data structures
│   ├── structure.py           # Hypergraph implementation
│   ├── extractors/            # LLM-driven extraction modules
│   ├── llm/                   # LLM, embedding, reranker clients
│   ├── prompts/               # Prompt templates
│   ├── utils/                 # Utilities
│   └── main/                  # Pipeline stages 1–6 + eval orchestrator
├── scripts/                   # Train/eval/debug/serve scripts
├── tests/                     # Test suite (mirror source structure)
├── data/                      # Datasets (LoCoMo-10, etc.)
├── results/                   # Experiment outputs (generated)
└── CONTRIBUTING.md            # (moved to repo root)
```

### Running Experiments

**Baseline training** (stages 1–5):

```bash
# Dry-run preview
bash scripts/train_baseline.sh --dry-run

# Actual training
bash scripts/train_baseline.sh
# Or:
make train-baseline
```

**Ablation studies**:

```bash
make train-ablation
# Or with custom config:
bash scripts/train_ablation.sh --config custom
```

**Evaluation**:

```bash
# Evaluate latest experiment
make eval-latest

# Evaluate specific experiment
bash scripts/eval_latest.sh --experiment exp-name

# View metrics
make view-metrics

# List all experiments
make list-experiments
```

### Configuration

HyperMem defaults match the paper (see `docs/CONFIGURATION.md#paper-defaults`):

- k^T = 15, k^E = 25, k^F = 30 (retrieval top-ks)
- λ = 0.5 (node embedding update weight)
- Reranker enabled
- BM25 + dense retrieval with RRF (k=60)
- Sum aggregation for hyperedge embeddings

Customize via `.env.local`:

```bash
HYPERMEM_EXPERIMENT_NAME=my-exp
HYPERMEM_TOPIC_TOP_K=15                 # Override (default: from code)
HYPERMEM_EPISODE_TOP_K=25
HYPERMEM_FACT_TOP_K=30
HYPERMEM_USE_RERANKER=true
OPENAI_API_KEY=sk-...
```

See `.env.example` for all options.

### Common Tasks

#### Adding a configuration parameter

1. Add to `hypermem/config/constants.py` (if a default constant) or `hypermem/config/__init__.py` (if runtime-configurable)
2. Document in `.env.example`
3. Use in code: `from hypermem.config import ExperimentConfig; config = ExperimentConfig()`

#### Adding an extraction module

1. Create `hypermem/extractors/my_extractor.py` with type hints
2. Write tests in `tests/test_my_extractor.py` (>80% coverage)
3. Integrate into the pipeline in `hypermem/main/stage*.py` if needed

#### Running stages manually

```bash
# Individual stages
python hypermem/main/stage1_memory_extraction.py
python hypermem/main/stage2_hypergraph_extraction.py
# ... stages 3–6

# Or via orchestrator
python hypermem/main/eval.py --start 1 --end 6
```

### Results & Metrics

Experiment results are saved to `results/<experiment_name>/`:

| File                             | Contents                       |
| -------------------------------- | ------------------------------ |
| `judged.json`                    | LLM-as-judge evaluation scores |
| `responses.json`                 | Generated answers              |
| `retrieval_logs.json`            | Retrieval pipeline logs        |
| `search_results.json`            | Search/ranking results         |
| `episodes/`, `topics/`, `facts/` | Extracted hypergraph nodes     |

View results:

```bash
make view-metrics
cat results/exp-name/judged.json | jq '.'
```

### Debugging

**Enable debug logging**:

```bash
export LOG_LEVEL=DEBUG
bash scripts/train_baseline.sh
```

**Dry-run mode** (preview without execution):

```bash
bash scripts/train_baseline.sh --dry-run
```

**Local model serving** (for self-hosted embedding/reranking):

```bash
# Terminal 1: Embedding server
bash scripts/serve_embedding.sh

# Terminal 2: Reranker server
bash scripts/serve_reranker.sh

# Terminal 3: Point to local servers
export EMBEDDING_BASE_URL=http://localhost:11810/v1
export RERANKER_BASE_URL=http://localhost:12810
bash scripts/run_eval.sh
```

### Troubleshooting

**Missing dependencies**:

```bash
pip install -e '.[hypermem]'
```

**API key errors**:

```bash
# Add to .env.local
echo "OPENAI_API_KEY=sk-..." >> .env.local
echo "OPENROUTER_API_KEY=sk-..." >> .env.local  # if using OpenRouter
```

**Tests failing**:

1. Check error message carefully
2. Run with verbose output: `pytest -vv tests/`
3. Confirm dependencies: `pip install -e '.[hypermem]'`
4. Try isolating: `pytest tests/test_config.py::test_retrieval_top_k_defaults_match_paper -v`

---

## Review Process

1. **Automated checks** (CI): Tests, linting, type checking must pass
2. **Code review**: At least one maintainer approval required
3. **Approval**: Resolve all review comments
4. **Merge**: Squash merge to main (clean history)

### Addressing review feedback

Push new commits to your feature branch; they'll be squashed on merge:

```bash
git add your-changes/
git commit -m "fix(scope): address review feedback"
git push origin feature/your-feature-name
```

---

## Citation

If you use HyperMem in your research, please cite:

```bibtex
@inproceedings{yue2026hypermem,
  title     = {HyperMem: Hypergraph Memory for Long-Term Conversations},
  author    = {Yue, Juwei and Hu, Chuanrui and Sheng, Jiawei and Zhou, Zuyi and Zhang, Wenyuan and Liu, Tingwen and Guo, Li and Deng, Yafeng},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```

---

Thank you for contributing to C-AIMMS!
