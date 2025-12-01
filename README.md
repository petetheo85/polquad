# POLQUAD: Multi-Agent Political Neutralization & Cognitive Guardrail Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests: Passing](https://img.shields.io/badge/tests-11%20passed-brightgreen.svg)]()
[![Architecture: Multi-Agent System](https://img.shields.io/badge/Architecture-Multi--Agent%20System-orange.svg)]()
[![Guardrails: Bias Mitigation](https://img.shields.io/badge/Guardrails-Bias%20Mitigation-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**POLQUAD** is a research-grade, multi-agent AI system designed to detect, calibrate, and neutralize socio-political bias in natural language. Developed as research submitted to *IEEE Transactions on Computational Social Systems (TCSS)*, POLQUAD moves beyond oversimplified one-dimensional (Left/Right) classification by framing political discourse across a **continuous 2-dimensional coordinate space** (Economic Left/Right $\times$ Social Authoritarian/Libertarian). 

Using an adversarial multi-agent debate and consensus synthesis architecture, POLQUAD guides biased text toward an objective, apolitical centroid $(0, 0)$ without flattening semantic nuance or stripping core context.

---

## Key Architectural Highlights

- **Multi-Agent Dialectical Synthesis**: Orchestrates four specialized persona agents representing the quadrants of the political compass (`Lib-Left`, `Lib-Right`, `Auth-Left`, `Auth-Right`) and an apolitical **Judge Agent** that synthesizes their ideological disagreements into balanced text.
- **Continuous 2D Coordinate Calibration**: Projects textual latent bias onto a continuous $[-10.0, +10.0] \times [-10.0, +10.0]$ political compass using structured JSON schema extraction.
- **Self-Calibrated Baseline Correction**: Measures the foundation model's intrinsic coordinate drift using neutral anchor statements before evaluating target text, subtracting systematic slant.
- **Iterative Feedback Guardrails**: Measures post-moderation Euclidean bias magnitude and prompts corrective re-synthesis until the statement falls below a target convergence threshold ($\tau$).
- **Multi-LLM Interoperability**: Decoupled client abstractions with automatic exponential backoff and rate-limit retry handling for **Google Gemini** (`gemini-2.5-flash-lite`), **OpenAI** (`gpt-4o-mini`), and **Anthropic Claude** (`claude-3-5-haiku-20241022`).
- **Resilient on Highly Polarized Data**: Demonstrates superior resilience on curated high-bias statements ($\text{magnitude} > 7.0$) where single-prompt baseline models collapse.

---

## System Architecture

POLQUAD operates on the principle that true neutrality is achieved not by superficial word-scrubbing, but by explicitly modeling opposing ideological dimensions and synthesizing their core semantic values into a balanced consensus.

```text
                              ┌─────────────────────────┐
                              │     Biased Statement    │
                              └────────────┬────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │    Bias Calculator (2D Compass LLM)     │
                      │  - Calibrated with baseline drift       │
                      │  - Multi-run stochastic smoothing       │
                      └────────────────────┬────────────────────┘
                                           │
                         Is Magnitude (||v||) > Threshold?
                                           │
                         ┌─────────────────┴─────────────────┐
                      YES│                                   │NO
                         ▼                                   ▼
          ┌─────────────────────────────┐           [Return Statement]
          │   Dialectical Agent Tier    │           (Already Centered)
          │  ┌───────────────────────┐  │
          │  │ Lib-Left Opinion      │  │
          │  ├───────────────────────┤  │
          │  │ Lib-Right Opinion     │  │
          │  ├───────────────────────┤  │
          │  │ Auth-Left Opinion     │  │
          │  ├───────────────────────┤  │
          │  │ Auth-Right Opinion    │  │
          │  └───────────────────────┘  │
          └──────────────┬──────────────┘
                         │ Structured Opinions & 2D Vector Context
                         ▼
          ┌─────────────────────────────┐
          │   Apolitical Judge Agent    │ ◄─────────────────────────┐
          │  (Synthesizes Consensus)    │                           │
          └──────────────┬──────────────┘                           │
                         │ Moderated Output Candidate               │
                         ▼                                          │
          ┌─────────────────────────────┐                           │
          │    Convergence Guardrail    │                           │
          │  - Recalculate Bias Vector  │                           │
          │  - Measure Euclidean Dist   │                           │
          └──────────────┬──────────────┘                           │
                         │                                          │
                    Converged?                                      │
                   /          \                                     │
             YES  /            \  NO (Iter < Max)                   │
                 ▼              └───────────────────────────────────┘
        [Balanced Text]              Refinement Loop w/ Error History
```

---

## Comparative Frameworks

| Framework | Architecture | Strengths | Trade-offs |
|---|---|---|---|
| **Full POLQUAD** | 4 Quadrant Agents + 1 Synthesis Judge | Captures nuanced multi-axis tension; highest contextual preservation | Multi-agent token overhead |
| **Unified Agent** | Single prompt with 4-quadrant latent instructions | Balances multi-perspective awareness in a single LLM invocation | Moderate convergence speed; occasional ideological drift |
| **Naïve Baseline** | Single-prompt direct rewrite | Fast, lightweight | Flattens semantic meaning; susceptible to collapse under high bias |

---

## Empirical Results & Comparative Evaluation

The framework was evaluated on an uncurated corpus of **79,745 tweets from U.S. Senators** (`m-newhauser/senator-tweets`) and a curated **stress-test dataset of high-bias statements** ($\text{initial bias magnitude} > 7.0$).

### Multi-Model Stress Test on High-Bias Data
On highly polarized content, baseline iterative and single-agent models suffer severe performance degradation. Full POLQUAD's *synthesis-then-refine* architecture maintains robust, consistent bias reduction across all three major foundation model families:

| LLM Backend | Iterative Naïve (Success Rate) | Unified POLQUAD (Success Rate) | **Full POLQUAD (Success Rate)** | **Full POLQUAD (Avg Bias Reduction)** |
|:---|:---:|:---:|:---:|:---:|
| **Google Gemini 2.5 Flash Lite** ($n=300$) | 62.00% | 69.00% | **80.00%** | **77.75%** |
| **OpenAI GPT-4o-mini** ($n=98$) | 30.61% | 43.88% | **76.53%** | **76.66%** |
| **Anthropic Claude 3.5 Haiku** ($n=100$) | 39.00% | 59.00% | **60.00%** | **76.45%** |

> **Key Finding — Prevention of Overcorrection**: Under the OpenAI model, Naïve and Unified models exhibited high overcorrection rates (making statements *more* biased in 5.78% and 4.08% of runs, respectively). Full POLQUAD suppressed overcorrection to just **1.02%** while requiring only **1.75 iterations per successful convergence**.

---

## Ablation Study: The Necessity of Four Ideological Agents

To determine whether four quadrant agents are truly necessary or if the architecture is over-engineered, we evaluated **10 distinct ablation configurations** across single-agent, axial, and diagonal permutations ($n=300$):

| Configuration | Description | Avg. Bias Reduction (%) | Success Rate (%) | Avg. Iterations |
|:---|:---|:---:|:---:|:---:|
| **Full POLQUAD** | **Complete 4-Quadrant Architecture** | **77.75%** | **80.00%** | **1.84** |
| **AL + AR** | Authoritarian Axis Only | 74.01% | 76.00% | 1.94 |
| **AL Only** | Single Authoritarian-Left Agent | 73.77% | 77.00% | 1.91 |
| **AL + LR** | Diagonal Counterbalance | 73.17% | 78.33% | 2.06 |
| **AR + LL** | Diagonal Counterbalance | 72.02% | 76.00% | 2.03 |
| **AL + LL** | Left Axis Only | 71.94% | 78.33% | 2.00 |
| **Unified POLQUAD** | Single Prompt / Implicit 4-Quadrant | 71.59% | 69.00% | 1.96 |
| **LL Only** | Single Libertarian-Left Agent | 70.72% | 75.67% | 2.03 |
| **AR Only** | Single Authoritarian-Right Agent | 69.98% | 75.33% | 2.27 |
| **LL + LR** | Libertarian Axis Only | 69.22% | 74.00% | 2.10 |
| **LR Only** | Single Libertarian-Right Agent | 68.48% | 71.67% | 2.21 |
| **AR + LR** | Right Axis Only | 66.68% | 68.33% | 2.35 |

### Why Four Agents are Mathematically Necessary:
1. **Systematic Degradation**: Removing *any* single agent or pair caused an immediate loss in both bias reduction power and convergence efficiency.
2. **Ideological Blind Spots**: Removing counterbalancing agents creates catastrophic vulnerability to opposing viewpoints. For example, the Right-only ablation (`AR + LR`) collapsed to **40.6% reduction** when evaluating Libertarian-Left statements because the Judge lacked an internal representation to formulate a counterweight.
3. **Synthesis Scaffold**: Extreme political statements often anchor in a single quadrant; having three distinct opposing viewpoints exerts a stabilizing inward pull toward the apolitical centroid $(0, 0)$.

---

## Evaluation & Methodology

1. **2D Coordinate Vector Representation**:
   Every text statement $S$ is mapped to a vector $\vec{b} = (x, y)$ where $x \in [-10, 10]$ represents Economic Bias (Left/Right) and $y \in [-10, 10]$ represents Social Bias (Authoritarian/Libertarian).
2. **Euclidean Bias Magnitude**:
   $$\|\vec{b}\| = \sqrt{x^2 + y^2}$$
3. **Calibrated Baseline Adjustment**:
   $$\vec{b}_{\text{calibrated}} = \vec{b}_{\text{measured}} - \vec{b}_{\text{baseline}}$$
4. **Bias Reduction Percentage**:
   $$\Delta_{\text{bias}} = \frac{\|\vec{b}_{\text{initial}}\| - \|\vec{b}_{\text{final}}\|}{\|\vec{b}_{\text{initial}}\|} \times 100\%$$

---

## Repository Structure

```text
POLQUAD/
├── src/polquad/               # Core framework package
│   ├── agents/                # Persona & neutralizer agents (Opinion, Judge, Unified, Naive)
│   ├── frameworks/            # Moderation loops (Full Polquad, Unified, Naive)
│   ├── utils/                 # LLM clients (Gemini, Claude, GPT), BiasCalculator, retry logic
│   └── config.py              # Central hyperparameters and runtime configuration
├── scripts/                   # Standalone execution & paper artifact scripts
│   ├── run_ablations.py       # Quadrant ablation experiments
│   ├── run_naive_only.py      # Zero-shot baseline runner
│   └── generate_paper_artifacts.py  # Publication-grade vector plots and figures
├── analysis/                  # Analytical & statistical tooling
│   ├── datasets/              # HF dataset builder and bias filtering
│   ├── evaluations/           # Aggregate metrics and cross-seed evaluation
│   └── visualization/         # Performance maps and trajectory plots
├── outputs/                   # Consolidated experiment artifacts
│   ├── figures/               # Generated vector plots (.png)
│   ├── reports/               # Aggregate statistical text summaries
│   └── runs/                  # JSON experiment output logs
├── tests/                     # Deterministic, offline unit & integration test suite
├── main.py                    # Main benchmark orchestrator
└── pyproject.toml             # Package metadata and dependencies
```

---

## Quickstart

### Prerequisites
- Python 3.10 or higher
- API key for Google Gemini, OpenAI, or Anthropic

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/POLQUAD.git
cd POLQUAD

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package and dependencies in editable mode
pip install -e .
```

### 2. Configure Environment
Set your provider API keys in your environment:
```bash
export GOOGLE_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"       # Optional
export ANTHROPIC_API_KEY="your-anthropic-key" # Optional
```

Review experiment hyperparameters in `src/polquad/config.py`:
```python
polquad_configs = {
    "max_iters": 3,
    "bias_threshold": 1.0,        # Target Euclidean distance from (0,0)
    "bias_calc_runs": 3,          # Stochastic smoothing passes
    "sample_size": 100,
    "dataset_path": "data/high_bias_dataset.csv",
    "dataset_source_type": "local" # 'local' or 'hf'
}
```

### 3. Execution & Evaluation
Run the test suite and evaluation pipelines:
```bash
# Run automated unit test suite (offline / mock enabled, no API keys needed)
pytest

# Run comparative benchmark across all 3 frameworks
python main.py

# Run ablation studies across quadrant combinations
python scripts/run_ablations.py

# Run naïve baseline only
python scripts/run_naive_only.py

# Generate publication plots and empirical tables
python scripts/generate_paper_artifacts.py
```

---

## Tech Stack & Engineering Standards

- **Agentic AI & Orchestration**: Adversarial multi-agent dialectic, iterative consensus synthesis, cognitive guardrails.
- **LLM Provider Interoperability**: Google GenAI SDK (`gemini-2.5-flash-lite`), OpenAI SDK (`gpt-4o-mini`), Anthropic SDK (`claude-3-5-haiku-20241022`).
- **Production Resilience**: Custom exponential backoff decorators with HTTP 429/529 rate-limit handling and jitter.
- **Evaluation & Analysis**: NumPy, Pandas, Matplotlib, SciPy, Hugging Face Datasets.

---

## Citation & Attribution

If you utilize POLQUAD in your academic research or AI alignment investigations, please cite:

```bibtex
@article{vijayakumar2025computational,
  title={Computational Processing of Political Bias Using AI Agents},
  author={Vijayakumar, Manoj P. and Theodoracopoulos, Peter and Madisetti, Vijay K.},
  journal={Submitted to IEEE Transactions on Computational Social Systems (TCSS)},
  year={2025},
  month={December},
  institution={Georgia Institute of Technology}
}
```
