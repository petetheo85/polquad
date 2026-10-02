# POLQUAD: Computational Processing of Political Bias Using AI Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/petetheo85/polquad/actions/workflows/test.yml/badge.svg)](https://github.com/petetheo85/polquad/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

```text
 ███████████     ███████    █████          ██████    █████  █████   █████████   ██████████  
▒▒███▒▒▒▒▒███  ███▒▒▒▒▒███ ▒▒███         ███▒▒▒▒███ ▒▒███  ▒▒███   ███▒▒▒▒▒███ ▒▒███▒▒▒▒███ 
 ▒███    ▒███ ███     ▒▒███ ▒███        ███    ▒▒███ ▒███   ▒███  ▒███    ▒███  ▒███   ▒▒███
 ▒██████████ ▒███      ▒███ ▒███       ▒███     ▒███ ▒███   ▒███  ▒███████████  ▒███    ▒███
 ▒███▒▒▒▒▒▒  ▒███      ▒███ ▒███       ▒███   ██▒███ ▒███   ▒███  ▒███▒▒▒▒▒███  ▒███    ▒███
 ▒███        ▒▒███     ███  ▒███      █▒▒███ ▒▒████  ▒███   ▒███  ▒███    ▒███  ▒███    ███ 
 █████        ▒▒▒███████▒   ███████████ ▒▒▒██████▒██ ▒▒████████   █████   █████ ██████████  
▒▒▒▒▒           ▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒ ▒▒   ▒▒▒▒▒▒▒▒   ▒▒▒▒▒   ▒▒▒▒▒ ▒▒▒▒▒▒▒▒▒▒   

                               Political Quadrant Analysis Framework
                                               v1.0.0
====================================================================================================
```

This repository contains the code and evaluation datasets for the paper:
> **"Computational Processing of Political Bias Using AI Agents"**  
> Peter Theodoracopoulos, Manoj P. Vijayakumar, and Vijay K. Madisetti (Georgia Institute of Technology)  
> *Accepted for publication in the Journal of Information Security (2026).*

POLQUAD is a system for detecting and neutralizing political bias in natural language. Rather than treating bias as a simple one-dimensional (Left/Right) label, it models statements across a continuous two-dimensional political compass (Economic Left/Right $\times$ Social Authoritarian/Libertarian). 

The system uses four persona agents representing each quadrant (`Lib-Left`, `Lib-Right`, `Auth-Left`, `Auth-Right`) and a neutral judge agent that synthesizes their perspectives into balanced text, moving the statement's coordinates toward $(0, 0)$ while retaining the original context.

## How It Works

1. **Bias Scoring**: An LLM maps input text onto continuous coordinates $[-10, +10] \times [-10, +10]$ using JSON schema extraction. To account for foundation model bias, POLQUAD measures coordinate drift on neutral anchor statements first and subtracts this baseline.
2. **Perspective Generation**: If bias magnitude exceeds the threshold ($\tau = 1.0$), four agents generate viewpoints grounded in their respective quadrants.
3. **Judge Synthesis**: A neutral judge prompt takes the original text and the four quadrant viewpoints to produce a balanced rewrite.
4. **Iterative Refinement**: The rewrite's bias coordinates are recalculated. If the Euclidean magnitude is still above threshold, the feedback loop repeats (up to a default limit of 3 iterations).

```text
                              ┌─────────────────────────┐
                              │     Biased Statement    │
                              └────────────┬────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │    Bias Calculator (2D Compass LLM)     │
                      │  - Calibrated with baseline drift       │
                      │  - Multi-pass stochastic smoothing      │
                      └────────────────────┬────────────────────┘
                                           │
                         Is Magnitude (||v||) > Threshold?
                                           │
                         ┌─────────────────┴─────────────────┐
                      YES│                                   │NO
                         ▼                                   ▼
          ┌─────────────────────────────┐           [Return Statement]
          │      Four Persona Agents    │           (Already Centered)
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
          │     Neutral Judge Agent     │ ◄─────────────────────────┐
          │    (Synthesizes Balance)    │                           │
          └──────────────┬──────────────┘                           │
                         │ Moderated Output Candidate               │
                         ▼                                          │
          ┌─────────────────────────────┐                           │
          │     Convergence Check       │                           │
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

## Example Walkthrough

To see this pipeline in practice, consider an empirical high-bias statement from our benchmark:

**Original statement**:
> *"The southern border is in complete chaos, and the Biden Administration is not lifting a finger to stop what is going to become an invasion of illegal immigrants."*  
> Initial coordinates: $(x = +4.50, y = +6.00)$ | Magnitude: $7.50$ (Authoritarian-Right)

**Quadrant perspectives**:
- **Libertarian-Left**: Emphasizes humanitarian concerns, human rights of migrants, root economic and climate causes of displacement, and safe legal processing over criminalization.
- **Libertarian-Right**: Focuses on labor mobility, questions the fiscal cost and bureaucracy of federal enforcement agencies, and balances enforcement with private property protections.
- **Authoritarian-Left**: Emphasizes orderly state regulation to protect domestic labor from wage suppression and corporate exploitation, alongside investment in border facilities.
- **Authoritarian-Right**: Prioritizes national sovereignty, border security, statutory immigration enforcement, and legal order.

**Synthesized output (Judge)**:
> *"The southern border faces considerable strain, prompting debate regarding the administration's policies on migration management and border security."*  
> Final coordinates: $(x = 0.00, y = 0.00)$ | Magnitude: $0.00$ (100% reduction, converged in 2 iterations)

## Comparative Frameworks & Cost Trade-offs

| Framework | Architecture | Strengths | Trade-offs & Compute Cost |
|---|---|---|---|
| **Full POLQUAD** | 4 Quadrant Agents + 1 Synthesis Judge | Captures cross-cutting ideological tension; highest context retention | ~5×–7× token overhead vs. single-prompt baseline |
| **Unified Agent** | Single prompt with 4-quadrant latent instructions | Balances multi-perspective awareness in one call | Moderate convergence speed; occasional drift; ~1.5× tokens |
| **Naïve Baseline** | Single-prompt direct rewrite | Fast, lightweight (~1× token cost) | Flattens meaning; higher failure rate on high-bias inputs |

**Token and compute cost**: Full POLQUAD uses roughly 5× to 7× more tokens per iteration than the single-prompt baseline (around 1,500–2,200 tokens vs. 250–350 tokens), because it runs four persona prompts, coordinate checks, and a synthesis judge. On heavily biased inputs, this extra compute reduces overcorrection and converges in fewer rounds (1.75–1.84 iterations on average).

## Empirical Results

The framework was evaluated on an uncurated corpus of **79,745 tweets from U.S. Senators** (`m-newhauser/senator-tweets`) and a curated **stress-test dataset of high-bias statements** ($\text{initial bias magnitude} > 7.0$).

### Multi-Model Stress Test on High-Bias Data

The table below shows performance across three foundation model backends:

| LLM Backend | Iterative Naïve (Success Rate) | Unified POLQUAD (Success Rate) | **Full POLQUAD (Success Rate)** | **Full POLQUAD (Avg Bias Reduction)** |
|:---|:---:|:---:|:---:|:---:|
| **Google Gemini 2.5 Flash Lite** ($n=300$) | 62.00% | 69.00% | **80.00%** | **77.75%** |
| **OpenAI GPT-4o-mini** ($n=98$) | 30.61% | 43.88% | **76.53%** | **76.66%** |
| **Anthropic Claude 3.5 Haiku** ($n=100$) | 39.00% | 59.00% | **60.00%** | **76.45%** |

On high-bias statements ($\text{magnitude} > 7.0$), single-prompt rewriting frequently overcorrects—making statements *more* biased in 5.78% of runs on GPT-4o-mini and 4.08% for the unified prompt. Full POLQUAD limits overcorrections to 1.02% while averaging 1.75 iterations to converge. 

The largest improvements over the baseline occur on Gemini and GPT-4o-mini. On Claude 3.5 Haiku, Full POLQUAD and the Unified variant perform similarly (60.00% vs. 59.00% convergence rate, $p = 1.00$).

## Ablation Study: Why Four Agents: Empirical Evidence

To test whether all four quadrant agents are necessary or whether a simpler setup works, we evaluated 10 ablation configurations across single-agent, axial, and diagonal permutations ($n=300$):

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

The ablation shows that the four-agent requirement is practical rather than a formal proof:
- **Partial setups can still perform well**: An Authoritarian-Left agent alone (`AL Only`) achieves a 73.77% bias reduction, close to the full system's 77.75%.
- **Blind spots with missing quadrants**: Configurations that omit an entire axis or diagonal leave the judge vulnerable to opposing statements. For instance, the Right-only ablation (`AR + LR`) drops to a 40.6% reduction when tested against Libertarian-Left statements because the judge lacks counterbalancing input.
- **Efficiency**: The complete system converges faster (1.84 iterations average vs. 2.21–2.35 for single-agent setups) and provides symmetric pull toward the center.

## Evaluation & Methodology

1. **2D Coordinate Vector Representation**:
   Every text statement $S$ is mapped to a vector $\vec{b} = (x, y)$ where $x \in [-10, 10]$ represents Economic Bias (Left/Right) and $y \in [-10, 10]$ represents Social Bias (Authoritarian/Libertarian).
2. **Euclidean Bias Magnitude**:
   $$
   \|\vec{b}\| = \sqrt{x^2 + y^2}
   $$
3. **Calibrated Baseline Adjustment**:
   $$
   \vec{b}_{\text{calibrated}} = \vec{b}_{\text{measured}} - \vec{b}_{\text{baseline}}
   $$
4. **Bias Reduction Percentage**:
   $$
   \Delta_{\text{bias}} = \frac{\|\vec{b}_{\text{initial}}\| - \|\vec{b}_{\text{final}}\|}{\|\vec{b}_{\text{initial}}\|} \times 100\%
   $$

## Limitations & Methodological Considerations

**Shared model families**: The scoring LLM and the generation agents come from the same model families, so measurement and rewriting are not fully independent.
**No human ground truth**: Evaluation relies on continuous coordinate estimates, embeddings, and automated blinded checks rather than human annotator labels.
**Political compass assumptions**: The 2D compass is a simplified model, and setting the target centroid at $(0, 0)$ is an operational design choice rather than an absolute definition of neutrality.
**Sample size differences**: Sample sizes vary across providers ($n=300$ for Gemini, $n=98$ for GPT-4o-mini, $n=100$ for Claude). As reported in the paired tests:
   - On **Gemini** ($n=300$), Full POLQUAD outperforms Naïve by $+18.00\%$ convergence ($95\%\text{ CI: } [+12.52\%, +23.48\%], p < 10^{-9}$) and Unified by $+11.00\%$ ($95\%\text{ CI: } [+5.72\%, +16.28\%], p < 10^{-4}$).
   - On **GPT-4o-mini** ($n=98$), Full POLQUAD outperforms Naïve by $+45.92\%$ convergence ($95\%\text{ CI: } [+33.86\%, +57.97\%], p < 10^{-9}$) and Unified by $+32.65\%$ ($95\%\text{ CI: } [+21.07\%, +44.24\%], p < 10^{-6}$).
   - On **Claude** ($n=100$), Full POLQUAD outperforms Naïve by $+21.00\%$ convergence ($95\%\text{ CI: } [+9.47\%, +32.53\%], p = 0.0011$), but is essentially tied with Unified POLQUAD ($+1.00\%$ difference, $95\%\text{ CI: } [-8.40\%, +10.40\%], p = 1.00$, not statistically significant).

## Data & Model Provenance

- **Model Versions**: Evaluations used `gemini-2.5-flash-lite`, `gpt-4o-mini`, and `claude-3-5-haiku-20241022`.
- **Dataset Licensing & Usage**:
  - The Senator tweets corpus (`m-newhauser/senator-tweets`) is hosted on Hugging Face without an explicit open-source license. The data consists of public social media posts from U.S. Senators and is utilized under research fair-use conventions.
  - The curated high-bias stress-test dataset is distributed under the project's [MIT License](LICENSE).

## Repository Structure

```text
POLQUAD/
├── .github/workflows/         # CI test workflow
├── src/polquad/               # Core framework package
│   ├── agents/                # Persona & judge agents
│   ├── frameworks/            # Moderation loops (Full Polquad, Unified, Naive)
│   ├── utils/                 # LLM clients, BiasCalculator, retry logic
│   └── config.py              # Hyperparameters and runtime config
├── scripts/                   # Evaluation & artifact scripts
│   ├── run_ablations.py       # Quadrant ablation experiments
│   ├── run_naive_only.py      # Zero-shot baseline runner
│   └── generate_paper_artifacts.py  # Plots and figures
├── analysis/                  # Analysis tooling
│   ├── datasets/              # Dataset extraction and filtering
│   ├── evaluations/           # Aggregate metrics
│   └── visualization/         # Performance maps
├── outputs/                   # Experiment artifacts (figures, reports, run logs)
├── tests/                     # Unit and integration test suite
├── main.py                    # Benchmark runner
└── pyproject.toml             # Package metadata and dependencies
```

## Quickstart

### Prerequisites
- Python 3.10+
- An API key for Google Gemini, OpenAI, or Anthropic

### 1. Installation
```bash
git clone https://github.com/petetheo85/polquad.git
cd polquad

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Configure Environment
Set provider API keys:
```bash
export GOOGLE_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"       # Optional
export ANTHROPIC_API_KEY="your-anthropic-key" # Optional
```

Key hyperparameters can be adjusted in `src/polquad/config.py`:
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
```bash
# Run unit test suite (offline / mock enabled)
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

## Citation

```bibtex
@article{theodoracopoulos2026computational,
  title={Computational Processing of Political Bias Using AI Agents},
  author={Theodoracopoulos, Peter and Vijayakumar, Manoj P. and Madisetti, Vijay K.},
  journal={Journal of Information Security},
  year={2026},
  note={Accepted for publication},
  institution={Georgia Institute of Technology}
}
```
