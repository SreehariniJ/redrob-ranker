# 🏆 Redrob Hackathon: Team InfiniCode — Intelligent Candidate Ranking

An AI-powered candidate ranking system that goes beyond keyword matching to understand who truly fits a role. Built for the Redrob Hackathon: *Intelligent Candidate Discovery & Ranking*.

## 🎯 What This Does

Given ~100K candidate profiles and a job description, the system:
1. **Reads and understands** the JD semantically — not just keyword extraction.
2. **Evaluates the full picture** — career history, skills, behavioral signals, and platform activity.
3. **Detects fraud** — filters honeypot profiles and penalizes resume padding.
4. **Delivers a ranked shortlist** — top 100 candidates with per-candidate reasoning.

## 🏗️ Architecture

```
candidates.jsonl ──► Pass 1: Heuristic Filter (all 100K) ──► Pass 2: Semantic Ranking (top 5K) ──► InfiniCode.csv
   (~100K)             │                                          │                                    (top 100)
                       ├── Red flag / honeypot detection           ├── Encode JD with BGE-small
                       ├── Career relevance scoring                ├── Batch-encode candidate texts
                       ├── Behavioral signal scoring               ├── Cosine similarity (top-2 avg)
                       └── Recency decay                           └── Weighted final formula
```

### Why 2-Pass?

- **Pass 1** streams 100K candidates with O(1) memory per candidate using fast heuristics. No model inference needed — runs in seconds.
- **Pass 2** uses `BAAI/bge-small-en-v1.5` (33M params, CPU-friendly) for deep semantic understanding on only the top 5,000 candidates.
- Total runtime: **~1 minute** on CPU. Well within the 5-minute budget.

## 📊 Scoring Formula

```
penalty_multiplier = 1.0 - (0.10 × honeypot_penalty)

final_score = penalty_multiplier × (
    0.35 × semantic_similarity     ← BGE cosine sim (JD vs candidate text)
  + 0.25 × career_relevance        ← Vector DB/ML keywords, seniority, depth
  + 0.20 × behavioral_signals      ← Recruiter response, GitHub, interviews
  + 0.10 × recency                 ← Exponential decay from last active date
)
```

### Why These Weights?

| Signal | Weight | Rationale |
|---|---|---|
| **Semantic Match** | 35% | The JD explicitly asks for *production ML systems* experience. Cosine similarity captures this far better than keywords. |
| **Career Relevance** | 25% | The JD says "not pure consulting, not pure research." Career history reveals this — skills lists don't. |
| **Behavioral Signals** | 20% | A great candidate who doesn't respond to recruiters or complete interviews isn't useful. |
| **Recency** | 10% | Recent activity suggests availability. But it's weighted low — it's a signal, not a requirement. |
| **Honeypot Penalty** | 10% | Proportional multiplier (not subtraction) so penalized candidates maintain relative ordering. |

## 🚨 Honeypot / Fraud Detection

| Type | Condition | Action |
|---|---|---|
| **Fake Profile** | 10+ YoE but zero GitHub AND zero recruiter response | **Drop entirely** |
| **Fake Expert** | "Expert" proficiency with 0 months duration | **Drop entirely** |
| **Ghosting Risk** | Inactive > 18 months | Penalize (-0.5) |
| **Resume Padding** | Lists a core JD skill but never mentions it in career descriptions | Penalize (-0.2 per skill, capped) |

## 🔧 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pre-download embedding model (requires network — run once)
python download_model.py

# 3. Run ranking (fully offline, <5 min)
python rank.py --candidates /path/to/candidates.jsonl --out ./InfiniCode.csv

# 4. Validate output
python validate_output.py --csv ./InfiniCode.csv
```

For CPU-only PyTorch (saves ~2GB download):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 📁 Project Structure

| File | Purpose |
|---|---|
| `rank.py` | Core ranking pipeline — the main deliverable |
| `download_model.py` | Downloads `BAAI/bge-small-en-v1.5` to `./models/` for offline use |
| `validate_output.py` | Validates output CSV — checks format, score monotonicity, and prints score distribution |
| `job_description.docx` | Source JD file — read dynamically at runtime |
| `submission_metadata.yaml` | Team info, compute specs, methodology summary |
| `requirements.txt` | Python dependencies |

## ⚙️ Compute Constraints Met

| Constraint | Limit | Actual |
|---|---|---|
| GPU | None | ✅ CPU-only |
| RAM | 16 GB | ✅ < 1 GB peak |
| Time | 5 min | ✅ ~1 min |
| Network during ranking | None | ✅ Model pre-downloaded |

## 👥 Team InfiniCode

| Name | Role |
|---|---|
| SREEHARINI J | Team Leader |
| SREEKUMAR M | Team Member |
| Sreya S | Team Member |
| sreejisha m gupta | Team Member |
