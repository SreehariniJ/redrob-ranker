# AGENTS.md — Redrob Hackathon: Candidate Ranking System

## Project Objective

Build an AI recruiter that **intelligently ranks** (not just filters) candidates for a Senior AI Engineer — Founding Team role. The system must demonstrate:

1. **Deep Job Understanding**: Interpret nuanced job descriptions, not just keyword match.
2. **Contextual Relevance**: Understand semantic fit — see beyond keywords to actual experience.
3. **Signal Integration**: Leverage all available data — profile attributes, career metadata, behavioral/activity signals.
4. **Output**: Lightning-fast, highly accurate, ranked shortlist of best-fit candidates.

## What We're Ranking

- **Input**: `candidates.jsonl` — ~100K candidate profiles with career history, skills, and Redrob behavioral signals.
- **Output**: Top 100 candidates as CSV (`candidate_id, rank, score, reasoning`).

## Compute Constraints

- **CPU-only** — no GPU available during ranking (`uses_gpu_for_inference: false`).
- **16GB RAM**, 8 cores.
- **No network** during ranking (`has_network_during_ranking: false`) — model must be pre-downloaded.
- **5-minute total budget** for ranking, including model load.
- Pre-computation step (model download) is allowed separately.

## Architecture: 2-Pass Pipeline

### Pass 1 — Fast Heuristic Filtering (all 100K candidates)
Stream JSONL line-by-line. For each candidate, compute:
- **Red flag / honeypot detection** — filter severe fakes (fake profiles, fake experts).
- **Career relevance** — vector DB keywords, ML keywords, seniority, consulting/research penalties.
- **Behavioral score** — recruiter response rate, GitHub activity, interview completion.
- **Recency** — continuous exponential decay from last active date.

Select top 2000 using `heapq.nsmallest`.

### Pass 2 — Batched Semantic Ranking (top 2000 only)
- Encode JD once with `BAAI/bge-small-en-v1.5` (with query instruction prefix).
- Batch-encode all candidate texts (profile summary + career descriptions) in one call.
- Compute cosine similarity (JD vs each candidate text), average top-2 entries.
- Combine with Pass 1 signals using weighted formula with proportional penalty.
- Use `heapq.nsmallest` for top-100 selection, write CSV.

## Scoring Formula (from `rank.py`)

```
penalty_multiplier = 1.0 - (0.10 * honeypot_penalty)

final_score = penalty_multiplier * (
    0.35 * semantic
  + 0.25 * career_relevance
  + 0.20 * behavioral
  + 0.10 * recency
)
```

## Key Files

| File | Purpose |
|---|---|
| `rank.py` | Core ranking pipeline — the main deliverable |
| `download_model.py` | Downloads `BAAI/bge-small-en-v1.5` to `./models/` |
| `generate_deck.py` | Generates presentation PDF via ReportLab |
| `fill_pptx.py` | Fills PPTX template with project details |
| `submission_metadata.yaml` | Team info, compute specs, declarations |

## Red Flag Detection Logic

### Severe (candidate dropped entirely):
- 10+ YoE but zero GitHub activity AND zero recruiter response rate.
- "Expert" proficiency with 0 months duration.

### Penalized (score reduced, candidate kept):
- Ghosting risk: inactive for >18 months.
- Resume padding: lists a core JD skill but never mentions it in career descriptions (fuzzy matching via `CORE_JD_SKILLS` variants).

## Running the Pipeline

```bash
pip install -r requirements.txt
python download_model.py
python rank.py --candidates /path/to/candidates.jsonl --out ./InfiniCode.csv
```

## Dependencies

- `sentence-transformers` — embedding model
- `torch` — tensor backend (CPU-only)
- `reportlab` — PDF generation
- `python-pptx` — PPTX template filling
