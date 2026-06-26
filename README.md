# Redrob Hackathon: Team InfiniCode Ranking System

This repository contains the code for Team InfiniCode's candidate ranking system.

## How to Run

```bash
pip install -r requirements.txt
python download_model.py
python rank.py --candidates /path/to/candidates.jsonl --out ./InfiniCode.csv
```

For CPU-only PyTorch (saves ~2GB download):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Methodology

Our system uses a 2-pass pipeline with an explicitly weighted scoring formula:

1. **Semantic Match (35%)**: Cosine similarity between the Job Description and the candidate's summary/experience using `BAAI/bge-small-en-v1.5`. Batch-encoded for speed.
2. **Career Relevance (25%)**: Extraction of vector DB and ML keywords from career descriptions, with penalties for purely academic or consulting backgrounds.
3. **Behavioral Signals (20%)**: Recruiter response rate, GitHub activity, and interview completion rate.
4. **Recency (10%)**: Continuous exponential decay based on last active date.
5. **Honeypot Penalty (-10%)**: Proportional multiplier applied to the final score for ghosting risk and resume padding.

## Generating the Presentation Deck

```bash
python generate_deck.py
```
