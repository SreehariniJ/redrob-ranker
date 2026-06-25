# Redrob Hackathon: Team InfiniCode Ranking System

This repository contains the code for Team InfiniCode's candidate ranking system.

## How to run in 60 seconds
If you just cloned this repo, you can generate the rankings immediately:
```bash
pip install -r requirements.txt
python download_model.py
python rank.py --candidates /path/to/candidates.jsonl --out ./InfiniCode.csv
```
This complete pipeline downloads the local embedding model and streams 100K candidates through a 2-pass semantic ranker in ~1 minute on a CPU-only 16GB machine.

## Setup Instructions

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the offline pre-computation step (downloads the `all-MiniLM-L6-v2` model):
```bash
python download_model.py
```

## Running the Ranker

To generate the ranking CSV, run the following command. The command will complete within 5 minutes on a standard CPU with 16GB RAM.

```bash
python rank.py --candidates /path/to/candidates.jsonl --out ./InfiniCode.csv
```

## Methodology

Our system uses an explicit weighted scoring formula:
1. **Semantic Match (40%)**: Cosine similarity between the Job Description and the candidate's summary/experience using the local `all-MiniLM-L6-v2` embedding model.
2. **Rule-Based Heuristics (30%)**: Extraction of vector DB and modern ML keywords from career descriptions, applying penalties for purely academic or consulting backgrounds.
3. **Behavioral Signals (30%)**: Scaled engagement metrics based on recruiter response rate, GitHub activity, and recent platform logins.

## Generating the Presentation Deck

We programmatically generate our presentation deck using ReportLab:
```bash
python generate_deck.py
```
