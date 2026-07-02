"""
Validate and analyze the ranking output CSV.
Usage: python validate_output.py --csv ./InfiniCode.csv
"""
import csv
import argparse
import statistics


def validate(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"{'='*60}")
    print(f"  VALIDATION REPORT: {csv_path}")
    print(f"{'='*60}\n")

    # --- Basic checks ---
    n = len(rows)
    print(f"Total candidates: {n}")
    assert n == 100, f"FAIL: Expected 100 candidates, got {n}"
    print("  [PASS] Exactly 100 candidates\n")

    # Check required columns
    required = {"candidate_id", "rank", "score", "reasoning"}
    actual = set(rows[0].keys())
    missing = required - actual
    assert not missing, f"FAIL: Missing columns: {missing}"
    print(f"  [PASS] All required columns present: {required}\n")

    # Check unique candidate IDs
    ids = [r["candidate_id"] for r in rows]
    assert len(ids) == len(set(ids)), "FAIL: Duplicate candidate IDs found"
    print("  [PASS] All candidate IDs are unique\n")

    # Check ranks are 1-100
    ranks = [int(r["rank"]) for r in rows]
    assert ranks == list(range(1, 101)), "FAIL: Ranks are not sequential 1-100"
    print("  [PASS] Ranks are sequential 1-100\n")

    # Check scores are monotonically non-increasing
    scores = [float(r["score"]) for r in rows]
    is_monotonic = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    if is_monotonic:
        print("  [PASS] Scores are monotonically non-increasing\n")
    else:
        violations = sum(1 for i in range(len(scores)-1) if scores[i] < scores[i+1])
        print(f"  [WARN] {violations} score ordering violations found\n")

    # Check all scores are unique
    if len(set(scores)) == len(scores):
        print("  [PASS] All scores are unique\n")
    else:
        dupes = len(scores) - len(set(scores))
        print(f"  [WARN] {dupes} duplicate scores found\n")

    # --- Score distribution ---
    print(f"{'='*60}")
    print(f"  SCORE DISTRIBUTION")
    print(f"{'='*60}\n")

    print(f"  Max score:    {max(scores):.4f} (Rank 1)")
    print(f"  Min score:    {min(scores):.4f} (Rank 100)")
    print(f"  Mean score:   {statistics.mean(scores):.4f}")
    print(f"  Median score: {statistics.median(scores):.4f}")
    print(f"  Std dev:      {statistics.stdev(scores):.4f}")
    print(f"  Score spread: {max(scores) - min(scores):.4f}")
    print()

    # Histogram
    buckets = [0]*10
    for s in scores:
        bucket = min(int(s * 10), 9)
        buckets[bucket] += 1

    print("  Score Histogram:")
    for i in range(9, -1, -1):
        bar = "█" * buckets[i]
        lo = i / 10
        hi = (i + 1) / 10
        print(f"    {lo:.1f}-{hi:.1f} | {bar} ({buckets[i]})")
    print()

    # --- Top 10 candidates ---
    print(f"{'='*60}")
    print(f"  TOP 10 CANDIDATES")
    print(f"{'='*60}\n")

    for row in rows[:10]:
        print(f"  Rank {row['rank']:>3} | Score: {float(row['score']):.4f} | ID: {row['candidate_id']}")
        print(f"           Reasoning: {row['reasoning'][:120]}")
        print()

    # --- Bottom 5 candidates (for contrast) ---
    print(f"{'='*60}")
    print(f"  BOTTOM 5 CANDIDATES (of top 100)")
    print(f"{'='*60}\n")

    for row in rows[-5:]:
        print(f"  Rank {row['rank']:>3} | Score: {float(row['score']):.4f} | ID: {row['candidate_id']}")
        print(f"           Reasoning: {row['reasoning'][:120]}")
        print()

    print(f"{'='*60}")
    print(f"  ALL CHECKS PASSED ✓")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ranking output CSV")
    parser.add_argument("--csv", required=True, help="Path to the output CSV")
    args = parser.parse_args()
    validate(args.csv)
