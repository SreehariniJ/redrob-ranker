import json
import csv
import argparse
import os
import heapq
import math
import re
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

WEIGHTS = {
    "semantic": 0.35,
    "career_relevance": 0.25,
    "behavioral": 0.20,
    "recency": 0.10,
    "honeypot_penalty": 0.10
}

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

JD_TEXT_FALLBACK = """
Senior AI Engineer — Founding Team. We need deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning. 
Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5) deployed to real users.
Production experience with vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS).
Strong Python code quality. Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B test).
Not pure research. Not just LangChain to OpenAI. Not pure consulting firms. Needs to write code.
"""


def load_jd_text():
    """Load JD from job_description.docx if available, otherwise use fallback."""
    docx_path = os.path.join(os.path.dirname(__file__), "job_description.docx")
    if os.path.exists(docx_path):
        try:
            from docx import Document
            doc = Document(docx_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            if text.strip():
                print(f"Loaded JD from {docx_path} ({len(text)} chars)")
                return text
        except ImportError:
            print("python-docx not installed, using fallback JD text.")
        except Exception as e:
            print(f"Error reading {docx_path}: {e}, using fallback JD text.")
    return JD_TEXT_FALLBACK.strip()

CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "deloitte", "mckinsey", "bcg", "bain", "pwc", "kpmg", "ey", "ey parthenon",
    "booz allen", "oliver wyman", "protiviti", "ntt data", "hcl", "tech mahindra",
    "lusabor", "mindtree", "mphasis", "hexaware"
}

CORE_JD_SKILLS = {
    "embedding": ["embed", "embedding", "embeddings"],
    "vector": ["vector", "vector db", "vector database", "vector search", "vector store"],
    "python": ["python"],
    "pytorch": ["pytorch"],
    "pinecone": ["pinecone"],
    "qdrant": ["qdrant"],
    "milvus": ["milvus"],
    "weaviate": ["weaviate"],
    "faiss": ["faiss"],
    "retrieval": ["retrieval", "retrieve", "rag"],
    "ranking": ["ranking", "ranker", "rerank"],
}

TECH_LIST = [
    "PyTorch", "Pinecone", "FAISS", "Qdrant", "Milvus", "Weaviate",
    "OpenSearch", "Elasticsearch", "Kafka", "Spark", "TensorFlow",
    "Sentence-Transformers", "LangChain", "OpenAI", "RAG"
]


def batch_semantic_scores(model, jd_embedding, candidates_data):
    """Batch-encode all candidates at once for speed."""
    all_texts = []
    candidate_text_counts = []

    for summary, career in candidates_data:
        texts = []
        if summary and summary.strip():
            texts.append(summary.strip())
        for job in career:
            desc = job.get("description", "")
            if desc and desc.strip():
                texts.append(desc.strip())
        candidate_text_counts.append(len(texts))
        all_texts.extend(texts)

    if not all_texts:
        return [0.0] * len(candidates_data)

    with torch.no_grad():
        all_embs = model.encode(all_texts, batch_size=256, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
        cos_scores = util.cos_sim(jd_embedding, all_embs)[0]

    scores = []
    idx = 0
    for count in candidate_text_counts:
        if count == 0:
            scores.append(0.0)
        else:
            cand_scores = cos_scores[idx:idx + count].tolist()
            cand_scores.sort(reverse=True)
            if len(cand_scores) >= 2:
                scores.append(max(0.0, (cand_scores[0] + cand_scores[1]) / 2.0))
            else:
                scores.append(max(0.0, cand_scores[0]))
        idx += count

    return scores


def parse_date(d_str):
    if not d_str:
        return None
    try:
        return datetime.strptime(d_str, "%Y-%m-%d")
    except ValueError:
        return None


def extract_tech(career_history):
    found = set()
    for job in career_history:
        desc = job.get("description", "").lower()
        for t in TECH_LIST:
            if t.lower() in desc:
                found.add(t)
    return list(found)


def skill_matches_career(skill_name, career_text):
    """Fuzzy check: does any variant of this skill appear in career text."""
    skill_lower = skill_name.lower()
    for category, variants in CORE_JD_SKILLS.items():
        if category in skill_lower:
            for variant in variants:
                if variant in career_text:
                    return True
    return False


def compute_career_relevance_score(profile, career_history, skills):
    score = 0.0

    yoe = profile.get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        score += 0.2
    elif yoe > 9:
        score += 0.1

    vector_dbs = {"pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "vector database"}
    ml_keywords = {"embedding", "retrieval", "rag", "ranking"}
    senior_keywords = {"lead", "architect", "owned", "built", "spearheaded", "designed"}

    has_vector_db = False
    has_ml_keywords = False
    pure_consulting = True
    pure_research = True

    full_career_text = " ".join([j.get("description", "").lower() for j in career_history])
    has_senior_context = any(s in full_career_text for s in senior_keywords)

    research_titles = {"research assistant", "postdoc", "postdoctoral", "phd researcher",
                       "research fellow", "principal researcher", "staff researcher",
                       "research intern", "visiting researcher", "research associate"}

    for job in career_history:
        desc = job.get("description", "").lower()
        title = job.get("title", "").lower()
        company = job.get("company", "").lower()

        if any(v in desc for v in vector_dbs):
            has_vector_db = True
        if any(m in desc for m in ml_keywords):
            has_ml_keywords = True

        # Word-boundary matching to avoid 'ey' matching 'journey', etc.
        is_consulting = any(re.search(r'\b' + re.escape(c) + r'\b', company) for c in CONSULTING_COMPANIES)
        if not is_consulting:
            pure_consulting = False

        if not any(r in title for r in research_titles):
            pure_research = False

    if has_vector_db:
        score += 0.3
    if has_ml_keywords:
        score += 0.2

    depth_bonus = 0.0
    for skill in skills:
        s_name = skill.get("name", "").lower()
        duration = skill.get("duration_months", 0)
        prof = skill.get("proficiency", "").lower()

        if duration > 24 and prof in ["advanced", "expert"] and s_name in full_career_text and has_senior_context:
            depth_bonus += 0.1

    score += min(0.3, depth_bonus)

    if pure_consulting:
        score -= 0.4
    if pure_research:
        score -= 0.4

    return max(0.0, min(1.0, score))


def check_red_flags(profile, career_history, skills, signals):
    yoe = profile.get("years_of_experience", 0)
    gh_score = signals.get("github_activity_score", 0)
    response_rate = signals.get("recruiter_response_rate", 0)

    # github_activity_score of -1 means no GitHub linked — treat as 0
    if yoe >= 10 and gh_score <= 0 and response_rate == 0:
        return True, 1.0

    for skill in skills:
        if skill.get("proficiency", "").lower() == "expert" and skill.get("duration_months", 1) == 0:
            return True, 1.0

    penalty = 0.0

    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        months_ago = (datetime.now() - last_active).days / 30.0
        if months_ago > 18:
            penalty += 0.5

    full_career_text = " ".join([j.get("description", "").lower() for j in career_history])

    padded_skills_count = 0
    for skill in skills:
        s_name = skill.get("name", "").lower()
        if any(category in s_name for category in CORE_JD_SKILLS):
            if not skill_matches_career(s_name, full_career_text):
                padded_skills_count += 1

    if padded_skills_count > 0:
        penalty += min(0.5, padded_skills_count * 0.2)

    return False, min(1.0, penalty)


def compute_behavioral_score(signals):
    score = 0.0

    response_rate = signals.get("recruiter_response_rate", 0)
    score += response_rate * 0.5

    gh_score = max(0, signals.get("github_activity_score", 0))  # -1 means no GitHub
    if gh_score > 0:
        score += (min(gh_score, 100) / 100.0) * 0.3

    interview_comp = signals.get("interview_completion_rate", 0)
    score += interview_comp * 0.2

    return max(0.0, min(1.0, score))


def compute_recency_score(signals):
    last_active = parse_date(signals.get("last_active_date"))
    if not last_active:
        return 0.0
    days_ago = (datetime.now() - last_active).days
    if days_ago <= 0:
        return 1.0
    return math.exp(-days_ago / 90.0)


def generate_reasoning(candidate, semantic, career_rel, behav, recency, penalty, found_tech, fast_score_rank):
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "a tech company")
    title = profile.get("current_title", "Engineer")
    signals = candidate.get("redrob_signals", {})
    gh = signals.get("github_activity_score", 0)
    response_rate = signals.get("recruiter_response_rate", 0)
    last_active = signals.get("last_active_date", "unknown")

    parts = []

    tech_str = f"{', '.join(found_tech[:3])}" if found_tech else "no specific JD tech"
    parts.append(f"{yoe}yr at {company} as {title} ({tech_str})")

    if semantic > 0.6:
        parts.append(f"strong semantic match ({semantic:.2f})")
    elif semantic > 0.4:
        parts.append(f"moderate semantic match ({semantic:.2f})")
    else:
        parts.append(f"weak semantic match ({semantic:.2f})")

    if career_rel >= 0.5:
        parts.append("high career relevance")
    elif career_rel < 0.2:
        parts.append("low career relevance")

    if gh > 70:
        parts.append(f"active on GitHub ({gh})")
    elif gh == 0:
        parts.append("no GitHub activity")

    if response_rate > 0.7:
        parts.append(f"high recruiter response ({response_rate:.0%})")

    if penalty > 0.3:
        parts.append(f"penalized (penalty={penalty:.1f})")
    elif penalty > 0:
        parts.append(f"minor penalty ({penalty:.1f})")

    days_ago = "unknown"
    if last_active and last_active != "unknown":
        try:
            d = parse_date(last_active)
            if d:
                days_ago = f"{(datetime.now() - d).days}d ago"
        except:
            pass
    parts.append(f"last active: {days_ago}")

    return "; ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    model_dir = os.path.join(os.path.dirname(__file__), "models", "bge-small-en-v1.5")
    if os.path.exists(model_dir):
        model = SentenceTransformer(model_dir)
    else:
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    jd_text = load_jd_text()
    jd_embedding = model.encode(QUERY_INSTRUCTION + jd_text.strip(), convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)

    first_pass_candidates = []

    print(f"Reading from {args.candidates}...")
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            cand = json.loads(line)
            cid = cand["candidate_id"]
            profile = cand.get("profile", {})
            career = cand.get("career_history", [])
            skills = cand.get("skills", [])
            signals = cand.get("redrob_signals", {})

            is_severe, penalty = check_red_flags(profile, career, skills, signals)
            if is_severe:
                continue

            career_score = compute_career_relevance_score(profile, career, skills)
            behav_score = compute_behavioral_score(signals)
            recency = compute_recency_score(signals)

            fast_score = (WEIGHTS["career_relevance"] * career_score) + \
                         (WEIGHTS["behavioral"] * behav_score) + \
                         (WEIGHTS["recency"] * recency) - \
                         (WEIGHTS["honeypot_penalty"] * penalty)

            summary = profile.get("summary", "")
            first_pass_candidates.append((-fast_score, cid, cand, summary, career, career_score, behav_score, recency, penalty))

    top_candidates = heapq.nsmallest(5000, first_pass_candidates)
    print(f"Pass 1 complete: selected top {len(top_candidates)} candidates")

    print(f"Running batched semantic similarity on {len(top_candidates)} candidates...")
    candidates_data = [(item[3], item[4]) for item in top_candidates]
    semantic_scores = batch_semantic_scores(model, jd_embedding, candidates_data)

    final_candidates = []
    for i, item in enumerate(top_candidates):
        _, cid, cand, summary, career, career_score, behav_score, recency, penalty = item
        sem_score = semantic_scores[i]

        penalty_multiplier = 1.0 - (WEIGHTS["honeypot_penalty"] * penalty)
        final_score = penalty_multiplier * (
            (WEIGHTS["semantic"] * sem_score) +
            (WEIGHTS["career_relevance"] * career_score) +
            (WEIGHTS["behavioral"] * behav_score) +
            (WEIGHTS["recency"] * recency)
        )
        final_score = round(final_score, 6)

        found_tech = extract_tech(career)
        reasoning = generate_reasoning(cand, sem_score, career_score, behav_score, recency, penalty, found_tech, i)

        final_candidates.append((-final_score, cid, final_score, reasoning))

    top_100 = heapq.nsmallest(100, final_candidates)

    print(f"Writing top 100 to {args.out}...")
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, item in enumerate(top_100, 1):
            _, cid, score, reason = item
            writer.writerow([cid, rank, round(score, 6), reason])

    print("Done!")


if __name__ == "__main__":
    main()
