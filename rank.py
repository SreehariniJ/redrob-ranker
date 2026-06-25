import json
import csv
import argparse
import os
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# Explicit Scoring Weights
WEIGHTS = {
    "semantic": 0.35,
    "career_relevance": 0.25,
    "behavioral": 0.20,
    "recency": 0.10,
    "honeypot_penalty": 0.10
}

# The core JD text used for semantic matching
JD_TEXT = """
Senior AI Engineer — Founding Team. We need deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning. 
Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5) deployed to real users.
Production experience with vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS).
Strong Python code quality. Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B test).
Not pure research. Not just LangChain to OpenAI. Not pure consulting firms. Needs to write code.
"""

def compute_multi_entry_semantic_score(model, jd_embedding, profile_summary, career_history):
    texts = []
    if profile_summary and profile_summary.strip():
        texts.append(profile_summary)
        
    for job in career_history:
        desc = job.get("description", "")
        if desc and desc.strip():
            texts.append(desc)
            
    if not texts:
        return 0.0
        
    with torch.no_grad():
        cand_embs = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        cosine_scores = util.cos_sim(jd_embedding, cand_embs)[0]
        
        scores = sorted(cosine_scores.tolist(), reverse=True)
        if len(scores) >= 2:
            return max(0.0, (scores[0] + scores[1]) / 2.0)
        else:
            return max(0.0, scores[0])

def parse_date(d_str):
    if not d_str:
        return None
    try:
        return datetime.strptime(d_str, "%Y-%m-%d")
    except ValueError:
        return None

def extract_tech(career_history):
    # Search for specific tech in career history for reasoning enrichment
    techs = ["PyTorch", "Pinecone", "FAISS", "Qdrant", "Milvus", "Weaviate", "OpenSearch", "Elasticsearch", "Kafka", "Spark", "TensorFlow", "Sentence-Transformers"]
    found = set()
    for job in career_history:
        desc = job.get("description", "").lower()
        for t in techs:
            if t.lower() in desc:
                found.add(t)
    return list(found)

def compute_career_relevance_score(profile, career_history, skills):
    score = 0.0
    
    # Years of experience (Optimal: 5-9)
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
    consulting_companies = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}
    pure_consulting = True
    pure_research = True
    
    full_career_text = " ".join([j.get("description", "").lower() for j in career_history])
    has_senior_context = any(s in full_career_text for s in senior_keywords)
    
    for job in career_history:
        desc = job.get("description", "").lower()
        title = job.get("title", "").lower()
        company = job.get("company", "").lower()
        
        if any(v in desc for v in vector_dbs):
            has_vector_db = True
        if any(m in desc for m in ml_keywords):
            has_ml_keywords = True
            
        if not any(c in company for c in consulting_companies):
            pure_consulting = False
            
        if "research assistant" not in title and "postdoc" not in title:
            pure_research = False

    if has_vector_db:
        score += 0.3
    if has_ml_keywords:
        score += 0.2
        
    # Skill Depth Signal
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
    # Returns (is_severe_honeypot: bool, penalty_score: float)
    
    yoe = profile.get("years_of_experience", 0)
    gh_score = signals.get("github_activity_score", 0)
    response_rate = signals.get("recruiter_response_rate", 0)
    
    # 1. Fake Profiles (Severe)
    if yoe >= 10 and gh_score == 0 and response_rate == 0:
        return True, 1.0
        
    # 2. Fake Experts (Severe)
    for skill in skills:
        if skill.get("proficiency", "").lower() == "expert" and skill.get("duration_months", 1) == 0:
            return True, 1.0
            
    penalty = 0.0
    
    # 3. Ghosting Risk
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        months_ago = (datetime.now() - last_active).days / 30.0
        if months_ago > 18:
            penalty += 0.5
            
    # 4. Resume Padding
    core_jd_skills = {"embedding", "vector", "python", "pytorch", "pinecone", "qdrant", "milvus", "weaviate", "faiss"}
    full_career_text = " ".join([j.get("description", "").lower() for j in career_history])
    
    padded_skills_count = 0
    for skill in skills:
        s_name = skill.get("name", "").lower()
        if any(c in s_name for c in core_jd_skills):
            if s_name not in full_career_text:
                padded_skills_count += 1
                
    if padded_skills_count > 0:
        penalty += min(0.5, padded_skills_count * 0.2)
        
    return False, min(1.0, penalty)

def compute_behavioral_score(signals):
    score = 0.0
    
    # Response rate
    response_rate = signals.get("recruiter_response_rate", 0)
    score += response_rate * 0.5
    
    # GitHub Activity
    gh_score = signals.get("github_activity_score", 0)
    if gh_score > 0:
        score += (min(gh_score, 100) / 100.0) * 0.3
        
    # Interview Completion
    interview_comp = signals.get("interview_completion_rate", 0)
    score += interview_comp * 0.2
            
    return max(0.0, min(1.0, score))

def compute_recency_score(signals):
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days_ago = (datetime.now() - last_active).days
        if days_ago <= 30:
            return 1.0
        elif days_ago <= 90:
            return 0.5
    return 0.0

def generate_reasoning(candidate, semantic, career_rel, behav, found_tech):
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "a tech company")
    title = profile.get("current_title", "Engineer")
    signals = candidate.get("redrob_signals", {})
    gh = signals.get("github_activity_score", 0)
    
    # Example format required by user:
    # "4.5 years at Flipkart as ML Engineer with direct FAISS and vector search experience; github_activity_score 87; no behavioral red flags"
    
    tech_str = ""
    if found_tech:
        tech_str = f" with direct {', '.join(found_tech[:2])} experience"
        
    behav_str = ""
    if gh > 50:
        behav_str = f"; github_activity_score {gh}"
        
    flags_str = "; no behavioral red flags."
    if behav < 0.3:
        flags_str = "; some behavioral red flags present."
        
    reason = f"{yoe} years at {company} as {title}{tech_str}{behav_str}{flags_str}"
    
    return reason

def main():
    parser = argparse.ArgumentParser(description="Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    # Load semantic model locally
    model_dir = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
    if os.path.exists(model_dir):
        model = SentenceTransformer(model_dir)
    else:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    jd_embedding = model.encode(JD_TEXT, convert_to_tensor=True, show_progress_bar=False)

    first_pass_candidates = []
    
    print(f"Reading from {args.candidates}...")
    with open(args.candidates, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
                
            cand = json.loads(line)
            cid = cand["candidate_id"]
            profile = cand.get("profile", {})
            career = cand.get("career_history", [])
            skills = cand.get("skills", [])
            signals = cand.get("redrob_signals", {})
            
            # 1. Advanced Red Flag & Honeypot Detection
            is_severe, penalty = check_red_flags(profile, career, skills, signals)
            if is_severe:
                continue
                
            # 2. Fast Heuristic Filtering
            career_score = compute_career_relevance_score(profile, career, skills)
            behav_score = compute_behavioral_score(signals)
            recency = compute_recency_score(signals)
            
            # Fast score approximation using weights
            fast_score = (WEIGHTS["career_relevance"] * career_score) + \
                         (WEIGHTS["behavioral"] * behav_score) + \
                         (WEIGHTS["recency"] * recency) - \
                         (WEIGHTS["honeypot_penalty"] * penalty)
            
            summary = profile.get("summary", "")
            first_pass_candidates.append((-fast_score, cid, cand, summary, career, career_score, behav_score, recency, penalty))

    # Take top 2000 for multi-entry semantic matching
    first_pass_candidates.sort()
    top_candidates = first_pass_candidates[:2000]
    
    print(f"Running multi-entry semantic similarity on top {len(top_candidates)} candidates...")
    final_candidates = []
    for item in top_candidates:
        _, cid, cand, summary, career, career_score, behav_score, recency, penalty = item
        
        sem_score = compute_multi_entry_semantic_score(model, jd_embedding, summary, career)
        
        # Explicit Final Score
        final_score = (WEIGHTS["semantic"] * sem_score) + \
                      (WEIGHTS["career_relevance"] * career_score) + \
                      (WEIGHTS["behavioral"] * behav_score) + \
                      (WEIGHTS["recency"] * recency) - \
                      (WEIGHTS["honeypot_penalty"] * penalty)
                      
        final_score = round(final_score, 4)
                      
        found_tech = extract_tech(career)
        reasoning = generate_reasoning(cand, sem_score, career_score, behav_score, found_tech)
        
        final_candidates.append((-final_score, cid, final_score, reasoning))

    # Final sort
    final_candidates.sort()
    top_100 = final_candidates[:100]
    
    print(f"Writing top 100 to {args.out}...")
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, item in enumerate(top_100, 1):
            _, cid, score, reason = item
            writer.writerow([cid, rank, round(score, 4), reason])
            
    print("Done!")

if __name__ == "__main__":
    main()
