from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

def generate_pdf():
    pdf_path = "presentation_deck.pdf"
    c = canvas.Canvas(pdf_path, pagesize=landscape(letter))
    width, height = landscape(letter)
    
    # Title Slide
    c.setFillColor(HexColor("#1A1A1D"))
    c.rect(0, 0, width, height, fill=1)
    
    c.setFillColor(HexColor("#C3073F"))
    c.setFont("Helvetica-Bold", 40)
    c.drawString(50, height - 100, "Intelligent Candidate Discovery & Ranking")
    
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont("Helvetica", 20)
    c.drawString(50, height - 150, "Team InfiniCode")
    
    c.setFont("Helvetica-Oblique", 14)
    c.drawString(50, height - 200, "An explicit, weighted hybrid scoring architecture.")
    c.showPage()
    
    # Slide 2: Architecture
    c.setFillColor(HexColor("#FFFFFF"))
    c.rect(0, 0, width, height, fill=1)
    
    c.setFillColor(HexColor("#000000"))
    c.setFont("Helvetica-Bold", 30)
    c.drawString(50, height - 80, "Architecture & Methodology")
    
    c.setFont("Helvetica", 16)
    text = [
        "Our ranker is built to run entirely locally, well within the 5-minute compute budget.",
        "We avoid heavy LLM calls at runtime by using a pre-computed local embedding model.",
        "",
        "The scoring formula is explicitly weighted into 3 pillars:",
        "1. Semantic Match (40% Weight)",
        "   - Uses all-MiniLM-L6-v2 via sentence-transformers.",
        "   - Computes Cosine Similarity between the JD and Candidate Summary + Experience.",
        "",
        "2. Rule-Based Heuristics (30% Weight)",
        "   - Extracts vector DB signals (Pinecone, Milvus, Qdrant) from role descriptions.",
        "   - Penalizes pure academic research and consulting backgrounds as per the JD.",
        "",
        "3. Behavioral Signals (30% Weight)",
        "   - Factors in Recruiter Response Rate, GitHub Activity, and Interview Completion Rate.",
        "   - Applies penalties to inactive or dormant accounts."
    ]
    
    y = height - 150
    for line in text:
        if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
            c.setFont("Helvetica-Bold", 16)
        elif line.startswith("   -"):
            c.setFont("Helvetica", 14)
        else:
            c.setFont("Helvetica", 16)
        c.drawString(50, y, line)
        y -= 25
        
    c.showPage()
    
    # Slide 3: Honeypot Detection
    c.setFillColor(HexColor("#FFFFFF"))
    c.rect(0, 0, width, height, fill=1)
    
    c.setFillColor(HexColor("#C3073F"))
    c.setFont("Helvetica-Bold", 30)
    c.drawString(50, height - 80, "Honeypot Detection & Avoidance")
    
    c.setFillColor(HexColor("#000000"))
    c.setFont("Helvetica", 16)
    
    text2 = [
        "Keyword stuffing is a major risk in naive recruitment systems.",
        "We proactively filter out candidates with impossible credentials:",
        "",
        "- 'Expert' proficiencies with 0 months of duration.",
        "- Titles that don't match the skills listed (e.g. Marketing Manager with PyTorch).",
        "",
        "By enforcing a hybrid architecture, our semantic layer ensures that the context",
        "of the skills matters, not just their presence. This robustly guards against",
        "the exact trap explicitly mentioned in the Job Description."
    ]
    
    y = height - 150
    for line in text2:
        c.drawString(50, y, line)
        y -= 25
        
    c.showPage()
    c.save()
    print(f"Generated {pdf_path}")

if __name__ == "__main__":
    generate_pdf()
