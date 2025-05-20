from flask import Flask, request, jsonify
from flask_cors import CORS
from research_analyzer import (
    search_core_papers,
    analyze_with_gemini,
    extract_limitations_scope,
    generate_new_ideas,
    elaborate_idea  # You may need to add this function if not present
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    # Extract and validate fields
    topic = data.get("topic", "")
    num_papers = int(data.get("num_papers", 10))
    num_ideas = int(data.get("num_ideas", 10))
    word_limit = int(data.get("word_limit", 250))
    sort = data.get("sort", "relevance")
    analysis_prompt = data.get("analysis_prompt", "")

    # --- Refactor process_papers to return a string result instead of printing ---
    # If your process_papers currently prints to terminal, you need to capture its output.
    # Example: Use io.StringIO to capture print output.
    import io
    import sys

    output = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = output
    try:
        result = process_papers(
            query=topic,
            num_papers=num_papers,
            num_ideas=num_ideas,
            word_limit=word_limit,
            analysis_prompt=analysis_prompt,
            sort_by=sort
        )
    finally:
        sys.stdout = sys_stdout
    printed_output = output.getvalue()
    output.close()

    # Prefer returning printed_output if process_papers prints, else return result
    return jsonify({"result": printed_output if printed_output.strip() else result})

@app.route("/analyze_papers", methods=["POST"])
def analyze_papers():
    data = request.get_json()
    topic = data.get("topic", "")
    num_papers = int(data.get("num_papers", 3))
    analysis_prompt = """Please provide a detailed analysis of this research paper covering:
    1. Main research question/hypothesis
    2. Methodology used
    3. Key findings
    4. Limitations
    5. Potential applications
    6. Relationship to other work in the field"""
    papers = search_core_papers(topic, limit=num_papers)
    result = []
    for paper in papers[:num_papers]:
        title = paper.get('title', 'Untitled')
        link = paper.get('url') or paper.get('downloadUrl') or paper.get('fullTextUrl')
        text = paper.get('fullText')
        if not text:
            download_url = paper.get('downloadUrl')
            if download_url:
                from research_analyzer import extract_text_from_pdf
                text = extract_text_from_pdf(download_url)
        if text:
            analysis = analyze_with_gemini(text, analysis_prompt)
            lim_scope = extract_limitations_scope(analysis)
        else:
            analysis = "Skipped: Full text not available for analysis"
            lim_scope = ""
        result.append({
            "title": title,
            "link": link,
            "analysis": analysis,
            "lim_scope": lim_scope
        })
    return jsonify({"papers": result})

@app.route("/generate_ideas", methods=["POST"])
def generate_ideas():
    data = request.get_json()
    limitations = data.get("limitations", "")
    topic = data.get("topic", "")
    num_ideas = int(data.get("num_ideas", 3))
    word_limit = int(data.get("word_limit", 150))
    ideas_text = generate_new_ideas(limitations, topic, num_ideas=num_ideas, word_limit=word_limit)
    # Split ideas as numbered list or bullet points
    import re
    ideas = []
    numbered = re.findall(r"\d+\.\s(.+?)(?=\n\d+\.|\n*$)", ideas_text, re.DOTALL)
    if numbered:
        for idea in numbered:
            ideas.append({"summary": idea.strip()})
    else:
        bullets = re.findall(r"[-*]\s(.+)", ideas_text)
        for idea in bullets:
            ideas.append({"summary": idea.strip()})
    return jsonify({"ideas": ideas})

@app.route("/elaborate", methods=["POST"])
def elaborate():
    data = request.get_json()
    topic = data.get("topic", "")
    idea_text = data.get("idea_text", "")
    word_limit = int(data.get("word_limit", 500))
    # Use the same elaborate_idea logic as before, or define it here
    if hasattr(elaborate_idea, "__call__"):
        result = elaborate_idea(idea_text, topic, word_limit=word_limit)
    else:
        # fallback: simple Gemini call
        from research_analyzer import genai
        prompt = (
            f"You are an expert research assistant. Please elaborate in detail (up to {word_limit} words) on the following research idea related to '{topic}'. "
            "Discuss its significance, possible methodology, expected challenges, and potential impact. Be thorough and insightful.\n\n"
            f"Idea:\n{idea_text}"
        )
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        result = response.text if hasattr(response, "text") else str(response)
    return jsonify({"result": result})

@app.route("/random_ideas", methods=["POST"])
@app.route("/api/random_ideas", methods=["POST"])
def random_ideas():
    """
    Returns a list of fresh AI-generated research ideas based on trending papers.
    Requires GEMINI_API_KEY and CORE_API_KEY as environment variables.
    """
    data = request.get_json(force=True)
    count = int(data.get("count", 5))

    # --- Fetch trending papers from CORE API ---
    import os
    import random
    import requests

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    CORE_API_KEY = os.environ.get("CORE_API_KEY", "")

    def fetch_trending_papers_from_core(limit=5):
        if not CORE_API_KEY:
            return []
        url = f"https://core.ac.uk:443/api-v2/search/articles"
        params = {
            "q": "machine learning OR artificial intelligence OR deep learning OR data science OR quantum computing",
            "page": 1,
            "pageSize": limit,
            "sort": "relevance"
        }
        headers = {
            "Authorization": f"Bearer {CORE_API_KEY}"
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            papers = []
            for hit in data.get("results", []):
                papers.append({
                    "title": hit.get("title", ""),
                    "authors": hit.get("authors", []),
                    "abstract": hit.get("description", ""),
                    "url": hit.get("url", "")
                })
            return papers
        except Exception as e:
            print("CORE API error:", e)
            return []

    def generate_gaps_with_gemini(papers, count=5):
        if not GEMINI_API_KEY or not papers:
            return []
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-pro")
        prompt = (
            "Given the following recent research papers, identify {} new research gaps or ideas that have not been addressed. "
            "For each, provide a concise and specific research idea or gap:\n\n".format(count)
        )
        for idx, paper in enumerate(papers, 1):
            prompt += f"Paper {idx}: {paper['title']}\nAbstract: {paper['abstract']}\n\n"
        prompt += "List the new research gaps or ideas as bullet points."
        try:
            response = model.generate_content(prompt)
            text = response.text
            ideas = []
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    ideas.append(line[2:].strip())
                elif line and not line.lower().startswith("paper"):
                    ideas.append(line.strip("-•*1234567890. ").strip())
            # Remove empty and duplicate ideas
            ideas = [i for i in ideas if i]
            seen = set()
            unique_ideas = []
            for idea in ideas:
                if idea not in seen:
                    unique_ideas.append(idea)
                    seen.add(idea)
            return unique_ideas[:count]
        except Exception as e:
            print("Gemini API error:", e)
            return []

    papers = fetch_trending_papers_from_core(limit=5)
    ideas = []
    if GEMINI_API_KEY and papers:
        ideas = generate_gaps_with_gemini(papers, count)
    if not ideas:
        fallback_ideas = [
            "Explainable AI for medical imaging diagnosis",
            "Quantum algorithms for large-scale optimization",
            "Privacy-preserving federated learning in healthcare",
            "Bias detection in language models for legal documents",
            "Energy-efficient deep learning for edge devices"
        ]
        ideas = random.sample(fallback_ideas, min(count, len(fallback_ideas)))
    return jsonify({"ideas": ideas})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

# To run this Flask app locally, open a terminal in your project directory and run:
# (Make sure you have Flask installed: pip install Flask)
#
#     python api.py
#
# This will start the server at http://127.0.0.1:8000
# You can access the interactive docs at http://127.0.0.1:8000/docs
#
# To deploy on Vercel, follow the previous instructions for requirements.txt and vercel.json,
# then run `vercel --prod` in your project directory.

# To test your Flask program locally:
#
# 1. Start the server:
#    python api.py
#
# 2. Open your browser and go to:
#    http://127.0.0.1:8000/docs
#    - Here you can interactively test the /analyze endpoint.
#
# 3. Or use curl/Postman:
#    curl -X POST "http://127.0.0.1:8000/analyze" -H "Content-Type: application/json" -d '{"query": "machine learning"}'
#
# Make sure your .env file is set up with the required API keys.
