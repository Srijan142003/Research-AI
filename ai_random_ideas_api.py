from flask import Flask, request, jsonify
import random
import requests
import os

app = Flask(__name__)

# Gemini API key (set as environment variable: GEMINI_API_KEY)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# CORE API key (set as environment variable: CORE_API_KEY)
CORE_API_KEY = os.environ.get("CORE_API_KEY", "")

# Helper: Fetch latest trending papers from CORE API (https://core.ac.uk/)
def fetch_trending_papers_from_core(limit=5):
    # You can get a free API key from https://core.ac.uk/services#api
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

# Gemini API call for generating gaps/ideas
def generate_gaps_with_gemini(papers, count=5):
    if not GEMINI_API_KEY or not papers:
        return []
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
    # Compose a prompt with paper titles and abstracts
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
        # Extract bullet points or lines
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

# --- Research Analyzer Section ---
def fetch_papers_for_topic(topic, num_papers=3):
    if not CORE_API_KEY:
        return []
    url = f"https://core.ac.uk:443/api-v2/search/articles"
    params = {
        "q": topic,
        "page": 1,
        "pageSize": num_papers,
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
                "url": hit.get("url", ""),
                "keywords": hit.get("topics", []) if "topics" in hit else []
            })
        return papers
    except Exception as e:
        print("CORE API error:", e)
        return []

def analyze_paper_with_gemini(title, abstract):
    if not GEMINI_API_KEY:
        return {"analysis": "AI analysis unavailable.", "lim_scope": ""}
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
    prompt = (
        f"Given the following research paper:\nTitle: {title}\nAbstract: {abstract}\n\n"
        "Provide a concise analysis of the paper, and then list any limitations or scope for future work."
        "\nFormat:\nAnalysis: ...\nLimitations/Scope: ..."
    )
    try:
        response = model.generate_content(prompt)
        text = response.text
        # Extract analysis and limitations/scope
        analysis = ""
        lim_scope = ""
        for line in text.split("\n"):
            if line.lower().startswith("analysis:"):
                analysis = line.partition(":")[2].strip()
            elif "limitation" in line.lower() or "scope" in line.lower():
                lim_scope = line.partition(":")[2].strip()
        return {"analysis": analysis, "lim_scope": lim_scope}
    except Exception as e:
        print("Gemini API error:", e)
        return {"analysis": "AI analysis unavailable.", "lim_scope": ""}

@app.route("/api/random_ideas", methods=["POST"])
@app.route("/random_ideas", methods=["POST"])
def random_ideas():
    data = request.get_json(force=True)
    count = int(data.get("count", 5))
    # Step 1: Fetch trending papers from CORE
    papers = fetch_trending_papers_from_core(limit=5)
    # Step 2: Use Gemini to generate new gaps/ideas
    ideas = []
    if GEMINI_API_KEY and papers:
        ideas = generate_gaps_with_gemini(papers, count)
    # Step 3: Fallback if Gemini or CORE fails
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

@app.route("/api/analyze_papers", methods=["POST"])
@app.route("/analyze_papers", methods=["POST"])
def analyze_papers():
    data = request.get_json(force=True)
    topic = data.get("topic", "")
    num_papers = int(data.get("num_papers", 3))
    papers = fetch_papers_for_topic(topic, num_papers)
    result = []
    for paper in papers:
        gemini_result = analyze_paper_with_gemini(paper["title"], paper["abstract"])
        result.append({
            "title": paper["title"],
            "link": paper["url"],
            "analysis": gemini_result["analysis"],
            "lim_scope": gemini_result["lim_scope"],
            "keywords": paper.get("keywords", [])
        })
    return jsonify({"papers": result})

@app.route("/api/generate_ideas", methods=["POST"])
@app.route("/generate_ideas", methods=["POST"])
def generate_ideas():
    data = request.get_json(force=True)
    limitations = data.get("limitations", "")
    topic = data.get("topic", "")
    num_ideas = int(data.get("num_ideas", 3))
    word_limit = int(data.get("word_limit", 150))
    if not GEMINI_API_KEY:
        return jsonify({"ideas": []})
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
    prompt = (
        f"Given the following research topic: {topic}\n"
        f"and these limitations or scope from recent papers:\n{limitations}\n\n"
        f"Generate {num_ideas} new research gaps or ideas, each within {word_limit} words. "
        "Format each as a summary paragraph."
    )
    try:
        response = model.generate_content(prompt)
        text = response.text
        ideas = [line.strip("-•*1234567890. ").strip() for line in text.split("\n") if line.strip()]
        ideas = [i for i in ideas if i]
        return jsonify({"ideas": [{"summary": i} for i in ideas[:num_ideas]]})
    except Exception as e:
        print("Gemini API error:", e)
        return jsonify({"ideas": []})

@app.route("/api/elaborate", methods=["POST"])
@app.route("/elaborate", methods=["POST"])
def elaborate():
    data = request.get_json(force=True)
    topic = data.get("topic", "")
    idea_text = data.get("idea_text", "")
    word_limit = int(data.get("word_limit", 500))
    if not GEMINI_API_KEY:
        return jsonify({"result": "AI elaboration unavailable."})
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
    prompt = (
        f"Given the research topic: {topic}\n"
        f"and the following research idea or gap:\n{idea_text}\n\n"
        f"Write a detailed elaboration (within {word_limit} words) covering significance, methodology, expected challenges, and potential impact."
    )
    try:
        response = model.generate_content(prompt)
        return jsonify({"result": response.text})
    except Exception as e:
        print("Gemini API error:", e)
        return jsonify({"result": "AI elaboration unavailable."})

if __name__ == "__main__":
    app.run(port=8000, debug=True)
