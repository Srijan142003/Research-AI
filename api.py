from flask import Flask, request, jsonify
from flask_cors import CORS
from research_analyzer import (
    search_core_papers,
    analyze_with_gemini,
    extract_limitations_scope,
    generate_new_ideas,
    elaborate_idea,
    process_papers
)
import base64
import re
import io
import sys
import os
import random
import requests

app = Flask(__name__)
CORS(app)

HF_API_KEY = os.environ.get("HF_API_KEY", "")
if not HF_API_KEY:
    print("ERROR: HF_API_KEY is not set. Please check your .env file.")

# --- Hugging Face image generation logic (from your app.py) ---
def extract_base64_from_error(error_msg):
    match = re.search(r'([A-Za-z0-9+/=]{100,})', error_msg)
    return match.group(1) if match else None

def generate_image_with_huggingface(prompt):
    HF_API_KEY = os.environ.get("HF_API_KEY", "")
    if not HF_API_KEY:
        print("ERROR: HF_API_KEY is not set. Please check your .env file.")
        return None, "HF_API_KEY not set"
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Accept": "application/json"
    }
    payload = {"inputs": prompt}
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        content_type = resp.headers.get("content-type", "")
        if resp.status_code == 200 and content_type.startswith("image/"):
            return base64.b64encode(resp.content).decode("utf-8"), None
        elif resp.status_code == 200 and content_type.startswith("application/json"):
            try:
                result = resp.json()
                base64_img = extract_base64_from_error(str(result))
                if base64_img:
                    return base64_img, None
                return None, result.get("error", str(result))
            except Exception:
                return None, f"Unknown error (could not parse JSON): {resp.text[:500]}"
        elif resp.status_code == 503:
            return None, "Model is loading. Please wait and try again."
        elif resp.status_code == 404:
            return None, "Model not found. Please check the model name or use a different model."
        else:
            try:
                result = resp.json()
                base64_img = extract_base64_from_error(str(result))
                if base64_img:
                    return base64_img, None
                return None, result.get("error", resp.text)
            except Exception:
                base64_img = extract_base64_from_error(resp.text)
                if base64_img:
                    return base64_img, None
                return None, resp.text
    except Exception as e:
        return None, f"Image generation error: {e}"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    topic = data.get("topic", "")
    num_papers = int(data.get("num_papers", 10))
    num_ideas = int(data.get("num_ideas", 10))
    word_limit = int(data.get("word_limit", 250))
    sort = data.get("sort", "relevance")
    analysis_prompt = data.get("analysis_prompt", "")
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
@app.route("/api/elaborate", methods=["POST"])
def elaborate():
    data = request.get_json()
    topic = data.get("topic", "")
    idea_text = data.get("idea_text", "")
    word_limit = int(data.get("word_limit", 500))
    # If topic is empty (as in Hot & Fresh), use idea_text as prompt for image
    try:
        result = elaborate_idea(idea_text, topic, word_limit=word_limit)
        # Use a more robust prompt for image generation
        if topic:
            prompt = f"An illustration of the following research idea: {idea_text} (Topic: {topic})"
        else:
            prompt = f"An illustration of the following research idea: {idea_text}"
        img_b64, img_error = generate_image_with_huggingface(prompt)
        image = f"data:image/png;base64,{img_b64}" if img_b64 else ""
        if not image and img_error:
            print(f"Image generation error: {img_error}")
        return jsonify({
            "result": result,
            "image": image,
            "image_error": img_error if not image else None
        })
    except Exception as e:
        print("Elaboration error:", e)
        return jsonify({
            "result": f"AI elaboration unavailable. (Exception: {e})",
            "image": "",
            "image_error": str(e)
        })

@app.route("/random_ideas", methods=["POST"])
@app.route("/api/random_ideas", methods=["POST"])
def random_ideas():
    data = request.get_json(force=True)
    count = int(data.get("count", 5))
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
    images = []
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

    # --- Generate images for each idea using Hugging Face ---
    for idea in ideas:
        prompt = f"An illustration of the following research idea: {idea}"
        img_b64, img_error = generate_image_with_huggingface(prompt)
        image = f"data:image/png;base64,{img_b64}" if img_b64 else ""
        images.append({
            "image": image,
            "image_error": img_error if not image else None
        })

    # Return both ideas and their images (frontend must be updated to use this)
    return jsonify({"ideas": ideas, "images": images})

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
