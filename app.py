from flask import Flask, render_template, request
import requests
import os
from dotenv import load_dotenv
import base64
import re

load_dotenv()

app = Flask(__name__)

HF_API_KEY = os.environ.get("HF_API_KEY", "")
if not HF_API_KEY:
    print("ERROR: HF_API_KEY is not set. Please check your .env file.")

def extract_base64_from_error(error_msg):
    # Extract base64 string from error message if present
    match = re.search(r'([A-Za-z0-9+/=]{100,})', error_msg)
    return match.group(1) if match else None

def generate_image_with_huggingface(prompt):
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
                # Sometimes the API returns a base64 string in the error
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

@app.route("/", methods=["GET", "POST"])
def index():
    image_data = None
    error = None
    if request.method == "POST":
        topic = request.form.get("topic", "")
        if not topic:
            error = "Please enter a topic."
        elif not HF_API_KEY:
            error = "Hugging Face API key is not set."
        else:
            img_b64, err = generate_image_with_huggingface(topic)
            if img_b64:
                image_data = img_b64
            else:
                error = err or "Unknown error occurred."
    return render_template("index.html", image_data=image_data, error=error)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
