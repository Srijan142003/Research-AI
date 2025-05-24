import os
import requests
import google.generativeai as genai
import argparse
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO

# Load environment variables
load_dotenv()
CORE_API_KEY = os.getenv("CORE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# If your version of google.generativeai does not support configure, set the API key as an environment variable or skip this line.
genai.configure(api_key=GOOGLE_API_KEY)

def get_user_topic():
    """Get research topic from user input"""
    print("\n" + "="*50)
    print("Research Paper Analysis System")
    print("="*50)
    return input("\nEnter your research topic of interest: ").strip()

def get_user_int(prompt: str, default: int = 10, min_value: int = 1, max_value: int = None) -> int:
    """Prompt user for an integer value with a default, minimum, and optional maximum value"""
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if not val:
            return default
        try:
            num = int(val)
            if num >= min_value and (max_value is None or num <= max_value):
                return num
            else:
                if max_value is not None:
                    print(f"Please enter a number between {min_value} and {max_value}")
                else:
                    print(f"Please enter a number >= {min_value}")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

from typing import List, Dict, Any

def search_core_papers(query: str, limit: int = 10, sort_by: str = "relevance") -> List[Dict[str, Any]]:
    """Search CORE API for research papers, sorted by relevance, views, or popularity"""
    if not query:
        raise ValueError("Search query cannot be empty")
    
    url = "https://api.core.ac.uk/v3/search/works"
    headers = {"Authorization": f"Bearer {CORE_API_KEY}"}
    params: dict[str, str] = {
        "q": query,
        "limit": str(limit),
        "sort": sort_by,
        "language": "en"
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get('results', [])
    except requests.HTTPError as e:
        print(f"\nCORE API error: {e} ({response.status_code})")
        if response.status_code == 500:
            print("The CORE API server encountered an error. Try a different topic, reduce the limit, or try again later.")
        else:
            print(f"Response content: {response.text}")
        return []
    except Exception as e:
        print(f"\nUnexpected error contacting CORE API: {e}")
        return []
def extract_text_from_pdf(url: str) -> str:
    """Download and extract text from PDF"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with BytesIO(response.content) as pdf_file:
            reader = PdfReader(pdf_file)
            # Filter out None values from extract_text()
            text = "\n".join([t for page in reader.pages if (t := page.extract_text())])
        return text
    except Exception as e:
        print(f"Error extracting PDF text from {url}: {e}")
        return ""
def analyze_with_gemini(text: str, prompt: str) -> str:
    """Analyze text using Gemini 1.5 Flash"""
    try:
        # Use the correct API call for Gemini 1.5 Flash
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + "\n\n" + text)
        if hasattr(response, "text"):
            return response.text
        else:
            return str(response)
    except Exception as e:
        return f"Error analyzing with Gemini: {e}"
def extract_limitations_scope(analysis: str) -> str:
    """
    Extract limitations and scope from the analysis text.
    This is a simple heuristic; for more accuracy, use an LLM.
    """
    # You can improve this extraction logic as needed.
    lines = []
    capture = False
    for line in analysis.splitlines():
        if "limitation" in line.lower() or "scope" in line.lower():
            capture = True
        if capture:
            lines.append(line)
            # Stop after a few lines or at next section
            if any(x in line.lower() for x in ["application", "potential", "relationship", "finding", "conclusion"]):
                break
    return "\n".join(lines).strip()
def generate_new_ideas(limitations_text: str, topic: str, num_ideas: int = 10, word_limit: int = 250) -> str:
    """
    Use Gemini to generate new research ideas based on limitations and gaps, with detailed elaboration for each idea.
    """
    prompt = (
        f"You are an expert research assistant. Based on the following limitations and scope found in recent research papers about '{topic}', "
        f"suggest {num_ideas} innovative research ideas or directions that address these gaps. "
        f"For each idea, elaborate thoroughly in a separate paragraph, ensuring each idea is explained in more than 100 words and within {word_limit} words. "
        "Number each idea and do not combine them. Be specific, detailed, and concise. List them as numbered points.\n\n"
        f"Limitations and Scope:\n{limitations_text}"
    )
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        if hasattr(response, "text"):
            return response.text
        else:
            return str(response)
    except Exception as e:
        return f"Error generating new ideas: {e}"
def elaborate_idea(idea_text: str, topic: str, word_limit: int = 1000) -> str:
    """
    Use Gemini to elaborate more deeply on a selected research idea.
    """
    prompt = (
        f"You are an expert research assistant. Please elaborate in detail (up to {word_limit} words) on the following research idea related to '{topic}'. "
        "Discuss its significance, possible methodology, expected challenges, and potential impact. Be thorough and insightful.\n\n"
        f"Idea:\n{idea_text}"
    )
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        if hasattr(response, "text"):
            return response.text
        else:
            return str(response)
    except Exception as e:
        return f"Error elaborating idea: {e}"

# Example usage for CLI or integration:
def display_idea_with_image(idea_text: str, topic: str, word_limit: int = 1000):
    """
    Elaborate an idea and display the AI-generated image (prints image URL or base64).
    """
    result = elaborate_idea(idea_text, topic, word_limit)
    print("\nElaboration:\n")
    print(result["elaboration"])
    print("\nAI-Generated Image (URL or base64):\n")
    print(result["image"])

def process_papers(
    query: str,
    analysis_prompt: str,
    sort_by: str = "relevance",
    num_papers: int = 10,
    num_ideas: int = 10,
    word_limit: int = 250
) -> str:
    """Main processing pipeline. Returns output as a string for web/API use."""
    output_lines = []
    try:
        papers = search_core_papers(query, limit=num_papers, sort_by=sort_by)
        if not papers:
            output_lines.append("\nNo papers found for your topic. Try a different search term.")
            return "\n".join(output_lines)

        output_lines.append(f"\nFound {len(papers)} relevant papers. Analyzing top {num_papers} (sorted by {sort_by})...")

        limitations_list = []

        for idx, paper in enumerate(papers[:num_papers]):
            output_lines.append(f"\n{'='*50}\nAnalyzing paper {idx+1}/{num_papers}")
            output_lines.append(f"Title: {paper.get('title', 'Untitled')}")
            # Show paper link if available
            paper_url = paper.get('url') or paper.get('downloadUrl') or paper.get('fullTextUrl')
            if paper_url:
                output_lines.append(f"Link: {paper_url}")
            else:
                output_lines.append("Link: Not available")

            # Extract text
            text = paper.get('fullText')
            if not text:
                download_url = paper.get('downloadUrl')
                if download_url:
                    text = extract_text_from_pdf(download_url)

            if text:
                # Analyze with Gemini
                analysis = analyze_with_gemini(text, analysis_prompt)
                output_lines.append(f"\nAnalysis:\n{analysis}")
                # Extract limitations/scope for idea generation
                lim_scope = extract_limitations_scope(analysis)
                if lim_scope:
                    limitations_list.append(lim_scope)
            else:
                output_lines.append("\nSkipped: Full text not available for analysis")

        # After all analyses, generate new research ideas
        if limitations_list:
            output_lines.append("\n" + "="*50)
            output_lines.append(f"Generating {num_ideas} new research ideas based on identified gaps (each >100 and <= {word_limit} words)...")
            all_limitations = "\n\n".join(limitations_list)
            new_ideas = generate_new_ideas(all_limitations, query, num_ideas=num_ideas, word_limit=word_limit)
            output_lines.append("\nSuggested Research Ideas:\n")
            output_lines.append(new_ideas)
        else:
            output_lines.append("\nCould not extract limitations/scope from the analyzed papers. No new ideas generated.")

    except Exception as e:
        output_lines.append(f"\nError processing papers: {str(e)}")

    return "\n".join(output_lines)

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Research Paper Analyzer')
    parser.add_argument('-t', '--topic', help='Research topic to analyze')
    parser.add_argument('-s', '--sort', choices=['relevance', 'views', 'popularity'], default='relevance',
                        help='Sort papers by: relevance (default), views, or popularity')
    parser.add_argument('-n', '--num-papers', type=int, default=None,
                        help='Number of papers to analyze')
    parser.add_argument('-i', '--num-ideas', type=int, default=None,
                        help='Number of new research ideas/gaps to generate')
    parser.add_argument('-w', '--word-limit', type=int, default=None,
                        help='Word limit for elaboration of new ideas (100-500)')
    args = parser.parse_args()
    
    # Get topic from command line or user input
    topic = args.topic if args.topic else get_user_topic()
    
    # Ensure valid topic input
    while not topic:
        print("\nPlease enter a valid research topic")
        topic = get_user_topic()

    # Get number of papers and ideas from command line or prompt
    if args.num_papers is not None:
        num_papers = args.num_papers
    else:
        num_papers = get_user_int("Enter the number of papers to analyze", default=10, min_value=1)
    if args.num_ideas is not None:
        num_ideas = args.num_ideas
    else:
        num_ideas = get_user_int("Enter the number of new research ideas/gaps to generate", default=10, min_value=1)
    min_word_limit = 100
    max_word_limit = 250
    if args.word_limit is not None and min_word_limit < args.word_limit <= max_word_limit:
        word_limit = args.word_limit
    else:
        word_limit = get_user_int(
            f"Enter the word limit for elaboration of each new idea (must be >{min_word_limit} and <= {max_word_limit})",
            default=max_word_limit,
            min_value=min_word_limit + 1,
            max_value=max_word_limit
        )
    
    ANALYSIS_PROMPT = """Please provide a detailed analysis of this research paper covering:
    1. Main research question/hypothesis
    2. Methodology used
    3. Key findings
    4. Limitations
    5. Potential applications
    6. Relationship to other work in the field"""
    
    process_papers(
        query=topic,
        analysis_prompt=ANALYSIS_PROMPT,
        sort_by=args.sort,
        num_papers=num_papers,
        num_ideas=num_ideas,
        word_limit=word_limit
    )
