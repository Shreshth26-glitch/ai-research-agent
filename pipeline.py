import os
import sys
import json
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field

# Import search_web from agent.py and SubQuestions, generate_sub_questions from structured_output.py
from agent import search_web
from structured_output import SubQuestions, generate_sub_questions

# Reconfigure stdout to use UTF-8 to prevent encoding crashes on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Define a Pydantic model ResearchFindings
class ResearchFindings(BaseModel):
    sub_question: str = Field(
        ..., 
        description="The sub-question that was researched."
    )
    answer: str = Field(
        ..., 
        description="The summary or synthesized answer for this sub-question based ONLY on the provided search results."
    )
    sources: list[str] = Field(
        ..., 
        description="A list of specific source URLs supporting the answer, extracted from the provided search results."
    )

def get_model_name() -> str:
    """Finds the most suitable available Gemini model, prioritizing stable ones."""
    preferred_models = ["gemini-3.5-flash", "gemini-3.6-flash"]
    try:
        available_models = [m.name.replace("models/", "") for m in genai.list_models()]
        for model in preferred_models:
            if model in available_models:
                return model
    except Exception:
        pass
    return "gemini-3.5-flash"  # default fallback

# Write a function planner(research_question: str) -> SubQuestions
def planner(research_question: str) -> SubQuestions:
    """Break a main research question into 3-5 sub-questions."""
    return generate_sub_questions(research_question)

# Write a function researcher(sub_question: str) -> ResearchFindings
def researcher(sub_question: str) -> ResearchFindings:
    """Search the web and extract structured findings answering the sub-question."""
    model_name = get_model_name()
    model = genai.GenerativeModel(model_name)
    
    # 1. Calls search_web(sub_question) to get search results
    search_results = search_web(sub_question)
    
    # 2. Sends the sub_question + search results to Gemini, asking it to answer using ONLY
    # the provided search results, and to list the specific source URLs used
    prompt = (
        f"You are a researcher. Answer the following sub-question using ONLY the provided search results.\n\n"
        f"Sub-question: {sub_question}\n\n"
        f"Search Results:\n{search_results}\n\n"
        f"Instructions:\n"
        f"1. Answer the sub-question as accurately and completely as possible based ONLY on the provided search results.\n"
        f"2. List the specific source URLs from the search results that support your answer. Do not invent any URLs.\n"
    )
    
    # 3. Use structured output (like in Phase 2) to get back a clean ResearchFindings object
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=ResearchFindings,
        temperature=0.2
    )
    
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )
    
    # Parse the response and return it
    try:
        data = json.loads(response.text)
        return ResearchFindings(**data)
    except Exception as e:
        print("\n--- DEBUG: RAW RESPONSE TEXT ---")
        print(response.text if hasattr(response, 'text') else response)
        print("--------------------------------\n")
        raise RuntimeError(
            f"Failed to parse response from Gemini into a ResearchFindings Pydantic model: {e}"
        )

# Write a function writer(research_question: str, findings: list[ResearchFindings]) -> str
def writer(research_question: str, findings: list[ResearchFindings]) -> str:
    """Synthesize all research findings into a final Markdown report."""
    model_name = get_model_name()
    model = genai.GenerativeModel(model_name)
    
    # Format the research findings for the prompt
    findings_text = ""
    for idx, f in enumerate(findings, 1):
        findings_text += f"Finding {idx}:\n"
        findings_text += f"- Sub-question: {f.sub_question}\n"
        findings_text += f"- Answer: {f.answer}\n"
        findings_text += f"- Sources: {', '.join(f.sources)}\n\n"
        
    prompt = (
        f"You are a professional technical writer. Synthesize the provided research findings into a clear, "
        f"well-organized report answering the main research question.\n\n"
        f"Main Research Question: {research_question}\n\n"
        f"Research Findings:\n"
        f"{findings_text}\n"
        f"Format and Content Requirements:\n"
        f"1. Structure: The report MUST have an Introduction, one main section per sub-question/finding, and a Conclusion.\n"
        f"2. Citations: Every factual claim must have an inline citation to its source URL (e.g., [Source Name](URL) or inline markdown link).\n"
        f"3. Style: Return ONLY the final report as a plain text/markdown string.\n"
    )
    
    response = model.generate_content(prompt)
    return response.text.strip()

# Write a function run_pipeline(research_question: str) -> str
def run_pipeline(research_question: str) -> str:
    """Run the complete planning, research, and writing pipeline."""
    print("Planning...")
    sub_questions_obj = planner(research_question)
    sub_qs = sub_questions_obj.sub_questions
    total_qs = len(sub_qs)
    
    findings = []
    for idx, sub_q in enumerate(sub_qs, 1):
        print(f"Researching sub-question {idx} of {total_qs}...")
        try:
            finding = researcher(sub_q)
            findings.append(finding)
        except Exception as e:
            # Basic error handling so that if one sub-question's research fails,
            # the pipeline logs it and continues with the rest rather than crashing entirely.
            print(f"Error: Research failed for sub-question {idx} ('{sub_q}'): {e}")
            
    print("Writing final report...")
    report = writer(research_question, findings)
    return report

if __name__ == "__main__":
    print("=========================================")
    print("Welcome to the Agentic Research Pipeline")
    print("=========================================\n")
    
    try:
        research_question = input("Enter your research question: ").strip()
        if not research_question:
            print("Error: Research question cannot be empty.")
            sys.exit(1)
            
        final_report = run_pipeline(research_question)
        
        # Save the final report to report.md
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(final_report)
            
        print("Report saved to report.md")
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as err:
        print(f"\nPipeline failed: {err}")
        sys.exit(1)
