import os
import sys
import json
import time
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

# Define a Pydantic model CriticReview
class CriticReview(BaseModel):
    is_sufficient: bool = Field(
        ...,
        description="True if findings adequately answer the research question, False otherwise."
    )
    gaps: list[str] = Field(
        ...,
        description="List of specific missing or weak sub-questions/topics; empty if none."
    )
    weak_sources: list[str] = Field(
        ...,
        description="Any findings whose sources seem unreliable or insufficient; empty if none."
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the judgment."
    )

def get_model_name() -> str:
    """Finds the most suitable available Gemini model, prioritizing stable ones."""
    preferred_models = ["gemini-3.5-flash-lite", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.6-flash"]
    try:
        available_models = [m.name.replace("models/", "") for m in genai.list_models()]
        for model in preferred_models:
            if model in available_models:
                return model
    except Exception:
        pass
    return "gemini-3.5-flash-lite"  # default fallback

def generate_content_with_retry(model, *args, **kwargs):
    """Wrapper around model.generate_content to handle rate limits (429/ResourceExhausted)."""
    max_retries = 5
    base_delay = 15.0
    for attempt in range(max_retries):
        try:
            return model.generate_content(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str or "resourceexhausted" in error_str:
                delay = base_delay + (attempt * 5)
                print(f"\n[Rate Limit] 429 Quota Exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise e
    # Final try that will raise the exception if it still fails
    return model.generate_content(*args, **kwargs)

# Write a function planner(research_question: str) -> SubQuestions
def planner(research_question: str) -> SubQuestions:
    """Break a main research question into 3-5 sub-questions."""
    return generate_sub_questions(research_question)

# Write a function researcher(sub_question: str) -> ResearchFindings
def researcher(sub_question: str) -> ResearchFindings:
    """Search the web and extract structured findings answering the sub-question."""
    model_name = get_model_name()
    model = genai.GenerativeModel(model_name)
    
    # Calls search_web(sub_question) to get search results
    search_results = search_web(sub_question)
    
    # Sends the sub_question + search results to Gemini, asking it to answer using ONLY
    # the provided search results, and to list the specific source URLs used
    prompt = (
        f"You are a researcher. Answer the following sub-question using ONLY the provided search results.\n\n"
        f"Sub-question: {sub_question}\n\n"
        f"Search Results:\n{search_results}\n\n"
        f"Instructions:\n"
        f"1. Answer the sub-question as accurately and completely as possible based ONLY on the provided search results.\n"
        f"2. List the specific source URLs from the search results that support your answer. Do not invent any URLs.\n"
    )
    
    # Use structured output to get back a clean ResearchFindings object
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=ResearchFindings,
        temperature=0.2
    )
    
    response = generate_content_with_retry(
        model,
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

# Write a function critic(research_question: str, findings: list[ResearchFindings]) -> CriticReview
def critic(research_question: str, findings: list[ResearchFindings]) -> CriticReview:
    """Critically evaluate the findings to find gaps or weak sources."""
    model_name = get_model_name()
    model = genai.GenerativeModel(model_name)
    
    # Format the research findings for the critic prompt
    findings_text = ""
    for idx, f in enumerate(findings, 1):
        findings_text += f"Finding {idx}:\n"
        findings_text += f"- Sub-question: {f.sub_question}\n"
        findings_text += f"- Answer: {f.answer}\n"
        findings_text += f"- Sources: {', '.join(f.sources)}\n\n"
        
    prompt = (
        f"You are a critical reviewer. Evaluate whether the provided findings adequately and completely "
        f"answer the main research question.\n\n"
        f"Main Research Question: {research_question}\n\n"
        f"Current Findings:\n"
        f"{findings_text}\n"
        f"Evaluate the findings for:\n"
        f"1. Completeness: Are there any gaps, unanswered aspects, or missing sub-questions?\n"
        f"2. Trustworthiness: Are any sources unreliable or insufficient?\n"
        f"3. Consistency: Are there any contradictions or weak claims?\n\n"
        f"Provide your review in the required structured JSON format."
    )
    
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=CriticReview,
        temperature=0.2
    )
    
    response = generate_content_with_retry(
        model,
        prompt,
        generation_config=generation_config
    )
    
    try:
        data = json.loads(response.text)
        return CriticReview(**data)
    except Exception as e:
        print("\n--- DEBUG: RAW RESPONSE TEXT ---")
        print(response.text if hasattr(response, 'text') else response)
        print("--------------------------------\n")
        raise RuntimeError(
            f"Failed to parse response from Gemini into a CriticReview Pydantic model: {e}"
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
    
    response = generate_content_with_retry(model, prompt)
    return response.text.strip()

# Modify run_pipeline() to add a self-correction loop
def run_pipeline(research_question: str) -> tuple[str, int, list[str]]:
    """Run the complete planning, research, critic self-correction, and writing pipeline."""
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
            
    # Critic loop
    max_loops = 2
    loop_count = 0
    all_gaps_fixed = []
    
    while loop_count < max_loops:
        print(f"Critic review: checking findings sufficiency (Iteration {loop_count + 1})...")
        try:
            review = critic(research_question, findings)
        except Exception as e:
            print(f"Error: Critic review failed: {e}")
            break
            
        if review.is_sufficient or not review.gaps:
            print("Critic review: findings sufficient, proceeding to writer.")
            break
            
        loop_count += 1
        gaps_to_research = review.gaps
        all_gaps_fixed.extend(gaps_to_research)
        print(f"Critic review: {len(gaps_to_research)} gaps found, re-researching...")
        
        for idx, gap_q in enumerate(gaps_to_research, 1):
            print(f"Researching gap sub-question {idx} of {len(gaps_to_research)}: '{gap_q}'...")
            try:
                finding = researcher(gap_q)
                findings.append(finding)
            except Exception as e:
                print(f"Error: Research failed for gap sub-question {idx} ('{gap_q}'): {e}")
                
    if loop_count == max_loops:
        print(f"Reached maximum critic loop iterations ({max_loops}). Proceeding to writer.")
        
    print("Writing final report...")
    report = writer(research_question, findings)
    return report, loop_count, all_gaps_fixed

if __name__ == "__main__":
    print("=========================================")
    print("Welcome to the Agentic Research Pipeline")
    print("=========================================\n")
    
    try:
        research_question = input("Enter your research question: ").strip()
        if not research_question:
            print("Error: Research question cannot be empty.")
            sys.exit(1)
            
        final_report, loop_iterations, gaps_found = run_pipeline(research_question)
        
        # Save the final report to report.md
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(final_report)
            
        print("Report saved to report.md")
        
        # Print a short summary of how many critic loop iterations ran and what gaps (if any) were found and fixed
        print("\n================ SUMMARY ================")
        print(f"Critic loop iterations run: {loop_iterations}")
        if gaps_found:
            print("Gaps found and fixed:")
            for idx, gap in enumerate(gaps_found, 1):
                print(f"  {idx}. {gap}")
        else:
            print("No gaps found. Findings were sufficient on the first pass.")
        print("=========================================")
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as err:
        print(f"\nPipeline failed: {err}")
        sys.exit(1)
