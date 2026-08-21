import os
import sys
import json
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field

# Reconfigure stdout to use UTF-8 to prevent encoding crashes on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Load GEMINI_API_KEY from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY is not set in the environment or .env file.")
    sys.exit(1)

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)


# 2. Define a Pydantic model called SubQuestions
class SubQuestions(BaseModel):
    main_question: str = Field(
        ..., 
        description="The original main research question."
    )
    sub_questions: list[str] = Field(
        ..., 
        description="A list of 3-5 specific, independently researchable sub-questions."
    )


# 3. Define a Pydantic model called Finding
class Finding(BaseModel):
    sub_question: str = Field(
        ..., 
        description="The specific sub-question that was researched."
    )
    answer: str = Field(
        ..., 
        description="The summary or synthesized answer for this sub-question."
    )
    sources: list[str] = Field(
        ..., 
        description="A list of sources or references (like URLs) supporting the answer."
    )


# 4. Write a function generate_sub_questions(research_question: str) -> SubQuestions
def generate_sub_questions(research_question: str) -> SubQuestions:
    """Break a main research question into 3-5 sub-questions using structured JSON outputs."""
    # Determine model availability to use a robust active model (fallback to gemini-3.6-flash if needed)
    model_name = "gemini-1.5-flash"
    try:
        available_models = [m.name.replace("models/", "") for m in genai.list_models()]
        if "gemini-1.5-flash" not in available_models and "gemini-3.6-flash" in available_models:
            model_name = "gemini-3.6-flash"
    except Exception as e:
        # Fallback to default model name if list_models fails
        pass

    model = genai.GenerativeModel(model_name)
    
    prompt = (
        f"Please break down the following research question into 3 to 5 specific, "
        f"independently researchable sub-questions:\n\n"
        f"Research Question: {research_question}"
    )
    
    # Configure the generation request to enforce a JSON schema using Pydantic
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=SubQuestions,
        temperature=0.2  # Use a lower temperature for deterministic extraction
    )
    
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )
    
    # Parse the response and handle failure
    try:
        data = json.loads(response.text)
        sub_questions_obj = SubQuestions(**data)
        return sub_questions_obj
    except Exception as e:
        print("\n--- DEBUG: RAW RESPONSE TEXT ---")
        print(response.text if hasattr(response, 'text') else response)
        print("--------------------------------\n")
        raise RuntimeError(
            f"Failed to parse response from Gemini into a SubQuestions Pydantic model: {e}"
        )


# 5. Add a if __name__ == "__main__": block
if __name__ == "__main__":
    print("==================================================")
    print("Gemini Structured JSON Output Demo (using Pydantic)")
    print("==================================================")
    
    try:
        research_question = input("Enter a research question: ").strip()
        if not research_question:
            print("Error: Research question cannot be empty.")
            sys.exit(1)
            
        print("\nGenerating sub-questions with Gemini (enforcing schema)...")
        result = generate_sub_questions(research_question)
        
        print("\nParsed SubQuestions Pydantic Object:")
        print(f"Main Question: {result.main_question}")
        print("Sub-questions:")
        for idx, sub_q in enumerate(result.sub_questions, 1):
            print(f"  {idx}. {sub_q}")
            
        # Also print the raw JSON for visual confirmation
        print("\nRaw JSON serialization:")
        print(result.model_dump_json(indent=2))
        
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as err:
        print(f"\nAn error occurred: {err}")
        sys.exit(1)
