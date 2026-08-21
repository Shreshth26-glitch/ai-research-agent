import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import protos
from tavily import TavilyClient

# Reconfigure stdout to use UTF-8 to prevent encoding crashes on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Load environment variables from .env
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 7. Basic error handling: check if keys are configured
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY is not set in the environment or .env file.")
if not TAVILY_API_KEY:
    print("Error: TAVILY_API_KEY is not set in the environment or .env file.")

# Configure the Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Configure the Tavily Client
tavily_client = None
if TAVILY_API_KEY:
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    except Exception as e:
        print(f"Error initializing Tavily Client: {e}")

# 3. Define the python function search_web
def search_web(query: str) -> str:
    """Search the web for information using Tavily.

    Args:
        query: The search query to run.

    Returns:
        A formatted string with the top search results, including title, URL, and a snippet.
    """
    if not tavily_client:
        return "Error: Tavily client is not configured."
    
    try:
        # Request top 5 results
        response = tavily_client.search(query=query, max_results=5)
        results = response.get("results", [])
        if not results:
            return "No results found."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No Title")
            url = result.get("url", "No URL")
            content = result.get("content", "No content snippet.")
            formatted_results.append(
                f"{i}. Title: {title}\n   URL: {url}\n   Snippet: {content}\n"
            )
        return "\n".join(formatted_results)
    except Exception as e:
        # 7. Basic error handling
        print(f"Error calling Tavily API: {e}")
        return f"Error executing search: {e}"

# 2 & 4. Set up Gemini model with tools
# We dynamically check which model is available to avoid 404 errors.
model_name = "gemini-1.5-flash"  # default requested model
if GEMINI_API_KEY:
    try:
        available_models = [m.name.replace("models/", "") for m in genai.list_models()]
        # If gemini-1.5-flash is not available, we fall back to gemini-3.6-flash which is the active model
        if "gemini-1.5-flash" not in available_models and "gemini-3.6-flash" in available_models:
            model_name = "gemini-3.6-flash"
    except Exception as e:
        print(f"Warning: Could not list models ({e}). Using default: {model_name}")

try:
    # Set up the GenerativeModel with search_web as a tool.
    # The SDK automatically uses function signature and docstrings to build the Tool/FunctionDeclaration schema.
    model = genai.GenerativeModel(
        model_name=model_name,
        tools=[search_web],
        system_instruction=(
            "You are a helpful research assistant. When answering questions, use the search_web tool "
            "to look up information. You must include inline citations (e.g., '[Source: URL]') for "
            "any factual claims you make, citing the specific URLs from the search results."
        )
    )
except Exception as e:
    print(f"Error configuring Gemini model: {e}")
    model = None

# 5. Write the run_agent function
def run_agent(question: str) -> str:
    if not model:
        return "Error: Gemini model is not configured."
    
    try:
        # Initialize conversation contents with the user's question
        contents = [
            protos.Content(
                role="user",
                parts=[protos.Part(text=question)]
            )
        ]
        
        # Max turns to prevent infinite loops (standard for agent loops)
        max_turns = 5
        for turn in range(max_turns):
            # Send current conversation context to Gemini
            response = model.generate_content(contents=contents)
            
            # Extract function calls from response manually (raw mechanics)
            function_calls = []
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.function_call:
                            function_calls.append(part.function_call)
            
            if function_calls:
                # Gemini decided to call one or more tools.
                # First, append the model's message containing the tool calls to history.
                contents.append(response.candidates[0].content)
                
                tool_parts = []
                for fc in function_calls:
                    if fc.name == "search_web":
                        query = fc.args.get("query")
                        print(f"\n[Agent] Gemini requested tool: search_web(query='{query}')")
                        
                        # Execute search_web
                        search_results = search_web(query)
                        
                        # Package the result into a FunctionResponse part
                        tool_part = protos.Part(
                            function_response=protos.FunctionResponse(
                                name="search_web",
                                response={"result": search_results}
                            )
                        )
                        tool_parts.append(tool_part)
                
                # Append the function response parts with role="user" to history
                if tool_parts:
                    contents.append(
                        protos.Content(
                            role="user",
                            parts=tool_parts
                        )
                    )
            else:
                # No more function calls, we have the final text answer
                return response.text
                
        return "Error: Agent reached maximum research turns without generating a final response."
    except Exception as e:
        # 7. Basic error handling
        print(f"Error during agent execution: {e}")
        return f"Error: {e}"

# 6. Main execution block
if __name__ == "__main__":
    print("=========================================")
    print("Welcome to the Gemini + Tavily Research Agent")
    print(f"Using model: {model_name}")
    print("=========================================\n")
    
    try:
        question = input("Enter your research question: ").strip()
        if not question:
            print("Question cannot be empty.")
            sys.exit(0)
            
        print("\nResearching, please wait...")
        final_answer = run_agent(question)
        
        print("\n---------------- FINAL ANSWER ----------------")
        print(final_answer)
        print("----------------------------------------------")
    except KeyboardInterrupt:
        print("\nExiting...")
