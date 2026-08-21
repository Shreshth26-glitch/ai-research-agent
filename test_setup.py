import os
from dotenv import load_dotenv
import google.generativeai as genai

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key == "your_key_here":
        print("Error: GEMINI_API_KEY is not set correctly in the .env file.")
        print("Please replace 'your_key_here' with your actual Gemini API key.")
        return

    # Configure the Gemini API
    genai.configure(api_key=api_key)

    print("Sending request to Gemini API...")
    try:
        # Create the model instance
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Generate the response
        response = model.generate_content("Say hello in one sentence.")
        
        print("\nResponse:")
        print(response.text.strip())
    except Exception as e:
        print(f"\nAn error occurred while calling the Gemini API: {e}")

if __name__ == "__main__":
    main()
