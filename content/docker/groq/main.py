import os
from groq import Groq
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Please set it in a .env file or pass it as an environment variable.")
        return

    try:
        client = Groq(
            api_key=api_key,
        )
        print("Groq client initialized successfully.")
        
        # simple test to list models if possible, or just print success
        # We won't make a call that costs money or requires complex setup, 
        # just checking if client init didn't crash immediately.
        
        print("Attempting to list models to verify connection...")
        models = client.models.list()
        print(f"Successfully retrieved {len(models.data)} models.")
        print("First 3 models:")
        for model in models.data[:3]:
            print(f"- {model.id}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
