import json
import os
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# OpenRouter configuration
# Make sure OPENROUTER_API_KEY is set in your environment variables
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please set it before running this script.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/ministral-8b-2512"
]

def generate_bio(model, entity_name):
    prompt = f"Write a short, purely factual biography (3-5 sentences) for {entity_name}. Include details like birth year, birth place, occupations, and education where applicable. Do not include any meta-commentary."
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful and strictly factual assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating bio for {entity_name} with {model}: {e}")
        return ""

def main():
    if not os.path.exists("data/entities.json"):
        print("data/entities.json not found! Please run 1_collect_entities.py first.")
        return
        
    # Check if bios.json already exists to resume
    bios_file = "data/bios.json"
    if os.path.exists(bios_file):
        with open(bios_file, "r", encoding="utf-8") as f:
            entities = json.load(f)
        print("Resuming from existing bios.json...")
    else:
        with open("data/entities.json", "r", encoding="utf-8") as f:
            entities = json.load(f)
            # Add bios dict if it doesn't exist
            for e in entities:
                e["bios"] = {}
        
    print(f"Loaded {len(entities)} entities.")
    
    for entity in tqdm(entities, desc="Generating Bios"):
        if "bios" not in entity:
            entity["bios"] = {}
            
        for model in MODELS:
            # Skip if already generated and valid
            if model in entity["bios"] and entity["bios"][model].strip():
                continue
                
            bio_text = generate_bio(model, entity["name"])
            entity["bios"][model] = bio_text
            
    os.makedirs("data", exist_ok=True)
    with open(bios_file, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=4)
        
    print(f"Generated bios saved to data/bios.json")

if __name__ == "__main__":
    main()
