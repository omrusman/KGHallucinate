import json
import os
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

DECOMPOSITION_MODEL = "meta-llama/llama-3.1-8b-instruct"

def decompose_bio(entity_name, bio_text):
    prompt = f"""
Given the following short biography of {entity_name}, break it down into atomic, independent factual claims. 
Each claim should be a single, standalone sentence about {entity_name}. 
Do not include any numbering or bullet points. Just output one claim per line.

Biography:
{bio_text}

Atomic Claims:
"""
    try:
        response = client.chat.completions.create(
            model=DECOMPOSITION_MODEL,
            messages=[
                {"role": "system", "content": "You extract simple, atomic facts from text. Output exactly one fact per line."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        claims_text = response.choices[0].message.content.strip()
        # Clean up the output to ensure it's a list of non-empty strings
        claims = [line.strip().lstrip("-*1234567890. ") for line in claims_text.split('\n') if line.strip()]
        return claims
    except Exception as e:
        print(f"Error decomposing bio for {entity_name}: {e}")
        return []

def main():
    if not os.path.exists("data/bios.json"):
        print("data/bios.json not found! Please run 2_generate_bios.py first.")
        return
        
    with open("data/bios.json", "r", encoding="utf-8") as f:
        bios_data = json.load(f)
        
    print(f"Loaded {len(bios_data)} entities with bios.")
    
    # Process each entity and decompose the bios
    for entity in tqdm(bios_data, desc="Decomposing Bios"):
        entity["decomposed_claims"] = {}
        for model, bio_text in entity["bios"].items():
            if bio_text:
                claims = decompose_bio(entity["name"], bio_text)
                entity["decomposed_claims"][model] = claims
            else:
                entity["decomposed_claims"][model] = []
                
    os.makedirs("data", exist_ok=True)
    with open("data/claims.json", "w", encoding="utf-8") as f:
        json.dump(bios_data, f, indent=4)
        
    print(f"Decomposed claims saved to data/claims.json")

if __name__ == "__main__":
    main()
