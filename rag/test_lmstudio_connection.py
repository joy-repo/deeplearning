import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()

def test_connection():
    print("Testing connection to LM Studio...")
    
    # LM Studio endpoint (OpenAI-compatible)
    endpoint = "http://127.0.0.1:1234/v1/embeddings"
    
    # Test with direct API call first
    headers = {"Content-Type": "application/json"}
    payload = {
        "input": ["Hello, this is a test."],
        "model": "embedding-model"  # Required for OpenAI compatibility
    }
    
    print("\nTesting direct API connection...")
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            embedding = data["data"][0]["embedding"]
            print("✓ Direct API call successful!")
            print(f"✓ Got embedding vector of length: {len(embedding)}")
        else:
            print("✗ Unexpected API response format:")
            print(json.dumps(data, indent=2))
    except Exception as e:
        print("✗ Direct API call failed:")
        print(f"Error: {str(e)}")
    
    print("\nTesting via wrapper class...")
    # Now test the wrapper class
    from ingest_database import get_embeddings_model
    
    embeddings = get_embeddings_model()
    try:
        embedding = embeddings.embed_query("Hello, this is a test.")
        print("✓ Wrapper class successful!")
        print(f"✓ Got embedding vector of length: {len(embedding)}")
    except Exception as e:
        print("✗ Wrapper class failed:")
        print(f"Error: {str(e)}")
    
    print("\nTroubleshooting tips if failed:")
    print("1. Make sure LM Studio is running")
    print("2. Check that the API server is started in LM Studio")
    print("3. Verify the endpoint URL matches LM Studio's interface")
    print("4. Check if LM Studio shows any errors")
    print("5. Try accessing http://127.0.0.1:1234/ in your browser")

if __name__ == "__main__":
    test_connection()