import os, requests
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

def _normalize_embedding(embedding):
    values = getattr(embedding, "values", None)
    if values is not None:
        return list(values)
    if isinstance(embedding, dict) and "values" in embedding:
        return embedding["values"]
    return embedding

def embed_texts(texts: list):
    contents = [text if isinstance(text, dict) else {"role": "user", "parts": [{"text": text}]} for text in texts]
    result = client.models.embed_content(
        model = "gemini-embedding-001",
        contents = contents
    )
    embeddings = getattr(result, "embeddings", None)
    if embeddings is None:
        single = getattr(result, "embedding", None)
        if single is None:
            raise ValueError("Gemini embed_content returned no embeddings")
        embeddings = [single]
    return [_normalize_embedding(emb) for emb in embeddings]

def embed_query(text: str):
    return embed_texts([text])[0]

def generate_answer(query, contexts, conversation_context=None):
    print("Gemini generating answer:")
    
    # Build prompt with optional conversation history
    prompt_parts = []
    
    if conversation_context:
        prompt_parts.append(conversation_context)
    
    prompt_parts.append(f"CONTEXTS:\n{contexts}")
    prompt_parts.append(f"\nQUESTION: {query}")
    
    if conversation_context:
        prompt_parts.append("\nProvide a response that takes into account the previous conversation. If the current question refers to previous messages (like 'it', 'that', 'the previous answer'), use the conversation history to understand the context. Answer concisely.")
    else:
        prompt_parts.append("\nAnswer concisely based on the provided contexts.")
    
    prompt = "\n".join(prompt_parts)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
        ),
    )
    return response.text