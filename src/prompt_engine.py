"""
Prompt formatting and LLM generation interface.
Supports Gemini API and offline deterministic cached answers for self-containment.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

STRICT_SYSTEM_PROMPT = """You are an official IFAB Laws of the Game adjudicator.
Your task is to answer the user's question strictly using ONLY the provided text passages.
Rules:
1. Base your answer solely on the supplied context chunks.
2. Cite the chunk ID (e.g., [L11_004]) supporting each claim.
3. If the context does not provide sufficient information to determine the answer, you MUST explicitly state: "Based on the provided context, the answer cannot be determined."
4. Do not invent or assume rules not present in the passages.
"""

def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Constructs a standardized prompt containing chunk IDs, headers, and text.
    Identical prompt format used for Answer 1, Answer 2, and Oracle control.
    """
    context_str = ""
    for i, c in enumerate(chunks, 1):
        cid = c.get("id", f"C{i}")
        header = c.get("header", "")
        text = c.get("text", "")
        context_str += f"\n--- Context Chunk [{cid}] ---\n{text}\n"

    prompt = f"""{STRICT_SYSTEM_PROMPT}

CONTEXT PASSAGES:
{context_str}

QUESTION:
{query}

ANSWER (grounded in context, citing chunk IDs):"""
    return prompt

class LLMInterface:
    def __init__(self, cached_answers_path: Optional[Path] = None, run_live_api: bool = True):
        self.run_live_api = run_live_api
        self.cached_answers_path = cached_answers_path
        self.cache = {}
        if cached_answers_path and cached_answers_path.exists():
            with open(cached_answers_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

        self.client = None
        self._init_gemini_client()

    def _init_gemini_client(self):
        """Initializes Gemini client if API key is present."""
        api_key = os.environ.get("GEMINI_API_KEY")
        # Kaggle user secrets fallback
        if not api_key:
            try:
                from kaggle_secrets import UserSecretsClient
                api_key = UserSecretsClient().get_secret("GEMINI_API_KEY")
            except Exception:
                pass

        if api_key and self.run_live_api:
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
            except Exception:
                try:
                    import google.generativeai as gai
                    gai.configure(api_key=api_key)
                    self.client = "legacy_gai"
                except Exception:
                    self.client = None

    def generate(self, prompt: str, cache_key: Optional[str] = None) -> str:
        """
        Generates text using live Gemini model or falls back to committed cached answers.
        """
        # Check cache first if available
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]

        if self.client and self.run_live_api:
            try:
                if self.client == "legacy_gai":
                    import google.generativeai as gai
                    model = gai.GenerativeModel("gemini-1.5-flash", generation_config={"temperature": 0.0})
                    response = model.generate_content(prompt)
                    answer = response.text
                else:
                    response = self.client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt,
                        config={"temperature": 0.0}
                    )
                    answer = response.text

                if cache_key:
                    self.cache[cache_key] = answer
                    if self.cached_answers_path:
                        with open(self.cached_answers_path, "w", encoding="utf-8") as f:
                            json.dump(self.cache, f, indent=2)
                return answer
            except Exception as e:
                print(f"[Warning] LLM API call failed ({e}). Falling back to cached answer.")

        # Return fallback placeholder if not cached and API unavailable
        return (
            f"[Offline/Cached Answer] The provided context was evaluated under zero temperature. "
            f"Please configure GEMINI_API_KEY to execute live generation."
        )
