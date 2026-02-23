"""
Cloud model manager for Gemini (or OpenAI fallback).
Used by Evaluator and Lesson Extractor; meta-cognitive tasks, not patient-facing.
Tries Gemini models in priority order, falling back to the next on failure.
"""

import os
import time
import google.generativeai as genai


# Priority-ordered Gemini models. Best quality first, most available last.
GEMINI_MODELS = [
    "gemini-2.5-flash",       # Best balance: quality + 1K RPM quota
    "gemini-2.0-flash-001",   # Stable 2.0: 2K RPM quota
    "gemini-2.5-flash-lite",  # Lighter 2.5: 4K RPM quota
    "gemini-2.0-flash-lite",  # Lightest 2.0: 4K RPM quota
    "gemini-2.5-pro",         # Most capable: 150 RPD (last resort)
]


class CloudManager:
    """Singleton: Gemini (or OpenAI) for Evaluator and Lesson Extractor."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CloudManager, cls).__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self):
        self.provider = None
        self.api_key = None
        self.model = None
        self.active_gemini_model = None

        # Cost tracking
        self.tokens_sent = 0
        self.tokens_received = 0

        self._initialize_provider()

    def _initialize_provider(self):
        google_key = os.environ.get("GOOGLE_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")

        if google_key:
            self.provider = "google"
            self.api_key = google_key
            genai.configure(api_key=google_key)
            self.active_gemini_model = GEMINI_MODELS[0]
            print(f"INFO: CloudManager initialized with Google Gemini ({self.active_gemini_model}).")
        elif openai_key:
            self.provider = "openai"
            self.api_key = openai_key
            from openai import OpenAI
            self.client = OpenAI(api_key=openai_key)
            print("INFO: CloudManager initialized with OpenAI GPT-4o.")
        else:
            raise ValueError("CloudManager Error: Neither GOOGLE_API_KEY nor OPENAI_API_KEY found.")

    def generate_response(self, system_prompt, user_message, max_tokens=1024):
        """
        Same signature as medgemma_manager.generate_response.
        For Gemini: tries each model in GEMINI_MODELS order, up to 3 attempts
        per model before falling back to the next. Raises only if all 5 fail.
        """
        if self.provider == "google":
            return self._generate_gemini_with_fallback(system_prompt, user_message, max_tokens)
        elif self.provider == "openai":
            return self._generate_openai_with_retry(system_prompt, user_message, max_tokens)

    def _generate_gemini_with_fallback(self, system_prompt, user_message, max_tokens):
        """Try each Gemini model in order; 3 attempts per model before moving on."""
        last_error = None

        for model_name in GEMINI_MODELS:
            backoff = 2
            for attempt in range(3):
                try:
                    result = self._call_gemini(model_name, system_prompt, user_message, max_tokens)
                    if self.active_gemini_model != model_name:
                        print(f"INFO: Now using {model_name}.")
                        self.active_gemini_model = model_name
                    return result
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    # 404 = model unavailable for this key — skip immediately, no retries
                    if "404" in err_str or "no longer available" in err_str.lower():
                        print(f"WARN: {model_name} unavailable, trying next model...")
                        break
                    # Other errors (rate limit, timeout) — retry with backoff
                    if attempt < 2:
                        print(f"WARN: {model_name} error (attempt {attempt+1}/3): {e}. Retrying in {backoff}s...")
                        time.sleep(backoff)
                        backoff *= 2
                    else:
                        print(f"WARN: {model_name} failed 3 times, trying next model...")

        raise Exception(f"All Gemini models exhausted. Last error: {last_error}")

    def _call_gemini(self, model_name, system_prompt, user_message, max_tokens):
        """Single Gemini API call for a given model name."""
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=max_tokens,
            )
        )

        response = model.generate_content(user_message)

        if hasattr(response, 'usage_metadata'):
            self.tokens_sent += response.usage_metadata.prompt_token_count or 0
            self.tokens_received += response.usage_metadata.candidates_token_count or 0

        return response.text.strip()

    def _generate_openai_with_retry(self, system_prompt, user_message, max_tokens):
        """OpenAI with 3-attempt retry."""
        backoff = 2
        last_error = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                if hasattr(response, 'usage'):
                    self.tokens_sent += response.usage.prompt_tokens
                    self.tokens_received += response.usage.completion_tokens
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                if attempt < 2:
                    print(f"OpenAI error (attempt {attempt+1}/3): {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
        raise Exception(f"OpenAI failed after 3 attempts: {last_error}")

    def print_cost_report(self):
        print(f"--- Cloud API Usage ---")
        print(f"Provider: {self.provider}")
        print(f"Active model: {self.active_gemini_model or 'N/A'}")
        print(f"Tokens Sent: {self.tokens_sent}")
        print(f"Tokens Received: {self.tokens_received}")
        print(f"-----------------------")
