"""
Cloud model manager for Gemini (or OpenAI fallback).
Used by Evaluator and Lesson Extractor; meta-cognitive tasks, not patient-facing.
"""

import os
import time
import google.generativeai as genai


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
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("INFO: CloudManager initialized with Google Gemini.")
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
        (Note: image argument is not supported/needed for cloud meta-tasks).
        """
        max_retries = 3
        backoff_delay = 2
        
        for attempt in range(max_retries):
            try:
                if self.provider == "google":
                    return self._generate_gemini(system_prompt, user_message, max_tokens)
                elif self.provider == "openai":
                    return self._generate_openai(system_prompt, user_message, max_tokens)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Cloud API error ({e}). Retrying in {backoff_delay}s...")
                    time.sleep(backoff_delay)
                    backoff_delay *= 2
                else:
                    raise Exception(f"Cloud API failed after {max_retries} attempts: {e}")

    def _generate_gemini(self, system_prompt, user_message, max_tokens):
        # Gemini uses system_instruction parameter in the model config, 
        # or we can combine it if needed. The specs requested:
        # "combine system_prompt and user_message into a single prompt
        # (Gemini handles system instructions differently — put system prompt as
        # the model's system_instruction parameter in GenerativeModel)"
        
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=system_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=max_tokens,
            )
        )
        
        response = model.generate_content(user_message)
        
        # Approximate token tracking (GenAI has usage_metadata)
        if hasattr(response, 'usage_metadata'):
            self.tokens_sent += response.usage_metadata.prompt_token_count
            self.tokens_received += response.usage_metadata.candidates_token_count
            
        return response.text.strip()
        
    def _generate_openai(self, system_prompt, user_message, max_tokens):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=max_tokens
        )
        
        # Approximate token tracking
        if hasattr(response, 'usage'):
            self.tokens_sent += response.usage.prompt_tokens
            self.tokens_received += response.usage.completion_tokens
            
        return response.choices[0].message.content.strip()

    def print_cost_report(self):
        print(f"--- Cloud API Usage ---")
        print(f"Provider: {self.provider}")
        print(f"Tokens Sent: {self.tokens_sent}")
        print(f"Tokens Received: {self.tokens_received}")
        print(f"-----------------------")
