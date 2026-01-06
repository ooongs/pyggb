#!/usr/bin/env python3
"""
Multimodal Interface for Vision LLMs
Supports GPT-4V, GPT-4o, Claude 3.5 Sonnet, OpenRouter, and vLLM models with vision.
"""

import os
import base64
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class MultimodalMessage:
    """Container for multimodal message with text and images."""
    
    def __init__(self, text: str, images: List[str] = None):
        """
        Initialize multimodal message.
        
        Args:
            text: Text content
            images: List of base64-encoded images
        """
        self.text = text
        self.images = images or []
    
    def add_image(self, image_base64: str):
        """Add an image to the message."""
        self.images.append(image_base64)


class MultimodalInterface:
    """Interface for vision-enabled LLMs."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None,
                 api_base: Optional[str] = None, use_cache: bool = True,
                 cache_ttl: str = "18000s"):
        """
        Initialize multimodal interface.

        Args:
            model: Model name (gpt-4o, gpt-4-vision-preview, claude-3-5-sonnet-20241022,
                              or vLLM model name like Qwen/Qwen2.5-VL-7B-Instruct)
            api_key: API key (if None, loads from environment)
            api_base: API base URL for vLLM or custom endpoints
            use_cache: Enable context caching for Gemini models (system_prompt caching)
            cache_ttl: Cache time-to-live (e.g., "3600s" for 1 hour, default)
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self._cache = None  # Store Gemini cache object

        # OpenRouter support (OpenAI-compatible)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_base = None
        if openrouter_key:
            openrouter_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
            if self.api_base is None:
                self.api_base = openrouter_base

        self.is_openrouter = self.api_base is not None and "openrouter.ai" in self.api_base
        self.openrouter_headers = {}
        if self.is_openrouter:
            referer = os.getenv("OPENROUTER_SITE_URL")
            title = os.getenv("OPENROUTER_APP_NAME") or os.getenv("OPENROUTER_TITLE")
            if referer:
                self.openrouter_headers["HTTP-Referer"] = referer
            if title:
                self.openrouter_headers["X-Title"] = title
        
        # Determine provider
        model_l = model.lower()
        if (not self.is_openrouter) and ("gemini" in model_l or model_l.startswith("models/gemini")):
            self.provider = "google"
            if api_key is None:
                self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        elif "claude" in model_l or "anthropic" in model_l:
            self.provider = "anthropic"
            if api_key is None:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            # Default to OpenAI-compatible API (works for GPT, OpenRouter, and vLLM)
            if api_key is None:
                self.api_key = openrouter_key or os.getenv("OPENAI_API_KEY")

            if self.is_openrouter:
                # Always treat OpenRouter as openrouter provider, even with "/" in model names
                self.provider = "openrouter"
            else:
                # Check if this is a vLLM model (typically has / in name)
                if "/" in model and self.api_base:
                    self.provider = "vllm"
                else:
                    self.provider = "openai"
        
        if not self.api_key:
            # For vLLM, API key might be optional
            if self.provider == "google":
                use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "y",
                    "on",
                )
                if not use_vertex:
                    raise ValueError("API key not found for google (set GOOGLE_API_KEY or GEMINI_API_KEY)")
            elif self.provider != "vllm":
                raise ValueError(f"API key not found for {self.provider}")
            else:
                # Use dummy key for vLLM
                self.api_key = "dummy-key"
        
        # Initialize client
        self._initialize_client()

    def __del__(self):
        """Cleanup: delete cache when interface is destroyed."""
        try:
            if hasattr(self, 'provider') and self.provider == "google" and hasattr(self, '_cache') and self._cache is not None:
                self.clear_cache()
        except:
            pass  # Ignore errors during cleanup
    
    def _initialize_client(self):
        """Initialize the appropriate API client."""
        if self.provider in ["openai", "vllm", "openrouter"]:
            try:
                from openai import OpenAI
                
                # Build client args
                client_args = {"api_key": self.api_key}
                if self.api_base:
                    client_args["base_url"] = self.api_base
                if self.provider == "openrouter" and self.openrouter_headers:
                    client_args["default_headers"] = self.openrouter_headers
                
                self.client = OpenAI(**client_args)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        elif self.provider == "google":
            try:
                from google import genai
                from google.genai import types
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. Run: pip install google-genai"
                )

            self._google_types = types

            use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
                "on",
            )
            if use_vertex:
                project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEXAI_PROJECT")
                location = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("VERTEXAI_LOCATION")
                if not project or not location:
                    raise ValueError(
                        "Vertex AI mode enabled but GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION not set"
                    )
                self.client = genai.Client(vertexai=True, project=project, location=location)
            else:
                self.client = genai.Client(api_key=self.api_key)
    
    def send_message(self, message: MultimodalMessage, 
                    system_prompt: Optional[str] = None,
                    temperature: float = 0,
                    max_tokens: int = 4000) -> str:
        """
        Send multimodal message to LLM.
        
        Args:
            message: MultimodalMessage with text and images
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Response text from LLM
        """
        if self.provider in ["openai", "vllm", "openrouter"]:
            return self._send_openai(message, system_prompt, temperature, max_tokens)
        elif self.provider == "anthropic":
            return self._send_anthropic(message, system_prompt, temperature, max_tokens)
        elif self.provider == "google":
            return self._send_google(message, system_prompt, temperature, max_tokens)
    
    def _send_openai(self, message: MultimodalMessage, system_prompt: Optional[str],
                     temperature: float, max_tokens: int) -> str:
        """Send message to OpenAI-compatible API (including vLLM)."""
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        

        if self.model.startswith("gpt-5"):
            # Build user message with text and images
            content = []

            # Add text
            content.append({"type": "input_text", "text": message.text})

            # Add images
            for img_base64 in message.images:
                # GPT-5 expects image_url as a string, not an object
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{img_base64}"
                })

            messages.append({"role": "user", "content": content})
        else:
            content = []
        
            # Add text
            content.append({"type": "text", "text": message.text})
            
            # Add images
            for img_base64 in message.images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}",
                        "detail": "high"  # Use high detail for geometric precision
                    }
                })
            
            messages.append({"role": "user", "content": content})
        
        # Call API
        try:
            use_responses_api = self.model.startswith("gpt-5") and self.provider == "openai"

            if use_responses_api:
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    reasoning={ "effort": "low" },
                    text={ "verbosity": "low" }
                )
                return response.output_text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
                return response.choices[0].message.content
        except Exception as e:
            # Handle vLLM specific errors
            if self.provider == "vllm":
                error_msg = str(e)
                if "model" in error_msg.lower():
                    raise ValueError(
                        f"vLLM model error: {error_msg}. "
                        f"Make sure the model '{self.model}' is loaded in vLLM server."
                    )
            raise
    
    def _send_anthropic(self, message: MultimodalMessage, system_prompt: Optional[str],
                       temperature: float, max_tokens: int) -> str:
        """Send message to Anthropic API."""
        content = []
        
        # Add images first (Anthropic prefers images before text)
        for img_base64 in message.images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_base64
                }
            })
        
        # Add text
        content.append({"type": "text", "text": message.text})
        
        # Call API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": content}],
        )
        
        return response.content[0].text

    def _create_gemini_cache(self, system_prompt: str) -> Any:
        """
        Create a Gemini context cache for the system prompt.

        Args:
            system_prompt: The system prompt to cache

        Returns:
            Cache object from Gemini API
        """
        if not hasattr(self, "_google_types"):
            raise RuntimeError("Google client not initialized (provider is not 'google').")

        types = self._google_types

        # Ensure model has explicit version suffix for caching
        model_for_cache = self.model
        if not any(model_for_cache.endswith(suffix) for suffix in ["-001", "-002", "-exp"]):
            # Try to append default version if not specified
            if "gemini-2.0-flash" in model_for_cache:
                model_for_cache = "models/gemini-2.0-flash-001"
            elif "gemini-1.5-pro" in model_for_cache:
                model_for_cache = "models/gemini-1.5-pro-002"
            elif "gemini-1.5-flash" in model_for_cache:
                model_for_cache = "models/gemini-1.5-flash-002"

        # Ensure "models/" prefix
        if not model_for_cache.startswith("models/"):
            model_for_cache = f"models/{model_for_cache}"

        try:
            cache = self.client.caches.create(
                model=model_for_cache,
                config=types.CreateCachedContentConfig(
                    display_name=f"pyggb-cache-{hash(system_prompt) % 10000}",
                    system_instruction=system_prompt,
                    ttl=self.cache_ttl
                )
            )
            print(f"✓ Cache created: {cache.name}")
            return cache
        except Exception as e:
            error_msg = str(e)
            # Don't show warning for "too small" errors - it's expected for short prompts
            if "too small" in error_msg.lower() or "min_total_token_count" in error_msg:
                print(f"Note: System prompt too short for caching (need ≥1024 tokens). Using non-cached requests.")
            else:
                print(f"Warning: Failed to create cache: {e}")
                print("Falling back to non-cached requests")
            return None

    def _get_or_create_cache(self, system_prompt: Optional[str]) -> Optional[Any]:
        """Get existing cache or create new one if caching is enabled."""
        if not self.use_cache or not system_prompt:
            return None

        # Skip caching for experimental models (they don't support caching)
        if "-exp" in self.model.lower() or "exp-" in self.model.lower():
            return None

        # Skip caching for Pro models (they require 4096+ tokens, while Flash models require 1024+)
        # Pro models: gemini-1.5-pro, gemini-2.5-pro, etc.
        if "pro" in self.model.lower() and "gemini" in self.model.lower():
            return None

        # Create cache if not exists
        if self._cache is None:
            self._cache = self._create_gemini_cache(system_prompt)

        return self._cache

    def clear_cache(self):
        """Delete the Gemini cache if it exists."""
        if self._cache is not None and hasattr(self.client, 'caches'):
            try:
                self.client.caches.delete(name=self._cache.name)
                print(f"Cache {self._cache.name} deleted successfully")
            except Exception as e:
                print(f"Warning: Failed to delete cache: {e}")
            finally:
                self._cache = None

    def _send_google(self, message: MultimodalMessage, system_prompt: Optional[str],
                     temperature: float = 1.0, max_tokens: int = 4000) -> str:
        """Send message to Google Gemini API via google-genai SDK."""
        if not hasattr(self, "_google_types"):
            raise RuntimeError("Google client not initialized (provider is not 'google').")

        types = self._google_types

        def _parse_data_url(image_b64_or_data_url: str) -> tuple[str, str]:
            s = (image_b64_or_data_url or "").strip()
            if s.startswith("data:") and ";base64," in s:
                header, payload = s.split(";base64,", 1)
                mime_type = header[5:] or "image/png"
                return mime_type, payload
            return "image/png", s

        parts: List[Any] = []

        for img in message.images:
            mime_type, payload_b64 = _parse_data_url(img)
            try:
                img_bytes = base64.b64decode(payload_b64, validate=True)
            except Exception:
                img_bytes = base64.b64decode(payload_b64)
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))

        if message.text:
            parts.append(types.Part.from_text(text=message.text))

        # Try to use cache if enabled
        cache = self._get_or_create_cache(system_prompt)

        config_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Add thinking_config only for models that support it (Gemini 2.0 Flash Thinking)
        model_lower = self.model.lower()
        if "thinking" in model_lower or "gemini-2.0-flash-thinking" in model_lower:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")

        # If cache is available, use it instead of system_instruction
        if cache is not None:
            config_kwargs["cached_content"] = cache.name
        elif system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        response = self.client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(**config_kwargs),
        )
        print(response)

        # Print cache usage statistics if available
        if hasattr(response, 'usage_metadata') and cache is not None:
            metadata = response.usage_metadata
            if hasattr(metadata, 'cached_content_token_count'):
                print(f"Cache hit! Cached tokens: {metadata.cached_content_token_count}")
                print(f"New tokens: {getattr(metadata, 'prompt_token_count', 0) - metadata.cached_content_token_count}")

        # google-genai returns .text for common text outputs, but fall back to parts
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            if content and getattr(content, "parts", None):
                joined = "".join(
                    [getattr(p, "text", "") for p in content.parts if getattr(p, "text", None)]
                ).strip()
                if joined:
                    return joined

        raise RuntimeError("Empty response from Gemini API")
    
    def send_conversation(self, messages: List[Dict[str, Any]], 
                         system_prompt: Optional[str] = None,
                         temperature: float = 0,
                         max_tokens: int = 4000) -> str:
        """
        Send multi-turn conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Response text
        """
        if self.provider in ["openai", "vllm", "openrouter"]:
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=messages
            )
            return response.content[0].text
        elif self.provider == "google":
            # Minimal support: concatenate user turns and ignore tool/function calls.
            text = "\n".join(
                [
                    f"{m.get('role','user')}: {m.get('content','')}"
                    for m in messages
                    if m.get("content") is not None
                ]
            )
            return self._send_google(MultimodalMessage(text=text), system_prompt, temperature, max_tokens)
    
    def check_connection(self) -> bool:
        """Check if the API connection is working."""
        try:
            message = MultimodalMessage(text="Hello, this is a connection test. Reply with 'OK'.")
            response = self.send_message(message, max_tokens=10)
            return len(response) > 0
        except Exception as e:
            print(f"Connection check failed: {e}")
            return False


# Test function
if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("Multimodal Interface Test")
    print("="*70)
    print()
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    
    print(f"OpenAI API Key: {'✓ Found' if openai_key else '✗ Not found'}")
    print(f"Anthropic API Key: {'✓ Found' if anthropic_key else '✗ Not found'}")
    print(f"OpenAI API Base: {api_base if api_base else 'Default (OpenAI)'}")
    print()
    
    if not openai_key and not anthropic_key and not api_base:
        print("No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file")
        print("Or set OPENAI_API_BASE for vLLM endpoint")
        sys.exit(1)
    
    # Test with available provider
    if openai_key:
        print("Testing OpenAI GPT-4o...")
        interface = MultimodalInterface(model="gpt-4o")
        
        message = MultimodalMessage(text="Hello! Can you describe geometric shapes?")
        response = interface.send_message(message)
        
        print(f"Response: {response[:200]}...")
        print("✓ OpenAI test successful")
    
    elif anthropic_key:
        print("Testing Anthropic Claude 3.5 Sonnet...")
        interface = MultimodalInterface(model="claude-3-5-sonnet-20241022")
        
        message = MultimodalMessage(text="Hello! Can you describe geometric shapes?")
        response = interface.send_message(message)
        
        print(f"Response: {response[:200]}...")
        print("✓ Anthropic test successful")
    
    elif api_base:
        print(f"Testing vLLM at {api_base}...")
        # You need to specify the model name
        print("Note: For vLLM, specify the model name loaded in your server")
        print("Example: python multimodal_interface.py --model Qwen/Qwen2.5-VL-7B-Instruct")
