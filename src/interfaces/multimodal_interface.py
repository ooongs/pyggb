#!/usr/bin/env python3
"""
Multimodal Interface for Vision LLMs
Supports GPT-4V, GPT-4o, Claude 3.5 Sonnet, and vLLM models with vision.
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
                 api_base: Optional[str] = None):
        """
        Initialize multimodal interface.
        
        Args:
            model: Model name (gpt-4o, gpt-4-vision-preview, claude-3-5-sonnet-20241022, 
                              or vLLM model name like Qwen/Qwen2.5-VL-7B-Instruct)
            api_key: API key (if None, loads from environment)
            api_base: API base URL for vLLM or custom endpoints
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        
        # Determine provider
        if "claude" in model.lower() or "anthropic" in model.lower():
            self.provider = "anthropic"
            if api_key is None:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            # Default to OpenAI-compatible API (works for GPT and vLLM)
            self.provider = "openai"
            if api_key is None:
                self.api_key = os.getenv("OPENAI_API_KEY")
            
            # Check if this is a vLLM model (typically has / in name)
            if "/" in model and self.api_base:
                self.provider = "vllm"
        
        if not self.api_key:
            # For vLLM, API key might be optional
            if self.provider != "vllm":
                raise ValueError(f"API key not found for {self.provider}")
            else:
                # Use dummy key for vLLM
                self.api_key = "dummy-key"
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate API client."""
        if self.provider in ["openai", "vllm"]:
            try:
                from openai import OpenAI
                
                # Build client args
                client_args = {"api_key": self.api_key}
                if self.api_base:
                    client_args["base_url"] = self.api_base
                
                self.client = OpenAI(**client_args)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
    
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
        if self.provider in ["openai", "vllm"]:
            return self._send_openai(message, system_prompt, temperature, max_tokens)
        elif self.provider == "anthropic":
            return self._send_anthropic(message, system_prompt, temperature, max_tokens)
    
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

            if self.model.startswith("gpt-5"):
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
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        
        return response.content[0].text
    
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
        if self.provider in ["openai", "vllm"]:
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
