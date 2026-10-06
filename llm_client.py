"""
Multi-provider LLM Client for 'Ask My Notes'.
Supports:
- Ollama (Local LLM, 100% private, no API key required)
- Google Gemini (Free tier, fast)
- Groq (Ultra-fast, Llama-3.3-70B)
- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic Claude (Claude 3.5 Sonnet / Haiku)
- Offline Extractive Mode (No API key, works out of the box)
"""

from typing import Generator, Optional, List, Dict, Any
import os
import json
import requests
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM wrapper supporting multiple cloud and local backends.
    """

    SUPPORTED_PROVIDERS = {
        "gemini": "Google Gemini (gemini-2.5-flash / gemini-1.5-flash)",
        "groq": "Groq Cloud (llama-3.3-70b-versatile)",
        "ollama": "Local Ollama (llama3.2, mistral, deepseek-r1)",
        "openai": "OpenAI (gpt-4o-mini / gpt-4o)",
        "claude": "Anthropic Claude (claude-3-5-haiku / sonnet)",
        "offline": "Local Extractive Mode (No API key / 100% Offline)",
    }

    DEFAULT_MODELS = {
        "gemini": "gemini-2.5-flash",
        "groq": "llama-3.3-70b-versatile",
        "ollama": "llama3.2",
        "openai": "gpt-4o-mini",
        "claude": "claude-3-5-haiku-20241022",
        "offline": "local-extractive",
    }

    def __init__(
        self,
        provider: str = "offline",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
    ):
        self.provider = provider.lower()
        self.api_key = (api_key or os.environ.get(f"{self.provider.upper()}_API_KEY", "")).strip()
        self.model_name = model_name or self.DEFAULT_MODELS.get(self.provider, "default")
        self.ollama_url = ollama_url.rstrip("/")

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generates a non-streaming completion for the given prompt.
        """
        full_response = ""
        for chunk in self.stream_generate(prompt, system_prompt):
            full_response += chunk
        return full_response

    def stream_generate(self, prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
        """
        Streams completion text chunks.
        """
        if self.provider == "offline":
            yield from self._stream_offline(prompt)
        elif self.provider == "ollama":
            yield from self._stream_ollama(prompt, system_prompt)
        elif self.provider == "gemini":
            yield from self._stream_gemini(prompt, system_prompt)
        elif self.provider == "groq":
            yield from self._stream_groq(prompt, system_prompt)
        elif self.provider == "openai":
            yield from self._stream_openai(prompt, system_prompt)
        elif self.provider == "claude":
            yield from self._stream_claude(prompt, system_prompt)
        else:
            yield f"Error: Unsupported provider '{self.provider}'."

    # -------------------------------------------------------------
    # Provider Implementations
    # -------------------------------------------------------------

    def _stream_ollama(self, prompt: str, system_prompt: str) -> Generator[str, None, None]:
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }

        try:
            with requests.post(url, json=payload, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    yield f"Ollama Error ({resp.status_code}): {resp.text}\nPlease check if model '{self.model_name}' is installed via `ollama pull {self.model_name}`."
                    return
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line.decode("utf-8"))
                        msg = data.get("message", {}).get("content", "")
                        if msg:
                            yield msg
        except requests.exceptions.ConnectionError:
            yield (
                "⚠️ **Could not connect to Ollama.**\n\n"
                "Please make sure Ollama is installed and running on your system:\n"
                "1. Download and run Ollama from [ollama.ai](https://ollama.ai).\n"
                f"2. Pull the model: `ollama run {self.model_name}`\n"
                "3. Or select another provider (Groq, Gemini, OpenAI, Claude) from the sidebar."
            )
        except Exception as e:
            yield f"Error calling Ollama: {str(e)}"

    def _stream_gemini(self, prompt: str, system_prompt: str) -> Generator[str, None, None]:
        if not self.api_key:
            yield "⚠️ **Gemini API Key missing.** Please enter your Google Gemini API key in the sidebar."
            return

        # Use Google Gemini's OpenAI-compatible endpoint
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }

        yield from self._stream_openai_compatible(url, headers, payload)

    def _stream_groq(self, prompt: str, system_prompt: str) -> Generator[str, None, None]:
        if not self.api_key:
            yield "⚠️ **Groq API Key missing.** Please enter your Groq API key (starts with `gsk_`) in the sidebar."
            return

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }

        yield from self._stream_openai_compatible(url, headers, payload)

    def _stream_openai(self, prompt: str, system_prompt: str) -> Generator[str, None, None]:
        if not self.api_key:
            yield "⚠️ **OpenAI API Key missing.** Please enter your OpenAI API key in the sidebar."
            return

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
        }

        yield from self._stream_openai_compatible(url, headers, payload)

    def _stream_openai_compatible(
        self, url: str, headers: dict, payload: dict
    ) -> Generator[str, None, None]:
        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    yield f"API Error ({resp.status_code}): {resp.text}"
                    return

                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode("utf-8").strip()
                        if line_str.startswith("data: "):
                            data_str = line_str[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            yield f"Connection error: {str(e)}"

    def _stream_claude(self, prompt: str, system_prompt: str) -> Generator[str, None, None]:
        if not self.api_key:
            yield "⚠️ **Anthropic Claude API Key missing.** Please enter your Claude API key in the sidebar."
            return

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    yield f"Claude API Error ({resp.status_code}): {resp.text}"
                    return

                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode("utf-8").strip()
                        if line_str.startswith("data: "):
                            data_str = line_str[6:].strip()
                            try:
                                event = json.loads(data_str)
                                if event.get("type") == "content_block_delta":
                                    text = event.get("delta", {}).get("text", "")
                                    if text:
                                        yield text
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            yield f"Error contacting Claude API: {str(e)}"

    def _stream_offline(self, prompt: str) -> Generator[str, None, None]:
        """
        Offline fallback mode:
        Performs direct extractive synthesis over the context passed in the prompt,
        highlighting the retrieved excerpts and their exact page locations.
        """
        lines = [
            "### 📖 Answer (Local Offline Extractive Mode)\n\n",
            "*(Operating in 100% offline mode. For full generative reasoning, you can enter a free Groq/Gemini key or run Ollama locally in the sidebar.)*\n\n",
            "Here is the synthesized excerpt directly retrieved from your notes matching your question:\n\n",
        ]
        for line in lines:
            yield line

        # Parse context from prompt if formatted by rag_chain
        if "=== RETRIEVED NOTES CONTEXT ===" in prompt:
            context_part = prompt.split("=== RETRIEVED NOTES CONTEXT ===")[1]
            if "=== END CONTEXT ===" in context_part:
                context_part = context_part.split("=== END CONTEXT ===")[0]

            yield context_part.strip() + "\n\n"
        else:
            yield prompt
