"""
Language model service for patient simulation.

Provides LLM integration for generating patient responses during voice interviews.
"""

from typing import Dict, List

import aiohttp


class OpenRouterChat:
    """
    OpenRouter LLM integration for patient simulation.

    Uses OpenRouter's API to generate patient responses based on conversation
    history and system prompts.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "anthropic/claude-haiku-4.5",
    ):
        self.api_key = api_key
        self.url = f"{base_url}/chat/completions"
        self.model = model

    async def chat(self, messages: List[Dict], system_prompt: str) -> str:
        payload = {
            "model": self.model,
            "reasoning": {"enabled": False},
            "messages": [],
        }

        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})

        payload["messages"].extend(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data["choices"][0]["message"]["content"]
