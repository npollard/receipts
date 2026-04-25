"""OpenAI chat model construction for parser infrastructure."""

import os

from langchain_openai import ChatOpenAI


def create_openai_chat_llm(model_name: str, temperature: float):
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
