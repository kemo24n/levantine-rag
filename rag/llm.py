"""
three free ways to get an llm, pick with LLM_PROVIDER in .env:
  gemini -> free tier from aistudio.google.com
  groq   -> free tier, very fast
  ollama -> runs on my laptop, no key, slower
  none   -> no llm, just returns the best passage (good for testing)

model names change a lot, so they're env vars too
"""
import os

import requests

TIMEOUT = 60


def load_env(path=".env"):
    # tiny .env reader so i dont need python-dotenv for 5 lines
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


def ask_gemini(system, prompt):
    key = os.environ["GEMINI_API_KEY"]
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }
    r = requests.post(url, params={"key": key}, json=body, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def ask_groq(system, prompt):
    key = os.environ["GROQ_API_KEY"]
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "temperature": 0.2,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def ask_ollama(system, prompt):
    # qwen2.5 is the best small model i found for arabic, llama3 mixes english in
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    r = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        },
        timeout=TIMEOUT * 3,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


PROVIDERS = {"gemini": ask_gemini, "groq": ask_groq, "ollama": ask_ollama}


def get_llm(name=None):
    name = (name or os.getenv("LLM_PROVIDER", "none")).lower()
    if name == "none":
        return None
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER should be one of {list(PROVIDERS)} or none, got '{name}'")
    return PROVIDERS[name]
