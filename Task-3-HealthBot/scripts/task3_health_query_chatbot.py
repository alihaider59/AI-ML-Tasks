from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


SYSTEM_PROMPT = (
    "You are a helpful medical assistant for general health education only. "
    "Use simple language and a calm, supportive tone. "
    "Never provide diagnosis, prescriptions, exact medication doses, or treatment plans. "
    "For symptom questions, provide general causes and safe self-care suggestions. "
    "Always include red-flag warning signs and when to seek urgent care. "
    "End every response with a short safety reminder to consult a licensed doctor."
)

HF_FALLBACK_MODELS = [
    "gpt2",
    "distilgpt2"
]
GROQ_FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

HF_ENDPOINTS = [
    "https://router.huggingface.co/hf-inference/models/{model}",
]

SAFETY_PATTERNS = [
    r"\b(suicide|kill myself|self harm|overdose|poison)\b",
    r"\b(chest pain|can'?t breathe|difficulty breathing|stroke)\b",
    r"\b(dosage|exact dose|prescribe)\b",
]

INTENT_PATTERNS = {
    "symptom": [
        r"\b(pain|fever|cough|cold|sore throat|headache|nausea|vomit|rash|dizzy|fatigue)\b",
    ],
    "medication": [
        r"\b(paracetamol|ibuprofen|medicine|drug|tablet|antibiotic|medication)\b",
    ],
    "prevention": [
        r"\b(prevent|prevention|avoid|get rid of|healthy|diet|exercise|lifestyle|sleep)\b",
    ],
}
SMALL_TALK_PATTERNS = [
    r"^(hi|hey|hello|hy)\b",
    r"\b(how are you|what's up|whats up)\b",
    r"^(ok|okay|thanks|thank you)\b",
]
GENERAL_HEALTH_HINTS = [
    "health",
    "symptom",
    "sick",
    "ill",
    "disease",
    "pain",
    "fever",
    "cough",
    "medicine",
    "doctor",
    "treatment",
    "prevent",
    "diet",
    "exercise",
]


class SafetyFilter:
    @staticmethod
    def check_query(text: str) -> Optional[str]:
        text = text.lower()
        for p in SAFETY_PATTERNS:
            if re.search(p, text):
                return (
                    "I cannot help with that request. "
                    "Please consult a qualified healthcare professional immediately."
                )
        return None


def build_user_prompt(user_text: str) -> str:
    """Wrap raw user text with response-quality instructions."""
    return (
        "Answer the following health query in this exact structure:\n"
        "1) Brief answer (1-2 lines)\n"
        "2) Possible common reasons (bullet points)\n"
        "3) Safe next steps at home (bullet points)\n"
        "4) Red flags: when to seek urgent care (bullet points)\n"
        "Keep it concise and non-diagnostic.\n\n"
        f"User query: {user_text}"
    )


def detect_intent(user_text: str) -> str:
    text = user_text.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return intent
    return "general"


def is_small_talk(user_text: str) -> bool:
    text = user_text.lower().strip()
    return any(re.search(pattern, text) for pattern in SMALL_TALK_PATTERNS)


def is_health_related(user_text: str) -> bool:
    text = user_text.lower()
    if detect_intent(user_text) != "general":
        return True
    return any(hint in text for hint in GENERAL_HEALTH_HINTS)


def build_intent_prompt(user_text: str, intent: str) -> str:
    if intent == "symptom":
        return (
            "Answer as general health education for symptom-related question.\n"
            "Use structure:\n"
            "1) What it might commonly be (non-diagnostic)\n"
            "2) Safe self-care steps\n"
            "3) Red flags requiring urgent care\n"
            "Keep concise and clear.\n\n"
            f"User query: {user_text}"
        )
    if intent == "medication":
        return (
            "Answer as medicine-safety education only.\n"
            "Do NOT provide exact dosage or prescription advice.\n"
            "Use structure:\n"
            "1) General safety info\n"
            "2) Who should be careful\n"
            "3) When to consult doctor urgently\n"
            "Keep concise and clear.\n\n"
            f"User query: {user_text}"
        )
    if intent == "prevention":
        return (
            "Answer as prevention-focused health guidance.\n"
            "Use structure:\n"
            "1) Key prevention habits\n"
            "2) Practical daily routine tips\n"
            "3) When medical check-up is needed\n"
            "Keep concise and clear.\n\n"
            f"User query: {user_text}"
        )
    return build_user_prompt(user_text)


@dataclass
class ChatConfig:
    provider: str
    openai_model: str
    groq_model: str
    hf_model: str
    save_transcript: bool


class HealthChatbot:
    def __init__(self, config: ChatConfig):
        self.config = config
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

        self.project_root = Path(__file__).resolve().parents[2]
        self.outputs_dir = self.project_root / "Task-3-HealthBot" / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- OPENAI ----------------
    def _openai_response(self, msg: str) -> str:
        if OpenAI is None:
            raise RuntimeError("OpenAI not installed")

        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY")

        client = OpenAI(api_key=key)

        messages = self.history + [{"role": "user", "content": msg}]
        res = client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages
        )

        return res.choices[0].message.content

    # ---------------- GROQ ----------------
    def _groq_response(self, msg: str) -> str:
        if OpenAI is None:
            raise RuntimeError("OpenAI-compatible client not installed")

        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Missing GROQ_API_KEY in Task-3-HealthBot/.env")

        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

        messages = self.history + [{"role": "user", "content": msg}]
        candidate_models = [self.config.groq_model] + [
            m for m in GROQ_FALLBACK_MODELS if m != self.config.groq_model
        ]
        last_error: Optional[str] = None

        for model_id in candidate_models:
            try:
                res = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=250,
                )
                return res.choices[0].message.content
            except Exception as exc:
                last_error = f"{model_id}: {exc}"
                continue

        raise RuntimeError(f"Groq failed for all models. Last: {last_error}")

    # ---------------- HF FIXED ----------------
    def _hf_request(self, model: str, prompt: str, headers: dict) -> tuple[Optional[str], Optional[str]]:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 120,
                "temperature": 0.5,
                "return_full_text": False
            }
        }

        last_error: Optional[str] = None
        errors: list[str] = []

        for endpoint in HF_ENDPOINTS:
            url = endpoint.format(model=model)
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=60)

                if r.status_code != 200:
                    body = r.text[:220].replace("\n", " ")
                    last_error = f"{model} -> {r.status_code} on {url}: {body}"
                    errors.append(last_error)
                    continue

                data = r.json()

                # list response
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    return data[0]["generated_text"], None

                # dict response
                if isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"], None

                # HF error format
                if isinstance(data, dict) and data.get("error"):
                    last_error = f"{model} -> {data['error']}"
                    errors.append(last_error)
                    continue
            except Exception as exc:
                last_error = f"{model} -> {type(exc).__name__}: {exc}"
                errors.append(last_error)
                continue

        if errors:
            return None, " | ".join(errors)
        return None, last_error

    def _huggingface_response(self, msg: str) -> str:
        key = os.getenv("HUGGINGFACE_API_KEY")
        if not key:
            raise RuntimeError("Missing HUGGINGFACE_API_KEY")

        headers = {"Authorization": f"Bearer {key}"}

        prompt = f"{SYSTEM_PROMPT}\nUser: {msg}\nAssistant:"

        candidates = [self.config.hf_model] + HF_FALLBACK_MODELS

        last_error = None

        for model in candidates:
            result, err = self._hf_request(model, prompt, headers)
            if result:
                return result.strip()
            last_error = err or f"{model} failed"

        raise RuntimeError(f"HF failed. Last: {last_error}")

    # ---------------- MAIN ----------------
    def ask(self, msg: str) -> str:
        blocked = SafetyFilter.check_query(msg)
        if blocked:
            return blocked

        # Keep non-health interactions short and natural.
        if is_small_talk(msg):
            reply = "Hey! I am here to help with your health questions. What would you like to ask?"
            self.history.append({"role": "user", "content": msg})
            self.history.append({"role": "assistant", "content": reply})
            return reply

        if not is_health_related(msg):
            reply = (
                "I am focused on health-related help. "
                "Ask me about symptoms, prevention, medicine safety, or healthy habits."
            )
            self.history.append({"role": "user", "content": msg})
            self.history.append({"role": "assistant", "content": reply})
            return reply

        intent = detect_intent(msg)
        prompted_msg = build_intent_prompt(msg, intent)

        if self.config.provider == "openai":
            reply = self._openai_response(prompted_msg)
        elif self.config.provider == "groq":
            reply = self._groq_response(prompted_msg)
        else:
            reply = self._huggingface_response(prompted_msg)

        if "urgent care" not in reply.lower() and "emergency" not in reply.lower():
            reply += "\n\nRed flags: If symptoms become severe, seek urgent medical care."
        if "consult a licensed doctor" not in reply.lower():
            reply += "\n\nNote: Consult a licensed doctor for medical advice."

        self.history.append({"role": "user", "content": msg})
        self.history.append({"role": "assistant", "content": reply})

        return reply

    def save(self):
        path = self.outputs_dir / f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(self.history, indent=2))
        return path


# ---------------- SETUP ----------------
def load_env():
    root = Path(__file__).resolve().parents[2]
    task_env = root / "Task-3-HealthBot" / ".env"
    root_env = root / ".env"
    if task_env.exists():
        load_dotenv(task_env, override=True)
    else:
        load_dotenv(root_env, override=True)


def main():
    load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=["groq", "openai", "huggingface"],
        default=os.getenv("HEALTH_CHAT_PROVIDER", "groq"),
    )
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    parser.add_argument("--groq-model", default=os.getenv("GROQ_MODEL", "llama3-8b-8192"))
    parser.add_argument("--hf-model", default=os.getenv("HF_MODEL", "gpt2"))
    parser.add_argument("--self-test", action="store_true")

    args = parser.parse_args()

    if args.self_test:
        print("Running self-test for safety filters...")
        tests = [
            "What causes a sore throat?",
            "Is paracetamol safe for children?",
            "Tell me exact dosage for my child",
        ]
        for q in tests:
            blocked = SafetyFilter.check_query(q)
            status = "BLOCKED" if blocked else "ALLOWED"
            print(f"- {status}: {q}")
        print("Self-test completed.")
        return

    bot = HealthChatbot(
        ChatConfig(
            provider=args.provider,
            openai_model=args.openai_model,
            groq_model=args.groq_model,
            hf_model=args.hf_model,
            save_transcript=True
        )
    )

    print("\nHealth Chatbot Ready\nType 'exit' to stop\n")

    while True:
        q = input("You: ").strip()
        if q == "exit":
            break

        try:
            print("\nBot:", bot.ask(q), "\n")
        except Exception as e:
            print("\nBot Error:", e, "\n")

    path = bot.save()
    print("Saved:", path)


if __name__ == "__main__":
    main()