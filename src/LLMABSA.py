import json
import re
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests

from src.base import ABSAAnalyzer, AspectSentiment


def _first_json_block(text: str) -> Optional[str]:
    """
    Extract the first JSON array/object from a string.
    """
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Fallback: greedy array/object
    m = re.search(r"(\[[\s\S]*\])", text)
    if m:
        return m.group(1).strip()

    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        return m.group(1).strip()

    return None


def _normalize_sentiment(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"pos", "positive", "favor", "favorable", "good"}:
        return "positive"
    if s in {"neg", "negative", "unfavorable", "bad"}:
        return "negative"
    # everything else becomes neutral
    return "neutral"


def _clip_confidence(x: Optional[float]) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _find_first_span(text: str, sub: str) -> List[int]:
    """
    Find first (start, end) for sub in text. If not found, return [0, 0].
    """
    if not sub:
        return [0, 0]
    i = text.lower().find(sub.lower())
    if i == -1:
        return [0, 0]
    return [i, i + len(sub)]


class LLMABSA(ABSAAnalyzer):
    """
    Aspect-Based Sentiment Analysis via a local LLM served by Ollama.
    - Uses /api/chat endpoint
    - Few-shot prompt with strict JSON output schema
    - Robust JSON parsing + validation
    """

    def __init__(
        self,
        model: str = "llama3.1",
        temperature: float = 0.2,
        max_retries: int = 2,
        timeout_s: int = 60,
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.timeout_s = int(timeout_s)
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ----------------------------- Public API -----------------------------

    def analyze(self, text: str) -> List[AspectSentiment]:
        """
        Call Ollama locally and return aspect-sentiment pairs.
        Always returns a list (possibly empty).
        """
        messages = self._build_messages(text)
        raw = self._call_ollama(messages)

        if raw is None:
            return []

        data = self._parse_llm_json(raw, original_text=text)
        return data


    def _build_messages(self, text: str) -> List[Dict[str, str]]:

        system = (
            "You are an ABSA (Aspect-Based Sentiment Analysis) extractor. "
            "Read the input review and produce ONLY valid JSON matching the schema:\n"
            "[{\"aspect\": str, \"sentiment\": \"positive|negative|neutral\", "
            "\"confidence\": number between 0 and 1, \"text_span\": [start, end]}].\n"
            "Rules:\n"
            "- Aspects are concrete entities/features from the text (e.g., 'service', 'ice cream', 'coffee', 'cupcake').\n"
            "- Sentiment is the stance toward EACH aspect only.\n"
            "- confidence reflects your certainty in [0..1].\n"
            "- text_span uses 0-based character offsets [start, end) locating the FIRST mention of the aspect in the original text.\n"
            "- If nothing relevant is found, return an empty JSON array [].\n"
            "- Do NOT include any commentary. Output JSON only."
        )

        ex1_user = (
            "The pizza was delicious but the service was terrible. "
            "The ice cream was just okay."
        )
        ex1_assistant = json.dumps(
            [
                {
                    "aspect": "pizza",
                    "sentiment": "positive",
                    "confidence": 0.85,
                    "text_span": [4, 9],
                },
                {
                    "aspect": "service",
                    "sentiment": "negative",
                    "confidence": 0.88,
                    "text_span": [37, 44],
                },
                {
                    "aspect": "ice cream",
                    "sentiment": "neutral",
                    "confidence": 0.55,
                    "text_span": [55, 64],
                },
            ],
            ensure_ascii=False,
        )

        ex2_user = (
            "Loved the affagato! The line was too long, but staff were friendly."
        )
        ex2_assistant = json.dumps(
            [
                {
                    "aspect": "affogato",
                    "sentiment": "positive",
                    "confidence": 0.82,
                    "text_span": [10, 18],
                },
                {
                    "aspect": "line",
                    "sentiment": "negative",
                    "confidence": 0.75,
                    "text_span": [25, 29],
                },
                {
                    "aspect": "staff",
                    "sentiment": "positive",
                    "confidence": 0.73,
                    "text_span": [52, 57],
                },
            ],
            ensure_ascii=False,
        )

        user = (
            "Analyze the following review and return ONLY JSON with the schema above.\n\n"
            f"REVIEW:\n{text}"
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": ex1_user},
            {"role": "assistant", "content": ex1_assistant},
            {"role": "user", "content": ex2_user},
            {"role": "assistant", "content": ex2_assistant},
            {"role": "user", "content": user},
        ]

    def _call_ollama(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        POST to Ollama /api/chat with retries and timeouts.
        Returns model text response (string) or None on hard failure.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": 4096,
            },
        }

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(
                    url, json=payload, timeout=self.timeout_s
                )
                resp.raise_for_status()
                data = resp.json()
                # /api/chat returns {'message': {'content': '...'}}
                content = data.get("message", {}).get("content", "")
                if content:
                    return content
                last_err = RuntimeError("Empty content from Ollama.")
            except requests.RequestException as e:
                last_err = e
            # small backoff before retry
            time.sleep(0.4 * (attempt + 1))

        # If we got here: failed
        print(f"[LLMABSA] Ollama call failed: {last_err}")
        return None

    def _parse_llm_json(self, llm_text: str, original_text: str) -> List[AspectSentiment]:
        """
        Parse LLM's output to a list of AspectSentiment.
        - Tolerates extra prose by extracting the first JSON block
        - Validates fields and repairs missing spans/confidence
        - Deduplicates identical aspects (case-insensitive)
        """
        raw = _first_json_block(llm_text) or "[]"

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            fixed = raw.replace("“", '"').replace("”", '"').replace("’", "'")
            fixed = re.sub(r",\s*([\]\}])", r"\1", fixed)
            try:
                data = json.loads(fixed)
            except Exception:
                print("[LLMABSA] Failed to parse JSON from LLM output.")
                return []

        # Ensure list
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []

        results: List[AspectSentiment] = []
        seen = set()

        for item in data:
            if not isinstance(item, dict):
                continue
            aspect = (item.get("aspect") or "").strip()
            if not aspect:
                continue

            sentiment = _normalize_sentiment(item.get("sentiment"))
            confidence = _clip_confidence(item.get("confidence"))

            span = item.get("text_span")
            if (
                isinstance(span, list)
                and len(span) == 2
                and all(isinstance(n, int) for n in span)
            ):
                text_span = span
            else:
                text_span = _find_first_span(original_text, aspect)

            key = aspect.lower()
            if key in seen:
                # keep the higher-confidence one
                for i, r in enumerate(results):
                    if r.aspect.lower() == key and confidence > r.confidence:
                        results[i] = AspectSentiment(
                            aspect=aspect,
                            sentiment=sentiment,
                            confidence=confidence,
                            text_span=text_span,
                        )
                continue

            results.append(
                AspectSentiment(
                    aspect=aspect,
                    sentiment=sentiment,
                    confidence=confidence,
                    text_span=text_span,
                )
            )
            seen.add(key)

        return results
