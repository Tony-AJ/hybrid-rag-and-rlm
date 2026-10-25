"""Provider-independent text generation: `generate(prompt) -> str`.

Design rules
------------
1. No model identifier is hard-coded as known-good. A provider must list the
   models its credentials can actually reach (`list_models`) and
   `verify_model` fails loudly if the configured id is not in that list. The id
   comes from configuration or an environment variable, never from a guess.
2. Every call returns a `GenerationResult` carrying the exact model id the
   provider reported, token counts and latency, so an experiment can record what
   actually ran rather than what was requested.
3. `MockProvider` exists to test the loop's plumbing offline and deterministically.
   It performs no reasoning and must never be used to report answer quality.

Known at the time of writing: no ANTHROPIC_API_KEY is present in this
environment, so `AnthropicProvider` cannot be verified here. The verification
step is implemented and will run as soon as a key exists; until then any
experiment must either use the mock provider or stop with a clear error.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv_keys() -> Dict[str, str]:
    """Read .env without overwriting real process environment variables."""
    env_path = ROOT / ".env"
    values: Dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def resolve_key(name: str) -> Optional[str]:
    return os.environ.get(name) or load_dotenv_keys().get(name)


def prompt_template_sha256(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GenerationConfig:
    """Everything that must be recorded for a generation run to be reproducible."""
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    stop_sequences: tuple = ()

    def as_record(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop_sequences": list(self.stop_sequences),
        }


@dataclass
class GenerationResult:
    text: str
    model: str
    provider: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: float = 0.0
    response_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class GenerationProvider(Protocol):
    """The whole contract the RLM loop depends on."""

    config: GenerationConfig

    def list_models(self) -> List[str]: ...

    def verify_model(self) -> Dict[str, Any]: ...

    def generate(self, prompt: str) -> GenerationResult: ...


class ProviderUnavailable(RuntimeError):
    """No usable credential, SDK or model. Never silently degraded to a mock."""


class QuotaExceeded(RuntimeError):
    """The provider refused the call for rate or quota reasons.

    Kept distinct from a generic failure because the correct response differs:
    a quota stop means the run is incomplete, not that the system is broken, so
    the evaluator saves partial results and says how far it got instead of
    reporting a score over fewer queries as if it were the whole set.
    """

    def __init__(self, provider: str, detail: str) -> None:
        super().__init__(f"{provider} quota or rate limit reached: {detail}")
        self.provider = provider
        self.detail = detail


def _is_quota_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"ResourceExhausted", "RateLimitError", "TooManyRequests"}:
        return True
    text = str(exc).lower()
    return "429" in text or "quota" in text or "rate limit" in text


RATE_LIMIT_MAX_WAITS = 6


def _rate_limit_wait_seconds(exc: BaseException) -> Optional[float]:
    """Seconds to wait before retrying a per-minute rate limit, or None.

    A per-minute limit (free-tier gemini-2.5-flash allows 5 requests a minute;
    Groq reports "tokens per minute (TPM)") clears by itself, so the call is
    retried after the delay the provider asks for. A per-day quota (Google
    "PerDay", Groq "per day (TPD)") does not clear within a run and stays a
    hard stop.
    """
    text = str(exc)
    if "PerDay" in text or "per day" in text:
        return None
    if "PerMinute" not in text and "per minute" not in text:
        return None
    google = re.search(r"retry in ([\d.]+)s", text) or re.search(r"seconds:\s*(\d+)", text)
    if google:
        return float(google.group(1)) + 1.0
    # Groq: "Please try again in 1m17.76s" or "in 7.66s" or "in 450ms".
    groq = re.search(r"try again in (?:(\d+)m(?!s))?(?:([\d.]+)(ms|s))?", text)
    if groq and (groq.group(1) or groq.group(2)):
        seconds = float(groq.group(1) or 0) * 60
        if groq.group(2):
            seconds += float(groq.group(2)) / (1000 if groq.group(3) == "ms" else 1)
        return seconds + 1.0
    return 60.0


# ================================ Anthropic =================================


class AnthropicProvider:
    """Hosted Claude via the official `anthropic` SDK.

    The model id is supplied by the caller or ANTHROPIC_MODEL. It is checked
    against `client.models.list()` before any generation call, so a wrong or
    retired id fails with a list of what is actually available instead of
    producing a confusing 404 mid-experiment.
    """

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1024):
        api_key = resolve_key("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderUnavailable(
                "ANTHROPIC_API_KEY is not set in the environment or .env. Set it, "
                "or run the experiment with --provider mock, which measures "
                "plumbing only and reports no answer quality."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderUnavailable("the `anthropic` package is not installed") from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        model_id = model or os.environ.get("ANTHROPIC_MODEL") or ""
        self.config = GenerationConfig(
            provider="anthropic", model=model_id, temperature=temperature, max_tokens=max_tokens
        )

    def list_models(self) -> List[str]:
        return [model.id for model in self._client.models.list(limit=100).data]

    def verify_model(self) -> Dict[str, Any]:
        available = self.list_models()
        if not self.config.model:
            raise ProviderUnavailable(
                "No model id given. Pass --model or set ANTHROPIC_MODEL. Ids this key can "
                f"reach: {available}. Model ids are never assumed."
            )
        if self.config.model not in available:
            raise ProviderUnavailable(
                f"Model {self.config.model!r} is not available to this key. "
                f"Available: {available}"
            )
        return {"provider": "anthropic", "model": self.config.model, "verified": True,
                "models_visible": len(available)}

    def generate(self, prompt: str) -> GenerationResult:
        if not self.config.model:
            raise ProviderUnavailable("No model id configured; run verify_model() first.")
        started = time.perf_counter()
        try:
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
                **({"stop_sequences": list(self.config.stop_sequences)} if self.config.stop_sequences else {}),
            )
        except Exception as exc:
            if _is_quota_error(exc):
                raise QuotaExceeded("anthropic", str(exc)) from exc
            raise RuntimeError(f"Anthropic generation failed for a {len(prompt)}-char prompt") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        return GenerationResult(
            text=text,
            model=response.model,
            provider="anthropic",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=round(latency_ms, 3),
            response_id=response.id,
            extra={"stop_reason": response.stop_reason},
        )


# ================================= Google ===================================


class GoogleProvider:
    """Hosted Gemini, present only because this repo already carries a GOOGLE_API_KEY.

    Same verification rule: the model id must appear in the provider's own list.
    """

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1024):
        api_key = resolve_key("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderUnavailable("GOOGLE_API_KEY is not set in the environment or .env")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ProviderUnavailable("the `google-generativeai` package is not installed") from exc
        genai.configure(api_key=api_key)
        self._genai = genai
        model_id = model or os.environ.get("GOOGLE_MODEL") or ""
        self.config = GenerationConfig(
            provider="google", model=model_id, temperature=temperature, max_tokens=max_tokens
        )

    def list_models(self) -> List[str]:
        return [
            model.name.split("/", 1)[-1]
            for model in self._genai.list_models()
            if "generateContent" in getattr(model, "supported_generation_methods", [])
        ]

    def verify_model(self) -> Dict[str, Any]:
        available = self.list_models()
        if not self.config.model:
            raise ProviderUnavailable(
                "No model id given. Pass --model or set GOOGLE_MODEL. Ids this key can reach: "
                f"{available}."
            )
        if self.config.model not in available:
            raise ProviderUnavailable(
                f"Model {self.config.model!r} is not available to this key. Available: {available}"
            )
        return {"provider": "google", "model": self.config.model, "verified": True,
                "models_visible": len(available)}

    def generate(self, prompt: str) -> GenerationResult:
        if not self.config.model:
            raise ProviderUnavailable("No model id configured; run verify_model() first.")
        model = self._genai.GenerativeModel(self.config.model)
        for attempt in range(RATE_LIMIT_MAX_WAITS + 1):
            # Timed per attempt so a rate-limit wait never inflates recorded latency.
            started = time.perf_counter()
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.config.temperature,
                        "max_output_tokens": self.config.max_tokens,
                    },
                )
                break
            except Exception as exc:
                if not _is_quota_error(exc):
                    raise RuntimeError(f"Google generation failed for a {len(prompt)}-char prompt") from exc
                wait = _rate_limit_wait_seconds(exc)
                if wait is None or attempt == RATE_LIMIT_MAX_WAITS:
                    raise QuotaExceeded("google", str(exc)) from exc
                print(f"  google per-minute limit hit; waiting {wait:.0f}s "
                      f"(retry {attempt + 1}/{RATE_LIMIT_MAX_WAITS})", flush=True)
                time.sleep(wait)
        latency_ms = (time.perf_counter() - started) * 1000
        usage = getattr(response, "usage_metadata", None)
        return GenerationResult(
            text=response.text,
            model=self.config.model,
            provider="google",
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            latency_ms=round(latency_ms, 3),
        )


# ================================== Mock ====================================


class MockProvider:
    """Deterministic offline stand-in. Tests plumbing, never quality.

    It reads the evidence block out of the prompt and answers with the entity
    codes it finds there, in document order. That is enough to exercise the
    loop, the token accounting and the evaluation path without a network call,
    and it deliberately has no reasoning ability: any answer-quality number
    produced with this provider is meaningless and the evaluator labels it as such.
    """

    EVALUATIVE = False

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1024):
        self.config = GenerationConfig(
            provider="mock",
            model=model or "mock-deterministic-1",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def list_models(self) -> List[str]:
        return [self.config.model]

    def verify_model(self) -> Dict[str, Any]:
        return {"provider": "mock", "model": self.config.model, "verified": True, "evaluative": False}

    def generate(self, prompt: str) -> GenerationResult:
        started = time.perf_counter()
        codes: List[str] = []
        for match in re.finditer(r"\b(MP-\d{3}|PL-[A-L]|[A-Z]{2}-\d{1,3})\b", prompt):
            code = match.group(1)
            if code not in codes:
                codes.append(code)
        if "DECIDE:" in prompt:
            # The loop asks whether to retrieve again; the mock always stops,
            # which keeps step counts deterministic.
            text = "STOP"
        else:
            text = ", ".join(codes[:6]) if codes else "no evidence found"
        latency_ms = (time.perf_counter() - started) * 1000
        return GenerationResult(
            text=text,
            model=self.config.model,
            provider="mock",
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
            latency_ms=round(latency_ms, 3),
        )


# =================================== Groq ===================================


class GroqProvider:
    """Hosted Groq through its OpenAI-compatible endpoint, via the `openai` SDK.

    Same verification rule: the model id must appear in the provider's own list.
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1024):
        api_key = resolve_key("GROQ_API_KEY")
        if not api_key:
            raise ProviderUnavailable("GROQ_API_KEY is not set in the environment or .env")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderUnavailable("the `openai` package is not installed") from exc
        self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)
        model_id = model or os.environ.get("GROQ_MODEL") or ""
        self.config = GenerationConfig(
            provider="groq", model=model_id, temperature=temperature, max_tokens=max_tokens
        )

    def list_models(self) -> List[str]:
        return [model.id for model in self._client.models.list().data]

    def verify_model(self) -> Dict[str, Any]:
        available = self.list_models()
        if not self.config.model:
            raise ProviderUnavailable(
                "No model id given. Pass --model or set GROQ_MODEL. Ids this key can reach: "
                f"{available}."
            )
        if self.config.model not in available:
            raise ProviderUnavailable(
                f"Model {self.config.model!r} is not available to this key. Available: {available}"
            )
        return {"provider": "groq", "model": self.config.model, "verified": True,
                "models_visible": len(available)}

    def generate(self, prompt: str) -> GenerationResult:
        if not self.config.model:
            raise ProviderUnavailable("No model id configured; run verify_model() first.")
        for attempt in range(RATE_LIMIT_MAX_WAITS + 1):
            # Timed per attempt so a rate-limit wait never inflates recorded latency.
            started = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as exc:
                if not _is_quota_error(exc):
                    raise RuntimeError(f"Groq generation failed for a {len(prompt)}-char prompt") from exc
                wait = _rate_limit_wait_seconds(exc)
                if wait is None or attempt == RATE_LIMIT_MAX_WAITS:
                    raise QuotaExceeded("groq", str(exc)) from exc
                print(f"  groq per-minute limit hit; waiting {wait:.0f}s "
                      f"(retry {attempt + 1}/{RATE_LIMIT_MAX_WAITS})", flush=True)
                time.sleep(wait)
        latency_ms = (time.perf_counter() - started) * 1000
        usage = response.usage
        return GenerationResult(
            text=response.choices[0].message.content or "",
            model=response.model,
            provider="groq",
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=round(latency_ms, 3),
        )


PROVIDERS = {
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "groq": GroqProvider,
    "mock": MockProvider,
}


def build_provider(
    name: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> GenerationProvider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}. Known: {sorted(PROVIDERS)}")
    return PROVIDERS[name](model=model, temperature=temperature, max_tokens=max_tokens)
