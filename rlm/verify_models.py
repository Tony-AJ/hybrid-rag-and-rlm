"""Verify that a configured model identifier is really reachable.

No model id is trusted because it looks plausible. This asks the provider which
models the current credential can reach and reports whether the configured id is
among them.

Run:
    python -m rlm.verify_models                       # every provider with a key
    python -m rlm.verify_models --provider anthropic --model <id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlm.providers import PROVIDERS, ProviderUnavailable, build_provider  # noqa: E402


def check(provider_name: str, model: str | None) -> dict:
    try:
        provider = build_provider(provider_name, model=model)
    except ProviderUnavailable as exc:
        return {"provider": provider_name, "status": "unavailable", "detail": str(exc)}
    try:
        available = provider.list_models()
    except Exception as exc:  # network, auth, SDK mismatch
        return {
            "provider": provider_name,
            "status": "list_failed",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    configured = provider.config.model
    return {
        "provider": provider_name,
        "status": "ok" if configured in available else "model_not_available",
        "configured_model": configured,
        "model_available": configured in available,
        "models_visible": len(available),
        "available_models": sorted(available),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    names = [args.provider] if args.provider else sorted(PROVIDERS)
    reports = [check(name, args.model if args.provider else None) for name in names]
    print(json.dumps(reports, indent=2))

    usable = [row for row in reports if row["status"] == "ok" and row["provider"] != "mock"]
    if not usable:
        print(
            "\nNo hosted provider is verified. Set a provider key (and a model id via "
            "ANTHROPIC_MODEL / GOOGLE_MODEL), then re-run. Until then the RLM evaluation "
            "can only run with --provider mock, which measures plumbing and reports no "
            "answer quality.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
