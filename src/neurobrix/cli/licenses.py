"""
License definitions and acceptance management for NeuroBrix models.

Handles license display in terminal and acceptance tracking.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


# ~/.neurobrix/license_acceptances.json
_ACCEPTANCES_FILE = Path.home() / ".neurobrix" / "license_acceptances.json"

# License metadata: name, URL, whether gate (explicit acceptance) is required
LICENSE_INFO = {
    "apache-2.0": {
        "name": "Apache License 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
        "gated": False,
    },
    "mit": {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
        "gated": False,
    },
    "cc-by-4.0": {
        "name": "Creative Commons Attribution 4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/",
        "gated": False,
    },
    "cc-by-nc-4.0": {
        "name": "Creative Commons Attribution-NonCommercial 4.0",
        "url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "gated": True,
    },
    "cc-by-nc-sa-4.0": {
        "name": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
        "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "gated": True,
    },
    "openrail++": {
        "name": "Open RAIL++-M License",
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md",
        "gated": False,
    },
    "llama3.1": {
        "name": "Llama 3.1 Community License Agreement",
        "url": "https://www.llama.com/llama3_1/license/",
        "gated": True,
    },
    "gemma": {
        "name": "Gemma Terms of Use",
        "url": "https://ai.google.dev/gemma/terms",
        "gated": True,
    },
    "deepseek": {
        "name": "DeepSeek License Agreement",
        "url": "https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-MODEL",
        "gated": False,
    },
    "cogvideox": {
        "name": "CogVideoX License",
        "url": "https://huggingface.co/THUDM/CogVideoX-2b/blob/main/LICENSE",
        "gated": True,
    },
}


def get_license_info(license_id: str) -> dict:
    """Get license metadata. Returns generic info for unknown licenses."""
    if license_id in LICENSE_INFO:
        return LICENSE_INFO[license_id]
    return {
        "name": license_id,
        "url": None,
        "gated": True,  # Unknown licenses require acceptance for safety
    }


def is_accepted(model_name: str) -> bool:
    """Check if user has already accepted the license for this model."""
    if not _ACCEPTANCES_FILE.exists():
        return False
    try:
        data = json.loads(_ACCEPTANCES_FILE.read_text())
        return model_name in data
    except (json.JSONDecodeError, OSError):
        return False


def record_acceptance(model_name: str, license_id: str):
    """Record that user accepted the license for a model."""
    _ACCEPTANCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if _ACCEPTANCES_FILE.exists():
        try:
            data = json.loads(_ACCEPTANCES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
    data[model_name] = {
        "license": license_id,
        "accepted_at": datetime.now(timezone.utc).isoformat(),
    }
    _ACCEPTANCES_FILE.write_text(json.dumps(data, indent=2))


def prompt_license_acceptance(
    model_name: str,
    license_id: str,
    vendor: str = "",
) -> bool:
    """Display license in terminal and prompt for acceptance.

    Returns True if accepted, False if declined.
    For non-gated permissive licenses, shows info but auto-accepts.
    """
    info = get_license_info(license_id)

    print(f"\n{'=' * 70}")
    print("LICENSE NOTICE")
    print("=" * 70)
    if vendor:
        print(f"   Vendor: {vendor}")
    print(f"   License: {info['name']}")
    if info["url"]:
        print(f"   Full text: {info['url']}")
    print()
    print("   This model is distributed under the license above.")
    print("   NeuroBrix does not modify the original license terms.")
    print("   You are responsible for complying with the license.")

    if not info["gated"]:
        # Permissive license — inform but don't block
        print(f"\n   (Permissive license — no acceptance required)")
        print("=" * 70)
        record_acceptance(model_name, license_id)
        return True

    # Gated license — require explicit acceptance
    print()
    print("   THIS LICENSE REQUIRES EXPLICIT ACCEPTANCE.")
    print("   By typing 'yes', you confirm that you have read and agree")
    print("   to the terms of the license above.")
    print("=" * 70)

    try:
        reply = input("\n   Accept license terms? [yes/No]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n   Declined.")
        return False

    if reply == "yes":
        record_acceptance(model_name, license_id)
        print("   License accepted.")
        return True
    else:
        print("   License declined. Download cancelled.")
        return False
