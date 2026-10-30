from __future__ import annotations

import os
import warnings
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.local.yaml"

_ENV_OVERRIDES = {
    "data.use_processed": "TRIPMIND_USE_PROCESSED",
    "data.bundle_root": "TRIPMIND_DATA_BUNDLE",
    "providers.route": "TRIPMIND_ROUTE_PROVIDER",
    "providers.realtime": "TRIPMIND_REALTIME_PROVIDER",
    "ors.api_key": "ORS_API_KEY",
    "ors.base_url": "ORS_BASE_URL",
    "dashscope.api_key": "DASHSCOPE_API_KEY",
    "dashscope.base_url": "DASHSCOPE_BASE_URL",
    "dashscope.model": "DASHSCOPE_MODEL",
    "dashscope.realtime_model": "DASHSCOPE_REALTIME_MODEL",
    "dashscope.search_strategy": "DASHSCOPE_SEARCH_STRATEGY",
    "realtime.ttl_seconds": "TRIPMIND_REALTIME_TTL_SECONDS",
    "realtime.poi_limit": "TRIPMIND_REALTIME_POI_LIMIT",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _get_path(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _coerce(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _set_path(config: dict[str, Any], dotted: str, value: str) -> None:
    parts = dotted.split(".")
    current = config
    for part in parts[:-1]:
        node = current.get(part)
        if not isinstance(node, dict):
            node = {}
            current[part] = node
        current = node
    old = current.get(parts[-1])
    current[parts[-1]] = _coerce(value, old)


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Load default YAML, optional local YAML, then environment overrides.

    Priority (highest last):
        configs/config.yaml
        configs/config.local.yaml
        .env / process environment

    `configs/config.local.yaml` and `.env` are gitignored and are the recommended
    places for real API keys.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    configured_path = os.getenv("TRIPMIND_CONFIG")
    config_path = Path(configured_path).expanduser() if configured_path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = _read_yaml(config_path)
    if LOCAL_CONFIG_PATH.exists() and LOCAL_CONFIG_PATH != config_path:
        config = _deep_merge(config, _read_yaml(LOCAL_CONFIG_PATH))

    for dotted, env_name in _ENV_OVERRIDES.items():
        if env_name in os.environ and os.environ[env_name] != "":
            if dotted.endswith("api_key"):
                file_key = _get_path(config, dotted)
                if file_key and str(file_key) != os.environ[env_name]:
                    warnings.warn(
                        f"{env_name} overrides a different API key from YAML. "
                        f"Unset {env_name} to use configs/config.local.yaml.",
                        RuntimeWarning, stacklevel=2,
                    )
            _set_path(config, dotted, os.environ[env_name])

    return config


def config_value(dotted: str, default: Any = None) -> Any:
    return _get_path(get_config(), dotted, default)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def clear_config_cache() -> None:
    get_config.cache_clear()
