from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent.parent / "Coverage_exp" / "Datasets"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_env_files() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(override=False)


@dataclass(frozen=True, slots=True)
class BackendConfig:
    api_url: str
    es_url: str
    dataset_root: Path
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_timeout_seconds: float
    llm_timeout_multiplier: float
    llm_timeout_max_seconds: float
    llm_max_retries: int
    llm_retry_backoff_seconds: float
    llm_retry_max_sleep_seconds: float
    llm_temperature: float
    llm_cache_enabled: bool
    llm_cache_dir: Path

    @classmethod
    def from_env(cls) -> "BackendConfig":
        _load_env_files()
        api_url = os.environ.get("ALPACA_API_URL", "http://roberto-vm.vpn.sintef:8004").rstrip("/")
        es_url = os.environ.get("ALPACA_ES_URL", "http://roberto-vm.vpn.sintef:9209").rstrip("/")
        dataset_root_raw = os.environ.get("ALPACA_DATASET_ROOT")
        dataset_root = Path(dataset_root_raw).expanduser() if dataset_root_raw else DEFAULT_DATASET_ROOT
        llm_api_key = os.environ.get("ALPACA_LLM_API_KEY", os.environ.get("DEEPINFRA_TOKEN", ""))
        llm_base_url = os.environ.get(
            "ALPACA_LLM_BASE_URL",
            "https://api.deepinfra.com/v1/openai",
        ).rstrip("/")
        llm_model = os.environ.get("ALPACA_LLM_MODEL", "openai/gpt-oss-20b")
        llm_timeout_seconds = float(os.environ.get("ALPACA_LLM_TIMEOUT_SECONDS", "90"))
        llm_timeout_multiplier = float(os.environ.get("ALPACA_LLM_TIMEOUT_MULTIPLIER", "1.8"))
        llm_timeout_max_seconds = float(os.environ.get("ALPACA_LLM_TIMEOUT_MAX_SECONDS", "300"))
        llm_max_retries = int(os.environ.get("ALPACA_LLM_MAX_RETRIES", "2"))
        llm_retry_backoff_seconds = float(os.environ.get("ALPACA_LLM_RETRY_BACKOFF_SECONDS", "2"))
        llm_retry_max_sleep_seconds = float(os.environ.get("ALPACA_LLM_RETRY_MAX_SLEEP_SECONDS", "20"))
        llm_temperature = float(os.environ.get("ALPACA_LLM_TEMPERATURE", "0"))
        llm_cache_enabled = os.environ.get("ALPACA_LLM_CACHE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
        llm_cache_dir_raw = os.environ.get("ALPACA_LLM_CACHE_DIR")
        llm_cache_dir = (
            Path(llm_cache_dir_raw).expanduser()
            if llm_cache_dir_raw
            else REPO_ROOT / "backend_lab" / ".cache"
        )
        return cls(
            api_url=api_url,
            es_url=es_url,
            dataset_root=dataset_root,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_timeout_seconds=llm_timeout_seconds,
            llm_timeout_multiplier=llm_timeout_multiplier,
            llm_timeout_max_seconds=llm_timeout_max_seconds,
            llm_max_retries=llm_max_retries,
            llm_retry_backoff_seconds=llm_retry_backoff_seconds,
            llm_retry_max_sleep_seconds=llm_retry_max_sleep_seconds,
            llm_temperature=llm_temperature,
            llm_cache_enabled=llm_cache_enabled,
            llm_cache_dir=llm_cache_dir,
        )

    def with_cache_mode(self, mode: str) -> "BackendConfig":
        normalized = mode.strip().lower()
        if normalized == "env":
            return self
        if normalized == "on":
            return replace(self, llm_cache_enabled=True)
        if normalized == "off":
            return replace(self, llm_cache_enabled=False)
        raise ValueError(f"Unsupported cache mode: {mode}")
