"""
Configuration Management
========================

Centralized configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Shared config that all settings classes use to load .env
_shared_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    env_prefix="",
    extra="ignore",
)


class LLMSettings(BaseSettings):
    """LLM configuration settings supporting Gemini and Groq providers."""

    model_config = _shared_config

    # Provider selection — "gemini" or "groq"
    llm_provider: Literal["gemini", "groq"] = Field(
        default="groq",
        description="LLM provider: 'groq' (free, recommended) or 'gemini'",
    )

    # Gemini settings
    gemini_api_key: str = Field(
        default="",
        description="Google AI API key for Gemini",
    )
    gemini_api_keys: str = Field(
        default="",
        description="Comma-separated list of Gemini API keys for circular rotation on rate limits",
    )

    # Groq settings
    groq_api_key: str = Field(
        default="",
        description="Groq API key for Llama 4 Scout vision model (free tier: 1000 RPD)",
    )
    groq_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description="Groq model ID (must support vision for screenshot analysis)",
    )

    # Shared LLM settings
    llm_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model for vision tasks (gemini-2.5-flash, gemini-2.0-flash, etc.)",
    )
    llm_max_output_tokens: int = Field(default=2048, description="Max output tokens for responses")
    llm_temperature: float = Field(default=0.1, description="Temperature for LLM")
    llm_top_p: float = Field(default=0.95, description="Top-p sampling parameter")
    llm_top_k: int = Field(default=40, description="Top-k sampling parameter")

    def get_all_api_keys(self) -> list[str]:
        """
        Return all available Gemini API keys as a de-duplicated, order-preserving list.

        Merges ``gemini_api_keys`` (comma-separated) with the single
        ``gemini_api_key`` to build the pool. Duplicates are removed
        while preserving insertion order.
        """
        seen: set[str] = set()
        keys: list[str] = []
        # Add keys from the comma-separated list first
        if self.gemini_api_keys:
            for k in self.gemini_api_keys.split(","):
                k = k.strip()
                if k and k not in seen:
                    seen.add(k)
                    keys.append(k)
        # Add the single key if it is not already in the list
        if self.gemini_api_key and self.gemini_api_key not in seen:
            keys.append(self.gemini_api_key)
        return keys if keys else [self.gemini_api_key]

    def get_active_api_key(self) -> str:
        """Return the primary API key for the selected provider."""
        if self.llm_provider == "groq":
            return self.groq_api_key
        return self.gemini_api_key

    def get_active_model(self) -> str:
        """Return the model name for the selected provider."""
        if self.llm_provider == "groq":
            return self.groq_model
        return self.llm_model


class DeviceSettings(BaseSettings):
    """Device provider configuration."""

    model_config = _shared_config

    device_provider: Literal["adb", "local", "emulator", "aws_device_farm"] = Field(
        default="adb",
        description="Device provider (adb/local/emulator = FREE local, aws_device_farm = AWS cloud)",
    )

    # ADB settings (FREE option - recommended for local dev)
    adb_device_serial: str = Field(
        default="",
        description="Specific ADB device serial (leave empty for auto-detect)",
    )

    # AWS Device Farm settings (cloud option)
    aws_device_farm_project_arn: str = Field(
        default="",
        description="ARN of the AWS Device Farm project (required for aws_device_farm provider)",
    )
    aws_device_farm_device_arn: str = Field(
        default="",
        description="ARN of a specific device to use (leave empty to auto-select)",
    )
    aws_access_key_id: str = Field(
        default="",
        description="AWS access key ID (optional – falls back to default credential chain)",
    )
    aws_secret_access_key: str = Field(
        default="",
        description="AWS secret access key (optional – falls back to default credential chain)",
    )


class ServerSettings(BaseSettings):
    """Server configuration settings."""

    model_config = _shared_config

    server_host: str = Field(default="0.0.0.0", description="Server host")
    server_port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=True, description="Debug mode")
    environment: str = Field(default="development", description="Environment name (development, staging, production)")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Log level",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for session storage",
    )
    cors_origins: str = Field(default="*", description="CORS origins")

    @field_validator("cors_origins")
    @classmethod
    def parse_cors_origins(cls, v: str) -> str:
        """Keep CORS origins as string, parse when needed."""
        return v

    def get_cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]


class AgentSettings(BaseSettings):
    """Agent configuration settings."""

    model_config = _shared_config

    max_steps: int = Field(default=30, description="Maximum steps per task")
    step_timeout: float = Field(default=30.0, description="Step timeout in seconds")
    screenshot_quality: int = Field(default=85, description="Screenshot quality (1-100)")
    enable_vision: bool = Field(default=False, description="Vision-based element detection (adds extra LLM call per step)")
    enable_accessibility_tree: bool = Field(
        default=True,
        description="Enable accessibility tree parsing",
    )
    min_step_interval: float = Field(
        default=3.0,
        description="Minimum seconds between LLM calls (3s for Groq 30 RPM, 12s for Gemini 5 RPM)",
    )
    rate_limit_max_retries: int = Field(
        default=5,
        description="Maximum retries for rate limit errors",
    )
    default_language: str = Field(default="en", description="Default language")


class Settings(BaseSettings):
    """
    Main settings class combining all configuration sections.

    Usage:
        from app.config import get_settings
        settings = get_settings()
        print(settings.llm.gemini_api_key)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    device: DeviceSettings = Field(default_factory=DeviceSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    def __init__(self, **kwargs):
        """Initialize settings with nested configuration."""
        super().__init__(**kwargs)
        # Re-initialize nested settings to pick up env vars
        self.llm = LLMSettings()
        self.device = DeviceSettings()
        self.server = ServerSettings()
        self.agent = AgentSettings()


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings loaded from environment.
    """
    return Settings()


# Convenience alias
settings = get_settings()