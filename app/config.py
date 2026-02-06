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
    """Google Gemini LLM configuration settings."""

    model_config = _shared_config

    gemini_api_key: str = Field(..., description="Google AI API key for Gemini")
    llm_model: str = Field(
        default="gemini-2.5-pro",
        description="Gemini model for vision tasks (gemini-2.0-flash, gemini-1.5-pro, etc.)",
    )
    llm_max_output_tokens: int = Field(default=8192, description="Max output tokens for responses")
    llm_temperature: float = Field(default=0.1, description="Temperature for LLM")
    llm_top_p: float = Field(default=0.95, description="Top-p sampling parameter")
    llm_top_k: int = Field(default=40, description="Top-k sampling parameter")


class DeviceSettings(BaseSettings):
    """Device provider configuration."""

    model_config = _shared_config

    device_provider: Literal["adb", "local", "emulator", "limrun", "browserstack"] = Field(
        default="adb",
        description="Device provider (adb/local = FREE, limrun/browserstack = paid)",
    )

    # ADB settings (FREE option - recommended)
    adb_device_serial: str = Field(
        default="",
        description="Specific ADB device serial (leave empty for auto-detect)",
    )

    # Limrun settings (paid)
    limrun_api_key: str = Field(default="", description="Limrun API key")
    limrun_api_url: str = Field(
        default="https://api.limrun.com/v1",
        description="Limrun API URL",
    )

    # BrowserStack settings (paid)
    browserstack_username: str = Field(default="", description="BrowserStack username")
    browserstack_access_key: str = Field(default="", description="BrowserStack access key")
    browserstack_api_url: str = Field(
        default="https://api-cloud.browserstack.com/app-automate",
        description="BrowserStack API URL",
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

    max_steps: int = Field(default=50, description="Maximum steps per task")
    step_timeout: float = Field(default=30.0, description="Step timeout in seconds")
    screenshot_quality: int = Field(default=85, description="Screenshot quality (1-100)")
    enable_vision: bool = Field(default=True, description="Enable vision-based detection")
    enable_accessibility_tree: bool = Field(
        default=True,
        description="Enable accessibility tree parsing",
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