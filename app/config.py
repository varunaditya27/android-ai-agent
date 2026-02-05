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


class LLMSettings(BaseSettings):
    """LLM configuration settings."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL",
    )
    llm_model: str = Field(default="gpt-4o", description="Model for vision tasks")
    llm_max_tokens: int = Field(default=4096, description="Max tokens for responses")
    llm_temperature: float = Field(default=0.1, description="Temperature for LLM")


class CloudDeviceSettings(BaseSettings):
    """Cloud device provider configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    cloud_provider: Literal["limrun", "browserstack"] = Field(
        default="limrun",
        description="Cloud device provider",
    )

    # Limrun settings
    limrun_api_key: str = Field(default="", description="Limrun API key")
    limrun_api_url: str = Field(
        default="https://api.limrun.com/v1",
        description="Limrun API URL",
    )

    # BrowserStack settings
    browserstack_username: str = Field(default="", description="BrowserStack username")
    browserstack_access_key: str = Field(default="", description="BrowserStack access key")
    browserstack_api_url: str = Field(
        default="https://api-cloud.browserstack.com/app-automate",
        description="BrowserStack API URL",
    )

    default_device_id: str = Field(default="", description="Default device ID")


class ServerSettings(BaseSettings):
    """Server configuration settings."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    server_host: str = Field(default="0.0.0.0", description="Server host")
    server_port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=True, description="Debug mode")
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

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

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
        print(settings.llm.openai_api_key)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    device: CloudDeviceSettings = Field(default_factory=CloudDeviceSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    def __init__(self, **kwargs):
        """Initialize settings with nested configuration."""
        super().__init__(**kwargs)
        # Re-initialize nested settings to pick up env vars
        self.llm = LLMSettings()
        self.device = CloudDeviceSettings()
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
