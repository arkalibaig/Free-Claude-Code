"""Minimalist configuration for Gemini-exclusive pipeline."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Gemini-specific settings for Claude Code bridge."""

    # ==================== Gemini Config ====================
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    
    # Default Gemini model
    model: str = "gemini-2.0-flash-exp"

    # ==================== Bridge Behavior ====================
    enable_thinking: bool = Field(default=True, validation_alias="ENABLE_THINKING")
    
    # ==================== HTTP Client Timeouts ====================
    http_read_timeout: float = 120.0
    http_write_timeout: float = 10.0
    http_connect_timeout: float = 5.0

    # ==================== Server ====================
    host: str = "0.0.0.0"
    port: int = 8082
    log_file: str = "server.log"
    
    # Optional server API key (Anthropic-style)
    anthropic_auth_token: str = Field(default="", validation_alias="ANTHROPIC_AUTH_TOKEN")

    @field_validator("gemini_api_key")
    @classmethod
    def require_api_key(cls, v: str) -> str:
        if not v.strip():
            # We don't raise here to allow startup, but provider will fail
            pass
        return v

    def resolve_model(self, claude_model_name: str) -> str:
        """Always resolve to the configured Gemini model."""
        return self.model

    @staticmethod
    def parse_model_name(model_string: str) -> str:
        """Extract model name from string (handles potential provider prefix)."""
        if "/" in model_string:
            return model_string.split("/")[-1]
        return model_string

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()
