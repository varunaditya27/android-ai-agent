"""
Android AI Agent
================

AI-powered mobile automation agent for blind users.

This package provides a FastAPI-based backend for controlling Android devices
through natural language commands, with a focus on accessibility.

Modules:
    - api: FastAPI routes and WebSocket handlers
    - agent: ReAct loop implementation and state management
    - device: Device abstraction (local ADB, AWS Device Farm)
    - perception: UI parsing and element detection
    - llm: LLM client and response parsing
    - accessibility: TalkBack integration and announcements
    - utils: Logging, security, and helper utilities
"""

__version__ = "1.0.0"
__author__ = "Android AI Agent Team"
