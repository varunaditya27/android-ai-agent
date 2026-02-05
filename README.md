# Android AI Agent 🤖📱

**AI-powered mobile automation agent designed for blind and visually impaired users.**

Transform natural language commands into Android device actions using advanced AI reasoning and cloud-based device farms.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The Android AI Agent is an intelligent automation system that enables users to control Android devices through natural language commands. Specifically designed with accessibility in mind, it helps blind and visually impaired users interact with mobile applications independently.

### How It Works

```
User: "Open YouTube and search for relaxing music"
         │
         ▼
    ┌─────────────────┐
    │   AI Agent      │ ◄── ReAct (Reasoning + Acting) Loop
    │   (GPT-4o)      │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Cloud Device    │ ◄── Limrun / BrowserStack
    │ (Android Phone) │
    └────────┬────────┘
             │
         ▼
    Task Completed! 🎉
```

The agent uses a **ReAct (Reasoning + Acting)** loop:
1. **Observe** - Capture screenshot and UI hierarchy
2. **Think** - LLM analyzes the screen and decides next action
3. **Act** - Execute the action (tap, swipe, type, etc.)
4. **Repeat** - Continue until task is complete

---

## Features

### 🎯 Core Capabilities

- **Natural Language Control** - Describe tasks in plain English
- **Multi-Step Reasoning** - Complex tasks broken into logical steps
- **Visual Understanding** - GPT-4o vision analyzes screenshots
- **Accessibility Tree Parsing** - Structured UI element detection
- **Authentication Handling** - Secure credential input prompts
- **Error Recovery** - Automatic retry with alternative strategies

### ♿ Accessibility Features

- **TalkBack Integration** - Works with Android screen reader
- **Voice Announcements** - Audio feedback for actions
- **Haptic Feedback** - Vibration patterns for events
- **Blind-Friendly Design** - Clear, concise status updates

### 🔧 Technical Features

- **Cloud Device Farms** - Limrun and BrowserStack support
- **WebSocket Streaming** - Real-time progress updates
- **Async Architecture** - High-performance async/await
- **Modular Design** - Easy to extend and customize
- **Comprehensive Testing** - Unit and integration tests
- **Docker Support** - Easy deployment

---

## Architecture

```
android-ai-agent/
├── app/
│   ├── __init__.py           # Package initialization
│   ├── main.py               # FastAPI application entry
│   ├── config.py             # Configuration management
│   │
│   ├── agent/                # ReAct Agent Core
│   │   ├── react_loop.py     # Main reasoning loop
│   │   ├── state.py          # Agent state management
│   │   ├── prompts.py        # System prompts
│   │   └── actions/          # Action handlers
│   │       ├── handler.py    # Action dispatcher
│   │       ├── tap.py        # Tap actions
│   │       ├── swipe.py      # Swipe/scroll actions
│   │       ├── type_text.py  # Text input
│   │       ├── launch_app.py # App launcher
│   │       └── system.py     # System actions
│   │
│   ├── device/               # Device Abstraction
│   │   ├── cloud_provider.py # Cloud device interface
│   │   ├── limrun_client.py  # Limrun integration
│   │   ├── browserstack.py   # BrowserStack integration
│   │   └── screenshot.py     # Screenshot utilities
│   │
│   ├── perception/           # UI Understanding
│   │   ├── ui_parser.py      # Accessibility tree parser
│   │   ├── element_detector.py # Element detection
│   │   ├── auth_detector.py  # Login screen detection
│   │   └── ocr.py            # Text recognition
│   │
│   ├── llm/                  # LLM Integration
│   │   ├── client.py         # OpenAI-compatible client
│   │   ├── models.py         # Model configurations
│   │   └── response_parser.py # Parse agent responses
│   │
│   ├── accessibility/        # Accessibility Features
│   │   ├── announcer.py      # Voice announcements
│   │   ├── talkback.py       # TalkBack integration
│   │   └── haptics.py        # Haptic feedback
│   │
│   ├── api/                  # REST & WebSocket API
│   │   ├── routes/
│   │   │   ├── health.py     # Health checks
│   │   │   ├── sessions.py   # Device sessions
│   │   │   └── agent.py      # Agent endpoints
│   │   └── websocket.py      # Real-time streaming
│   │
│   └── utils/                # Utilities
│       ├── logger.py         # Structured logging
│       └── security.py       # Credential handling
│
├── tests/                    # Test suite
├── scripts/                  # Utility scripts
├── Dockerfile                # Container image
├── docker-compose.yml        # Service orchestration
└── requirements.txt          # Dependencies
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key (for GPT-4o)
- Cloud device provider credentials (Limrun or BrowserStack)

### 1. Clone and Install

```bash
git clone https://github.com/varunaditya27/android-ai-agent.git
cd android-ai-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit with your credentials
nano .env
```

Required settings:
```env
LLM_API_KEY=your-openai-api-key
CLOUD_DEVICE_API_KEY=your-device-provider-key
```

### 3. Run the Server

```bash
# Development mode with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or use the demo script
python scripts/run_demo.py
```

### 4. Test It Out

```bash
# Health check
curl http://localhost:8000/health

# Or open the interactive demo
python scripts/run_demo.py --task "Open YouTube"
```

---

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/varunaditya27/android-ai-agent.git
cd android-ai-agent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Using Docker

```bash
# Build image
docker build -t android-ai-agent .

# Run container
docker run -p 8000:8000 \
  -e LLM_API_KEY=your-key \
  -e CLOUD_DEVICE_API_KEY=your-key \
  android-ai-agent
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

---

## Configuration

### Environment Variables

Create a `.env` file with the following settings:

```env
# ===========================================
# LLM Configuration
# ===========================================
LLM_API_KEY=your-openai-api-key
LLM_MODEL_NAME=gpt-4o              # or gpt-4o-mini for cost savings
LLM_API_BASE=https://api.openai.com/v1
LLM_MAX_TOKENS=4096
LLM_TEMPERATURE=0.1

# ===========================================
# Cloud Device Configuration
# ===========================================
CLOUD_DEVICE_PROVIDER=limrun       # or browserstack
CLOUD_DEVICE_API_KEY=your-provider-key
CLOUD_DEVICE_BASE_URL=https://api.limrun.com/v1

# ===========================================
# Server Configuration
# ===========================================
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_DEBUG=false
SERVER_LOG_LEVEL=INFO
SERVER_ENVIRONMENT=production
SERVER_CORS_ORIGINS=*

# ===========================================
# Agent Configuration
# ===========================================
AGENT_MAX_STEPS=50
AGENT_STEP_TIMEOUT=30
AGENT_ACTION_DELAY=0.5
```

### Supported Cloud Providers

#### Limrun (Recommended)
- High-performance real device cloud
- Excellent API for automation
- Good pricing for development

#### BrowserStack
- Wide device selection
- Appium-based automation
- Enterprise-grade reliability

---

## Usage

### Interactive Demo

```bash
# Start interactive session
python scripts/run_demo.py

# With a specific task
python scripts/run_demo.py --task "Open Chrome and search for weather"
```

### REST API

```python
import requests

# Create a session
session = requests.post("http://localhost:8000/sessions", json={
    "device_type": "android",
    "timeout_minutes": 30
}).json()

session_id = session["session_id"]

# Execute a task
result = requests.post("http://localhost:8000/agent/execute", json={
    "session_id": session_id,
    "task": "Open YouTube and search for cooking videos",
    "max_steps": 30
}).json()

print(f"Success: {result['success']}")
print(f"Result: {result['result']}")

# Cleanup
requests.delete(f"http://localhost:8000/sessions/{session_id}")
```

### WebSocket Streaming

```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);

ws.onopen = () => {
  // Start a task
  ws.send(JSON.stringify({
    type: "start_task",
    data: {
      task: "Open Settings and enable WiFi",
      max_steps: 30
    }
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  switch (message.type) {
    case "step_update":
      console.log(`Step: ${message.data.action_type}`);
      console.log(`Thinking: ${message.data.thinking}`);
      break;
      
    case "input_required":
      // Handle credential input
      const password = prompt(message.data.prompt);
      ws.send(JSON.stringify({
        type: "provide_input",
        data: { value: password }
      }));
      break;
      
    case "task_completed":
      console.log(`Done! ${message.data.result}`);
      break;
  }
};
```

### Python SDK

```python
import asyncio
from app.agent import ReActAgent, AgentConfig
from app.device.cloud_provider import create_cloud_device
from app.llm.client import LLMClient

async def main():
    # Setup device
    device = create_cloud_device(
        provider="limrun",
        api_key="your-key",
    )
    await device.allocate()
    
    # Setup LLM
    llm = LLMClient(
        api_key="your-openai-key",
        model="gpt-4o",
    )
    
    # Create agent
    agent = ReActAgent(
        llm_client=llm,
        device=device,
        config=AgentConfig(max_steps=30),
    )
    
    # Run task
    result = await agent.run("Open YouTube and play trending videos")
    
    print(f"Success: {result.success}")
    print(f"Result: {result.result}")
    
    # Cleanup
    await device.release()

asyncio.run(main())
```

---

## API Reference

### Health Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/health/ready` | GET | Readiness probe |
| `/health/live` | GET | Liveness probe |
| `/health/info` | GET | Service information |

### Session Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sessions` | POST | Create new session |
| `/sessions` | GET | List all sessions |
| `/sessions/{id}` | GET | Get session details |
| `/sessions/{id}` | DELETE | Delete session |
| `/sessions/{id}/screenshot` | GET | Capture screenshot |

### Agent Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agent/execute` | POST | Execute task (blocking) |
| `/agent/status/{session_id}` | GET | Get agent status |
| `/agent/input` | POST | Provide user input |
| `/agent/cancel/{session_id}` | POST | Cancel task |
| `/agent/quick-action` | POST | Execute single action |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/{session_id}` | Real-time task streaming |

---

## Development

### Setup Development Environment

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Format code
black app/ tests/

# Lint
ruff check app/ tests/

# Type checking
mypy app/
```

### Project Structure Guidelines

- **Modular Design**: Each module has a single responsibility
- **Async First**: Use `async/await` for I/O operations
- **Type Hints**: Full type annotations for all functions
- **Documentation**: Docstrings for all public APIs
- **Error Handling**: Graceful error recovery

---

## Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=app --cov-report=html

# Specific test file
pytest tests/test_agent.py -v

# Specific test
pytest tests/test_agent.py::TestAgentState::test_start_task -v
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Multi-component workflows
- **API Tests**: FastAPI endpoint testing

### Docker Test Runner

```bash
docker-compose --profile test up test
```

---

## Deployment

### Production Checklist

- [ ] Set `SERVER_DEBUG=false`
- [ ] Set `SERVER_ENVIRONMENT=production`
- [ ] Configure proper `SERVER_CORS_ORIGINS`
- [ ] Use strong API keys
- [ ] Enable HTTPS (reverse proxy)
- [ ] Set up monitoring/logging
- [ ] Configure rate limiting

### Docker Compose Production

```bash
# Production deployment
docker-compose -f docker-compose.yml up -d

# Scale workers
docker-compose up -d --scale app=3
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: android-ai-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: android-ai-agent
  template:
    metadata:
      labels:
        app: android-ai-agent
    spec:
      containers:
      - name: agent
        image: android-ai-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: llm-api-key
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
```

---

## Contributing

We welcome contributions! Please see our contributing guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Run linting (`ruff check . && black --check .`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings (Google style)
- Keep functions focused and small
- Write tests for new features

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- OpenAI for GPT-4o vision capabilities
- Limrun and BrowserStack for cloud device infrastructure
- The accessibility community for invaluable feedback

---

## Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our community](https://discord.gg/example)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/android-ai-agent/issues)

---

<p align="center">
  Made with ❤️ for accessibility
</p>
