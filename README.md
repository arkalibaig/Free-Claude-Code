# Free Claude Code Gemini Bridge

A high performance bridge for running Claude Code using Gemini powered backends. This project provides a lean and lightweight pipeline optimized specifically for Google Gemini models.

## Project Status
The bridge is now operational and focused strictly on Gemini integration. All redundant providers and legacy modules have been removed to ensure maximum performance and minimal latency.

## Key Features
* Gemini Optimization: Built specifically for Gemini 2.0 and 1.5 models.
* Handshake Stability: Resolved connection issues and unauthorized errors.
* Tool Index Management: Implemented tool index offsets to stabilize streaming between reasoning and action blocks.
* Minimalist Architecture: Stripped of all unnecessary bloat for a pure local inference environment.

## Setup
1. Configure your environment variables in the .env file.
2. Install dependencies using uv.
3. Start the server using the command python server.py.

## Usage
Once the server is running on localhost port 8082, point your Claude Code CLI to the local bridge:
export ANTHROPIC_BASE_URL=http://localhost:8082/v1
export ANTHROPIC_API_KEY=your_token
claude

## License
MIT
