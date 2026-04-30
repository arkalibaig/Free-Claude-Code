# Free-Claude-Code (Gemini Optimized)

> **⚠️ Status: Work in Progress**
> This project is currently under active development. The bridge logic is being refined and it is **not yet fully functional**. 

A high-performance bridge for running Claude Code using Gemini-powered backends. 

### Key Optimizations (In Progress)
* **Handshake Fix**: Working on resolving 401 unauthorized errors during the initial connection.
* **Tool-Index Offset**: Implementing tool index offsets (+2) to stabilize streaming between Thinking and Tool blocks.
* **Minimalist Build**: Stripping unnecessary bloat for a leaner local inference environment.

### Setup
1. Configure your environment in `.env`.
2. Run the server: `python server.py`
