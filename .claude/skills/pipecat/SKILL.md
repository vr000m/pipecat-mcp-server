---
name: pipecat
description: Start a voice conversation using the Pipecat MCP server
---

Start a voice conversation using the Pipecat MCP server's start(), listen(), speak(), and stop() tools.

## Flow

1. Call `start()` to initialize the voice agent
2. Greet the user with `speak()`, then loop: `listen()` → `speak()`
3. If the user wants to end the conversation, ask for verbal confirmation before stopping. When in doubt, keep listening.
4. Once confirmed, say goodbye with `speak()`, then call `stop()`

## Guidelines

- Keep responses to 1-2 short sentences. Brevity is critical for voice.
- Before any change (files, PRs, issues, etc.), show the proposed change in the terminal and ask for verbal confirmation.
- Always call `stop()` when the conversation ends.
