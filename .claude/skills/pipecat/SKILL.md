---
name: pipecat
description: Start a voice conversation using the Pipecat MCP server
---

Start a voice conversation using the Pipecat MCP server's start(), listen(), speak(), and stop() tools.

## Flow

1. Print a nicely formatted message with bullet points in the terminal with the following information:
   - The voice session is starting
   - Once ready, they can connect via the transport of their choice (Pipecat Playground, Daily room, or phone call)
   - Models are downloaded on the first user connection, so the first connection may take a moment
   - If the connection is not established and the user cannot hear any audio, they should check the terminal for errors from the Pipecat MCP server
2. Call `start()` to initialize the voice agent
3. Greet the user with `speak()`, then call `listen()` to wait for input
4. When the user asks you to perform a task:
   - Acknowledge the request with `speak()` (do NOT call `listen()` yet)
   - Perform the work (edit files, run commands, etc.)
   - Use `speak()` to provide progress updates as needed during the task
   - Once the task is complete, use `speak()` to report the result
   - Only then call `listen()` to wait for the next user input
5. When the user asks a simple question or makes conversation (no task to perform), respond with `speak()` then immediately call `listen()`
6. If the user wants to end the conversation, ask for verbal confirmation before stopping. When in doubt, keep listening.
7. Once confirmed, say goodbye with `speak()`, then call `stop()`

The key principle: `listen()` means "I'm done and ready for the user to talk." Never call it while you still have work to do or updates to communicate.

## Guidelines

- Keep all responses and progress updates to 1-2 short sentences. Brevity is critical for voice.
- When the user asks you to perform a task (e.g., edit a file, create a PR), verbally acknowledge the request first, then start working on it. Do not work in silence.
- Before any change (files, PRs, issues, etc.), show the proposed change in the terminal, use `speak()` to ask for verbal confirmation, then call `listen()` to get the user's response before proceeding.
- Always call `stop()` when the conversation ends.
