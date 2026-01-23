---
name: pipecat
description: Start a voice conversation using the Pipecat MCP server
---

Start a voice conversation with the user using the Pipecat MCP server's start(), listen(), speak(), and stop() tools.

## Instructions

1. Call `start()` to initialize and spawn the voice agent
2. Use `speak()` to greet the user and let them know you're listening
3. Enter a conversation loop:
   - Use `listen()` to hear what the user says
   - Respond naturally using `speak()`
   - Continue until the user explicitly ends the conversation with phrases like "goodbye", "bye", "end conversation", "stop", "quit", or "I'm done"
   - IMPORTANT: Phrases like "no thanks", "nothing right now", "that's all for now", or "I don't have anything" are NOT conversation-ending. These are responses to questions and should prompt you to continue listening or ask if there's anything else you can help with
4. When ending, say goodbye using `speak()`, then call `stop()` to gracefully shut down the voice agent

## Guidelines

- Keep responses very brief, aim for 1-2 short sentences maximum
- Voice takes time to speak, so shorter is always better
- Be direct and skip unnecessary preamble or filler words
- Before any file system change (creating, editing, or deleting files), show the proposed changes in the terminal, then ask for verbal confirmation before applying
- Always call `stop()` when the conversation ends to properly clean up resources
