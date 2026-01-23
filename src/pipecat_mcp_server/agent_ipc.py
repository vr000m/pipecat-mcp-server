#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inter-process communication for the Pipecat MCP server.

This module manages the IPC queues and child process lifecycle for communication
between the MCP server (parent) and the Pipecat voice agent (child). The child
process runs separately to avoid stdio collisions with the MCP protocol.
"""

import asyncio
import atexit
import multiprocessing
from typing import Optional

# Use spawn to avoid issues with forking from async context Fork copies the
# parent's state (event loop, file descriptors, locks) which can cause
# issues. Spawn creates a fresh Python interpreter.
multiprocessing.set_start_method("spawn", force=True)

_cmd_queue: Optional[multiprocessing.Queue] = None
_response_queue: Optional[multiprocessing.Queue] = None
_pipecat_process: Optional[multiprocessing.Process] = None


def _cleanup():
    """Clean up the pipecat child process."""
    global _pipecat_process
    if _pipecat_process is not None and _pipecat_process.is_alive():
        # Try graceful shutdown first
        try:
            if _cmd_queue is not None:
                _cmd_queue.put({"cmd": "stop"})
                _pipecat_process.join(timeout=2.0)
        except Exception:
            pass

        # Force terminate if still alive
        if _pipecat_process.is_alive():
            _pipecat_process.terminate()
            _pipecat_process.join(timeout=1.0)

        # Kill if terminate didn't work
        if _pipecat_process.is_alive():
            _pipecat_process.kill()

        _pipecat_process = None


def start_pipecat_process():
    """Start the Pipecat child process.

    Creates IPC queues and spawns a new process to run the Pipecat voice agent.
    Cleans up any existing process before starting a new one.
    """
    global _cmd_queue, _response_queue, _pipecat_process

    # Clean up any existing process first
    _cleanup()

    # Create IPC queues using spawn context
    _cmd_queue = multiprocessing.Queue()
    _response_queue = multiprocessing.Queue()

    # Start pipecat as separate process (daemon so it exits when MCP stops)
    _pipecat_process = multiprocessing.Process(
        target=run_pipecat_process,
        args=(_cmd_queue, _response_queue),
        daemon=True,
    )
    _pipecat_process.start()

    atexit.register(_cleanup)


def stop_pipecat_process():
    """Stop the pipecat child process (explicit cleanup)."""
    _cleanup()


def run_pipecat_process(cmd_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue):
    """Entry point for the Pipecat child process.

    Initializes logging and runs the Pipecat main loop. This function is called
    in a separate process to avoid stdio collisions with the MCP protocol.

    Args:
        cmd_queue: Queue for receiving commands from the MCP server.
        response_queue: Queue for sending responses back to the MCP server.

    """
    global _cmd_queue, _response_queue

    import os
    import sys

    from loguru import logger

    # Configure logging for child process (to stderr, not stdout)
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    _cmd_queue = cmd_queue
    _response_queue = response_queue

    # Change to package directory so pipecat_main() can find bot.py
    package_dir = os.path.dirname(__file__)
    os.chdir(package_dir)

    # Import and run the pipecat main (which will call our bot() function)
    from pipecat.runner.run import main as pipecat_main

    pipecat_main()


async def send_response(response: dict):
    """Send a response from the child process to the MCP server.

    Args:
        response: Response dictionary to send.

    Raises:
        RuntimeError: If the Pipecat process has not been started.

    """
    if _response_queue is None:
        raise RuntimeError("Pipecat process not started")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _response_queue.put, response)


async def read_request() -> dict:
    """Read a request from the MCP server in the child process.

    Blocks until a command is available in the queue.

    Returns:
        Request dictionary containing the command and arguments.

    Raises:
        RuntimeError: If the Pipecat process has not been started.

    """
    if _cmd_queue is None:
        raise RuntimeError("Pipecat process not started")
    loop = asyncio.get_event_loop()
    request = await loop.run_in_executor(None, _cmd_queue.get)
    return request


async def send_command(cmd: str, **kwargs) -> dict:
    """Send a command to the Pipecat child process and wait for response.

    Args:
        cmd: Command name (e.g., "listen", "speak", "stop").
        **kwargs: Additional arguments for the command.

    Returns:
        Response dictionary from the child process.

    Raises:
        RuntimeError: If the Pipecat process has not been started or if
            the child process returns an error.

    """
    if _cmd_queue is None or _response_queue is None:
        raise RuntimeError("Pipecat process not started")

    request = {"cmd": cmd, **kwargs}

    # Use thread pool for blocking queue operations
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _cmd_queue.put, request)
    response = await loop.run_in_executor(None, _response_queue.get)

    if "error" in response:
        raise RuntimeError(response["error"])
    return response
