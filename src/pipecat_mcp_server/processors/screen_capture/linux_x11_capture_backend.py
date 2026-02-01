#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Linux X11 screen capture backend using python-xlib."""

import asyncio
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from .base_capture_backend import BaseCaptureBackend, WindowInfo

# Lazy references populated by _ensure_xlib()
_display_module = None
_X_module = None
_Xatom_module = None


def _ensure_xlib():
    """Import python-xlib on first use."""
    global _display_module, _X_module, _Xatom_module
    if _display_module is not None:
        return

    import Xlib.display
    import Xlib.X
    import Xlib.Xatom

    _display_module = Xlib.display
    _X_module = Xlib.X
    _Xatom_module = Xlib.Xatom


def _find_window_by_id(display, window_id: int):
    """Find a window by its X11 window ID.

    Args:
        display: Xlib Display instance.
        window_id: The X11 window ID.

    Returns:
        An Xlib Window object or None.

    """
    try:
        window = display.create_resource_object("window", window_id)
        # Verify the window exists by querying its attributes
        window.get_attributes()
        return window
    except Exception:
        return None


def _get_window_title(display, window) -> str:
    """Get the title of an X11 window.

    Args:
        display: Xlib Display instance.
        window: Xlib Window object.

    Returns:
        The window title, or empty string if not found.

    """
    try:
        net_wm_name = display.intern_atom("_NET_WM_NAME")
        prop = window.get_full_property(net_wm_name, 0)
        if prop and prop.value:
            title = prop.value
            if isinstance(title, bytes):
                title = title.decode("utf-8", errors="replace")
            return title
    except Exception:
        pass
    try:
        prop = window.get_full_property(_Xatom_module.XA_WM_NAME, 0)
        if prop and prop.value:
            title = prop.value
            if isinstance(title, bytes):
                title = title.decode("utf-8", errors="replace")
            return title
    except Exception:
        pass
    return ""


def _capture_x11(display, window) -> Optional[Tuple[bytes, Tuple[int, int]]]:
    """Capture a window's content via XGetImage.

    Args:
        display: Xlib Display instance.
        window: Xlib Window to capture.

    Returns:
        Tuple of (rgb_bytes, (width, height)) or None on failure.

    """
    try:
        geom = window.get_geometry()
        width = geom.width
        height = geom.height
    except Exception as e:
        logger.error(f"Failed to get window geometry: {e}")
        return None

    if width == 0 or height == 0:
        return None

    try:
        raw = window.get_image(0, 0, width, height, _X_module.ZPixmap, 0xFFFFFFFF)
    except Exception as e:
        logger.error(f"XGetImage failed: {e}")
        return None

    # X11 ZPixmap returns BGRA (32-bit depth)
    data = raw.data
    if isinstance(data, str):
        data = data.encode("latin-1")

    arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 4)
    # BGRA → RGB
    rgb = np.ascontiguousarray(arr[:, [2, 1, 0]]).tobytes()
    return (rgb, (width, height))


class LinuxX11CaptureBackend(BaseCaptureBackend):
    """Linux screen capture backend using X11/Xlib."""

    def __init__(self):
        """Initialize the Linux X11 capture backend."""
        self._display = None
        self._window = None

    async def list_windows(self) -> List[WindowInfo]:
        """List all open windows via X11. Not yet implemented."""
        raise NotImplementedError("list_windows is not yet implemented for Linux X11")

    async def start(self, window_id: Optional[int], monitor: int) -> Optional[int]:
        """Set up the X11 display and find the target window.

        Args:
            window_id: Optional X11 window ID to capture.
            monitor: Monitor index (used for root window capture).

        Returns:
            The window ID if found, or None if not found or capturing full screen.

        """
        _ensure_xlib()

        self._display = _display_module.Display()

        matched_id = None
        if window_id is not None:
            self._window = _find_window_by_id(self._display, window_id)
            if self._window:
                matched_id = window_id
                title = _get_window_title(self._display, self._window) or str(window_id)
                logger.debug(f"Found window: '{title}' (ID: {window_id})")
            else:
                logger.warning(f"Window ID {window_id} not found, falling back to full screen")

        if self._window is None:
            self._window = self._display.screen(monitor).root
            logger.debug(f"Capturing root window (screen {monitor})")

        return matched_id

    async def capture(self) -> Optional[Tuple[bytes, Tuple[int, int]]]:
        """Capture a single frame from the X11 window.

        Returns:
            Tuple of (rgb_bytes, (width, height)) or None on failure.

        """
        if self._display is None or self._window is None:
            return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _capture_x11, self._display, self._window)

    async def stop(self) -> None:
        """Release X11 resources."""
        if self._display:
            self._display.close()
            self._display = None
        self._window = None
