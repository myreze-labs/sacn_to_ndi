"""
NDI Sender Module

Handles sending video frames and metadata over NDI.
Implementations can be swapped (cyndilib, ndi-python, etc.)
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from encoders import EncodedFrame, EncodedMetadata


@dataclass
class NDISenderConfig:
    """Configuration for NDI sender."""
    source_name: str
    frame_rate: int = 30
    clock_video: bool = True  # Use NDI's internal frame timing
    width: int = 512           # Initial frame width
    height: int = 1            # Initial frame height
    fourcc: str = "RGBA"       # Pixel format: RGBA, BGRX, BGRA


class NDISenderProtocol(ABC):
    """Abstract base class for NDI senders."""
    
    @abstractmethod
    def start(self) -> None:
        """Initialize and start the NDI sender."""
        ...
    
    @abstractmethod
    def stop(self) -> None:
        """Stop and clean up the NDI sender."""
        ...
    
    @abstractmethod
    def send_video_frame(self, frame: EncodedFrame) -> None:
        """Send a video frame."""
        ...
    
    @abstractmethod
    def send_metadata(self, metadata: EncodedMetadata) -> None:
        """Send metadata."""
        ...
    
    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if sender is running."""
        ...
    
    @property
    @abstractmethod
    def num_connections(self) -> int:
        """Get number of connected receivers."""
        ...


class CyndilibNDISender(NDISenderProtocol):
    """
    NDI sender implementation using cyndilib.

    cyndilib is a Cython wrapper for the NDI SDK that provides
    better performance than pure Python wrappers.

    Verified API (cyndilib ~0.1.0):
      - Sender.set_video_frame(vf) must be called BEFORE Sender.open()
      - VideoSendFrame.set_frame_rate() takes a fractions.Fraction
      - VideoSendFrame.write_data() expects a flat 1-D numpy array
      - Sender.send_video() takes no arguments (uses attached frame)
      - Sender.send_metadata(tag, attrs) takes tag string + attrs dict
    """

    def __init__(self, config: NDISenderConfig):
        self._config = config
        self._sender = None
        self._video_frame = None
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Initialize and start the NDI sender."""
        if self._running:
            return

        # Import here so the module loads even without cyndilib installed
        from fractions import Fraction
        from cyndilib.wrapper.ndi_structs import FourCC
        from cyndilib.sender import Sender
        from cyndilib.video_frame import VideoSendFrame

        # Map fourcc string to cyndilib enum
        fourcc_map = {
            "BGRX": FourCC.BGRX,
            "BGRA": FourCC.BGRA,
            "RGBA": FourCC.RGBA,
            "RGBX": FourCC.RGBX,
            "UYVY": FourCC.UYVY,
        }
        fourcc = fourcc_map.get(self._config.fourcc)
        if fourcc is None:
            raise ValueError(
                f"Unsupported FourCC: {self._config.fourcc}"
            )

        # 1. Create sender (not yet open)
        self._sender = Sender(self._config.source_name)

        # 2. Create and configure video frame BEFORE open
        #    cyndilib does not allow altering the frame after sender.open()
        self._video_frame = VideoSendFrame()
        self._video_frame.set_fourcc(fourcc)
        self._video_frame.set_resolution(
            self._config.width, self._config.height
        )
        self._video_frame.set_frame_rate(
            Fraction(self._config.frame_rate, 1)
        )

        # 3. Attach frame to sender (required before open)
        self._sender.set_video_frame(self._video_frame)

        # 4. Open sender
        self._sender.open()

        self._running = True

    def stop(self) -> None:
        """Stop and clean up the NDI sender."""
        if not self._running:
            return

        with self._lock:
            if self._sender is not None:
                self._sender.close()
                self._sender = None

            self._video_frame = None
            self._running = False

    def send_video_frame(self, frame: EncodedFrame) -> None:
        """
        Send a video frame over NDI.

        The frame dimensions and format MUST match what was configured
        in NDISenderConfig. cyndilib does not allow altering the frame
        after the sender is opened.

        Args:
            frame: EncodedFrame with numpy array data and dimensions
        """
        if not self._running or self._sender is None:
            raise RuntimeError("NDI sender not running")

        with self._lock:
            if self._video_frame is None or self._sender is None:
                return

            # write_data expects a flat 1-D uint8 array
            flat_data = frame.data.ravel()
            self._video_frame.write_data(flat_data)

            # send_video() uses the attached video frame (no arguments)
            self._sender.send_video()

    def send_metadata(self, metadata: EncodedMetadata) -> None:
        """
        Send metadata over NDI.

        cyndilib's Sender.send_metadata(tag, attrs) takes a tag string
        and a dict of attributes. We parse our XML string back into
        tag + attrs to match the API.

        Args:
            metadata: EncodedMetadata with XML string
        """
        if not self._running or self._sender is None:
            raise RuntimeError("NDI sender not running")

        with self._lock:
            if self._sender is None:
                return

            # Parse XML string into tag + attributes for cyndilib API
            import xml.etree.ElementTree as ET
            root = ET.fromstring(metadata.xml_string)
            tag = root.tag
            attrs = dict(root.attrib)

            # Include text content as a 'data' attribute if present
            # (e.g. base64-encoded channel data from child elements)
            for child in root:
                if child.text:
                    key = child.tag
                    if child.attrib.get("encoding"):
                        key += "_encoding"
                        attrs[key] = child.attrib["encoding"]
                    attrs[child.tag] = child.text

            self._sender.send_metadata(tag, attrs)

    @property
    def is_running(self) -> bool:
        """Check if sender is running."""
        return self._running

    @property
    def num_connections(self) -> int:
        """Get number of connected receivers."""
        if self._sender is None:
            return 0
        try:
            # cyndilib requires a timeout_ms argument
            return self._sender.get_num_connections(0)
        except TypeError:
            return 0


class MockNDISender(NDISenderProtocol):
    """
    Mock NDI sender for testing without NDI SDK.
    
    Logs frames instead of sending them over NDI.
    """
    
    def __init__(self, config: NDISenderConfig):
        self._config = config
        self._running = False
        self._frame_count = 0
        self._metadata_count = 0
    
    def start(self) -> None:
        """Start the mock sender."""
        self._running = True
        print(f"[MockNDI] Started sender: {self._config.source_name}")
    
    def stop(self) -> None:
        """Stop the mock sender."""
        self._running = False
        print(f"[MockNDI] Stopped sender. Sent {self._frame_count} frames, {self._metadata_count} metadata")
    
    def send_video_frame(self, frame: EncodedFrame) -> None:
        """Log video frame info."""
        if not self._running:
            raise RuntimeError("Mock sender not running")
        
        self._frame_count += 1
        if self._frame_count % 30 == 0:  # Log every 30 frames
            print(f"[MockNDI] Frame {self._frame_count}: {frame.width}x{frame.height} {frame.fourcc}")
    
    def send_metadata(self, metadata: EncodedMetadata) -> None:
        """Log metadata info."""
        if not self._running:
            raise RuntimeError("Mock sender not running")
        
        self._metadata_count += 1
        if self._metadata_count % 30 == 0:
            print(f"[MockNDI] Metadata {self._metadata_count}: {len(metadata.xml_string)} bytes")
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def num_connections(self) -> int:
        return 1  # Pretend we have one connection
