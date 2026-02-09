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
    """
    
    def __init__(self, config: NDISenderConfig):
        self._config = config
        self._sender = None
        self._video_frame = None
        self._metadata_frame = None
        self._running = False
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Initialize and start the NDI sender."""
        if self._running:
            return
        
        # Import here to allow module to load even if cyndilib not installed
        from cyndilib.wrapper.ndi_structs import FourCC
        from cyndilib.sender import Sender
        from cyndilib.video_frame import VideoSendFrame
        from cyndilib.metadata_frame import MetadataSendFrame
        
        self._sender = Sender(self._config.source_name)
        self._sender.open()
        
        # Create video frame object
        self._video_frame = VideoSendFrame()
        self._fourcc = FourCC
        
        # Create metadata frame object
        self._metadata_frame = MetadataSendFrame()
        
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
            self._metadata_frame = None
            self._running = False
    
    def send_video_frame(self, frame: EncodedFrame) -> None:
        """
        Send a video frame over NDI.
        
        Args:
            frame: EncodedFrame with numpy array data and dimensions
        """
        if not self._running or self._sender is None:
            raise RuntimeError("NDI sender not running")
        
        from cyndilib.wrapper.ndi_structs import FourCC
        
        with self._lock:
            if self._video_frame is None or self._sender is None:
                return
            
            # Map fourcc string to cyndilib FourCC enum
            fourcc_map = {
                "BGRX": FourCC.BGRX,
                "BGRA": FourCC.BGRA,
                "UYVY": FourCC.UYVY,
                "RGBA": FourCC.RGBA,
                "RGBX": FourCC.RGBX,
            }
            
            fourcc = fourcc_map.get(frame.fourcc)
            if fourcc is None:
                raise ValueError(f"Unsupported FourCC: {frame.fourcc}")
            
            # Configure video frame
            self._video_frame.set_fourcc(fourcc)
            self._video_frame.set_resolution(frame.width, frame.height)
            self._video_frame.set_frame_rate(self._config.frame_rate, 1)
            
            # Write frame data
            self._video_frame.write_data(frame.data)
            
            # Send the frame
            self._sender.send_video(self._video_frame)
    
    def send_metadata(self, metadata: EncodedMetadata) -> None:
        """
        Send metadata over NDI.
        
        Args:
            metadata: EncodedMetadata with XML string
        """
        if not self._running or self._sender is None:
            raise RuntimeError("NDI sender not running")
        
        with self._lock:
            if self._metadata_frame is None or self._sender is None:
                return
            
            # Set the metadata content
            self._metadata_frame.set_data(metadata.xml_string)
            
            # Send the metadata
            self._sender.send_metadata(self._metadata_frame)
    
    @property
    def is_running(self) -> bool:
        """Check if sender is running."""
        return self._running
    
    @property
    def num_connections(self) -> int:
        """Get number of connected receivers."""
        if self._sender is None:
            return 0
        return self._sender.get_num_connections()


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
