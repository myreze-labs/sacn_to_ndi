"""
DMX Encoding Module

Provides different strategies for encoding DMX data for NDI transport.
Can be extended with new encoding schemes as needed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sacn_receiver import DMXData


@dataclass
class EncodedFrame:
    """Container for encoded video frame data."""
    data: np.ndarray
    width: int
    height: int
    fourcc: str  # e.g., "BGRX", "BGRA", "UYVY"


@dataclass
class EncodedMetadata:
    """Container for encoded metadata."""
    xml_string: str


class DMXEncoder(ABC):
    """Abstract base class for DMX encoders."""
    
    @abstractmethod
    def encode_video(self, dmx_data: DMXData | list[DMXData]) -> EncodedFrame:
        """
        Encode DMX data as a video frame.
        
        Args:
            dmx_data: Single DMXData or list of DMXData for multiple universes
            
        Returns:
            EncodedFrame with numpy array and dimensions
        """
        ...
    
    @abstractmethod
    def encode_metadata(self, dmx_data: DMXData | list[DMXData]) -> EncodedMetadata:
        """
        Encode DMX data as XML metadata.
        
        Args:
            dmx_data: Single DMXData or list of DMXData for multiple universes
            
        Returns:
            EncodedMetadata with XML string
        """
        ...


class SingleUniverseEncoder(DMXEncoder):
    """
    Encoder for single universe DMX data.
    
    Video encoding: 512x1 BGRX image where Blue channel = DMX value
    Metadata encoding: XML with channel values
    """
    
    def __init__(self):
        # Pre-allocate frame buffer for efficiency
        self._frame_buffer = np.zeros((1, 512, 4), dtype=np.uint8)
    
    def encode_video(self, dmx_data: DMXData | list[DMXData]) -> EncodedFrame:
        """
        Encode DMX as 512x1 video frame.
        
        Each pixel's Blue channel represents one DMX channel value.
        Green and Red are set to same value for grayscale visualization.
        Alpha is set to 255.
        """
        if isinstance(dmx_data, list):
            if len(dmx_data) != 1:
                raise ValueError("SingleUniverseEncoder only supports one universe")
            dmx_data = dmx_data[0]
        
        # Convert DMX data to numpy array
        dmx_array = np.frombuffer(dmx_data.data, dtype=np.uint8)
        
        # BGRX format: B=DMX, G=DMX, R=DMX, X=255 (grayscale representation)
        self._frame_buffer[0, :, 0] = dmx_array  # Blue
        self._frame_buffer[0, :, 1] = dmx_array  # Green
        self._frame_buffer[0, :, 2] = dmx_array  # Red
        self._frame_buffer[0, :, 3] = 255        # Alpha/X
        
        return EncodedFrame(
            data=self._frame_buffer.copy(),
            width=512,
            height=1,
            fourcc="BGRX",
        )
    
    def encode_metadata(self, dmx_data: DMXData | list[DMXData]) -> EncodedMetadata:
        """
        Encode DMX as XML metadata.
        
        Format:
        <dmx universe="1" sequence="0" priority="100">
            <channels encoding="base64">...</channels>
        </dmx>
        """
        import base64
        
        if isinstance(dmx_data, list):
            if len(dmx_data) != 1:
                raise ValueError("SingleUniverseEncoder only supports one universe")
            dmx_data = dmx_data[0]
        
        root = ET.Element("dmx")
        root.set("universe", str(dmx_data.universe))
        root.set("sequence", str(dmx_data.sequence))
        root.set("priority", str(dmx_data.priority))
        
        # Base64 encode the DMX data for compact representation
        channels = ET.SubElement(root, "channels")
        channels.set("encoding", "base64")
        channels.text = base64.b64encode(dmx_data.data).decode("ascii")
        
        return EncodedMetadata(
            xml_string=ET.tostring(root, encoding="unicode")
        )


class MultiUniverseEncoder(DMXEncoder):
    """
    Encoder for multiple universe DMX data.
    
    Video encoding: 512xN BGRX image where N = number of universes
                   Each row is one universe, Blue channel = DMX value
    Metadata encoding: XML with all universes
    """
    
    def __init__(self, num_universes: int):
        self._num_universes = num_universes
        # Pre-allocate frame buffer
        self._frame_buffer = np.zeros((num_universes, 512, 4), dtype=np.uint8)
    
    def encode_video(self, dmx_data: DMXData | list[DMXData]) -> EncodedFrame:
        """
        Encode multiple universes as 512xN video frame.
        
        Each row represents one universe.
        Each pixel's Blue channel represents one DMX channel value.
        """
        if isinstance(dmx_data, DMXData):
            dmx_data = [dmx_data]
        
        if len(dmx_data) > self._num_universes:
            raise ValueError(
                f"Too many universes: got {len(dmx_data)}, max {self._num_universes}"
            )
        
        # Clear buffer
        self._frame_buffer.fill(0)
        self._frame_buffer[:, :, 3] = 255  # Alpha
        
        # Fill in each universe
        for i, data in enumerate(dmx_data):
            dmx_array = np.frombuffer(data.data, dtype=np.uint8)
            self._frame_buffer[i, :, 0] = dmx_array  # Blue
            self._frame_buffer[i, :, 1] = dmx_array  # Green
            self._frame_buffer[i, :, 2] = dmx_array  # Red
        
        return EncodedFrame(
            data=self._frame_buffer.copy(),
            width=512,
            height=self._num_universes,
            fourcc="BGRX",
        )
    
    def encode_metadata(self, dmx_data: DMXData | list[DMXData]) -> EncodedMetadata:
        """
        Encode multiple universes as XML metadata.
        
        Format:
        <dmx_data>
            <universe id="1" sequence="0" priority="100">
                <channels encoding="base64">...</channels>
            </universe>
            ...
        </dmx_data>
        """
        import base64
        
        if isinstance(dmx_data, DMXData):
            dmx_data = [dmx_data]
        
        root = ET.Element("dmx_data")
        root.set("universes", str(len(dmx_data)))
        
        for data in dmx_data:
            universe_elem = ET.SubElement(root, "universe")
            universe_elem.set("id", str(data.universe))
            universe_elem.set("sequence", str(data.sequence))
            universe_elem.set("priority", str(data.priority))
            
            channels = ET.SubElement(universe_elem, "channels")
            channels.set("encoding", "base64")
            channels.text = base64.b64encode(data.data).decode("ascii")
        
        return EncodedMetadata(
            xml_string=ET.tostring(root, encoding="unicode")
        )


class ResizableEncoder(DMXEncoder):
    """
    Wraps any DMXEncoder and resizes/reformats its video output.
    
    Takes the inner encoder's raw frame (e.g. 512x1 BGRX) and
    reshapes/tiles it into the target resolution and pixel format.
    
    DMX channel values are laid out left-to-right, top-to-bottom.
    All channels of a pixel carry the same DMX value (grayscale).
    Unused pixels are black with full alpha.
    """
    
    def __init__(
        self,
        inner: DMXEncoder,
        width: int = 128,
        height: int = 128,
        fourcc: str = "RGBA",
    ):
        self._inner = inner
        self._width = width
        self._height = height
        self._fourcc = fourcc
        
        # Pre-allocate output buffer
        self._frame_buffer = np.zeros(
            (self._height, self._width, 4), dtype=np.uint8
        )
    
    def encode_video(self, dmx_data: DMXData | list[DMXData]) -> EncodedFrame:
        """
        Encode DMX data into a WxH RGBA/BGRX frame.
        
        512 DMX channels are mapped into the pixel grid
        left-to-right, top-to-bottom. Each pixel's RGB channels
        carry the DMX value; alpha is always 255.
        """
        # Get raw DMX values from the inner encoder's data path
        if isinstance(dmx_data, list):
            all_bytes = bytearray()
            for d in dmx_data:
                all_bytes.extend(d.data)
        else:
            all_bytes = dmx_data.data
        
        total_pixels = self._width * self._height
        dmx_array = np.frombuffer(all_bytes, dtype=np.uint8)
        
        # Clear buffer
        self._frame_buffer.fill(0)
        # Alpha channel always 255
        self._frame_buffer[:, :, 3] = 255
        
        # Map DMX channels to pixels (fill what we have)
        num_channels = min(len(dmx_array), total_pixels)
        
        if self._fourcc in ("RGBA", "RGBX"):
            # R=DMX, G=DMX, B=DMX, A=255
            for i in range(num_channels):
                row = i // self._width
                col = i % self._width
                v = dmx_array[i]
                self._frame_buffer[row, col, 0] = v  # R
                self._frame_buffer[row, col, 1] = v  # G
                self._frame_buffer[row, col, 2] = v  # B
        else:
            # BGRX/BGRA: B=DMX, G=DMX, R=DMX, X/A=255
            for i in range(num_channels):
                row = i // self._width
                col = i % self._width
                v = dmx_array[i]
                self._frame_buffer[row, col, 0] = v  # B
                self._frame_buffer[row, col, 1] = v  # G
                self._frame_buffer[row, col, 2] = v  # R
        
        return EncodedFrame(
            data=self._frame_buffer.copy(),
            width=self._width,
            height=self._height,
            fourcc=self._fourcc,
        )
    
    def encode_metadata(self, dmx_data: DMXData | list[DMXData]) -> EncodedMetadata:
        """Delegate metadata encoding to inner encoder."""
        return self._inner.encode_metadata(dmx_data)


# Decoder utilities for receiver side

def decode_video_single_universe(frame_data: np.ndarray) -> bytearray:
    """
    Decode a single universe from video frame.
    
    Args:
        frame_data: numpy array of shape (1, 512, 4) or (512, 4)
        
    Returns:
        bytearray of 512 DMX channel values
    """
    if frame_data.ndim == 3:
        # Take first row, Blue channel
        return bytearray(frame_data[0, :, 0].tobytes())
    elif frame_data.ndim == 2:
        # Take Blue channel
        return bytearray(frame_data[:, 0].tobytes())
    else:
        raise ValueError(f"Unexpected frame shape: {frame_data.shape}")


def decode_video_multi_universe(frame_data: np.ndarray) -> list[bytearray]:
    """
    Decode multiple universes from video frame.
    
    Args:
        frame_data: numpy array of shape (N, 512, 4)
        
    Returns:
        List of bytearrays, one per universe
    """
    if frame_data.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape: {frame_data.shape}")
    
    universes = []
    for row in range(frame_data.shape[0]):
        universes.append(bytearray(frame_data[row, :, 0].tobytes()))
    
    return universes


def decode_metadata_xml(xml_string: str) -> dict[int, bytearray]:
    """
    Decode DMX data from XML metadata.
    
    Returns:
        Dictionary mapping universe ID to DMX data
    """
    import base64
    
    root = ET.fromstring(xml_string)
    result = {}
    
    if root.tag == "dmx":
        # Single universe format
        universe = int(root.get("universe", 1))
        channels_elem = root.find("channels")
        if channels_elem is not None and channels_elem.text:
            result[universe] = bytearray(base64.b64decode(channels_elem.text))
    
    elif root.tag == "dmx_data":
        # Multi-universe format
        for universe_elem in root.findall("universe"):
            universe = int(universe_elem.get("id", 1))
            channels_elem = universe_elem.find("channels")
            if channels_elem is not None and channels_elem.text:
                result[universe] = bytearray(base64.b64decode(channels_elem.text))
    
    return result
