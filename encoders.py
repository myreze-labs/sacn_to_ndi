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
    
    Channel depth modes:
      - "grayscale": R=G=B=DMX value per pixel (1 DMX ch/pixel)
      - "rgb": R, G, B each carry a separate DMX channel (3 DMX ch/pixel)
    
    Unused pixels are black with full alpha.
    """
    
    VALID_CHANNEL_DEPTHS = ("grayscale", "rgb")
    
    def __init__(
        self,
        inner: DMXEncoder,
        width: int = 128,
        height: int = 128,
        fourcc: str = "BGRX",
        channel_depth: str = "grayscale",
    ):
        self._inner = inner
        self._width = width
        self._height = height
        self._fourcc = fourcc
        
        if channel_depth not in self.VALID_CHANNEL_DEPTHS:
            raise ValueError(
                f"Invalid channel_depth '{channel_depth}', "
                f"expected one of {self.VALID_CHANNEL_DEPTHS}"
            )
        self._channel_depth = channel_depth
        
        # Pre-allocate output buffer (always 4 bytes per pixel for NDI)
        self._frame_buffer = np.zeros(
            (self._height, self._width, 4), dtype=np.uint8
        )
    
    @property
    def channel_depth(self) -> str:
        return self._channel_depth
    
    def _channels_per_pixel(self) -> int:
        """How many DMX channels are packed into each pixel."""
        return 3 if self._channel_depth == "rgb" else 1
    
    def encode_video(self, dmx_data: DMXData | list[DMXData]) -> EncodedFrame:
        """
        Encode DMX data into a WxH video frame.
        
        In grayscale mode: each pixel R=G=B=DMX[i] (1 DMX channel per pixel).
        In RGB mode: each pixel R=DMX[i], G=DMX[i+1], B=DMX[i+2] (3 per pixel).
        
        Alpha/X byte is always 255.
        """
        # Get raw DMX values
        if isinstance(dmx_data, list):
            all_bytes = bytearray()
            for d in dmx_data:
                all_bytes.extend(d.data)
        else:
            all_bytes = dmx_data.data
        
        dmx_array = np.frombuffer(all_bytes, dtype=np.uint8)
        num_dmx = len(dmx_array)
        total_pixels = self._width * self._height
        
        # Clear buffer and set alpha
        self._frame_buffer.fill(0)
        self._frame_buffer[:, :, 3] = 255
        
        is_bgr = self._fourcc in ("BGRX", "BGRA")
        
        if self._channel_depth == "rgb":
            # Pack 3 DMX channels per pixel: R, G, B each carry a
            # separate DMX value. This is 3x more efficient.
            num_pixels = min(num_dmx // 3, total_pixels)
            remainder = num_dmx % 3 if num_pixels < total_pixels else 0
            
            for i in range(num_pixels):
                row = i // self._width
                col = i % self._width
                di = i * 3  # DMX index
                if is_bgr:
                    self._frame_buffer[row, col, 0] = dmx_array[di + 2]  # B = ch[i+2]
                    self._frame_buffer[row, col, 1] = dmx_array[di + 1]  # G = ch[i+1]
                    self._frame_buffer[row, col, 2] = dmx_array[di]      # R = ch[i]
                else:
                    self._frame_buffer[row, col, 0] = dmx_array[di]      # R = ch[i]
                    self._frame_buffer[row, col, 1] = dmx_array[di + 1]  # G = ch[i+1]
                    self._frame_buffer[row, col, 2] = dmx_array[di + 2]  # B = ch[i+2]
            
            # Handle leftover channels (1 or 2 remaining)
            if remainder > 0 and num_pixels < total_pixels:
                row = num_pixels // self._width
                col = num_pixels % self._width
                di = num_pixels * 3
                if is_bgr:
                    if remainder >= 1:
                        self._frame_buffer[row, col, 2] = dmx_array[di]      # R
                    if remainder >= 2:
                        self._frame_buffer[row, col, 1] = dmx_array[di + 1]  # G
                else:
                    if remainder >= 1:
                        self._frame_buffer[row, col, 0] = dmx_array[di]      # R
                    if remainder >= 2:
                        self._frame_buffer[row, col, 1] = dmx_array[di + 1]  # G
        else:
            # Grayscale: R=G=B=DMX value (1 DMX channel per pixel)
            num_channels = min(num_dmx, total_pixels)
            
            if is_bgr:
                for i in range(num_channels):
                    row = i // self._width
                    col = i % self._width
                    v = dmx_array[i]
                    self._frame_buffer[row, col, 0] = v  # B
                    self._frame_buffer[row, col, 1] = v  # G
                    self._frame_buffer[row, col, 2] = v  # R
            else:
                for i in range(num_channels):
                    row = i // self._width
                    col = i % self._width
                    v = dmx_array[i]
                    self._frame_buffer[row, col, 0] = v  # R
                    self._frame_buffer[row, col, 1] = v  # G
                    self._frame_buffer[row, col, 2] = v  # B
        
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
