#!/usr/bin/env python3
"""
NDI Receiver Example

Demonstrates how to receive and decode DMX data from the sACN-to-NDI bridge.
This is a reference implementation for the receiving side.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
import threading

import numpy as np

from encoders import (
    decode_video_single_universe,
    decode_video_multi_universe,
    decode_metadata_xml,
)


class NDIDMXReceiver:
    """
    Receives NDI video frames and extracts DMX data.
    
    Uses cyndilib to receive NDI frames and decodes them
    using the encoder module's decode functions.
    """
    
    def __init__(
        self,
        source_name: str | None = None,
        on_dmx_update: callable = None,
    ):
        """
        Initialize NDI DMX receiver.
        
        Args:
            source_name: Specific NDI source to connect to (None = first found)
            on_dmx_update: Callback(universe: int, data: bytearray) for DMX updates
        """
        self._source_name = source_name
        self._on_dmx_update = on_dmx_update
        self._receiver = None
        self._running = False
        self._stop_event = threading.Event()
        self._receive_thread: threading.Thread | None = None
        
        # Latest DMX data per universe
        self._dmx_data: dict[int, bytearray] = {}
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start receiving NDI frames."""
        if self._running:
            return
        
        from cyndilib.finder import Finder
        from cyndilib.receiver import Receiver
        from cyndilib.video_frame import VideoRecvFrame
        from cyndilib.metadata_frame import MetadataRecvFrame
        
        # Find NDI sources
        print("Searching for NDI sources...")
        finder = Finder()
        finder.open()
        
        # Wait for sources to be discovered
        time.sleep(2.0)
        
        sources = finder.get_source_names()
        if not sources:
            raise RuntimeError("No NDI sources found")
        
        print(f"Found sources: {sources}")
        
        # Select source
        if self._source_name:
            if self._source_name not in sources:
                raise RuntimeError(f"Source '{self._source_name}' not found")
            source_name = self._source_name
        else:
            source_name = sources[0]
            print(f"Using first source: {source_name}")
        
        # Create receiver
        self._receiver = Receiver()
        self._receiver.set_source_name(source_name)
        self._receiver.open()
        
        # Create frame objects
        self._video_frame = VideoRecvFrame()
        self._metadata_frame = MetadataRecvFrame()
        
        # Start receive thread
        self._stop_event.clear()
        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()
        
        self._running = True
        print(f"Connected to: {source_name}")
    
    def stop(self) -> None:
        """Stop receiving."""
        if not self._running:
            return
        
        self._stop_event.set()
        
        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
        
        if self._receiver:
            self._receiver.close()
            self._receiver = None
        
        self._running = False
        print("Receiver stopped.")
    
    def get_dmx_data(self, universe: int = 1) -> bytearray | None:
        """Get latest DMX data for a universe."""
        with self._lock:
            return self._dmx_data.get(universe)
    
    def _receive_loop(self) -> None:
        """Main receive loop."""
        from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
        
        while not self._stop_event.is_set():
            try:
                # Try to receive a frame (with timeout)
                frame_type = self._receiver.receive(
                    video_frame=self._video_frame,
                    metadata_frame=self._metadata_frame,
                    timeout_ms=100,
                )
                
                if frame_type == "video":
                    self._handle_video_frame()
                elif frame_type == "metadata":
                    self._handle_metadata_frame()
                
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"Receive error: {e}")
                time.sleep(0.1)
    
    def _handle_video_frame(self) -> None:
        """Process received video frame."""
        # Get frame data as numpy array
        data = self._video_frame.get_data()
        if data is None:
            return
        
        height = self._video_frame.yres
        
        with self._lock:
            if height == 1:
                # Single universe
                dmx = decode_video_single_universe(data)
                self._dmx_data[1] = dmx
                if self._on_dmx_update:
                    self._on_dmx_update(1, dmx)
            else:
                # Multiple universes
                universes = decode_video_multi_universe(data)
                for i, dmx in enumerate(universes, start=1):
                    self._dmx_data[i] = dmx
                    if self._on_dmx_update:
                        self._on_dmx_update(i, dmx)
    
    def _handle_metadata_frame(self) -> None:
        """Process received metadata frame."""
        xml_data = self._metadata_frame.get_data()
        if not xml_data:
            return
        
        try:
            dmx_map = decode_metadata_xml(xml_data)
            
            with self._lock:
                for universe, dmx in dmx_map.items():
                    self._dmx_data[universe] = dmx
                    if self._on_dmx_update:
                        self._on_dmx_update(universe, dmx)
        except Exception as e:
            print(f"Metadata decode error: {e}")


def print_dmx_update(universe: int, data: bytearray) -> None:
    """Example callback that prints DMX channel values."""
    # Show first 16 channels
    values = [str(data[i]) for i in range(min(16, len(data)))]
    print(f"Universe {universe}: [{', '.join(values)}, ...]")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NDI DMX Receiver - Receive DMX data from sACN-to-NDI bridge"
    )
    
    parser.add_argument(
        "-s", "--source",
        type=str,
        default=None,
        help="NDI source name to connect to (default: first found)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print DMX values as they arrive"
    )
    
    args = parser.parse_args()
    
    # Create receiver
    callback = print_dmx_update if args.verbose else None
    receiver = NDIDMXReceiver(
        source_name=args.source,
        on_dmx_update=callback,
    )
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        receiver.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        receiver.start()
        
        # Main loop - just keep running
        while True:
            time.sleep(1.0)
            
            if not args.verbose:
                # Periodically print status
                dmx = receiver.get_dmx_data(1)
                if dmx:
                    print(f"Channel 1: {dmx[0]}, Channel 2: {dmx[1]}, ...")
    
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
