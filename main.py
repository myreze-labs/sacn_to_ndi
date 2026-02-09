#!/usr/bin/env python3
"""
sACN to NDI Bridge

Receives DMX data via sACN/E1.31 and transmits it over NDI
as video frames and/or metadata.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from sacn_receiver import SACNReceiver, DMXData
from encoders import SingleUniverseEncoder, MultiUniverseEncoder, ResizableEncoder, DMXEncoder
from ndi_sender import NDISenderConfig, CyndilibNDISender, MockNDISender, NDISenderProtocol


class EncodingMode(Enum):
    """How to encode DMX data for NDI transport."""
    VIDEO_ONLY = "video"
    METADATA_ONLY = "metadata"
    BOTH = "both"


@dataclass
class BridgeConfig:
    """Configuration for the sACN to NDI bridge."""
    universes: list[int]
    ndi_source_name: str
    frame_rate: int = 30
    encoding_mode: EncodingMode = EncodingMode.VIDEO_ONLY
    use_multicast: bool = True
    bind_address: str = "0.0.0.0"
    use_mock_ndi: bool = False  # For testing without NDI SDK
    ndi_width: int = 512        # NDI output frame width
    ndi_height: int = 1         # NDI output frame height (1 = raw, >1 = resized)
    ndi_fourcc: str = "RGBA"    # Pixel format: RGBA, BGRX, BGRA


class SACNtoNDIBridge:
    """
    Main bridge class that connects sACN input to NDI output.
    
    Coordinates the receiver, encoder, and sender modules.
    """
    
    def __init__(self, config: BridgeConfig):
        self._config = config
        self._running = False
        self._stop_event = threading.Event()
        self._send_thread: threading.Thread | None = None
        self._frame_count: int = 0
        self._start_time: float | None = None
        
        # Initialize receiver
        self._receiver = SACNReceiver(
            universes=config.universes,
            use_multicast=config.use_multicast,
            bind_address=config.bind_address,
        )
        
        # Initialize encoder based on number of universes
        if len(config.universes) == 1:
            base_encoder: DMXEncoder = SingleUniverseEncoder()
        else:
            base_encoder = MultiUniverseEncoder(len(config.universes))
        
        # Wrap with resizable encoder for custom NDI output resolution
        self._encoder: DMXEncoder = ResizableEncoder(
            inner=base_encoder,
            width=config.ndi_width,
            height=config.ndi_height,
            fourcc=config.ndi_fourcc,
        )
        
        # Initialize sender with the configured resolution and format
        sender_config = NDISenderConfig(
            source_name=config.ndi_source_name,
            frame_rate=config.frame_rate,
            width=config.ndi_width,
            height=config.ndi_height,
            fourcc=config.ndi_fourcc,
        )
        
        if config.use_mock_ndi:
            self._sender: NDISenderProtocol = MockNDISender(sender_config)
        else:
            self._sender = CyndilibNDISender(sender_config)
    
    def start(self) -> None:
        """Start the bridge."""
        if self._running:
            return
        
        print(f"Starting sACN to NDI bridge...")
        print(f"  Universes: {self._config.universes}")
        print(f"  NDI Source: {self._config.ndi_source_name}")
        print(f"  Frame Rate: {self._config.frame_rate} fps")
        print(f"  Encoding: {self._config.encoding_mode.value}")
        print(f"  NDI Output: {self._config.ndi_width}x{self._config.ndi_height} {self._config.ndi_fourcc}")
        print(f"  Multicast: {self._config.use_multicast}")
        
        # Start components
        self._receiver.start()
        self._sender.start()
        
        # Start send thread
        self._stop_event.clear()
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()
        
        self._frame_count = 0
        self._start_time = time.time()
        self._running = True
        print("Bridge started. Press Ctrl+C to stop.")
    
    def stop(self) -> None:
        """Stop the bridge."""
        if not self._running:
            return
        
        print("\nStopping bridge...")
        
        # Signal send thread to stop
        self._stop_event.set()
        
        if self._send_thread is not None:
            self._send_thread.join(timeout=2.0)
        
        # Stop components
        self._receiver.stop()
        self._sender.stop()
        
        self._running = False
        print("Bridge stopped.")
    
    def _send_loop(self) -> None:
        """Main loop that sends NDI frames at the configured rate."""
        frame_period = 1.0 / self._config.frame_rate
        
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            
            try:
                self._send_frame()
            except Exception as e:
                print(f"Error sending frame: {e}")
            
            # Sleep for remaining time in frame period
            elapsed = time.perf_counter() - loop_start
            sleep_time = frame_period - elapsed
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)
    
    def _send_frame(self) -> None:
        """Encode and send current DMX data."""
        # Get current DMX data for all universes
        if len(self._config.universes) == 1:
            dmx_data = self._receiver.get_universe_data(self._config.universes[0])
            if dmx_data is None:
                return
        else:
            all_data = self._receiver.get_all_universe_data()
            dmx_data = [all_data.get(u) for u in self._config.universes]
            dmx_data = [d for d in dmx_data if d is not None]
            if not dmx_data:
                return
        
        # Encode and send based on mode
        mode = self._config.encoding_mode
        
        if mode in (EncodingMode.VIDEO_ONLY, EncodingMode.BOTH):
            frame = self._encoder.encode_video(dmx_data)
            self._sender.send_video_frame(frame)
        
        if mode in (EncodingMode.METADATA_ONLY, EncodingMode.BOTH):
            metadata = self._encoder.encode_metadata(dmx_data)
            self._sender.send_metadata(metadata)
        
        self._frame_count += 1
    
    @property
    def is_running(self) -> bool:
        """Check if bridge is running."""
        return self._running
    
    @property
    def config(self) -> BridgeConfig:
        """Get current configuration."""
        return self._config
    
    @property
    def receiver(self) -> SACNReceiver:
        """Get the sACN receiver instance."""
        return self._receiver
    
    @property
    def sender(self) -> NDISenderProtocol:
        """Get the NDI sender instance."""
        return self._sender
    
    def get_stats(self) -> dict:
        """Get bridge statistics for monitoring."""
        now = time.time()
        return {
            "running": self._running,
            "frame_count": self._frame_count,
            "uptime_s": round(now - self._start_time, 1) if self._start_time else 0,
            "config": {
                "universes": self._config.universes,
                "ndi_source_name": self._config.ndi_source_name,
                "frame_rate": self._config.frame_rate,
                "encoding_mode": self._config.encoding_mode.value,
                "use_multicast": self._config.use_multicast,
                "bind_address": self._config.bind_address,
                "use_mock_ndi": self._config.use_mock_ndi,
                "ndi_width": self._config.ndi_width,
                "ndi_height": self._config.ndi_height,
                "ndi_fourcc": self._config.ndi_fourcc,
            },
            "sacn": self._receiver.get_stats(),
            "ndi": {
                "running": self._sender.is_running,
                "num_connections": self._sender.num_connections,
            },
        }
    
    def get_dmx_snapshot(self) -> dict[int, list[int]]:
        """Get current DMX values for all universes as JSON-serializable dict."""
        result = {}
        for universe in self._config.universes:
            data = self._receiver.get_universe_data(universe)
            if data is not None:
                result[universe] = list(data.data)
        return result
    
    def wait(self) -> None:
        """Wait for bridge to stop (blocking)."""
        if self._send_thread is not None:
            self._send_thread.join()


def create_bridge(config: BridgeConfig) -> SACNtoNDIBridge:
    """Factory function for creating bridge instances."""
    return SACNtoNDIBridge(config)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="sACN to NDI Bridge - Convert DMX over sACN to NDI video/metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -u 1                     # Single universe, video encoding
  %(prog)s -u 1 2 3 -n "DMX Data"   # Multiple universes with custom name
  %(prog)s -u 1 -e metadata         # Metadata-only encoding
  %(prog)s -u 1 --unicast           # Unicast mode (no multicast join)
  %(prog)s -u 1 --mock              # Test without NDI SDK
  %(prog)s -u 1 --web               # Start with web UI on port 8080
  %(prog)s -u 1 --web --web-port 9000  # Custom web UI port
        """
    )
    
    parser.add_argument(
        "-u", "--universes",
        type=int,
        nargs="+",
        default=[1],
        help="sACN universe(s) to receive (default: 1)"
    )
    
    parser.add_argument(
        "-n", "--name",
        type=str,
        default=None,
        help="NDI source name (default: 'sACN Universe N' or 'sACN Universes N-M')"
    )
    
    parser.add_argument(
        "-f", "--fps",
        type=int,
        default=30,
        help="NDI frame rate (default: 30)"
    )
    
    parser.add_argument(
        "-e", "--encoding",
        type=str,
        choices=["video", "metadata", "both"],
        default="video",
        help="Encoding mode (default: video)"
    )
    
    parser.add_argument(
        "--unicast",
        action="store_true",
        help="Use unicast mode (don't join multicast groups)"
    )
    
    parser.add_argument(
        "--bind",
        type=str,
        default="0.0.0.0",
        help="Address to bind sACN receiver to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock NDI sender for testing"
    )
    
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start web UI for monitoring and control"
    )
    
    parser.add_argument(
        "--web-port",
        type=int,
        default=8080,
        help="Port for web UI (default: 8080)"
    )
    
    parser.add_argument(
        "--web-host",
        type=str,
        default="0.0.0.0",
        help="Host for web UI (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--resolution",
        type=str,
        default="512x1",
        help="NDI output resolution WxH (default: 512x1, e.g. 128x128)"
    )
    
    parser.add_argument(
        "--fourcc",
        type=str,
        choices=["RGBA", "BGRX", "BGRA"],
        default="RGBA",
        help="NDI pixel format (default: RGBA)"
    )
    
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="With --web: don't start bridge automatically; use the GUI to start"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Generate default NDI source name if not provided
    if args.name is None:
        if len(args.universes) == 1:
            ndi_name = f"sACN Universe {args.universes[0]}"
        else:
            ndi_name = f"sACN Universes {min(args.universes)}-{max(args.universes)}"
    else:
        ndi_name = args.name
    
    # Parse resolution
    try:
        res_parts = args.resolution.lower().split("x")
        ndi_width = int(res_parts[0])
        ndi_height = int(res_parts[1])
    except (ValueError, IndexError):
        print(f"Invalid resolution '{args.resolution}', expected WxH (e.g. 128x128)")
        return 1
    
    # Create config
    config = BridgeConfig(
        universes=sorted(args.universes),
        ndi_source_name=ndi_name,
        frame_rate=args.fps,
        encoding_mode=EncodingMode(args.encoding),
        use_multicast=not args.unicast,
        bind_address=args.bind,
        use_mock_ndi=args.mock,
        ndi_width=ndi_width,
        ndi_height=ndi_height,
        ndi_fourcc=args.fourcc,
    )
    
    # Create bridge
    bridge = SACNtoNDIBridge(config)
    
    if args.web:
        # ── Web UI mode ──
        import uvicorn
        from web_server import BridgeWebServer
        
        server = BridgeWebServer()
        server.set_bridge(bridge)
        server.set_bridge_factory(create_bridge)
        
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            bridge.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        if not args.no_autostart:
            bridge.start()
        
        print(f"Web UI available at http://{args.web_host}:{args.web_port}")
        print("Press Ctrl+C to stop.")
        
        try:
            uvicorn.run(
                server.app,
                host=args.web_host,
                port=args.web_port,
                log_level="info",
            )
        except KeyboardInterrupt:
            pass
        finally:
            bridge.stop()
    else:
        # ── Headless mode (original behavior) ──
        def signal_handler(sig, frame):
            bridge.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            bridge.start()
            bridge.wait()
        except KeyboardInterrupt:
            pass
        finally:
            bridge.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
