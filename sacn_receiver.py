"""
sACN (E1.31) Receiver Module

Handles receiving DMX data over sACN/E1.31 protocol.
Can be replaced with other DMX input sources (Art-Net, USB DMX, etc.)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import sacn


@dataclass
class DMXData:
    """Container for DMX universe data."""
    universe: int
    data: bytearray = field(default_factory=lambda: bytearray(512))
    sequence: int = 0
    priority: int = 100
    
    def copy(self) -> DMXData:
        """Create a copy of this DMX data."""
        new_data = DMXData(universe=self.universe)
        new_data.data[:] = self.data
        new_data.sequence = self.sequence
        new_data.priority = self.priority
        return new_data


class DMXReceiverProtocol(Protocol):
    """Protocol for DMX receivers - allows swapping implementations."""
    
    def start(self) -> None:
        """Start receiving DMX data."""
        ...
    
    def stop(self) -> None:
        """Stop receiving DMX data."""
        ...
    
    def get_universe_data(self, universe: int) -> DMXData | None:
        """Get current DMX data for a universe."""
        ...
    
    def set_callback(self, callback: Callable[[DMXData], None]) -> None:
        """Set callback for DMX data updates."""
        ...


class SACNReceiver:
    """
    sACN/E1.31 DMX receiver implementation.
    
    Receives DMX data via sACN multicast or unicast and stores
    the latest values per universe.
    """
    
    def __init__(
        self,
        universes: list[int],
        use_multicast: bool = True,
        bind_address: str = "0.0.0.0",
    ):
        """
        Initialize sACN receiver.
        
        Args:
            universes: List of universe numbers to listen to (1-63999)
            use_multicast: Whether to join multicast groups for universes
            bind_address: Address to bind the receiver to
        """
        self._universes = universes
        self._use_multicast = use_multicast
        self._bind_address = bind_address
        
        self._receiver: sacn.sACNreceiver | None = None
        self._universe_data: dict[int, DMXData] = {}
        self._lock = threading.Lock()
        self._callback: Callable[[DMXData], None] | None = None
        self._running = False
        
        # Stats tracking
        self._packet_counts: dict[int, int] = {}
        self._last_packet_times: dict[int, float] = {}
        self._start_time: float | None = None
        
        # Initialize data containers for each universe
        for universe in universes:
            self._universe_data[universe] = DMXData(universe=universe)
            self._packet_counts[universe] = 0
            self._last_packet_times[universe] = 0.0
    
    def set_callback(self, callback: Callable[[DMXData], None]) -> None:
        """Set callback for DMX data updates."""
        self._callback = callback
    
    def start(self) -> None:
        """Start the sACN receiver."""
        if self._running:
            return
        
        self._receiver = sacn.sACNreceiver(bind_address=self._bind_address)
        self._receiver.start()
        
        # Register listeners for each universe
        for universe in self._universes:
            self._register_universe_listener(universe)
            
            if self._use_multicast:
                self._receiver.join_multicast(universe)
        
        self._start_time = time.time()
        self._running = True
    
    def stop(self) -> None:
        """Stop the sACN receiver."""
        if not self._running or self._receiver is None:
            return
        
        if self._use_multicast:
            for universe in self._universes:
                try:
                    self._receiver.leave_multicast(universe)
                except Exception:
                    pass  # May already have left
        
        self._receiver.stop()
        self._receiver = None
        self._running = False
    
    def get_universe_data(self, universe: int) -> DMXData | None:
        """
        Get a copy of current DMX data for a universe.
        
        Returns None if universe is not being listened to.
        """
        with self._lock:
            if universe not in self._universe_data:
                return None
            return self._universe_data[universe].copy()
    
    def get_all_universe_data(self) -> dict[int, DMXData]:
        """Get copies of current DMX data for all universes."""
        with self._lock:
            return {u: d.copy() for u, d in self._universe_data.items()}
    
    def _register_universe_listener(self, universe: int) -> None:
        """Register a listener callback for a specific universe."""
        if self._receiver is None:
            raise RuntimeError("Receiver not started")
        
        @self._receiver.listen_on("universe", universe=universe)
        def _on_packet(packet: sacn.DataPacket) -> None:
            self._handle_packet(universe, packet)
    
    def _handle_packet(self, universe: int, packet: sacn.DataPacket) -> None:
        """Handle incoming sACN packet."""
        # Only process DMX data packets (start code 0x00)
        if packet.dmxStartCode != 0x00:
            return
        
        with self._lock:
            dmx_data = self._universe_data.get(universe)
            if dmx_data is None:
                return
            
            # Update DMX data
            data = packet.dmxData
            data_len = len(data)
            dmx_data.data[:data_len] = bytes(data)
            
            # Zero out remaining channels if packet is short
            if data_len < 512:
                dmx_data.data[data_len:] = b"\x00" * (512 - data_len)
            
            dmx_data.sequence = packet.sequence
            dmx_data.priority = packet.priority
            
            # Update stats
            self._packet_counts[universe] = self._packet_counts.get(universe, 0) + 1
            self._last_packet_times[universe] = time.time()
            
            # Call callback with a copy
            if self._callback is not None:
                self._callback(dmx_data.copy())
    
    @property
    def is_running(self) -> bool:
        """Check if receiver is running."""
        return self._running
    
    @property
    def universes(self) -> list[int]:
        """Get list of universes being listened to."""
        return list(self._universes)
    
    def get_stats(self) -> dict:
        """Get receiver statistics."""
        now = time.time()
        with self._lock:
            universe_stats = {}
            for u in self._universes:
                last_time = self._last_packet_times.get(u, 0.0)
                age = (now - last_time) if last_time > 0 else None
                universe_stats[u] = {
                    "packet_count": self._packet_counts.get(u, 0),
                    "last_packet_age_s": round(age, 2) if age is not None else None,
                    "receiving": age is not None and age < 5.0,
                }
            return {
                "running": self._running,
                "uptime_s": round(now - self._start_time, 1) if self._start_time else 0,
                "universes": universe_stats,
            }
