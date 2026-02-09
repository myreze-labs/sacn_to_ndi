"""
Web Server Module

Provides a FastAPI-based web UI and API for monitoring and controlling
the sACN to NDI bridge.

NOTE: We intentionally do NOT use `from __future__ import annotations`
here because FastAPI needs runtime access to type annotations for
request body parsing and dependency injection.
"""

import asyncio
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)


class BridgeWebServer:
    """
    Web server providing GUI and API for the sACN-to-NDI bridge.

    Exposes:
    - GET  /              -> HTML GUI
    - GET  /api/status    -> bridge status + stats
    - GET  /api/dmx       -> current DMX values (all universes)
    - GET  /api/dmx/{u}   -> current DMX values for universe u
    - POST /api/bridge/start  -> start the bridge
    - POST /api/bridge/stop   -> stop the bridge
    - POST /api/config        -> update configuration
    - WS   /ws/live           -> live DMX data stream
    """

    def __init__(self) -> None:
        self._app = FastAPI(
            title="sACN to NDI Bridge",
            version="0.1.0",
        )
        self._bridge = None  # SACNtoNDIBridge
        self._bridge_factory = None  # Callable
        self._ws_clients: list = []

        self._setup_routes()

    @property
    def app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self._app

    def set_bridge(self, bridge: Any) -> None:
        """Set the bridge instance to monitor/control."""
        self._bridge = bridge

    def set_bridge_factory(self, factory: Callable) -> None:
        """Set factory callable for creating bridges from config."""
        self._bridge_factory = factory

    def _setup_routes(self) -> None:  # noqa: C901
        """Register all API routes."""

        @self._app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            html_path = (
                Path(__file__).parent / "static" / "index.html"
            )
            if not html_path.exists():
                return HTMLResponse(
                    content="<h1>Error: static/index.html not found</h1>"
                    f"<p>Looked at: {html_path}</p>",
                    status_code=500,
                )
            return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

        @self._app.get("/api/status")
        async def get_status() -> JSONResponse:
            if self._bridge is None:
                return JSONResponse(
                    {"running": False, "error": "No bridge configured"}
                )
            try:
                return JSONResponse(self._bridge.get_stats())
            except Exception as exc:
                return JSONResponse(
                    {"error": f"Stats error: {exc}"}, status_code=500
                )

        @self._app.get("/api/dmx")
        async def get_dmx_all() -> JSONResponse:
            if self._bridge is None:
                return JSONResponse(
                    {"error": "No bridge configured"},
                    status_code=503,
                )
            return JSONResponse(self._bridge.get_dmx_snapshot())

        @self._app.get("/api/dmx/{universe}")
        async def get_dmx_universe(universe: int) -> JSONResponse:
            if self._bridge is None:
                return JSONResponse(
                    {"error": "No bridge configured"},
                    status_code=503,
                )
            snapshot = self._bridge.get_dmx_snapshot()
            if universe not in snapshot:
                return JSONResponse(
                    {"error": f"Universe {universe} not found"},
                    status_code=404,
                )
            return JSONResponse({universe: snapshot[universe]})

        @self._app.post("/api/bridge/start")
        async def start_bridge() -> JSONResponse:
            if self._bridge is None:
                return JSONResponse(
                    {"error": "No bridge configured"},
                    status_code=503,
                )
            if self._bridge.is_running:
                return JSONResponse({"status": "already_running"})
            try:
                self._bridge.start()
                return JSONResponse({"status": "started"})
            except Exception as exc:
                return JSONResponse(
                    {"error": str(exc)}, status_code=500
                )

        @self._app.post("/api/bridge/stop")
        async def stop_bridge() -> JSONResponse:
            if self._bridge is None:
                return JSONResponse(
                    {"error": "No bridge configured"},
                    status_code=503,
                )
            if not self._bridge.is_running:
                return JSONResponse({"status": "already_stopped"})
            try:
                self._bridge.stop()
                return JSONResponse({"status": "stopped"})
            except Exception as exc:
                return JSONResponse(
                    {"error": str(exc)}, status_code=500
                )

        @self._app.post("/api/config")
        async def update_config(request: Request) -> JSONResponse:
            """Update bridge config. Stops/restarts as needed."""
            if (
                self._bridge is None
                or self._bridge_factory is None
            ):
                return JSONResponse(
                    {"error": "No bridge or factory configured"},
                    status_code=503,
                )

            try:
                new_config = await request.json()
            except Exception as exc:
                return JSONResponse(
                    {"error": f"Invalid JSON body: {exc}"},
                    status_code=400,
                )

            from main import BridgeConfig, EncodingMode

            was_running = self._bridge.is_running
            if was_running:
                self._bridge.stop()

            current = self._bridge.config
            try:
                enc_mode = new_config.get(
                    "encoding_mode",
                    current.encoding_mode.value,
                )
                config = BridgeConfig(
                    universes=sorted(
                        new_config.get("universes", current.universes)
                    ),
                    ndi_source_name=new_config.get(
                        "ndi_source_name",
                        current.ndi_source_name,
                    ),
                    frame_rate=int(
                        new_config.get(
                            "frame_rate", current.frame_rate
                        )
                    ),
                    encoding_mode=EncodingMode(enc_mode),
                    use_multicast=new_config.get(
                        "use_multicast", current.use_multicast
                    ),
                    bind_address=new_config.get(
                        "bind_address", current.bind_address
                    ),
                    use_mock_ndi=new_config.get(
                        "use_mock_ndi", current.use_mock_ndi
                    ),
                    ndi_width=int(
                        new_config.get(
                            "ndi_width", current.ndi_width
                        )
                    ),
                    ndi_height=int(
                        new_config.get(
                            "ndi_height", current.ndi_height
                        )
                    ),
                    ndi_fourcc=new_config.get(
                        "ndi_fourcc", current.ndi_fourcc
                    ),
                )
            except (ValueError, TypeError) as exc:
                return JSONResponse(
                    {"error": f"Invalid config: {exc}"},
                    status_code=400,
                )

            self._bridge = self._bridge_factory(config)

            if was_running:
                try:
                    self._bridge.start()
                except Exception as exc:
                    return JSONResponse(
                        {"error": f"Failed to restart: {exc}"},
                        status_code=500,
                    )

            resp_config = {
                "universes": config.universes,
                "ndi_source_name": config.ndi_source_name,
                "frame_rate": config.frame_rate,
                "encoding_mode": config.encoding_mode.value,
                "use_multicast": config.use_multicast,
                "bind_address": config.bind_address,
                "use_mock_ndi": config.use_mock_ndi,
                "ndi_width": config.ndi_width,
                "ndi_height": config.ndi_height,
                "ndi_fourcc": config.ndi_fourcc,
            }
            return JSONResponse({
                "status": "reconfigured",
                "running": self._bridge.is_running,
                "config": resp_config,
            })

        @self._app.websocket("/ws/live")
        async def ws_live(ws: WebSocket) -> None:
            await ws.accept()
            self._ws_clients.append(ws)
            connected = True
            try:
                while connected:
                    if self._bridge is not None:
                        try:
                            payload = {
                                "type": "dmx",
                                "ts": time.time(),
                                "dmx": self._bridge.get_dmx_snapshot(),
                                "stats": self._bridge.get_stats(),
                            }
                        except Exception:
                            logger.error(
                                "Error building WS payload:\n%s",
                                traceback.format_exc(),
                            )
                            await asyncio.sleep(0.1)
                            continue
                        try:
                            await ws.send_text(json.dumps(payload))
                        except Exception:
                            # Any send failure means client is gone
                            connected = False
                            break
                    await asyncio.sleep(0.1)
            except Exception:
                # Catch-all for any disconnect variant
                pass
            finally:
                if ws in self._ws_clients:
                    self._ws_clients.remove(ws)
