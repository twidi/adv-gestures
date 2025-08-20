import asyncio
import json
import logging
import os
import queue
import threading
import webbrowser
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np
from aiohttp import web  # type: ignore[import-not-found]
from aiohttp_sse import sse_response  # type: ignore[import-not-found]

if TYPE_CHECKING:
    # if imported normally, it, for whatever reason, blocks opencv to open a window in the run command
    from aiortc import RTCPeerConnection  # type: ignore[import-not-found]

from ...config import Config
from ...models.hands import Hands
from ...models.utils import InfinityJSONEncoder
from ...recognizer import Recognizer

logger = logging.getLogger("adv_gestures.playground")
logger.setLevel(logging.INFO)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"

# Configure handler if logger doesn't have one
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False  # Don't propagate to root logger


# Store session data per UID
class Session:
    def __init__(self, uid: str, mirror: bool, use_gpu: bool = True):
        self.uid = uid
        self.recognizer = Recognizer(
            model_path=os.getenv("GESTURE_RECOGNIZER_MODEL_PATH", "").strip() or "gesture_recognizer.task",
            use_gpu=use_gpu,
            mirroring=mirror,
        )
        logger.info(f"Recognizer initialized for UID {uid} with mirroring={mirror}, use_gpu={use_gpu}")
        self.hands = Hands(Config())
        self.pc: RTCPeerConnection | None = None
        self.sse_connection: Any | None = None

        # Queues for async-to-sync bridge
        self.frame_queue: queue.Queue[np.ndarray[Any, Any] | None] = queue.Queue(maxsize=5)
        self.result_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.processing_thread: threading.Thread | None = None
        self.stop_processing = False

    def frames_generator(self) -> Iterator[np.ndarray[Any, Any]]:
        """Generator that yields frames from the queue."""
        while not self.stop_processing:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if frame is None:  # Sentinel value
                    break
                yield frame
            except queue.Empty:
                continue

    def process_frames(self) -> None:
        """Process frames using the recognizer (runs in separate thread)."""
        try:
            for _frame, _stream_info, _result in self.recognizer.handle_frames_from_opencv(
                self.frames_generator(), self.hands
            ):
                # Convert hands data to dict and put in result queue
                hands_data = self.hands.to_dict()
                self.result_queue.put(hands_data)
        except Exception as e:
            logger.exception(f"Error in frame processing thread for {self.uid}: {e}")
        finally:
            # Signal that processing is done
            self.stop_processing = True

    def start_processing(self) -> None:
        """Start the frame processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_processing = False
            self.processing_thread = threading.Thread(target=self.process_frames)
            self.processing_thread.start()

    def stop(self) -> None:
        """Stop processing and clean up."""
        self.stop_processing = True
        # Put sentinel value to unblock the generator
        try:
            self.frame_queue.put(None, block=False)
        except queue.Full:
            pass
        # Wait for thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1)


sessions: dict[str, Session] = {}


async def index(request: web.Request) -> web.FileResponse:
    """Serve the main HTML page."""
    index_file = STATIC_DIR / "index.html"

    return web.FileResponse(
        index_file,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"},
    )


async def serve_static(request: web.Request) -> web.FileResponse:
    """Serve static files with no-cache headers."""
    filename = request.match_info["filename"]

    # Security: prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        raise web.HTTPNotFound()

    file_path = STATIC_DIR / filename

    # Resolve to absolute path and ensure it's within STATIC_DIR
    try:
        file_path = file_path.resolve()
        file_path.relative_to(STATIC_DIR.resolve())
    except (ValueError, RuntimeError):
        # Path is outside STATIC_DIR
        raise web.HTTPNotFound() from None

    if not file_path.exists() or not file_path.is_file():
        raise web.HTTPNotFound()

    return web.FileResponse(
        file_path,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"},
    )


async def healthz(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.Response(text="OK", status=200)


async def webrtc_offer(request: web.Request) -> web.Response:
    from aiortc import (  # type: ignore[import-not-found]
        RTCPeerConnection,
        RTCSessionDescription,
    )

    """Handle WebRTC offer and return answer."""
    uid = request.query.get("uid")

    if not uid:
        return web.Response(text="Missing UID", status=400)

    # Validate UID format
    try:
        UUID(uid)
    except ValueError:
        return web.Response(text="Invalid UID format", status=400)

    # Parse offer
    try:
        data = await request.json()
        sdp = data.get("sdp")
        mirror = data.get("mirror", True)

        if not sdp:
            return web.Response(text="Missing SDP", status=400)
    except Exception as e:
        logger.error(f"Failed to parse offer: {e}")
        return web.Response(text="Invalid JSON", status=400)

    # Create session if not exists
    if uid not in sessions:
        use_gpu = request.app.get("use_gpu", True)
        session = Session(uid, mirror, use_gpu)
        sessions[uid] = session
    else:
        session = sessions[uid]

    # Create peer connection
    pc = RTCPeerConnection()
    session.pc = pc

    @pc.on("track")  # type: ignore[misc]
    async def on_track(track: Any) -> None:
        """Handle incoming video track."""
        if track.kind == "video":
            # Start the processing thread
            session.start_processing()

            # Process video frames
            async def receive_frames() -> None:
                """Receive frames from WebRTC and put them in the queue."""
                from aiortc import MediaStreamError  # type: ignore[import-not-found]

                try:
                    while True:
                        frame = await track.recv()

                        # Convert aiortc frame to numpy array
                        img = frame.to_ndarray(format="bgr24")

                        # Try to put frame in queue (drop if full)
                        try:
                            session.frame_queue.put(img, block=False)
                        except queue.Full:
                            # Drop frame if queue is full
                            pass

                except MediaStreamError:
                    # This is normal when the stream ends (e.g., browser tab closed)
                    logger.info(f"Media stream ended for {uid}")
                except Exception as e:
                    # This is an actual error
                    logger.error(f"Frame receiving error for {uid}: {e}")
                finally:
                    # Stop processing when done
                    session.stop()

            async def send_results() -> None:
                """Send results from the processing thread via SSE."""
                try:
                    while not session.stop_processing:
                        try:
                            # Check for results with a small timeout
                            hands_data = await asyncio.get_event_loop().run_in_executor(
                                None, session.result_queue.get, True, 0.1
                            )

                            if session.sse_connection:
                                try:
                                    await session.sse_connection.send(
                                        json.dumps(hands_data, cls=InfinityJSONEncoder), event="snapshot"
                                    )
                                except Exception as e:
                                    logger.error(f"Failed to send SSE for {uid}: {e}")

                        except queue.Empty:
                            await asyncio.sleep(0.005)

                except Exception as e:
                    logger.error(f"Result sending error for {uid}: {e}")

            # Start both tasks
            asyncio.create_task(receive_frames())
            asyncio.create_task(send_results())

    @pc.on("connectionstatechange")  # type: ignore[misc]
    async def on_connectionstatechange() -> None:
        logger.info(f"Connection state for {uid}: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            await cleanup_session(uid)

    # Set remote description and create answer
    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp, "offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({"sdp": pc.localDescription.sdp})
    except Exception as e:
        logger.error(f"WebRTC negotiation failed for {uid}: {e}")
        await cleanup_session(uid)
        return web.Response(text="WebRTC negotiation failed", status=500)


async def sse_handler(request: web.Request) -> web.StreamResponse:
    """Handle SSE connection for real-time updates."""
    uid = request.query.get("uid")

    if not uid:
        return web.Response(text="Missing UID", status=400)

    # Validate UID format
    try:
        UUID(uid)
    except ValueError:
        return web.Response(text="Invalid UID format", status=400)

    # Extract mirror parameter from query string
    mirror_str = request.query.get("mirror", "true")
    mirror = mirror_str.lower() == "true"

    # Create session if not exists
    if uid not in sessions:
        use_gpu = request.app.get("use_gpu", True)
        session = Session(uid, mirror, use_gpu)
        sessions[uid] = session
    else:
        session = sessions[uid]

    async with sse_response(request) as resp:
        session.sse_connection = resp

        try:
            # Send connected event
            await resp.send("", event="connected")

            # Keep connection alive
            while True:
                await asyncio.sleep(5)  # Heartbeat
                if uid not in sessions or session.sse_connection is None:
                    break

        except Exception as e:
            logger.error(f"SSE error for {uid}: {e}")
            try:
                await resp.send(json.dumps({"error": str(e)}, cls=InfinityJSONEncoder), event="error")
            except Exception:
                pass
        finally:
            logger.info(f"Closing SSE connection for {uid}")
            await cleanup_session(uid)

    return resp


async def cleanup_session(uid: str) -> None:
    """Clean up resources for a session."""
    if uid not in sessions:
        return

    session = sessions[uid]

    # Stop processing thread
    session.stop()

    # Remove SSE connection
    session.sse_connection = None

    # Close peer connection
    if session.pc:
        await session.pc.close()
        session.pc = None

    # Close recognizer
    session.recognizer.close()

    # Remove session
    del sessions[uid]

    logger.info(f"Cleaned up session {uid}")


async def on_shutdown(app: web.Application) -> None:
    """Clean up all sessions on shutdown."""
    # Close all sessions
    for uid in list(sessions.keys()):
        await cleanup_session(uid)


def create_app() -> web.Application:
    """Create the aiohttp application."""
    app = web.Application()

    # Routes
    app.router.add_get("/", index)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/static/{filename:.+}", serve_static)
    app.router.add_post("/webrtc/offer", webrtc_offer)
    app.router.add_get("/sse", sse_handler)

    # Cleanup on shutdown
    app.on_shutdown.append(on_shutdown)

    return app


def playground_server(
    host: str = "127.0.0.1",
    port: int = 9810,
    open_browser: bool = False,
    use_gpu: bool = True,
) -> None:
    """Run the gesture recognition playground server."""
    app = create_app()
    app["use_gpu"] = use_gpu

    async def start_server() -> None:
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        logger.info(f"Playground server running at http://{host}:{port}")

        if open_browser:
            # Wait a bit for server to be ready
            await asyncio.sleep(0.5)

            # Check if server is responding
            import aiohttp

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"http://{host}:{port}/healthz") as resp:
                        if resp.status == 200:
                            webbrowser.open(f"http://{host}:{port}")
                except Exception:
                    pass

        # Keep running
        await asyncio.Event().wait()

    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
