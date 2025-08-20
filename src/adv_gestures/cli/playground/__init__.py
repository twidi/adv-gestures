"""Playground command for the CLI."""

from pathlib import Path

import typer

from ...config import Config
from .. import options
from ..common import (
    app,
    determine_gpu_usage,
    determine_mirror_mode,
)


@app.command(name="playground")
def playground_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    port: int = typer.Option(9810, "--port", help="Port to bind to"),
    open_browser: bool = typer.Option(False, "--open", help="Open browser after starting"),
    mirror: bool | None = options.mirror,
    gpu: bool | None = options.gpu,
    config_path: Path | None = options.config,  # noqa: B008
) -> None:
    """Run the gesture recognition playground server."""
    # Load configuration
    config = Config.load(config_path)

    # Determine GPU usage
    use_gpu = determine_gpu_usage(gpu, config)

    # Determine mirror mode
    use_mirror = determine_mirror_mode(mirror, config)

    from .server import playground_server

    playground_server(host=host, port=port, open_browser=open_browser, use_gpu=use_gpu, default_mirror=use_mirror)
