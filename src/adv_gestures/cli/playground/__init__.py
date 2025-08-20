"""Playground command for the CLI."""

from pathlib import Path

import typer

from ...config import Config
from ..common import (
    DEFAULT_USER_CONFIG_PATH,
    app,
    determine_gpu_usage,
    determine_mirror_mode,
)


@app.command(name="playground")
def playground_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    port: int = typer.Option(9810, "--port", help="Port to bind to"),
    open_browser: bool = typer.Option(False, "--open", help="Open browser after starting"),
    mirror: bool = typer.Option(False, "--mirror", help="Force mirror mode (overrides environment variable)"),
    no_mirror: bool = typer.Option(
        False, "--no-mirror", help="Force no mirror mode (overrides environment variable)"
    ),
    gpu: bool = typer.Option(False, "--gpu", help="Force GPU acceleration (overrides environment variable)"),
    no_gpu: bool = typer.Option(False, "--no-gpu", help="Force CPU processing (overrides environment variable)"),
    config_path: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help=f"Path to config file. Default: {DEFAULT_USER_CONFIG_PATH}"
    ),
) -> None:
    """Run the gesture recognition playground server."""
    # Load configuration
    config = Config.load(config_path)

    # Determine GPU usage
    use_gpu = determine_gpu_usage(gpu, no_gpu, config)

    # Determine mirror mode
    use_mirror = determine_mirror_mode(mirror, no_mirror, config)

    from .server import playground_server

    playground_server(host=host, port=port, open_browser=open_browser, use_gpu=use_gpu, default_mirror=use_mirror)
