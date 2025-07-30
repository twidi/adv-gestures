"""Playground command for the CLI."""

import typer

from ..common import app


@app.command(name="playground")
def playground_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    port: int = typer.Option(9810, "--port", help="Port to bind to"),
    open_browser: bool = typer.Option(False, "--open", help="Open browser after starting"),
) -> None:
    """Run the gesture recognition playground server."""
    from .server import playground_server

    playground_server(host=host, port=port, open_browser=open_browser)
