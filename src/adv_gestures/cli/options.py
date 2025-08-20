"""Shared CLI option definitions."""

from __future__ import annotations

import typer

from .common import DEFAULT_USER_CONFIG_PATH

camera = typer.Option(None, "--camera", "--cam", "-cm", help="Camera name filter (case insensitive)")

preview = typer.Option(True, "--preview/--no-preview", "-p/-np", help="Show visual preview window")

mirror = typer.Option(
    None, "--mirror/--no-mirror", "-m/-nm", help="Force mirror mode (overrides environment variable)"
)

size = typer.Option(None, "--size", "-s", help="Maximum dimension of the camera capture")

config = typer.Option(None, "--config", "-c", help=f"Path to config file. Default: {DEFAULT_USER_CONFIG_PATH}")

gpu = typer.Option(None, "--gpu/--no-gpu", "-g/-ng", help="Force GPU acceleration (overrides environment variable)")
