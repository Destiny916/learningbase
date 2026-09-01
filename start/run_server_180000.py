"""Python 3.10-compatible entrypoint for the isolated 180000 server."""

from __future__ import annotations

import runpy
import typing

import typing_extensions


# The deployed PC2 interpreter is Python 3.10; ACT-DINOv3's LeRobot source
# imports typing names introduced after 3.10.
if not hasattr(typing, "Self"):
    typing.Self = typing_extensions.Self
if not hasattr(typing, "Unpack"):
    typing.Unpack = typing_extensions.Unpack

runpy.run_module("xwiz_act_server.server_160000", run_name="__main__")
