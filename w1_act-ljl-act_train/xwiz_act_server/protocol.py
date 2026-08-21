"""Length-prefixed pickle transport used by the vendor W1 inference client."""

from __future__ import annotations

from collections.abc import Mapping
import pickle
import socket
import struct
from typing import Any


MAX_FRAME_BYTES = 64 * 1024 * 1024


class ProtocolError(RuntimeError):
    """Raised when a peer sends a malformed protocol frame."""


def recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        packet = connection.recv(size - len(chunks))
        if not packet:
            raise ProtocolError("connection closed before frame completed")
        chunks.extend(packet)
    return bytes(chunks)


def recv_message(connection: socket.socket) -> dict[str, Any]:
    payload_size = struct.unpack(">I", recv_exact(connection, 4))[0]
    if payload_size > MAX_FRAME_BYTES:
        raise ProtocolError(
            f"frame too large: {payload_size} bytes exceeds {MAX_FRAME_BYTES}"
        )
    try:
        message = pickle.loads(recv_exact(connection, payload_size))
    except Exception as exc:
        raise ProtocolError(f"invalid pickle payload: {exc}") from exc
    if not isinstance(message, Mapping):
        raise ProtocolError("message payload must be a mapping")
    return dict(message)


def send_message(connection: socket.socket, message: Any) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > MAX_FRAME_BYTES:
        raise ProtocolError(
            f"frame too large: {len(payload)} bytes exceeds {MAX_FRAME_BYTES}"
        )
    connection.sendall(struct.pack(">I", len(payload)))
    connection.sendall(payload)
