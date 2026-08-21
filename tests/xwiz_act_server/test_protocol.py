import socket
import struct

import pytest

from xwiz_act_server.protocol import (
    MAX_FRAME_BYTES,
    ProtocolError,
    recv_message,
    send_message,
)


def test_round_trip_frame_over_socketpair():
    left, right = socket.socketpair()
    try:
        send_message(left, {"type": "STATUS", "request_id": 7})
        assert recv_message(right) == {"type": "STATUS", "request_id": 7}
    finally:
        left.close()
        right.close()


def test_rejects_frame_larger_than_limit():
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack(">I", MAX_FRAME_BYTES + 1))
        with pytest.raises(ProtocolError, match="too large"):
            recv_message(right)
    finally:
        left.close()
        right.close()


def test_rejects_non_mapping_payload():
    left, right = socket.socketpair()
    try:
        send_message(left, ["not", "a", "mapping"])
        with pytest.raises(ProtocolError, match="mapping"):
            recv_message(right)
    finally:
        left.close()
        right.close()
