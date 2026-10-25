# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""``torch.cuda.current_stream`` for kernel-launch hot paths.

``torch.cuda.current_stream()`` builds a fresh ``torch.cuda.Stream`` wrapper on
every call (device-index resolution, lazy-init check, ``Stream.__new__``), a few
microseconds per launch that add up over the hundreds of FlyDSL launches of a
prefill step. The wrapper is a plain handle: two wrappers of the same
(stream_id, device_index, device_type) are equal and drive the same stream. This
returns a memoised wrapper for the *current* stream -- queried from the runtime
on every call, so stream switches (``torch.cuda.stream(...)`` contexts, graph
capture streams) are honoured exactly -- and defers to torch for anything else.
"""

import torch

_STREAMS: dict = {}


def current_stream(device=None):
    """Same stream as ``torch.cuda.current_stream(device)``."""
    if not torch.cuda.is_initialized():
        return torch.cuda.current_stream(device)
    if device is None:
        index = torch._C._cuda_getDevice()
    elif (
        type(device) is torch.device
        and device.type == "cuda"
        and device.index is not None
    ):
        index = device.index
    else:
        return torch.cuda.current_stream(device)
    key = torch._C._cuda_getCurrentStream(index)
    stream = _STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(
            stream_id=key[0], device_index=key[1], device_type=key[2]
        )
        _STREAMS[key] = stream
    return stream
