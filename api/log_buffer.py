"""
Keeps the most recent log lines in memory so the frontend's Console
panel can display real backend activity - not a separate logging
system, just a small buffer that taps into the same logging your
terminal already shows.
"""

import logging
from collections import deque

log_buffer: deque[str] = deque(maxlen=200)


class _BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "/logs" in message:
            return  # don't let polling this endpoint spam itself
        log_buffer.append(self.format(record))


def attach_log_buffer() -> None:
    """Call once, at startup, to start capturing log lines."""
    handler = _BufferHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    # Uvicorn's own loggers don't forward to the root logger by default,
    # so listening there alone misses every request log line. Attach
    # directly to uvicorn's loggers too, not just root.
    logging.getLogger().addHandler(handler)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).addHandler(handler)