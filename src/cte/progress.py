from __future__ import annotations

import sys
from typing import TextIO


class ProgressReporter:
    def __init__(self, *, enabled: bool, stream: TextIO | None = None) -> None:
        self.enabled = enabled
        self.stream = sys.stderr if stream is None else stream

    def stage(self, stage: str, message: str) -> None:
        if not self.enabled:
            return
        print(f"[cte][{stage}] {message}", file=self.stream, flush=True)

    def doc(self, stage: str, current: int, total: int, doc_name: str) -> None:
        if not self.enabled:
            return
        print(f"[cte][{stage}] {current}/{total} {doc_name}", file=self.stream, flush=True)
