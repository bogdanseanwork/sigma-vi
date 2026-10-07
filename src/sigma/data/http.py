"""Polite HTTP: per-provider rate limits, retries with backoff, redacted errors, streaming downloads."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from sigma.core.security import redact


class RateLimiter:
    """Evenly spaces calls so a provider's per-minute limit is never exceeded."""

    def __init__(self, per_minute: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.interval = 60.0 / per_minute
        self._clock, self._sleep = clock, sleep
        self._next = 0.0

    def wait(self) -> None:
        now = self._clock()
        if now < self._next:
            self._sleep(self._next - now)
            now = self._next
        self._next = now + self.interval


class HttpError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class HttpClient:
    def __init__(self, base_url: str, headers: Mapping[str, str], per_minute: float,
                 retries: int = 4, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers)
        self.limiter = RateLimiter(per_minute)
        self.retries, self.timeout = retries, timeout

    def _open(self, url: str) -> Any:
        req = urllib.request.Request(url, headers=self.headers)
        delay = 2.0
        for attempt in range(self.retries + 1):
            self.limiter.wait()
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:200]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise HttpError(e.code, redact(body)) from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise HttpError(0, redact(str(e))) from None
        raise HttpError(0, "unreachable")

    def get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        query = {k: v for k, v in (params or {}).items() if v is not None}
        url = f"{self.base_url}{path}" + (f"?{urllib.parse.urlencode(query)}" if query else "")
        with self._open(url) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def download(self, path: str, dest: Path, progress: Callable[[int], None] | None = None) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with self._open(f"{self.base_url}{path}") as resp, open(tmp, "wb") as out:
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done)
        tmp.replace(dest)
        return dest
