"""Resume an HTTP download with Range requests."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument("--progress-mb", type=float, default=256.0)
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = output.stat().st_size if output.exists() else 0

    head = Request(args.url, method="HEAD")
    with urlopen(head, timeout=60) as response:
        total = int(response.headers.get("Content-Length", "0"))
        accept_ranges = response.headers.get("Accept-Ranges", "")

    if existing >= total > 0:
        print(f"Already complete: {existing} bytes")
        return 0
    if existing and "bytes" not in accept_ranges.lower():
        raise RuntimeError("Server does not advertise byte-range support")

    headers = {}
    mode = "ab" if existing else "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = Request(args.url, headers=headers)
    print(f"Downloading {args.url}")
    print(f"Existing: {existing} / {total}")
    next_report = existing + int(args.progress_mb * 1024 * 1024)
    with urlopen(request, timeout=120) as response, output.open(mode + "") as stream:
        while True:
            chunk = response.read(args.chunk_size)
            if not chunk:
                break
            stream.write(chunk)
            existing += len(chunk)
            if existing >= next_report:
                percent = 100.0 * existing / total if total else 0.0
                print(f"{existing}/{total} ({percent:.1f}%)", flush=True)
                next_report = existing + int(args.progress_mb * 1024 * 1024)
    print(f"{existing}/{total} (complete)")
    if total and output.stat().st_size != total:
        raise RuntimeError(f"Incomplete download: {output.stat().st_size} != {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
