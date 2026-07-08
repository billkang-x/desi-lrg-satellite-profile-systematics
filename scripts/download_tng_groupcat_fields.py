"""Download a minimal IllustrisTNG group-catalog field set for HOD pilots."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_FIELDS = {
    "Subhalo": ["SubhaloPos", "SubhaloVel", "SubhaloMassType", "SubhaloGrNr", "SubhaloFlag"],
    "Group": ["GroupFirstSub"],
}


def request_json(url: str, api_key: str, timeout: int) -> dict:
    req = urllib.request.Request(url, headers={"api-key": api_key, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_snapshot(base_url: str, sim: str, redshift: float | None, snapshot: int | None, api_key: str, timeout: int) -> dict:
    if snapshot is not None:
        return request_json(f"{base_url}/{sim}/snapshots/{snapshot}/", api_key, timeout)
    if redshift is None:
        raise ValueError("Need either --snapshot or --redshift")
    return request_json(f"{base_url}/{sim}/snapshots/z={redshift}/", api_key, timeout)


def content_disposition_filename(header_value: str | None) -> str | None:
    if not header_value:
        return None
    match = re.search(r'filename="?([^";]+)"?', header_value)
    return match.group(1) if match else None


def field_url(base_url: str, sim: str, snapshot: int, group: str, field: str) -> str:
    return f"{base_url}/{sim}/files/groupcat-{snapshot}/?{urllib.parse.urlencode({group: field})}"


def download_file(url: str, api_key: str, output_path: Path, timeout: int, overwrite: bool) -> dict:
    if output_path.exists() and not overwrite:
        return {"path": str(output_path), "bytes": output_path.stat().st_size, "status": "exists", "url": url}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()
    req = urllib.request.Request(url, headers={"api-key": api_key})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            disposition_name = content_disposition_filename(response.headers.get("content-disposition"))
            if disposition_name and output_path.name == "AUTO":
                output_path = output_path.parent / disposition_name
                temp_path = output_path.with_suffix(output_path.suffix + ".part")
            total = int(response.headers.get("content-length") or 0)
            written = 0
            with temp_path.open("wb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    written += len(chunk)
                    if total and written % (64 * 1024 * 1024) < len(chunk):
                        print(f"{output_path.name}: {written / 1e6:.1f}/{total / 1e6:.1f} MB")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:500]}") from exc
    temp_path.replace(output_path)
    return {"path": str(output_path), "bytes": output_path.stat().st_size, "status": "downloaded", "seconds": time.time() - started, "url": url}


def parse_fields(items: list[str]) -> dict[str, list[str]]:
    if not items:
        return DEFAULT_FIELDS
    fields: dict[str, list[str]] = {}
    for item in items:
        group, field_list = item.split(":", 1)
        fields[group] = [field.strip() for field in field_list.split(",") if field.strip()]
    return fields


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", default="TNG300-3")
    parser.add_argument("--redshift", type=float, default=0.7)
    parser.add_argument("--snapshot", type=int, default=None)
    parser.add_argument("--output-dir", default="data/tng300_pilot/raw")
    parser.add_argument("--base-url", default="https://www.tng-project.org/api")
    parser.add_argument("--api-key-env", default="TNG_API_KEY")
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before running this script")
    snapshot_info = resolve_snapshot(args.base_url, args.sim, args.redshift, args.snapshot, api_key, args.timeout)
    sim_info = request_json(f"{args.base_url}/{args.sim}/", api_key, args.timeout)
    snapshot = int(snapshot_info["number"])
    fields = parse_fields(args.field)
    output_dir = Path(args.output_dir) / args.sim / f"snap{snapshot:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    downloads = []
    if not args.metadata_only:
        for group, group_fields in fields.items():
            for field in group_fields:
                url = field_url(args.base_url, args.sim, snapshot, group, field)
                filename = f"{args.sim}_snap{snapshot:03d}_{group}_{field}.hdf5"
                print(f"Downloading {group}/{field}")
                downloads.append(download_file(url, api_key, output_dir / filename, args.timeout, args.overwrite))

    metadata = {
        "sim": args.sim,
        "snapshot": snapshot,
        "redshift": float(snapshot_info["redshift"]),
        "boxsize_ckpc_h": float(sim_info["boxsize"]),
        "filesize_groupcat_bytes": snapshot_info.get("filesize_groupcat"),
        "num_groups_subfind": snapshot_info.get("num_groups_subfind"),
        "num_groups_fof": snapshot_info.get("num_groups_fof"),
        "fields": fields,
        "downloads": downloads,
    }
    metadata_path = output_dir / f"{args.sim}_snap{snapshot:03d}_minimal_groupcat_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
