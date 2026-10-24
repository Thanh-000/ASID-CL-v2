from __future__ import annotations

import argparse
import csv
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


URL = "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz"
TOTAL_BYTES = 1_902_053_684


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable CrisisMMD v2.0 media downloader.")
    parser.add_argument("--output-dir", type=Path, default=Path("external_data/crisismmd_v2"))
    parser.add_argument("--connections", type=int, default=4)
    args = parser.parse_args()
    if args.connections <= 0:
        raise ValueError("connections must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "download_parts"
    parts_dir.mkdir(exist_ok=True)
    segments = _segments(args.connections)
    _write_manifest(parts_dir, segments)

    stop = threading.Event()
    monitor = threading.Thread(
        target=_monitor,
        args=(parts_dir, segments, stop),
        daemon=True,
    )
    monitor.start()
    try:
        with ThreadPoolExecutor(max_workers=args.connections) as executor:
            list(executor.map(lambda segment: _download_segment(parts_dir, segment), segments))
    finally:
        stop.set()
        monitor.join()

    archive = output_dir / "CrisisMMD_v2.0.tar.gz"
    temporary = archive.with_suffix(".tar.gz.assembling")
    with temporary.open("wb") as destination:
        for index, _, _ in segments:
            with (parts_dir / f"part{index:03d}.bin").open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    destination.write(chunk)
        destination.flush()
        os.fsync(destination.fileno())
    if temporary.stat().st_size != TOTAL_BYTES:
        raise RuntimeError(f"assembled size mismatch: {temporary.stat().st_size}")
    os.replace(temporary, archive)
    print(f"archive={archive}")
    print(f"bytes={archive.stat().st_size}")
    print(f"sha256={_sha256(archive)}")


def _segments(count: int) -> list[tuple[int, int, int]]:
    chunk = (TOTAL_BYTES + count - 1) // count
    return [
        (index, index * chunk, min(TOTAL_BYTES - 1, (index + 1) * chunk - 1))
        for index in range(count)
        if index * chunk < TOTAL_BYTES
    ]


def _download_segment(parts_dir: Path, segment: tuple[int, int, int]) -> None:
    index, start, end = segment
    path = parts_dir / f"part{index:03d}.bin"
    expected = end - start + 1
    if path.exists() and path.stat().st_size > expected:
        raise RuntimeError(f"oversized segment: {path}")
    failures = 0
    while (path.stat().st_size if path.exists() else 0) < expected:
        current = path.stat().st_size if path.exists() else 0
        absolute_start = start + current
        try:
            with requests.get(
                URL,
                headers={"Range": f"bytes={absolute_start}-{end}"},
                stream=True,
                timeout=(30, 90),
            ) as response:
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                if response.status_code != 206 or not content_range.startswith(
                    f"bytes {absolute_start}-{end}/"
                ):
                    raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
                with path.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            failures = 0
        except (requests.RequestException, OSError, RuntimeError) as exc:
            failures += 1
            if failures > 30:
                raise RuntimeError(f"segment {index} failed repeatedly") from exc
            time.sleep(min(30, 2 * failures))
        actual = path.stat().st_size if path.exists() else 0
        if actual > expected:
            raise RuntimeError(f"segment {index} exceeded expected size")


def _monitor(
    parts_dir: Path,
    segments: list[tuple[int, int, int]],
    stop: threading.Event,
) -> None:
    while not stop.wait(15):
        downloaded = sum(
            (parts_dir / f"part{index:03d}.bin").stat().st_size
            if (parts_dir / f"part{index:03d}.bin").exists()
            else 0
            for index, _, _ in segments
        )
        print(
            f"downloaded={downloaded}/{TOTAL_BYTES} ({100 * downloaded / TOTAL_BYTES:.1f}%)",
            flush=True,
        )


def _write_manifest(parts_dir: Path, segments: list[tuple[int, int, int]]) -> None:
    with (parts_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("part", "start", "end", "expected"))
        for index, start, end in segments:
            writer.writerow((f"part{index:03d}.bin", start, end, end - start + 1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
