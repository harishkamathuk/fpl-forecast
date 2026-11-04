from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fpl_forecast.operations.config import LATEST_SUCCESSFUL_PATH, OPERATIONAL_FAILED_DIR, OPERATIONAL_RUNS_DIR


def latest_successful(path: Path | None = None) -> dict[str, Any] | None:
    pointer_path = path or LATEST_SUCCESSFUL_PATH
    if not pointer_path.exists():
        return None
    return json.loads(pointer_path.read_text(encoding="utf-8"))


def publish_success(
    temp_dir: Path,
    *,
    run_id: str,
    manifest: dict[str, Any],
    runs_dir: Path | None = None,
    pointer_path: Path | None = None,
) -> Path:
    output_runs_dir = runs_dir or OPERATIONAL_RUNS_DIR
    latest_path = pointer_path or LATEST_SUCCESSFUL_PATH
    output_runs_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_runs_dir / run_id
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing operational run: {final_dir}")
    temp_manifest = temp_dir / "run_manifest.json"
    temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_tmp = latest_path.with_suffix(".json.tmp")
    pointer = {
        "run_id": run_id,
        "run_dir": str(final_dir),
        "manifest_path": str(final_dir / "run_manifest.json"),
        "published_at": manifest["completed_at"],
        "schema_version": manifest["frontend_schema_version"],
    }
    if "input_fingerprint" in manifest:
        pointer["input_fingerprint"] = manifest["input_fingerprint"]
    renamed = False
    try:
        pointer_tmp.write_text(json.dumps(pointer, indent=2, sort_keys=True), encoding="utf-8")
        temp_dir.rename(final_dir)
        renamed = True
        # This atomic replacement is the publication commit point. No pointer
        # mutation may follow it in the caller.
        pointer_tmp.replace(latest_path)
    except Exception as exc:
        if renamed:
            try:
                final_dir.rename(temp_dir)
            except OSError as rollback_exc:
                raise RuntimeError(
                    f"Publication failed: {exc}; rollback failed: {rollback_exc}; "
                    f"unreferenced run retained at {final_dir}"
                ) from exc
        raise
    finally:
        # A stale pointer candidate is never authoritative. Cleanup must not
        # turn a committed publication into an apparent failure.
        try:
            pointer_tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return final_dir


def publish_failure(
    temp_dir: Path,
    *,
    run_id: str,
    manifest: dict[str, Any],
    failed_runs_dir: Path | None = None,
) -> Path:
    output_failed_dir = failed_runs_dir or OPERATIONAL_FAILED_DIR
    output_failed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = output_failed_dir / run_id
    if failed_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing failed operational run: {failed_dir}")
    (temp_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temp_dir.rename(failed_dir)
    return failed_dir
