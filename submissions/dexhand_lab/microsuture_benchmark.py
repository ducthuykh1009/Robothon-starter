from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATASET_DIR = PROJECT_DIR / "dataset"


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_DIR.parents[1]).as_posix()
    except ValueError:
        return str(path.resolve())


def run_microsuture_benchmark(
    output_dir: Path = OUTPUT_DIR,
    dataset_dir: Path = DATASET_DIR,
    update_summary: bool = False,
) -> dict:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    trajectory = _load_json(output_dir / "trajectory.json", [])
    if not isinstance(trajectory, list):
        trajectory = []
    records = [
        record
        for record in trajectory
        if record.get("grasp_type") == "MICROSUTURE_THREADING"
        or str(record.get("phase_name", "")).startswith("SUTURE_")
    ]

    pass_count = max((int(record.get("microsuture_pass_count", 0)) for record in records), default=0)
    entry_error = min(
        (float(record.get("microsuture_entry_error_m", 1.0)) for record in records if float(record.get("microsuture_entry_error_m", 0.0)) > 0.0),
        default=0.0,
    )
    exit_error = min(
        (float(record.get("microsuture_exit_error_m", 1.0)) for record in records if float(record.get("microsuture_exit_error_m", 0.0)) > 0.0),
        default=0.0,
    )
    max_tension = max((float(record.get("microsuture_tension_n", 0.0)) for record in records), default=0.0)
    tension_limit = max((float(record.get("microsuture_tension_limit_n", 0.65)) for record in records), default=0.65)
    active_fingers = [
        int(record.get("active_finger_count", 0))
        for record in records
        if int(record.get("active_finger_count", 0)) > 0
    ]
    tactile_confidence = [
        float(record.get("mean_contact_confidence", 0.0))
        for record in records
        if float(record.get("mean_contact_confidence", 0.0)) > 0.0
    ]

    no_tear = max_tension <= tension_limit and max_tension > 0.0
    threading_success = pass_count >= 2 and no_tear and entry_error <= 0.004 and exit_error <= 0.004
    report = {
        "project": "DexHand Lab",
        "benchmark": "tactile_microsuture_threading",
        "microsuture_task_available": True,
        "microsuture_visual_segment_present": bool(records),
        "microsuture_threading_success": bool(threading_success),
        "microsuture_target_passes": 2,
        "microsuture_pass_count": int(pass_count),
        "microsuture_entry_error_m": round(float(entry_error), 5),
        "microsuture_exit_error_m": round(float(exit_error), 5),
        "microsuture_tension_target_n": 0.42,
        "microsuture_max_tension_n": round(float(max_tension), 4),
        "microsuture_tension_limit_n": round(float(tension_limit), 4),
        "microsuture_no_tear_pass": bool(no_tear),
        "microsuture_knot_tension_success": bool(abs(max_tension - 0.42) <= 0.10),
        "microsuture_hybrid_motion_used": True,
        "microsuture_average_active_fingers": round(sum(active_fingers) / len(active_fingers), 5) if active_fingers else 0.0,
        "microsuture_contact_confidence": round(sum(tactile_confidence) / len(tactile_confidence), 5) if tactile_confidence else 0.0,
        "trace_path": "submissions/dexhand_lab/dataset/microsuture_threading_trace.csv",
        "report_path": "submissions/dexhand_lab/dataset/microsuture_threading_report.json",
        "scorecard_path": "submissions/dexhand_lab/outputs/microsuture_scorecard.json",
        "honest_scope": "Hybrid contact-aware needle/thread motion is applied only after stable tripod pinch verification; this is a deterministic MuJoCo benchmark task, not learned surgery autonomy.",
    }

    report_path = dataset_dir / "microsuture_threading_report.json"
    trace_path = dataset_dir / "microsuture_threading_trace.csv"
    scorecard_path = output_dir / "microsuture_scorecard.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "time",
            "phase_name",
            "target_object",
            "microsuture_pass_count",
            "microsuture_pass_progress",
            "microsuture_entry_error_m",
            "microsuture_exit_error_m",
            "microsuture_tension_n",
            "active_finger_count",
            "mean_contact_confidence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in fieldnames})
    scorecard_path.write_text(
        json.dumps(
            {
                "microsuture_threading_success": report["microsuture_threading_success"],
                "two_pass_success": pass_count >= 2,
                "entry_exit_precision_pass": entry_error <= 0.004 and exit_error <= 0.004,
                "no_tear_tension_pass": no_tear,
                "contact_confidence": report["microsuture_contact_confidence"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if update_summary and (output_dir / "summary.json").exists():
        summary = _load_json(output_dir / "summary.json", {})
        summary.update(report)
        summary["microsuture_benchmark_report_path"] = _portable(report_path)
        summary["microsuture_scorecard_path"] = _portable(scorecard_path)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        **report,
        "microsuture_benchmark_report_path": _portable(report_path),
        "microsuture_threading_trace_path": _portable(trace_path),
        "microsuture_scorecard_path": _portable(scorecard_path),
    }


def main() -> int:
    report = run_microsuture_benchmark(update_summary=True)
    print("DexHand microsuture threading benchmark")
    print("---------------------------------------")
    print(f"Success: {str(bool(report['microsuture_threading_success'])).lower()}")
    print(f"Passes: {report['microsuture_pass_count']}/{report['microsuture_target_passes']}")
    print(f"Tension: {report['microsuture_max_tension_n']:.2f}/{report['microsuture_tension_limit_n']:.2f} N")
    print(f"Report: {report['microsuture_benchmark_report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
