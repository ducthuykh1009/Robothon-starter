from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATASET_DIR = PROJECT_DIR / "dataset"
VIAL_FORCE_LIMIT_N = 4.5
VIAL_ROTATION_TARGET_DEG = 162.0


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def run_vial_uncap_delivery_benchmark(
    *,
    output_dir: Path = OUTPUT_DIR,
    dataset_dir: Path = DATASET_DIR,
) -> dict[str, Any]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = _load_json(output_dir / "trajectory.json", [])
    vial_rows = [row for row in trajectory if row.get("vial_task_phase")]

    cap_rotation = max((float(row.get("vial_cap_rotation_achieved_deg", 0.0)) for row in vial_rows), default=0.0)
    max_force = max((float(row.get("vial_no_crush_force_n", 0.0)) for row in vial_rows), default=0.0)
    contact_confidences = [
        float(row.get("mean_contact_confidence", 0.0))
        for row in vial_rows
        if float(row.get("mean_contact_confidence", 0.0)) > 0.0
    ]
    active_fingers = [int(row.get("active_finger_count", 0) or 0) for row in vial_rows]
    cap_removed = any(bool(row.get("vial_cap_removed")) for row in vial_rows)
    sample_delivered = any(bool(row.get("pill_delivery_success")) for row in vial_rows)
    no_crush = max_force <= VIAL_FORCE_LIMIT_N and max_force > 0.0
    verified = any(bool(row.get("vial_grasp_verified")) for row in vial_rows)
    success = bool(verified and cap_removed and sample_delivered and no_crush and cap_rotation >= 150.0)

    report = {
        "project": "DexHand Lab",
        "benchmark": "vial_uncap_delivery",
        "purpose": "Visible no-crush vial handling: grasp body, twist/lift cap, tilt vial, and deliver sample into tray.",
        "vial_uncap_deliver_task_available": True,
        "vial_uncap_deliver_success": success,
        "vial_grasp_verified": verified,
        "vial_cap_rotation_target_deg": VIAL_ROTATION_TARGET_DEG,
        "vial_cap_rotation_achieved_deg": round(cap_rotation, 3),
        "vial_cap_rotation_error_deg": round(abs(VIAL_ROTATION_TARGET_DEG - cap_rotation), 3),
        "vial_cap_removed": cap_removed,
        "pill_delivery_success": sample_delivered,
        "pill_in_tray": sample_delivered,
        "vial_no_crush_force_limit_n": VIAL_FORCE_LIMIT_N,
        "vial_max_force_n": round(max_force, 3),
        "vial_no_crush_force_pass": no_crush,
        "mean_vial_contact_confidence": round(mean(contact_confidences), 5) if contact_confidences else 0.0,
        "mean_active_fingers_during_vial_task": round(mean(active_fingers), 5) if active_fingers else 0.0,
        "vial_task_frame_count": len(vial_rows),
        "vial_hybrid_manipulation_used": True,
        "honest_scope": "Hybrid contact-aware cap/sample motion is applied only after verified multi-finger vial contact.",
        "trace_path": "submissions/dexhand_lab/dataset/vial_uncap_delivery_trace.csv",
    }

    report_path = dataset_dir / "vial_uncap_delivery_report.json"
    trace_path = dataset_dir / "vial_uncap_delivery_trace.csv"
    scorecard_path = output_dir / "vial_uncap_delivery_scorecard.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    scorecard_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "time",
            "phase_name",
            "target_object",
            "active_finger_count",
            "vial_grasp_verified",
            "vial_cap_rotation_achieved_deg",
            "vial_cap_removed",
            "vial_no_crush_force_n",
            "vial_delivery_progress",
            "pill_in_tray",
            "pill_delivery_success",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in vial_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    return {
        "vial_uncap_delivery_benchmark_available": True,
        "vial_uncap_delivery_report_path": "submissions/dexhand_lab/dataset/vial_uncap_delivery_report.json",
        "vial_uncap_delivery_trace_path": "submissions/dexhand_lab/dataset/vial_uncap_delivery_trace.csv",
        "vial_uncap_delivery_scorecard_path": "submissions/dexhand_lab/outputs/vial_uncap_delivery_scorecard.json",
        **{key: value for key, value in report.items() if key.startswith("vial_") or key.startswith("pill_")},
    }


def main() -> int:
    report = run_vial_uncap_delivery_benchmark()
    print("DexHand vial uncap-delivery benchmark")
    print("-------------------------------------")
    print(f"Success: {str(bool(report['vial_uncap_deliver_success'])).lower()}")
    print(f"Cap rotation: {float(report['vial_cap_rotation_achieved_deg']):.1f} deg")
    print(f"No-crush force pass: {str(bool(report['vial_no_crush_force_pass'])).lower()}")
    print(f"Sample delivered: {str(bool(report['pill_delivery_success'])).lower()}")
    return 0 if report["vial_uncap_deliver_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
