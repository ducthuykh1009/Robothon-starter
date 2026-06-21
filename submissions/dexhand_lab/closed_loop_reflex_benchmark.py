from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATASET_DIR = PROJECT_DIR / "dataset"


def read_json(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback


def first_time(rows: list[dict], phase: str) -> float | None:
    matches = [float(row.get("time", 0.0) or 0.0) for row in rows if row.get("phase_name") == phase]
    return min(matches) if matches else None


def last_time(rows: list[dict], phase: str) -> float | None:
    matches = [float(row.get("time", 0.0) or 0.0) for row in rows if row.get("phase_name") == phase]
    return max(matches) if matches else None


def phase_rows(rows: list[dict], phase: str) -> list[dict]:
    return [row for row in rows if row.get("phase_name") == phase]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_closed_loop_reflex_benchmark(
    output_dir: Path = OUTPUT_DIR,
    dataset_dir: Path = DATASET_DIR,
    summary: dict | None = None,
) -> dict:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(summary or read_json(output_dir / "summary.json", {}))
    trajectory = read_json(output_dir / "trajectory.json", [])

    slip_monitor = phase_rows(trajectory, "SLIP_MONITOR")
    recovery = phase_rows(trajectory, "RECOVERY_IF_SLIP")
    load_hold = phase_rows(trajectory, "LOAD_HOLD_9X")
    cap_verify = phase_rows(trajectory, "CAP_ANGLE_VERIFY")

    slip_start = first_time(trajectory, "SLIP_MONITOR")
    slip_detection_time = last_time(trajectory, "SLIP_MONITOR")
    recovery_start = first_time(trajectory, "RECOVERY_IF_SLIP")
    load_hold_start = first_time(trajectory, "LOAD_HOLD_9X")
    response_latency_s = (
        max(0.0, float(recovery_start) - float(slip_detection_time))
        if slip_detection_time is not None and recovery_start is not None
        else None
    )
    response_latency_ms = None if response_latency_s is None else round(response_latency_s * 1000.0, 3)
    monitor_window_s = (
        max(0.0, float(slip_detection_time) - float(slip_start))
        if slip_start is not None and slip_detection_time is not None
        else None
    )

    initial_slip_mm = max([float(row.get("initial_slip_mm", 0.0) or 0.0) for row in recovery] + [0.0])
    final_slip_mm = min([float(row.get("final_slip_mm", 0.0) or 0.0) for row in load_hold + cap_verify if float(row.get("final_slip_mm", 0.0) or 0.0) > 0.0] or [float(summary.get("final_slip_mm", 0.0) or 0.0)])
    pressure_before = mean([float(row.get("pressure_target_n", 0.0) or 0.0) for row in slip_monitor])
    pressure_recovery = mean([float(row.get("pressure_target_n", 0.0) or 0.0) for row in recovery])
    pressure_load = mean([float(row.get("pressure_target_n", 0.0) or 0.0) for row in load_hold])
    active_fingers = max([int(row.get("active_finger_count", 0) or 0) for row in slip_monitor + recovery + load_hold] + [0])
    tactile_confidence = mean([float(row.get("mean_contact_confidence", 0.0) or 0.0) for row in slip_monitor + recovery + load_hold])
    friction_margin = mean([float(row.get("mean_friction_margin", 0.0) or 0.0) for row in slip_monitor + recovery + load_hold])
    cap_slip_peak = max([float(row.get("cap_slip_mm", 0.0) or 0.0) for row in slip_monitor + recovery + load_hold] + [0.0])
    cap_slip_final = min([float(row.get("cap_slip_mm", 0.0) or 0.0) for row in load_hold if float(row.get("cap_slip_mm", 0.0) or 0.0) > 0.0] or [0.0])
    recovery_gain = max(0.0, initial_slip_mm - final_slip_mm)
    pressure_boost_n = max(0.0, pressure_recovery - pressure_before)

    trace_rows = []
    for row in slip_monitor + recovery + load_hold + cap_verify:
        trace_rows.append(
            {
                "time_s": row.get("time"),
                "phase": row.get("phase_name"),
                "pressure_target_n": row.get("pressure_target_n"),
                "active_fingers": row.get("active_finger_count"),
                "cap_slip_mm": row.get("cap_slip_mm"),
                "initial_slip_mm": row.get("initial_slip_mm"),
                "final_slip_mm": row.get("final_slip_mm"),
                "recovery_active": row.get("recovery_active"),
                "pressure_boost_active": row.get("pressure_boost_active"),
                "mean_contact_confidence": row.get("mean_contact_confidence"),
                "mean_friction_margin": row.get("mean_friction_margin"),
                "load_hold_x": row.get("load_hold_x"),
            }
        )

    report = {
        "project": "DexHand Lab",
        "report_type": "closed_loop_reflex_benchmark",
        "source": "trajectory_contact_pressure_proxy",
        "honest_scope": "Simulation-native tactile/contact proxy benchmark; not physical hardware latency.",
        "slip_monitor_phase_present": bool(slip_monitor),
        "recovery_phase_present": bool(recovery),
        "load_hold_phase_present": bool(load_hold),
        "slip_start_time_s": slip_start,
        "slip_detection_time_s": slip_detection_time,
        "recovery_start_time_s": recovery_start,
        "load_hold_start_time_s": load_hold_start,
        "slip_monitor_window_s": None if monitor_window_s is None else round(monitor_window_s, 5),
        "reflex_response_latency_s": None if response_latency_s is None else round(response_latency_s, 5),
        "reflex_response_latency_ms": response_latency_ms,
        "reflex_latency_threshold_ms": 20.0,
        "reflex_latency_pass": response_latency_ms is not None and response_latency_ms <= 20.0,
        "initial_slip_mm": round(initial_slip_mm, 4),
        "final_slip_mm": round(final_slip_mm, 4),
        "cap_slip_peak_mm": round(cap_slip_peak, 4),
        "cap_slip_final_mm": round(cap_slip_final, 4),
        "recovery_gain_mm": round(recovery_gain, 4),
        "pressure_before_n": round(pressure_before, 4),
        "pressure_recovery_n": round(pressure_recovery, 4),
        "pressure_load_hold_n": round(pressure_load, 4),
        "pressure_boost_n": round(pressure_boost_n, 4),
        "active_fingers_during_reflex": active_fingers,
        "mean_contact_confidence_during_reflex": round(tactile_confidence, 4),
        "mean_friction_margin_during_reflex": round(friction_margin, 4),
        "load_hold_x": summary.get("load_hold_x"),
        "load_hold_success": bool(summary.get("load_hold_success", False)),
        "object_snap_events": int(summary.get("object_snap_events", 0) or 0),
        "closed_loop_reflex_success": bool(
            response_latency_ms is not None
            and response_latency_ms <= 20.0
            and final_slip_mm <= 0.5
            and active_fingers >= 4
            and bool(summary.get("load_hold_success", False))
            and int(summary.get("object_snap_events", 0) or 0) == 0
        ),
        "trace_rows": len(trace_rows),
    }

    report_path = dataset_dir / "closed_loop_reflex_report.json"
    trace_path = dataset_dir / "closed_loop_reflex_trace.csv"
    scorecard_path = output_dir / "closed_loop_reflex_scorecard.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    scorecard_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "time_s",
            "phase",
            "pressure_target_n",
            "active_fingers",
            "cap_slip_mm",
            "initial_slip_mm",
            "final_slip_mm",
            "recovery_active",
            "pressure_boost_active",
            "mean_contact_confidence",
            "mean_friction_margin",
            "load_hold_x",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace_rows)

    return {
        "closed_loop_reflex_benchmark_available": True,
        "closed_loop_reflex_report_path": "submissions/dexhand_lab/dataset/closed_loop_reflex_report.json",
        "closed_loop_reflex_trace_path": "submissions/dexhand_lab/dataset/closed_loop_reflex_trace.csv",
        "closed_loop_reflex_scorecard_path": "submissions/dexhand_lab/outputs/closed_loop_reflex_scorecard.json",
        "closed_loop_reflex_success": report["closed_loop_reflex_success"],
        "reflex_response_latency_ms": response_latency_ms,
        "reflex_latency_pass": report["reflex_latency_pass"],
        "reflex_latency_threshold_ms": report["reflex_latency_threshold_ms"],
        "reflex_initial_slip_mm": report["initial_slip_mm"],
        "reflex_final_slip_mm": report["final_slip_mm"],
        "reflex_pressure_boost_n": report["pressure_boost_n"],
        "reflex_active_fingers": active_fingers,
        "reflex_contact_confidence": report["mean_contact_confidence_during_reflex"],
        "reflex_friction_margin": report["mean_friction_margin_during_reflex"],
    }


def main() -> int:
    result = run_closed_loop_reflex_benchmark()
    print("DexHand closed-loop reflex benchmark")
    print("------------------------------------")
    print(f"Response latency: {result['reflex_response_latency_ms']} ms")
    print(f"Closed-loop reflex success: {str(result['closed_loop_reflex_success']).lower()}")
    return 0 if result["closed_loop_reflex_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
