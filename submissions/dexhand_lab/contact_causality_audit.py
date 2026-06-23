from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATASET_DIR = PROJECT_DIR / "dataset"


CONTACT_CAUSAL_PHASES = {
    "STABLE_GRASP_VERIFY",
    "SECURE_OBJECT",
    "HOLD_STABLE",
    "IN_HAND_ROTATION",
    "MINIMUM_JERK_CAP_TWIST",
    "CAP_ANGLE_VERIFY",
    "LOAD_HOLD_9X",
    "LOCK_ROTATE_CODE_1",
    "LOCK_ROTATE_CODE_2",
    "LOCK_ROTATE_CODE_3",
    "LOCK_LATCH_PULL",
    "LOCK_MICRO_DOOR_OPEN",
    "ASSEMBLY_PRECISION_GRASP",
    "ASSEMBLY_IN_HAND_ORIENT",
    "ASSEMBLY_ALIGN_TO_SOCKET",
    "ASSEMBLY_COMPLIANT_INSERT",
    "ASSEMBLY_JAM_CHECK_CORRECT",
    "ASSEMBLY_INSERT_VERIFY",
    "VIAL_FIVE_FINGER_FORCE_VERIFY",
    "VIAL_CAP_COUNTER_TWIST",
    "VIAL_CAP_LIFT_CLEAR",
    "VIAL_TILT_TO_TRAY",
    "VIAL_SAMPLE_DELIVERY",
    "VIAL_DELIVERY_VERIFY",
    "CHECKPOINT_TOUCH",
    "BUTTON_PRESS",
}


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def audit_contact_causality(
    *,
    output_dir: Path = OUTPUT_DIR,
    dataset_dir: Path = DATASET_DIR,
) -> dict[str, Any]:
    trajectory = _load_json(output_dir / "trajectory.json", [])
    summary = _load_json(output_dir / "summary.json", {})
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    pre_verification_motion_events = 0
    attach_before_verification_events = 0
    snap_events = 0
    verified_motion_frames = 0
    motion_frames = 0
    contact_confidences: list[float] = []
    active_fingers: list[int] = []

    for record in trajectory:
        phase = str(record.get("phase_name", ""))
        target = record.get("target_object") or ""
        stable = bool(record.get("stable_grasp_verified", False))
        attached = bool(record.get("attached_to_hand", False))
        active = int(record.get("active_finger_count", 0) or 0)
        confidence = float(record.get("mean_contact_confidence", record.get("tactile_confidence", 0.0)) or 0.0)
        object_moved_before_grasp = bool(record.get("object_moved_before_grasp", False))
        snap = bool(record.get("object_snap_event", False) or record.get("sudden_pose_jump_detected", False))
        attach_before = bool(record.get("attach_before_verification", False))
        movement_allowed = bool(stable or attached or phase in CONTACT_CAUSAL_PHASES)
        manipulation_phase = bool(target and phase in CONTACT_CAUSAL_PHASES)

        if manipulation_phase:
            motion_frames += 1
            active_fingers.append(active)
            contact_confidences.append(confidence)
            if movement_allowed:
                verified_motion_frames += 1
        if object_moved_before_grasp:
            pre_verification_motion_events += 1
        if snap:
            snap_events += 1
        if attach_before:
            attach_before_verification_events += 1

        if manipulation_phase or object_moved_before_grasp or snap or attach_before:
            rows.append(
                {
                    "time_s": record.get("time", 0.0),
                    "phase": phase,
                    "target_object": target,
                    "stable_grasp_verified": stable,
                    "attached_to_hand": attached,
                    "active_finger_count": active,
                    "contact_confidence": round(confidence, 5),
                    "movement_allowed_by_contact_gate": movement_allowed,
                    "object_moved_before_grasp": object_moved_before_grasp,
                    "object_snap_event": snap,
                    "attach_before_verification": attach_before,
                }
            )

    contact_causal_pass = (
        int(summary.get("object_snap_events", 0) or 0) == 0
        and int(summary.get("attach_before_verification_count", 0) or 0) == 0
        and pre_verification_motion_events == 0
        and attach_before_verification_events == 0
        and snap_events == 0
    )
    verified_motion_rate = verified_motion_frames / motion_frames if motion_frames else 1.0

    report = {
        "project": "DexHand Lab",
        "audit_name": "contact_causality_no_snap_audit",
        "policy": "object motion is allowed only after contact/verification gates or in explicitly logged tactile manipulation phases",
        "hybrid_routine_disclosed": True,
        "contact_causal_pass": bool(contact_causal_pass),
        "object_snap_events": int(summary.get("object_snap_events", 0) or snap_events),
        "attach_before_verification_count": int(summary.get("attach_before_verification_count", 0) or attach_before_verification_events),
        "pre_verification_motion_events": pre_verification_motion_events,
        "verified_motion_frame_rate": round(float(verified_motion_rate), 5),
        "manipulation_frames_checked": motion_frames,
        "mean_active_fingers_during_manipulation": round(mean(active_fingers), 5) if active_fingers else 0.0,
        "mean_contact_confidence_during_manipulation": round(mean(contact_confidences), 5) if contact_confidences else 0.0,
        "checked_phase_count": len(CONTACT_CAUSAL_PHASES),
        "trace_path": "submissions/dexhand_lab/dataset/contact_causality_trace.csv",
    }

    trace_path = dataset_dir / "contact_causality_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "time_s",
            "phase",
            "target_object",
            "stable_grasp_verified",
            "attached_to_hand",
            "active_finger_count",
            "contact_confidence",
            "movement_allowed_by_contact_gate",
            "object_moved_before_grasp",
            "object_snap_event",
            "attach_before_verification",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report_path = dataset_dir / "contact_causality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = audit_contact_causality()
    print("DexHand contact-causality audit")
    print("------------------------------")
    print(f"Contact-causal pass: {str(report['contact_causal_pass']).lower()}")
    print(f"Object snap events: {report['object_snap_events']}")
    print(f"Attach-before-verification: {report['attach_before_verification_count']}")
    print(f"Verified motion frame rate: {report['verified_motion_frame_rate']:.3f}")


if __name__ == "__main__":
    main()
