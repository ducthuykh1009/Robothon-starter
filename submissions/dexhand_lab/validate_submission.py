from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATASET_DIR = PROJECT_DIR / "dataset"

SRT_TIME_RE = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})"
)


def valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def srt_last_timestamp_s(path: Path) -> float:
    latest = 0.0
    for match in SRT_TIME_RE.finditer(path.read_text(encoding="utf-8")):
        latest = max(
            latest,
            int(match.group("h")) * 3600.0
            + int(match.group("m")) * 60.0
            + int(match.group("s"))
            + int(match.group("ms")) / 1000.0,
        )
    return latest


def main() -> int:
    required_files = [
        PROJECT_DIR / "registration.json",
        PROJECT_DIR / "README.md",
        PROJECT_DIR / "JUDGE_BRIEF.md",
        PROJECT_DIR / "EVIDENCE_INDEX.md",
        PROJECT_DIR / "run_demo.py",
        PROJECT_DIR / "run_stress_eval.py",
        PROJECT_DIR / "arena_task_suite.py",
        PROJECT_DIR / "minimum_jerk_controller.py",
        PROJECT_DIR / "contact_feedback_audit.py",
        PROJECT_DIR / "hardware_adaptation_audit.py",
        PROJECT_DIR / "tactile_active_perception.py",
        PROJECT_DIR / "tactile_shape_classifier.py",
        PROJECT_DIR / "adaptive_regrasp_policy.py",
        PROJECT_DIR / "tactile_pose_estimator.py",
        PROJECT_DIR / "precision_assembly_controller.py",
        PROJECT_DIR / "combination_lock_controller.py",
        PROJECT_DIR / "contact_causality_audit.py",
        PROJECT_DIR / "judge_replay_index.py",
        PROJECT_DIR / "closed_loop_reflex_benchmark.py",
        PROJECT_DIR / "vial_uncap_delivery_benchmark.py",
        PROJECT_DIR / "microsuture_benchmark.py",
        PROJECT_DIR / "quality_gate.py",
        PROJECT_DIR / "tests" / "test_submission_contract.py",
        PROJECT_DIR / "scene.xml",
        PROJECT_DIR / "human_grasp_library.py",
        PROJECT_DIR / "object_classifier.py",
        PROJECT_DIR / "dexhand_controller.py",
        PROJECT_DIR / "rubric_scorecard.json",
        PROJECT_DIR / "submission_manifest.json",
        OUTPUT_DIR / "demo.mp4",
        OUTPUT_DIR / "summary.json",
        OUTPUT_DIR / "trajectory.json",
        OUTPUT_DIR / "contact_timeline.json",
        OUTPUT_DIR / "final_report.txt",
        OUTPUT_DIR / "event_rules_report.json",
        OUTPUT_DIR / "submission_readiness_report.json",
        OUTPUT_DIR / "rubric_readiness_report.json",
        OUTPUT_DIR / "rubric_readiness_scorecard.csv",
        OUTPUT_DIR / "narration.srt",
        OUTPUT_DIR / "policy_card.json",
        OUTPUT_DIR / "sensor_manifest.json",
        OUTPUT_DIR / "judge_summary.json",
        OUTPUT_DIR / "video_replay_scorecard.json",
        OUTPUT_DIR / "closed_loop_reflex_scorecard.json",
        OUTPUT_DIR / "vial_uncap_delivery_scorecard.json",
        OUTPUT_DIR / "microsuture_scorecard.json",
        OUTPUT_DIR / "blind_tactile_summary.json",
        OUTPUT_DIR / "assembly_summary.json",
        OUTPUT_DIR / "combination_lock_summary.json",
        OUTPUT_DIR / "stress_eval.json",
        OUTPUT_DIR / "baseline_vs_feedback.json",
        OUTPUT_DIR / "stress_eval_summary.csv",
        PROJECT_DIR / "media" / "keyframes.png",
        PROJECT_DIR / "media" / "blind_tactile_keyframes.png",
        PROJECT_DIR / "media" / "tactile_classifier_panel.png",
        PROJECT_DIR / "media" / "assembly_keyframes.png",
        PROJECT_DIR / "media" / "tactile_pose_estimation_panel.png",
        PROJECT_DIR / "media" / "combination_lock_keyframes.png",
        PROJECT_DIR / "media" / "demo.mp4",
        DATASET_DIR / "task_suite_report.json",
        DATASET_DIR / "task_suite.csv",
        DATASET_DIR / "tactile_feedback_report.json",
        DATASET_DIR / "tactile_taxels.csv",
        DATASET_DIR / "tactile_exploration_trace.csv",
        DATASET_DIR / "tactile_classifier_report.json",
        DATASET_DIR / "tactile_confusion_matrix.json",
        DATASET_DIR / "adaptive_regrasp_report.json",
        DATASET_DIR / "adaptive_regrasp_trace.csv",
        DATASET_DIR / "unknown_arena_report.json",
        DATASET_DIR / "blind_tactile_stress_eval.json",
        DATASET_DIR / "blind_tactile_baseline_comparison.json",
        DATASET_DIR / "tactile_pose_estimator_report.json",
        DATASET_DIR / "tactile_pose_trace.csv",
        DATASET_DIR / "precision_assembly_report.json",
        DATASET_DIR / "precision_assembly_trace.csv",
        DATASET_DIR / "jam_recovery_report.json",
        DATASET_DIR / "jam_recovery_trace.csv",
        DATASET_DIR / "no_ground_truth_control_audit.json",
        DATASET_DIR / "combination_lock_report.json",
        DATASET_DIR / "combination_lock_trace.csv",
        DATASET_DIR / "contact_causality_report.json",
        DATASET_DIR / "contact_causality_trace.csv",
        DATASET_DIR / "judge_video_replay_index.json",
        DATASET_DIR / "judge_video_replay_index.csv",
        DATASET_DIR / "closed_loop_reflex_report.json",
        DATASET_DIR / "closed_loop_reflex_trace.csv",
        DATASET_DIR / "vial_uncap_delivery_report.json",
        DATASET_DIR / "vial_uncap_delivery_trace.csv",
        DATASET_DIR / "microsuture_threading_report.json",
        DATASET_DIR / "microsuture_threading_trace.csv",
        DATASET_DIR / "minimum_jerk_report.json",
        DATASET_DIR / "minimum_jerk_trace.csv",
        DATASET_DIR / "stress_eval.json",
        DATASET_DIR / "hardware_adaptation_report.json",
        DATASET_DIR / "hardware_command_stream.csv",
        DATASET_DIR / "sim2real_safety_case.json",
        DATASET_DIR / "code_quality_report.json",
        DATASET_DIR / "unit_test_report.json",
        PROJECT_DIR / "hardware_transfer.json",
        PROJECT_DIR / "HARDWARE_ADAPTATION.md",
        OUTPUT_DIR / "episodes" / "episode_000" / "trajectory.json",
        OUTPUT_DIR / "episodes" / "episode_000" / "metadata.json",
    ]
    json_files = [
        PROJECT_DIR / "registration.json",
        PROJECT_DIR / "rubric_scorecard.json",
        PROJECT_DIR / "submission_manifest.json",
        OUTPUT_DIR / "summary.json",
        OUTPUT_DIR / "trajectory.json",
        OUTPUT_DIR / "contact_timeline.json",
        OUTPUT_DIR / "event_rules_report.json",
        OUTPUT_DIR / "submission_readiness_report.json",
        OUTPUT_DIR / "rubric_readiness_report.json",
        OUTPUT_DIR / "policy_card.json",
        OUTPUT_DIR / "sensor_manifest.json",
        OUTPUT_DIR / "judge_summary.json",
        OUTPUT_DIR / "video_replay_scorecard.json",
        OUTPUT_DIR / "closed_loop_reflex_scorecard.json",
        OUTPUT_DIR / "vial_uncap_delivery_scorecard.json",
        OUTPUT_DIR / "microsuture_scorecard.json",
        OUTPUT_DIR / "blind_tactile_summary.json",
        OUTPUT_DIR / "assembly_summary.json",
        OUTPUT_DIR / "combination_lock_summary.json",
        OUTPUT_DIR / "stress_eval.json",
        OUTPUT_DIR / "baseline_vs_feedback.json",
        DATASET_DIR / "task_suite_report.json",
        DATASET_DIR / "tactile_feedback_report.json",
        DATASET_DIR / "tactile_classifier_report.json",
        DATASET_DIR / "tactile_confusion_matrix.json",
        DATASET_DIR / "adaptive_regrasp_report.json",
        DATASET_DIR / "unknown_arena_report.json",
        DATASET_DIR / "blind_tactile_stress_eval.json",
        DATASET_DIR / "blind_tactile_baseline_comparison.json",
        DATASET_DIR / "tactile_pose_estimator_report.json",
        DATASET_DIR / "precision_assembly_report.json",
        DATASET_DIR / "jam_recovery_report.json",
        DATASET_DIR / "no_ground_truth_control_audit.json",
        DATASET_DIR / "combination_lock_report.json",
        DATASET_DIR / "contact_causality_report.json",
        DATASET_DIR / "judge_video_replay_index.json",
        DATASET_DIR / "closed_loop_reflex_report.json",
        DATASET_DIR / "vial_uncap_delivery_report.json",
        DATASET_DIR / "microsuture_threading_report.json",
        DATASET_DIR / "minimum_jerk_report.json",
        DATASET_DIR / "stress_eval.json",
        DATASET_DIR / "hardware_adaptation_report.json",
        DATASET_DIR / "sim2real_safety_case.json",
        DATASET_DIR / "code_quality_report.json",
        DATASET_DIR / "unit_test_report.json",
        PROJECT_DIR / "hardware_transfer.json",
        OUTPUT_DIR / "episodes" / "episode_000" / "trajectory.json",
        OUTPUT_DIR / "episodes" / "episode_000" / "metadata.json",
    ]
    missing = [path for path in required_files if not path.exists()]
    invalid = [path for path in json_files if path.exists() and not valid_json(path)]
    if missing or invalid:
        print("DexHand validation failed")
        if missing:
            print("Missing files:")
            for path in missing:
                print(f"- {path}")
        if invalid:
            print("Invalid JSON:")
            for path in invalid:
                print(f"- {path}")
        return 1
    summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    registration = json.loads((PROJECT_DIR / "registration.json").read_text(encoding="utf-8"))
    manifest = json.loads((PROJECT_DIR / "submission_manifest.json").read_text(encoding="utf-8"))
    rubric = json.loads((PROJECT_DIR / "rubric_scorecard.json").read_text(encoding="utf-8"))
    event_rules = json.loads((OUTPUT_DIR / "event_rules_report.json").read_text(encoding="utf-8"))
    readiness = json.loads((OUTPUT_DIR / "submission_readiness_report.json").read_text(encoding="utf-8"))
    quality_report = json.loads((DATASET_DIR / "code_quality_report.json").read_text(encoding="utf-8"))
    rubric_readiness = json.loads((OUTPUT_DIR / "rubric_readiness_report.json").read_text(encoding="utf-8"))
    required_metrics = [
        "hand_skeleton_valid",
        "five_fingers_present",
        "thumb_opposition_joint_present",
        "object_snap_events",
        "attach_before_verification_count",
        "verified_grasp_before_attach_rate",
        "sphere_enclosure_grasp_success",
        "cube_opposing_face_grasp_success",
        "cylinder_side_body_grasp_success",
        "top_down_cylinder_grasp_count",
        "in_hand_rotation_success",
        "achieved_rotation_deg",
        "rotation_error_deg",
        "stylus_tripod_success",
        "checkpoint_touch_success",
        "index_only_button_press_success",
        "stress_eval_available",
        "dataset_stress_eval_path",
        "stress_eval_summary",
        "tactile_channels",
        "touch_sensor_count",
        "mujoco_touch_sensors_present",
        "sensorized_fingertip_count",
        "active_contact_confidence",
        "dexterous_contact_confidence",
        "tactile_taxel_audit_confidence",
        "cap_rotation_target_deg",
        "cap_rotation_achieved_deg",
        "cap_rotation_error_deg",
        "cap_rotation_success",
        "final_slip_mm",
        "max_slip_mm",
        "slip_recovery_success",
        "load_hold_x",
        "load_hold_success",
        "object_drop_count",
        "task_gate_count",
        "task_gates_passed",
        "task_gate_success_rate",
        "stress_rollouts",
        "stress_success_rate",
        "baseline_success_rate",
        "feedback_success_rate",
        "improvement_percentage",
        "average_active_fingers_dexterous_grasps",
        "average_multi_side_contact_score_dexterous_grasps",
        "minimum_jerk_controller_pass",
        "hardware_audit_pass",
        "object_center_between_fingers_rate",
        "contact_timeline_path",
        "judge_summary_path",
        "evidence_index_path",
        "overall_task_success",
        "blind_tactile_mode_available",
        "unknown_object_arena_available",
        "tactile_classifier_accuracy",
        "classification_confidence_mean",
        "average_probes_per_object",
        "blind_tactile_success_rate",
        "adaptive_regrasp_success_rate",
        "unknown_arena_success_rate",
        "blind_tactile_summary_path",
        "no_ground_truth_pose_mode_available",
        "ground_truth_pose_hidden_from_controller",
        "ground_truth_used_only_for_scoring",
        "tactile_pose_estimator_enabled",
        "pose_estimation_success",
        "estimated_object_center_error_m",
        "estimated_axis_error_deg",
        "estimated_orientation_error_deg",
        "precision_assembly_arena_available",
        "assembly_success",
        "insertion_depth_ratio",
        "socket_alignment_error_m",
        "socket_angle_error_deg",
        "jam_detection_available",
        "jam_recovery_report_path",
        "precision_assembly_report_path",
        "assembly_stress_eval_available",
        "assembly_success_rate",
        "jam_recovery_success_rate",
        "mean_pose_estimation_error_m",
        "combination_lock_task_available",
        "combination_lock_success",
        "combination_lock_code_sequence",
        "combination_lock_detected_sequence",
        "combination_lock_steps",
        "detent_detection_success",
        "detent_count",
        "latch_pull_success",
        "micro_door_opened",
        "combination_lock_max_error_deg",
        "combination_lock_contact_confidence",
        "combination_lock_report_path",
        "combination_lock_trace_path",
        "event_rules_report_path",
        "submission_readiness_report_path",
        "rubric_readiness_report_path",
        "rubric_readiness_scorecard_path",
        "code_quality_report_path",
        "unit_test_report_path",
        "local_readiness_score_estimate_not_official",
        "rubric_readiness_pass",
        "code_quality_pass",
        "demo_video_duration_rule_pass",
        "video_render_mode",
        "runability_status",
        "rules_alignment_pass",
        "judge_replay_index_available",
        "judge_replay_index_path",
        "video_replay_scorecard_path",
        "video_replay_milestone_count",
        "video_replay_milestones_present",
        "video_replay_coverage_rate",
        "rubric_replay_category_count",
        "rubric_replay_all_categories_present",
        "closed_loop_reflex_benchmark_available",
        "closed_loop_reflex_report_path",
        "closed_loop_reflex_trace_path",
        "closed_loop_reflex_scorecard_path",
        "closed_loop_reflex_success",
        "reflex_response_latency_ms",
        "reflex_latency_pass",
        "reflex_latency_threshold_ms",
        "reflex_final_slip_mm",
        "reflex_pressure_boost_n",
        "reflex_active_fingers",
        "vial_uncap_deliver_task_available",
        "vial_uncap_deliver_visual_segment_present",
        "vial_uncap_delivery_benchmark_available",
        "vial_uncap_delivery_report_path",
        "vial_uncap_delivery_trace_path",
        "vial_uncap_delivery_scorecard_path",
        "vial_uncap_deliver_success",
        "vial_grasp_verified",
        "vial_cap_rotation_target_deg",
        "vial_cap_rotation_achieved_deg",
        "vial_cap_removed",
        "vial_max_force_n",
        "vial_no_crush_force_limit_n",
        "vial_no_crush_force_pass",
        "pill_delivery_success",
        "pill_in_tray",
        "microsuture_task_available",
        "microsuture_visual_segment_present",
        "microsuture_threading_success",
        "microsuture_target_passes",
        "microsuture_pass_count",
        "microsuture_entry_error_m",
        "microsuture_exit_error_m",
        "microsuture_max_tension_n",
        "microsuture_tension_limit_n",
        "microsuture_no_tear_pass",
        "microsuture_knot_tension_success",
        "microsuture_benchmark_report_path",
        "microsuture_threading_trace_path",
        "microsuture_scorecard_path",
    ]
    missing_metrics = [metric for metric in required_metrics if metric not in summary]
    if missing_metrics:
        print("DexHand validation failed")
        print("Missing summary metrics:")
        for metric in missing_metrics:
            print(f"- {metric}")
        return 1
    expected_values = {
        "hand_skeleton_valid": True,
        "five_fingers_present": True,
        "thumb_opposition_joint_present": True,
        "sphere_enclosure_grasp_success": True,
        "cube_opposing_face_grasp_success": True,
        "cylinder_side_body_grasp_success": True,
        "in_hand_rotation_success": True,
        "stylus_tripod_success": True,
        "checkpoint_touch_success": True,
        "index_only_button_press_success": True,
        "cap_rotation_success": True,
        "slip_recovery_success": True,
        "load_hold_success": True,
        "minimum_jerk_controller_pass": True,
        "hardware_audit_pass": True,
        "overall_task_success": True,
        "ground_truth_pose_hidden_from_controller": True,
        "ground_truth_used_only_for_scoring": True,
        "tactile_pose_estimator_enabled": True,
        "pose_estimation_success": True,
        "precision_assembly_arena_available": True,
        "assembly_success": True,
        "combination_lock_task_available": True,
        "combination_lock_success": True,
        "detent_detection_success": True,
        "latch_pull_success": True,
        "micro_door_opened": True,
        "contact_causality_audit_available": True,
        "contact_causality_pass": True,
        "assembly_visual_segment_present": True,
        "combination_lock_visual_segment_present": True,
        "judge_replay_index_available": True,
        "rubric_replay_all_categories_present": True,
        "closed_loop_reflex_benchmark_available": True,
        "closed_loop_reflex_success": True,
        "reflex_latency_pass": True,
        "vial_uncap_deliver_task_available": True,
        "vial_uncap_deliver_visual_segment_present": True,
        "vial_uncap_delivery_benchmark_available": True,
        "vial_uncap_deliver_success": True,
        "vial_grasp_verified": True,
        "vial_cap_removed": True,
        "vial_no_crush_force_pass": True,
        "pill_delivery_success": True,
        "pill_in_tray": True,
        "microsuture_task_available": True,
        "microsuture_visual_segment_present": True,
        "microsuture_threading_success": True,
        "microsuture_no_tear_pass": True,
        "microsuture_knot_tension_success": True,
    }
    bad_values = [
        f"{metric} expected {expected!r}, got {summary.get(metric)!r}"
        for metric, expected in expected_values.items()
        if summary.get(metric) != expected
    ]
    if int(summary.get("object_snap_events", 1)) != 0:
        bad_values.append("object_snap_events expected 0")
    if int(summary.get("attach_before_verification_count", 1)) != 0:
        bad_values.append("attach_before_verification_count expected 0")
    if int(summary.get("top_down_cylinder_grasp_count", 1)) != 0:
        bad_values.append("top_down_cylinder_grasp_count expected 0")
    if float(summary.get("verified_grasp_before_attach_rate", 0.0)) < 0.99:
        bad_values.append("verified_grasp_before_attach_rate expected >= 0.99")
    if float(summary.get("object_center_between_fingers_rate", 0.0)) < 0.99:
        bad_values.append("object_center_between_fingers_rate expected >= 0.99")
    if not bool(summary.get("stress_eval_available", False)):
        bad_values.append("stress_eval_available expected true; run run_stress_eval.py --seeds 32")
    if not summary.get("dataset_stress_eval_path"):
        bad_values.append("dataset_stress_eval_path expected in summary")
    if not isinstance(summary.get("stress_eval_summary"), dict) or not summary.get("stress_eval_summary"):
        bad_values.append("stress_eval_summary expected non-empty dict in summary")
    if int(summary.get("tactile_channels", 0)) != 5:
        bad_values.append("tactile_channels expected 5")
    if int(summary.get("touch_sensor_count", 0)) != 5:
        bad_values.append("touch_sensor_count expected 5")
    if not bool(summary.get("mujoco_touch_sensors_present", False)):
        bad_values.append("mujoco_touch_sensors_present expected true")
    if int(summary.get("sensorized_fingertip_count", 0)) != 5:
        bad_values.append("sensorized_fingertip_count expected 5")
    if float(summary.get("cap_rotation_target_deg", 0.0)) != 224.0:
        bad_values.append("cap_rotation_target_deg expected 224")
    if float(summary.get("cap_rotation_achieved_deg", 0.0)) < 214.0:
        bad_values.append("cap_rotation_achieved_deg expected >= 214")
    if float(summary.get("final_slip_mm", 999.0)) > 0.5:
        bad_values.append("final_slip_mm expected <= 0.5")
    if float(summary.get("load_hold_x", 0.0)) < 5.0:
        bad_values.append("load_hold_x expected >= 5.0")
    if float(summary.get("task_gate_success_rate", 0.0)) < 0.90:
        bad_values.append("task_gate_success_rate expected >= 0.90")
    if float(summary.get("feedback_success_rate", 0.0)) < float(summary.get("baseline_success_rate", 0.0)):
        bad_values.append("feedback_success_rate expected >= baseline_success_rate")
    if float(summary.get("average_active_fingers_dexterous_grasps", 0.0)) < 4.0:
        bad_values.append("average_active_fingers_dexterous_grasps expected >= 4.0")
    if float(summary.get("average_multi_side_contact_score_dexterous_grasps", 0.0)) < 0.80:
        bad_values.append("average_multi_side_contact_score_dexterous_grasps expected >= 0.80")
    if not bool(summary.get("blind_tactile_mode_available", False)):
        bad_values.append("blind_tactile_mode_available expected true")
    if not bool(summary.get("unknown_object_arena_available", False)):
        bad_values.append("unknown_object_arena_available expected true")
    if float(summary.get("tactile_classifier_accuracy", 0.0)) < 0.90:
        bad_values.append("tactile_classifier_accuracy expected >= 0.90")
    if float(summary.get("blind_tactile_success_rate", 0.0)) < 0.80:
        bad_values.append("blind_tactile_success_rate expected >= 0.80")
    if float(summary.get("adaptive_regrasp_success_rate", 0.0)) < 0.80:
        bad_values.append("adaptive_regrasp_success_rate expected >= 0.80")
    if float(summary.get("average_probes_per_object", 0.0)) <= 0.0:
        bad_values.append("average_probes_per_object expected > 0")
    if not bool(summary.get("no_ground_truth_pose_mode_available", False)):
        bad_values.append("no_ground_truth_pose_mode_available expected true")
    if float(summary.get("estimated_object_center_error_m", 999.0)) > 0.012:
        bad_values.append("estimated_object_center_error_m expected <= 0.012")
    if float(summary.get("estimated_axis_error_deg", 999.0)) > 12.0:
        bad_values.append("estimated_axis_error_deg expected <= 12")
    if float(summary.get("insertion_depth_ratio", 0.0)) < 0.85:
        bad_values.append("insertion_depth_ratio expected >= 0.85")
    if not bool(summary.get("jam_detection_available", False)):
        bad_values.append("jam_detection_available expected true")
    if not bool(summary.get("assembly_stress_eval_available", False)):
        bad_values.append("assembly_stress_eval_available expected true; run run_stress_eval.py --seeds 32 --arena assembly --blind-tactile --no-ground-truth-pose")
    if float(summary.get("assembly_success_rate", 0.0)) < 0.80:
        bad_values.append("assembly_success_rate expected >= 0.80")
    if float(summary.get("jam_recovery_success_rate", 0.0)) < 0.80:
        bad_values.append("jam_recovery_success_rate expected >= 0.80")
    if int(summary.get("detent_count", 0)) < 3:
        bad_values.append("detent_count expected >= 3")
    if float(summary.get("combination_lock_max_error_deg", 999.0)) > 4.0:
        bad_values.append("combination_lock_max_error_deg expected <= 4.0")
    if float(summary.get("combination_lock_contact_confidence", 0.0)) < 0.85:
        bad_values.append("combination_lock_contact_confidence expected >= 0.85")
    if float(summary.get("verified_motion_frame_rate", 0.0)) < 0.95:
        bad_values.append("verified_motion_frame_rate expected >= 0.95")
    if int(summary.get("pre_verification_motion_events", 1)) != 0:
        bad_values.append("pre_verification_motion_events expected 0")
    if int(summary.get("video_replay_milestones_present", 0)) < 10:
        bad_values.append("video_replay_milestones_present expected >= 10")
    if float(summary.get("video_replay_coverage_rate", 0.0)) < 0.90:
        bad_values.append("video_replay_coverage_rate expected >= 0.90")
    if int(summary.get("rubric_replay_category_count", 0)) < 8:
        bad_values.append("rubric_replay_category_count expected >= 8")
    if float(summary.get("reflex_response_latency_ms", 999.0)) > float(summary.get("reflex_latency_threshold_ms", 20.0)):
        bad_values.append("reflex_response_latency_ms expected <= threshold")
    if float(summary.get("reflex_final_slip_mm", 999.0)) > 0.5:
        bad_values.append("reflex_final_slip_mm expected <= 0.5")
    if int(summary.get("reflex_active_fingers", 0)) < 4:
        bad_values.append("reflex_active_fingers expected >= 4")
    if float(summary.get("vial_cap_rotation_achieved_deg", 0.0)) < 150.0:
        bad_values.append("vial_cap_rotation_achieved_deg expected >= 150")
    if float(summary.get("vial_max_force_n", 99.0)) > float(summary.get("vial_no_crush_force_limit_n", 0.0)):
        bad_values.append("vial_max_force_n expected <= vial_no_crush_force_limit_n")
    if int(summary.get("microsuture_pass_count", 0)) < int(summary.get("microsuture_target_passes", 2)):
        bad_values.append("microsuture_pass_count expected >= microsuture_target_passes")
    if float(summary.get("microsuture_entry_error_m", 99.0)) > 0.004:
        bad_values.append("microsuture_entry_error_m expected <= 0.004")
    if float(summary.get("microsuture_exit_error_m", 99.0)) > 0.004:
        bad_values.append("microsuture_exit_error_m expected <= 0.004")
    if float(summary.get("microsuture_max_tension_n", 99.0)) > float(summary.get("microsuture_tension_limit_n", 0.0)):
        bad_values.append("microsuture_max_tension_n expected <= microsuture_tension_limit_n")
    if not bool(summary.get("demo_video_duration_rule_pass", False)):
        bad_values.append("demo_video_duration_rule_pass expected true for 1-3 minute event video")
    if float(summary.get("duration_s", 0.0)) < 60.0 or float(summary.get("duration_s", 0.0)) > 180.0:
        bad_values.append("duration_s expected inside 60-180 second event window")
    if str(summary.get("runability_status", "")).lower() != "pass":
        bad_values.append("runability_status expected pass")
    if not bool(summary.get("rules_alignment_pass", False)):
        bad_values.append("rules_alignment_pass expected true")
    expected_uuid = str(registration.get("uuid", "")).strip()
    uuid_sources = {
        "registration.json": registration.get("uuid"),
        "outputs/summary.json": summary.get("uuid"),
        "submission_manifest.json": manifest.get("registration_uuid"),
        "rubric_scorecard.json": rubric.get("registration_uuid"),
        "outputs/event_rules_report.json": event_rules.get("registration_uuid"),
        "outputs/submission_readiness_report.json": readiness.get("registration_uuid"),
    }
    mismatched_uuid_sources = [
        f"{source}={value!r}" for source, value in uuid_sources.items() if str(value).strip() != expected_uuid
    ]
    if not expected_uuid:
        bad_values.append("registration.json uuid must be present")
    if mismatched_uuid_sources:
        bad_values.append("UUID mismatch across submission evidence: " + ", ".join(mismatched_uuid_sources))
    if not bool(readiness.get("uuid_consistency_pass", False)):
        bad_values.append("submission_readiness_report uuid_consistency_pass expected true")
    if not bool(readiness.get("submission_readiness_pass", False)):
        bad_values.append("submission_readiness_report submission_readiness_pass expected true")
    if bool(summary.get("submission_readiness_pass")) != bool(readiness.get("submission_readiness_pass")):
        bad_values.append("summary submission_readiness_pass must match submission_readiness_report")
    stale_pr_markers = tuple(f"pull/{number}" for number in (151, 160, 477))
    pr_target = str(readiness.get("pr_target", ""))
    if any(marker in pr_target for marker in stale_pr_markers):
        bad_values.append("submission_readiness_report pr_target points to a stale PR")
    deprecated_backup_files = [
        PROJECT_DIR / "BACKUP_874_SCORE_STATE.md",
        PROJECT_DIR / "BACKUP_PR151_STATE.md",
    ]
    existing_backups = [path.name for path in deprecated_backup_files if path.exists()]
    if existing_backups:
        bad_values.append("deprecated score/PR backup files should not ship: " + ", ".join(existing_backups))
    srt_end_s = srt_last_timestamp_s(OUTPUT_DIR / "narration.srt")
    if srt_end_s > float(summary.get("duration_s", 0.0)) + 1.0:
        bad_values.append("narration.srt extends past demo duration")
    if not bool(quality_report.get("code_quality_pass", False)):
        bad_values.append("code_quality_report code_quality_pass expected true")
    if not bool(rubric_readiness.get("all_rubric_rows_pass", False)):
        bad_values.append("rubric_readiness_report all_rubric_rows_pass expected true")
    if float(rubric_readiness.get("local_readiness_score_estimate_not_official", 0.0)) < 90.0:
        bad_values.append("local_readiness_score_estimate_not_official expected >= 90")
    if bad_values:
        print("DexHand validation failed")
        print("Unexpected summary values:")
        for item in bad_values:
            print(f"- {item}")
        return 1
    validator_report = {
        "project": "DexHand Lab",
        "validation_passed": True,
        "blind_tactile_evidence": {
            "status": "pass",
            "blind_tactile_mode_available": bool(summary.get("blind_tactile_mode_available")),
            "unknown_object_arena_available": bool(summary.get("unknown_object_arena_available")),
            "tactile_classifier_accuracy": summary.get("tactile_classifier_accuracy"),
            "blind_tactile_success_rate": summary.get("blind_tactile_success_rate"),
            "adaptive_regrasp_success_rate": summary.get("adaptive_regrasp_success_rate"),
            "average_probes_per_object": summary.get("average_probes_per_object"),
            "object_snap_events": summary.get("object_snap_events"),
        },
        "core_evidence": {
            "cap_rotation_success": summary.get("cap_rotation_success"),
            "load_hold_success": summary.get("load_hold_success"),
            "task_gate_success_rate": summary.get("task_gate_success_rate"),
            "feedback_success_rate": summary.get("feedback_success_rate"),
            "baseline_success_rate": summary.get("baseline_success_rate"),
        },
        "microsuture_evidence": {
            "status": "pass",
            "microsuture_task_available": bool(summary.get("microsuture_task_available")),
            "microsuture_visual_segment_present": bool(summary.get("microsuture_visual_segment_present")),
            "microsuture_threading_success": bool(summary.get("microsuture_threading_success")),
            "microsuture_pass_count": summary.get("microsuture_pass_count"),
            "microsuture_target_passes": summary.get("microsuture_target_passes"),
            "microsuture_entry_error_m": summary.get("microsuture_entry_error_m"),
            "microsuture_exit_error_m": summary.get("microsuture_exit_error_m"),
            "microsuture_max_tension_n": summary.get("microsuture_max_tension_n"),
            "microsuture_tension_limit_n": summary.get("microsuture_tension_limit_n"),
            "checked_files": [
                "dataset/microsuture_threading_report.json",
                "dataset/microsuture_threading_trace.csv",
                "outputs/microsuture_scorecard.json",
            ],
        },
        "precision_assembly_evidence": {
            "status": "pass",
            "no_ground_truth_pose_mode_available": bool(summary.get("no_ground_truth_pose_mode_available")),
            "ground_truth_pose_hidden_from_controller": bool(summary.get("ground_truth_pose_hidden_from_controller")),
            "ground_truth_used_only_for_scoring": bool(summary.get("ground_truth_used_only_for_scoring")),
            "pose_estimation_success": bool(summary.get("pose_estimation_success")),
            "estimated_object_center_error_m": summary.get("estimated_object_center_error_m"),
            "estimated_axis_error_deg": summary.get("estimated_axis_error_deg"),
            "assembly_success": bool(summary.get("assembly_success")),
            "insertion_depth_ratio": summary.get("insertion_depth_ratio"),
            "jam_detection_available": bool(summary.get("jam_detection_available")),
            "assembly_stress_eval_available": bool(summary.get("assembly_stress_eval_available")),
            "assembly_success_rate": summary.get("assembly_success_rate"),
            "jam_recovery_success_rate": summary.get("jam_recovery_success_rate"),
            "mean_pose_estimation_error_m": summary.get("mean_pose_estimation_error_m"),
            "checked_files": [
                "dataset/tactile_pose_estimator_report.json",
                "dataset/precision_assembly_report.json",
                "dataset/jam_recovery_report.json",
                "dataset/no_ground_truth_control_audit.json",
                "outputs/assembly_summary.json",
                "media/assembly_keyframes.png",
                "media/tactile_pose_estimation_panel.png",
            ],
        },
        "combination_lock_evidence": {
            "status": "pass",
            "combination_lock_task_available": bool(summary.get("combination_lock_task_available")),
            "combination_lock_success": bool(summary.get("combination_lock_success")),
            "combination_lock_code_sequence": summary.get("combination_lock_code_sequence"),
            "combination_lock_detected_sequence": summary.get("combination_lock_detected_sequence"),
            "detent_detection_success": bool(summary.get("detent_detection_success")),
            "detent_count": summary.get("detent_count"),
            "latch_pull_success": bool(summary.get("latch_pull_success")),
            "micro_door_opened": bool(summary.get("micro_door_opened")),
            "combination_lock_max_error_deg": summary.get("combination_lock_max_error_deg"),
            "combination_lock_contact_confidence": summary.get("combination_lock_contact_confidence"),
            "checked_files": [
                "dataset/combination_lock_report.json",
                "dataset/combination_lock_trace.csv",
                "outputs/combination_lock_summary.json",
                "media/combination_lock_keyframes.png",
            ],
        },
        "contact_causality_evidence": {
            "status": "pass",
            "contact_causality_pass": bool(summary.get("contact_causality_pass")),
            "verified_motion_frame_rate": summary.get("verified_motion_frame_rate"),
            "pre_verification_motion_events": summary.get("pre_verification_motion_events"),
            "object_snap_events": summary.get("object_snap_events"),
            "attach_before_verification_count": summary.get("attach_before_verification_count"),
            "checked_files": [
                "dataset/contact_causality_report.json",
                "dataset/contact_causality_trace.csv",
            ],
        },
        "judge_replay_evidence": {
            "status": "pass",
            "judge_replay_index_available": bool(summary.get("judge_replay_index_available")),
            "video_replay_milestones_present": summary.get("video_replay_milestones_present"),
            "video_replay_milestone_count": summary.get("video_replay_milestone_count"),
            "video_replay_coverage_rate": summary.get("video_replay_coverage_rate"),
            "rubric_replay_category_count": summary.get("rubric_replay_category_count"),
            "rubric_replay_all_categories_present": bool(summary.get("rubric_replay_all_categories_present")),
            "checked_files": [
                "dataset/judge_video_replay_index.json",
                "dataset/judge_video_replay_index.csv",
                "outputs/video_replay_scorecard.json",
            ],
        },
        "closed_loop_reflex_evidence": {
            "status": "pass",
            "closed_loop_reflex_benchmark_available": bool(summary.get("closed_loop_reflex_benchmark_available")),
            "closed_loop_reflex_success": bool(summary.get("closed_loop_reflex_success")),
            "reflex_response_latency_ms": summary.get("reflex_response_latency_ms"),
            "reflex_latency_threshold_ms": summary.get("reflex_latency_threshold_ms"),
            "reflex_pressure_boost_n": summary.get("reflex_pressure_boost_n"),
            "reflex_final_slip_mm": summary.get("reflex_final_slip_mm"),
            "reflex_active_fingers": summary.get("reflex_active_fingers"),
            "checked_files": [
                "dataset/closed_loop_reflex_report.json",
                "dataset/closed_loop_reflex_trace.csv",
                "outputs/closed_loop_reflex_scorecard.json",
            ],
        },
        "event_rules_alignment": {
            "status": "pass",
            "event_rules_report_path": summary.get("event_rules_report_path"),
            "demo_video_duration_rule_pass": summary.get("demo_video_duration_rule_pass"),
            "duration_s": summary.get("duration_s"),
            "runability_status": summary.get("runability_status"),
            "rules_alignment_pass": summary.get("rules_alignment_pass"),
        },
        "submission_readiness_evidence": {
            "status": "pass",
            "submission_readiness_report_path": "submissions/dexhand_lab/outputs/submission_readiness_report.json",
            "uuid_consistency_pass": readiness.get("uuid_consistency_pass"),
            "submission_readiness_pass": readiness.get("submission_readiness_pass"),
            "registration_uuid": expected_uuid,
            "pr_target": readiness.get("pr_target"),
        },
        "quality_gate_evidence": {
            "status": "pass",
            "code_quality_report_path": "submissions/dexhand_lab/dataset/code_quality_report.json",
            "rubric_readiness_report_path": "submissions/dexhand_lab/outputs/rubric_readiness_report.json",
            "unit_test_report_path": "submissions/dexhand_lab/dataset/unit_test_report.json",
            "code_quality_pass": quality_report.get("code_quality_pass"),
            "python_compile_pass": quality_report.get("python_compile_pass"),
            "source_health_pass": quality_report.get("source_health_pass"),
            "local_readiness_score_estimate_not_official": rubric_readiness.get("local_readiness_score_estimate_not_official"),
            "all_rubric_rows_pass": rubric_readiness.get("all_rubric_rows_pass"),
        },
        "event_hygiene": {
            "status": "pass",
            "narration_last_timestamp_s": round(srt_last_timestamp_s(OUTPUT_DIR / "narration.srt"), 3),
            "demo_duration_s": summary.get("duration_s"),
            "summary_readiness_matches_report": bool(summary.get("submission_readiness_pass"))
            == bool(readiness.get("submission_readiness_pass")),
            "stale_pr_target_present": any(
                marker in str(readiness.get("pr_target", ""))
                for marker in (f"pull/{number}" for number in (151, 160, 477))
            ),
            "deprecated_backup_files_present": False,
            "stress_eval_summary_embedded": isinstance(summary.get("stress_eval_summary"), dict)
            and bool(summary.get("stress_eval_summary")),
        },
    }
    validator_report_path = OUTPUT_DIR / "validator_report.json"
    validator_report_path.write_text(json.dumps(validator_report, indent=2), encoding="utf-8")
    summary["validator_report_path"] = "submissions/dexhand_lab/outputs/validator_report.json"
    summary["validation_passed"] = True
    summary["uuid_consistency_pass"] = readiness.get("uuid_consistency_pass")
    summary["required_outputs_present"] = readiness.get("required_outputs_present")
    readiness["validation_passed"] = True
    readiness["required_commands"]["python submissions/dexhand_lab/validate_submission.py"] = {
        "status": "pass",
        "evidence": "submissions/dexhand_lab/outputs/validator_report.json",
    }
    readiness["validator_report_path"] = "submissions/dexhand_lab/outputs/validator_report.json"
    readiness["submission_readiness_pass"] = bool(
        readiness.get("uuid_consistency_pass")
        and readiness.get("required_outputs_present")
        and readiness.get("event_rule_alignment", {}).get("rules_alignment_pass")
        and all(bool(value) for value in readiness.get("scoring_rubric_evidence", {}).values())
    )
    summary["submission_readiness_pass"] = readiness["submission_readiness_pass"]
    (OUTPUT_DIR / "submission_readiness_report.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DexHand validation passed")
    print(f"Summary: {OUTPUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
