from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"


def load_json(relative_path: str) -> dict:
    return json.loads((PROJECT_DIR / relative_path).read_text(encoding="utf-8"))


class DexHandSubmissionContractTest(unittest.TestCase):
    def test_uuid_consistency(self) -> None:
        registration = load_json("registration.json")
        summary = load_json("outputs/summary.json")
        manifest = load_json("submission_manifest.json")
        readiness = load_json("outputs/submission_readiness_report.json")
        expected_uuid = registration["uuid"]
        self.assertEqual(summary["uuid"], expected_uuid)
        self.assertEqual(summary["registration_uuid"], expected_uuid)
        self.assertEqual(manifest["registration_uuid"], expected_uuid)
        self.assertEqual(readiness["registration_uuid"], expected_uuid)
        self.assertTrue(readiness["uuid_consistency_pass"])

    def test_core_metrics_stay_event_ready(self) -> None:
        summary = load_json("outputs/summary.json")
        validator = load_json("outputs/validator_report.json")
        self.assertTrue(summary.get("validation_passed", validator.get("validation_passed", False)))
        self.assertTrue(summary["rules_alignment_pass"])
        self.assertTrue(summary["demo_video_duration_rule_pass"])
        self.assertGreaterEqual(float(summary["duration_s"]), 60.0)
        self.assertLessEqual(float(summary["duration_s"]), 180.0)
        self.assertEqual(int(summary["object_snap_events"]), 0)
        self.assertEqual(int(summary["attach_before_verification_count"]), 0)
        self.assertEqual(float(summary["cap_rotation_target_deg"]), 224.0)
        self.assertGreaterEqual(float(summary["cap_rotation_achieved_deg"]), 214.0)
        self.assertGreaterEqual(float(summary["load_hold_x"]), 5.0)
        self.assertEqual(int(summary["tactile_channels"]), 5)

    def test_advanced_evidence_is_present(self) -> None:
        summary = load_json("outputs/summary.json")
        required_true = [
            "blind_tactile_mode_available",
            "unknown_object_arena_available",
            "no_ground_truth_pose_mode_available",
            "pose_estimation_success",
            "precision_assembly_arena_available",
            "assembly_success",
            "jam_detection_available",
            "combination_lock_task_available",
            "combination_lock_success",
            "detent_detection_success",
            "latch_pull_success",
            "micro_door_opened",
            "assembly_visual_segment_present",
            "combination_lock_visual_segment_present",
            "contact_causality_audit_available",
            "contact_causality_pass",
            "minimum_jerk_controller_pass",
            "hardware_audit_pass",
        ]
        for metric in required_true:
            with self.subTest(metric=metric):
                self.assertTrue(summary[metric])
        self.assertGreaterEqual(float(summary["tactile_classifier_accuracy"]), 0.90)
        self.assertGreaterEqual(float(summary["assembly_success_rate"]), 0.80)
        self.assertGreaterEqual(float(summary["jam_recovery_success_rate"]), 0.80)
        self.assertLessEqual(float(summary["combination_lock_max_error_deg"]), 4.0)
        self.assertGreaterEqual(float(summary["verified_motion_frame_rate"]), 0.95)
        self.assertEqual(int(summary["pre_verification_motion_events"]), 0)

    def test_required_media_and_reports_exist(self) -> None:
        required_paths = [
            "outputs/demo.mp4",
            "media/demo.mp4",
            "media/keyframes.png",
            "media/blind_tactile_keyframes.png",
            "media/assembly_keyframes.png",
            "media/tactile_pose_estimation_panel.png",
            "media/combination_lock_keyframes.png",
            "outputs/contact_timeline.json",
            "outputs/final_report.txt",
            "outputs/event_rules_report.json",
            "outputs/submission_readiness_report.json",
            "dataset/task_suite_report.json",
            "dataset/tactile_feedback_report.json",
            "dataset/tactile_pose_estimator_report.json",
            "dataset/precision_assembly_report.json",
            "dataset/jam_recovery_report.json",
            "dataset/combination_lock_report.json",
            "dataset/combination_lock_trace.csv",
            "dataset/contact_causality_report.json",
            "dataset/contact_causality_trace.csv",
            "dataset/hardware_adaptation_report.json",
        ]
        for relative_path in required_paths:
            with self.subTest(path=relative_path):
                path = PROJECT_DIR / relative_path
                self.assertTrue(path.exists(), relative_path)
                self.assertGreater(path.stat().st_size, 0, relative_path)


if __name__ == "__main__":
    unittest.main()
