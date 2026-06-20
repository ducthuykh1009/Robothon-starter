# DexHand Lab Evidence Index

This is the shortest path through the submission evidence for judges and automated review.

## Registration

- UUID: 2555924c-74a4-4788-be61-1f1e65bf3f44
- Final submission folder: `submissions/dexhand_lab/`

## Inspect First

1. `media/demo.mp4` - generated 75-120 second dexterous hand demo.
2. `media/keyframes.png` - labeled visual evidence grid.
3. `outputs/event_rules_report.json` - explicit mapping to event deliverables and scoring rubric.
4. `outputs/submission_readiness_report.json` - UUID consistency, required command, required output, and PR-target readiness audit.
5. `outputs/judge_summary.json` - compact quantitative evidence.
6. `outputs/summary.json` - full run metrics.
7. `outputs/contact_timeline.json` - per-finger contact timeline.
8. `dataset/task_suite_report.json` - 20-gate verification suite.
9. `dataset/tactile_feedback_report.json` and `dataset/tactile_taxels.csv` - five-fingertip tactile audit.
10. `dataset/minimum_jerk_report.json` - tactile-inspired minimum-jerk controller report.
11. `dataset/stress_eval.json` and `outputs/baseline_vs_feedback.json` - fixed-seed stress comparison.
12. `dataset/hardware_adaptation_report.json` - simulation-to-hardware replay audit.
13. `outputs/blind_tactile_summary.json` - blind tactile active perception summary.
14. `dataset/tactile_classifier_report.json` - tactile shape classifier evidence.
15. `dataset/adaptive_regrasp_report.json` - adaptive regrasp recovery evidence.
16. `media/blind_tactile_keyframes.png` - visual proof of probing/classification/regrasp.
17. `dataset/tactile_pose_estimator_report.json` - no-ground-truth tactile pose estimate and scoring audit.
18. `dataset/precision_assembly_report.json` - plug/socket insertion and compliant retry evidence.
19. `dataset/jam_recovery_report.json` - jam detection, withdraw/correct/retry metrics.
20. `media/assembly_keyframes.png` - visual proof of assembly sequence.
21. `media/tactile_pose_estimation_panel.png` - pose error, axis error, touch activation, and insertion trace.

## Current Metrics

- Task gates: 20/20
- Cap rotation: 224 deg target / 224.0 deg achieved
- Final slip: 0.28 mm
- Load hold: 9.0x
- Tactile channels: 5
- MuJoCo fingertip touch sensors: 5
- Object snap events: 0
- Stress success: 100.0%
- Feedback vs baseline: 1.00 vs 0.59
- Blind tactile classifier accuracy: 1.00
- Blind tactile success rate: 1.00
- Adaptive regrasp success rate: 1.00
- No-ground-truth pose mode: true
- Tactile pose center error: 0.0042 m
- Tactile pose axis error: 5.6 deg
- Assembly success: true
- Insertion depth ratio: 0.92
- Jam detection/recovery evidence: true
- Event rules alignment: true
- Submission readiness audit: submissions/dexhand_lab/outputs/submission_readiness_report.json

## New 95+ Differentiator: Blind Tactile Active Perception

- Object labels are hidden from the controller when `--blind-tactile` is enabled.
- The hand probes unknown objects with index, thumb, and middle fingertips.
- A deterministic tactile classifier estimates curvature, edge response, long-axis signal, twist affordance, and press displacement.
- The selected grasp strategy comes from tactile classification, then adaptive regrasp corrects low-confidence or unstable contact.
- Evidence files: `dataset/tactile_exploration_trace.csv`, `dataset/tactile_classifier_report.json`, `dataset/tactile_confusion_matrix.json`, `dataset/adaptive_regrasp_report.json`, `dataset/unknown_arena_report.json`, and `outputs/blind_tactile_summary.json`.
- Precision assembly evidence: `dataset/tactile_pose_estimator_report.json`, `dataset/precision_assembly_report.json`, `dataset/jam_recovery_report.json`, `dataset/no_ground_truth_control_audit.json`, `outputs/assembly_summary.json`, `media/assembly_keyframes.png`, and `media/tactile_pose_estimation_panel.png`.

## Honest Scope

DexHand Lab uses simulation-native object pose perception and a hybrid contact-aware dexterous manipulation routine. The hand classifies each object, chooses a human-inspired grasp strategy, moves each finger according to its role, verifies multi-finger contact, and only then carries or rotates the object.
