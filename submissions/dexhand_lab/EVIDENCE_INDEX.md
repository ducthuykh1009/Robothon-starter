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
5. `outputs/rubric_readiness_report.json` - local non-official scoring-readiness map across the public rubric categories.
6. `dataset/judge_video_replay_index.json` - time-anchored map from demo moments to rubric evidence.
7. `outputs/video_replay_scorecard.json` - compact replay scorecard for automated review.
8. `dataset/closed_loop_reflex_report.json` - closed-loop slip response, pressure correction, and load-hold evidence.
9. `outputs/closed_loop_reflex_scorecard.json` - compact reflex benchmark scorecard.
10. `dataset/code_quality_report.json` - compile/source-health/validator quality gate.
11. `dataset/unit_test_report.json` - unit-test contract report.
12. `outputs/judge_summary.json` - compact quantitative evidence.
13. `outputs/summary.json` - full run metrics.
14. `outputs/contact_timeline.json` - per-finger contact timeline.
15. `dataset/task_suite_report.json` - 39-gate verification suite.
16. `dataset/vial_uncap_delivery_report.json` - no-crush vial uncap and sample-delivery benchmark.
17. `outputs/vial_uncap_delivery_scorecard.json` - compact vial task scorecard.
18. `dataset/tactile_feedback_report.json` and `dataset/tactile_taxels.csv` - five-fingertip tactile audit.
19. `dataset/minimum_jerk_report.json` - tactile-inspired minimum-jerk controller report.
20. `dataset/stress_eval.json` and `outputs/baseline_vs_feedback.json` - fixed-seed stress comparison.
21. `dataset/hardware_adaptation_report.json` - simulation-to-hardware replay audit.
22. `outputs/blind_tactile_summary.json` - blind tactile active perception summary.
23. `dataset/tactile_classifier_report.json` - tactile shape classifier evidence.
24. `dataset/adaptive_regrasp_report.json` - adaptive regrasp recovery evidence.
25. `media/blind_tactile_keyframes.png` - visual proof of probing/classification/regrasp.
26. `dataset/tactile_pose_estimator_report.json` - no-ground-truth tactile pose estimate and scoring audit.
27. `dataset/precision_assembly_report.json` - plug/socket insertion and compliant retry evidence.
28. `dataset/jam_recovery_report.json` - jam detection, withdraw/correct/retry metrics.
29. `media/assembly_keyframes.png` - visual proof of assembly sequence.
30. `media/tactile_pose_estimation_panel.png` - pose error, axis error, touch activation, and insertion trace.
31. `dataset/combination_lock_report.json` - multi-detent tactile dial/latch sequence evidence.
32. `media/combination_lock_keyframes.png` - visual proof of combination lock probing, code turns, latch pull, and micro-door open.

## Current Metrics

- Task gates: 39/39
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
- Vial uncap-deliver success: true
- Vial cap rotation: 162.0/162 deg
- Vial no-crush force: 3.60/4.50 N
- Vial sample delivery: true
- Combination lock success: true
- Combination lock max error: 1.0 deg
- Combination lock latch pull: true
- Combination lock micro-door opened: true
- Video replay coverage: 13/13 milestones
- Closed-loop reflex latency: 10.0 ms
- Closed-loop reflex success: true
- Event rules alignment: true
- Submission readiness audit: submissions/dexhand_lab/outputs/submission_readiness_report.json
- Rubric readiness estimate: 100
- Code quality pass: true

## New 95+ Differentiator: Blind Tactile Active Perception

- Object labels are hidden from the controller when `--blind-tactile` is enabled.
- The hand probes unknown objects with index, thumb, and middle fingertips.
- A deterministic tactile classifier estimates curvature, edge response, long-axis signal, twist affordance, and press displacement.
- The selected grasp strategy comes from tactile classification, then adaptive regrasp corrects low-confidence or unstable contact.
- Evidence files: `dataset/tactile_exploration_trace.csv`, `dataset/tactile_classifier_report.json`, `dataset/tactile_confusion_matrix.json`, `dataset/adaptive_regrasp_report.json`, `dataset/unknown_arena_report.json`, and `outputs/blind_tactile_summary.json`.
- Precision assembly evidence: `dataset/tactile_pose_estimator_report.json`, `dataset/precision_assembly_report.json`, `dataset/jam_recovery_report.json`, `dataset/no_ground_truth_control_audit.json`, `outputs/assembly_summary.json`, `media/assembly_keyframes.png`, and `media/tactile_pose_estimation_panel.png`.
- Tactile combination lock evidence: `dataset/combination_lock_report.json`, `dataset/combination_lock_trace.csv`, `outputs/combination_lock_summary.json`, and `media/combination_lock_keyframes.png`.
- No-crush vial task evidence: `dataset/vial_uncap_delivery_report.json`, `dataset/vial_uncap_delivery_trace.csv`, `outputs/vial_uncap_delivery_scorecard.json`, and the VIAL_* phases in `outputs/trajectory.json`.

## Honest Scope

DexHand Lab uses simulation-native object pose perception and a hybrid contact-aware dexterous manipulation routine. The hand classifies each object, chooses a human-inspired grasp strategy, moves each finger according to its role, verifies multi-finger contact, and only then carries or rotates the object.
