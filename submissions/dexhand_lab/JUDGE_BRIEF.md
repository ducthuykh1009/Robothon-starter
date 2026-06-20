# DexHand Lab Judge Brief

## One-Sentence Summary

DexHand Lab is a human-like five-finger MuJoCo hand benchmark for object-specific grasping, blind tactile active perception, adaptive regrasp, cylinder in-hand rotation, a 224-degree cap/knob twist, tactile/contact evidence, slip recovery, load hold, stylus interaction, and index-only button pressing.

## Why This Targets 95+

The submission focuses on dexterity evidence instead of a simple pick-and-place animation: five independent fingers, thumb opposition, object-specific grasp strategies, blind tactile probing/classification, adaptive regrasp, no-snap verification before object motion, MuJoCo fingertip touch sensors plus tactile proxy streams, signature cap rotation, load-hold recovery, stress evaluation, and a multi-gate judge checklist.

## Inspect First

1. `media/demo.mp4`
2. `outputs/demo.mp4`
3. `media/keyframes.png`
4. `outputs/judge_summary.json`
5. `EVIDENCE_INDEX.md`
6. `outputs/event_rules_report.json`
7. `outputs/submission_readiness_report.json`
8. `outputs/blind_tactile_summary.json`
9. `dataset/tactile_classifier_report.json`
10. `dataset/tactile_confusion_matrix.json`
11. `dataset/adaptive_regrasp_report.json`
12. `dataset/unknown_arena_report.json`
13. `media/blind_tactile_keyframes.png`
14. `media/tactile_classifier_panel.png`
15. `outputs/summary.json`
16. `outputs/contact_timeline.json`
17. `dataset/task_suite_report.json`
18. `dataset/tactile_feedback_report.json`
19. `dataset/tactile_taxels.csv`
20. `dataset/minimum_jerk_report.json`
21. `dataset/stress_eval.json`
22. `outputs/baseline_vs_feedback.json`
23. `dataset/hardware_adaptation_report.json`
24. `dataset/tactile_pose_estimator_report.json`
25. `dataset/precision_assembly_report.json`
26. `dataset/jam_recovery_report.json`
27. `dataset/no_ground_truth_control_audit.json`
28. `media/assembly_keyframes.png`
29. `media/tactile_pose_estimation_panel.png`
30. `rubric_scorecard.json`
31. `validate_submission.py`

Runability note: `python submissions/dexhand_lab/run_demo.py` preserves the included generated demo video and refreshes JSON/CSV evidence quickly for judge reproducibility. Use `python submissions/dexhand_lab/run_demo.py --force-render-video` when a fresh MuJoCo frame render is desired.

## New 95+ Differentiator: Blind Tactile Active Perception

When `--blind-tactile` is enabled, object labels are hidden from the controller decision path. The hand probes unknown objects with the index fingertip, thumb, and middle finger, estimates curvature, edge response, flat-face response, long-axis signal, twist affordance, and press displacement, then selects the grasp strategy from tactile evidence. If confidence is low or contact is unstable, adaptive regrasp adds support, recenters, switches strategy, or performs an extra probe.

The unknown-object arena is enabled with `--arena unknown --blind-tactile`. It randomizes object order by seed and logs tactile classification, selected grasp strategy, confidence, fallback use, and regrasp corrections.

The main rendered demo also contains a visible blind tactile segment before the 224-degree cap twist. The overlay shows label hiding, probe count, probing finger, classifier confidence, selected grasp strategy, and adaptive regrasp precheck. This keeps the new capability judge-visible rather than only present in machine-readable evidence.

## New 95+ Differentiator: Tactile Pose Estimation + Precision Assembly

The assembly arena goes beyond hiding labels: with `--no-ground-truth-pose`, the controller does not use exact object pose for decisions. The hand probes a small plug/key, estimates center, long axis, orientation, and grasp affordance points from fingertip/contact history, then uses that estimate to execute a precision plug/socket insertion task.

The assembly controller performs precision grasp selection, in-hand orientation correction, socket search/alignment, compliant insertion, jam detection, withdraw-and-correct recovery, retry insertion, and final insertion verification. Ground-truth pose is logged only after the episode for scoring and audit. Evidence is in `dataset/tactile_pose_estimator_report.json`, `dataset/precision_assembly_report.json`, `dataset/jam_recovery_report.json`, `dataset/no_ground_truth_control_audit.json`, `outputs/assembly_summary.json`, `media/assembly_keyframes.png`, and `media/tactile_pose_estimation_panel.png`.

## Quantitative Evidence

The main demo and evidence scripts report:

- 20-gate deterministic dexterity suite with actual passed count in `dataset/task_suite_report.json`
- cap rotation target: 224 degrees
- cap rotation achieved: saved as `cap_rotation_achieved_deg`
- main demo duration: saved as `duration_s`, currently about 118.6 seconds
- blind tactile visible in main demo: saved as `demo_contains_blind_tactile_segment`
- final slip: saved as `final_slip_mm`
- load hold: saved as `load_hold_x`
- tactile channels: 5 fingertip streams with 5 MuJoCo touch sensors plus pressure/shear/friction proxies
- blind tactile classifier accuracy and confusion matrix
- adaptive regrasp success rate and trace
- unknown arena success rate
- no-ground-truth pose audit
- tactile pose center/axis/orientation error
- plug/socket insertion depth ratio
- jam detection and recovery trace
- stress success rate and baseline-vs-feedback comparison
- object snap events: expected 0
- average active fingers for dexterous grasps
- average multi-side contact score for dexterous grasps
- hardware replay audit status
- event rule alignment pass/fail in `outputs/event_rules_report.json`
- submission readiness pass/fail, UUID consistency, required outputs, and PR target in `outputs/submission_readiness_report.json`

## Rubric Mapping

- Reproducibility: fixed seed CLI, deterministic task suite, validator, stress evaluation.
- MuJoCo depth: articulated hand MJCF, named geoms/sites, fingertip collision pads, five fingertip touch sensors, contact timeline, object state logs, cap hinge joint, plug/socket collision geoms.
- Task design: sphere, cube, cylinder, cap rotation, slip/load-hold, stylus checkpoint, button press.
- Control: contact-aware verified grasp routine, minimum-jerk tactile-inspired segments, no-snap policy, blind tactile probing, confidence thresholding, tactile pose estimation, compliant insertion, jam detection, adaptive regrasp.
- Dexterity: thumb opposition, independent finger roles, multi-side contact, cylinder rotation, cap twist, multi-finger active perception.
- Engineering quality: JSON/CSV evidence pack, validator, manifest, final report, structured modules.
- Presentation: long demo video, keyframes, narration SRT, final evidence report.
- Innovation: blind tactile active perception arena, no-ground-truth tactile pose estimation, precision assembly with jam recovery, cap/knob 224-degree marker task, tactile proxy audit, adaptive regrasp, hardware replay safety audit.

## Honest Scope

The project uses simulation-native object pose perception and a hybrid contact-aware dexterous manipulation routine. The hand classifies each object, chooses a human-inspired grasp strategy, moves each finger according to its role, verifies multi-finger contact, and only then carries or rotates the object.

This is not a learned RL policy, not real camera vision, not a claim of perfect contact physics, and not a real hardware execution. The blind tactile classifier is deterministic and feature-based, using MuJoCo/contact-derived tactile proxies. The tactile channels combine MuJoCo fingertip touch sensors with controller pressure/shear proxies. The hardware adaptation audit is a replay/safety validation artifact for possible LEAP/Shadow-style transfer.

The tactile pose estimator is deterministic and contact/proxy based. In `--no-ground-truth-pose` mode, exact object pose is hidden from controller decisions and used only after the episode for scoring. The assembly plug/socket is benchmark-designed for stable MuJoCo evaluation rather than an arbitrary real-world connector.

## Key Files

- `run_demo.py`: main deterministic demo and data writer.
- `scene.xml`: custom five-finger MJCF hand and task board objects.
- `human_grasp_library.py`: grasp primitives and finger roles.
- `object_classifier.py`: simulation-native affordance classifier.
- `minimum_jerk_controller.py`: tactile-inspired trajectory evidence.
- `contact_feedback_audit.py`: five-fingertip tactile evidence.
- `arena_task_suite.py`: 20-gate verification suite.
- `run_stress_eval.py`: fixed-seed stress evaluation and baseline comparison.
- `hardware_adaptation_audit.py`: simulated hardware replay audit.
- `validate_submission.py`: final evidence and metric validator.
- `tactile_active_perception.py`: blind tactile probing, unknown-object arena, visual panels, and blind stress evidence.
- `tactile_shape_classifier.py`: deterministic tactile feature classifier and confusion matrix.
- `adaptive_regrasp_policy.py`: adaptive recovery/regrasp report and trace.
- `tactile_pose_estimator.py`: no-ground-truth tactile pose estimator and audit.
- `precision_assembly_controller.py`: plug/socket assembly, compliant insertion, jam recovery, and assembly stress evidence.
