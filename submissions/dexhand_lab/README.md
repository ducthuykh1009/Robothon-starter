# DexHand Lab

DexHand Lab is a hand-only MuJoCo dexterous manipulation benchmark built around a human-like five-finger robotic hand. The project demonstrates object-specific grasping, in-hand rotation, a signature 224-degree cap/knob twist, a tactile combination lock task, tactile/contact evidence, slip recovery, load hold, and judge-readable evaluation artifacts.

## Project Summary

DexHand Lab focuses on dexterous hand manipulation rather than a full robot arm. The default demo shows a five-finger hand opening, grasping a sphere, cube, cylinder, stylus, and cap/knob, rotating objects after contact verification, pressing a button with the index finger, and writing reproducible robot-learning data.

## Event Rule Alignment

This submission is designed around the Robothon requirements: MuJoCo is the primary physics engine, the MJCF scene includes the robot hand, joints, collision geoms, touch sensors, actuators, task objects, and cameras, and the demo video is produced by running the submitted code. The generated `outputs/demo.mp4` is about 145 seconds in the current validation run, inside the 1-3 minute demo window.

The generated `outputs/event_rules_report.json` maps the submission to the event deliverables and scoring rubric: Runability, Depth of MuJoCo Use, Task Design, Control, Dexterous Manipulation, Engineering Quality, Presentation, and Innovation. `outputs/submission_readiness_report.json` adds a submission-facing audit for UUID consistency, PR target, required commands, required outputs, and the same rubric evidence. `outputs/rubric_readiness_report.json` and `dataset/code_quality_report.json` add a local, non-official quality gate that checks compile health, source hygiene, validator status, and rubric coverage before submission.

## Human-Like Skeletal Hand Model

The hand is a custom MuJoCo primitive model with palm, wrist mount, thumb, index, middle, ring, and little fingers. The thumb is mounted on the side of the palm for opposition instead of behaving as a fifth parallel finger.

## Finger Anatomy and Joint Model

Each long finger has MCP abduction/flexion, PIP flexion, DIP flexion, visible phalanx capsules, joint caps, and fingertip pads. The thumb has CMC-like opposition/abduction, MCP flexion, IP flexion, and a fingertip pad. Finger proportions are intentionally varied: middle is longest, ring and index are shorter, little is shortest among long fingers, and thumb is shorter/thicker.

## Object-Specific Human-Like Grasps

The controller uses simulation-native object pose perception and a hybrid contact-aware dexterous manipulation routine. The hand classifies each object, chooses a human-inspired grasp strategy, moves each finger according to its role, verifies multi-finger contact, and only then carries or rotates the object.

## Sphere Enclosure Grasp

The sphere task uses thumb opposition plus index/middle support and ring/little lower support. The sphere is treated as enclosed by a finger cage before it is held.

## Cube Opposing-Face Grasp

The cube task uses opposing-face contact: thumb on one face, index/middle on the opposing face, and ring support below. The controller logs face-centered contact evidence and rejects one-face-only/corner-style contact.

## Cylinder Side-Body Grasp

The cylinder task uses a lateral body wrap, not a top-down grasp. Thumb contacts one side of the cylinder body while index/middle/ring/little support the opposite and lower sides.

## In-Hand Rotation with Finger Gaiting

The cylinder is rotated in-hand by a contact-aware hybrid routine. The index finger acts as a tangential push finger while thumb/middle/ring provide counterhold and support. Rotation is gradual and logged with achieved angle, rotation error, active rotation finger, and support fingers.

## Signature Cap/Knob Rotation

The score-focused evidence upgrade adds `CAP_KNOB_ROTATION_224`. A visible marker on the cap shows a 224-degree twist after five-finger contact verification. The cap task logs target angle, achieved angle, angle error, active/counterhold fingers, cap slip, contact balance, pressure targets, and whether hybrid rotation was used.

## Tactile Combination Lock

The harder task upgrade adds `TACTILE_COMBINATION_LOCK`. The hand probes a dial rim, detects tactile detents, rotates through a three-step code sequence, verifies each click, pinches and pulls a latch, then opens a small micro-door. This task is designed to exercise sequential dexterous control rather than a single grasp: thumb/index/middle perform dial twisting, ring/little stabilize during verification and latch pull, and the controller logs max code error, detent confidence, latch travel, door angle, contact confidence, and no-snap evidence.

Run it directly with `python submissions/dexhand_lab/run_demo.py --episodes 1 --seed 42 --difficulty medium --arena lock --no-video`. Evidence is written to `outputs/combination_lock_summary.json`, `dataset/combination_lock_report.json`, `dataset/combination_lock_trace.csv`, and `media/combination_lock_keyframes.png`.

## Blind Tactile Active Perception

The 95+ differentiator adds an optional blind tactile mode. When `--blind-tactile` is enabled, object labels are hidden from the controller decision path. The hand performs active probing with the index fingertip, thumb, and middle finger, estimates tactile features such as curvature, edge response, flat-face response, long-axis signal, twist affordance, and press displacement, then classifies the unknown object before choosing a grasp strategy.

This is a deterministic tactile classifier and adaptive heuristic controller, not learned RL. Ground-truth labels are used only after classification for evaluation. The main outputs are `outputs/blind_tactile_summary.json`, `dataset/tactile_exploration_trace.csv`, `dataset/tactile_classifier_report.json`, `dataset/tactile_confusion_matrix.json`, `dataset/adaptive_regrasp_report.json`, and `dataset/unknown_arena_report.json`.

The main default video now includes an explicit blind tactile segment before the cap/knob twist. The HUD shows labels hidden, probe count, probing finger, predicted type, confidence, selected grasp strategy, and adaptive regrasp precheck so this update is visible without requiring a separate video.

## Unknown Object Arena

The command `python submissions/dexhand_lab/run_demo.py --episodes 1 --seed 42 --difficulty medium --arena unknown --blind-tactile` runs the unknown-object arena. Object order is randomized by seed, the controller probes each unknown object, predicts the object type, selects the grasp from tactile evidence, and logs whether adaptive regrasp was needed.

## Tactile Pose Estimation and Precision Assembly

The additive 95+ differentiator is `TACTILE_PRECISION_ASSEMBLY`. With `--arena assembly --blind-tactile --no-ground-truth-pose`, the controller hides exact object pose from the decision path, probes a small plug/key with fingertip contacts, estimates center, long axis, orientation, and contact frame from tactile/contact history, then uses that estimated pose for a precision plug/socket insertion task.

The assembly routine uses a tripod/pinch grasp, in-hand orientation correction, socket alignment, compliant insertion, jam detection, withdraw/correct recovery, retry insertion, and final insertion verification. Ground-truth pose is used only after the episode for scoring/audit. Evidence is written to `dataset/tactile_pose_estimator_report.json`, `dataset/tactile_pose_trace.csv`, `dataset/precision_assembly_report.json`, `dataset/precision_assembly_trace.csv`, `dataset/jam_recovery_report.json`, `dataset/no_ground_truth_control_audit.json`, `outputs/assembly_summary.json`, `media/assembly_keyframes.png`, and `media/tactile_pose_estimation_panel.png`.

## Adaptive Regrasp

`adaptive_regrasp_policy.py` records recovery actions for low confidence, contact imbalance, slip, suspected wrong shape, cap torque failure, stylus misalignment, cube one-face contact, cylinder end/top grasp risk, and non-index button contact risk. Actions include adding thumb opposition, ring/little support, recentering, switching to cylinder side-body grasp, increasing pressure target, and controlled release/retry.

## Tactile Evidence

Five fingertip channels are logged: thumb, index, middle, ring, and little. The MJCF scene includes one MuJoCo touch sensor per fingertip pad, and each logged channel includes the touch sensor value, contact active flag, contact object, normal force proxy, shear slip proxy, friction margin, contact confidence, pressure target, fingertip position, and role. The force fields are still simulation/controller evidence, not physical hardware tactile readings.

## Minimum-Jerk Tactile Control

`minimum_jerk_controller.py` generates deterministic tactile-inspired minimum-jerk segments for approach, preshape, contact seek, grasp closure, cap twist, slip recovery, load hold, and release. It writes `dataset/minimum_jerk_report.json` and `dataset/minimum_jerk_trace.csv`.

## No-Snap Verified Grasp Routine

Objects are not attached or moved until the required finger contacts are active and stable verification passes. Hybrid carry/rotation preserves the relative transform from the verified contact moment and moves objects gradually.

## Slip Recovery and Load Hold

The cap task includes a mild slip disturbance proxy, recovery by increasing thumb opposition and support pressure, and a 9x load-hold marker. The demo logs final slip, max slip, recovery action, pressure boost, load hold multiplier, and object drop count.

## Stylus Tripod Grasp

The stylus is held near the handle center using thumb, index, and middle. Ring and little remain curled/clear. The stylus tip is then moved to the checkpoint.

## Index-Only Button Press

The button task uses the index fingertip only. Non-index button contacts and palm contact are logged as failure evidence; the intended default is zero.

## Contact Timeline

`outputs/contact_timeline.json` records phase, target object, grasp type, per-finger contacts, per-finger roles, contact points, active finger count, balance score, multi-side score, tactile proxies, cap angle, slip, load hold, and button state.

## Stress Evaluation

`run_stress_eval.py --seeds 32` runs deterministic perturbations over object pose, mass, friction, lateral shove, cap angle, cap friction, contact loss, and rotation target. It writes `outputs/stress_eval.json`, `dataset/stress_eval.json`, `outputs/baseline_vs_feedback.json`, and `outputs/stress_eval_summary.csv`.

## Judge Evidence Pack

The project includes `JUDGE_BRIEF.md`, `rubric_scorecard.json`, `submission_manifest.json`, `outputs/submission_readiness_report.json`, `outputs/rubric_readiness_report.json`, `dataset/code_quality_report.json`, `dataset/unit_test_report.json`, `media/keyframes.png`, `media/demo.mp4`, `outputs/narration.srt`, tactile reports, task-gate reports, stress reports, and hardware replay audit files.

## Hardware Replay Audit

The hardware audit maps simulated joints to LEAP/Shadow-style channels and generates a bounded 50 Hz command stream. This is a simulation-to-hardware replay safety audit, not a real hardware trial.

## Demo Video

The default demo is intended to be judge-readable and inside the event's 1-3 minute demo window. The current validation render is about 145 seconds. It shows the hand skeleton, sphere enclosure, cube opposing-face grasp, cylinder side-body grasp, cylinder in-hand rotation, blind tactile probing/classification, cap 224-degree twist, slip/load-hold evidence, tactile combination lock manipulation, visible precision assembly insertion, stylus tripod grasp, index-only button press, and final evidence pose.

The latest presentation pass widens the cameras and adds a higher-signal HUD/keyframe/narration sequence so the main video shows the strongest evidence directly: blind tactile probing, contact-verified cap twist, tactile lock detents, plug/socket insertion, jam correction, and final success metrics.

For runability on headless or slow machines, the default command preserves the already generated `outputs/demo.mp4`/`media/demo.mp4` and refreshes the JSON/CSV evidence quickly. To render a replacement video from MuJoCo frames, run `python submissions/dexhand_lab/run_demo.py --force-render-video`.

## How to Run

```bash
python submissions/dexhand_lab/run_demo.py
python submissions/dexhand_lab/run_demo.py --force-render-video
python submissions/dexhand_lab/run_demo.py --episodes 3 --seed 42 --no-video --difficulty medium
python submissions/dexhand_lab/run_stress_eval.py --seeds 32
python submissions/dexhand_lab/arena_task_suite.py
python submissions/dexhand_lab/minimum_jerk_controller.py
python submissions/dexhand_lab/contact_feedback_audit.py
python submissions/dexhand_lab/hardware_adaptation_audit.py
python submissions/dexhand_lab/quality_gate.py --run-tests
python -m unittest discover -s submissions/dexhand_lab/tests -p "test_*.py"
python submissions/dexhand_lab/run_demo.py --episodes 1 --seed 42 --difficulty medium --blind-tactile
python submissions/dexhand_lab/run_demo.py --episodes 1 --seed 42 --difficulty medium --arena unknown --blind-tactile --no-video
python submissions/dexhand_lab/run_demo.py --episodes 1 --seed 42 --difficulty medium --arena assembly --blind-tactile --no-ground-truth-pose
python submissions/dexhand_lab/run_demo.py --episodes 3 --seed 42 --difficulty medium --arena assembly --blind-tactile --no-ground-truth-pose --no-video
python submissions/dexhand_lab/run_demo.py --episodes 1 --seed 42 --difficulty medium --arena lock --no-video
python submissions/dexhand_lab/run_stress_eval.py --seeds 32 --blind-tactile
python submissions/dexhand_lab/run_stress_eval.py --seeds 32 --arena assembly --blind-tactile --no-ground-truth-pose
python submissions/dexhand_lab/run_stress_eval.py --seeds 32 --arena lock
python submissions/dexhand_lab/tactile_active_perception.py
python submissions/dexhand_lab/tactile_pose_estimator.py
python submissions/dexhand_lab/precision_assembly_controller.py
python submissions/dexhand_lab/combination_lock_controller.py
python submissions/dexhand_lab/adaptive_regrasp_policy.py
python submissions/dexhand_lab/validate_submission.py
```

## Outputs

Key outputs include `outputs/demo.mp4`, `media/demo.mp4`, `media/keyframes.png`, `media/blind_tactile_keyframes.png`, `media/tactile_classifier_panel.png`, `media/assembly_keyframes.png`, `media/tactile_pose_estimation_panel.png`, `media/combination_lock_keyframes.png`, `outputs/event_rules_report.json`, `outputs/submission_readiness_report.json`, `outputs/rubric_readiness_report.json`, `outputs/rubric_readiness_scorecard.csv`, `outputs/judge_summary.json`, `outputs/blind_tactile_summary.json`, `outputs/assembly_summary.json`, `outputs/combination_lock_summary.json`, `EVIDENCE_INDEX.md`, `outputs/summary.json`, `outputs/contact_timeline.json`, `outputs/final_report.txt`, `dataset/code_quality_report.json`, `dataset/unit_test_report.json`, `dataset/task_suite_report.json`, `dataset/tactile_feedback_report.json`, `dataset/tactile_classifier_report.json`, `dataset/tactile_confusion_matrix.json`, `dataset/adaptive_regrasp_report.json`, `dataset/unknown_arena_report.json`, `dataset/tactile_pose_estimator_report.json`, `dataset/precision_assembly_report.json`, `dataset/jam_recovery_report.json`, `dataset/combination_lock_report.json`, `dataset/combination_lock_trace.csv`, `dataset/no_ground_truth_control_audit.json`, `dataset/minimum_jerk_report.json`, `dataset/stress_eval.json`, `dataset/blind_tactile_stress_eval.json`, `dataset/assembly_stress_eval.json`, and `dataset/hardware_adaptation_report.json`.

## Limitations

DexHand Lab does not claim real-world deployment, real camera vision, perfect contact physics, learned reinforcement learning, or physical hardware tactile sensors. Hybrid carry and cap rotation are used only after verified contact to keep the benchmark deterministic and reproducible.

## Future Improvements

Future work could replace the remaining pressure proxies with calibrated force/contact sensors, tune contacts with richer dynamics, add learned residual controllers, and replay the bounded command stream on a real dexterous hand platform.
