<!-- Robothon 2026 submission -->

## Registration UUID

Registration UUID: 2555924c-74a4-4788-be61-1f1e65bf3f44

The same UUID should be stored in:

submissions/dexhand_lab/registration.json

## Project Summary

Project name: DexHand Lab — Blind Tactile Active Perception and 5-Finger Manipulation

Robothon platform: MuJoCo dexterous hand simulation

Task goal: demonstrate a hand-only human-like 5-finger robot that performs object-specific grasping, in-hand manipulation, tactile evidence logging, blind tactile active perception, no-ground-truth tactile pose estimation, precision plug/socket assembly, adaptive regrasp, cap rotation, slip recovery, stylus interaction, and index-only button pressing.

DexHand Lab is a custom MuJoCo benchmark centered on a skeletal 5-finger hand rather than a robot arm. The hand has visible thumb opposition, MCP/PIP/DIP-style finger joints, fingertip pads, tactile/contact evidence streams, and object-specific grasp primitives for sphere, cube, cylinder, cap/knob, stylus, button, and assembly tasks.

## Event Rule Alignment

- MuJoCo is the primary physics engine.
- The submission includes runnable code, MJCF scene/model, actuators, joints, collisions, touch sensors, task objects, outputs, README, judge brief, manifest, validator, and generated demo video.
- The demo video is included at `submissions/dexhand_lab/outputs/demo.mp4`.
- The current demo duration is about 118.6 seconds, inside the event 1–3 minute target.
- The project is reproducible from fixed seeds.
- `outputs/event_rules_report.json` maps the submission to the event rubric: Runability, Depth of MuJoCo Use, Task Design, Control, Dexterous Manipulation, Engineering Quality, Presentation, and Innovation.

## What This Submission Demonstrates

- Human-like 5-finger skeletal hand with thumb opposition.
- Independent finger timing and role-specific motion.
- Sphere enclosure grasp with multi-side contact.
- Cube opposing-face grasp.
- Cylinder side-body grasp and in-hand rotation.
- 224-degree cap/knob rotation with visible marker.
- Five fingertip tactile channels.
- Minimum-jerk tactile-inspired control.
- Slip detection and recovery.
- 9x load-hold evidence.
- Blind tactile active perception.
- Tactile shape classification without using object labels in blind mode.
- Adaptive regrasp policy.
- No-ground-truth tactile pose estimation.
- Precision plug/socket assembly with jam detection and recovery.
- Stylus tripod grasp and checkpoint touch.
- Index-only button press.
- Stress evaluation and baseline-vs-feedback comparison.
- Hardware replay/sim-to-real safety audit.

## How to Run

```bash
python submissions/dexhand_lab/run_demo.py
python submissions/dexhand_lab/run_demo.py --episodes 3 --seed 42 --no-video --difficulty medium
python submissions/dexhand_lab/run_stress_eval.py --seeds 32
python submissions/dexhand_lab/validate_submission.py
```

Additional evidence commands:

```bash
python submissions/dexhand_lab/run_demo.py --episodes 1 --seed 42 --difficulty medium --blind-tactile
python submissions/dexhand_lab/run_demo.py --episodes 1 --seed 42 --difficulty medium --arena unknown --blind-tactile --no-video
python submissions/dexhand_lab/run_demo.py --episodes 1 --seed 42 --difficulty medium --arena assembly --blind-tactile --no-ground-truth-pose
python submissions/dexhand_lab/run_stress_eval.py --seeds 32 --arena assembly --blind-tactile --no-ground-truth-pose
```

## Demo Video

- Demo video path: `submissions/dexhand_lab/outputs/demo.mp4`
- Mirrored copy: `submissions/dexhand_lab/media/demo.mp4`
- Duration: about 118.6 seconds
- The video shows the five-finger hand, tactile probing, object-specific grasps, cap rotation, slip/load-hold evidence, stylus interaction, button press, and final evidence frame.

## Key Evidence Files

- `submissions/dexhand_lab/README.md`
- `submissions/dexhand_lab/JUDGE_BRIEF.md`
- `submissions/dexhand_lab/rubric_scorecard.json`
- `submissions/dexhand_lab/submission_manifest.json`
- `submissions/dexhand_lab/outputs/summary.json`
- `submissions/dexhand_lab/outputs/final_report.txt`
- `submissions/dexhand_lab/outputs/contact_timeline.json`
- `submissions/dexhand_lab/outputs/event_rules_report.json`
- `submissions/dexhand_lab/dataset/task_suite_report.json`
- `submissions/dexhand_lab/dataset/tactile_feedback_report.json`
- `submissions/dexhand_lab/dataset/tactile_classifier_report.json`
- `submissions/dexhand_lab/dataset/tactile_pose_estimator_report.json`
- `submissions/dexhand_lab/dataset/precision_assembly_report.json`
- `submissions/dexhand_lab/dataset/jam_recovery_report.json`
- `submissions/dexhand_lab/dataset/stress_eval.json`
- `submissions/dexhand_lab/dataset/assembly_stress_eval.json`
- `submissions/dexhand_lab/media/keyframes.png`
- `submissions/dexhand_lab/media/assembly_keyframes.png`
- `submissions/dexhand_lab/media/tactile_pose_estimation_panel.png`

## Current Validation Snapshot

- Validator: pass
- Demo duration: 118.6 seconds
- Object snap events: 0
- Attach-before-verification count: 0
- Task gates: 20/20
- Cap rotation target/achieved: 224/224 degrees
- Load hold: 9.0x
- Tactile channels: 5
- Blind tactile classifier accuracy: 1.00
- Adaptive regrasp success rate: 1.00
- Assembly success rate: 96.9%
- Jam recovery success rate: 92.3%

## Honest Limitations

This is a deterministic/contact-aware MuJoCo benchmark, not a learned RL policy. It uses simulation-native object pose/contact information and tactile/contact proxies, not real camera vision or physical tactile hardware. The hardware audit is a replay/safety validation artifact, not a real hardware trial. Hybrid carry/rotation is used only after verified multi-finger contact to keep the simulation reproducible.

## Checklist

- [x] `submissions/dexhand_lab/registration.json` contains my UUID.
- [x] PR description contains the same UUID.
- [x] Code runs from documented commands.
- [x] Demo video was generated by submitted code.
- [x] README and judge evidence pack are included.
- [x] Validator passes on generated evidence.
