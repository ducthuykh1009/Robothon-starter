# DexHand Lab 87.4 Score State Backup

Current leaderboard score: 87.4/100

Branch at backup time: dexhand-lab-final-90-20260619

Latest known commit at backup time: 4ed00cc Upgrade DexHand Lab tactile evidence pack

Preserved baseline evidence:
- Human-like 5-finger hand with thumb opposition
- Object-specific grasps for sphere, cube, cylinder, stylus, button, and cap/knob
- 224-degree cap/knob rotation
- Five tactile channels and five MuJoCo fingertip touch sensors
- Contact timeline and tactile taxel CSV
- 9x load hold
- Slip recovery
- Minimum-jerk tactile controller
- 32-seed stress evaluation
- Hardware replay audit
- Judge evidence pack and validator

Existing headline metrics from the preserved run:
- Demo duration: 101.6 seconds
- Task gates: 20/20
- Cap rotation: 224 deg target / 224 deg achieved
- Final slip: 0.28 mm
- Load hold: 9.0x
- Object snap events: 0
- Baseline vs feedback: 0.59375 vs 1.0

Validation commands to preserve:
```bash
python submissions/dexhand_lab/run_demo.py --episodes 1 --seed 42 --debug-grasp --difficulty medium
python submissions/dexhand_lab/run_demo.py --episodes 3 --seed 42 --no-video --difficulty medium
python submissions/dexhand_lab/run_stress_eval.py --seeds 32
python submissions/dexhand_lab/validate_submission.py
```

Additive upgrade rule:
Blind tactile active perception must be optional and controlled by `--blind-tactile` and `--arena unknown`. The default commands should keep the existing DexHand pipeline behavior.
