# DexHand Lab PR #151 Backup State

This file records the active state before the additive tactile precision assembly upgrade.

- Current branch: `dexhand-lab-final-clean-20260620`
- Current PR number: `151`
- UUID: `ae3845b8-7246-4fc9-8655-31d46dbeba99`
- Submission folder: `submissions/dexhand_lab/`
- Update policy: additive only. Existing DexHand Lab features, outputs, validator behavior, stress evaluation, cap rotation, blind tactile classifier, adaptive regrasp, tactile reports, and hardware replay audit are preserved.

## Current Headline Metrics

- Demo duration: about 118.6 seconds
- Object snap events: 0
- Attach-before-verification count: 0
- Cap rotation target / achieved: 224 / 224 degrees
- Task gates: 20 / 20
- Tactile channels: 5
- Final slip: 0.28 mm
- Max slip: 0.46 mm
- Load hold: 9.0x
- Blind tactile classifier accuracy: 1.00
- Adaptive regrasp success rate: 1.00
- Stress rollouts: 32
- Feedback success rate: 100.0%
- Baseline success rate: 59.4%
- Validator: pass

## Required Commands To Preserve

```bash
python submissions/dexhand_lab/run_demo.py
python submissions/dexhand_lab/run_demo.py --episodes 3 --seed 42 --no-video --difficulty medium
python submissions/dexhand_lab/run_stress_eval.py --seeds 32
python submissions/dexhand_lab/validate_submission.py
```

## Additive Upgrade

The next change adds tactile pose estimation and precision assembly as an optional arena:

```bash
python submissions/dexhand_lab/run_demo.py --episodes 1 --seed 42 --difficulty medium --arena assembly --blind-tactile --no-ground-truth-pose
```

Ground truth pose is used only after the episode for scoring/evaluation in the new assembly evidence.
