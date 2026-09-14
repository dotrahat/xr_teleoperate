# Main-behavior head-relative baseline

This branch starts at `main` commit
`3c841ae39f53dae0b943c68012e6417ba3022a5b` and adds measurement-only
head-relative logging. It does not include the G1 head-origin calibration or
retargeting changes from the feature branch.

The instrumentation is deliberately restricted to the configuration used for
the physical comparison: `G1_23`, BrainCo, hand tracking, and `head_yaw` arm
reference. Every run records `comparison_role=main_behavior_baseline`, the
exact main behavior commit, and `head_origin_calibration=legacy_virtual_head`
in `run_meta.json`.

## Capture the baseline

Use the same robot support, operator, headset fit, control frequency, starting
pose, and movement sequence that will be used on the feature branch. On
hardware, keep the robot supported and use motion mode:

```bash
cd teleop
python teleop_hand_and_arm.py \
  --arm=G1_23 \
  --ee=brainco \
  --input-mode=hand \
  --arm-reference-mode=head_yaw \
  --motion \
  --head-relative-monitor \
  --head-relative-monitor-no-viewer
```

The output is written below `teleop/utils/head_relative_logs/` in a directory
whose name begins with `main_baseline_real_`. Copy or retain that entire run
directory before switching branches.

## Repeatable movement sequence

After starting collection, hold the neutral pose for five seconds. Then use
slow, symmetric movements and hold each endpoint for roughly five seconds:

1. Near forward reach.
2. Mid forward reach.
3. Long forward reach within the already tested safe workspace.
4. Left/right lateral sweep at mid reach.
5. Up/down sweep at mid reach.
6. Return to neutral and hold for five seconds.

Repeat the same sequence on the feature branch. Do not add calibration during
the baseline run; this branch intentionally has no calibration option.

## Compare on the feature branch

Run the analyzer from the feature branch and pass both runs explicitly:

```bash
python utils/analyze_head_relative_logs.py \
  /path/to/feature-run \
  --baseline /path/to/main-baseline-run
```

Use the target-matched low-motion result as the primary comparison. Also check
coverage: low overlap means the two movement sequences were not similar enough
for a strong conclusion. The monitor matches wrist translation only; it does
not record or match wrist orientation.
