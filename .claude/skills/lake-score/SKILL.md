---
name: lake-score
description: Score a change to the uwslam odometry or maps on the lake clip against a control run of the last commit - full clip and 10 s windows, 16 seeds, paired run for run. Use before accepting any change to core/ or the scoring tools, or when asked whether a change is more accurate.
argument-hint: "<label> [rates, default 5,10]"
---

# Lake score

Shows whether the working tree is more accurate than the last commit on the lake clip, under identical settings.

Arguments: $ARGUMENTS (a label for the output folder; the rates, default `5,10`). The Python, video and
reconstruction come from `CLAUDE.local.md`. Run from the repository root; `<out>` is `local/lake-score/<label>`. Every
run builds the core in the configuration `UWSLAM_CONFIG` names (`s3` by default, or `p4`); set it the same for both trees
unless the configuration is what is being compared (the scorings record it and `compare` prints it).

## Steps

1. Control: `git status --short` shows only the change being scored; then
   `git worktree add <out>/control HEAD` → a copy of the last commit.
2. Windows, both trees (they can run at the same time):
   - `PYTHONPATH=<out>/control/src <python> .claude/skills/lake-score/lake_score.py windows <video> --reconstruction <folder> --output <out>/windows_control.json --rates <rates>`
   - the same with `PYTHONPATH=src` and `--output <out>/windows_candidate.json`
   → per run (12 windows × 16 seeds × each rate): ATE after similarity alignment, heading RMSE and maximum, lost
   steps; the medians are printed.
3. Full clip, both trees: `PYTHONPATH=<tree>/src <python> -m uwslam.mvo compare <video> --reconstruction <folder> --output <out>/mvo_<control|candidate>`
   → `pc_summary.json` with 16-seed medians and ranges at 10, 5 and 3 Hz.
4. Compare:
   - `<python> .claude/skills/lake-score/lake_score.py compare <out>/windows_control.json <out>/windows_candidate.json`
     → paired medians and how often the candidate is lower (it refuses scorings made with different settings);
   - `<python> .claude/skills/lake-score/lake_score.py full <out>/mvo_control/pc_summary.json <out>/mvo_candidate/pc_summary.json`.
5. If the change touches the maps, directions or plan, also run `uwslam.voxel_map`, `obstacle_dir`, `path_plan` and
   `mono_map` for both trees into `<out>` and compare their `summary.json` scores.
6. `git worktree remove <out>/control`.

## Done when

- Both trees were scored with the same settings, and the control's `core_sha256` is the last commit's.
- For each rate there is a control median, a candidate median and the candidate's paired win rate for every metric.

## Report

- Per rate: metric, control median, candidate median, candidate lower in % of paired runs; the full-clip medians and
  ranges.
- Say plainly when the result is mixed or within the seed spread. A lower median with a win rate near 50% is not an
  improvement.
- Results stay in `<out>`. Regenerate `results/lake_clip/` only for a change the user accepts, then run `board-check`.
- Update `.claude/progress.md`.
