---
name: board-check
description: Build and flash the uwslam firmware to the ESP32-S3 and check that every step, map, voxel, direction and plan hash equals the PC's from the same frames and masks, with per-step timing and free RAM. Use after any change to core/, its headers or the firmware, or when asked to run, test or time the board.
argument-hint: "<label> [rate 5|10, default 5]"
---

# Board check

Shows that the board computes exactly what the PC computes from the same frames and masks, and how long it takes.

Arguments: $ARGUMENTS (a label for the record, e.g. `240mhz_5hz`; the rate, default 5). Paths, the port and both
Pythons come from `CLAUDE.local.md`. Run from the repository root; `<out>` is `local/board/<label>`.

## Steps

1. Tests: `python -m unittest discover -s tests -t .` → `OK`, and again with `UWSLAM_CONFIG=p4` → `OK`.
2. PC expectations for the tree being flashed, with the project Python and `PYTHONPATH=src`:
   - `python -m uwslam.mvo compare <video> --reconstruction <folder> --output <out>/mono_vo_c --board-frames <out>/frames_10hz.bin`
   - `python -m uwslam.mono_map <video> --reconstruction <folder> --output <out>/mono_map --board-masks <out>/masks_10hz.bin`
   - `python -m uwslam.voxel_map`, `obstacle_dir` and `path_plan`, each with `--output <out>/<tool>`
   → `board_expected_*_<rate>hz.json` in each folder, carrying the tree's `core_sha256`.
3. Build and flash, in a shell where ESP-IDF is activated as `CLAUDE.local.md` says:
   `idf.py -C firmware/esp32s3_mono_vo -B firmware/esp32s3_mono_vo/build_240 -p <port> flash` → the flash is verified.
   If the change adds static data, check `idf.py -C firmware/esp32s3_mono_vo -B firmware/esp32s3_mono_vo/build_240 size`
   first: libmain `.bss` must stay near 94 KB or the pyramids fail to allocate.
   For the ESP32-P4: `idf.py -C firmware/esp32s3_mono_vo -B firmware/esp32s3_mono_vo/build_p4 -D SDKCONFIG=build_p4/sdkconfig
   -D IDF_TARGET=esp32p4 -p <p4 port> flash`, labels `p4_360mhz[_5hz]`; its UART link is slower (5 Hz about 3 min).
   The P4 firmware is the `p4` build configuration: make its step-2 expectations with `UWSLAM_CONFIG=p4`, and its
   accepted results go to `results/lake_clip_p4/`.
4. Stream, with the ESP-IDF Python and `PYTHONPATH=src`:
   `python -m uwslam.mvo_stream --port <port> --frames <out>/frames_10hz.bin --masks <out>/masks_10hz.bin --expected <out>/mono_vo_c/board_expected_<rate>hz.json --expected-map <out>/mono_map/board_expected_map_<rate>hz.json --expected-voxels <out>/voxel_map/board_expected_voxels_<rate>hz.json --expected-dirs <out>/obstacle_dir/board_expected_dirs_<rate>hz.json --expected-plan <out>/path_plan/board_expected_plan_<rate>hz.json --label <label>`
   → `<out>/mono_vo_c/board/<label>.json` and `<label>_serial.log`. The firmware refuses expectations whose source
   SHA-256 differs from its own build.

## Done when

- `mvo_stream` reports every step, map, voxel, direction and plan equal to the PC's (a partial run whose steps differ
  fails).
- At 5 Hz the slowest step is under 200 ms; internal RAM free at the end is in the record.

## Report

- Equal or not, per check; ms per step (mean / median / max) and the stage times; RAM free; the record's path.
- For a change the user accepts: regenerate `results/lake_clip/` and its board records, and update the ESP32-S3 table
  in README.md and README.zh-CN.md together.
- Update `.claude/progress.md`.
