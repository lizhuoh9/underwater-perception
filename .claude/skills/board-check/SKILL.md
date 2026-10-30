---
name: board-check
description: Build and flash the uwslam firmware to the ESP32-P4 and check that every step, map, voxel, direction and plan hash equals the PC's from the same frames and masks, with per-step timing and free RAM. Use after any change to core/, its headers or the firmware, or when asked to run, test or time the board.
---

# Board check

Shows that the board computes exactly what the PC computes from the same frames and masks, and how long it takes.

Arguments: $ARGUMENTS (a label for the record, e.g. `p4_360mhz_usj_5hz`; the rate, default 5). Paths, the port and
both Pythons come from `CLAUDE.local.md`. Run from the repository root; `<out>` is `local/board/<label>`. The board is
the ESP32-P4.

## Steps

1. Tests: `python -m unittest discover -s tests -t .` → `OK`.
2. PC expectations for the tree being flashed, with the project Python and `PYTHONPATH=src`:
   - `python -m uwslam.mvo compare <video> --reconstruction <folder> --output <out>/mono_vo_c --board-frames <out>/frames_10hz.bin`
   - `python -m uwslam.mono_map <video> --reconstruction <folder> --output <out>/mono_map --board-masks <out>/masks_10hz.bin`
   - `python -m uwslam.voxel_map`, `obstacle_dir` and `path_plan`, each with `--output <out>/<tool>`
   → `board_expected_*_<rate>hz.json` in each folder, carrying the tree's source SHA-256s and `"config": "p4"`. Run at
   most three tools at a time: each loads the clip and the reference cloud.
3. Check the port's chip first: `python -m esptool --port <port> chip_id` must report an ESP32-P4 (and the MAC
   `CLAUDE.local.md` gives for the dev board). Then build and flash, in a shell where ESP-IDF is activated as
   `CLAUDE.local.md` says:
   `idf.py -C firmware/mono_vo_stream -B firmware/mono_vo_stream/build -D SDKCONFIG=build/sdkconfig -p <port> flash`
   → the flash is verified. That build links over the console UART (a USB-UART bridge). For a board connected through
   the P4's own USB-Serial-JTAG port, build the console there instead, in a build folder of its own:
   `idf.py -C firmware/mono_vo_stream -B firmware/mono_vo_stream/build_usj -D SDKCONFIG=build_usj/sdkconfig -D "SDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.usb_serial_jtag" -p <port> flash`.
4. Stream, with the ESP-IDF Python and `PYTHONPATH=src`:
   `python -m uwslam.mvo_stream --port <port> --frames <out>/frames_10hz.bin --masks <out>/masks_10hz.bin --expected <out>/mono_vo_c/board_expected_<rate>hz.json --expected-map <out>/mono_map/board_expected_map_<rate>hz.json --expected-voxels <out>/voxel_map/board_expected_voxels_<rate>hz.json --expected-dirs <out>/obstacle_dir/board_expected_dirs_<rate>hz.json --expected-plan <out>/path_plan/board_expected_plan_<rate>hz.json --label <label>`
   → `<out>/mono_vo_c/board/<label>.json` and `<label>_serial.log`. The firmware refuses expectations whose source
   SHA-256 differs from its own build. Over the UART the 5 Hz clip takes about 3 min, over USB-Serial-JTAG about 40 s.
5. The IMU, for any change to it (or to the odometry): for each prepared AQUALOC sequence (`CLAUDE.local.md`),
   `python -m uwslam.aqualoc board <prepared> --output <out>/aqualoc_<seq> --calibration <the other prepared>`, then
   stream it with `--frames <out>/aqualoc_<seq>/frames.bin --imu <out>/aqualoc_<seq>/imu.bin --expected <out>/aqualoc_<seq>/board_expected_imu.json`,
   with `--gyro <out>/aqualoc_<seq>/gyro.bin --expected <out>/aqualoc_<seq>/board_expected_gyro.json`, and once with
   `board_expected_camera.json` alone, for the camera alone's time. For the metric scale (any change to it or to the
   odometry), also with `--imu <out>/aqualoc_<seq>/imu.bin --sense <out>/aqualoc_<seq>/sense.bin --expected
   <out>/aqualoc_<seq>/board_expected_depth.json` (from a sequence prepared with its depths). Sequence 07 takes about
   2 min over USB-Serial-JTAG, sequence 01 about 4. The lake's simulated ranger: with the project Python,
   `python -m uwslam.ranger <video> --reconstruction <folder> --output <out>/ranger --board-frames <out>/frames_5hz.bin`
   (about 3 min), then stream `--frames <out>/frames_5hz.bin --sense <out>/ranger/sense_5hz.bin --expected
   <out>/ranger/board_expected_ranger_5hz.json`.
6. The live build, for any change to `core/frame_prep.*`, the firmware's live path or the odometry. Build and flash it
   in a build folder of its own, after the chip check:
   `idf.py -C firmware/mono_vo_stream -B firmware/mono_vo_stream/build_live -D SDKCONFIG=build_live/sdkconfig -D "SDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.usb_serial_jtag;sdkconfig.live" -p <port> flash`
   (a changed `sdkconfig.live` only reaches a build whose `build_live/sdkconfig` is deleted first). With the project
   Python, `python -m uwslam.live_feed prepare <video> --reconstruction <folder> --output <out>/live`; with the ESP-IDF
   Python, `python -m uwslam.live_feed stream <out>/live --port <port> --label check --check` (about 1 min), then the
   same with `--label live` and no `--check` (real time, 35 s); with the project Python,
   `python -m uwslam.live_feed check <out>/live --label check --reconstruction <folder> --output <out>/live_check`
   (about 5 min) → `<out>/live/board/{check,live}.json` and `<out>/live_check/summary.json`.

## Done when

- `mvo_stream` reports every step, map, voxel, direction and plan equal to the PC's (a partial run whose steps differ
  fails), with the gyro every step's use of it and the final bias estimate too, with the IMU also the world's up, and
  with `--sense` every step's metric centre and scale.
- At 5 Hz the slowest step is under 200 ms, with the gyro and the IMU as well; internal RAM free at the end is in the
  record.
- The live build: `steps_equal` equals `steps` in the check's summary.json, `decode_difference.max_abs` is at most 1,
  and the real-time record's frames stepped, dropped and work per frame are in its `results` and `timing`.

## Report

- Equal or not, per check; ms per step (mean / median / max) and the stage times; RAM free; the record's path.
- For a change the user accepts: regenerate `results/lake_clip/` and its board records (the stream writes the
  RESULTS.md boards section and notes which link `*_usj*` records used), and update the ESP32-P4 section of README.md
  and README.zh-CN.md together.
- Update `.claude/progress.md`.
