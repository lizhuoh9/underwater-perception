---
name: board-check
description: Build and flash the uwslam firmware to the ESP32-P4 and check that every step, map, voxel, direction and plan hash equals the PC's from the same frames and masks, with per-step timing and free RAM. Use after any change to core/, its headers or the firmware, or when asked to run, test or time the board.
---

# Board check

Shows that the board computes exactly what the PC computes from the same frames and masks, and how long it takes.

Arguments: $ARGUMENTS (a label for the record, e.g. `p4_360mhz_usj_5hz`; the rate, default 5). Paths, the port and
both Pythons come from `CLAUDE.local.md`. Run from the repository root with `UWSLAM_CONFIG=p4`; `<out>` is
`local/board/<label>`. The board is the ESP32-P4 (the S3 is no longer used).

## Steps

1. Tests: `python -m unittest discover -s tests -t .` with `UWSLAM_CONFIG=p4` → `OK`.
2. PC expectations for the tree being flashed, with the project Python, `PYTHONPATH=src` and `UWSLAM_CONFIG=p4`:
   - `python -m uwslam.mvo compare <video> --reconstruction <folder> --output <out>/mono_vo_c --board-frames <out>/frames_10hz.bin`
   - `python -m uwslam.mono_map <video> --reconstruction <folder> --output <out>/mono_map --board-masks <out>/masks_10hz.bin`
   - `python -m uwslam.voxel_map`, `obstacle_dir` and `path_plan`, each with `--output <out>/<tool>`
   → `board_expected_*_<rate>hz.json` in each folder, carrying the tree's source SHA-256s and `"config": "p4"`. Run at
   most three tools at a time: each loads the clip and the reference cloud.
3. Check the port's chip first: `python -m esptool --port <port> chip_id` must report an ESP32-P4 (and the MAC
   `CLAUDE.local.md` gives for the dev board). Then build and flash, in a shell where ESP-IDF is activated as
   `CLAUDE.local.md` says:
   `idf.py -C firmware/esp32s3_mono_vo -B firmware/esp32s3_mono_vo/build_p4 -D SDKCONFIG=build_p4/sdkconfig -D IDF_TARGET=esp32p4 -p <port> flash`
   → the flash is verified. That build links over the console UART (a USB-UART bridge). For a board connected through
   the P4's own USB-Serial-JTAG port, build the console there instead: a build folder of its own, with
   `-D "SDKCONFIG_DEFAULTS=sdkconfig.defaults;<file holding CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y>"` (`CLAUDE.local.md`
   names the file and folder on this machine).
4. Stream, with the ESP-IDF Python and `PYTHONPATH=src`:
   `python -m uwslam.mvo_stream --port <port> --frames <out>/frames_10hz.bin --masks <out>/masks_10hz.bin --expected <out>/mono_vo_c/board_expected_<rate>hz.json --expected-map <out>/mono_map/board_expected_map_<rate>hz.json --expected-voxels <out>/voxel_map/board_expected_voxels_<rate>hz.json --expected-dirs <out>/obstacle_dir/board_expected_dirs_<rate>hz.json --expected-plan <out>/path_plan/board_expected_plan_<rate>hz.json --label <label>`
   → `<out>/mono_vo_c/board/<label>.json` and `<label>_serial.log`. The firmware refuses expectations whose source
   SHA-256 differs from its own build. Over the UART the 5 Hz clip takes about 3 min, over USB-Serial-JTAG about 40 s.

## Done when

- `mvo_stream` reports every step, map, voxel, direction and plan equal to the PC's (a partial run whose steps differ
  fails).
- At 5 Hz the slowest step is under 200 ms; internal RAM free at the end is in the record.

## Report

- Equal or not, per check; ms per step (mean / median / max) and the stage times; RAM free; the record's path.
- For a change the user accepts: regenerate `results/lake_clip_p4/` and its board records (the stream writes the
  RESULTS.md boards section; say in a note there which link the records used), and update the ESP32-P4 section of
  README.md and README.zh-CN.md together.
- Update `.claude/progress.md`.
