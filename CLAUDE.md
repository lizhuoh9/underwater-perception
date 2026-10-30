# uwslam

Monocular-camera SLAM for a small underwater robot, with an optional IMU (gyro, and an accelerometer that levels the
world poses). A C core (`core/`) runs bit-identically on a PC and an ESP32-P4; Python tools (`src/uwslam/`) bind it
with ctypes and score it against a reconstruction of a lake clip, and the IMU path on AQUALOC's harbour sequences
(camera and IMU on one clock). The board's live build takes a camera's JPEG frames and prepares them itself.

## Layout

- `core/`: C99, no heap, no libm. Odometry `mono_vo*.c` (five units, listed in `mono_vo_internal.h`), `voxel_map.c`,
  `obstacle_dir.c`, `path_plan.c`, `mono_map.c`, `mono_cloud.c`, `frame_prep.c` (a camera's image averaged onto the
  320×240 frame); prefixes `mvo_`, `vm_`, `od_`, `pp_`, `mm_`, `mc_`, `fp_`.
- `src/uwslam/`: bindings and lake-clip tools (`mvo`, `voxel_map`, `obstacle_dir`, `path_plan`, `mono_map`,
  `mono_cloud`), AQUALOC's sequences with their IMU (`aqualoc`: prepare, board files; `aqualoc_score`), the board host
  `mvo_stream`, the live build's camera `live_feed`, and `experiments/` (OpenCV prototype `mono_vo`, reference poses
  and masks `mask_motion`, `segmenter`).
- `tests/`: unittest suite; `tests/c/` holds the C contract checks and the firmware host harnesses.
- `firmware/mono_vo_stream/`: the ESP32-P4 firmware; frames and masks arrive over the console UART, or over the chip's
  own USB-Serial-JTAG port when built with `sdkconfig.usb_serial_jtag`. With `sdkconfig.live` on top, the live build
  (`live.c`): PSRAM on at 200 MHz, a camera's JPEG frames (`jpeg`) checked (`jpeg_check.c`), decoded by the JPEG engine
  and averaged on the board while a worker task steps the newest frame.
- `results/lake_clip/`: the committed results, including the board's expected-hash files and its records, and the
  inputs the tools share (`mask_motion`, the OpenCV prototype `mono_vo`); `live/` holds the live build's records.
  `results/aqualoc/`: the IMU's scores and, per sequence, its checks, the board's expectations and records (the data
  itself is not in the repository).
- `local/`: experiment folders, not in git.

## Commands

From the repository root, with the Python and gcc that `CLAUDE.local.md` names:

```bash
python -m unittest discover -s tests -t .    # about 3 min; builds the C core with gcc
python -m uwslam.mvo compare <video> --reconstruction <folder> --output <fresh folder> [--board-frames <file>]
python -m uwslam.mono_map <video> --reconstruction <folder> --output <fresh folder> [--board-masks <file>]
python -m uwslam.voxel_map <video> --reconstruction <folder> --output <fresh folder>    # obstacle_dir, path_plan alike
python -m uwslam.aqualoc prepare <raw_data> --trajectory <txt> --camera-calib <yaml> --imu-calib <yaml> --output <fresh>
python -m uwslam.aqualoc_score <prepared seq01> <prepared seq07> --output <fresh folder> --jobs 8
python -m uwslam.aqualoc board <prepared> --output <fresh folder> --calibration <other prepared>    # for mvo_stream
python -m uwslam.live_feed prepare <video> --reconstruction <folder> --output <fresh folder>    # the live build's camera
python -m uwslam.live_feed stream <prepared> --port <port> --label <name> [--check]    # ESP-IDF's Python (pyserial)
python -m uwslam.live_feed check <prepared> --label <check run> --reconstruction <folder> --output <fresh folder>
ruff format src tests .claude/skills && ruff check src tests .claude/skills    # the configuration is in pyproject.toml
```

Every tool refuses to overwrite its output: give it a fresh folder (scratch runs go under `local/`). The core is built
one way, for the ESP32-P4 (`core/mono_vo.h`: 300 features in 19×19 windows; a voxel map of 4096 voxels that the plan
reads two to a side); the board names it `p4` and the host refuses expectations made from another build.

## Verifying a change

- The tests pass.
- Accuracy: the `lake-score` skill. Score against a control run of the last commit with identical settings, over the
  16 seeds in `uwslam.mvo.SEEDS`, on the 10 s clip windows as well as the full clip. One run or one seed proves
  nothing: the seed alone can change a score twofold.
- The IMU path: `uwslam.aqualoc_score` on both prepared sequences (10 s windows and whole sequences, 16 seeds; gyro
  runs paired with the camera alone, IMU runs with the gyro alone; the accelerometer calibrated with the other
  sequence's fitted bias); without readings a step must stay bit-identical (the lake's step hashes), and gyro-only
  readings must give the gyro's steps bit for bit.
- Board: the `board-check` skill. Every step, map, voxel, direction and plan hash must equal the PC's; for the IMU,
  stream a sequence from `uwslam.aqualoc board` with `--imu`, and with `--gyro`. For the live build (`frame_prep`,
  the firmware's live path): a `live_feed stream --check` run whose steps `live_feed check` finds all equal to the PC's
  on the board's own frames, those frames within one grey level of the PC's decoding, and a real-time run's drops.
- A change to `core/` or its headers changes the source SHA-256 the firmware checks (`uwslam.mvo.CORE_FILES` and
  `uwslam.mvo.prep_sha256`, guarded by `tests/test_mvo_sources.py`): regenerate the expected files in
  `results/lake_clip/` and reflash. Finish comment edits in `core/` before generating results, since a comment changes
  the hashes too.

## Conventions

- English code, comments and reports; README.md and README.zh-CN.md change together.
- No personal paths or names in the repository: machine paths enter through CLI arguments or environment variables,
  and outputs record file names, not paths.
- Structs whose bytes are hashed have no padding; static asserts check it.
- The board's input rate is 5 Hz; 10 Hz is scored as well.

## Limits

- The board is an ESP32-P4 dev board (two RISC-V cores at 360 MHz; PSRAM, at 200 MHz, only in the live build). Its
  firmware tracks with `mvo_vector.S` (the 19×19 window kernels; the block copy stays C).
- At real 5 Hz input a step must stay under 200 ms on the board; a lake step takes about 127 ms (at most 154), with
  the gyro or the IMU on AQUALOC at most 183 ms; about 97 KB of internal RAM stay free.
- The live build decodes a 960×544 JPEG in 6.1 ms and averages it in about 22 ms; a frame's whole work takes 164 ms
  in the median when frames wait for the step, 198 ms at real 5 Hz, where receiving the next frame over USB slows the
  step (two frames of 170 dropped); 66,452 B of internal RAM stay free. Frames are checked as they arrive
  (`jpeg_check.c`): the driver's own parser has no bounds.

Paths on this machine are in `CLAUDE.local.md`. Repeated procedures are skills in `.claude/skills/`.
