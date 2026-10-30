# uwslam

Monocular-camera SLAM for a small underwater robot, with an optional IMU (gyro, and an accelerometer that levels the
world poses). A C core (`core/`) runs bit-identically on a PC and an ESP32-P4; Python tools (`src/uwslam/`) bind it
with ctypes and score it against a reconstruction of a lake clip, and the IMU path on AQUALOC's harbour sequences
(camera and IMU on one clock).

## Layout

- `core/`: C99, no heap, no libm. Odometry `mono_vo*.c` (five units, listed in `mono_vo_internal.h`), `voxel_map.c`,
  `obstacle_dir.c`, `path_plan.c`, `mono_map.c`, `mono_cloud.c`; prefixes `mvo_`, `vm_`, `od_`, `pp_`, `mm_`, `mc_`.
- `src/uwslam/`: bindings and lake-clip tools (`mvo`, `voxel_map`, `obstacle_dir`, `path_plan`, `mono_map`,
  `mono_cloud`), AQUALOC's sequences with their IMU (`aqualoc`: prepare, board files; `aqualoc_score`), the board host
  `mvo_stream`, and `experiments/` (OpenCV prototype `mono_vo`, reference poses and masks `mask_motion`, `segmenter`).
- `tests/`: unittest suite; `tests/c/` holds the C contract checks and the firmware host harnesses.
- `firmware/mono_vo_stream/`: the ESP32-P4 firmware; frames and masks arrive over the console UART, or over the chip's
  own USB-Serial-JTAG port when built with `sdkconfig.usb_serial_jtag`.
- `results/lake_clip/`: the committed results, including the board's expected-hash files and its records, and the
  inputs the tools share (`mask_motion`, the OpenCV prototype `mono_vo`). `results/aqualoc/`: the IMU's scores and,
  per sequence, its checks, the board's expectations and records (the data itself is not in the repository).
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
  stream a sequence from `uwslam.aqualoc board` with `--imu`, and with `--gyro`.
- A change to `core/` or its headers changes the source SHA-256 the firmware checks (`uwslam.mvo.CORE_FILES`, guarded
  by `tests/test_mvo_sources.py`): regenerate the expected files in `results/lake_clip/` and reflash. Finish comment
  edits in `core/` before generating results, since a comment changes the hashes too.

## Conventions

- English code, comments and reports; README.md and README.zh-CN.md change together.
- No personal paths or names in the repository: machine paths enter through CLI arguments or environment variables,
  and outputs record file names, not paths.
- Structs whose bytes are hashed have no padding; static asserts check it.
- The board's input rate is 5 Hz; 10 Hz is scored as well.

## Limits

- The board is an ESP32-P4 dev board (two RISC-V cores at 360 MHz, no PSRAM used). Its firmware tracks with
  `mvo_vector.S` (the 19×19 window kernels; the block copy stays C).
- At real 5 Hz input a step must stay under 200 ms on the board; a lake step takes about 127 ms (at most 154), with
  the gyro or the IMU on AQUALOC at most 183 ms; about 97 KB of internal RAM stay free.

Paths on this machine are in `CLAUDE.local.md`. Repeated procedures are skills in `.claude/skills/`.
