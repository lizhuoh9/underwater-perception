# uwslam

Monocular-camera SLAM for a small underwater robot. A C core (`core/`) runs bit-identically on a PC and an ESP32-P4
(earlier an ESP32-S3); Python tools (`src/uwslam/`) bind it with ctypes and score it against a reconstruction of a
lake clip.

## Layout

- `core/`: C99, no heap, no libm. Odometry `mono_vo*.c` (five units, listed in `mono_vo_internal.h`), `voxel_map.c`,
  `obstacle_dir.c`, `path_plan.c`, `mono_map.c`, `mono_cloud.c`; prefixes `mvo_`, `vm_`, `od_`, `pp_`, `mm_`, `mc_`.
- `src/uwslam/`: bindings and lake-clip tools (`mvo`, `voxel_map`, `obstacle_dir`, `path_plan`, `mono_map`,
  `mono_cloud`), the board host `mvo_stream`, and `experiments/` (OpenCV prototype `mono_vo`, reference poses and
  masks `mask_motion`, `segmenter`).
- `tests/`: unittest suite; `tests/c/` holds the C contract checks and the firmware host harnesses.
- `firmware/esp32s3_mono_vo/`: the only firmware, for the ESP32-P4 (and the ESP32-S3, its default target, no longer
  used; `sdkconfig.defaults.<target>`); on the P4 frames and masks arrive over the console UART, or over its own
  USB-Serial-JTAG port when the console is built there.
- `results/lake_clip_p4/`: the P4's committed results, including the board's expected-hash files and its records.
- `results/lake_clip/`: the inputs both configurations share (`mask_motion`, the OpenCV prototype `mono_vo`) and the
  frozen s3 set.
- `local/`: experiment folders, not in git.

## Commands

From the repository root, with the Python and gcc that `CLAUDE.local.md` names, and `UWSLAM_CONFIG=p4`:

```bash
python -m unittest discover -s tests -t .    # about 3 min; builds the C core with gcc
python -m uwslam.mvo compare <video> --reconstruction <folder> --output <fresh folder> [--board-frames <file>]
python -m uwslam.mono_map <video> --reconstruction <folder> --output <fresh folder> [--board-masks <file>]
python -m uwslam.voxel_map <video> --reconstruction <folder> --output <fresh folder>    # obstacle_dir, path_plan alike
```

Every tool refuses to overwrite its output: give it a fresh folder (scratch runs go under `local/`). The core has two
build configurations (`core/mono_vo.h`): `p4` (`MVO_CONFIG_P4`: 300 features in 19×19 windows, a finer voxel map of
4096 voxels that the plan reads two to a side), the one in use, and `s3` (250 features in 15×15 windows), the code's
default, no longer used. Tools and tests build the one `UWSLAM_CONFIG` names.

## Verifying a change

- The tests pass with `UWSLAM_CONFIG=p4`. The s3 configuration is no longer tested, scored, regenerated or
  board-checked (the user, 2026-10-29: only the P4).
- Accuracy: the `lake-score` skill. Score against a control run of the last commit with identical settings, over the
  16 seeds in `uwslam.mvo.SEEDS`, on the 10 s clip windows as well as the full clip. One run or one seed proves
  nothing: the seed alone can change a score twofold.
- Board: the `board-check` skill. Every step, map, voxel, direction and plan hash must equal the PC's.
- A change to `core/` or its headers changes the source SHA-256 the firmware checks (`uwslam.mvo.CORE_FILES`, guarded
  by `tests/test_mvo_sources.py`): regenerate the expected files in `results/lake_clip_p4/` and reflash. Finish comment
  edits in `core/` before generating results, since a comment changes the hashes too.

## Conventions

- English code, comments and reports; README.md and README.zh-CN.md change together.
- No personal paths or names in the repository: machine paths enter through CLI arguments or environment variables,
  and outputs record file names, not paths.
- Structs whose bytes are hashed have no padding; static asserts check it.
- The board's input rate is 5 Hz; 10 Hz is scored as well.

## Limits

- The board is an ESP32-P4 dev board (two RISC-V cores at 360 MHz, no PSRAM used). Its firmware builds the `p4`
  configuration and tracks with `mvo_pie_p4.S` (the window kernels for both window sizes, chosen by `MVO_CONFIG_P4`;
  the block copy stays C). Its build folder has its own sdkconfig (`-D SDKCONFIG=build_p4/sdkconfig`); the board
  reports its configuration and the host refuses expectations of another.
- At real 5 Hz input a step must stay under 200 ms on the board; a p4 step takes about 130 ms (at most 155) and leaves
  about 108 KB of internal RAM. Changes may be P4-only (`MVO_CONFIG_P4`).
- The S3's limits (internal RAM with the two 102 KB pyramids, libmain `.bss` near 94 KB) bind only the s3
  configuration, which is no longer used.

Paths on this machine are in `CLAUDE.local.md`. Repeated procedures are skills in `.claude/skills/`.
