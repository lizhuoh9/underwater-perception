# uwslam

Monocular-camera SLAM for a small underwater robot. A C core (`core/`) runs bit-identically on a PC and an ESP32-S3;
Python tools (`src/uwslam/`) bind it with ctypes and score it against a reconstruction of a lake clip.

## Layout

- `core/`: C99, no heap, no libm. Odometry `mono_vo*.c` (five units, listed in `mono_vo_internal.h`), `voxel_map.c`,
  `obstacle_dir.c`, `path_plan.c`, `mono_map.c`, `mono_cloud.c`; prefixes `mvo_`, `vm_`, `od_`, `pp_`, `mm_`, `mc_`.
- `src/uwslam/`: bindings and lake-clip tools (`mvo`, `voxel_map`, `obstacle_dir`, `path_plan`, `mono_map`,
  `mono_cloud`), the board host `mvo_stream`, and `experiments/` (OpenCV prototype `mono_vo`, reference poses and
  masks `mask_motion`, `segmenter`).
- `tests/`: unittest suite; `tests/c/` holds the C contract checks and the firmware host harnesses.
- `firmware/esp32s3_mono_vo/`: the only firmware, for the ESP32-S3 (default target) and the ESP32-P4
  (`sdkconfig.defaults.<target>`); frames and masks arrive over USB-Serial-JTAG (S3) or the console UART (P4).
- `results/lake_clip/`: committed results, including the board's expected-hash files.
- `local/`: experiment folders, not in git.

## Commands

From the repository root, with the Python and gcc that `CLAUDE.local.md` names:

```bash
python -m unittest discover -s tests -t .    # about 1.5 min; builds the C core with gcc
python -m uwslam.mvo compare <video> --reconstruction <folder> --output <fresh folder> [--board-frames <file>]
python -m uwslam.mono_map <video> --reconstruction <folder> --output <fresh folder> [--board-masks <file>]
python -m uwslam.voxel_map <video> --reconstruction <folder> --output <fresh folder>    # obstacle_dir, path_plan alike
```

Every tool refuses to overwrite its output: give it a fresh folder (scratch runs go under `local/`). The core has two
build configurations (`core/mono_vo.h`): `s3`, the default, and `p4` (`MVO_CONFIG_P4`: 300 features instead of 250).
Tools and tests build the one `UWSLAM_CONFIG` names; the `p4` results live in `results/lake_clip_p4/`.

## Verifying a change

- The tests pass, in both configurations (again with `UWSLAM_CONFIG=p4`).
- Accuracy: the `lake-score` skill. Score against a control run of the last commit with identical settings, over the
  16 seeds in `uwslam.mvo.SEEDS`, on the 10 s clip windows as well as the full clip. One run or one seed proves
  nothing: the seed alone can change a score twofold.
- Board: the `board-check` skill. Every step, map, voxel, direction and plan hash must equal the PC's.
- A change to `core/` or its headers changes the source SHA-256 the firmware checks (`uwslam.mvo.CORE_FILES`, guarded
  by `tests/test_mvo_sources.py`): regenerate the expected files and reflash. Finish comment edits in `core/` before
  generating results, since a comment changes the hashes too.

## Conventions

- English code, comments and reports; README.md and README.zh-CN.md change together.
- No personal paths or names in the repository: machine paths enter through CLI arguments or environment variables,
  and outputs record file names, not paths.
- Structs whose bytes are hashed have no padding; static asserts check it.
- The board's input rate is 5 Hz; 10 Hz is scored as well.

## Limits

- The board is an ESP32-S3 with its PSRAM unused. The two 102 KB image pyramids need separate internal blocks
  allocated before the USB driver, libmain `.bss` must stay near 94 KB, and the voxel map is allocated last. Internal
  RAM is the limit.
- At real 5 Hz input a step must stay under 200 ms on the board.
- The ESP32-P4 firmware builds the `p4` configuration and tracks with `mvo_pie_p4.S` (the window kernels; the block
  copy stays C). Its build folder has its own sdkconfig (`-D SDKCONFIG=build_p4/sdkconfig`); the board reports its
  configuration and the host refuses expectations of another.
- A change that touches both configurations is scored and board-checked in both (`UWSLAM_CONFIG=p4` for the second).

Paths on this machine are in `CLAUDE.local.md`. Repeated procedures are skills in `.claude/skills/`.
