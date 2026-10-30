# Monocular visual odometry in C (core/mono_vo*.c) on the lake clip

The C port of the *pairwise* method runs on the robot camera's 320×240 grey frames and is scored exactly as the OpenCV prototype in `results/lake_clip/mono_vo` (similarity alignment for positions; rotation, direction of travel and heading directly). It follows the prototype's design with its own numerics: its own pyramid and fixed-point Lucas-Kanade, and instead of OpenCV's five-point solver an eight-point MSAC with least-squares refits, a search over travel directions and a robust Levenberg-Marquardt, pairs behind a camera counting as outliers (see core/mono_vo.h). Six changes in design. The prototype stops tracking the pairs that recoverPose drops as farther than 50 baselines, while the C port keeps every epipolar inlier tracked and gives only the nearer ones a point: on short steps the far pairs were most of the features, and dropping them broke the chain of shared depths that carries the scale. The prototype takes each step's median depth ratio as it comes, while the C port keeps it within 3 times the median of the last 10 step lengths (and takes that median when a step shares fewer than 8 points): on a fast turn a median from a few points could be far off, and the chain of shared depths then kept the wrong scale. The prototype takes every essential matrix's translation, while the C port takes none where the pairs give it no evidence over a separately fitted rotation: such a step counts as rotation only when that rotation has 15 tracks and half of them, else as lost. A step lost on a fast turn keeps the last pose in the prototype, while in the C port it keeps a turn when at least 8 tracks and half of them agree on one (at 5 Hz the lost steps turn by 11 degrees in the median, all of which the heading lost): the fitted rotation's, and where only the homography has that support, the essential pose's when it had 15 inliers, else the homography's. A homography the fitted rotation does not share has mostly read the travel's parallax as a turn, over the lake's floor as pitch: on such lost steps at 5 Hz it errs +1.5 degrees in pitch a step against -0.2 for the essential pose, though by as much in all (6.0 and 6.2 degrees), and that bias made the path climb. Any lost step whose essential pose had 15 inliers carries the chain of shared depths on with that pose, still without its translation, when at least 8 depth ratios it shares with the earlier points agree on one scale (their interquartile range at most half their median): dropping every point at each of the 11 lost steps a run restarted the scale from the recent speed each time. Last, where the homography and the essential matrix both turn a step by more than 5 degrees, the C port pulls the essential rotation's angle about the homography's turn axis towards the homography's: on a fast turn two views trade the turn against sideways travel for a fraction of a pixel, and the essential matrix took up to a quarter of each fast step as travel, 14 degrees over the clip's two-second swing at 5 Hz, which one long-baseline estimate across the swing does not lose. Only along that axis: pulling the whole rotation also dragged in the homography's tilt, which the floor's parallax biases, 10 degrees more pitch over the clip. At 10 Hz the steps are half as large and the pull mends the swing's first half more than its second, so the heading over the whole clip comes out worse there than without it.

Each row of the first table is one run with one RANSAC seed. On this clip the seed alone can change the position and heading errors by a factor of two or more, so a difference between two rows is not significant on its own; the second table gives the C port's median and range over 16 seeds.

| Rate | Version | Steps (lost / rotation only) | Position RMSE after alignment | Heading RMSE / max | Direction of travel, median | Scale ratio over 5 s windows | ms per step (PC) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 Hz | C | 338 (4 / 10) | 0.36 m | 33.0° / 57° | 13.8° | 0.16–1.43 | 13.8 |
| 10 Hz | OpenCV prototype | 338 (0 / 1) | 0.41 m | 27.4° / 46° | 13.7° | 0.74–1.70 | 8.4 |
| 5 Hz | C | 169 (15 / 5) | 0.17 m | 11.8° / 23° | 11.9° | 0.78–1.30 | 18.6 |
| 5 Hz | OpenCV prototype | 169 (3 / 0) | 0.44 m | 25.7° / 49° | 11.5° | 0.48–1.65 | 9.7 |
| 3 Hz | C | 101 (18 / 1) | 0.36 m | 26.3° / 54° | 10.9° | 0.32–1.45 | 17.6 |
| 3 Hz | OpenCV prototype | 101 (7 / 0) | 0.33 m | 18.4° / 49° | 12.1° | 0.35–2.12 | 10.7 |

C port over 16 RANSAC seeds, median [smallest–largest]:

| Rate | Position RMSE after alignment | Heading RMSE | Scale ratio spread over 5 s windows (largest / smallest) | Lost steps |
|---:|---:|---:|---:|---:|
| 10 Hz | 0.33 m [0.26–0.37] | 39.5° [30–46] | 5.3× [2.3–9.2] | 2 [1–4] |
| 5 Hz | 0.19 m [0.16–0.26] | 19.4° [5–26] | 1.8× [1.4–2.6] | 13 [11–15] |
| 3 Hz | 0.38 m [0.34–0.56] | 39.1° [26–66] | 4.1× [1.6–8.1] | 17 [16–19] |

`board_expected_10hz.json` holds the FNV-1a hash of every 10 Hz step (338 steps) for the board stream test (uwslam.mvo_stream).

Re-run (the output folder must not exist or be empty):

```bash
python -m uwslam.mvo compare <video> --reconstruction <folder with images.txt> --output results/lake_clip/mono_vo_c_2 --board-frames <file for the board's frames>
```

## ESP32 boards

The host streams grayscale frames and optional masks over the board's serial link (`python -m uwslam.mvo_stream`) of an ESP32-P4: its USB-UART bridge, or the chip's own USB-Serial-JTAG port. The board reports each step's hash and stage times. These timings exclude capture and transfer. Each record provides its own clock, build (core/mono_vo.h), rate, core count and kernel choice (the vector kernels are mvo_vector.S). Before streaming, the board checks its vector dispatch against C on random windows. Records and raw serial lines are in `board/`.

| Record | Clock | Config | Rate | Cores | Kernels | Frames | Steps equal to the PC | Map | Per step, mean / median / max | Tracking | Motion | Detection | Map update | Voxel map / directions / plan / gyro (update) |
|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| `p4_360mhz_usj` | 360 MHz | p4 | 10 Hz | 2 | vector | 339 | 338 of 338 | 2555 observations, hash equal to the PC's | 102 / 99 / 163 ms | 65 ms | 28 ms | 8 ms | 0.0 ms | 3660 voxels, hash equal to the PC's (0.5 ms); directions equal to the PC's (0.18 ms); plans equal to the PC's (1.83 ms) |
| `p4_360mhz_usj_5hz` | 360 MHz | p4 | 5 Hz | 2 | vector | 170 | 169 of 169 | 1628 observations, hash equal to the PC's | 127 / 132 / 154 ms | 86 ms | 25 ms | 16 ms | 0.0 ms | 3530 voxels, hash equal to the PC's (0.5 ms); directions equal to the PC's (0.18 ms); plans equal to the PC's (1.78 ms) |

Records named `*_usj*` were streamed over the ESP32-P4's own USB-Serial-JTAG port, not its USB-UART bridge: the firmware was built as in the second line below, which changes only the console.

Re-run (after `uwslam.mvo compare --board-frames` and `uwslam.mono_map --board-masks`):

```bash
idf.py -C firmware/mono_vo_stream -B firmware/mono_vo_stream/build -D SDKCONFIG=build/sdkconfig -p <port> flash   # the console on a USB-UART bridge
idf.py -C firmware/mono_vo_stream -B firmware/mono_vo_stream/build_usj -D SDKCONFIG=build_usj/sdkconfig -D "SDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.usb_serial_jtag" -p <port> flash   # on USB-Serial-JTAG
python -m uwslam.mvo_stream --port <port> --frames <frames file> --masks <masks file> --label <name>
```
