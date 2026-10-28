# Monocular visual odometry in C (core/mono_vo*.c) on the lake clip

The C port of the *pairwise* method runs on the robot camera's 320×240 grey frames and is scored exactly as the OpenCV prototype in `results/lake_clip/mono_vo` (similarity alignment for positions; rotation, direction of travel and heading directly). It follows the prototype's design with its own numerics: its own pyramid and fixed-point Lucas-Kanade, and instead of OpenCV's five-point solver an eight-point MSAC with least-squares refits, a search over travel directions and a robust Levenberg-Marquardt, pairs behind a camera counting as outliers (see core/mono_vo.h). Five changes in design. The prototype stops tracking the pairs that recoverPose drops as farther than 50 baselines, while the C port keeps every epipolar inlier tracked and gives only the nearer ones a point: on short steps the far pairs were most of the features, and dropping them broke the chain of shared depths that carries the scale. The prototype takes each step's median depth ratio as it comes, while the C port keeps it within 3 times the median of the last 10 step lengths (and takes that median when a step shares fewer than 8 points): on a fast turn a median from a few points could be far off, and the chain of shared depths then kept the wrong scale. The prototype takes every essential matrix's translation, while the C port takes none where the pairs give it no evidence over a separately fitted rotation: such a step counts as rotation only when that rotation has 15 tracks and half of them, else as lost. A step lost on a fast turn keeps the last pose in the prototype, while in the C port it keeps a turn when at least 8 tracks and half of them agree on one (at 5 Hz the lost steps turn by 11 degrees in the median, all of which the heading lost): the fitted rotation's, else the essential pose's when it had 15 inliers, else the homography's. A homography the fitted rotation does not share has read the travel's parallax as a turn, over the lake's floor as pitch: on the lost 5–10 degree turns at 5 Hz, +3.8 degrees a step against +0.7 for the essential pose, which made the path climb. Last, where the homography and the essential matrix both turn a step by more than 5 degrees, the C port pulls the essential rotation's angle about the homography's turn axis towards the homography's: on a fast turn two views trade the turn against sideways travel for a fraction of a pixel, and the essential matrix took up to a quarter of each fast step as travel, 14 degrees over the clip's two-second swing at 5 Hz, which one long-baseline estimate across the swing does not lose. Only along that axis: pulling the whole rotation also dragged in the homography's tilt, which the floor's parallax biases, 10 degrees more pitch over the clip. At 10 Hz the steps are half as large and the pull mends the swing's first half more than its second, so the heading over the whole clip comes out worse there than without it.

Each row of the first table is one run with one RANSAC seed. On this clip the seed alone can change the position and heading errors by a factor of two or more, so a difference between two rows is not significant on its own; the second table gives the C port's median and range over 16 seeds.

| Rate | Version | Steps (lost / rotation only) | Position RMSE after alignment | Heading RMSE / max | Direction of travel, median | Scale ratio over 5 s windows | ms per step (PC) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 Hz | C | 338 (1 / 12) | 0.36 m | 26.0° / 48° | 12.6° | 0.21–1.41 | 9.2 |
| 10 Hz | OpenCV prototype | 338 (0 / 1) | 0.41 m | 27.4° / 46° | 14.4° | 0.73–1.69 | 5.8 |
| 5 Hz | C | 169 (12 / 6) | 0.32 m | 8.5° / 32° | 11.5° | 0.31–1.35 | 9.6 |
| 5 Hz | OpenCV prototype | 169 (3 / 0) | 0.44 m | 25.7° / 49° | 12.7° | 0.48–1.61 | 7.9 |
| 3 Hz | C | 101 (20 / 0) | 0.43 m | 9.8° / 48° | 9.4° | 0.28–2.40 | 10.5 |
| 3 Hz | OpenCV prototype | 101 (7 / 0) | 0.33 m | 18.5° / 47° | 11.2° | 0.35–2.12 | 9.1 |

C port over 16 RANSAC seeds, median [smallest–largest]:

| Rate | Position RMSE after alignment | Heading RMSE | Scale ratio spread over 5 s windows (largest / smallest) | Lost steps |
|---:|---:|---:|---:|---:|
| 10 Hz | 0.31 m [0.25–0.37] | 34.2° [26–39] | 3.4× [1.6–12.5] | 2 [1–5] |
| 5 Hz | 0.40 m [0.29–0.51] | 18.3° [9–25] | 5.8× [2.6–8.0] | 11 [9–14] |
| 3 Hz | 0.43 m [0.38–0.52] | 27.1° [8–43] | 9.0× [6.9–11.7] | 21 [19–23] |

`board_expected_10hz.json` holds the FNV-1a hash of every 10 Hz step (338 steps) for the ESP32-S3 stream test.

Re-run (the output folder must not exist or be empty):

```bash
python -m uwslam.mvo compare <video> --reconstruction <folder with images.txt> --output results/lake_clip/mono_vo_c_2 --board-frames <file for the board's frames>
```

## ESP32-S3

The host streams grayscale frames and optional masks over USB (`python -m uwslam.mvo_stream`); the board reports each step's hash and stage times. These timings exclude capture and USB transfer. Each record provides its own rate, core count and kernel choice. Before streaming, the board checks its vector dispatch against C on random windows. Records and raw serial lines are in `board/`.

| Record | Rate | Cores | Kernels | Frames | Steps equal to the PC | Map | Per step, mean / median / max | Tracking | Motion | Detection | Map update | Voxel map / directions / plan (update) |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| `240mhz` | 10 Hz | 2 | vector | 339 | 338 of 338 | 2108 observations, hash equal to the PC's | 135 / 136 / 191 ms | 55 ms | 64 ms | 16 ms | 0.0 ms | 1006 voxels, hash equal to the PC's (1.1 ms); directions equal to the PC's (0.26 ms); plans equal to the PC's (1.28 ms) |
| `240mhz_5hz` | 5 Hz | 2 | vector | 170 | 169 of 169 | 1348 observations, hash equal to the PC's | 160 / 169 / 207 ms | 71 ms | 57 ms | 32 ms | 0.0 ms | 918 voxels, hash equal to the PC's (0.9 ms); directions equal to the PC's (0.26 ms); plans equal to the PC's (1.13 ms) |

Re-run (after `uwslam.mvo compare --board-frames` and `uwslam.mono_map --board-masks`):

```bash
idf.py -C firmware/esp32s3_mono_vo -B firmware/esp32s3_mono_vo/build_240 -p <port> flash
python -m uwslam.mvo_stream --port <port> --frames <frames file> --masks <masks file> --label <name>
```
