# Monocular visual odometry on the robot camera's frames (lake clip, reference poses)

Source: `lake.mp4`, 1015 frames (33.8 s at 30 FPS); reference: the reconstruction's keyframes at the 3× pose timing, interpolated to every frame. No IMU, depth or ranger data are used.

- Frames take the robot camera's path (320×240, RGB565) and turn grey; 160×120 is the 2×2 average the segmentation model sees. Intrinsics: a 110° horizontal field of view (fitted in the mask_motion probe).
- Both methods track Shi-Tomasi corners with pyramidal Lucas-Kanade and a forward-backward check. *pairwise*: the essential matrix by RANSAC between consecutive processed frames (a homography when the tracks barely move), with scale carried from step to step by the points both steps triangulate. *keyframe*: the essential matrix only to initialise, then PnP against a map of points triangulated between keyframes (made when the rotation-compensated parallax reaches 6 px at 320×240), and tracks predicted by the last rotation.
- Positions are compared after one similarity alignment of the whole run (scale, rotation, offset); rotation, the direction of travel and the heading are compared directly.

## Results

| Method | Frames | Rate | Steps (lost / rotation only) | Position RMSE after alignment | Final error | Heading RMSE / max | Rotation per pair, median (no rotation) | Direction of travel, median | Scale ratio over 5 s windows | ms per frame (PC) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pairwise | 320x240 | 10 Hz | 338 (0 / 1) | 0.41 m | 0.41 m | 27.4° / 46° | 1.29° (0.93°) | 14.4° | 0.73–1.69 | 5.8 |
| pairwise | 320x240 | 5 Hz | 169 (3 / 0) | 0.44 m | 0.50 m | 25.7° / 49° | 2.24° (1.74°) | 12.7° | 0.48–1.61 | 7.9 |
| pairwise | 320x240 | 3 Hz | 101 (7 / 0) | 0.33 m | 0.55 m | 18.5° / 47° | 3.22° (2.43°) | 11.2° | 0.35–2.12 | 9.1 |
| pairwise | 160x120 | 10 Hz | 338 (0 / 0) | 0.42 m | 0.64 m | 50.5° / 81° | 1.30° (0.93°) | 16.1° | 0.14–1.62 | 4.1 |
| pairwise | 160x120 | 5 Hz | 169 (4 / 0) | 0.84 m | 1.03 m | 9.4° / 49° | 2.08° (1.74°) | 14.4° | 0.45–2.50 | 5.0 |
| pairwise | 160x120 | 3 Hz | 101 (8 / 0) | 1.10 m | 1.08 m | 31.7° / 79° | 3.35° (2.43°) | 11.4° | 0.42–4.60 | 5.7 |
| keyframe | 320x240 | 10 Hz | 338 (0 / 1) | 0.71 m | 1.12 m | 61.7° / 99° | 1.17° (0.93°) | 19.4° | 0.25–2.32 | 4.6 |
| keyframe | 320x240 | 5 Hz | 169 (12 / 5) | 1.13 m | 1.52 m | 71.7° / 106° | 1.92° (1.74°) | 16.2° | 0.12–4.04 | 6.9 |
| keyframe | 320x240 | 3 Hz | 101 (13 / 0) | 1.11 m | 1.37 m | 56.0° / 75° | 3.45° (2.43°) | 13.1° | 0.23–3.72 | 7.3 |
| keyframe | 160x120 | 10 Hz | 338 (0 / 2) | 0.78 m | 1.15 m | 39.9° / 79° | 1.20° (0.93°) | 19.9° | 0.05–4.94 | 3.4 |
| keyframe | 160x120 | 5 Hz | 169 (8 / 5) | 1.09 m | 1.45 m | 46.5° / 63° | 1.94° (1.74°) | 17.4° | 0.03–4.32 | 4.0 |
| keyframe | 160x120 | 3 Hz | 101 (19 / 2) | 0.38 m | 0.71 m | 30.6° / 71° | 3.23° (2.43°) | 13.7° | 0.41–1.41 | 4.3 |

The reference camera travels 8.9 reconstruction units (taken as metres). "Scale ratio over 5 s windows" is the aligned estimated path length over the reference path length in each 5 s window: 1 everywhere means no scale drift.

## Limits

- One diver clip, not the robot; the reference is a reconstruction interpolated between keyframes, in its own units.
- A monocular run has no metric scale: the alignment supplies it after the fact.
- PC timings are for the OpenCV implementation and say nothing about the ESP32-P4.

Re-run (the output folder must not exist or be empty):

```bash
python -m uwslam.experiments.mono_vo <video> --reconstruction <folder with images.txt> --output results/lake_clip/mono_vo_2
```
