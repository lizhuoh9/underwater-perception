# Monocular visual odometry on the robot camera's frames (lake clip, reference poses)

Source: `lake.mp4`, 1015 frames (33.8 s at 30 FPS); reference: the reconstruction's keyframes at the 3× pose timing, one frame later, interpolated to every frame. No IMU, depth or ranger data are used.

- Frames take the robot camera's path (320×240, RGB565) and turn grey; 160×120 is the 2×2 average the segmentation model sees. Intrinsics: a 110° horizontal field of view (the mask_motion probe's fit, kept after the reference was re-timed).
- Both methods track Shi-Tomasi corners with pyramidal Lucas-Kanade and a forward-backward check. *pairwise*: the essential matrix by RANSAC between consecutive processed frames (a homography when the tracks barely move), with scale carried from step to step by the points both steps triangulate. *keyframe*: the essential matrix only to initialise, then PnP against a map of points triangulated between keyframes (made when the rotation-compensated parallax reaches 6 px at 320×240), and tracks predicted by the last rotation.
- Positions are compared after one similarity alignment of the whole run (scale, rotation, offset); rotation, the direction of travel and the heading are compared directly.

## Results

| Method | Frames | Rate | Steps (lost / rotation only) | Position RMSE after alignment | Final error | Heading RMSE / max | Rotation per pair, median (no rotation) | Direction of travel, median | Scale ratio over 5 s windows | ms per frame (PC) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pairwise | 320x240 | 10 Hz | 338 (0 / 1) | 0.41 m | 0.41 m | 27.4° / 46° | 1.29° (0.87°) | 13.7° | 0.74–1.70 | 8.4 |
| pairwise | 320x240 | 5 Hz | 169 (3 / 0) | 0.44 m | 0.50 m | 25.7° / 49° | 2.24° (1.72°) | 11.5° | 0.48–1.65 | 9.7 |
| pairwise | 320x240 | 3 Hz | 101 (7 / 0) | 0.33 m | 0.56 m | 18.4° / 49° | 3.24° (2.46°) | 12.1° | 0.35–2.12 | 10.7 |
| pairwise | 160x120 | 10 Hz | 338 (0 / 0) | 0.42 m | 0.64 m | 50.5° / 81° | 1.29° (0.87°) | 16.6° | 0.14–1.62 | 4.4 |
| pairwise | 160x120 | 5 Hz | 169 (4 / 0) | 0.84 m | 1.03 m | 9.3° / 42° | 2.05° (1.72°) | 13.9° | 0.45–2.50 | 5.0 |
| pairwise | 160x120 | 3 Hz | 101 (8 / 0) | 1.10 m | 1.09 m | 31.7° / 78° | 3.26° (2.46°) | 11.3° | 0.42–4.59 | 5.5 |
| keyframe | 320x240 | 10 Hz | 338 (0 / 1) | 0.71 m | 1.13 m | 61.7° / 100° | 1.16° (0.87°) | 19.1° | 0.25–2.37 | 4.8 |
| keyframe | 320x240 | 5 Hz | 169 (12 / 5) | 1.13 m | 1.52 m | 71.7° / 106° | 1.97° (1.72°) | 16.8° | 0.12–4.10 | 6.0 |
| keyframe | 320x240 | 3 Hz | 101 (13 / 0) | 1.11 m | 1.37 m | 56.0° / 76° | 3.47° (2.46°) | 12.7° | 0.23–3.72 | 7.7 |
| keyframe | 160x120 | 10 Hz | 338 (0 / 2) | 0.78 m | 1.15 m | 40.0° / 80° | 1.20° (0.87°) | 20.2° | 0.05–5.02 | 3.5 |
| keyframe | 160x120 | 5 Hz | 169 (8 / 5) | 1.09 m | 1.46 m | 46.5° / 64° | 1.95° (1.72°) | 17.3° | 0.03–4.38 | 4.5 |
| keyframe | 160x120 | 3 Hz | 101 (19 / 2) | 0.38 m | 0.71 m | 30.6° / 72° | 3.31° (2.46°) | 13.6° | 0.40–1.41 | 4.7 |

The reference camera travels 8.9 reconstruction units (taken as metres). "Scale ratio over 5 s windows" is the aligned estimated path length over the reference path length in each 5 s window: 1 everywhere means no scale drift.

## Limits

- One diver clip, not the robot; the reference is a reconstruction interpolated between keyframes, in its own units.
- A monocular run has no metric scale: the alignment supplies it after the fact.
- PC timings are for the OpenCV implementation and say nothing about the ESP32-P4.

Re-run (the output folder must not exist or be empty):

```bash
python -m uwslam.experiments.mono_vo <video> --reconstruction <folder with images.txt> --output results/lake_clip/mono_vo_2
```
