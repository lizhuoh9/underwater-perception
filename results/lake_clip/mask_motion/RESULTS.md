# Camera rotation from water masks alone (lake clip, reference poses)

Source: `lake.mp4`, 1014 frames (33.8 s at 30 FPS); reference: the reconstruction's 88 keyframes, interpolated to every frame.

- **Reference timing.** The reconstruction's time column is image index / 30 but its images were taken at 10 FPS, so video time = 3 × that column, plus one frame (33.3 ms): image i is video frame 3 i + 1, which tracking over the keyframes 0.1 s apart showed (see the module's docstring). Read so, SIFT rotations between full-resolution frames 0.1 s apart follow the reference: yaw correlation 0.86, median difference 1.1° (334 pairs); without the frame 0.86 and 1.1° (334 pairs): over all these pairs the frame hardly matters, since most turn slowly, while over the keyframes 0.1 s apart, which sit in the swings, rotations tracked on the robot's 320×240 frames differ from it by a median 1.09° one frame later against 1.92° without. Read 1:1 they do not follow it: -0.01 and 3.9° (109 pairs).
- **Masks.** Each frame takes the robot's input path to 160×120 and the int8 water-segmentation model at its deployment threshold gives the 60×80 non-water mask. Boundary cells are non-water cells with a water neighbour.
- **Mask gyro.** A pure rotation moves image points by K R K⁻¹ whatever their depth. One frame's boundary cells are turned into the other frame and scored by the 3-4 chamfer distance to its boundary (capped at 5 cells), in both directions. A coarse grid (yaw ±60°, pitch ±15°, roll ±12° in 3° steps) is refined twice (1°, 0.25°). Frames with fewer than 20 boundary cells get no estimate; those pairs count as zero rotation.
- **References.** *Edges*: the same search on Canny edges of the 160×120 grey frame, pooled onto the same 60×80 grid. *Features*: SIFT and the essential matrix on the full-resolution frames. *Zero*: no rotation.
- **Intrinsics.** Not calibrated. All sources use the tools' 110° horizontal field of view, which this fit chose before the reference was re-timed: among 60–130°, the smallest median difference between the 10 Hz feature rotations and the reference on pairs turning by more than 3° (table below; on smaller turns feature noise dominates and favours any narrow view). Re-timed, the fit picks 105° (median 1.29° against 1.56° at 110°), but in the C odometry narrower views won heading at 5 and 10 Hz on most measures and lost position at 5 Hz, so the tools keep 110°.

## Rotation error (angle between estimated and reference rotation)

| Rate | Pairs | Reference median / P90 | Zero | Mask median / P90 | Mask < 2° | Mask beats zero | Mask yaw corr. | Edges median / P90 | Edges yaw corr. | Features median / P90 | Features answered |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 Hz | 337 | 0.9° / 4.7° | 0.9° | 1.9° / 7.0° | 52% | 19% | 0.55 | 0.9° / 4.4° | 0.56 | 1.1° / 2.6° | 99% |
| 5 Hz | 336 | 1.6° / 9.2° | 1.6° | 3.2° / 10.1° | 32% | 25% | 0.62 | 1.6° / 8.4° | 0.48 | 1.9° / 4.0° | 98% |
| 3 Hz | 335 | 2.6° / 16.1° | 2.6° | 4.5° / 13.5° | 16% | 32% | 0.74 | 2.5° / 15.9° | -0.05 | 2.8° / 7.3° | 96% |

Pairs start every 3 frames. Medians and P90 count pairs without an estimate as zero rotation; "beats zero" counts answered pairs only. Yaw correlation is between estimated and reference yaw over answered pairs.

## How much of the mask change is rotation

| Rate | Reference translation (median, reconstruction units) | Cells changed (raw) | Still changed after the reference rotation |
|---:|---:|---:|---:|
| 10 Hz | 0.025 | 5% | 5% |
| 5 Hz | 0.049 | 6% | 6% |
| 3 Hz | 0.082 | 7% | 7% |

What the reference rotation leaves comes from translation (parallax), the visibility edge moving with the camera and segmentation flicker; they cannot be told apart here.

## Field-of-view choice

| HFOV | Median feature-vs-reference angle at 10 Hz (turns > 3°) | Pairs |
|---:|---:|---:|
| 60° | 6.1° | 52 |
| 65° | 4.7° | 52 |
| 70° | 3.5° | 53 |
| 75° | 3.3° | 52 |
| 80° | 3.0° | 51 |
| 85° | 2.3° | 47 |
| 90° | 1.8° | 43 |
| 95° | 1.5° | 48 |
| 100° | 1.4° | 47 |
| 105° | 1.3° | 49 |
| 110° | 1.6° | 52 |
| 115° | 1.9° | 52 |
| 120° | 1.9° | 51 |
| 125° | 2.6° | 53 |
| 130° | 3.0° | 52 |

## Limits

- One 34 s lake clip filmed by a diver, not the robot in a pool; the reference is a reconstruction whose keyframes are interpolated, with units of its own.
- The field of view was fitted to the same reference it is scored against, which pins it no better than about ten degrees.
- The search estimates rotation only, so translation parallax leaks into it.

Re-run (the output folder must not exist or be empty):

```bash
python -m uwslam.experiments.mask_motion <video> --reconstruction <folder with images.txt> --model <segmentation model folder> --output results/lake_clip/mask_motion_2
```
