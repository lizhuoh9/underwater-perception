# Camera rotation from water masks alone (lake clip, reference poses)

Source: `lake.mp4`, 1014 frames (33.8 s at 30 FPS); reference: the reconstruction's 88 keyframes, interpolated to every frame.

- **Reference timing.** The reconstruction's time column is image index / 30 but its images were taken at 10 FPS, so video time = 3 × that column. With this scale, SIFT rotations between full-resolution frames 0.1 s apart follow the reference: yaw correlation 0.86, median difference 1.1° (334 pairs). Read 1:1 they do not: -0.01 and 3.9° (109 pairs).
- **Masks.** Each frame takes the robot's input path to 160×120 and the int8 water-segmentation model at its deployment threshold gives the 60×80 non-water mask. Boundary cells are non-water cells with a water neighbour.
- **Mask gyro.** A pure rotation moves image points by K R K⁻¹ whatever their depth. One frame's boundary cells are turned into the other frame and scored by the 3-4 chamfer distance to its boundary (capped at 5 cells), in both directions. A coarse grid (yaw ±60°, pitch ±15°, roll ±12° in 3° steps) is refined twice (1°, 0.25°). Frames with fewer than 20 boundary cells get no estimate; those pairs count as zero rotation.
- **References.** *Edges*: the same search on Canny edges of the 160×120 grey frame, pooled onto the same 60×80 grid. *Features*: SIFT and the essential matrix on the full-resolution frames. *Zero*: no rotation.
- **Intrinsics.** Not calibrated. A 110° horizontal field of view was chosen because, among 60–130°, it gives the smallest median difference between the 10 Hz feature rotations and the reference on pairs turning by more than 3° (table below; on smaller turns feature noise dominates and favours any narrow view). All sources use it.

## Rotation error (angle between estimated and reference rotation)

| Rate | Pairs | Reference median / P90 | Zero | Mask median / P90 | Mask < 2° | Mask beats zero | Mask yaw corr. | Edges median / P90 | Edges yaw corr. | Features median / P90 | Features answered |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 Hz | 337 | 0.9° / 4.6° | 0.9° | 2.0° / 7.0° | 50% | 21% | 0.58 | 0.9° / 4.1° | 0.56 | 1.1° / 2.6° | 99% |
| 5 Hz | 336 | 1.7° / 9.4° | 1.7° | 3.3° / 10.1° | 32% | 24% | 0.63 | 1.6° / 9.2° | 0.46 | 2.0° / 4.3° | 98% |
| 3 Hz | 335 | 2.5° / 15.4° | 2.5° | 4.5° / 13.6° | 17% | 33% | 0.74 | 2.5° / 15.6° | -0.05 | 2.8° / 7.4° | 96% |

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
| 60° | 5.9° | 52 |
| 65° | 4.2° | 52 |
| 70° | 3.4° | 54 |
| 75° | 2.9° | 53 |
| 80° | 2.5° | 52 |
| 85° | 2.2° | 47 |
| 90° | 1.7° | 43 |
| 95° | 1.6° | 48 |
| 100° | 1.6° | 47 |
| 105° | 1.6° | 49 |
| 110° | 1.4° | 52 |
| 115° | 1.7° | 52 |
| 120° | 2.1° | 51 |
| 125° | 2.6° | 53 |
| 130° | 3.0° | 54 |

## Limits

- One 34 s lake clip filmed by a diver, not the robot in a pool; the reference is a reconstruction whose keyframes are interpolated, with units of its own.
- The field of view is fitted to the same reference it is scored against.
- The search estimates rotation only, so translation parallax leaks into it.

Re-run (the output folder must not exist or be empty):

```bash
python -m uwslam.experiments.mask_motion <video> --reconstruction <folder with images.txt> --model <segmentation model folder> --output results/lake_clip/mask_motion_2
```
