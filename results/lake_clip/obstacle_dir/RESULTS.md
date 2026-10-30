# Obstacle directions from the water masks (lake clip)

The segmentation's 60×80 mask of each frame (not water: red in the figures) is split into 3 × 5 directions of the camera's view by `core/obstacle_dir.c`: bands up / level / down at ±15° and sectors at ±15° and ±40° from the optical axis. A direction is blocked as soon as half of it is not water and freed only after two frames below 30%; the choice is the free direction nearest straight ahead (fewest steps, turning before rising or diving, then the more open). It is scored against the reference reconstruction, seen through the same grid from the reference camera of each frame: a direction counts as blocked at range R when half of its cells hold a reconstructed surface nearer than R.

The reconstruction has no water surface, so in the upper band the mask blocks directions the reference calls open: the surface bounds the robot too. The mask says nothing about range; the voxel map does.

## 10 Hz (339 frames)

No free direction in 3% of the frames.

| Range | Band | Reference blocked | Mask blocked | Reference-blocked directions the mask blocks | Mask-blocked directions the reference blocks |
|---:|---|---:|---:|---:|---:|
| 0.5 m | up | 25% | 89% | 100% | 28% |
| 0.5 m | level | 26% | 60% | 100% | 43% |
| 0.5 m | down | 61% | 100% | 100% | 61% |
| 0.5 m | all | 37% | 83% | 100% | 45% |
| 1 m | up | 35% | 89% | 100% | 40% |
| 1 m | level | 45% | 60% | 100% | 75% |
| 1 m | down | 83% | 100% | 100% | 83% |
| 1 m | all | 55% | 83% | 100% | 66% |
| 2 m | up | 36% | 89% | 100% | 40% |
| 2 m | level | 54% | 60% | 99% | 89% |
| 2 m | down | 86% | 100% | 100% | 86% |
| 2 m | all | 59% | 83% | 100% | 70% |

| Range | The choice is free | Straight ahead is free | A random direction is free | Share correlation |
|---:|---:|---:|---:|---:|
| 0.5 m | 100% | 88% | 66% | 0.54 |
| 1 m | 100% | 61% | 45% | 0.68 |
| 2 m | 99% | 20% | 40% | 0.71 |

## 5 Hz (170 frames)

No free direction in 4% of the frames.

| Range | Band | Reference blocked | Mask blocked | Reference-blocked directions the mask blocks | Mask-blocked directions the reference blocks |
|---:|---|---:|---:|---:|---:|
| 0.5 m | up | 25% | 90% | 100% | 27% |
| 0.5 m | level | 26% | 61% | 100% | 42% |
| 0.5 m | down | 61% | 100% | 100% | 61% |
| 0.5 m | all | 37% | 84% | 100% | 44% |
| 1 m | up | 36% | 90% | 100% | 40% |
| 1 m | level | 45% | 61% | 100% | 74% |
| 1 m | down | 84% | 100% | 100% | 84% |
| 1 m | all | 55% | 84% | 100% | 65% |
| 2 m | up | 36% | 90% | 100% | 40% |
| 2 m | level | 54% | 61% | 100% | 88% |
| 2 m | down | 86% | 100% | 100% | 86% |
| 2 m | all | 59% | 84% | 100% | 70% |

| Range | The choice is free | Straight ahead is free | A random direction is free | Share correlation |
|---:|---:|---:|---:|---:|
| 0.5 m | 100% | 87% | 64% | 0.54 |
| 1 m | 100% | 61% | 40% | 0.68 |
| 2 m | 99% | 20% | 36% | 0.71 |

Figures: `directions_<rate>hz.png` (example frames with their directions, mask and reference depth) and `level_band_<rate>hz.png` (the level band over the clip).

Re-run (the output folder must not exist or be empty):

```bash
python -m uwslam.obstacle_dir <video> --reconstruction <folder with images.txt and images.ply> --output results/lake_clip/obstacle_dir_2
```
