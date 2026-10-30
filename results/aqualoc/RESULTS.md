# The odometry with and without the gyro on AQUALOC harbour sequences

5 Hz, 16 seeds; 10 s windows every 5 s and whole sequences. Each cell is the median over the runs and, for a gyro mode, the share of the runs paired with `off` (same window, same seed) in which it is lower. `gyro` learns the bias from zero at the run's start; `warm` starts each window from the whole-sequence run's estimate at that point. The camera looks down at the seabed: yaw is about the vertical, orientation the whole rotation error, heading the turn about the first camera's y axis (a tilt here).

## 10 s windows

| Sequence | Mode | Runs | ATE (m) | yaw RMSE (°) | orientation RMSE (°) | heading RMSE (°) | lost steps |
|---|---|---:|---:|---:|---:|---:|---:|
| seq01 | off | 704 | 0.247 | 3.34 | 21.02 | 8.69 | 0 |
| seq01 | gyro | 704 | 0.055 (96%) | 4.13 (39%) | 4.81 (98%) | 0.74 (94%) | 10 (0%) |
| seq01 | warm | 704 | 0.076 (96%) | 0.91 (86%) | 2.51 (98%) | 0.90 (98%) | 12 (0%) |
| seq07 | off | 336 | 0.180 | 7.49 | 23.70 | 14.72 | 1 |
| seq07 | gyro | 336 | 0.062 (88%) | 1.78 (94%) | 7.06 (99%) | 3.62 (88%) | 4 (38%) |
| seq07 | warm | 336 | 0.059 (93%) | 1.52 (98%) | 4.88 (100%) | 2.69 (92%) | 4 (43%) |
| all | off | 1040 | 0.233 | 3.74 | 21.23 | 9.02 | 0 |
| all | gyro | 1040 | 0.060 (93%) | 2.84 (57%) | 5.39 (98%) | 1.25 (92%) | 8 (12%) |
| all | warm | 1040 | 0.067 (95%) | 1.07 (90%) | 3.00 (98%) | 1.18 (96%) | 9 (14%) |

## Whole sequences

| Sequence | Mode | Runs | ATE (m) | yaw RMSE (°) | orientation RMSE (°) | heading RMSE (°) | lost steps |
|---|---|---:|---:|---:|---:|---:|---:|
| seq01 | off | 16 | 3.763 | 60.14 | 111.51 | 101.42 | 3 |
| seq01 | gyro | 16 | 2.105 (100%) | 6.65 (100%) | 18.33 (100%) | 8.05 (100%) | 320 (0%) |
| seq07 | off | 16 | 2.703 | 83.03 | 116.56 | 92.80 | 55 |
| seq07 | gyro | 16 | 2.013 (100%) | 11.40 (100%) | 27.91 (100%) | 21.12 (100%) | 62 (0%) |
| all | off | 32 | 3.248 | 66.24 | 112.30 | 98.68 | 28 |
| all | gyro | 32 | 2.035 (100%) | 8.35 (100%) | 20.90 (100%) | 9.97 (100%) | 187 (0%) |

## The bias the whole-sequence runs learnt (deg/s, the IMU's axes)

- seq01: median x +0.413, y +0.979, z -0.180 over 16 seeds; fitted to the reference: x +0.093, y +1.296, z +0.054
- seq07: median x -0.184, y +0.078, z +0.673 over 16 seeds; fitted to the reference: x -0.273, y +0.379, z +0.172

Re-run (the output folder must not exist or be empty):

```bash
python -m uwslam.aqualoc_score <prepared seq01> <prepared seq07> --output <fresh folder>
```
