# The odometry with and without the IMU on AQUALOC harbour sequences

5 Hz, 16 seeds; 10 s windows every 5 s and whole sequences. Each cell is the median over the runs and, for a mode with a base, the share of the runs paired with the base (same window, same seed) in which it is lower: `gyro` and `warm` against `off`, `imu` and `imu_raw` against `gyro`. `gyro` learns the bias from zero at the run's start; `warm` starts each window from the whole-sequence run's estimate at that point; `imu` adds the accelerometer, its bias calibrated with the other sequences' fitted one; `imu_raw` adds it uncalibrated. The camera looks down at the seabed: yaw is about the vertical, tilt the angle between the estimated and the reference's up, orientation the whole rotation error, heading the turn about the first camera's y axis (a tilt here); the levelled ATE turns the path only about the vertical, and height is its vertical part.

## 10 s windows

| Sequence | Mode | Runs | ATE (m) | levelled ATE (m) | height (m) | yaw RMSE (°) | tilt RMSE (°) | orientation RMSE (°) | heading RMSE (°) | lost steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seq01 | off | 704 | 0.247 | 0.291 | 0.071 | 3.34 | 20.86 | 21.02 | 8.69 | 0 |
| seq01 | gyro | 704 | 0.055 (96%) | 0.075 (98%) | 0.027 (80%) | 4.13 (39%) | 1.52 (100%) | 4.81 (98%) | 0.74 (94%) | 10 (0%) |
| seq01 | warm | 704 | 0.076 (96%) | 0.081 (96%) | 0.030 (73%) | 0.91 (86%) | 2.24 (100%) | 2.51 (98%) | 0.90 (98%) | 12 (0%) |
| seq01 | imu | 704 | 0.055 (46%) | 0.075 (56%) | 0.023 (52%) | 3.95 (82%) | 0.56 (97%) | 4.00 (100%) | 0.52 (69%) | 10 (0%) |
| seq01 | imu_raw | 704 | 0.055 (51%) | 0.076 (51%) | 0.023 (51%) | 3.98 (75%) | 1.13 (63%) | 4.19 (55%) | 0.86 (44%) | 10 (0%) |
| seq07 | off | 336 | 0.180 | 0.221 | 0.117 | 7.49 | 21.68 | 23.70 | 14.72 | 1 |
| seq07 | gyro | 336 | 0.062 (88%) | 0.075 (91%) | 0.052 (79%) | 1.78 (94%) | 6.80 (99%) | 7.06 (99%) | 3.62 (88%) | 4 (38%) |
| seq07 | warm | 336 | 0.059 (93%) | 0.070 (96%) | 0.049 (85%) | 1.52 (98%) | 4.69 (98%) | 4.88 (100%) | 2.69 (92%) | 4 (43%) |
| seq07 | imu | 336 | 0.060 (64%) | 0.069 (69%) | 0.042 (68%) | 1.38 (78%) | 2.20 (97%) | 2.69 (97%) | 1.32 (91%) | 4 (0%) |
| seq07 | imu_raw | 336 | 0.060 (63%) | 0.067 (73%) | 0.037 (66%) | 1.43 (59%) | 2.59 (94%) | 3.27 (94%) | 1.45 (77%) | 4 (0%) |
| all | off | 1040 | 0.233 | 0.276 | 0.078 | 3.74 | 20.93 | 21.23 | 9.02 | 0 |
| all | gyro | 1040 | 0.060 (93%) | 0.075 (96%) | 0.036 (80%) | 2.84 (57%) | 2.04 (99%) | 5.39 (98%) | 1.25 (92%) | 8 (12%) |
| all | warm | 1040 | 0.067 (95%) | 0.079 (96%) | 0.032 (77%) | 1.07 (90%) | 2.69 (99%) | 3.00 (98%) | 1.18 (96%) | 9 (14%) |
| all | imu | 1040 | 0.058 (52%) | 0.070 (60%) | 0.031 (57%) | 2.26 (80%) | 0.68 (97%) | 3.51 (99%) | 0.64 (76%) | 8 (0%) |
| all | imu_raw | 1040 | 0.058 (55%) | 0.070 (58%) | 0.031 (56%) | 2.46 (70%) | 1.61 (73%) | 3.85 (68%) | 1.12 (55%) | 8 (0%) |

## Whole sequences

| Sequence | Mode | Runs | ATE (m) | levelled ATE (m) | height (m) | yaw RMSE (°) | tilt RMSE (°) | orientation RMSE (°) | heading RMSE (°) | lost steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seq01 | off | 16 | 3.763 | 4.314 | 0.832 | 60.14 | 99.70 | 111.51 | 101.42 | 3 |
| seq01 | gyro | 16 | 2.105 (100%) | 2.168 (100%) | 0.670 (81%) | 6.65 (100%) | 17.43 (100%) | 18.33 (100%) | 8.05 (100%) | 320 (0%) |
| seq01 | imu | 16 | 2.095 (62%) | 2.104 (100%) | 0.386 (100%) | 6.84 (31%) | 0.81 (100%) | 6.87 (100%) | 0.85 (100%) | 320 (0%) |
| seq01 | imu_raw | 16 | 2.063 (100%) | 2.100 (100%) | 0.337 (100%) | 7.26 (6%) | 5.16 (100%) | 8.58 (100%) | 2.78 (100%) | 320 (0%) |
| seq07 | off | 16 | 2.703 | 2.913 | 0.530 | 83.03 | 92.99 | 116.56 | 92.80 | 55 |
| seq07 | gyro | 16 | 2.013 (100%) | 2.317 (100%) | 0.907 (12%) | 11.40 (100%) | 25.35 (100%) | 27.91 (100%) | 21.12 (100%) | 62 (0%) |
| seq07 | imu | 16 | 2.028 (25%) | 2.030 (100%) | 0.232 (100%) | 9.44 (94%) | 2.81 (100%) | 9.64 (100%) | 2.66 (100%) | 62 (0%) |
| seq07 | imu_raw | 16 | 2.010 (56%) | 2.034 (100%) | 0.246 (100%) | 10.25 (56%) | 5.84 (100%) | 12.14 (100%) | 4.82 (100%) | 62 (0%) |
| all | off | 32 | 3.248 | 3.587 | 0.782 | 66.24 | 98.51 | 112.30 | 98.68 | 28 |
| all | gyro | 32 | 2.035 (100%) | 2.277 (100%) | 0.761 (47%) | 8.35 (100%) | 19.32 (100%) | 20.90 (100%) | 9.97 (100%) | 187 (0%) |
| all | imu | 32 | 2.048 (44%) | 2.055 (100%) | 0.317 (100%) | 7.60 (62%) | 1.63 (100%) | 7.65 (100%) | 1.75 (100%) | 187 (0%) |
| all | imu_raw | 32 | 2.029 (78%) | 2.053 (100%) | 0.292 (100%) | 8.16 (31%) | 5.29 (100%) | 9.81 (100%) | 3.66 (100%) | 187 (0%) |

## The accelerometer's calibration (m/s², the IMU's axes)

- seq01: fitted to its reference x +0.173, y -0.691, z +0.579; calibrated with x +0.265, y -0.035, z +0.601
- seq07: fitted to its reference x +0.265, y -0.035, z +0.601; calibrated with x +0.173, y -0.691, z +0.579

## The bias the whole-sequence gyro runs learnt (deg/s, the IMU's axes)

- seq01: median x +0.413, y +0.979, z -0.180 over 16 seeds; fitted to the reference: x +0.093, y +1.296, z +0.054
- seq07: median x -0.184, y +0.078, z +0.673 over 16 seeds; fitted to the reference: x -0.273, y +0.379, z +0.172

Re-run (the output folder must not exist or be empty):

```bash
python -m uwslam.aqualoc_score <prepared seq01> <prepared seq07> --output <fresh folder>
```
