# Obstacle map from the monocular odometry and the water masks (lake clip)

Camera only: `core/mono_vo*.c` gives the motion (up to scale), `core/mono_map.c` files the tracked points that the segmentation marks as not water in a top-view grid. Scored after one similarity alignment of the camera path to the reference: the distance from every obstacle observation to the reconstruction's nearest cloud point (thinned to 3 cm). Each row of the first table is one run with one RANSAC seed; on this clip the seed alone can change these distances by a factor of two or more, so the second table gives the median and range over 16 seeds. A LOST step freezes the map; scores use only the preceding continuous prefix and are unavailable when its 3D alignment is underdetermined. These scores do not establish continuity or accuracy after loss. Different seeds can have different continuous prefix lengths. The alignment spread ratio is the second divided by the largest singular value of the centred cross-covariance; values near zero indicate weak alignment constraints, often from a nearly straight path. A larger ratio does not guarantee accuracy. The ratio in the first table is for that run's seed only.

| Rate | Observations (frames) | Distance to cloud, median | Within 0.1 / 0.2 / 0.3 m | Path RMSE after alignment | Continuous frames / total | Alignment spread ratio | Grid cell (odometry units) | Occupied / unknown cells | Slice height / resets |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 Hz | 4540 (43) | 0.24 m | 12% / 38% / 72% | 0.13 m | 44 / 339 | 0.014 | 1.43 | 58 / 16326 | -2.74 / 2 |
| 5 Hz | 1985 (20) | 0.20 m | 14% / 50% / 82% | 0.08 m | 21 / 170 | 0.019 | 1.01 | 201 / 16183 | -0.92 / 1 |

Over 16 RANSAC seeds, median [smallest–largest]:

| Rate | Distance to cloud, median of the run | Path RMSE after alignment |
|---:|---:|---:|
| 10 Hz | 0.21 m [0.12–0.25] (16/16 seeds) | 0.10 m [0.09–0.32] (16/16 seeds) |
| 5 Hz | 0.20 m [0.15–0.27] (16/16 seeds) | 0.08 m [0.04–0.10] (16/16 seeds) |

The grid's cell is an eighth of the scene's median depth when the map starts; it stays in odometry units, whose length in metres drifts with the odometry's scale, so the grid has no fixed metric size. The grid takes a slice within half a scene depth of its fixed height; the grid resets when the estimated camera height moves one cell from it. Unobserved cells remain unknown: a ray at one height cannot prove the whole slice is free. On this clip most observations are the lake bed below the robot, so the slice is sparse.

Top views: `map_10hz.png`, `map_5hz.png` (left: observations over the cloud; right: the grid the robot keeps, in its own units).

Re-run (the output folder must not exist or be empty):

```bash
python -m uwslam.mono_map <video> --reconstruction <folder with images.txt and images.ply> --output results/lake_clip/mono_map_2
```
