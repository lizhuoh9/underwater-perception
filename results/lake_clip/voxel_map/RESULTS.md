# The continuing voxel map from the monocular odometry (lake clip)

`core/voxel_map.c` files the odometry's well-triangulated tracked points on cells the segmentation marks as not water, voxel by voxel, after every step and through LOST steps (the map goes on in the odometry's frame; it is not frozen at the first loss). The voxel side follows the scene: the median depth over 16, reset when the scene asks for twice or half of it (the monocular unit drifts tenfold over the clip). A voxel seen in two frames is an obstacle. It holds at most 4096 voxels (56 KB): far voxels, then single sightings, then the longest unseen obstacles make room.

Scores, after similarity alignments of the camera path (the map's unit is the odometry's): **global**, the obstacles held at the end with one alignment of the whole path (drift included); **local**, every 5 s the obstacles held at that moment with an alignment of the last 5 s of the path, what avoidance uses. Distances are to the reconstruction's nearest cloud point (thinned to 3 cm). Medians and ranges over 16 RANSAC seeds.

| Rate | Local: median distance | Local: within 0.1 / 0.2 m | Global: median distance | Global: within 0.2 m | Obstacles held at the end | Their age, median / 90% | Voxel side at the end | Side resets | LOST steps gone through | A map frozen at the first LOST would cover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 Hz | 0.055 m [0.046–0.063] | 68% [62%–75%] / 85% [81%–90%] | 0.05 m [0.04–0.08] | 100% [84%–100%] | 2078 [1359–2621] | 4.6 s [3.1–5.4] / 8.2 s [5.8–9.5] | 0.003 m [0.001–0.013] | 4 [2–7] | 2 [1–4] | 4.2 s [4.0–4.4] of 33.8 s |
| 5 Hz | 0.068 m [0.061–0.087] | 60% [53%–63%] / 76% [65%–82%] | 0.07 m [0.05–0.11] | 86% [76%–97%] | 1068 [795–1278] | 5.7 s [4.4–6.8] / 12.8 s [7.2–13.8] | 0.014 m [0.010–0.016] | 6 [4–11] | 13 [11–15] | 4.4 s [4.2–4.4] of 33.8 s |

The default seed's run is drawn in `map_<rate>hz.png`. Its board expectation (`board_expected_voxels_<rate>hz.json`) holds every step's voxel count and filed observations and the final map's hash for `uwslam.mvo_stream --expected-voxels`.

Limits: one clip; the reference's own errors; positions drift with the odometry (no loop closure); unobserved space is unknown, not free.

Re-run (the output folder must not exist or be empty):

```bash
python -m uwslam.voxel_map <video> --reconstruction <folder with images.txt and images.ply> --output results/lake_clip/voxel_map_2
```
