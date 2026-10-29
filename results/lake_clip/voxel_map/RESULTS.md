# The continuing voxel map from the monocular odometry (lake clip)

`core/voxel_map.c` files the odometry's well-triangulated tracked points on cells the segmentation marks as not water, voxel by voxel, after every step and through LOST steps (the map goes on in the odometry's frame; it is not frozen at the first loss). The voxel side follows the scene: the median depth over 8, reset when the scene asks for twice or half of it (the monocular unit drifts tenfold over the clip). A voxel seen in two frames is an obstacle. It holds at most 1024 voxels (14 KB): far voxels, then single sightings, then the longest unseen obstacles make room.

Scores, after similarity alignments of the camera path (the map's unit is the odometry's): **global**, the obstacles held at the end with one alignment of the whole path (drift included); **local**, every 5 s the obstacles held at that moment with an alignment of the last 5 s of the path, what avoidance uses. Distances are to the reconstruction's nearest cloud point (thinned to 3 cm). Medians and ranges over 16 RANSAC seeds.

| Rate | Local: median distance | Local: within 0.1 / 0.2 m | Global: median distance | Global: within 0.2 m | Obstacles held at the end | Their age, median / 90% | Voxel side at the end | Side resets | LOST steps gone through | A map frozen at the first LOST would cover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 Hz | 0.058 m [0.045–0.063] | 69% [64%–79%] / 85% [81%–91%] | 0.09 m [0.04–0.18] | 82% [54%–100%] | 867 [770–905] | 3.0 s [2.8–3.8] / 5.7 s [5.1–6.5] | 0.026 m [0.011–0.045] | 5 [3–8] | 2 [1–3] | 4.3 s [4.0–4.4] of 33.8 s |
| 5 Hz | 0.065 m [0.056–0.119] | 62% [46%–67%] / 79% [63%–82%] | 0.15 m [0.05–0.24] | 64% [44%–93%] | 704 [529–767] | 4.2 s [3.4–5.0] / 8.6 s [6.6–9.4] | 0.023 m [0.019–0.035] | 4 [3–8] | 12 [9–13] | 4.4 s [4.2–4.4] of 33.8 s |

The default seed's run is drawn in `map_<rate>hz.png`. Its board expectation (`board_expected_voxels_<rate>hz.json`) holds every step's voxel count and filed observations and the final map's hash for `uwslam.mvo_stream --expected-voxels`.

Limits: one clip; the reference's own errors; positions drift with the odometry (no loop closure); unobserved space is unknown, not free.

Re-run (the output folder must not exist or be empty):

```bash
python -m uwslam.voxel_map <video> --reconstruction <folder with images.txt and images.ply> --output results/lake_clip/voxel_map_2
```
