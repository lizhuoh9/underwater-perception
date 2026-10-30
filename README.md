# uwslam

Monocular-camera SLAM for a small underwater robot: one camera, no IMU or rangers. A C core that gives the same results
bit for bit on a PC and on an ESP32-P4 turns the robot camera's grey frames into the camera's motion and a continuing
3D map, turns the segmentation model's water mask into the directions of the view that are blocked, and from both
plans where to head next. Everything is checked on a lake clip against a 3D reconstruction of it, and on an ESP32-P4 by
streaming the clip to the board over USB.

[中文说明](README.zh-CN.md)

```text
camera frame, 320×240 grey
├── core/mono_vo        corners, Lucas-Kanade tracking, the camera's motion, triangulated sparse points
│   └── core/voxel_map      the points accumulated in a 3D voxel map that goes on through tracking losses ──┐
│                                                                                                         ├─ core/path_plan
└── segmentation model (ESP-DL, outside this repository): 60×80 water / not-water mask                    │  where to head next:
    └── core/obstacle_dir   which directions of the view are blocked, and the free one nearest ahead ─────┘  avoidance in 3D
```

Two older maps stay beside them: [core/mono_map](core/mono_map.h), a top-view obstacle slice, and
[core/mono_cloud](core/mono_cloud.h), the tracked features' sparse points. Both freeze at the first tracking loss.

## Findings

All numbers come from the lake clip: the first 33.8 s of an underwater video filmed along a wall in a lake, at 30 FPS.
The reference is a WaterSplat reconstruction of it: 88 camera poses (their stamps times 3, one video frame later, see
[mask_motion](results/lake_clip/mask_motion/RESULTS.md)) and its point cloud, taken as metres. The water masks are the
segmentation model's output on the robot camera's 160×120 frames. Medians are over 16 RANSAC seeds, because on this
clip the seed alone can change an error twofold. Comparisons with earlier settings below were scored before the
reference was moved by that frame, which changed the scores by under a millimetre in position, under a degree in median
heading and about a point in the plan's shares.

1. **The odometry tracks the clip at the board's rate.** The C port's camera path, after one similarity alignment, is
   0.33 m RMSE at 10 Hz, 0.19 m at 5 Hz and 0.38 m at 3 Hz over the 8.9 m the camera swims. The OpenCV prototype it
   was ported from, which keeps a pinhole camera, gives 0.41, 0.44 and 0.33 m on one seed. Tracking is lost on 2, 13
   and 17 steps. The monocular scale drifts: the largest over the smallest 5 s window ratio is 5.3×, 1.8× and 4.1×.
   The C tools' runs take the camera's lens, whose distortion is not centred in the frame: without it every step
   pitched up a little.
2. **The 3D map goes on for the whole clip.** The voxel map holds what the robot needs around it: every 5 s, the
   obstacles it holds lie a median 5.5 cm (10 Hz) and 6.7 cm (5 Hz) from the reconstruction's surfaces, and 85% and
   76% of them lie within 0.2 m. That is with the last 5 s of the path aligned to the reference. With one alignment
   for the whole drifting path, the obstacles held at the end lie a median 0.05 m and 0.07 m from them. It goes on
   through every tracking loss. The older maps, frozen at the first loss, stop after 4.2–4.4 s of the 33.8 s.
3. **The water mask tells the blocked directions reliably.** When the reconstruction has a surface within 0.5, 1 or
   2 m filling half of one of the 15 directions, the mask blocks that direction too, in at least 99% of the cases in
   every band. At 5 Hz the direction it picks as free really is free 100%, 100% and 99% of the time at those ranges.
   Straight ahead is free only 87%, 61% and 20% of the time, and a random direction 64%, 40% and 36%. The mask also
   blocks the water surface above, which the reconstruction lacks.
4. **The planner heads into free water and knows how far it is.** Going, at 5 Hz, it heads into a corridor free beyond
   0.5 m in 97% of its steps and beyond 1 m in 93%, never into one blocked within 0.25 m, where straight ahead is free
   beyond 1 m in only 51%. Where the mask sees no free direction (3–4% of the steps, facing the wall) it creeps into the
   free water the map knows instead of having no answer. Its ranges, counted in steps of the robot's own speed, agree
   with the reference's within a factor of 2 for 86% (5 Hz) and 82% (10 Hz) of the directions the map knows. Where the
   mask sees water the map holds no points, so the plan bounds those directions by the surfaces in view instead: free
   for a median 9.5 steps at 5 Hz (15.3 at 10 Hz), and the reference was nearer than that bound in at most 1% of the
   water directions.
5. **The ESP32-P4 gives the PC's results, at 5 Hz.** With the clip streamed at 5 and 10 Hz, the board gives every
   odometry step, the voxel map, the directions, the plan and the obstacle map exactly as the PC does. At 360 MHz on
   two cores a 5 Hz step takes 127 ms (median 132, at most 155); the voxel map adds 0.48 ms, the directions 0.18 ms and
   the plan 1.78 ms, within the 200 ms between frames. At 10 Hz a step takes 102 ms (median 99, at most 163), and 158
   of the 338 steps take longer than the 100 ms between frames. 107,928 B of internal RAM stay free.

## Repository layout

```text
.
├── core/                       C99, no heap, no libm
│   ├── mono_vo*.c, mono_vo.h   monocular visual odometry (five units, see mono_vo_internal.h)
│   ├── voxel_map.c/.h          the continuing 3D voxel map
│   ├── obstacle_dir.c/.h       blocked directions from the water mask
│   ├── path_plan.c/.h          where to head next: local avoidance and 3D path choice
│   ├── mono_map.c/.h           top-view obstacle slice (frozen at the first loss)
│   └── mono_cloud.c/.h         sparse points of the tracked features (frozen at the first loss)
├── firmware/mono_vo_stream/    ESP-IDF app for the ESP32-P4: runs the core on streamed frames and masks, reports hashes and times
├── src/uwslam/                 Python package
│   ├── mvo.py                  ctypes binding of the core; the odometry on the lake clip against the OpenCV prototype
│   ├── voxel_map.py            the voxel map on the lake clip, scored against the reconstruction
│   ├── obstacle_dir.py         the directions on the lake clip, scored against the reconstruction's depth
│   ├── path_plan.py            the plan on the lake clip, scored against the reconstruction's free corridors
│   ├── mono_map.py, mono_cloud.py   the older maps
│   ├── mvo_stream.py           streams frames and masks to the board and checks every step against the PC
│   └── experiments/            the OpenCV prototype (mono_vo), the reference poses and masks (mask_motion), the
│                               segmentation model runner (segmenter)
├── tests/                      unittest suite; tests/c/ holds the C contract checks and firmware host harnesses
└── results/lake_clip/          results, one folder per tool, each with RESULTS.md; the board's expectations and records
```

## Getting started

Needs Python 3.10 or newer and gcc on the PATH (MinGW-w64 on Windows): the C core is compiled into a shared library on
the fly. Run everything from the repository root.

```bash
pip install -e ".[experiments]"             # numpy, Pillow, OpenCV (torch and esp-ppq only for the segmenter)
python -m unittest discover -s tests -t .
```

The lake clip and its reconstruction are not part of the repository. The tools take the video and a reconstruction
folder holding `images.txt` (camera poses, TUM format) and `images.ply` (point cloud). Every tool refuses to overwrite
an existing output; choose a fresh folder.

```bash
python -m uwslam.mvo compare <video> --reconstruction <folder> --output results/lake_clip/mono_vo_c_2 --board-frames <frames file>
python -m uwslam.voxel_map <video> --reconstruction <folder> --output results/lake_clip/voxel_map_2
python -m uwslam.obstacle_dir <video> --reconstruction <folder> --output results/lake_clip/obstacle_dir_2
python -m uwslam.path_plan <video> --reconstruction <folder> --output results/lake_clip/path_plan_2
python -m uwslam.mono_map <video> --reconstruction <folder> --output results/lake_clip/mono_map_2 --board-masks <masks file>
```

The water masks come from [mask_motion](src/uwslam/experiments/mask_motion.py), which runs an exported ESP-DL int8
segmentation model on the clip ([segmenter.py](src/uwslam/experiments/segmenter.py) describes the model folder).

The Python code is formatted and linted with ruff, configured in [pyproject.toml](pyproject.toml):
`ruff format src tests` and `ruff check src tests`.

## Monocular odometry

[core/mono_vo.h](core/mono_vo.h) tracks up to 300 Shi-Tomasi corners in 19×19 windows with a fixed-point pyramidal
Lucas-Kanade and a forward-backward check. When the tracks fit a pure turn (a homography within 1 px), the step is that
rotation. Otherwise an eight-point MSAC essential matrix, a search over travel directions and a robust
Levenberg-Marquardt give the motion, unless a separately fitted rotation explains the tracks as well. A fast turn adds
a pull towards the homography's turn. Translation is accepted only when the evidence favours it. The step length comes
from the depth ratios of the points two steps share, counting those the new pair puts within 60 baselines, and is kept
within three times the recent speed. A step with too few inliers is LOST: it keeps the position, takes the turn it can
measure and, where its depths agree with the earlier points', carries the scale on. The C port follows the OpenCV
prototype ([experiments/mono_vo.py](src/uwslam/experiments/mono_vo.py)) in design, not in numbers; the comparison of
the two is [results/lake_clip/mono_vo_c/RESULTS.md](results/lake_clip/mono_vo_c/RESULTS.md), the prototype's own run
[results/lake_clip/mono_vo/RESULTS.md](results/lake_clip/mono_vo/RESULTS.md).

| Rate | Position RMSE after alignment | Heading RMSE | Scale spread over 5 s windows | Lost steps |
|---:|---:|---:|---:|---:|
| 10 Hz | 0.33 m [0.26–0.37] | 39.5° [30–46] | 5.3× [2.3–9.2] | 2 [1–4] |
| 5 Hz | 0.19 m [0.16–0.26] | 19.4° [5–26] | 1.8× [1.4–2.6] | 13 [11–15] |
| 3 Hz | 0.38 m [0.34–0.56] | 39.1° [26–66] | 4.1× [1.6–8.1] | 17 [16–19] |

Median [smallest–largest] over 16 seeds. A monocular run has no metric scale: the alignment supplies it.

The settings favour 5 Hz, the board's input rate, and spend the P4's time on it. With the pinhole camera of the time,
300 features instead of 250 brought the 5 Hz path from 0.33 to 0.26 m in the median of 16 seeds and from 0.079 to
0.067 m over 10 s windows of the clip (lower in 68% of 192 paired runs), the heading unchanged; at 10 Hz it was worse.
The other ways tried to spend that time on accuracy did not help: 640×480 frames, 10 Hz input, more RANSAC samples,
refinement or tracking iterations. With the lake's lens, 19×19 windows instead of 15×15 brought the heading over the
5 Hz windows from 8.1° to 6.2° in the median (lower in 59% of the paired runs) and its largest error from 20.1° to
15.3°, the position staying about the same (0.066 against 0.067 m, lower in 57%); at 3 and 10 Hz the heading was worse.
With them, 350 or 400 features were worse at 5 Hz (full clip 0.23 and 0.24 m against 0.21). The pull towards the
homography's turn mends the fast swings at 5 Hz, where the essential matrix took up to a quarter of each fast step as
travel (14° over the clip's two-second swing); at 10 Hz the steps are half as large, the pull mends a swing's first
half more than its second, and the heading over the whole clip comes out worse than without it.

A step LOST on a fast turn keeps a turn when at least 8 tracks and half of them agree on one (at 5 Hz the lost steps
turn by 11° in the median, all of which keeping the last pose lost from the heading): the separately fitted rotation's,
and where only the homography has that support, the essential pose's when it had 15 inliers, else the homography's. A
homography the fitted rotation does not share has mostly read the travel's parallax as a turn, over the lake's floor as
pitch: on such lost steps at 5 Hz it errs +1.5° in pitch a step against −0.2° for the essential pose, though by as much
in all (6.0° and 6.2°), and that bias made the path climb. Lost steps with no supported rotation still keep none: on a
still camera with noisy tracks the essential pose's rotation would add turns that are not there.

Translation is refused where the tracks give no evidence for it over a rotation, and a LOST step takes none. What such
steps cost was the scale: a lost step dropped the triangulated points that carry the step length from one step to the
next, so each time the next step restarted at the recent speed. A lost step whose essential pose had 15 inliers
therefore carries its points on with that pose, still taking no translation, when at least 8 of the depth ratios it
shares with the earlier points agree on one scale (their interquartile range at most half their median). A still
camera's noise fits poses whose depth ratios scatter, so it nearly always drops its points.

The lake camera's distortion is not centred in height: to first order it is a tangential term p1 of −0.02 in OpenCV's
model, as a frame cropped off the lens's axis has, and without it every step pitched up a little. It was found without
the reference, on the core's own tracks: with that term, the chain of 30 FPS steps agrees in pitch with direct
rotations across 1.6 s of the same tracks, and the floating dock's straight bottom edge keeps its vanishing point
through the clip. The C tools' lake runs therefore take it (`LENS` in
[experiments/mono_vo.py](src/uwslam/experiments/mono_vo.py); the OpenCV prototype keeps the pinhole).

What is left at 5 Hz is mostly scale, not rotation: rebuilt from a run's own steps (16 seeds, before the 60-baseline
limit below), the full clip's path came out 0.200 m off with the reference's attitude and 0.140 m with its step lengths,
against 0.214 m as it was. The scale itself had a bias. The step length is the median ratio of each shared point's
earlier depth to its depth in the new pair, and the points counted were those the new pair put within 50 baselines
(`MVO_MAX_DEPTH`). A ratio grows with the new pair's parallax, so cutting at the pair's own noisy depths dropped the
points whose noise put them far and, on short steps, where many sit near that limit, biased the median. The points
counted are now those the new pair puts within 60 baselines (`MVO_RATIO_DEPTH` in [core/mono_vo.h](core/mono_vo.h)),
against the 50 before (16 seeds):

| | Within 50 baselines | Within 60 baselines |
|---|---:|---:|
| 5 Hz, full clip: position RMSE / scale spread | 0.214 m [0.148–0.275] / 2.0× | 0.190 m [0.157–0.257] / 1.8× |
| 10 Hz, full clip: position RMSE / scale spread | 0.394 m [0.359–0.434] / 8.3× | 0.327 m [0.257–0.366] / 5.3× |
| 5 Hz, 10 s windows: position RMSE, median | 0.066 m | 0.064 m (lower in 66%) |
| 10 Hz, 10 s windows: position RMSE, median | 0.088 m | 0.076 m (lower in 47%) |
| 5 Hz voxel map: local / global median distance | 0.065 / 0.072 m | 0.068 / 0.070 m |
| 10 Hz voxel map: local / global median distance | 0.057 / 0.055 m | 0.055 / 0.045 m |
| 5 Hz plan: the chosen direction's known range within twice the reference's | 71% | 81% |

The headings and the lost steps do not change. The plan row rests on few steps: the chosen direction had a known range
in 12 of the 169 steps on the default seed (14 before). The limit of 60 was chosen from a scan of 50, 60, 75 and 100
baselines and none on this one clip; it should be checked again on the robot's own footage. A wider limit gave the
5 Hz path more (0.173 m at 100 baselines) for a worse local map (0.076 m) and worse 10 Hz windows (0.105 m); with none,
the points too noisy to help made 10 Hz worse.

Tried on the scale and dropped: a second decentring term of the lens (p2 of −0.02 or −0.03; the chained 30 FPS steps'
yaw against direct estimates shrinks to about zero at −0.03) made the full clip's heading better (19.0° to 13.4° and
12.7°) but not the windows', and the scale worse: the 5 Hz full clip came out 0.34–0.38 m instead of 0.21. A
sliding-window fit of the last five frames (their poses, positions or step lengths, with the points tracked through
them), prototyped on the PC, let the scale collapse: at 5 Hz 49% of the tracks survive a step and about a third two,
and over tracks that short the window's scale drifted more than the odometry's chain; at 10 Hz, where tracks live
longer, it came out 1.0–1.3 m against 0.39 (4 seeds). Tracking lost features again from their neighbours'
displacement cut the lost steps from 13 to 11 but put the 5 Hz full clip at 0.38 m (started from one guess, both
directions of the forward-backward check agree on wrong tracks too), also with new corners kept 12 px from the border.
A pull of each step length towards the last gave 0.24–0.37 m (the diver's speed changes, and the ratios carry it), the
length that best reprojects the shared points 0.55 m, and a median weighted by the inverse of the earlier depth 0.24 m.
The tracks are short mostly because new corners die: just detected, they survive a step in 29% of the cases; tracked
through four steps or more, in 80%.

The camera is `mvo_camera_t`: the 320×240 frame's pinhole intrinsics and its lens distortion in OpenCV's model and
order, k1, k2, p1, p2 and k3, as `cv2.calibrateCamera` gives them for a calibration at 320×240. Calibrate the robot's
camera under water, behind its port: the water and a flat port change the lens. Before any model sees a step's pairs,
the core undistorts them by Newton steps, which agree with OpenCV's undistortion to within 0.001 px in the tests. The
corners, the tracking and the masks stay in the raw image, and the obstacle directions take each mask column's and
row's undistorted ray. With all five zero nothing changes: the core is bit-identical to a pinhole core. The lake clip's
lens was never calibrated; the C tools' runs on it take the one term the clip itself shows, p1 = −0.02 (above), and the
host sends it to the board with the rest of the camera. The tests render a moving camera through a strong barrel lens:
told the lens, the odometry keeps a pinhole's accuracy; taking the same frames for a pinhole's, it errs far more in
turn and in direction of travel. From Python: `mvo.Camera(fx, fy, cx, cy, k1, k2, p1, p2, k3)` (`mvo.clip_camera()`
is the lake clip's), or `python -m uwslam.mono_cloud <video> --fx … --k1 … --k3 …`; the board's `start` line carries
the five at its end, after the seed, cores and simd.

## Continuing voxel map

[core/voxel_map.h](core/voxel_map.h) files, after every odometry step, the well-triangulated tracked points that the
mask marks as not water, voxel by voxel, in the odometry's frame. It goes on through LOST steps, since the odometry
keeps its frame through them. The monocular unit drifts, so the voxel side follows the scene: a sixteenth of the median
depth. It is reset, and the voxels re-keyed and merged, when the scene asks for twice or half the current side. A voxel
seen in two frames is an obstacle; unobserved space is unknown, not free. The map holds at most 4096 voxels (56 KB).
When it is full, far voxels make room first, then voxels seen only once, then the obstacles unseen longest. At 5 Hz on
the lake clip that keeps the last 5.7 s of obstacles (median; 90% within 12.8 s).

| Rate | Local: median distance | Local: within 0.1 / 0.2 m | Global: median distance | Obstacles held | LOST steps gone through |
|---:|---:|---:|---:|---:|---:|
| 10 Hz | 0.055 m [0.046–0.063] | 68% / 85% | 0.05 m [0.04–0.08] | 2078 | 2 |
| 5 Hz | 0.067 m [0.061–0.085] | 60% / 76% | 0.07 m [0.05–0.11] | 1068 | 13 |

Local: every 5 s, the obstacles the robot holds, with the last 5 s of its path aligned to the reference (what avoidance
uses). Global: the obstacles held at the end, with one alignment of the whole path. Details and figures:
[results/lake_clip/voxel_map/RESULTS.md](results/lake_clip/voxel_map/RESULTS.md).

![The obstacles the robot held over the clip, seen from above and from the side](results/lake_clip/voxel_map/map_5hz.png)

The voxels are this fine, and this many, because the P4 has the memory. Against voxels twice as large and four times
fewer, scored on the odometry of 15×15 windows, the obstacles held at 5 Hz lay 0.074 m from the reference's surface
over the whole path instead of 0.106, and 0.064 m near the camera instead of 0.067 (at 10 Hz 0.055 and 0.050 m instead
of 0.075 and 0.060); a larger map of the larger voxels had not helped. Single sightings land in different fine voxels
more often, so the planner reads them two to a side (below).

The extra features triangulate less well, so the map's points lie a little farther from the reference: with the
pinhole camera of the time, 300 features put the 5 Hz map's local error at 7.1 cm instead of 6.1. Filing only points
from rays at least 1.5° apart instead of 1° took that back but halved how often the range straight ahead was known (34%
of the 5 Hz steps instead of 71%), because forward motion gives the least parallax there: driving straight at a wall in
the synthetic test, it knew that range in none of 11 steps instead of 7, and with 1.25° in none either. The maps
therefore file from 1°.

## Obstacle directions

[core/obstacle_dir.h](core/obstacle_dir.h) splits each frame's 60×80 mask into 3 × 5 directions of the camera's view:
bands up, level and down at ±15°, and sectors at ±15° and ±40° from the optical axis. A direction is blocked as soon as
half of it is not water, and freed only after two frames below 30%. The choice is the free direction nearest straight
ahead: fewest steps first, then turning before rising or diving, then the more open one. It is scored against the
reconstruction seen through the same grid from each frame's reference camera.
[results/lake_clip/obstacle_dir/RESULTS.md](results/lake_clip/obstacle_dir/RESULTS.md) has the tables per band and
range.

![Example frames with their directions, masks and reference depth](results/lake_clip/obstacle_dir/directions_5hz.png)

The mask gives direction, not range, and "up" is the image's up, not gravity's. The voxel map gives the ranges, and the
planner joins the two.

## Avoidance and path choice

[core/path_plan.h](core/path_plan.h) decides after every step where to head, among the same 3 × 5 directions, towards a
goal direction (straight ahead unless the caller gives one in the odometry's world). A single camera fixes no scale, so
ranges count in steps: the odometry's recent speed is one step, and a range of n steps is n frames away at that speed.
A direction's range is the distance to the second nearest obstacle in it (one alone is too easily an outlier). A
direction is passable when the mask shows it free and the map puts no obstacle within 8 steps (1.6 s at 5 Hz) there.
The map adds vetoes but lifts none: it holds only textured points, and where it holds none the mask may still see a
surface. Letting a far map range reopen a direction the mask blocked sent the plan into corridors blocked within 0.25 m
in 2–4% of its steps; without it, in none. The plan heads for the passable direction nearest the goal, turning before
rising or diving. With none passable it creeps into the longest free water the map knows (at least 3 steps), else turns
in place away from the side where the map holds more obstacles, and with nothing known it stops.

It is scored open-loop (the camera follows the diver, not the plan) against each direction's free corridor in the
reconstruction: the 10th percentile range over the central half of the direction's cells.

| 5 Hz, 16 seeds | Free beyond 0.25 m | Free beyond 0.5 m | Free beyond 1 m |
|---|---:|---:|---:|
| The plan, going (96% of the steps) | 100% | 97% | 93% |
| The plan, creeping (4% of the steps, where the mask sees no free direction) | 100% | 83% | 75% |
| Straight ahead | 92% | 86% | 51% |
| The best direction, in hindsight | 100% | 100% | 100% |

While it goes, its corridors are as free as those of the mask's own choice (100%, 97% and 93%). On the default seed
the map vetoed a direction the mask saw as water in 5 steps at 5 Hz and in 3 at 10 Hz (once, at 10 Hz, the mask's own
choice), and the reference had a surface within 8 steps for none of those vetoes. That is no failure of the map: the
reference had no surface within 8 steps in any direction the mask saw as water, so there was nothing to veto. The map
also holds almost no points there: the voxel map files none on water cells, and open water has no texture to track.
The few vetoes it cast in water directions came from stray points.

So the plan bounds the water directions instead. Anything nearer than the visibility along a water ray would itself be
seen, and every surface in view is nearer than the visibility, so a water direction is free at least as far as the
surfaces the camera sees now. `clear` is 0.75 of the median range of the voxels filed in this frame, when there are at
least 10. Over 16 seeds it was known for 82% (5 Hz) and 92% (10 Hz) of the water directions, a median 9.5 and 15.3
steps (beyond the 8-step margin in 70% and 98%), and the reference was nearer than it in at most 1% at 5 Hz and never at
10 Hz. In a probe, every such direction still had 14 steps or more of free water; the full median as the bound missed
in 1.4%. What the map adds is an answer where the mask has none, and the ranges, the bound included: a robot can slow
down as the steps to the nearest obstacle run out.

The plan reads the fine voxels two to a side (`PP_GROUP`), adding their frames, so that two voxels seen once each, even
in the same frame, make their cell an obstacle. Read one by one, with 15×15 windows, the finer voxels left the range
straight ahead known in 68% of the 5 Hz steps instead of 77%; grouped, in 83%, with as much free water ahead and 87% of
the known ranges instead of 90% within twice the reference's. As built, the range straight ahead is known in 83% of the
5 Hz steps and 52% of the 10 Hz ones. [results/lake_clip/path_plan/RESULTS.md](results/lake_clip/path_plan/RESULTS.md)
has 10 Hz too.

![Example frames with the plan: green passable, numbers the map's range in steps, yellow the plan](results/lake_clip/path_plan/plan_frames_5hz.png)

## ESP32-P4

[firmware/mono_vo_stream](firmware/mono_vo_stream) runs the odometry, the voxel map, the obstacle directions, the
planner and the two older maps on an ESP32-P4 (two RISC-V cores at 360 MHz, PSRAM not used), on frames and masks
streamed from the PC, and answers every frame with its step's hash, map counts and stage times.
[mvo_stream.py](src/uwslam/mvo_stream.py) checks all of it against the PC's expectations. The board names its build,
`p4`, and the sources it was built from in its `ready` line; the host refuses a board built from other sources, and
expectations made by another build.

The link is the console. By default ([sdkconfig.defaults](firmware/mono_vo_stream/sdkconfig.defaults)) that is UART0,
which P4 boards bring out on a USB-UART bridge; the firmware switches it to 921,600 baud once it runs (`--baud` on the
host). At 2 Mbaud a CH340 bridge lost single bytes of the board's replies in two of three lake runs, so the host also
refuses a damaged step line. For the chip's own USB-Serial-JTAG port, add
[sdkconfig.usb_serial_jtag](firmware/mono_vo_stream/sdkconfig.usb_serial_jtag) to the defaults, in a build folder of
its own; only the console changes.

```bash
# in an ESP-IDF 5.3+ shell; the console on a USB-UART bridge:
idf.py -C firmware/mono_vo_stream -B firmware/mono_vo_stream/build -D SDKCONFIG=build/sdkconfig -p <PORT> flash
# or on the chip's USB-Serial-JTAG port:
idf.py -C firmware/mono_vo_stream -B firmware/mono_vo_stream/build_usj -D SDKCONFIG=build_usj/sdkconfig \
    -D "SDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.usb_serial_jtag" -p <PORT> flash

python -m uwslam.mvo_stream --port <PORT> --frames <frames file> --masks <masks file> \
    --expected results/lake_clip/mono_vo_c/board_expected_5hz.json \
    --expected-map results/lake_clip/mono_map/board_expected_map_5hz.json \
    --expected-voxels results/lake_clip/voxel_map/board_expected_voxels_5hz.json \
    --expected-dirs results/lake_clip/obstacle_dir/board_expected_dirs_5hz.json \
    --expected-plan results/lake_clip/path_plan/board_expected_plan_5hz.json --label <name>
```

The frames file is what `uwslam.mvo compare --board-frames` writes, the masks file what `uwslam.mono_map --board-masks`
writes; with the 5 Hz expectations the board gets every second frame. The host needs only pyserial
(`pip install -e ".[board]"`, or ESP-IDF's own Python). Before every stream it has the board check its vector kernels
against the portable C ones on 2,000 random windows (`selftest`), and it refuses a board where any result differs.

[mvo_vector.S](firmware/mono_vo_stream/main/mvo_vector.S) holds the tracker's window kernels in the P4's vector
instructions: a row of the 19×19 window takes three vectors of eight 16-bit lanes, and the kernels give exactly the
integers of the C ones. A call costs 19.6 µs (a window) and 6.5 µs (its sums), against 118 and 36 µs for the C kernels
on the P4; the block copy stays in C (8 µs a call).

Measured with the lake clip's lens over the USB-Serial-JTAG port (records `p4_360mhz_usj*` in
[results/lake_clip/mono_vo_c/board](results/lake_clip/mono_vo_c/board)):

| | 5 Hz (169 steps) | 10 Hz (338 steps) |
|---|---:|---:|
| Steps, voxel map, directions, plans and obstacle map equal to the PC's | all | all |
| Odometry per step, mean / median / max | 127 / 132 / 155 ms | 102 / 99 / 163 ms |
| Tracking / motion / detection, mean | 86 / 26 / 16 ms | 66 / 28 / 8 ms |
| Voxel map update, mean / max | 0.48 / 2.6 ms | 0.55 / 4.1 ms |
| Obstacle directions, mean | 0.18 ms | 0.18 ms |
| Plan, mean / max | 1.78 / 3.0 ms | 1.83 / 2.9 ms |
| Internal RAM free at the end | 107,928 B | 107,928 B |

A 5 Hz step stays within the 200 ms between frames; at 10 Hz 158 of the 338 steps take longer than the 100 ms between
frames. Everything lives in internal RAM: the two image pyramids (about 100 KB each), the odometry's state and the
voxel map (56 KB). These are the board's compute times: the replay is paced by the link (over the USB-Serial-JTAG port
the 5 Hz clip's 170 frames took 38 s), and the segmentation model is not in them.

## Limitations

- One clip, filmed by a diver rather than the robot, with the reference's own errors. The camera's 110° field of view
  was fitted on the same clip, which pins it only to about ten degrees (re-timed, the fit prefers a narrower view; in
  the C odometry narrower views won heading at 5 and 10 Hz on most measures but lost position at 5 Hz), and its lens
  was never calibrated: the C tools' lake runs take only the decentring term the clip itself shows (p1 = −0.02).
  Settings such as the 60 baselines of the step length's points were chosen on this clip alone.
- A single camera: no metric scale, and the odometry drifts, in scale above all. The map closes no loops and does not
  relocalise after a loss.
- The mask's directions are the camera's, not gravity-aligned; the mask cannot tell range.
- The planner is local and scored open-loop on a clip the robot did not steer: it has not driven a robot yet.
- The board check streams recorded frames and masks over a serial link; it does not include capture, the segmentation
  model or the robot's other tasks.
