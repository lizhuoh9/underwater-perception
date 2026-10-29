# Local avoidance and 3D path choice (lake clip)

`core/path_plan.c` chooses, after every odometry step, which of the camera's 3 × 5 directions to head for (towards straight ahead): passable means water in the mask and no obstacle in the voxel map within 8 steps (1.6 s at 5 Hz); the map adds vetoes but lifts none. With none passable it creeps into the longest free water the map knows, else turns in place. Ranges count in steps of the odometry's recent speed, so they need no metric scale. Scored open-loop against the reconstruction: a direction's free corridor is the 10th percentile range over the central half of its cells; the camera follows the diver, not the plan. Medians and ranges over 16 RANSAC seeds.

## 10 Hz (338 steps)

| Heading for | Free beyond 0.25 m | Free beyond 0.5 m | Free beyond 1 m |
|---|---:|---:|---:|
| the plan, going | 100% [100%–100%] | 97% [97%–97%] | 91% [91%–91%] |
| the plan, creeping | 100% [91%–100%] | 73% [45%–91%] | 73% [45%–91%] |
| the mask's own choice (when it has one) | 100% [100%–100%] | 97% [97%–97%] | 91% [91%–91%] |
| straight ahead | 92% [92%–92%] | 84% [84%–84%] | 51% [51%–51%] |
| the best direction (hindsight) | 100% [100%–100%] | 100% [100%–100%] | 100% [100%–100%] |

Actions: stop 0%, go 97%, creep 3%, turn 0%. The plan went into a corridor blocked within 0.25 m in 0% of its go steps; it creeps where the mask has no free direction.
Its ranges in steps agree with the reference's within a factor of 2 for 82% of the directions with a known range (the default seed: 3017 direction-steps, median log2 ratio -0.35; below zero is short, the safe side). The direction it heads for seldom has one (22 steps): it mostly heads into open water, where the map holds nothing. The range straight ahead, where forward motion gives the least parallax, is known in 52% [49%–58%] of the steps.

Water directions (the default seed: 863 of them over the steps): the map vetoed one in 3 steps, the mask's own choice in 1 of them; the reference had a surface within 8 steps for 0 of those vetoes and none for 3. The reference had a surface within 8 steps in 0 water directions in all: nothing to veto. The plan bounds them instead (`clear`, 0.75 of the median range of the surfaces in view); in the median of the seeds, known in 92% of them, 15.3 steps, beyond 8 steps in 98%; the reference was nearer than it in 0% [0%–0%].

## 5 Hz (169 steps)

| Heading for | Free beyond 0.25 m | Free beyond 0.5 m | Free beyond 1 m |
|---|---:|---:|---:|
| the plan, going | 100% [100%–100%] | 97% [97%–97%] | 92% [91%–92%] |
| the plan, creeping | 100% [100%–100%] | 83% [67%–100%] | 75% [33%–100%] |
| the mask's own choice (when it has one) | 100% [100%–100%] | 97% [97%–97%] | 91% [91%–91%] |
| straight ahead | 92% [92%–92%] | 85% [85%–85%] | 51% [51%–51%] |
| the best direction (hindsight) | 100% [100%–100%] | 100% [100%–100%] | 100% [100%–100%] |

Actions: stop 0%, go 96%, creep 4%, turn 0%. The plan went into a corridor blocked within 0.25 m in 0% of its go steps; it creeps where the mask has no free direction.
Its ranges in steps agree with the reference's within a factor of 2 for 87% of the directions with a known range (the default seed: 1606 direction-steps, median log2 ratio -0.23; below zero is short, the safe side). The direction it heads for seldom has one (12 steps): it mostly heads into open water, where the map holds nothing. The range straight ahead, where forward motion gives the least parallax, is known in 83% [78%–88%] of the steps.

Water directions (the default seed: 418 of them over the steps): the map vetoed one in 5 steps, the mask's own choice in 0 of them; the reference had a surface within 8 steps for 0 of those vetoes and none for 5. The reference had a surface within 8 steps in 0 water directions in all: nothing to veto. The plan bounds them instead (`clear`, 0.75 of the median range of the surfaces in view); in the median of the seeds, known in 82% of them, 9.5 steps, beyond 8 steps in 70%; the reference was nearer than it in 0% [0%–1%].

Figures: `plan_<rate>hz.png` (the planned direction's corridor over the clip) and `plan_frames_<rate>hz.png` (example frames with the plan). `board_expected_plan_<rate>hz.json` holds every step's plan hash and the final one for `uwslam.mvo_stream --expected-plan`.

Limits: one clip, open-loop; the goal is straight ahead; the map holds seconds, not the whole dive.

Re-run (the output folder must not exist or be empty):

```bash
python -m uwslam.path_plan <video> --reconstruction <folder with images.txt and images.ply> --output results/lake_clip/path_plan_2
```
