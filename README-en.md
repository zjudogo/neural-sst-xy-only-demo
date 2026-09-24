# XY-only Neural-SST demo

**Language:** English | [简体中文](README-zh-CN.md) | [README home](README.md)

This demo predicts one `224 × 224` spatial probability map `P_xy`. SST uses
that map to choose some XY state samples. The model does not predict yaw,
steering, control, or duration.

The bundled checkpoint was trained with the start yaw as input, so the U-Net
receives `sin(start_yaw)` and `cos(start_yaw)` as input channels.

During planning, OMPL samples yaw, steering, and duration.

## Included files

- `checkpoint/best.pt`: the compatible path-image U-Net checkpoint;
- `dataset/test/`: 500 real Neural-SST test tasks, each as one readable JSON;
- `src/run.py`: the only Python command-line entry point;
- `src/NeuralSSTXY.cpp`: the native C++ OMPL SST implementation;
- `src/ompl_demo_cmake.patch`: the one-target OMPL demo build patch;
- `result/`: saved JSON, JPG, and the completed bias comparison CSV.

## Native OMPL SST

Python only performs the GPU U-Net forward pass and transfers one static input
file per task: occupancy, start/goal, seed, `xy_bias`, and `P_xy`.
`NeuralSSTXY.cpp` keeps the entire SST loop in C++: XY mixture sampling, yaw
sampling, collision checking, Dubins propagation, witness pruning, first
arrival counting, and path extraction. There are no Python callbacks during
SST expansion.

This repository ships source, never a platform-specific executable. To build
the native demo on Ubuntu/WSL, first install the C++ build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build git patch \
  libboost-program-options-dev libboost-serialization-dev libeigen3-dev
```

Clone the tested OMPL release and initialize its bundled dependencies:

```bash
git clone --branch 2.0.1 --recursive https://github.com/ompl/ompl.git ompl-2.0.1
cd ompl-2.0.1
git submodule update --init --recursive
```

Set these paths from the Neural-SST demo root. Replace the two absolute paths
with your own locations:

```bash
DEMO_DIR=/absolute/path/to/45.neural_sst_xy_only_demo
OMPL_SRC=/absolute/path/to/ompl-2.0.1
OMPL_BUILD=/absolute/path/to/ompl-2.0.1/build/release
```

Copy the native planner source into OMPL's demo directory:

```bash
cp "$DEMO_DIR/src/NeuralSSTXY.cpp" "$OMPL_SRC/demos/NeuralSSTXY.cpp"
```

Apply the supplied patch from the OMPL source root. It adds only one demo
target to OMPL's existing `demos/CMakeLists.txt`:

```cmake
add_ompl_demo(NeuralSSTXY NeuralSSTXY.cpp)
```

```bash
cd "$OMPL_SRC"
patch -p1 < "$DEMO_DIR/src/ompl_demo_cmake.patch"
```

Configure OMPL with demos enabled and compile only this target:

```bash
cmake -S "$OMPL_SRC" -B "$OMPL_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOMPL_BUILD_DEMOS=ON \
  -DOMPL_BUILD_PYTHON_BINDINGS=OFF
cmake --build "$OMPL_BUILD" --target NeuralSSTXY -j"$(nproc)"
```

The executable is now generated locally at:

```text
<OMPL build>/demos/NeuralSSTXY
```

No `cmake --install` step is required. Point the runner at the locally built
executable without adding another CLI option:

```bash
export NEURAL_SST_XY_EXECUTABLE="$OMPL_BUILD/demos/NeuralSSTXY"
cd "$DEMO_DIR"
python src/run.py --source-jobs dataset/test/0000_0000_000.json --xy-bias 0.5
```

`NEURAL_SST_XY_EXECUTABLE` is optional when `NeuralSSTXY` is already on `PATH`.


## Two input modes

`src/run.py` has exactly two input modes. `--source-jobs` accepts one task JSON
file. `--dataset-dir` accepts a directory of task JSON files. The two options
are mutually exclusive.

```bash
# Run one task JSON.
python src/run.py \
  --source-jobs dataset/test/0000_0000_000.json

# Run every task JSON in a directory.
python src/run.py \
  --dataset-dir dataset/test
```

Both commands use these defaults:

| Option | Default |
| --- | --- |
| `--checkpoint` | `checkpoint/best.pt` |
| `--xy-bias` | `0.5` |
| `--seed` | `20260713` |
| `--result-dir` | `result/` |

For example, this runs one task with a different bias and stores its artifacts
in another directory:

```bash
python src/run.py \
  --source-jobs dataset/test/0000_0000_000.json \
  --xy-bias 0.7 \
  --result-dir result_bias_0.7
```

The runner uses CUDA automatically when it is available. Each task creates:

- `{task_id}-{xy_bias}.json` with planning status, first-arrival sampling
  counters, `state_sequence`, `control_sequence`, and `control_segments`;
- `{task_id}-{xy_bias}.jpg` with obstacles, start, goal, and the returned SST
  path. When `xy_bias != 0`, the JPG also overlays `P_xy` on free space.
- optionally, when `--tree-dir` is supplied, `{task_id}-{xy_bias}.tree.json`
  and `{task_id}-{xy_bias}.tree.jpg` with the SST planner data and rendered
  search tree. The tree snapshot contains the nodes and parent edges retained
  by SST when `solve()` stops, including ancestors of active nodes. SST-pruned
  nodes and collision-rejected propagation attempts are not available through
  this planner-data export.

Before recording `model_time_ms`, the runner performs one unrecorded CUDA/cuDNN
warm-up on the first selected task. This keeps one-off GPU process-start
latency separate from the U-Net forward-pass measurement.

## Task JSON format

Each file contains one environment and exactly one start/goal pair:

```json
{
  "env_dims": [224, 224],
  "rectangle_obstacles": [[x, y, width, height]],
  "circle_obstacles": [[center_x, center_y, radius]],
  "start": [start_x_px, start_y_px],
  "goal": [goal_x_px, goal_y_px],
  "start_yaw_rad": 0.29670597
}
```

The task filename follows `{map_id}_{pair_id}_{start_yaw_deg}.json`. It has no
bias suffix. Result filenames carry the chosen bias.

## Historical first-arrival sampling reference

`result/compare_xy_bias_0.0_vs_0.5.csv` is the saved 500-task reference used
for the historical claim below. It compares the old direct-allocator SST
implementation with the same test tasks and seed protocol. Each sampler's mean
uses only its own successful tasks; an unsolved task has no first-arrival count
and is excluded from that sampler's mean.

| XY bias | Successful tasks | Mean first-arrival samples |
| ---: | ---: | ---: |
| `0.0` | 448 / 500 | 3,629.230 |
| `0.5` | 456 / 500 | 1,728.864 |

The recorded reduction is `1 - 1728.864 / 3629.230 = 52.363%`.

### `path_image_unet` full-data timing reference

The following is the separate 500-task **`path_image_unet`** reference, measured
with CUDA, no probability-map cache, and per-run inference timing after warm-up.
The SST-only column is native planning time; total time adds map inference once.
Means are over successful runs at each bias, matching the corresponding
first-arrival sample averages. The final column is the end-to-end time saved
relative to the SST-only (`bias=0.0`) baseline.

| Model | Bias | Runs | Success | Map inference | SST-only planning | Total first-arrival time | First-arrival samples | Net time saved vs. SST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `path_image_unet` | 0.0 | 500 | 89.6% | 0.000 ms | 32.674 ms | 32.674 ms | 3,618.2 | 0.000 ms |
| `path_image_unet` | 0.3 | 500 | 90.6% | 6.902 ms | 21.732 ms | 28.688 ms | 2,570.7 | 3.986 ms |
| `path_image_unet` | 0.5 | 500 | 90.4% | 6.804 ms | 11.632 ms | 18.375 ms | 1,622.6 | 14.299 ms |
| `path_image_unet` | 0.7 | 500 | 90.6% | 6.879 ms | 15.532 ms | 22.361 ms | 1,945.4 | 10.313 ms |
| `path_image_unet` | 0.9 | 500 | 90.2% | 6.896 ms | 12.095 ms | 18.949 ms | 1,446.0 | 13.725 ms |

At `bias=0.5`, planning takes about 21.042 ms less than SST-only. After
subtracting the measured 6.804 ms U-Net inference time, this yields about
**14.299 ms** in net end-to-end savings.

At `bias=0.7`, the net savings are about **10.313 ms**. The source report gives
a mean inference time of 6.804 ms and a median of 5.086 ms for `bias=0.5`.

The experiment configuration records `device=cuda`, CUDA availability, and
inference timing after warm-up.

The saved numbers were produced by this native C++ SST design. A future rerun
should use the same compiled source, OMPL version, seed protocol, and test set
before being compared with this reference.

## Paired visualization cases

The saved historical JSON files contain the returned solution path, but not the
search tree. The 12 selected tasks below were therefore rerun with tree export,
once for SST (`xy_bias=0.0`) and once for Neural-SST (`xy_bias=0.5`). Each table
cell shows the planner status, the exact `state_sample_calls`, the number of
retained tree vertices/edges, and the tree image from that same run. Green lines
show the retained SST tree; the returned solution path is overlaid in blue when
one was found. No composite/subplot images are used.

To export a tree for another run, pass `--tree-dir`:

```bash
python src/run.py \
  --source-jobs dataset/test/0104_0001_270.json \
  --xy-bias 0.5 \
  --result-dir result/tree_run/results \
  --tree-dir result/tree_run/trees
```

These are case studies, not a replacement for the aggregate 500-task table
above. Planning is stochastic, and `P_xy` may vary slightly across inference
devices. Four Neural-SST reruns differ from the old path-only records; table
classifications and counts below intentionally follow the new tree-producing
runs, so that every displayed image, status, and count refer to one run. The
manifest at `result/trees/manifest.json` records whether each run matched its
saved historical JSON. A failure at the sample limit shows `50,000` state
samples; `first_solution_iterations` is one higher (`50,001`) in failed runs
because it includes the final termination-condition check.

The tree is the planner data retained by SST when `solve()` stops, including
ancestors of active nodes. It is not a full history: nodes already removed by
SST pruning and collision-rejected propagation attempts are not available in
this export. Each image below is linked directly to its individual run.

### Neural-SST substantially reduces sampling

All four paired runs below found a solution with both planners, and Neural-SST
used fewer state samples in each case.

| Task | SST (`xy_bias=0.0`) | Neural-SST (`xy_bias=0.5`) |
| --- | --- | --- |
| `0104_0001_270` | Success · 6,365 samples · 3,610 vertices / 3,608 edges<br><img src="result/trees/0104_0001_270-0.0.tree.jpg" alt="SST retained tree for task 0104_0001_270" width="320"> | Success · 46 samples · 46 vertices / 44 edges<br><img src="result/trees/0104_0001_270-0.5.tree.jpg" alt="Neural-SST retained tree for task 0104_0001_270" width="320"> |
| `0115_0002_180` | Success · 12,382 samples · 6,959 vertices / 6,957 edges<br><img src="result/trees/0115_0002_180-0.0.tree.jpg" alt="SST retained tree for task 0115_0002_180" width="320"> | Success · 147 samples · 128 vertices / 126 edges<br><img src="result/trees/0115_0002_180-0.5.tree.jpg" alt="Neural-SST retained tree for task 0115_0002_180" width="320"> |
| `0056_0002_300` | Success · 3,839 samples · 2,236 vertices / 2,234 edges<br><img src="result/trees/0056_0002_300-0.0.tree.jpg" alt="SST retained tree for task 0056_0002_300" width="320"> | Success · 46 samples · 40 vertices / 38 edges<br><img src="result/trees/0056_0002_300-0.5.tree.jpg" alt="Neural-SST retained tree for task 0056_0002_300" width="320"> |
| `0082_0002_180` | Success · 40,134 samples · 14,220 vertices / 14,218 edges<br><img src="result/trees/0082_0002_180-0.0.tree.jpg" alt="SST retained tree for task 0082_0002_180" width="320"> | Success · 26,028 samples · 11,212 vertices / 11,210 edges<br><img src="result/trees/0082_0002_180-0.5.tree.jpg" alt="Neural-SST retained tree for task 0082_0002_180" width="320"> |

### Neural-SST substantially increases sampling

These two selected reruns still solve with both planners, but use substantially
more samples with the learned spatial bias. The two former increase examples
`0046_0000_120` and `0049_0001_150` instead hit the sample limit in their new
tree-producing reruns; they are therefore listed in the failure table below.

| Task | SST (`xy_bias=0.0`) | Neural-SST (`xy_bias=0.5`) |
| --- | --- | --- |
| `0063_0001_030` | Success · 109 samples · 89 vertices / 87 edges<br><img src="result/trees/0063_0001_030-0.0.tree.jpg" alt="SST retained tree for task 0063_0001_030" width="320"> | Success · 7,409 samples · 3,902 vertices / 3,900 edges (+6,697.25%)<br><img src="result/trees/0063_0001_030-0.5.tree.jpg" alt="Neural-SST retained tree for task 0063_0001_030" width="320"> |
| `0067_0000_120` | Success · 669 samples · 215 vertices / 213 edges<br><img src="result/trees/0067_0000_120-0.0.tree.jpg" alt="SST retained tree for task 0067_0000_120" width="320"> | Success · 7,015 samples · 3,309 vertices / 3,307 edges (+948.58%)<br><img src="result/trees/0067_0000_120-0.5.tree.jpg" alt="Neural-SST retained tree for task 0067_0000_120" width="320"> |

### SST fails while Neural-SST succeeds

Here uniform SST exhausts its complete 50,000-state-sample budget without an
exact solution, while Neural-SST finds one in the corresponding tree-producing
run.

| Task | SST (`xy_bias=0.0`) | Neural-SST (`xy_bias=0.5`) |
| --- | --- | --- |
| `0065_0002_300` | Failed · 50,000 samples · 20,218 vertices / 20,216 edges<br><img src="result/trees/0065_0002_300-0.0.tree.jpg" alt="SST retained tree for task 0065_0002_300" width="320"> | Success · 349 samples · 239 vertices / 237 edges<br><img src="result/trees/0065_0002_300-0.5.tree.jpg" alt="Neural-SST retained tree for task 0065_0002_300" width="320"> |
| `0062_0000_240` | Failed · 50,000 samples · 18,527 vertices / 18,525 edges<br><img src="result/trees/0062_0000_240-0.0.tree.jpg" alt="SST retained tree for task 0062_0000_240" width="320"> | Success · 764 samples · 568 vertices / 566 edges<br><img src="result/trees/0062_0000_240-0.5.tree.jpg" alt="Neural-SST retained tree for task 0062_0000_240" width="320"> |
| `0070_0001_150` | Failed · 50,000 samples · 22,427 vertices / 22,425 edges<br><img src="result/trees/0070_0001_150-0.0.tree.jpg" alt="SST retained tree for task 0070_0001_150" width="320"> | Success · 771 samples · 498 vertices / 496 edges<br><img src="result/trees/0070_0001_150-0.5.tree.jpg" alt="Neural-SST retained tree for task 0070_0001_150" width="320"> |

### Neural-SST fails while SST succeeds

This is the complementary failure mode in the tree-producing runs. The
Neural-SST planner exhausts its 50,000-sample budget, while uniform SST finds a
solution.

| Task | SST (`xy_bias=0.0`) | Neural-SST (`xy_bias=0.5`) |
| --- | --- | --- |
| `0046_0000_120` | Success · 730 samples · 415 vertices / 413 edges<br><img src="result/trees/0046_0000_120-0.0.tree.jpg" alt="SST retained tree for task 0046_0000_120" width="320"> | Failed · 50,000 samples · 12,871 vertices / 12,869 edges<br><img src="result/trees/0046_0000_120-0.5.tree.jpg" alt="Neural-SST retained tree for task 0046_0000_120" width="320"> |
| `0049_0001_150` | Success · 1,003 samples · 387 vertices / 385 edges<br><img src="result/trees/0049_0001_150-0.0.tree.jpg" alt="SST retained tree for task 0049_0001_150" width="320"> | Failed · 50,000 samples · 22,489 vertices / 22,487 edges<br><img src="result/trees/0049_0001_150-0.5.tree.jpg" alt="Neural-SST retained tree for task 0049_0001_150" width="320"> |
| `0094_0002_180` | Success · 2,917 samples · 935 vertices / 933 edges<br><img src="result/trees/0094_0002_180-0.0.tree.jpg" alt="SST retained tree for task 0094_0002_180" width="320"> | Failed · 50,000 samples · 14,837 vertices / 14,835 edges<br><img src="result/trees/0094_0002_180-0.5.tree.jpg" alt="Neural-SST retained tree for task 0094_0002_180" width="320"> |

## Current Challenges and Limitations

- **The number of samples is still high.** In the 500-task `path_image_unet` results, the mean number of first-arrival samples on successful Neural-SST runs is still about 1,446–2,571, depending on `xy_bias`; at `bias=0.5`, it is 1,622.6. Learned guidance reduces search effort, but does not eliminate the need for many state samples.
- **Probability-map errors can make guidance counterproductive.** If `P_xy` misses a feasible corridor or concentrates probability on locations that are dynamically unreachable, the sampler can waste additional expansions. This effect is stronger at higher `xy_bias`. On individual tasks Neural-SST may therefore need more samples than uniform SST, or fail on a task that SST solves. The counterexamples in the tree tables below are illustrative cases, not estimates of aggregate success rates.
- **Yaw is still sampled randomly.** The network outputs only the two-dimensional `P_xy` map; it does not predict or condition on yaw. Even an XY location in a high-probability path region may be paired with a randomly sampled heading that is incompatible with a narrow passage or the vehicle's steering constraints, causing propagation to fail or collision checking to reject the state. The current goal test also checks XY only and does not constrain the terminal heading.
- **The guidance map is static.** The probability map is inferred once per task. The sampling distribution is not updated online using the current SST tree, failed extensions, or newly discovered corridors. If the initial prediction is biased, subsequent search continues to be influenced by the same map.
- **The current evaluation focuses on the first solution.** The planner stops after finding its first exact solution. Thus, the reported sample counts and times measure first-arrival efficiency; they do not measure continued path-quality optimization or establish improved asymptotic optimality.

These observations suggest retaining uniform exploration alongside neural guidance and evaluating success rate, first-arrival samples, end-to-end time, and path quality together. The probability map or mean sample count alone is not enough to determine whether the planner is better overall.

## Relation to NI-RRT* and Reference

This demo is related to the learned-guidance idea in Neural Informed RRT* (NI-RRT*) [1], but it is not a direct reproduction of that method. NI-RRT* uses point-cloud state representations and PointNet++ to predict guidance states, together with the admissible sampling-ellipsoid constraint from Informed RRT* (Neural Focus) and a guidance-state connectivity mechanism (Neural Connect). This demo instead uses `path_image_unet` to predict a dense two-dimensional probability map, `P_xy`, from raster inputs and mixes it into XY state sampling in OMPL SST. The current implementation does not include NI-RRT*'s Informed RRT* ellipsoid constraint or Neural Connect.

1. Z. Huang, H. Chen, J. Pohovey, and K. Driggs-Campbell, “Neural Informed RRT*: Learning-based Path Planning with Point Cloud State Representations under Admissible Ellipsoidal Constraints,” *2024 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 8742–8748, 2024. [DOI](https://doi.org/10.1109/ICRA57147.2024.10611099) · [arXiv](https://arxiv.org/abs/2309.14595) · [official code](https://github.com/tedhuang96/nirrt_star).
