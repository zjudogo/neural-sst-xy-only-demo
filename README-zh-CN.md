# XY-only Neural-SST 演示

**语言：** [English](README-en.md) | 简体中文 | [文档首页](README.md)

本演示预测一张 `224 × 224` 的空间概率图 `P_xy`，由 SST 使用该概率图选择部分 XY 状态进行采样。模型不预测 yaw、转向、控制量或持续时间。

随附 checkpoint 的训练输入包含起始航向，U-Net 的输入包含 `sin(start_yaw)` 和 `cos(start_yaw)` 两个通道。

规划期间，OMPL 自行采样 yaw、转向及持续时间。

## 文件说明

- `checkpoint/best.pt`：兼容的 path-image U-Net checkpoint；
- `dataset/test/`：500 个 Neural-SST 测试任务，每个任务保存为可读 JSON；
- `src/run.py`：唯一的 Python 命令行入口；
- `src/NeuralSSTXY.cpp`：原生 C++ OMPL SST 实现；
- `src/ompl_demo_cmake.patch`：仅添加一个构建目标的 OMPL demo 补丁；
- `result/`：已保存的 JSON、JPG 和 bias 对比 CSV。

## 原生 OMPL SST

Python 只负责执行 GPU U-Net 前向推理，并为每个任务传递一个静态输入文件，其中包含占据图、起点/终点、随机种子、`xy_bias` 和 `P_xy`。

`NeuralSSTXY.cpp` 在 C++ 中完成整个 SST 循环：XY 混合采样、yaw 采样、碰撞检查、Dubins 状态传播、witness 剪枝、首次到达计数及路径提取。SST 扩展期间不会调用 Python 回调。

仓库只提供源代码，不包含平台专用的可执行文件。要在 Ubuntu/WSL 中构建原生 demo，请先安装 C++ 构建依赖：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build git patch \
  libboost-program-options-dev libboost-serialization-dev libeigen3-dev
```

克隆经过测试的 OMPL 版本并初始化其依赖：

```bash
git clone --branch 2.0.1 --recursive https://github.com/ompl/ompl.git ompl-2.0.1
cd ompl-2.0.1
git submodule update --init --recursive
```

在 Neural-SST demo 根目录下设置以下路径，并将绝对路径替换为你自己的位置：

```bash
DEMO_DIR=/absolute/path/to/45.neural_sst_xy_only_demo
OMPL_SRC=/absolute/path/to/ompl-2.0.1
OMPL_BUILD=/absolute/path/to/ompl-2.0.1/build/release
```

将原生规划器源文件复制到 OMPL 的 demo 目录：

```bash
cp "$DEMO_DIR/src/NeuralSSTXY.cpp" "$OMPL_SRC/demos/NeuralSSTXY.cpp"
```

在 OMPL 源码根目录应用随附补丁。该补丁只向 OMPL 现有的 `demos/CMakeLists.txt` 添加一个 demo 目标：

```cmake
add_ompl_demo(NeuralSSTXY NeuralSSTXY.cpp)
```

```bash
cd "$OMPL_SRC"
patch -p1 < "$DEMO_DIR/src/ompl_demo_cmake.patch"
```

启用 demos 并只编译此目标：

```bash
cmake -S "$OMPL_SRC" -B "$OMPL_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOMPL_BUILD_DEMOS=ON \
  -DOMPL_BUILD_PYTHON_BINDINGS=OFF
cmake --build "$OMPL_BUILD" --target NeuralSSTXY -j"$(nproc)"
```

生成的可执行文件位于：

```text
<OMPL build>/demos/NeuralSSTXY
```

无需执行 `cmake --install`。无需增加命令行参数，只需将 runner 指向本地构建的可执行文件：

```bash
export NEURAL_SST_XY_EXECUTABLE="$OMPL_BUILD/demos/NeuralSSTXY"
cd "$DEMO_DIR"
python src/run.py --source-jobs dataset/test/0000_0000_000.json --xy-bias 0.5
```

如果 `NeuralSSTXY` 已经位于 `PATH` 中，则无需设置 `NEURAL_SST_XY_EXECUTABLE`。

## 两种输入模式

`src/run.py` 有且只有两种输入模式。`--source-jobs` 接收一个任务 JSON 文件；`--dataset-dir` 接收一个包含任务 JSON 文件的目录。两者互斥。

```bash
# 运行一个任务 JSON。
python src/run.py \
  --source-jobs dataset/test/0000_0000_000.json

# 运行目录中的所有任务 JSON。
python src/run.py \
  --dataset-dir dataset/test
```

两种命令的默认参数如下：

| 参数 | 默认值 |
| --- | --- |
| `--checkpoint` | `checkpoint/best.pt` |
| `--xy-bias` | `0.5` |
| `--seed` | `20260713` |
| `--result-dir` | `result/` |

例如，以下命令使用另一个 bias 运行单个任务，并将产物保存到其他目录：

```bash
python src/run.py \
  --source-jobs dataset/test/0000_0000_000.json \
  --xy-bias 0.7 \
  --result-dir result_bias_0.7
```

如果 CUDA 可用，runner 会自动使用 CUDA。每个任务会生成：

- `{task_id}-{xy_bias}.json`：规划状态、首次到达采样计数、`state_sequence`、`control_sequence` 和 `control_segments`；
- `{task_id}-{xy_bias}.jpg`：包含障碍物、起点、终点和 SST 返回路径；当 `xy_bias != 0` 时，图片也会在自由空间上叠加 `P_xy`；
- 可选地，传入 `--tree-dir` 后会生成 `{task_id}-{xy_bias}.tree.json` 和 `{task_id}-{xy_bias}.tree.jpg`，分别保存 SST planner data 和渲染后的搜索树。树快照包含 `solve()` 结束时 SST 保留的节点和父子边，也包括活动节点的祖先。此导出接口无法提供已经被 SST 剪枝移除的节点，也无法提供碰撞检查拒绝的传播尝试。

记录 `model_time_ms` 前，runner 会针对第一个选中的任务执行一次不计入测量的 CUDA/cuDNN 预热，以将一次性的 GPU 进程启动开销与 U-Net 前向推理耗时分开。

## 任务 JSON 格式

每个文件包含一个环境和且仅有一组起点/终点：

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

任务文件名格式为 `{map_id}_{pair_id}_{start_yaw_deg}.json`，不包含 bias 后缀；结果文件名则包含所用 bias。

## 历史首次到达采样参考

`result/compare_xy_bias_0.0_vs_0.5.csv` 是以下历史结论所使用的 500 任务参考数据。它使用相同测试任务和种子协议，对比旧版直接分配器 SST。每种采样器的均值只统计各自成功的任务；未求解任务没有首次到达次数，因此从该采样器的均值中排除。

| XY bias | 成功任务数 | 首次到达采样均值 |
| ---: | ---: | ---: |
| `0.0` | 448 / 500 | 3,629.230 |
| `0.5` | 456 / 500 | 1,728.864 |

记录的降幅为 `1 - 1728.864 / 3629.230 = 52.363%`。

### `path_image_unet` 全数据集耗时参考

以下是另一组 500 任务的 **`path_image_unet`** 参考数据：使用 CUDA，不缓存概率图，并在预热后逐次计量推理耗时。SST-only 列是原生规划耗时；总耗时在此基础上加上一次概率图推理。均值针对每种 bias 下成功的运行计算，与对应的首次到达采样均值使用相同成功任务口径。最后一列表示相较于 SST-only（`bias=0.0`）基线节省的端到端时间。

| 模型 | Bias | 运行数 | 成功率 | 概率图推理 | SST-only 规划 | 首次到达总时间 | 首次到达采样数 | 相较 SST 净节省时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `path_image_unet` | 0.0 | 500 | 89.6% | 0.000 ms | 32.674 ms | 32.674 ms | 3,618.2 | 0.000 ms |
| `path_image_unet` | 0.3 | 500 | 90.6% | 6.902 ms | 21.732 ms | 28.688 ms | 2,570.7 | 3.986 ms |
| `path_image_unet` | 0.5 | 500 | 90.4% | 6.804 ms | 11.632 ms | 18.375 ms | 1,622.6 | 14.299 ms |
| `path_image_unet` | 0.7 | 500 | 90.6% | 6.879 ms | 15.532 ms | 22.361 ms | 1,945.4 | 10.313 ms |
| `path_image_unet` | 0.9 | 500 | 90.2% | 6.896 ms | 12.095 ms | 18.949 ms | 1,446.0 | 13.725 ms |

`bias=0.5` 的规划耗时比 SST-only 少约 21.042 ms，扣除测得的 6.804 ms U-Net 推理后，端到端净省约 **14.299 ms**；

`bias=0.7` 的净节省约 **10.313 ms**。源报告中 `bias=0.5` 的平均推理耗时为 6.804 ms，中位数为 5.086 ms。

实验配置记录了 `device=cuda`、CUDA 可用以及预热后的推理计时。

## 当前困难与局限

- **采样次数仍然偏高。** 在 `path_image_unet` 的 500 任务结果中，Neural-SST 成功任务的首次到达采样均值仍约为 1,446–2,571 次（取决于 `xy_bias`）；`bias=0.5` 时为 1,622.6 次。学习引导减少了搜索工作量，但尚未消除大量状态采样。
- **概率图误差可能使引导适得其反。** `P_xy` 如果未能覆盖真实可行走廊，或者将概率集中在对动力学不可达的位置，采样器会浪费更多扩展；高 `xy_bias` 时这种偏置影响更强。因此个别任务中 Neural-SST 可能比均匀 SST 需要更多采样，甚至在 SST 成功时未能求解。末尾案例表中的新树导出运行包含这类反例；它们是案例而非总体成功率估计。
- **航向仍是随机采样。** 网络仅输出二维 `P_xy`，没有预测或条件化 yaw。一个 XY 位置即便处于高概率路径区域，其随机采到的航向也可能与狭窄通道或车辆转向约束不匹配，导致传播失败或碰撞检查拒绝。当前目标判定也只检查 XY 位置，不约束最终航向。
- **引导图是静态的。** 每个任务只推理一次概率图；采样分布不会根据当前 SST 树、失败扩展或已发现的新通路在线更新。若早期预测偏差较大，后续搜索仍会持续受到相同引导图影响。
- **当前评测侧重首次解。** 规划器在找到首个精确解后停止，因此现有采样和耗时结果衡量的是首次到达效率，不代表路径质量持续优化后的结果，也不能据此声称路径渐近最优性更好。

这些现象说明，神经引导应与保留的均匀探索共同使用，并且需要按成功率、首次到达采样、端到端耗时和路径质量一起评估；单看概率图或平均采样数不足以判断规划器是否整体更好。


## 成对结果与搜索树可视化

已保存的历史 JSON 文件包含返回的解路径，但不包含搜索树。

因此，下列 12 个选定任务各自重新运行并导出树：SST 使用 `xy_bias=0.0`，Neural-SST 使用 `xy_bias=0.5`。
每个表格单元格展示规划状态、精确的 `state_sample_calls`、保留树的顶点/边数量，以及来自同一次运行的树图。
绿色细线显示 SST 保留树；如果找到解，则以蓝色叠加返回路径。这里没有使用拼接图或子图。

如需为其他任务导出树，可传入 `--tree-dir`：

```bash
python src/run.py \
  --source-jobs dataset/test/0104_0001_270.json \
  --xy-bias 0.5 \
  --result-dir result/tree_run/results \
  --tree-dir result/tree_run/trees
```

这些案例用于说明情况，并非替代上方 500 任务总体表。规划具有随机性，推理设备也可能使 `P_xy` 略有差异。四个 Neural-SST 重跑结果与旧的仅路径记录不同；下表分类和计数均以本次生成树的运行结果为准，从而确保图片、状态及计数指向同一次运行。`result/trees/manifest.json` 记录了各项运行是否与历史 JSON 相符。达到采样上限的失败案例显示 `50,000` 次状态采样；对应的 `first_solution_iterations` 多 1（即 `50,001`），因为它还计入了最后一次终止条件检查。

树图表示 `solve()` 结束时 SST 仍保留在 planner data 中的节点和边，包括活动节点的祖先。它不是完整搜索历史：已被 SST 剪枝移除的节点以及碰撞检查拒绝的传播尝试不会包含在导出结果中。下表中的每张图片都单独引用对应运行的树图。

### Neural-SST 显著减少采样

以下四个成对重跑案例均由两种方法成功求解，且 Neural-SST 使用的状态采样次数更少。

| 任务 | SST（`xy_bias=0.0`） | Neural-SST（`xy_bias=0.5`） |
| --- | --- | --- |
| `0104_0001_270` | 成功 · 6,365 次采样 · 3,610 个顶点 / 3,608 条边<br><img src="result/trees/0104_0001_270-0.0.tree.jpg" alt="任务 0104_0001_270 的 SST 保留树" width="320"> | 成功 · 46 次采样 · 46 个顶点 / 44 条边<br><img src="result/trees/0104_0001_270-0.5.tree.jpg" alt="任务 0104_0001_270 的 Neural-SST 保留树" width="320"> |
| `0115_0002_180` | 成功 · 12,382 次采样 · 6,959 个顶点 / 6,957 条边<br><img src="result/trees/0115_0002_180-0.0.tree.jpg" alt="任务 0115_0002_180 的 SST 保留树" width="320"> | 成功 · 147 次采样 · 128 个顶点 / 126 条边<br><img src="result/trees/0115_0002_180-0.5.tree.jpg" alt="任务 0115_0002_180 的 Neural-SST 保留树" width="320"> |
| `0056_0002_300` | 成功 · 3,839 次采样 · 2,236 个顶点 / 2,234 条边<br><img src="result/trees/0056_0002_300-0.0.tree.jpg" alt="任务 0056_0002_300 的 SST 保留树" width="320"> | 成功 · 46 次采样 · 40 个顶点 / 38 条边<br><img src="result/trees/0056_0002_300-0.5.tree.jpg" alt="任务 0056_0002_300 的 Neural-SST 保留树" width="320"> |
| `0082_0002_180` | 成功 · 40,134 次采样 · 14,220 个顶点 / 14,218 条边<br><img src="result/trees/0082_0002_180-0.0.tree.jpg" alt="任务 0082_0002_180 的 SST 保留树" width="320"> | 成功 · 26,028 次采样 · 11,212 个顶点 / 11,210 条边<br><img src="result/trees/0082_0002_180-0.5.tree.jpg" alt="任务 0082_0002_180 的 Neural-SST 保留树" width="320"> |

### Neural-SST 显著增加采样

以下两个案例中，两种方法都成功求解，但采用学习得到的空间偏置后采样次数显著增加。原先另外两个增加采样案例 `0046_0000_120` 和 `0049_0001_150` 在新的树导出重跑中达到采样上限，因此移至下方失败案例表。

| 任务 | SST（`xy_bias=0.0`） | Neural-SST（`xy_bias=0.5`） |
| --- | --- | --- |
| `0063_0001_030` | 成功 · 109 次采样 · 89 个顶点 / 87 条边<br><img src="result/trees/0063_0001_030-0.0.tree.jpg" alt="任务 0063_0001_030 的 SST 保留树" width="320"> | 成功 · 7,409 次采样 · 3,902 个顶点 / 3,900 条边（+6,697.25%）<br><img src="result/trees/0063_0001_030-0.5.tree.jpg" alt="任务 0063_0001_030 的 Neural-SST 保留树" width="320"> |
| `0067_0000_120` | 成功 · 669 次采样 · 215 个顶点 / 213 条边<br><img src="result/trees/0067_0000_120-0.0.tree.jpg" alt="任务 0067_0000_120 的 SST 保留树" width="320"> | 成功 · 7,015 次采样 · 3,309 个顶点 / 3,307 条边（+948.58%）<br><img src="result/trees/0067_0000_120-0.5.tree.jpg" alt="任务 0067_0000_120 的 Neural-SST 保留树" width="320"> |

### SST 失败而 Neural-SST 成功

以下案例中，均匀采样的 SST 在完整的 50,000 次状态采样预算内未找到精确解，而对应的 Neural-SST 树导出运行成功找到了路径。

| 任务 | SST（`xy_bias=0.0`） | Neural-SST（`xy_bias=0.5`） |
| --- | --- | --- |
| `0065_0002_300` | 失败 · 50,000 次采样 · 20,218 个顶点 / 20,216 条边<br><img src="result/trees/0065_0002_300-0.0.tree.jpg" alt="任务 0065_0002_300 的 SST 保留树" width="320"> | 成功 · 349 次采样 · 239 个顶点 / 237 条边<br><img src="result/trees/0065_0002_300-0.5.tree.jpg" alt="任务 0065_0002_300 的 Neural-SST 保留树" width="320"> |
| `0062_0000_240` | 失败 · 50,000 次采样 · 18,527 个顶点 / 18,525 条边<br><img src="result/trees/0062_0000_240-0.0.tree.jpg" alt="任务 0062_0000_240 的 SST 保留树" width="320"> | 成功 · 764 次采样 · 568 个顶点 / 566 条边<br><img src="result/trees/0062_0000_240-0.5.tree.jpg" alt="任务 0062_0000_240 的 Neural-SST 保留树" width="320"> |
| `0070_0001_150` | 失败 · 50,000 次采样 · 22,427 个顶点 / 22,425 条边<br><img src="result/trees/0070_0001_150-0.0.tree.jpg" alt="任务 0070_0001_150 的 SST 保留树" width="320"> | 成功 · 771 次采样 · 498 个顶点 / 496 条边<br><img src="result/trees/0070_0001_150-0.5.tree.jpg" alt="任务 0070_0001_150 的 Neural-SST 保留树" width="320"> |

### Neural-SST 失败而 SST 成功

这是相反的失败情况：Neural-SST 耗尽 50,000 次采样预算，而均匀采样 SST 成功找到了路径。

| 任务 | SST（`xy_bias=0.0`） | Neural-SST（`xy_bias=0.5`） |
| --- | --- | --- |
| `0046_0000_120` | 成功 · 730 次采样 · 415 个顶点 / 413 条边<br><img src="result/trees/0046_0000_120-0.0.tree.jpg" alt="任务 0046_0000_120 的 SST 保留树" width="320"> | 失败 · 50,000 次采样 · 12,871 个顶点 / 12,869 条边<br><img src="result/trees/0046_0000_120-0.5.tree.jpg" alt="任务 0046_0000_120 的 Neural-SST 保留树" width="320"> |
| `0049_0001_150` | 成功 · 1,003 次采样 · 387 个顶点 / 385 条边<br><img src="result/trees/0049_0001_150-0.0.tree.jpg" alt="任务 0049_0001_150 的 SST 保留树" width="320"> | 失败 · 50,000 次采样 · 22,489 个顶点 / 22,487 条边<br><img src="result/trees/0049_0001_150-0.5.tree.jpg" alt="任务 0049_0001_150 的 Neural-SST 保留树" width="320"> |
| `0094_0002_180` | 成功 · 2,917 次采样 · 935 个顶点 / 933 条边<br><img src="result/trees/0094_0002_180-0.0.tree.jpg" alt="任务 0094_0002_180 的 SST 保留树" width="320"> | 失败 · 50,000 次采样 · 14,837 个顶点 / 14,835 条边<br><img src="result/trees/0094_0002_180-0.5.tree.jpg" alt="任务 0094_0002_180 的 Neural-SST 保留树" width="320"> |

## 与 NI-RRT* 的关系与参考文献

本 demo 的学习引导采样思路与 Neural Informed RRT*（NI-RRT*）相关 [1]，但并非该方法的直接复现。NI-RRT* 使用点云状态表示和 PointNet++ 预测引导状态，并结合 Informed RRT* 的可采样椭球约束（Neural Focus）以及引导状态连通机制（Neural Connect）。本 demo 则使用 `path_image_unet` 从栅格输入预测稠密的二维概率图 `P_xy`，将其混入 OMPL SST 的 XY 状态采样；当前实现不包含 NI-RRT* 的 Informed RRT* 椭球约束或 Neural Connect。

1. Z. Huang, H. Chen, J. Pohovey, and K. Driggs-Campbell, “Neural Informed RRT*: Learning-based Path Planning with Point Cloud State Representations under Admissible Ellipsoidal Constraints,” *2024 IEEE International Conference on Robotics and Automation (ICRA)*, pp. 8742–8748, 2024. [DOI](https://doi.org/10.1109/ICRA57147.2024.10611099) · [arXiv](https://arxiv.org/abs/2309.14595) · [official code](https://github.com/tedhuang96/nirrt_star).
