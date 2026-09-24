"""Run XY-only Neural-SST: Python predicts P_xy and native C++ OMPL plans.

This entry point deliberately separates responsibilities: Python handles batched,
accelerated U-Net inference and visualization, while the native program performs
each SST state expansion, control propagation, and collision check. They exchange
fixed task data and one probability map through a temporary text file instead of
re-entering Python repeatedly during the planning loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from geometry import GRID_SIZE, RESOLUTION_M, WORLD_MAX_M, WORLD_MIN_M, load_task
from model import XYProbabilityUNet


# Source lives in <project>/src. Resolve the default checkpoint and result directory
# relative to the project root so their locations do not depend on the current cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The C++ side uses both termination-condition query count and wall-clock time as budgets.
MAX_ITERATIONS = 50_000
MAX_TIME_S = 30.0
# This default can be overridden by NEURAL_SST_XY_EXECUTABLE to support local or
# non-standard installation paths.
NATIVE_EXECUTABLE = "NeuralSSTXY"


def parse_args() -> argparse.Namespace:
    """Parse the command line and require choosing exactly one input mode.

    ``xy_bias=0`` is the purely uniform SST baseline, while ``xy_bias=1`` draws every
    XY sample from the network distribution. Intermediate values are mixed randomly
    per sample by the C++ sampler. Numeric range validation is handled in ``main`` so
    this function is limited to defining and parsing arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-jobs", type=Path, help="One readable task JSON file.")
    source.add_argument("--dataset-dir", type=Path, help="Directory containing readable task JSON files.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoint" / "best.pt")
    parser.add_argument("--xy-bias", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--result-dir", type=Path, default=PROJECT_ROOT / "result")
    parser.add_argument("--tree-dir", type=Path,
                        help="Optionally export the retained SST tree JSON and rendered JPG per task.")
    return parser.parse_args()


def task_paths(args: argparse.Namespace) -> list[Path]:
    """Validate the input source and return task paths in deterministic order."""
    if args.source_jobs:
        if not args.source_jobs.is_file():
            raise FileNotFoundError(f"Task JSON not found: {args.source_jobs}")
        return [args.source_jobs]
    if not args.dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {args.dataset_dir}")
    paths = sorted(args.dataset_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No task JSON files found in {args.dataset_dir}")
    return paths


def load_model(checkpoint_path: Path) -> tuple[XYProbabilityUNet, torch.device]:
    """Load the checkpoint and place the U-Net on CUDA when available.

    The checkpoint's ``config.base_channels`` is part of the model topology and must
    be read before constructing the network so its weights can be loaded safely.
    ``eval()`` disables training-time behavior such as BatchNorm/Dropout (this model
    uses GroupNorm, but the general inference convention is retained).
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    channels = int(checkpoint.get("config", {}).get("base_channels", 32))
    model = XYProbabilityUNet(base_channels=channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


def network_input(task: dict[str, object]) -> np.ndarray:
    """Build the five-channel model input with shape ``[5, 224, 224]``.

    Channels are, in order: occupancy map, start Gaussian heatmap, goal Gaussian
    heatmap, image-wide constant ``sin(yaw)``, and image-wide constant ``cos(yaw)``.
    Encoding yaw with sine/cosine instead of the raw angle removes the numerical
    discontinuity between ``-π`` and ``π``. The heatmap denominator 18.0 matches the
    training preprocessing.
    """
    occupancy = np.asarray(task["occupancy"], dtype=np.float32)
    start_xy = np.asarray(task["start_xy"], dtype=np.float32)
    goal_xy = np.asarray(task["goal_xy"], dtype=np.float32)
    yy, xx = np.mgrid[:GRID_SIZE, :GRID_SIZE]

    def point_map(point: np.ndarray) -> np.ndarray:
        # This inverts pixel_to_world; use the converted point as a continuous center
        # on the image grid.
        pixel_x = (point[0] - WORLD_MIN_M) / RESOLUTION_M
        pixel_y = (WORLD_MAX_M - point[1]) / RESOLUTION_M
        # Preserve subpixel centers so the network can use continuous start/goal
        # locations. Explicitly keep the output as float32.
        return np.exp(-((xx - pixel_x) ** 2 + (yy - pixel_y) ** 2) / 18.0).astype(np.float32)

    yaw = float(task["start_yaw"])
    return np.stack((occupancy, point_map(start_xy), point_map(goal_xy),
                     np.full_like(occupancy, math.sin(yaw)),
                     np.full_like(occupancy, math.cos(yaw))))


@torch.inference_mode()
def probability_map(model: XYProbabilityUNet, device: torch.device,
                    task: dict[str, object]) -> tuple[np.ndarray, float]:
    """Predict ``P_xy`` and measure one warmed-up end-to-end inference pass.

    ``inference_mode`` disables autograd metadata more thoroughly than ``no_grad``
    and is suitable for inference. CUDA operations are asynchronous by default, so
    synchronize before and after timing; otherwise only kernel-launch time is measured.
    Return probabilities rather than logits so C++ can use them directly as
    nonnegative discrete-sampling weights.
    """
    inputs = torch.from_numpy(network_input(task)[None]).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    probability = torch.sigmoid(model(inputs)[0, 0]).cpu().numpy().astype(np.float32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return probability, time.perf_counter() - started


@torch.inference_mode()
def warm_up_model(model: XYProbabilityUNet, device: torch.device, task: dict[str, object]) -> None:
    """Run an unrecorded inference pass to exclude first-use CUDA context and kernel setup."""
    probability_map(model, device, task)


def native_executable() -> str:
    """Locate the compiled OMPL executable without committing a platform-specific binary."""
    configured = os.environ.get("NEURAL_SST_XY_EXECUTABLE", NATIVE_EXECUTABLE)
    executable = shutil.which(configured) if not Path(configured).is_file() else configured
    if executable is None:
        raise FileNotFoundError(
            "Native SST executable not found. Build src/NeuralSSTXY.cpp with the two OMPL demo "
            "patch files, then set NEURAL_SST_XY_EXECUTABLE to its absolute path."
        )
    return executable


def run_native(task: dict[str, object], probability: np.ndarray, seed: int,
               xy_bias: float, model_time_s: float,
               tree_path: Path | None = None) -> dict[str, object]:
    """Write the C++ input protocol, run native SST once, and parse its final JSON line.

    The file layout must match ``NeuralSSTXY.cpp/read_input``: line one contains
    scalar settings, line two is a row-major flattened binary occupancy map, and line
    three is P_xy in the same index order. Subtract inference time from the 30-second
    total budget so comparisons across networks use a consistent end-to-end time limit.
    """
    start, goal = np.asarray(task["start_xy"]), np.asarray(task["goal_xy"])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as handle:
        input_path = Path(handle.name)
        # Pass a positive value even if inference exceeds the budget to keep the OMPL
        # termination condition valid.
        remaining_time = max(1e-6, MAX_TIME_S - model_time_s)
        handle.write(f"{seed} {xy_bias} {MAX_ITERATIONS} {remaining_time} "
                     f"{start[0]} {start[1]} {task['start_yaw']} {goal[0]} {goal[1]}\n")
        np.savetxt(handle, np.asarray(task["occupancy"], dtype=np.uint8).reshape(1, -1), fmt="%d")
        np.savetxt(handle, probability.reshape(1, -1), fmt="%.8g")
    try:
        # The native program may print OMPL logs; by convention, its final line is the
        # single machine-readable JSON record.
        command = [native_executable(), str(input_path)]
        if tree_path is not None:
            # The optional third program argument asks C++ to export the SST tree
            # retained when solve finishes.
            command.append(str(tree_path))
        output = subprocess.run(command, check=True, text=True,
                                capture_output=True).stdout.strip()
        return json.loads(output.splitlines()[-1])
    finally:
        # delete=False allows child processes on platforms such as Windows to open the
        # temporary file; it is unconditionally cleaned up afterward.
        input_path.unlink(missing_ok=True)


def world_to_image(x_m: float, y_m: float, scale: int) -> tuple[float, float]:
    """Convert metric world coordinates to continuous pixels in the enlarged Pillow image."""
    return ((x_m - WORLD_MIN_M) / RESOLUTION_M * scale,
            (WORLD_MAX_M - y_m) / RESOLUTION_M * scale)


def render_result(path: Path, task: dict[str, object], probability: np.ndarray,
                  result: dict[str, object], xy_bias: float) -> None:
    """Draw obstacles, an optional probability heatmap, the returned path, and endpoints.

    Display the heatmap only when probability sampling is enabled so the uniform
    ``xy_bias=0`` baseline is not mistaken for network-guided sampling. Normalize
    probability colors using the 1st and 99th percentiles in free space to reduce the
    impact of a few extreme pixels. Never overlay the heatmap on obstacles.
    """
    scale = 3
    occupancy = np.asarray(task["occupancy"], dtype=bool)
    # White denotes free space and black denotes obstacles. repeat enlarges with
    # nearest-neighbor interpolation so grid edges remain sharp.
    background = np.where(occupancy, 0, 255).astype(np.uint8)
    background = np.repeat(np.repeat(background, scale, axis=0), scale, axis=1)
    canvas = np.stack((background, background, background), axis=-1).astype(np.float32)
    if xy_bias != 0.0:
        free = ~occupancy
        # Percentile stretching reveals sampling preferences better than global min/max.
        low, high = np.percentile(probability[free], (1.0, 99.0))
        heat = np.clip((probability - low) / max(1e-8, high - low), 0.0, 1.0)
        heat = np.repeat(np.repeat(heat, scale, axis=0), scale, axis=1)
        free = np.repeat(np.repeat(free, scale, axis=0), scale, axis=1)
        # Increase the yellow-red overlay's alpha with heat; sqrt keeps low but nonzero
        # probabilities visible.
        layer = np.stack((np.full_like(heat, 255.0), 235.0 * np.sqrt(heat),
                          30.0 * (1.0 - heat)), axis=-1)
        alpha = 0.58 * heat
        canvas[free] = (1.0 - alpha[free, None]) * canvas[free] + alpha[free, None] * layer[free]
    image = Image.fromarray(canvas.astype(np.uint8), mode="RGB")
    drawing = ImageDraw.Draw(image)
    # C++ returns states as [x, y, yaw]; only planar positions are needed to draw the path.
    states = result["state_sequence"]
    if len(states) >= 2:
        drawing.line([world_to_image(state[0], state[1], scale) for state in states], fill=(0, 90, 255), width=3)
    for point, color in ((task["start_xy"], (0, 190, 0)), (task["goal_xy"], (235, 30, 30))):
        x, y = world_to_image(float(point[0]), float(point[1]), scale)
        drawing.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=(255, 255, 255), width=2)
    image.save(path, quality=95)


def render_tree_result(path: Path, result_image: Path, tree_path: Path,
                       result: dict[str, object]) -> None:
    """Draw the retained SST tree over the result image and emphasize the final path.

    The exported tree contains nodes and parent-child edges that survive SST pruning,
    including ancestors of active nodes. Pruned nodes and collision-rejected
    propagation attempts are not part of OMPL's final PlannerData, so this image is a
    snapshot of the tree when solve stops, not a complete sampling-history log.
    """
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    image = Image.open(result_image).convert("RGB")
    drawing = ImageDraw.Draw(image, "RGBA")
    vertices = tree["vertices"]
    edges = tree["edges"]
    scale = image.width / GRID_SIZE

    for source, destination in edges:
        start = world_to_image(vertices[source][0], vertices[source][1], scale)
        end = world_to_image(vertices[destination][0], vertices[destination][1], scale)
        drawing.line((start, end), fill=(20, 125, 95, 105), width=1)

    states = result["state_sequence"]
    if len(states) >= 2:
        route = [world_to_image(state[0], state[1], scale) for state in states]
        drawing.line(route, fill=(20, 65, 220, 245), width=max(2, round(scale)))

    image.save(path, quality=95)


def save_record(result_path: Path, task_path: Path, task: dict[str, object], result: dict[str, object],
                xy_bias: float, seed: int, model_time_s: float, image_path: Path) -> None:
    """Complete the native result as a reproducible experiment record and write JSON.

    Preserve the C++ counters and control sequences while adding input provenance,
    model time, image name, path length, and final-state error. Path length sums the
    Euclidean distances between consecutive discrete states. This is suitable for
    reporting, but is not a substitute for the vehicle's integrated arc length.
    """
    states = np.asarray(result["state_sequence"], dtype=np.float64)
    segments = result["control_segments"]
    record = {
        "task_file": str(task_path), "sample_id": task_path.stem, "xy_bias": xy_bias, "seed": seed,
        "start": {"x": float(task["start_xy"][0]), "y": float(task["start_xy"][1]), "yaw": float(task["start_yaw"])},
        "goal": {"x": float(task["goal_xy"][0]), "y": float(task["goal_xy"][1])},
        "model_time_ms": 1000.0 * model_time_s, "visualization": image_path.name,
        # C++ provides each control-segment duration; summing the empty sequence for
        # an unsolved task naturally yields zero.
        "duration_s": float(sum(segment["duration_s"] for segment in segments)),
        "path_length_m": float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum()) if len(states) >= 2 else 0.0,
        "final_error_m": float(np.linalg.norm(states[-1, :2] - task["goal_xy"])) if len(states) else None,
        **result,
    }
    result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run the full pipeline: validate arguments, load/warm up the model, plan each
    task, and save its result pair."""
    args = parse_args()
    if not 0.0 <= args.xy_bias <= 1.0:
        raise ValueError("--xy-bias must be in [0, 1]")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.tree_dir is not None:
        args.tree_dir.mkdir(parents=True, exist_ok=True)
    paths = task_paths(args)
    model, device = load_model(args.checkpoint)
    # Warm up with the first real task so model_time_ms is more comparable across tasks.
    warm_up_model(model, device, load_task(paths[0]))
    for task_path in paths:
        task = load_task(task_path)
        probability, model_time_s = probability_map(model, device, task)
        # Include bias in the filename so uniform and guided runs for the same task
        # can coexist without overwriting each other.
        run_name = f"{task_path.stem}-{args.xy_bias:g}"
        image_path = args.result_dir / f"{run_name}.jpg"
        result_path = args.result_dir / f"{run_name}.json"
        tree_path = args.tree_dir / f"{run_name}.tree.json" if args.tree_dir else None
        result = run_native(task, probability, args.seed, args.xy_bias, model_time_s, tree_path)
        render_result(image_path, task, probability, result, args.xy_bias)
        if tree_path is not None:
            render_tree_result(args.tree_dir / f"{run_name}.tree.jpg", image_path, tree_path, result)
        save_record(result_path, task_path, task, result, args.xy_bias, args.seed, model_time_s, image_path)
        print(f"{run_name}: success={result['success']} first_samples={result['first_solution_iterations']}")


if __name__ == "__main__":
    main()
