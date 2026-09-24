"""Geometric conversions between task JSON and Neural-SST world coordinates.

The original ``random_2d`` dataset represents maps as 224×224 image pixels, with
``(0, 0)`` at the upper-left and row indices increasing downward. The planner instead
uses metric coordinates centered on the map, with the Y axis pointing upward. This
module is the sole conversion point between the two representations so the Python
network input, C++ planner input, and final visualization all share the same obstacle
and coordinate conventions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 224 pixels at 0.05 m per pixel exactly cover the square workspace [-5.6, 5.6] m.
GRID_SIZE = 224
RESOLUTION_M = 0.05
WORLD_MIN_M = -5.6
WORLD_MAX_M = 5.6


def rasterize_obstacles(env_dims: list[int], rectangles: list[list[int]], circles: list[list[int]]) -> np.ndarray:
    """Rasterize the dataset's rectangular and circular obstacles into a boolean map.

    All input geometry uses pixel units. Rectangles use ``[left, top, width, height]``
    and circles use ``[center_x, center_y, radius]``. The right/bottom rectangle
    boundaries are open intervals, consistent with NumPy and the data generator;
    circle boundaries are included in the obstacle using ``<=``. The returned array
    has shape ``[H, W]`` with ``True`` denoting occupied cells. This meaning is passed
    unchanged to the neural network, native planner, and visualization code.
    """
    width, height = map(int, env_dims)
    if (width, height) != (GRID_SIZE, GRID_SIZE):
        raise ValueError("This demo expects env_dims = [224, 224]")
    # Start with a free map; writing each obstacle type in turn naturally preserves
    # occupancy where shapes overlap.
    occupancy = np.zeros((height, width), dtype=bool)
    for x, y, rect_width, rect_height in rectangles:
        # Clip to image bounds to support edge-touching or slightly out-of-range source
        # data and prevent negative-index wraparound.
        occupancy[max(0, int(y)):min(height, int(y + rect_height)),
                  max(0, int(x)):min(width, int(x + rect_width))] = True
    # ogrid creates broadcastable row/column coordinates and avoids constructing two
    # full coordinate grids for every circle.
    yy, xx = np.ogrid[:height, :width]
    for center_x, center_y, radius in circles:
        # Keep squared distances to avoid square roots and match the circular pixel-mask
        # definition exactly.
        occupancy[(xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2] = True
    return occupancy


def pixel_to_world(pixel_xy: list[int]) -> np.ndarray:
    """Convert a dataset coordinate ``[pixel_x, pixel_y]`` into ``[x_m, y_m]``.

    The X axis only needs translation to the map center. The Y axis must also be
    flipped because image rows increase downward while OMPL world coordinates increase
    upward. The result uses ``float32`` to match the network input while still precisely
    representing the 5 cm grid resolution.
    """
    px, py = map(float, pixel_xy)
    return np.asarray([(px - GRID_SIZE / 2.0) * RESOLUTION_M,
                       (GRID_SIZE / 2.0 - py) * RESOLUTION_M], dtype=np.float32)


def load_task(path: Path) -> dict[str, object]:
    """Read a task file and build the standard task dictionary shared by later stages.

    In addition to retaining the original obstacle fields for traceability, this
    function caches the rasterized ``occupancy``, the original pixel coordinates of
    the start and goal, and their converted world coordinates. The start yaw is kept
    in radians. Goal yaw is unconstrained for XY-only tasks, so it is neither read nor
    returned.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    occupancy = rasterize_obstacles(payload["env_dims"], payload["rectangle_obstacles"],
                                    payload["circle_obstacles"])
    return {"name": path.stem, "path": str(path),
            "rectangle_obstacles": payload["rectangle_obstacles"],
            "circle_obstacles": payload["circle_obstacles"], "occupancy": occupancy,
            "start_pixel": payload["start"], "goal_pixel": payload["goal"],
            "start_xy": pixel_to_world(payload["start"]), "goal_xy": pixel_to_world(payload["goal"]),
            "start_yaw": float(payload["start_yaw_rad"])}
