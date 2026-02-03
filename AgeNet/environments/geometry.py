from typing import List, Tuple
from matplotlib.path import Path


def do_intersect(points: List[Tuple[float, float]], obstacles) -> bool:
    # Generic geometric intersection test between a path and obstacles.
    
    path = Path(points)
    return any(path.intersects_bbox(obstacle) for obstacle in obstacles)
