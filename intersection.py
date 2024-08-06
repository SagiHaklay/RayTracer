import numpy as np

from material import Material

class Intersection:
    def __init__(self, point: np.ndarray, norm: np.ndarray, material: Material, surface_idx) -> None:
        self.point = point
        self.normal = norm
        self.material = material
        self.surface_idx = surface_idx