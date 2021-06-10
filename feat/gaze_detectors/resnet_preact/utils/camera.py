import cv2
import numpy as np


class Camera:

    def __init__(self, width, height, camera_matrix, dist_coeffs):
        self.width = width
        self.height = height
        self.camera_matrix = camera_matrix
        self.dist_coefficients = dist_coeffs

    def project_points(self, points3d, rvec=None, tvec=None) -> np.ndarray:
        assert points3d.shape[1] == 3
        if rvec is None:
            rvec = np.zeros(3, dtype=np.float)
        if tvec is None:
            tvec = np.zeros(3, dtype=np.float)
        points2d, _ = cv2.projectPoints(points3d, rvec, tvec,
                                        self.camera_matrix,
                                        self.dist_coefficients)
        return points2d.reshape(-1, 2)
