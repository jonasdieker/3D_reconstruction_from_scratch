import numpy as np

from src.EssentialMatrix import CalcEssentialMatrix


class DepthReconstruction:
    """Takes corresponding pairs of points and rotation and translation in order to reconstruct
    the 3D information."""

    def __init__(self, pts1, pts2, R, T):
        """
        Inputs:
            - pts1: coordinates in image 1
            - pts2: corresponding coordinates in image 2
            - R: rotation of camera 2 wrt camera 1
            - T: translation of camera 2 wrt camera 1"""

        self.pts1 = CalcEssentialMatrix._make_2D_coords_homogeneous(pts1)
        self.pts2 = CalcEssentialMatrix._make_2D_coords_homogeneous(pts2)
        self.R = R
        self.T = T
        self.l = None

    def _depth_reconstruction(self) -> np.ndarray:
        """Uses the geometric solution to recover the depth of points.
        Returns:
            - l: scaling for each point and T for point set 1.
                X1 = l1 * R @ x1 + l[-1]*T
        """

        # Build the M matrix of the linear system
        n = self.pts1.shape[0]
        M = np.zeros((3 * n, n + 1))
        for i, (coord1, coord2) in enumerate(zip(self.pts1, self.pts2)):
            M[3 * (i) : 3 * (i + 1), i] = (
                DepthReconstruction._convert_to_skew_symmetric(coord2) @ self.R @ coord1
            )

        M_last_col = [
            DepthReconstruction._convert_to_skew_symmetric(coord) @ self.T
            for coord in self.pts2
        ]
        M_last_col = np.asarray(M_last_col).flatten()
        M[:, -1] = M_last_col

        # scaling vector is singular vector corresponding to smallest singular value
        U, _, VT = np.linalg.svd(M)
        l = VT[-1] / np.linalg.norm(VT[-1], ord=2)
        return l

    def get_3D_pts(self) -> np.ndarray:
        """Uses scaling vector l and point set 1 to recover the 3D position of points."""

        self.l = self._depth_reconstruction()

        pts_3D = np.asarray(
            [
                self.l[i] * self.R @ pt + self.l[-1] * self.T
                for i, pt in enumerate(self.pts1)
            ]
        )
        return pts_3D

    @staticmethod
    def _convert_to_skew_symmetric(coord: list[int]):
        """Takes homegeneous coordinate and turns it into skew-symmetric matrix."""

        return np.array(
            [
                [0, -coord[2], coord[1]],
                [coord[2], 0, -coord[0]],
                [-coord[1], coord[0], 0],
            ]
        )
