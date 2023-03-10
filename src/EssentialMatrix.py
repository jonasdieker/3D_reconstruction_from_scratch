import numpy as np


class EssentialMatrix:
    def __init__(self, pts1, pts2):

        # make 2D coords homogeneous
        pts1 = self._make_2D_coords_homogeneous(pts1)
        pts2 = self._make_2D_coords_homogeneous(pts2)
        self.pts1, self.pts2 = pts1, pts2

    def calc_E(self):
        """
        Uses the coplanarity constraint. Turned into least squares problem by flattening F.

        Returns the 3x3 fundamental matrix.
        """

        # normalise coordinates to be zero centred and scaled to range [-1, 1]
        pts1, _ = self._map_coords_to_interval(self.pts1, (-1, 1))
        pts2, _ = self._map_coords_to_interval(self.pts2, (-1, 1))

        # obtain N many equations stacked in A
        A = np.asarray([np.kron(pts2[i], pts1[i]) for i in range(len(pts1))])

        # Stacked coplanarity constraint Ae = 0. Use SVD to find eigenvector which minimises equation.
        e = np.linalg.svd(A)

        # take singular vector corresponding to the smallest singular value
        e = e[2][-1]

        # reshape f with col-major order
        E = e.reshape((3, 3), order="F")

        # TODO: transform back to image space using transformation T

        # need to enforce rank=2 deficiency in E and [1, 1, 0] singular values.
        svd = np.linalg.svd(E)
        svd[1][0], svd[1][1], svd[1][2] = 1, 1, 0
        E = svd[0] @ np.diag(svd[1]) @ svd[2]
        return E

    def extract_R_b(self, E: np.ndarray):
        R, b = 0, 0
        return R, b

    def _make_2D_coords_homogeneous(self, coords: np.ndarray):
        padded_coords = np.ones((coords.shape[0], coords.shape[1] + 1), dtype=int)
        padded_coords[:, :2] = coords
        return padded_coords

    def _map_coords_to_interval(self, coords: np.ndarray, interval: tuple):
        """Normalises coordinates to be zero centred and scaled to interval e.g. [-1, 1].

        It returns the transformed coordinates and the original interval in order perform
        reverse transformation."""

        if coords.shape[-1] == 3:
            coords = coords[:, :2]

        max = coords.max(axis=0)
        min = coords.min(axis=0)
        orig_interval = (min, max)

        transformed_coords = ((interval[1] - interval[0]) / (max - min)) * (
            coords - min
        ) + interval[0]

        transformed_coords = self._make_2D_coords_homogeneous(transformed_coords)

        return transformed_coords, orig_interval
