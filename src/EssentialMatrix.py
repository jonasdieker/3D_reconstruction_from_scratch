import numpy as np


class CalcEssentialMatrix:
    """Takes two sets of corresponding points and calculates the essential matrix E."""

    def __init__(self, pts1, pts2):

        self.pts1 = self._make_2D_coords_homogeneous(pts1)
        self.pts2 = self._make_2D_coords_homogeneous(pts2)
        self.E = None

    def calc_E(self):
        """
        Uses the coplanarity constraint to calculate E.
        Turned into least squares problem by first flattening E computing its values then reshaping.

        Returns the 3x3 essential matrix.
        """

        # # normalise coordinates to be zero centred and scaled to range [-1, 1]
        # pts1, _ = self._map_coords_to_interval(self.pts1, (-1, 1))
        # pts2, _ = self._map_coords_to_interval(self.pts2, (-1, 1))
        pts1, pts2 = self.pts1, self.pts2

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
        self.E = E
        return E

    def extract_R_T(self):
        """Two possible solutions -> only one provides positive depth.
        Therefore try both and see which one returns a legal depth."""

        U, Sig, VT = np.linalg.svd(self.E)  # type: ignore
        R_z_90 = np.zeros((3, 3))
        R_z_90[0, 1], R_z_90[1, 0], R_z_90[2, 2] = 1, -1, 1

        # two possible solutions
        T_hat1, R1 = U @ R_z_90 @ np.diag(Sig) @ U.T, U @ R_z_90.T @ VT
        T_hat2, R2 = (
            U @ np.linalg.inv(R_z_90) @ np.diag(Sig) @ U.T,
            U @ np.linalg.inv(R_z_90).T @ VT,
        )

        T1 = np.array([T_hat1[1, 0], T_hat1[0, 2], T_hat1[2, 1]])
        T2 = np.array([T_hat2[1, 0], T_hat2[0, 2], T_hat2[2, 1]])

        # check which solution yields a positive depth
        test_pt = self.pts1[0]
        X1 = R1 @ test_pt + T1
        X2 = R2 @ test_pt + T2

        return (R1, T1) if X1[-1] >= 0 else (R2, T2)

    @staticmethod
    def _make_2D_coords_homogeneous(coords: np.ndarray):
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
