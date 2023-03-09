import numpy as np


def get_F(points1: np.ndarray, points2: np.ndarray):
    """
    Uses the coplanarity constraint. Turned into least squares problem by flattening F.

    Returns the 3x3 fundamental matrix.
    """

    # make 2D coords homogeneous
    points1 = _make_2D_coords_homogeneous(points1)
    points2 = _make_2D_coords_homogeneous(points2)

    # normalise coordinates to be zero centred and scaled to range [-1, 1]
    points1, points2, T = _normalise_coords(points1, points2)

    # obtain N many equations stacked in A
    A = np.asarray([np.kron(points2[i], points1[i]) for i in range(len(points1))])

    # Stacked coplanarity constraint Af = 0. Use SVD to find eigenvector which minimises equation.
    f = np.linalg.svd(A)

    # take singular vector corresponding to the smallest singular value
    f = f[2][-1]

    # reshape f with col-major order
    F = f.reshape((3, 3), order="F")

    # TODO: transform back to image space using transformation T

    # need to enforce rank=2 deficiency in F
    svd = np.linalg.svd(F)
    svd[1][-1] = 0
    F = svd[0] @ np.diag(svd[1]) @ svd[2]

    return F


def extract_R_b(E: np.ndarray):
    R, b = 0, 0
    return R, b


def _make_2D_coords_homogeneous(coords: np.ndarray):
    padded_coords = np.ones((coords.shape[0], coords.shape[1] + 1), dtype=int)
    padded_coords[:, :2] = coords
    return padded_coords


def _normalise_coords(coords1: np.ndarray, coords2: np.ndarray):
    # TODO: normalise coordinates to be zero centred and scaled to range [-1, 1]
    T = 0
    return coords1, coords2, T
