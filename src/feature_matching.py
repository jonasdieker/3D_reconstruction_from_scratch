import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.stats import binned_statistic

__all__ = [
    "compute_corners",
    "compute_descriptors",
    "compute_matches",
    "visualise_corners",
    "visualise_matches",
]


def compute_corners(I: np.ndarray, T: int):
    """
    inputs:
        - I (np.ndarray): image
        - T (int): threshold from Shi-Tomasi corner detection
    returns:
        - corners (np.ndarray [num_points x 2]): Coordinates of detected corners."""

    # specifying kernels in x and y and perform convolutions to find image gradients
    kernelx = (1 / 8) * np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernely = (1 / 8) * np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Jx = ndimage.convolve(I, kernelx)
    Jy = ndimage.convolve(I, kernely)

    # calculate squared gradients/gradient combination
    Jx_squared = np.square(Jx)
    Jy_squared = np.square(Jy)
    JxJy = np.multiply(Jx, Jy)

    # accumulate gradients in local neighbourhood W for each point
    kernel_window = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    Jx_2_acc = ndimage.convolve(Jx_squared, kernel_window)
    JxJy_acc = ndimage.convolve(JxJy, kernel_window)
    Jy_2_acc = ndimage.convolve(Jy_squared, kernel_window)

    # use Shi-Tomasi to calculate threshold value 'T' for each pixel
    corners = []
    for i in range(0, I.shape[1]):
        for j in range(I.shape[0]):
            M = [[Jx_2_acc[j, i], JxJy_acc[j, i]], [JxJy_acc[j, i], Jy_2_acc[j, i]]]
            threshold = np.trace(M) / 2 - 0.5 * np.sqrt(
                (np.trace(M)) ** -4 * np.linalg.det(M)
            )
            if threshold >= T:
                corners.append([i, j])
    corners = np.asarray(corners)
    return corners


def _get_submatrices(arr: np.ndarray):
    submatrices = []
    nrows, ncols = arr.shape
    assert nrows % 4 == 0
    assert ncols % 4 == 0
    cols = np.split(arr, 4, axis=1)
    for i in range(len(cols)):
        submatrices.extend(np.split(cols[i], 4, axis=0))
    return submatrices


def _get_histogram(orientation, magnitude):
    # calculates the orientation histogram based on 8 bins for all subgroups

    feature_vec = []
    for i in range(len(orientation)):
        bin_sum = binned_statistic(
            magnitude[i].flatten(),
            orientation[i].flatten(),
            "sum",
            bins=8,
            range=(0, 360),
        )
        feature_vec.extend(bin_sum[0])
    return feature_vec


def compute_descriptors(I: np.ndarray, corners: np.ndarray):
    """
    For each supplied corner point this function computes a 128 dimensional descriptor based
    on Histogram of Orientated Gradients (HOG).

    Inputs:
        - I (np.ndarray): Grayscale image.
        - corners (np.ndarray [num_corners x 2]): Coordinates of detected corners.
    Returns:
        - feature_vectors (list[list[float]]): 128-dim vector for each corner."""

    # specifying kernels in x and y and perform convolutions to find image gradients
    kernelx = (1 / 8) * np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernely = (1 / 8) * np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Jx = ndimage.convolve(I, kernelx)
    Jy = ndimage.convolve(I, kernely)

    # calculate squared gradients
    Jx_squared = np.square(Jx)
    Jy_squared = np.square(Jy)

    # pad squared gradients to find neighbourhoods even for point on edge of image
    padded_Jx_squared = np.zeros((Jx_squared.shape[0] + 16, Jx_squared.shape[1] + 16))
    padded_Jx_squared[
        8 : Jx_squared.shape[0] + 8, 8 : Jx_squared.shape[1] + 8
    ] = Jx_squared

    padded_Jy_squared = np.zeros((Jy_squared.shape[0] + 16, Jy_squared.shape[1] + 16))
    padded_Jy_squared[
        8 : Jy_squared.shape[0] + 8, 8 : Jy_squared.shape[1] + 8
    ] = Jy_squared

    padded_Jx = np.zeros((Jx.shape[0] + 16, Jx.shape[1] + 16))
    padded_Jx[8 : Jx.shape[0] + 8, 8 : Jx.shape[1] + 8] = Jx

    padded_Jy = np.zeros((Jy.shape[0] + 16, Jy.shape[1] + 16))
    padded_Jy[8 : Jy.shape[0] + 8, 8 : Jy.shape[1] + 8] = Jy

    # find descriptor for each keypoint
    feature_vectors = []
    for i in range(len(corners)):
        x = corners[i, 0] + 8
        y = corners[i, 1] + 8

        # find magnitude and orientation of gradient for each neighbourhood W (size 16x16) around keypoint
        magn = np.sqrt(
            padded_Jx_squared[y - 8 : y + 8, x - 8 : x + 8]
            + padded_Jy_squared[y - 8 : y + 8, x - 8 : x + 8]
        )
        orient = (
            np.arctan2(
                padded_Jy[y - 8 : y + 8, x - 8 : x + 8],
                padded_Jx[y - 8 : y + 8, x - 8 : x + 8],
            )
            * 180
        ) / np.pi

        # split into sub-matrices
        orient_subgroups = _get_submatrices(orient)
        magn_subgroups = _get_submatrices(magn)

        # compute orientation histograms
        feature_vector = _get_histogram(orient_subgroups, magn_subgroups)

        # normalise feature_vector and append (make sure we dont divide by zero)
        if np.sum(feature_vector) == 0:
            feature_vectors.append(feature_vector)
        else:
            feature_vectors.append(np.divide(feature_vector, np.sum(feature_vector)))

    return feature_vectors


def compute_matches(descr1, descr2, ratio=0.6):
    """
    Uses Lowe's Ratio Test to find query for feature q:
        1. Find two closest descriptors to q according to euclidean norm
        2. Test if distance to smallest match is <T (threshold)
        3. Accept match if d(q,p1)/d(q,p2) < 1/2 (best feature much closer than second feature)
    Inputs:
        - descr1 (list[list[float]]): 128-dim vector for each key point for image 1.
        - descr2 same as descr1 for image 2.
    Returns:
        - matches (list[list[int]]): List of lists matching indices of corresponding key points.
    """

    matches = []
    for i in range(len(descr1)):
        temp_lst = []
        for j in range(len(descr2)):
            # using L2 norm between descriptor pairs
            temp_lst.append(
                np.linalg.norm(np.asarray(descr1[i]) - np.asarray(descr2[j]), ord=2)  # type: ignore
            )
        # print(max(temp_lst), min(temp_lst))
        p1 = min(temp_lst)
        p1_idx = temp_lst.index(p1)
        temp_lst[p1_idx] = max(temp_lst)
        p2 = min(temp_lst)

        if p2 != 0 and p1 / p2 <= ratio:
            matches.append([i, p1_idx])

    return matches


def visualise_corners(image, corners):
    print(f"Detected {len(corners)} corners.")
    fig = plt.figure(figsize=(5, 5), dpi=100)
    plt.imshow(image, cmap="gray")
    plt.scatter(np.array(corners)[:, 0], np.array(corners)[:, 1])
    plt.show()


def visualise_matches(I1, I2, corners1, corners2, matches, title="Matches"):
    # get corners indices for set 1 and 2
    matches1 = np.array(matches, dtype=int)[:, 0]
    matches2 = np.array(matches, dtype=int)[:, 1]

    # find points from corners using indices
    points1 = corners1[matches1]
    points2 = corners2[matches2]

    fig = plt.figure(figsize=(9, 4), dpi=100)
    # horizontally stack images to display side by side seamlessly
    total = np.hstack((I1, I2))
    width_offset = I1.shape[1]
    dpi = 100
    plt.imshow(total, cmap="gray")
    plt.scatter(points1[:, 0], points1[:, 1], marker="x")  # type: ignore
    plt.scatter(points2[:, 0] + width_offset, points2[:, 1], marker="x")  # type: ignore
    for i in range(len(matches)):
        plt.plot(
            [points1[i, 0], points2[i, 0] + width_offset],
            [points1[i, 1], points2[i, 1]],
            "ro-",
        )
    plt.title(title)
    plt.show()
