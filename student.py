import numpy as np
import cv2
import random

def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points. See the handout, Q5
    of the written questions, or the lecture slides for how to set up these
    equations.

    Don't forget to set M_34 = 1 in this system to fix the scale.

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points
    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
    """
    ######################
    # Do not change this #
    ######################

    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in our scanning setup
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters)
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)
    points3d = np.array(points3d)

    ########################
    # TODO: Your code here #
    new_points = np.reshape(points2d, (-1,))

    mat = []
    for i in range(len(points3d)):
        mat.append([
                points3d[i][0],
                points3d[i][1],
                points3d[i][2],
                1,
                0,
                0,
                0,
                0,
                -points3d[i][0] * points2d[i][0],
                -points3d[i][1] * points2d[i][0],
                -points3d[i][2] * points2d[i][0],
            ])
        mat.append([
                0,
                0,
                0,
                0,
                points3d[i][0],
                points3d[i][1],
                points3d[i][2],
                1,
                -points3d[i][0] * points2d[i][1],
                -points3d[i][1] * points2d[i][1],
                -points3d[i][2] * points2d[i][1],
            ])
    M = np.zeros(12)
    M[:11] = np.linalg.lstsq(mat, new_points, rcond=None)[0]
    M[11] = 1
    M = np.reshape(M, (3, 4))
    ########################

    return M

def normalize_coordinates(points):
    """
    ============================ EXTRA CREDIT ============================
    Normalize the given Points before computing the fundamental matrix. You
    should perform the normalization to make the mean of the points 0
    and the average magnitude 1.0.

    The transformation matrix T is the product of the scale and offset matrices

    Offset Matrix
    Find c_u and c_v and create a matrix of the form in the handout for T_offset

    Scale Matrix
    Subtract the means of the u and v coordinates, then take the reciprocal of
    their standard deviation i.e. 1 / np.std([...]). Then construct the scale
    matrix in the form provided in the handout for T_scale

    :param points: set of [n x 2] 2D points
    :return: a tuple of (normalized_points, T) where T is the [3 x 3] transformation
    matrix
    """
    ########################
    # TODO: Your code here #
    scale_matrix = np.eye(3)
    offset_matrix = np.eye(3)

    mean = np.mean(points, axis=0)
    std = 1 / np.std(points[:, :2], axis=0)
    scale_matrix[0][0] = std[0]
    scale_matrix[1][1] = std[1]
    offset_matrix[0][2] = -mean[0]
    offset_matrix[1][2] = -mean[1]

    T = np.matmul(scale_matrix, offset_matrix)
    points = T @ points.T
    points = points.T

    ########################
    # This is a placeholder with the identity matrix for T replace with the
    # real transformation matrix for this set of points

    return points, T

def estimate_fundamental_matrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Try to implement this function as efficiently as possible. It will be
    called repeatedly for part IV of the project

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
    """
    ########################
    # TODO: Your code here #
    n = points1.shape[0]

    x1 = np.concatenate((points1, np.ones(n).reshape(-1, 1)), axis=1)
    x2 = np.concatenate((points2, np.ones(n).reshape(-1, 1)), axis=1)

    x1, mat_1 = normalize_coordinates(x1) 
    x2, mat_2 = normalize_coordinates(x2)

    x_u = np.multiply(x1, x2[:, 0].reshape(-1, 1))
    x_v = np.multiply(x1, x2[:, 1].reshape(-1, 1))
    A = np.concatenate((x_u, x_v, x1), axis=1)

    U, S, V = np.linalg.svd(A)
    F_matrix = V[-1].reshape((3,3))

    U, S, V = np.linalg.svd(F_matrix)
    S[-1] = 0
    F_matrix = U @ np.diagflat(S) @ V
    F_matrix = mat_2.T @ F_matrix @ mat_1
    F_matrix /= F_matrix[-1][-1]
    ########################

    # This is an intentionally incorrect Fundamental matrix placeholder
    return F_matrix

def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Run RANSAC for num_iters.

    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points from two pictures. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    best_Fmatrix is the [3 x 3] fundamental matrix, inliers1 and inliers2 are
    the [M x 2] corresponding points (some subset of matches1 and matches2) that
    are inliners with respect to best_Fmatrix

    For this section, use RANSAC to find the best fundamental matrix by randomly
    sampling interest points. You would call the function that estimates the 
    fundamental matrix (either the "cheat" function or your own 
    estimate_fundamental_matrix) iteratively within this function.

    If you are trying to produce an uncluttered visualization of epipolar lines,
    you may want to return no more than 30 points for either image.

    :return: best_Fmatrix, inliers1, inliers2
    """
    # DO NOT TOUCH THE FOLLOWING LINES
    random.seed(0)
    np.random.seed(0)
    
    ########################
    # TODO: Your code here #

    PassThreshold = 1e-2
    num_sample = 8
    n = matches1.shape[0]
    best_inlier_count = 0
    best_Fmatrix = estimate_fundamental_matrix(matches1[0:9, :], matches2[0:9, :])
    inliers_a = matches1[0:29, :]
    inliers_b = matches2[0:29, :]

    for i in range(0, num_iters):
        RandIdx = np.random.randint(n, size=num_sample)

        estimate_Fmatrix = estimate_fundamental_matrix(
            matches1[RandIdx, :], matches2[RandIdx, :]
        )
        # estimate_Fmatrix, _ = cv2.findFundamentalMat(
        #     matches1[RandIdx], matches2[RandIdx], cv2.FM_8POINT, 1e10, 0, 1
        # )

        inlier_count = 0
        curr_inliers1 = []
        curr_inliers2 = []

        for j in range(n):
            new_matches1 = np.append(matches1[j, :], 1)
            new_matches2 = np.append(matches2[j, :], 1)
            error = np.matmul(new_matches2.T, estimate_Fmatrix)
            error = np.matmul(error, new_matches1)

            if abs(error) < PassThreshold:
                curr_inliers1.append([matches1[j, 0], matches1[j, 1]])
                curr_inliers2.append([matches2[j, 0], matches2[j, 1]])
                inlier_count += 1

        if inlier_count > best_inlier_count:
            best_Fmatrix = estimate_Fmatrix
            best_inlier_count = inlier_count
            inliers_a = curr_inliers1
            inliers_b = curr_inliers2

    inliers_a = np.asarray(inliers_a)
    inliers_b = np.asarray(inliers_b)
    ########################

    # Your RANSAC loop should contain a call to 'estimate_fundamental_matrix()'
    # that you wrote for part II.

    return best_Fmatrix, inliers_a, inliers_b

def matches_to_3d(points1, points2, M1, M2):
    """
    Given two sets of points and two projection matrices, you will need to solve
    for the ground-truth 3D points using np.linalg.lstsq(). For a brief reminder
    of how to do this, please refer to Question 5 from the written questions for
    this project.


    :param points1: [N x 2] points from image1
    :param points2: [N x 2] points from image2
    :param M1: [3 x 4] projection matrix of image2
    :param M2: [3 x 4] projection matrix of image2
    :return: [N x 3] NumPy array of solved ground truth 3D points for each pair of 2D
    points from points1 and points2
    """
    ########################
    # TODO: Your code here #

    # Fill in the correct shape
    #points3d = [np.zeros((_, _))]
    points3d = []
    for i in range(points1.shape[0]):
        A = []
        A.append(
            [
                (M1[0][0] - M1[2][0] * points1[i][0]),
                (M1[0][1] - M1[2][1] * points1[i][0]),
                (M1[0][2] - M1[2][2] * points1[i][0]),
            ]
        )
        A.append(
            [
                (M1[1][0] - M1[2][0] * points1[i][1]),
                (M1[1][1] - M1[2][1] * points1[i][1]),
                (M1[1][2] - M1[2][2] * points1[i][1]),
            ]
        )
        A.append(
            [
                (M2[0][0] - M2[2][0] * points2[i][0]),
                (M2[0][1] - M2[2][1] * points2[i][0]),
                (M2[0][2] - M2[2][2] * points2[i][0]),
            ]
        )
        A.append(
            [
                (M2[1][0] - M2[2][0] * points2[i][1]),
                (M2[1][1] - M2[2][1] * points2[i][1]),
                (M2[1][2] - M2[2][2] * points2[i][1]),
            ]
        )
        b = [
            M1[2][3] * points1[i][0] - M1[0][3],
            M1[2][3] * points1[i][1] - M1[1][3],
            M2[2][3] * points2[i][0] - M2[0][3],
            M2[2][3] * points2[i][1] - M2[1][3],
        ]
        points3d.append(np.linalg.lstsq(A, b, rcond=None)[0])

    # Solve for ground truth points

    ########################

    return points3d
