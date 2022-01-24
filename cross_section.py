"""Extract cross sections of detected objects along their main axis through their origin."""
from typing import Tuple

import numpy as np
from scipy import ndimage
import cv2
from sklearn.decomposition import PCA


def denoise(binary_image: np.ndarray) -> np.ndarray:
    """
    Denoise a binary image by closing and opening.

    Args:
        binary image (np.ndarray): input binary image

    Returns:
        np.ndarray: denoised binary image
    """

    closing_kernel = np.ones((30, 30), np.uint8)
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, closing_kernel)

    opening_kernel = np.ones((60, 60), np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, opening_kernel)

    return opened


def threshold_image(image: np.ndarray) -> np.ndarray:
    """
    Create a binary image using an adaptive threshold.

    Args:
        image (np.ndarray): input image

    Returns:
        np.ndarray: binary image
    """
    binary_image = cv2.adaptiveThreshold(image, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    return denoise(binary_image)


def get_sections(binary_image: np.ndarray, min_box_size: int) -> Tuple[list, list]:
    """
    Extract contours & bounding boxes from binarized image.

    Args:
        binary_image (np.ndarray): input binary image

        min_box_size (int): bounding boxes with smaller width or height get suppressed

    Returns:
        tuple containing

        - non_suppressed_contours (list): list of contours
        - bounding_boxes (list): list of bounding boxes
    """
    _, contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    bounding_boxes = []
    non_suppressed_contours = []
    for cntr in contours:
        box = cv2.boundingRect(cntr)

        # suppress small boxes
        if box[2] > min_box_size and box[3] > min_box_size:
            bounding_boxes.append(cv2.boundingRect(cntr))
            non_suppressed_contours.append(cntr)

    return non_suppressed_contours, bounding_boxes


def get_orientation(contour: np.ndarray) \
        -> Tuple[np.float32, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the orientation of a contour using principal component analysis.

    Args:
        contour (np.ndarray): input contour

    Returns:
        tuple containing

        - angle (float): orientation angle in rad
        - mean (np.ndarray): center of mass
        - eigenvector (np.ndarray): eigenvector of the contour
        - eigenvalue (np.ndarray): eigenvalue of the contour
    """
    contour = np.squeeze(contour).astype(np.float32)

    # principal component analysis
    pca = PCA(n_components=1).fit(contour)

    mean = pca.mean_.astype(int)
    eigenvector = pca.components_[0]
    eigenvalue = pca.explained_variance_[0]
    angle = np.arctan2(eigenvector[1], eigenvector[0])

    return (angle, mean, eigenvector, eigenvalue)


def get_cross_section(section: np.ndarray, offset: Tuple[int, int],
                      start: np.ndarray, angle: np.float32, avg: int=1) -> np.ndarray:
    """
    Extract a cross section of an image given by a point and angle.

    Args:
        section (np.ndarray): input image part

        offset (Tuple[int, int]): offset of the upper left image part
                                  edge relative to the complete image

        start (np.ndarray): point to define the cross section

        angle (np.ndarray): angle to define the cross section

        avg (int): average values over +- (int-1)/2 lines next to the cross section

    Returns:
        origin (np.ndarray): image part which lies in the line through point with given angle
    """
    rotation_center_x = start[0] - offset[0]
    rotation_center_y = start[1] - offset[1]

    # put rotation center in center of the section to rotate correctly
    pad_x = [section.shape[1] - rotation_center_x, rotation_center_x]
    pad_y = [section.shape[0] - rotation_center_y, rotation_center_y]
    padded_section = np.pad(section, [pad_y, pad_x], 'constant')

    rotated = ndimage.rotate(padded_section, np.rad2deg(angle), reshape=False)

    center_v = int(rotated.shape[0] / 2)
    offset = int(avg - 1 / 2)

    assert offset >= 0

    # average over (avg - 1) lines
    cross_section = np.mean(rotated[center_v - offset:center_v + offset, :], axis=0)
    cross_section = np.trim_zeros(cross_section)

    return cross_section


def plot_image(image: np.ndarray, plot_data: Tuple[np.ndarray, np.ndarray,
                    Tuple[int, int, int, int], np.ndarray]) -> None:
    """
    Plot the image with bounding boxes, contours and cross sections. Blocks
    after plotting until key is pressed

    Args:
        image (np.ndarray): input image

        plot data (Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], np.ndarray]:
            tuple containing

                - origin (np.ndarray): origin of the object
                - eigenvector (np.ndarray): eigenvector of the object
                - bounding_box (Tuple[int, int, int, int]): bounding box of the object
                - contour (np.ndarray): contour of the object
    """
    # convert image to color
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for i, (origin, eigenvec, eigenval, box, cntr) in enumerate(plot_data):
        x, y, w, h = box
        # plot bounding box
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.putText(rgb, str(i), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        # plot contour
        cv2.drawContours(rgb, cntr, -1, (0, 255, 0), 8)
        # plot extracted lines
        m = eigenvec[1] / eigenvec[0]
        start_x = x
        end_x = x + w
        start_y = int((start_x - origin[0]) * m + origin[1])
        end_y = int((end_x - origin[0]) * m + origin[1])

        if min(start_y, end_y) < y or max(start_y, end_y) > (y + h):
            m = eigenvec[0] / eigenvec[1]
            start_y = y
            end_y = y + h
            start_x = int((start_y - origin[1]) * m + origin[0])
            end_x = int((end_y - origin[1]) * m + origin[0])

        cv2.line(rgb, (start_x, start_y), (end_x, end_y), (255, 0, 0), 4) 

    rgb = cv2.resize(rgb, (0,0), fx=0.2, fy=0.2)
    cv2.imshow('image', rgb)
    cv2.waitKey(0)


def main(image: np.ndarray, min_box_size: int, plot: bool=False) -> list:
    """
    Extract cross sections of each object in the image, where the origin of the
    cross section is the average center of all circles/ellipses in the object
    and the cross section is extracted along the main principal component.

    Args:
        image (np.ndarray): input image

        min_box_size (int): objects with bounding boxes with smaller width or height get suppressed

        plot (bool): flag to plot image. Blocks after plot is shown until key is pressed

    Returns:
        cross_sections (list): list of cross sections of detected objects & their contours
    """
    mask = threshold_image(image)
    contours, bounding_boxes = get_sections(mask, min_box_size)

    # invert image
    inverted = cv2.bitwise_not(image)
    # black out background
    inverted[mask == 0] = 0

    cross_sections = []
    plot_data = []
    for (cntr, box) in zip(contours, bounding_boxes):
        x, y, w, h = box
        section = inverted[y:y+h, x:x+w]

        angle, origin, eigenvec, eigenval = get_orientation(cntr)

        line = get_cross_section(section, (x, y), origin, angle, avg=3)

        cross_sections.append((line, cntr))

        if plot:
            plot_data.append((origin, eigenvec, eigenval, box, cntr))

    if plot:
        plot_image(image, plot_data)

    return cross_sections


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='image to be analyzed')
    parser.add_argument('--min_box_size', help='minimal bounding box height/width',
        default=200, required=False)
    args = parser.parse_args()

    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)

    lines = main(image, args.min_box_size, True)
