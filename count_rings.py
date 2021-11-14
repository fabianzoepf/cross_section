"""Count rings of a fish scale."""
from typing import Tuple

import numpy as np
from scipy import signal


def denoise(cross_section: np.ndarray) -> np.ndarray:
    """
    Denoise the cross section: apply bandpass filter

    Args:
        cross_section (np.ndarray): cross section of the scale

    Returns:
        cross_section (np.ndarray): denoised cross section of the scale
    """
    # remove mean
    mean = np.mean(cross_section).astype(np.uint8)
    cross_section -= mean
    
    # bandpass FIR filter
    filter_order = 51
    b = signal.firwin(filter_order, cutoff=[.15, .4], fs=1, pass_zero=False)
    cross_section = signal.lfilter(b, [1.0], cross_section)

    # compensate filter delay
    delay = int((filter_order - 1) / 2)
    tmp = np.zeros_like(cross_section)
    tmp = cross_section
    cross_section[:-delay] = tmp[delay:]

    return cross_section


def plot_data(cross_section: np.ndarray, filtered: np.ndarray,
        peaks: np.ndarray, diff: np.ndarray) -> None:
    """
    Plot cross section, detected rings and distances

    Args:
        cross_section (np.ndarray): cross section of the scale

        filtered (np.ndarray): filtered/denoised cross section

        peaks (np.ndarray): indices of detected peaks
        
        diff (np.ndarray): distances between peak indices
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.tight_layout()

    ax[0].title.set_text('Original Data')
    ax[0].plot(cross_section)

    ax[1].title.set_text('Filtered Data')
    ax[1].plot(filtered)
    ax[1].scatter(peaks, filtered[peaks], c='red', marker='x')

    ax[2].title.set_text('Diff between detected lines')
    ax[2].plot(diff)
    plt.show()


def main(cross_section: np.ndarray, plot: bool=False) -> Tuple[int, np.ndarray]:
    """
    Count the rings on a fish scales cross section

    Args:
        cross_section (np.ndarray): cross section of the scale

        plot (bool): flag to plot image. Blocks after plot is shown until plots are closed

    Returns:
        Returns:
        tuple containing

        - num_peaks (int): number of rings detected
        - diff (np.ndarray): distance of each ring to its neighbors in pixel
    """
    filtered = denoise(cross_section)

    pks = signal.find_peaks(filtered)[0]
    diff = np.diff(pks)

    if plot:
        plot_data(cross_section, filtered, pks, diff)

    return len(pks), diff


if __name__ == '__main__':
    import argparse
    import cv2
    import cross_section

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='image to be analyzed')
    parser.add_argument('--min_box_size', help='minimal bounding box height/width',
        default=200, required=False)
    args = parser.parse_args()

    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)

    lines = cross_section.main(image, args.min_box_size)

    for l in lines:
        rings = main(l, True)
