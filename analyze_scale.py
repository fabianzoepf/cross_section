"""Analyze fish scales: extract their cross section and count the rings."""
import argparse
import os
import glob
import imghdr

import pandas as pd
import cv2
from tqdm import tqdm

import cross_section
import count_rings


def export(scale_data: pd.DataFrame, out_file: str) -> None:
    """
    Export date to file. File type can either be json or feather: this
    depends on the extenstion of the out_file argument.

    Args:
        scale_data (pd.DataFrame): data to be written to file

        out_file (str): path to the output file. Only json or feather extension allowed
    """
    _, ext = os.path.splitext(out_file)

    if ext == '.json':
        with open(out_file, 'w') as f:
            scale_data.to_json(f, orient='records')
    elif ext == '.feather':
        scale_data.to_feather(out_file)
    else:
        raise ValueError('Output file type not available. Use json or feather.')


def convert_to_mm(data: float, dpi: int, power: int=1) -> pd.DataFrame:
    """
    Convert pixel data to mm.

    Args:
        data (float): data in pixel

        dpi (int): dots per inch

        power (int): power to be used to convert higher dimensions

    Returns:
        data (float): data in mm
    """
    inches = data / (dpi**power)
    mm = (25.4**power) * inches

    return mm


def get_files(path: str) -> list:
    """
    Get all files image files within in the path.

    Args:
        path (str): path to image or folder

    Returns:
        files (list): paths to all image files in folder and subfolders
    """
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'))
    else:
        files = []

    # filter out non-images
    image_files = [x for x in files if imghdr.what(x) is not None]

    return image_files


def main(path: str, min_box_size: int, out_file: str, dpi: int=0) -> None:
    """
    Analyze fish scales: extract their cross section and count the rings as well as the pixel
    distance between detected rings.
    Data can be exported to either json or feather files for further use.

    Args:
        path (str): path to input images. Can be a single image or a folder, in the latter case all
                    images contained in the folder and subfolders will be analyzed.

        min_box_size (int): minimum bounding box size of a scale in the image in pixel.
                            Used to filter out noise.

        out_file (str): path to the output file. Only json or feather extension allowed.

        dpi (int): dots per inch, if specified, all output data will be in mm, otherwise in pixel
    """
    files = get_files(path)

    scale_data = []

    for image_path in tqdm(files):
        #TODO: logging if something fails
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        cross_sections = cross_section.main(image, min_box_size)

        for cs, cnt in cross_sections:
            num_rings, diff = count_rings.main(cs)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)

            if 0 < dpi:
                diff = convert_to_mm(diff, dpi)
                area = convert_to_mm(area, dpi, power=2)
                perimeter = convert_to_mm(perimeter, dpi)

            scale_data.append([image_path, cs, num_rings, diff, area, perimeter])

    scale_data = pd.DataFrame(scale_data,
        columns=['image', 'gray values', '# rings',
                 'ring distance', 'area', 'perimeter'])

    export(scale_data, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to image or folder to be analyzed')
    parser.add_argument('--min_box_size', help='minimal bounding box height/width',
        type=int, default=200, required=False)
    parser.add_argument('--out_file',
        help='output path, determines the type also. Use .json or .feather extension',
        default=os.path.join(os.getcwd(),'scale_data.feather'), required=False)
    parser.add_argument('--dpi',
        help='dots per inch. If specified, all output data will be in mm, otherwise in pixel',
        type=int, default=0, required=False)
    args = parser.parse_args()

    main(args.path, args.min_box_size, args.out_file, args.dpi)
