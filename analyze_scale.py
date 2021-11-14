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


def export(scale_data: list, out_file: str) -> None:
    """
    Export date to file. File type can either be json or feather: this
    depends on the extenstion of the out_file argument.

    Args:
        scale_data (list): data to be written to file

        out_file (str): path to the output file. Only json or feather extension allowed
    """

    scale_data = pd.DataFrame(scale_data,
        columns=['image', 'gray values', '# rings', 'ring distance'])

    _, ext = os.path.splitext(out_file)

    if ext == '.json':
        with open(out_file, 'w') as f:
            scale_data.to_json(f, orient='records')
    elif ext == '.feather':
        scale_data.to_feather(out_file)
    else:
        raise ValueError('Output file type not available. Use json or feather.')


def main(path: str, min_box_size: int, out_file: str) -> None:
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
    """
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'))
    else:
        files = []

    scale_data = []

    for image_path in tqdm(files):
        # check if file is an image
        if imghdr.what(image_path) is not None:
            #TODO: logging if something fails
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            cross_sections = cross_section.main(image, min_box_size)

            for cs in cross_sections:
                num_rings, diff = count_rings.main(cs)

                scale_data.append([image_path, cs, num_rings, diff])

    export(scale_data, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to image or folder to be analyzed')
    parser.add_argument('--min_box_size', help='minimal bounding box height/width',
        default=200, required=False)
    parser.add_argument('--out_file',
        help='output path, determines the type also. Use .json or .feather extension',
        default=os.path.join(os.getcwd(),'scale_data.feather'), required=False)
    args = parser.parse_args()

    main(args.path, args.min_box_size, args.out_file)
