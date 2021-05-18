#!/usr/bin/env python3.6
import sys
import shutil
import re
import random
import argparse
import warnings
from pathlib import Path
from pprint import pprint
from functools import partial
from typing import Any, Callable, List, Tuple

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
from numpy import unique as uniq
from skimage.io import imread, imsave
from skimage.transform import resize

from utils import mmap_, uc_, map_


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    return res.astype(np.uint8)


def get_p_id(path: Path, regex: str = "(Case\d+)(_segmentation)?") -> str:
    matched = re.match(regex, path.stem)

    if matched:
        return matched.group(1)
    raise ValueError(regex, path)


def save_slices(img_p: Path, 
                dest_dir: Path, shape: Tuple[int], 
                img_dir: str = "img", gt_dir: str = "gt") -> Tuple[int, int, int]:
    p_id: str = get_p_id(img_p)
    assert "Case" in p_id

    # Load the data
    img = imread(str(img_p), plugin='simpleitk')
    # print(img.shape, img.dtype)
    # print(img.min(), img.max(), len(np.unique(img)))
    # print(np.unique(gt))

    assert img.dtype in [np.int16]

    img_nib = sitk.ReadImage(str(img_p))
    dx, dy, dz = img_nib.GetSpacing()
    # print(dx, dy, dz)
    assert np.abs(dx - dy) <= 0.0000041, (dx, dy, dx - dy)
    assert 0.27 <= dx <= 0.75, dx
    assert 2.19994 <= dz <= 4.00001, dz

    x, y, z = img.shape
    assert (y, z) in [(320, 320), (512, 512), (256, 256), (384, 384)], (y, z)
    assert 15 <= x <= 54, x

    # Normalize and check data content
    norm_img = norm_arr(img)  # We need to normalize the whole 3d img, not 2d slices
    assert 0 == norm_img.min() and norm_img.max() == 255, (norm_img.min(), norm_img.max())
    assert norm_img.dtype == np.uint8

    save_dir_img: Path = Path(dest_dir, img_dir)
    for j in range(len(img)):
        img_s = norm_img[j, :, :]

        # Resize and check the data are still what we expect
        resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)
        r_img: np.ndarray = resize_(img_s, shape).astype(np.uint8)
        # print(time() - tic)
        assert 0 <= r_img.min() and r_img.max() <= 255  # The range might be smaller

        a_img = r_img

        k = 0
        for save_dir, data in zip([save_dir_img], [a_img]):
            filename = f"{p_id}_{k}_{j:02d}.png"
            save_dir.mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(Path(save_dir, filename)), data)
    


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the cleaning up is done before calling the script
    assert src_path.exists()
    # assert not dest_path.exists()

    # Get all the file names, avoid the temporal ones
    nii_paths: List[Path] = [p for p in src_path.rglob('*.mhd')]
    assert len(nii_paths) % 2 == 0, "Uneven number of .nii, one+ pair is broken"

    # We sort now, but also id matching is checked while iterating later on
    img_nii_paths: List[Path] = sorted(p for p in nii_paths if "_segmentation" not in str(p))
    # gt_nii_paths: List[Path] = sorted(p for p in nii_paths if "_segmentation" in str(p))
    # assert len(img_nii_paths) == len(gt_nii_paths)
    paths: List[Tuple[Path, Path]] = list(img_nii_paths)

    print(f"Found {len(img_nii_paths)} pairs in total")
    pprint(paths[:5])

    assert len(paths) == 30

    mode = "test"
    img_paths = paths

    dest_dir = Path(dest_path, mode)
    print(f"Slicing {len(img_paths)} pairs to {dest_dir}")

    pfun = partial(save_slices, dest_dir=dest_dir, shape=args.shape)
    # mmap_(uc_(pfun), [img_paths])
    for paths in img_paths:
        uc_(pfun)([paths])


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, default='../data/prostate/Test') #done
    parser.add_argument('--dest_dir', type=str, default='../data/prostate/PROSTATE') #done

    parser.add_argument('--img_dir', type=str, default="IMG")
    parser.add_argument('--gt_dir', type=str, default="GT")
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)

    main(args)
