import torch
from torch import Tensor
from torch.jit.annotations import List, Tuple
from typing import Optional, Dict, Tuple
import numpy as np
import math

import torchvision_wj.utils.functional_tensor as functional_tensor

torch.pi = torch.acos(torch.zeros(1))[0] * 2

def _get_inverse_affine_matrix(
        center: List[float], angle: float, translate: List[float], scale: float, shear: List[float]
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    #
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x / scale for x in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    return matrix

def rotate(
        img: Tensor, angle: float, resample: int = 0, expand: bool = False,
        center: Optional[List[int]] = None, fill: Optional[int] = None
) -> Tensor:
    """Rotate the image by angle.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): image to be rotated.
        angle (float or int): rotation angle value in degrees, counter-clockwise.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (list or tuple, optional): Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands. This option is only available for ``pillow>=5.2.0``.
            This option is not supported for Tensor input. Fill value for the area outside the transform in the output
            image is always 0.

    Returns:
        PIL Image or Tensor: Rotated image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """
    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    center_f = [0.0, 0.0]
    if center is not None:
        img_size = functional_tensor._get_image_size(img)
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, img_size)]

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
    return functional_tensor.rotate(img, matrix=matrix, resample=resample, expand=expand, fill=fill)

def parallel_transform(image, box_height, box_width, angle, is_mask=True, epsilon=1e-6):
    if abs(angle)>epsilon:
        image_rot = rotate(image, angle, resample=2, expand=True)
    else:
        image_rot = image.clone()

    if is_mask:
        scale = 1/torch.cos(angle/180.*torch.pi)
        rot_h = torch.floor(box_height*scale)
        rot_w = torch.floor(box_width*scale)
        # print('**********',angle,scale,rot_h,rot_w)
        # print(torch.sum(image_rot>=0.5,dim=(0,1)))
        # print(torch.sum(image_rot>=0.5,dim=(0,2)))
        
        flag = torch.sum(image_rot>0.5,dim=(0,1))<rot_h-0.5
        rot0 = image_rot.clone()
        rot0[:,:,flag] = 0
            
        flag = torch.sum(image_rot>0.5,dim=(0,2))<rot_w-0.5
        rot1 = image_rot.clone()
        rot1[:,flag,:] = 0
        return rot0, rot1
    else:
        return image_rot


if __name__=='__main__':
    import sys
    import matplotlib.pyplot as plt
    
    h,w,r = 51, 41, 8
    img = np.zeros((h,w))
    img[r:h-r,r:w-r] = 1
    img[r:h//2+1,r:w//2+1] = 2
    # img = img[np.newaxis]
    angle = 30
    image = torch.from_numpy(img)
    index = torch.nonzero(image>0.5, as_tuple=True)
    y0,y1 = index[0].min(), index[0].max()
    x0,x1 = index[1].min(), index[1].max()
    box_h = y1-y0+1
    box_w = x1-x0+1
    y = torch.tan(angle/180.*torch.pi)*torch.arange(x0,x1+1)
    image = image.unsqueeze(0)

    degree_max = torch.acos(box_w.float()/box_h.float())/torch.pi*180

    image_rot0 = rotate(image, angle, resample=2, expand=True)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image[0].numpy())
    plt.subplot(1,2,2)
    plt.imshow(image_rot0[0].numpy()) 
    plt.show()
    sys.exit()
    
    rot0,rot1 = parallel_transform(image, box_h, box_w, angle)
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(rot0[0])
    # plt.subplot(1,2,2)
    # plt.imshow(rot1[0])
    
    image_rot1 = rotate(image, -angle, resample=2, expand=True)
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(image[0].numpy())
    # plt.subplot(1,2,2)
    # plt.imshow(image_rot1[0].numpy()) 
    
    rot0,rot1 = parallel_transform(image, box_h, box_w, -angle)
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(rot0[0])
    # plt.subplot(1,2,2)
    # plt.imshow(rot1[0])
    
    print('all mask',torch.sum(image_rot0[0]>=0.5),torch.sum(image_rot1[0]>=0.5))
    ind_a = torch.nonzero(image_rot0[0]>=0.5, as_tuple=True)
    ind_b = torch.nonzero(image_rot1[0]>=0.5, as_tuple=True)
    print(ind_a[0].min(),ind_a[0].max(),ind_a[1].min(),ind_a[1].max())
    print(ind_b[0].min(),ind_b[0].max(),ind_b[1].min(),ind_b[1].max())
    
    # image_rot = rotate(image, 90-angle, resample=2, expand=True)
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(image[0].numpy())
    # plt.subplot(1,2,2)
    # plt.imshow(image_rot[0].numpy()) 
    
    # rot0,rot1 = parallel_transform(image, box_h, box_w, angle)
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(rot0[0])
    # plt.subplot(1,2,2)
    # plt.imshow(rot1[0])
    
    # print(torch.sum(image_rot0>=0.5,dim=(0,1)))
    # print(torch.sum(image_rot0>=0.5,dim=(0,2)))
    # print(torch.sum(image_rot1>=0.5,dim=(0,1)))
    # print(torch.sum(image_rot1>=0.5,dim=(0,2)))
    # print(torch.sum(rot0>0.5,dim=(0,1)))
    # print(torch.sum(rot1>0.5,dim=(0,2)))
