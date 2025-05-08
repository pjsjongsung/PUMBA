from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K

import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.align.reslice import reslice
from dipy.nn.utils import pad_crop
from scipy.ndimage import affine_transform, label

from skimage.morphology import binary_dilation

def post_process(pred):
    chunks, n_chunk = label(np.round(pred[..., 0]))
    if n_chunk > 1:
        u, c = np.unique(chunks[chunks!=0], return_counts=True)
        t = u[np.argmax(c)]
        new_mask = np.where(chunks==t, 1, 0)
    else:
        new_mask = np.round(pred[..., 0])
    dilated_mask = binary_dilation(new_mask, [(np.ones((5, 1, 1)), 1), (np.ones((1, 5, 1)), 1), (np.ones((1, 1, 5)), 1)])
    new_mask = new_mask + dilated_mask * np.round(pred[..., 1])
    return new_mask

def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    smooth = 1.
    dice_sum = 0.
    for i in tf.range(2):
        y_pred_seg = y_pred[..., i]
        y_true_seg = y_true[..., i]
        y_true_f = K.flatten(y_true_seg)
        y_pred_f = K.flatten(y_pred_seg)
        intersection = K.sum(y_true_f * y_pred_f)
        dice_sum += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice_sum


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def transform_img(
    image,
    affine,
    *,
    voxsize=(1, 1, 1),
    target_voxsize=None,
    considered_points="corners",
    init_shape=None,
    ratio=None,
    set_size=None,
    need_isotropic=False,
):
    """
    Function to transform images for Deep Learning models

    Parameters
    ----------
    image : np.ndarray
        Image to transform.
    affine : np.ndarray
        Affine matrix provided by the image file.
    voxsize : tuple (3,)
        Voxel size provided by the image file.
    target_voxsize : tuple (3,)
        The voxel size we want to start from.
        Ignored if need_isotropic is True.
    considered_points : str
        Considered points when calculating the transformed shape.
        \"corners\" will consider only corners of the image shape.
        \"all\" will consider all voxels. Might be needed when shearing is applied.
    init_shape : tuple (3,)
        What we want the initial shape to be before last resizing step.
        Ignored if need_isotropic is True.
    ratio : float
        The ratio of change in the last resizing step.
        Ignored if need_isotropic is True.
    set_size : tuple (3,)
        The final size of the image array.
    need_isotropic : bool
        Whether the output needs to be isotropic in the end.



    Returns
    -------
    new_image : np.ndarray
        Transformed image to be used in the Deep Learning model.
    params : tuple
        Parameters that are used when recovering the original image space.
    """
    ori_shape = image.shape
    if need_isotropic:
        target_voxsize = tuple(np.max(voxsize) * np.ones(3))
    if target_voxsize is not None and np.any(target_voxsize != np.ones(3)):
        image2, affine2 = reslice(image, affine, voxsize, target_voxsize)
    else:
        image2 = image.copy()
        affine2 = affine.copy()

    shape = image2.shape

    if considered_points == "corners":
        corners = np.array(
            [
                [0, 0, 0, 1],
                [shape[0] - 1, 0, 0, 1],
                [0, shape[1] - 1, 0, 1],
                [0, 0, shape[2] - 1, 1],
                [shape[0] - 1, shape[1] - 1, shape[2] - 1, 1],
                [shape[0] - 1, 0, shape[2] - 1, 1],
                [0, shape[1] - 1, shape[2] - 1, 1],
                [shape[0] - 1, shape[1] - 1, 0, 1],
            ],
            dtype=np.float64,
        )
    elif considered_points == "all":
        temp1 = np.arange(shape[0])
        temp2 = np.arange(shape[1])
        temp3 = np.arange(shape[2])
        grid1, grid2, grid3 = np.meshgrid(temp1, temp2, temp3)
        corners = np.vstack([grid1.ravel(), grid2.ravel(), grid3.ravel()]).T
        corners = np.hstack([corners, np.full((corners.shape[0], 1), 1)])
        corners = corners.astype(np.float64)
    else:
        ValueError('considered points should be "corners" or "all"')

    transformed_corners = (affine2 @ corners.T).T
    min_bounds = transformed_corners.min(axis=0)[:3]
    max_bounds = transformed_corners.max(axis=0)[:3]

    # Calculate the required offset to ensure
    # all necessary coordinates are positive
    offset = np.ceil(-min_bounds)
    new_shape = (np.ceil(max_bounds) + offset).astype(int)
    offset_array = np.array(
        [[1, 0, 0, offset[0]], [0, 1, 0, offset[1]], [0, 0, 1, offset[2]], [0, 0, 0, 1]]
    )

    new_affine = affine2.copy()
    new_affine = np.matmul(offset_array, new_affine)

    inv_affine = np.linalg.inv(new_affine)
    new_image = np.zeros(tuple(new_shape))
    affine_transform(
        image2, inv_affine, output_shape=tuple(new_shape), output=new_image
    )

    mid_image = new_image.copy()

    crop_vs = None
    pad_vs = None
    if not need_isotropic:
        if init_shape:
            new_image, pad_vs, crop_vs = pad_crop(new_image, init_shape)

        if (ratio is not None and ratio != 1) and set_size is None:
            new_image, _ = reslice(
                new_image, np.eye(4), (1, 1, 1), (ratio, ratio, ratio)
            )
        elif set_size:
            new_image, _ = reslice(
                new_image,
                np.eye(4),
                (1, 1, 1),
                (
                    new_image.shape[0] / set_size[0],
                    new_image.shape[1] / set_size[1],
                    new_image.shape[2] / set_size[2],
                ),
            )

    else:
        ratio = np.max(np.array(mid_image.shape) / np.array(set_size))
        new_size = np.ceil(np.array(set_size) * ratio)
        new_image, pad_vs, crop_vs = pad_crop(mid_image, tuple(new_size))
        new_image, _ = reslice(new_image, np.eye(4), (1, 1, 1), (ratio, ratio, ratio))

    params = (
        inv_affine,
        image2.shape,
        offset_array,
        crop_vs,
        pad_vs,
        ratio,
        voxsize,
        target_voxsize,
        need_isotropic,
        set_size,
        ori_shape,
    )

    return new_image, params


def recover_img(image, params):
    """
    Function to recover image from transform_img

    Parameters
    ----------
    image : np.ndarray
        Image to recover.
    params : tuple
        Parameters for recover_img function.
        Returned from transform_img.

    Returns
    -------
    new_image : np.ndarray
        Recovered image
    affine : np.ndarray
        Recovered affine.
        This should be same as the original affine.
    """
    (
        inv_affine,
        mid_shape,
        offset_array,
        crop_vs,
        pad_vs,
        ratio,
        voxsize,
        target_voxsize,
        need_isotropic,
        set_size,
        ori_shape,
    ) = params
    new_affine = np.linalg.inv(inv_affine)
    if need_isotropic:
        new_image, _ = reslice(image, np.eye(4), (ratio, ratio, ratio), (1, 1, 1))
    else:
        if (ratio is not None and ratio != 1) and set_size is None:
            new_image, _ = reslice(image, np.eye(4), (ratio, ratio, ratio), (1, 1, 1))
        elif set_size:
            new_image, _ = reslice(
                image,
                np.eye(4),
                (
                    new_image.shape[0] / set_size[0],
                    new_image.shape[1] / set_size[1],
                    new_image.shape[2] / set_size[2],
                ),
                (1, 1, 1),
            )
        else:
            new_image = image

    if crop_vs is not None and pad_vs is not None:
        crop_vs = crop_vs.astype(int)
        pad_vs = pad_vs.astype(int)
        new_image = np.pad(
            new_image,
            (
                (crop_vs[0, 0], crop_vs[0, 1]),
                (crop_vs[1, 0], crop_vs[1, 1]),
                (crop_vs[2, 0], crop_vs[2, 1]),
            ),
        )
        new_image = new_image[
            pad_vs[0, 0] : new_image.shape[0] - pad_vs[0, 1],
            pad_vs[1, 0] : new_image.shape[1] - pad_vs[1, 1],
            pad_vs[2, 0] : new_image.shape[2] - pad_vs[2, 1],
        ]

    new_image = affine_transform(new_image, new_affine, output_shape=mid_shape)
    affine = np.matmul(np.linalg.inv(offset_array), new_affine)

    if target_voxsize is not None and np.any(target_voxsize != np.ones(3)):
        new_image, affine = reslice(new_image, affine, target_voxsize, voxsize)
        if new_image.shape != ori_shape:
            new_image = pad_crop(new_image, ori_shape)

    return new_image, affine

if __name__ == "__main__":
    model_path = 'pumba_model'
    model = load_model(model_path, custom_objects={'dice_loss':dice_loss, 'dice_coefficient':dice_coefficient})

    file_name = ''
    output_name = ''
    image, affine, voxsize = load_nifti(file_name, return_voxsize=True)
    image = np.interp(image, (np.percentile(image, 1), np.percentile(image, 99)), (0, 1))
    image, params = transform_img(image, affine, (128, 128, 128))
    shape = image.shape

    pred = np.squeeze(model(np.reshape(image, (1, 128, 128, 128, 1))))

    pred = post_process(pred)

    pred = recover_img(pred, params)
    save_nifti(output_name, np.round(pred), affine)