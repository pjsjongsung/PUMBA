from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K

import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.align.reslice import reslice
from dipy.nn.utils import pad_crop, inv_pad_crop, get_bounds
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
    target_voxsize=None,
    final_size=None,
    order=3,
    considered_points="corners",
):
    """
    Function to transform images for Deep Learning models

    Parameters
    ----------
    image : np.ndarray
        Image to transform.
    affine : np.ndarray
        Affine matrix provided by the image file.
    target_voxsize : tuple (3,), optional
        The voxel size we want to start from.
        If none is provided, we calculate the closest isotropic voxel size.
    final_size : tuple (3,), optional
        The final size of the image array.
    order : int, optional
        The order of the spline interpolation.
        The order has to be in the range 0-5.
        If transforming an int image, order 0 is recommended.
    considered_points : str, optional
        Considered points when calculating the transformed shape.
        \"corners\" will consider only corners of the image shape.
        \"all\" will consider all voxels. Might be needed when shearing is applied.

    Returns
    -------
    new_image : np.ndarray
        Transformed image to be used in the Deep Learning model.
    params : dict
        Parameters that are used when recovering the original image space.
    """
    ori_shape = image.shape

    R = affine[:3, :3]
    voxsize = np.sqrt(np.sum(R * R, axis=0))
    if target_voxsize is None:
        target_voxsize = np.array([np.min(voxsize)] * 3)
    else:
        target_voxsize = np.array(target_voxsize)

    if not np.allclose(voxsize, target_voxsize):
        image, resliced_affine = reslice(
            image, affine, tuple(voxsize), tuple(target_voxsize)
        )
    else:
        resliced_affine = None

    init_shape = image.shape

    min_bounds, max_bounds = get_bounds(
        init_shape, affine, considered_points=considered_points
    )

    offset = np.ceil(-min_bounds)
    new_shape = (np.ceil(max_bounds) + offset).astype(int)
    offset_array = np.array(
        [[1, 0, 0, offset[0]], [0, 1, 0, offset[1]], [0, 0, 1, offset[2]], [0, 0, 0, 1]]
    )

    new_affine = affine.copy()
    new_affine = np.matmul(offset_array, new_affine)

    inv_affine = np.linalg.inv(new_affine)

    new_image = np.zeros(tuple(new_shape))
    affine_transform(
        image,
        inv_affine,
        output_shape=tuple(new_shape),
        output=new_image,
        order=order,
    )

    crop_vs = None
    pad_vs = None
    if final_size is None:
        final_size = new_image.shape
    final_size = np.array(final_size)
    new_image, pad_vs, crop_vs = pad_crop(new_image, final_size)

    params = {
        "inv_affine": inv_affine,
        "offset": offset_array,
        "crop_value": crop_vs,
        "pad_value": pad_vs,
        "voxsize": voxsize,
        "target_voxsize": target_voxsize,
        "ori_shape": ori_shape,
        "init_shape": init_shape,
        "affine": resliced_affine,
    }

    return new_image, params


def recover_img(image, params, *, order=3):
    """
    Function to recover image from transform_img

    Parameters
    ----------
    image : np.ndarray
        Image to recover.
    params : tuple
        Parameters for recover_img function.
        Returned from transform_img.
    order : int, optional
        The order of the spline interpolation.
        The order has to be in the range 0-5.
        If recovering an int image, order 0 is recommended.

    Returns
    -------
    new_image : np.ndarray
        Recovered image
    affine : np.ndarray
        Recovered affine.
        This should be same as the original affine.
    """
    expected_keys = {
        "inv_affine",
        "crop_value",
        "pad_value",
        "voxsize",
        "target_voxsize",
        "ori_shape",
        "init_shape",
        "affine",
    }
    missing = expected_keys - set(params.keys())
    if missing:
        raise ValueError(f"params is missing keys: {missing}")

    inv_affine = params["inv_affine"]
    crop_vs = params["crop_value"]
    pad_vs = params["pad_value"]
    voxsize = params["voxsize"]
    target_voxsize = params["target_voxsize"]
    ori_shape = params["ori_shape"]
    init_shape = params["init_shape"]
    affine = params["affine"]

    if crop_vs is not None and pad_vs is not None:
        new_image = inv_pad_crop(image, crop_vs, pad_vs)

    new_affine = np.linalg.inv(inv_affine)
    new_image = affine_transform(
        new_image, new_affine, output_shape=init_shape, order=order
    )

    if not np.allclose(voxsize, target_voxsize):
        new_image, affine = reslice(
            new_image,
            affine,
            tuple(target_voxsize),
            tuple(voxsize),
            new_shape=ori_shape,
            order=order,
        )

    return new_image, affine

if __name__ == "__main__":
    model_path = 'pumba_model'
    model = load_model(model_path, custom_objects={'dice_loss':dice_loss, 'dice_coefficient':dice_coefficient})

    file_name = ''
    output_name = ''
    image, affine = load_nifti(file_name, return_voxsize=True)
    image = np.interp(image, (np.percentile(image, 1), np.percentile(image, 99)), (0, 1))
    image, params = transform_img(image, affine, (2, 2, 2), (128, 128, 128))
    shape = image.shape

    pred = np.squeeze(model(np.reshape(image, (1, 128, 128, 128, 1))))

    pred = post_process(pred)

    pred, _ = recover_img(pred, params)

    save_nifti(output_name, np.round(pred).astype(np.int32), affine)
