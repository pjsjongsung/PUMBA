import tensorflow as tf
import tensorflow.keras.backend as K

import numpy as np
from dipy.io.image import load_nifti, save_nifti
from pumba_utils import transform_img, recover_img, post_process, compute_isotropic_spacing
from scipy.ndimage import label

from skimage.morphology import binary_dilation

import sys

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

if __name__ == "__main__":
    model_path = 'pumba_model'
    model = tf.saved_model.load(model_path)

    file_name = sys.argv[1]
    output_name = sys.argv[2]
    transform_method = sys.argv[3] if len(sys.argv) > 3 else "transform_img"

    image, affine, voxsize = load_nifti(file_name, return_voxsize=True)
    shape = image.shape
    if transform_method == "transform_img":
        target_voxsize = compute_isotropic_spacing(image.shape, voxsize)
        image, params = transform_img(image, affine, target_voxsize=tuple([target_voxsize] * 3), final_size=(128, 128, 128))
    elif transform_method == "resize":
        from skimage.transform import resize
        image = resize(image, (128, 128, 128), anti_aliasing=True)
    else:
        raise ValueError(f"Unknown transform method: {transform_method}")
    image = np.interp(image, (np.percentile(image, 1), np.percentile(image, 99)), (0, 1))
    image = tf.convert_to_tensor(image.reshape((1, 128, 128, 128, 1)), dtype=tf.float32)

    pred = np.squeeze(model(image))

    if skip_postprocess := ("--skip-postprocess" in sys.argv):
        pred = np.argmax(pred, axis=-1)
    else:
        pred = post_process(pred)

    if transform_method == "transform_img":
        pred = recover_img(pred, params, order=0)
    elif transform_method == "resize":
        pred = resize(pred, shape, order=0, preserve_range=True)

    save_nifti(output_name, np.round(pred).astype(np.uint8), affine)
