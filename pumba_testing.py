import tensorflow as tf
import tensorflow.keras.backend as K

import numpy as np
from dipy.io.image import load_nifti, save_nifti
from pumba_utils import transform_img, recover_img
from scipy.ndimage import label

from skimage.morphology import binary_dilation

import sys

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

if __name__ == "__main__":
    model_path = 'pumba_model'
    model = tf.saved_model.load(model_path)

    file_name = sys.argv[1]
    output_name = sys.argv[2]

    image, affine = load_nifti(file_name)
    image = np.interp(image, (np.percentile(image, 1), np.percentile(image, 99)), (0, 1))
    image, params = transform_img(image, affine, (2, 2, 2), (128, 128, 128))
    shape = image.shape
    image = tf.convert_to_tensor(image.reshape((1, 128, 128, 128, 1)), dtype=tf.float32)

    pred = np.squeeze(model(image))

    pred = post_process(pred)

    pred = recover_img(pred, params)

    save_nifti(output_name, np.round(pred).astype(np.int32), affine)
