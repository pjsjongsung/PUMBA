import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interpn
from scipy.ndimage import rotate
from utils_neuron import integrate_vec, resize, transform
import tensorflow as tf
from skimage.morphology import binary_closing, ball
from skimage.segmentation import find_boundaries
from tensorflow.data import AUTOTUNE


def apply_displacement_field(image, displacement_field):
    """Applies a displacement field to an image.

    Args:
        image: A numpy array representing the image.
        displacement_field: A numpy array representing the displacement field.

    Returns:
        A numpy array representing the warped image.
    """

    # Create a grid of coordinates for the input image.

    # Apply the displacement field to the grid.
    warped_grid = displacement_field

    # Warp the image using the warped grid.
    image_tensor = tf.convert_to_tensor(image.astype(np.float32))
    field_tensor = tf.convert_to_tensor(warped_grid.astype(np.float32))
    warped_image = transform(image_tensor, field_tensor, 'nearest').numpy()

    return warped_image


def ss_smooth(displacement_field, N=7):
    return integrate_vec(displacement_field, nb_steps=N)


def phi_V(x, N):
    x = np.moveaxis(x, 0, -1)
    x = x / (2 ** N)
    grid = np.mgrid[:x.shape[0], :x.shape[1], :x.shape[2]]
    i = np.arange(x.shape[0])
    j = np.arange(x.shape[1])
    k = np.arange(x.shape[2])
    points = (i, j, k)
    for _ in range(N):
        loc = np.stack([grid[d] + x[..., d] for d in range(3)], axis=-1)
        loc = np.reshape(loc, (64*64*64, 3))
        loc = np.clip(loc, 0, 63)
        temp = interpn(points, x, loc)
        x = np.reshape(temp, (64, 64, 64, 3))

    x = np.moveaxis(x, -1, 0)

    return x



def upscale_volume(input_volume, scale_factor):
    # Create the interpolation function
    grid = [np.arange(dim) for dim in input_volume.shape]
    interpolator = RegularGridInterpolator(grid, input_volume, method='linear', bounds_error=False, fill_value=0)

    # Calculate the new dimensions of the upscaled volume
    new_shape = [int(dim * scale_factor) for dim in input_volume.shape]
    output_volume = np.zeros(new_shape)

    # Generate new coordinates for the upscaled volume
    new_coordinates = []
    for i in range(len(new_shape)):
        coord_range = np.linspace(0, input_volume.shape[i] - 1, new_shape[i])
        new_coordinates.append(coord_range)

    new_coordinates = np.meshgrid(*new_coordinates, indexing='ij')
    new_coordinates = np.stack(new_coordinates, axis=-1)

    # Perform trilinear interpolation
    output_volume = interpolator(new_coordinates)

    return output_volume


def generate_random_ellipsoid(size_range, thickness_range, brain_range, deformation_range, move_range, small_ellipsoid_range, hole_number=10, hole_size_range=(7,15)):
    rng = np.random.default_rng()
    # Generate random size for the outer ellipsoid
    size_outer = rng.integers(*size_range, size=3)

    # Generate random thickness for the hollow ellipsoid
    thickness = rng.integers(*thickness_range, size=3)

    # Generate random size for the inner ellipsoid
    size_inner = size_outer - thickness - np.array([np.random.randint(*brain_range[0]), np.random.randint(*brain_range[1]), np.random.randint(*brain_range[2])])
    size_smalls = [rng.integers(10, 30, size=3) for _ in range(small_ellipsoid_range)]
    size_skull = size_outer - thickness

    x, y, z = np.ogrid[0:128, 0:128, 0:128]
    ellipsoid_skull = ((x - 64) / (size_skull[0]//2)) ** 2 + ((y - 64) / (size_skull[1]//2)) ** 2 + ((z - 64) / (size_skull[2]//2)) ** 2 <= 1

    displacement_field = tf.convert_to_tensor(rng.normal(0, rng.integers(*deformation_range), size=(10, 10, 10, 3)).astype(np.float32))

    displacement_field = resize(displacement_field, [6.4, 6.4, 6.4], [64, 64, 64])
    displacement_field = ss_smooth(displacement_field, 7)
    displacement_field = resize(displacement_field, [2.0, 2.0, 2.0], [128, 128, 128]).numpy()


    ellipsoid_outer_range = np.array([[64  - size_outer[0]//2, 64 + size_outer[0]//2],
                                      [64  - size_outer[1]//2, 64 + size_outer[1]//2],
                                      [64  - size_outer[2]//2, 64 + size_outer[2]//2]])
    ellipsoid_outer = ((x - 64) / (size_outer[0]//2)) ** 2 + ((y - 64) / (size_outer[1]//2)) ** 2 + ((z - 64) / (size_outer[2]//2)) ** 2 <= 1
    mx, my, mz = rng.integers(*move_range, size=3)
    ellipsoid_outer = np.roll(ellipsoid_outer, (mx, my, mz), axis=(0, 1, 2))
    hollow_ellipsoid = np.logical_xor(ellipsoid_outer, ellipsoid_skull)

    ellipsoid_inner_range = np.array([[64  - size_inner[0]//2, 64 + size_inner[0]//2],
                                      [64  - size_inner[1]//2, 64 + size_inner[1]//2],
                                      [64  - size_inner[2]//2, 64 + size_inner[2]//2]])
    ellipsoid_inner = ((x - 64) / (size_inner[0]//2)) ** 2 + ((y - 64) / (size_inner[1]//2)) ** 2 + ((z - 64) / (size_inner[2]//2)) ** 2 <= 1

    hole_mask = np.ones((128, 128, 128))
    for _ in range(rng.integers(hole_number//2, hole_number)):
        hole_size = rng.integers(*hole_size_range, size=3)
        rnd_trans = [rng.integers(0, size_inner[0]//2-hole_size[0]//2),
                     rng.integers(0, size_inner[1]//2-hole_size[1]//2),
                     rng.integers(0, size_inner[2]//2-hole_size[2]//2)]
        hole = ((x - 64 - rnd_trans[0]) / (hole_size[0]//2)) ** 2 + ((y - 64 - rnd_trans[0]) / (hole_size[1]//2)) ** 2 + ((z - 64 - rnd_trans[0]) / (hole_size[2]//2)) ** 2 <= 1
        hole_mask = hole_mask * (1-hole)

    small_ellipsoids = []
    small_ellipsoids_range = []
    for size_small in size_smalls:
        rnd_trans = rng.integers(20, 31, size=3) * rng.choice([-1, 1], size=3)
        temp_range = [[64 + rnd_trans[0] - size_small[0]//2, 64 + rnd_trans[0] + size_small[0]//2],
                      [64 + rnd_trans[1] - size_small[1]//2, 64 + rnd_trans[1] + size_small[1]//2],
                      [64 + rnd_trans[2] - size_small[2]//2, 64 + rnd_trans[2] + size_small[2]//2]]
        small_ellipsoids_range.append(temp_range)
        small_ellipsoids.append(((x - 64 - rnd_trans[0]) / (size_small[0]//2)) ** 2 + ((y - 64 - rnd_trans[1]) / (size_small[1]//2)) ** 2 + ((z - 64 - rnd_trans[2]) / (size_small[2]//2)) ** 2 <= 1)
    small_ellipsoids = np.stack(small_ellipsoids, axis=0)
    small_ellipsoids_range = np.stack(small_ellipsoids_range, axis=0)

    small_ellipsoids_range = np.clip(small_ellipsoids_range, 0, 128)

    # Create the hollow ellipsoid by subtracting the inner ellipsoid from the inner ellipsoid

    return hollow_ellipsoid, ellipsoid_outer_range, ellipsoid_inner, ellipsoid_inner_range, small_ellipsoids, small_ellipsoids_range, ellipsoid_skull, displacement_field, hole_mask

def divide_ellipsoid_continuous(ellipsoid, cube_range=None, divisions=3, mean_v=1.0, std_v=0.4, mean_range=(0.4, 1.0), std_range=(0, 0.4)):
    divided_ellipsoid = np.zeros_like(ellipsoid, dtype=float)
    divided_ellipsoid_label = np.zeros_like(ellipsoid, dtype=np.int32)

    idx = 1
    if cube_range is None:
        cube_shape = np.array([ellipsoid.shape[0], ellipsoid.shape[1], ellipsoid.shape[2]])
        cube_range = []
        cube_range.append([0, cube_shape[0]])
        cube_range.append([0, cube_shape[1]])
        cube_range.append([0, cube_shape[2]])
        cube_range = np.array(cube_range)
    else:
        cube_shape = np.array([[cube_range[0][1] - cube_range[0][0]],
                               [cube_range[1][1] - cube_range[1][0]],
                               [cube_range[2][1] - cube_range[2][0]]])
    if divisions == 1:
        start_idx_x, end_idx_x = cube_range[0][0], cube_range[0][1]
        start_idx_y, end_idx_y = cube_range[1][0], cube_range[1][1]
        start_idx_z, end_idx_z = cube_range[2][0], cube_range[2][1]
        mean = mean_v  # Random mean
        std_dev = std_v  # Random standard deviation
        random_values = np.random.normal(mean, std_dev, [end_idx_x-start_idx_x, end_idx_y-start_idx_y, end_idx_z-start_idx_z])
        divided_ellipsoid[start_idx_x:end_idx_x, start_idx_y:end_idx_y, start_idx_z:end_idx_z] = random_values * ellipsoid[start_idx_x:end_idx_x, start_idx_y:end_idx_y, start_idx_z:end_idx_z]
        divided_ellipsoid_label[start_idx_x:end_idx_x, start_idx_y:end_idx_y, start_idx_z:end_idx_z] = idx * ellipsoid[start_idx_x:end_idx_x, start_idx_y:end_idx_y, start_idx_z:end_idx_z]

    else:
        for i in range(divisions):
            for j in range(divisions):
                for k in range(divisions):
                    start_idx_x, end_idx_x = int(i * (cube_shape[0] // divisions) + cube_range[0][0]), int((i + 1) * (cube_shape[0] // divisions) + cube_range[0][0])
                    start_idx_y, end_idx_y = int(j * (cube_shape[1] // divisions) + cube_range[1][0]), int((j + 1) * (cube_shape[1] // divisions) + cube_range[1][0])
                    start_idx_z, end_idx_z = int(k * (cube_shape[2] // divisions) + cube_range[2][0]), int((k + 1) * (cube_shape[2] // divisions) + cube_range[2][0])

                    mean = np.random.uniform(0.4, 1)  # Random mean
                    std_dev = np.random.uniform(0, 0.4)  # Random standard deviation
                    # std_dev = 0.1
                    random_values = np.random.normal(mean, std_dev, [end_idx_x-start_idx_x, end_idx_y-start_idx_y, end_idx_z-start_idx_z])
                    divided_ellipsoid[start_idx_x:end_idx_x, start_idx_y:end_idx_y, start_idx_z:end_idx_z] = random_values * ellipsoid[start_idx_x:end_idx_x, start_idx_y:end_idx_y, start_idx_z:end_idx_z]
                    divided_ellipsoid_label[start_idx_x:end_idx_x, start_idx_y:end_idx_y, start_idx_z:end_idx_z] = idx * ellipsoid[start_idx_x:end_idx_x, start_idx_y:end_idx_y, start_idx_z:end_idx_z]
                    idx += 1

    return np.clip(divided_ellipsoid, 0, 1), divided_ellipsoid_label


def gen():
    size_range = (120, 126)
    thickness_range = (5, 50)
    brain_range = [(0, 20), (0, 20), (10, 30)]
    deformation_range = (10, 15)
    move_range = (0, 10)
    small_ellipsoid_range = 5
    hole_number = 5
    hole_size_range = (3, 10)
    rng = np.random.default_rng()

    for i_idx in range(3000):
        # Generate a random ellipsoid
        hollow_ellipsoid, hollow_ellipsoid_range, inner_ellipsoid, inner_ellipsoid_range, small_ellipsoids, small_ellipsoids_range, skull_ellipsoid, displacement_field, hole_mask = generate_random_ellipsoid(size_range, thickness_range, brain_range, deformation_range, move_range, small_ellipsoid_range, hole_number=hole_number, hole_size_range=hole_size_range)

        background_mask = np.where(np.any([hollow_ellipsoid, inner_ellipsoid], axis=0), 0, 1).astype(np.int32)

        divided_hollow_ellipsoid, divided_hollow_ellipsoid_label = divide_ellipsoid_continuous(hollow_ellipsoid, divisions=2)
        divided_background, _ = divide_ellipsoid_continuous(background_mask, divisions=1, mean_v=0.1, std_v=0.1)
        divided_inner_ellipsoid, divided_inner_ellipsoid_label = divide_ellipsoid_continuous(inner_ellipsoid, inner_ellipsoid_range, divisions=2)

        for s_idx in range(small_ellipsoid_range):
            temp, _ = divide_ellipsoid_continuous(small_ellipsoids[s_idx], small_ellipsoids_range[s_idx], divisions=2)
            divided_hollow_ellipsoid = np.where(small_ellipsoids[s_idx], temp, divided_hollow_ellipsoid)

        mask = inner_ellipsoid.astype(np.float32)

        whole = divided_inner_ellipsoid+(divided_hollow_ellipsoid*np.abs(1-inner_ellipsoid))+divided_background

        whole2 = apply_displacement_field(whole, displacement_field)
        mask2 = np.round(apply_displacement_field(mask, displacement_field)).astype(np.int32)

        temp = np.concatenate([whole2, mask2], axis=-1)
        temp = rotate(temp, rng.uniform(-15.0, 15.0), axes=(0, 1), reshape=False)
        temp = rotate(temp, rng.uniform(-15.0, 15.0), axes=(0, 2), reshape=False)
        temp = rotate(temp, rng.uniform(-15.0, 15.0), axes=(1, 2), reshape=False)

        whole2 = np.clip(temp[..., 0] * hole_mask, 0, 1)
        mask2 = temp[..., 1].astype(np.int32)
        mask2 = binary_closing(mask2, ball(2))
        boundary_mask = find_boundaries(mask2, 3).astype(np.int32)
        mask2 = np.stack([np.where(np.all([mask2==1, boundary_mask!=1], axis=0), 1, 0).astype(np.int32), boundary_mask], axis=-1)

        yield (whole2[..., np.newaxis], mask2)

def augment_using_ops(x, y):

    def random_ss(vol):
        if tf.random.uniform([]) < 0.2:
            temp = vol[..., 0] * vol[..., 1]
            vol = tf.stack([temp, vol[..., 1]], axis=-1)
        return vol

    def random_flip(vol):
        ps = (0.5, 0.5, 0.5)
        axes = []

        for ax, p in enumerate(ps):
            if rng.uniform() < p:
                axes.append(ax)
        if len(axes) != 0:
            vol = tf.reverse(vol, axes)

        return vol

    def random_tp(vol):
        max_tp = 10
        tps = [int(np.round(rng.uniform()*max_tp*2-max_tp)) for _ in range(3)]

        vol = tf.roll(vol, shift=tps, axis=[0,1,2])

        return vol

    images = x
    labels = tf.cast(y, tf.float32)
    rng = np.random.default_rng()
    vol = tf.concat([images, labels], axis=-1)

    # vol = random_ss(vol)
    vol = random_flip(vol)
    vol = random_tp(vol)
    images = vol[..., :1]
    labels = vol[..., 1:]

    return (images, tf.cast(labels, tf.int32))

dataset = tf.data.Dataset.from_generator(gen,
                                         output_signature=(
                                            tf.TensorSpec(shape=(128, 128, 128, 1), dtype=tf.float32),
                                            tf.TensorSpec(shape=(128, 128, 128, 2), dtype=tf.int32)
                                         ))

dataset = (dataset
         .map(augment_using_ops, num_parallel_calls=AUTOTUNE)
         .prefetch(tf.data.AUTOTUNE))

dataset.save('pumba_dataset')