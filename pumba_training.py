import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Concatenate
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam


initializer = tf.keras.initializers.HeNormal
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

def conv_block(input_tensor, num_filters):
    encoder = Conv3D(num_filters, 3, padding='same', activation='relu', kernel_initializer=initializer())(input_tensor)
    encoder = Conv3D(num_filters, 3, padding='same', activation='relu', kernel_initializer=initializer())(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = MaxPooling3D(2, strides=2)(encoder)
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = UpSampling3D(2)(input_tensor)
    decoder = Concatenate(axis=-1)([concat_tensor, decoder])
    decoder = Conv3D(num_filters, 3, padding='same', activation='relu', kernel_initializer=initializer())(decoder)
    decoder = Conv3D(num_filters, 3, padding='same', activation='relu', kernel_initializer=initializer())(decoder)
    return decoder

def unet_3d(input_shape, num_classes, model_scale=8):
    inputs = Input(input_shape)
    
    encoder0_pool, encoder0 = encoder_block(inputs, model_scale)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, model_scale*2)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, model_scale*4)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, model_scale*8)
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, model_scale*16)
    
    center = conv_block(encoder4_pool, model_scale*32)
    
    decoder4 = decoder_block(center, encoder4, model_scale*16)
    decoder3 = decoder_block(decoder4, encoder3, model_scale*8)
    decoder2 = decoder_block(decoder3, encoder2, model_scale*4)
    decoder1 = decoder_block(decoder2, encoder1, model_scale*2)
    decoder0 = decoder_block(decoder1, encoder0, model_scale)
    
    outputs = Conv3D(num_classes, 1, activation='softmax')(decoder0)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

batch_size = 1

train_generator = tf.data.Dataset.load('pumba_dataset')
train_generator = train_generator.batch(batch_size)
# Example usage:
input_shape = (128, 128, 128, 1)  # assuming input images are grayscale
num_classes = 3  # assuming binary segmentation
num_epochs = 50

previous_epoch = 0

# Define custom hyperparameters
learning_rate = 0.0001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-7  # small value to prevent division by zero

# Create custom Adam optimizer
custom_adam = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

if os.path.exists('epochs.txt'):
    with open('epochs.txt', 'r') as f:
        previous_epoch = int(f.read())
    model_path = 'pumba_model'
    model = load_model(model_path, custom_objects={'dice_loss':dice_loss, 'dice_coefficient':dice_coefficient})
else:
    model = unet_3d(input_shape, num_classes)
    model.compile(optimizer=custom_adam, loss=dice_loss, metrics=[dice_coefficient])

print("Previous epoch number:", previous_epoch)

for epoch in range(previous_epoch+1, num_epochs+previous_epoch):
    history = model.fit(train_generator,
                        steps_per_epoch=3000 // batch_size)
    print(history.history['loss'])

    model.save('pumba_model')
    
    # Save current epoch number
    with open('epochs.txt', 'w') as f:
        f.write(str(epoch))