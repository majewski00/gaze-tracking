import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as plt_Rectangle
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Concatenate, Input, Conv2D, Add, MaxPool2D
from keras.models import Model
from tensorflow_addons.optimizers import CyclicalLearningRate
import sys
import os


def euclidean_distance(y_true, y_pred):
    """Custom Euclidean Distance Accuracy for tf.keras Model"""
    y_true = tf.multiply(y_true, tf.constant([[1920, 1080]], dtype='float32'))
    y_pred = tf.multiply(y_pred, tf.constant([[1920, 1080]], dtype='float32'))
    norm = tf.norm(y_true - y_pred, ord='euclidean', axis=-1)
    return tf.reduce_mean(norm, axis=-1)


def learning_results(hist):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    if list(hist.keys())[3] == 'val_euclidean_distance':
        name = 'Euclidean Distance'
        ax[1].set_title('Gaze Tracking Distance Error')
        ax[0].set_title('Gaze Tracking Loss Function')
    else:
        name = 'Accuracy'
        ax[1].set_title('Training Accuracy')
        ax[0].set_title('Training Loss Function')

    ax[1].plot(range(len(hist['loss'])), list(hist.values())[1], label=name, color='blue')
    ax[1].plot(range(len(hist['loss'])), list(hist.values())[3], label='Validation ' + name,
               color='red', linestyle=':')

    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend(loc='best')

    ax[0].plot(range(len(hist['loss'])), hist['loss'], label='Loss Function', color=(0.8, 0.42, 0.97))
    ax[0].plot(range(len(hist['loss'])), hist['val_loss'], label='Validation Loss Function', color=(0.41, 0.01, 0.62))
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='best')

    plt.show()


class ResnetBlock(Model):
    def __init__(self, filters: int, down_sample: bool = False):
        super().__init__()
        self.__filters = filters
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        self.conv_1 = Conv2D(self.__filters, strides=self.__strides[0], kernel_size=(3, 3),
                             padding='same', kernel_initializer='he_normal')
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__filters, strides=self.__strides[1], kernel_size=(3, 3),
                             padding='same', kernel_initializer='he_normal')
        self.bn_2 = BatchNormalization()
        self.merge = Add()
        self.drop = Dropout(0.1)
        self.relu = Activation(activation='relu')

        if self.__down_sample:
            self.res_conv = Conv2D(self.__filters, strides=2, kernel_size=(1, 1), padding='same',
                                   kernel_initializer='he_normal')
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        skip = inputs
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            skip = self.res_conv(skip)
            skip = self.res_bn(skip)

        x = self.merge([x, skip])
        out = self.relu(x)
        return out


class ResNet18(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                             kernel_initializer='he_normal')
        self.bn = BatchNormalization()
        self.max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)

        self.relu = Activation(activation='relu')
        self.drop = Dropout(0.1)
        self.flat = Flatten()

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.bn(out)
        out = self.relu(out)
        out = self.max_pool(out)

        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1,
                          self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
            out = self.drop(out)

        out = self.flat(out)

        return out


class ResNet34(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                             kernel_initializer='he_normal')
        self.bn = BatchNormalization()
        self.max_pool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_1_3 = ResnetBlock(64)

        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_2_3 = ResnetBlock(128)
        self.res_2_4 = ResnetBlock(128)

        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_3_3 = ResnetBlock(256)
        self.res_3_4 = ResnetBlock(256)
        self.res_3_5 = ResnetBlock(256)
        self.res_3_6 = ResnetBlock(256)

        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.res_4_3 = ResnetBlock(512)

        self.relu = Activation(activation='relu')
        self.drop = Dropout(0.1)
        self.flat = Flatten()

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.bn(out)
        out = self.relu(out)
        out = self.max_pool(out)

        for res_block in [self.res_1_1, self.res_1_2, self.res_1_3,
                          self.res_2_1, self.res_2_2, self.res_2_3, self.res_2_4,
                          self.res_3_1, self.res_3_2, self.res_3_3, self.res_3_4, self.res_3_5, self.res_3_6,
                          self.res_4_1, self.res_4_2, self.res_4_3]:
            out = res_block(out)
            out = self.drop(out)

        out = self.flat(out)
        return out


def create_model(type: int = 18, eyes_connected: bool = True, pretrain_res_net: str = None, name: str = None):
    """Create model based on ResNet18 or ResNet34 backbone (type=18/34). Set the path to the corresponding ResNet weights
        with 'pretrain_res_net'. Also there is a experimental possibility to disconnect the weights from left and right
        eye images. By default they are connected"""

    if not eyes_connected:
        if type == 18:
            left_eye_model = ResNet18()
            right_eye_model = ResNet18()
            face_model = ResNet18()
        elif type == 34:
            left_eye_model = ResNet34()
            right_eye_model = ResNet34()
            face_model = ResNet34()
    else:
        if type == 18:
            eyes_model = ResNet18()
            face_model = ResNet18()
        elif type == 34:
            eyes_model = ResNet34()
            face_model = ResNet34()

    left_input = Input((224, 224, 3), name='left_input')
    right_input = Input((224, 224, 3), name='right_input')
    face_input = Input((224, 224, 3), name='face_input')
    binary_input = Input((25, 25, 1), name='binary_input')

    if not eyes_connected:
        left = left_eye_model(left_input)
        right = right_eye_model(right_input)
    else:
        left = eyes_model(left_input)
        right = eyes_model(right_input)

    eyes = Concatenate()([left, right])
    eyes = Dropout(0.1)(eyes)
    eyes = Dense(units=128, activation='relu')(eyes)
    eyes = BatchNormalization()(eyes)
    eyes = Dropout(0.1)(eyes)

    face = face_model(face_input)
    face = Dense(units=128, activation='relu')(face)
    face = BatchNormalization()(face)
    face = Dropout(0.1)(face)
    face = Dense(units=64, activation='relu')(face)
    face = BatchNormalization()(face)
    face = Dropout(0.1)(face)

    binary = Flatten()(binary_input)
    binary = Dense(units=256, activation='relu')(binary)
    binary = BatchNormalization()(binary)
    binary = Dropout(0.1)(binary)
    binary = Dense(units=128, activation='relu')(binary)
    binary = BatchNormalization()(binary)
    binary = Dropout(0.1)(binary)

    output = Concatenate()([eyes, face, binary])
    output = Dense(units=128, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.1)(output)
    output = Dense(units=2, activation='sigmoid', name='output')(output)

    if name is not None:
        model = Model(inputs=[left_input, right_input, face_input, binary_input], outputs=[output], name=name)
    else:
        model = Model(inputs=[left_input, right_input, face_input, binary_input], outputs=[output])

    if pretrain_res_net is not None and not eyes_connected:
        left_eye_model.load_weights(pretrain_res_net)
        right_eye_model.load_weights(pretrain_res_net)
        face_model.load_weights(pretrain_res_net)
    elif pretrain_res_net is not None and eyes_connected:
        eyes_model.load_weights(pretrain_res_net)
        face_model.load_weights(pretrain_res_net)

    return model




## Load Dataset ##
image = image_extraction.Image()
dataset, dataset_size = image.create_dataset(shuffle=True, mirror=True)



## Initialize model and necessary parameters ##

model = create_model(18, eyes_connected=True, pretrain_res_net='...', name="Gaze_Tracking_Model")

BATCH_SIZE = 64
VALIDATION_SPLIT = 0.15
EPOCHS = 20
INIT_LR = 1e-4
MAX_LR = 2e-3

train_dataset = dataset.take(int((1 - VALIDATION_SPLIT) * dataset_size)).batch(BATCH_SIZE, drop_remainder=True).repeat()
val_dataset = dataset.skip(int((1 - VALIDATION_SPLIT) * dataset_size)).batch(BATCH_SIZE, drop_remainder=True)


steps_per_epoch = int((1 - VALIDATION_SPLIT) * dataset_size) // BATCH_SIZE
val_steps = int(VALIDATION_SPLIT * dataset_size) // BATCH_SIZE
clr = CyclicalLearningRate(initial_learning_rate=INIT_LR, maximal_learning_rate=MAX_LR, scale_fn=lambda x: 1/(2.**(x-1)),
                           step_size=2 * steps_per_epoch)

model.compile(optimizer=keras.optimizers.Adam(clr),
              loss=keras.losses.MeanSquaredError(),
              metrics=[euclidean_distance])

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, start_from_epoch=5, restore_best_weights=True)



## Train Model ##
hist = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[callback], batch_size=BATCH_SIZE,
                 steps_per_epoch=steps_per_epoch, validation_steps=val_steps, validation_batch_size=BATCH_SIZE)

learning_results(hist.history)
model.save_weights('...')



