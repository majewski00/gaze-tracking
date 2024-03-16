import os
from typing import Optional

import tensorflow as tf
from keras.models import Model
from keras.layers import (Dense, Dropout, Flatten,
                          BatchNormalization, Activation,
                          Concatenate, Input, Conv2D, Add,
                          MaxPool2D)


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


class GazeTracker:
    def __init__(self):
        pass

    def build_model(self,
                    resnet_18: Optional[bool] = False,
                    connect_eyes: Optional[bool] = True,
                    pretrained_params_path: Optional[str] = None,
                    model_name: Optional[str] = "Gaze-Tracking Model"
                    ) -> any:
        """
        Create ResNet model for Gaze-Tracking. There is a experimental possibility to disconnect the weights from both eyes. By default they are connected.


        Args:
            resnet_18(bool, optional): If true, model will run on ResNet18 backbone. Otherwise the ResNet34 will be used.
            connect_eyes(bool, optional): When set to False left and right eye will have independent parameters. Default is True.
            pretrained_params_path(str, optional): Path to advised pretrained model parameters.
            model_name(str, optional): TensorFlow model name

        Returns:
            TensorFlow functional model
        """

        if not os.path.exists(pretrained_params_path):
            print("Error: Pretrained weights file does not exist.")
            exit()

        if not connect_eyes:
            if resnet_18:
                left_eye_model = ResNet18()
                right_eye_model = ResNet18()
                face_model = ResNet18()
            else:
                left_eye_model = ResNet34()
                right_eye_model = ResNet34()
                face_model = ResNet34()
        else:
            if resnet_18:
                eyes_model = ResNet18()
                face_model = ResNet18()
            else:
                eyes_model = ResNet34()
                face_model = ResNet34()

        left_input = Input((224, 224, 3), name='left_input')
        right_input = Input((224, 224, 3), name='right_input')
        face_input = Input((224, 224, 3), name='face_input')
        binary_input = Input((25, 25, 1), name='binary_input')

        if not connect_eyes:
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

        model = Model(inputs=[left_input, right_input, face_input, binary_input], outputs=[output], name=model_name)

        if pretrained_params_path is not None and not connect_eyes:
            left_eye_model.load_weights(pretrained_params_path)
            right_eye_model.load_weights(pretrained_params_path)
            face_model.load_weights(pretrained_params_path)
        elif pretrained_params_path is not None and connect_eyes:
            eyes_model.load_weights(pretrained_params_path)
            face_model.load_weights(pretrained_params_path)

        return model

    def _build_tl_model(self,
                        resnet_18: Optional[bool] = False,
                        ) -> any:
        """
        Function for building Transfer Learning model.

        Args:
            resnet_18(bool, optional): If true, model will run on ResNet18 backbone. Otherwise the ResNet34 will be used.

        Returns:
            TensorFlow Model design for pretraining
        """

        if resnet_18:
            resnet = ResNet18()
        else:
            resnet = ResNet34()

        face_input = Input((224, 224, 3), name='face_input')
        out = resnet(face_input)
        out = Dense(units=128, activation='relu')(out)
        out = BatchNormalization()(out)
        out = Dropout(0.1)(out)
        out = Dense(units=64, activation='relu')(out)
        out = BatchNormalization()(out)
        out = Dropout(0.1)(out)
        out = Dense(units=4, activation='sigmoid')(out)

        model = Model(inputs=[face_input], outputs=[out])

        return model


