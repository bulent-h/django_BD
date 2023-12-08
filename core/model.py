import tensorflow as tf
import keras
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add

class Conv2DBNLayer(layers.Layer):
    def __init__(self, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu',**kwargs):
        super().__init__(**kwargs)
        self.filters=filters
        self.num_row=num_row
        self.num_col=num_col
        self.padding=padding
        self.strides=strides
        self.activation=activation
        self.conv2d = layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)
        self.batch_norm = layers.BatchNormalization(axis=3, scale=False)
        self.activation = activation

    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.batch_norm(x)
        if self.activation is None:
            return x
        x = layers.Activation(self.activation, name=self.name)(x)
        return x
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "num_row":self.num_row,
            "num_col": self.num_col,
            "padding": self.padding,
            "strides": self.strides,
            "activation": self.activation,
            "name": self.name,
        })
        return config
    
class MultiResBlock(layers.Layer):
    def __init__(self, U, alpha=1.67,**kwargs):
        super().__init__(**kwargs)
        self.W = alpha * U
        self.alpha = alpha
        self.U = U
        self.C = int(self.W * 0.167) + int(self.W * 0.333) + int(self.W * 0.5)
        self.shortcut = Conv2DBNLayer(self.C, 1, 1, activation=None, padding='same')
        self.conv3x3 = Conv2DBNLayer(int(self.W * 0.167), 3, 3, activation='relu', padding='same')
        self.conv5x5 = Conv2DBNLayer(int(self.W * 0.333), 3, 3, activation='relu', padding='same')
        self.conv7x7 = Conv2DBNLayer(int(self.W * 0.5), 3, 3, activation='relu', padding='same')
        self.concat = layers.Concatenate(axis=3)
        self.batch_norm_1 = layers.BatchNormalization(axis=3)
        self.add_layer = layers.Add()
        self.activation = layers.Activation('relu')
        self.batch_norm_2 = layers.BatchNormalization(axis=3)

    def call(self, inputs):
        shortcut = self.shortcut(inputs)
        conv3x3 = self.conv3x3(inputs)
        conv5x5 = self.conv5x5(conv3x3)
        conv7x7 = self.conv7x7(conv5x5)
        concatenated = self.concat([conv3x3, conv5x5, conv7x7])
        batch_normed = self.batch_norm_1(concatenated)
        added = self.add_layer([shortcut, batch_normed])
        activated = self.activation(added)
        out = self.batch_norm_2(activated)
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update ({
            "alpha":self.alpha,
            "U": self.U,
            "name": self.name,
        })
        return config
    
class AttentionBlock(layers.Layer):
    def __init__(self, f_out,alpha=1.67,**kwargs ):

        super(AttentionBlock, self).__init__()
        self.W = alpha * f_out
        self.C = int(self.W * 0.167) + int(self.W * 0.333) + int(self.W * 0.5)
        self.c1 = layers.Conv2D(f_out, kernel_size=1, strides=1, padding='valid', use_bias=True)
        self.c_spa1 = layers.Conv2D(f_out, kernel_size=1, strides=1, padding='valid', use_bias=True)
        self.relu = layers.ReLU()
        self.c2 = layers.Conv2D(self.C, kernel_size=1, strides=1, padding='valid', use_bias=True)
        self.sigmoid = layers.Activation('sigmoid')
        self.add = layers.Add()
    def call(self, g, spa):
        
        g1 = self.c1(g)
        x1 = self.c_spa1(spa)
        f = self.relu(self.add([g1,x1]))
        f = self.c2(f)
        f = self.sigmoid(f)
        f = layers.multiply([spa, f])

        return f