from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import json
import numpy as np
from custom_layers import Wrap
from etmiss_utils import del_etmiss_loss

config = json.load(open("config.json","r"))
eta_bins = config["imageparams"]["eta_bins"]
phi_bins = config["imageparams"]["phi_bins"]
batch_size = config["trainingconfig"]["batch_size"]

def BB_model(tracks=False):
    '''
    Bernard's original architecture.
    '''
    n_channels = 4 if not tracks else 6
    layer_opts = {'kernel_initializer': 'RandomUniform',
                  'padding': 'valid',
                  'data_format': 'channels_last',
                  'strides': (1,1)}

    ip = layers.Input((eta_bins, phi_bins, n_channels))
    x = Wrap(half_kernel_size=4)(ip)
    x = layers.Conv2D(15,9,**layer_opts,activation='relu')(x)
    x = Wrap(half_kernel_size=3)(x)
    x = layers.Conv2D(20,7,**layer_opts,activation='relu')(x)
    x = Wrap(half_kernel_size=2)(x)
    x = layers.Conv2D(25,5,**layer_opts,activation='relu')(x)
    x = layers.Conv2D(1,1,**layer_opts,activation='relu')(x)

    mdl = Model(inputs=ip,outputs=x)
    mdl.summary()
    mdl.compile(optimizer=optimizers.Adam(), loss=del_etmiss_loss)

    return mdl


# Reconstructed the basic architecture of UNet as best I could. Hopefully this illustrates the way network graphs are built in Keras, so you can implement your own networks with e.g. residual connectivity.
def unet(tracks=False):
    '''
    UNet. I no longer have the parameters used in my report. I know I used significantly more filters (~32?) per convolutional
    layer.
    '''
    n_channels = 4 if not tracks else 6
    layer_opts = {'kernel_initializer': 'RandomUniform',
                'padding': 'valid',
                'data_format': 'channels_first',
                'strides': (1,1)}

    ip = layers.Input((eta_bins, phi_bins, n_channels))

    x1 = Wrap(half_kernel_size=2)(ip)
    x1 = layers.Conv2D(filters=7, kernel_size=5, activation = 'relu',**layer_opts)(x1)
    x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
    
    x2 = Wrap(half_kernel_size=2)(x1)
    x2 = layers.Conv2D(filters=7, kernel_size=5, activation = 'relu',**layer_opts)(x2)
    x2 = layers.MaxPooling2D(pool_size=(5, 2))(x2)

    x3 = Wrap(half_kernel_size=2)(x2)
    x3 = layers.Conv2D(filters=7, kernel_size=5, activation = 'relu',**layer_opts)(x3)
    x3 = layers.UpSampling2D(size=(5, 2))(x3)
    
    x4 = Wrap(half_kernel_size=2)(x3)
    x4 = layers.Conv2D(filters=7, kernel_size=5, activation = 'relu',**layer_opts)(x4)
    x4 = layers.UpSampling2D(size=(2, 2))(x4)
    
    x5 = layers.Conv2D(filters=1, kernel_size=1, activation = 'relu',**layer_opts)(x4)

    mdl = Model(inputs=ip,outputs=x5)
    mdl.summary()
    mdl.compile(optimizer=optimizers.Adam(), loss=del_etmiss_loss)

    return mdl