import tensorflow as tf
import tensorflow.python.keras.backend as K
import numpy as np
import math
import json

config = json.load(open("config.json","r"))

eta_bins = config["imageparams"]["eta_bins"]
eta_min = config["imageparams"]["eta_min"]
eta_max = config["imageparams"]["eta_max"]
phi_bins = config["imageparams"]["phi_bins"]
phi_min = config["imageparams"]["phi_min"]
phi_max = config["imageparams"]["phi_max"]

def get_centres(n_bins, b_min, b_max):
    """Calculate bin centers from bins."""
    hist, edges = np.histogram([], n_bins, range=(b_min, b_max))
    centers = [(edges[i + 1] + edges[i]) / 2 for i in range(len(edges) - 1)]
    return np.array(centers)
    
def get_phi_tensor(get_numpy=False):
    """Create the matrix used to calculate ETmiss from a pt 2d hist in eta, phi."""
    eta_centres = get_centres(eta_bins, eta_min, eta_max)
    phi_centres = get_centres(phi_bins, phi_min, phi_max)

    phi_x = np.zeros((eta_bins, phi_bins))
    phi_y = np.zeros((eta_bins, phi_bins))
    for phi, phi_centre in enumerate(phi_centres):
        cosinus = math.cos(phi_centre)
        sinus = math.sin(phi_centre)
        for eta, eta_centre in enumerate(eta_centres):
            phi_x[eta][phi] = -1. * cosinus
            phi_y[eta][phi] = -1. * sinus

    phi_x = phi_x[:, np.newaxis]
    phi_y = phi_y[:, np.newaxis]

    if get_numpy:
        return phi_x, phi_y
    else:
        phi_x_tf = tf.convert_to_tensor(phi_x, np.float32)
        phi_y_tf = tf.convert_to_tensor(phi_y, np.float32)
        return phi_x_tf, phi_y_tf
        
Phi_x,Phi_y = get_phi_tensor(get_numpy=True)

def get_etmiss(img):
    etmiss_x_img = Phi_x*img
    etmiss_y_img = Phi_y*img
    
    etmiss_x = np.sum(etmiss_x_img)
    etmiss_y = np.sum(etmiss_y_img)
    
    return math.sqrt(np.square(etmiss_x)+np.square(etmiss_y))

def get_etmiss_components(img):
    etmiss_x_img = Phi_x*img
    etmiss_y_img = Phi_y*img
    
    etmiss_x = np.sum(etmiss_x_img)
    etmiss_y = np.sum(etmiss_y_img)
    
    return etmiss_x,etmiss_y

phi_x_tf, phi_y_tf = get_phi_tensor()
def del_etmiss_loss(y_true, y_pred):
    '''
    loss function for CNN.
    '''
    
    # y_pred = K.permute_dimensions(y_pred, (0,3,1,2))
    # y_true = K.permute_dimensions(y_true, (0,3,1,2))
    
    diff = y_pred - y_true
    
    et_miss_x = diff * phi_x_tf
    et_miss_x = K.sum(K.sum(et_miss_x, axis=2), axis=1)

    et_miss_y = diff * phi_y_tf
    et_miss_y = K.sum(K.sum(et_miss_y, axis=2), axis=1)

    diff = K.sum(K.sum(diff, axis=2), axis=1)
    return K.mean(K.square(et_miss_x) + K.square(et_miss_y))