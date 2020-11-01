import tensorflow as tf
import numpy as np
import h5py
from etmiss_utils import get_phi_tensor, get_etmiss
import json

config = json.load(open("config.json","r"))
eta_bins = config["imageparams"]["eta_bins"]
phi_bins = config["imageparams"]["phi_bins"]

phi_x, phi_y = get_phi_tensor(get_numpy=True)

# following two functions no longer used; were for data augmentation.

def phi_rotation(X,Y,probability = 0.5):
    '''randomly rotate images through a random phi angle (i.e. translate by a random number of phi bins)'''
    
    if np.random.binomial(1,probability) == False:
        return X,Y
    
    cells_to_shift = randint(-phi_bins,phi_bins)
    return np.roll(X,cells_to_shift,axis=3),np.roll(Y,cells_to_shift,axis=3)

def eta_reflection(X,Y,probability = 0.5):
    '''randomly reflect images through eta=0'''
    
    if np.random.binomial(1,probability) == False:
        return X,Y
    
    return np.flip(X,axis=2),np.flip(Y,axis=2)


class BaseDataGenerator(tf.keras.utils.Sequence):
    '''
    BaseDataGenerator: object for more efficient access to data during training. Inherit from this class
    and redefine data_generation if you need to change the input/output formats.
    '''
    def __init__(self, entries, batch_size, shuffle=True, tracks=False):

        self.entries = entries
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tracks=tracks

        self.on_epoch_end()

    def __len__(self):
        return int(self.entries.shape[0] // self.batch_size)

    def __getitem__(self, index):
        # Generate indices of the batch
        batch_entries = self.entries[index * self.batch_size: (index + 1) * self.batch_size]
        X, Y = self.data_generation(batch_entries)
        return X, Y

    def on_epoch_end(self):
        #Updates indices after each epoch
        if self.shuffle:
            np.random.shuffle(self.entries)

    def data_generation(self, batch_entries):
        
        if self.tracks == True:
            n_channels = 6
        else:
            n_channels = 4
            
        batch_size = self.batch_size
        X = np.zeros((batch_size, n_channels, eta_bins, phi_bins))
        Y = np.zeros((batch_size, 1, eta_bins, phi_bins))

        for entry in range(self.batch_size):
            batch_entry = int(batch_entries[entry][1])
            file = batch_entries[entry][0]
            dataset = h5py.File(file, "r")

            Y[entry][0] = dataset["truth_nm_barcode"][batch_entry]
            
            X[entry][0] = dataset["cluster"][batch_entry]
            X[entry][1] = dataset["SK"][batch_entry]
            X[entry][2] = dataset["VorSK"][batch_entry]
            X[entry][3] = dataset["CSSK"][batch_entry]
            if self.tracks == True:
                X[entry][4] = dataset["tracks_nm"][batch_entry]
                X[entry][5] = dataset["tracks_prime_nm"][batch_entry]

            dataset.close()
        
        X = np.moveaxis(X,1,3)
        Y = np.moveaxis(Y,1,3)

        return X, Y