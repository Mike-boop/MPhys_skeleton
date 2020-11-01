#!/usr/bin/python

import numpy as np
import h5py
import tensorflow as tf
import pickle
import os
import json
from datetime import date
from random import randint
import models
import data_generators
from etmiss_utils import get_phi_tensor, get_etmiss

#setup environment: configurations stored in ../config.json
config = json.load(open("config.json","r"))
for key in config["environment"].keys():
    os.environ[key] = config["environment"][key]

eta_bins = config["imageparams"]["eta_bins"]
phi_bins = config["imageparams"]["phi_bins"]
eta_min = config["imageparams"]["eta_min"]
eta_max = config["imageparams"]["eta_max"]
phi_min = config["imageparams"]["phi_min"]
phi_max = config["imageparams"]["phi_max"]
batch_size = config["trainingconfig"]["batch_size"]


# change n_entries for debug purposes
def get_sets(files, split_ratio=0.8):
    n_entries = 0
    file_entries = [0 for file in files]
    entries = []
    for i_f in range(len(files)):
        dataset = h5py.File(files[i_f], "r")
        file_entries[i_f] = dataset["entries"][0]
        for i in range(file_entries[i_f]):
            entries.append([files[i_f], i])
        dataset.close()

    n_entries = sum(file_entries)#100
    train_entries = int(n_entries * split_ratio)
    entries_train = entries[:train_entries]
    entries_valid = entries[train_entries:n_entries]
    entries_train = np.array(entries_train)
    entries_valid = np.array(entries_valid)

    return entries_train, entries_valid
    
def get_modelname(tracks,model_id,additional_info=""):
    '''Book-keeping info. Trying to make modelname as verbose as possible to avoid confusion/overwrites.'''
    today = str(date.today())
     
    modelname =  str(model_id) + "_"
    modelname += "tracks" + str(tracks) + "_"
    modelname += today + "_"
    modelname += additional_info
                 
    print(modelname)
    return modelname

def main():
    
    #1. Define the model and its name
    files = ["/Users/michaelthornton/Code/MPhys_skeleton/data/sample_00_fixed.h5"]
    tracks = False
    mdl = models.BB_model(tracks=tracks)
    DataGenerator = data_generators.BaseDataGenerator
    epochs=10
                                                
    model_id = randint(1000,9999)
    modelname =  get_modelname(tracks,model_id)
    
    #2. Get training info
    entries_train, entries_valid = get_sets(files)
    print(entries_train.shape, entries_valid.shape)
    
    steps_per_epoch = entries_train.shape[0] // batch_size
    validation_steps = entries_valid.shape[0] // batch_size
    
    train_gen = DataGenerator(entries_train, batch_size,tracks=tracks)
    valid_gen = DataGenerator(entries_valid, batch_size,tracks=tracks)

    #3. Define checkpoints
    loss_model = tf.keras.callbacks.ModelCheckpoint("models/{}_loss_best.hdf".format(modelname),
                                 save_best_only=True, monitor='val_loss',
                                 mode='min')
    acc_model = tf.keras.callbacks.ModelCheckpoint("models/{}_acc_best.hdf".format(modelname),
                                save_best_only=True, monitor='val_acc',
                                mode='max')

    callbacks = [loss_model, acc_model]

    #4. Train model on dataset
    history = mdl.fit(train_gen,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=valid_gen,
                        validation_steps=validation_steps,
                        callbacks=callbacks,
                        epochs=epochs,
                        #verbose=1,
                        #use_multiprocessing=True,
                        workers=1)
                            
    print("finished training")
    
    #5. Save weights of trained model + training history
    with open('jars/{}_history.pickle'.format(modelname), 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    mdl.save_weights("trained_models/{}_weights.h5".format(modelname))
        
    print("saved model")

if __name__ == "__main__":
    main()