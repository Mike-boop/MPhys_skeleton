# Setup & Requirements

This code was tested with Python 3.8.5 on MacOS. The package requirements for running the examples are given in requirements.txt. Install these with: pip install -r requirements.txt (run within a blank virtual environment.)

If you want to use xgboost you'll need OpenMP. On my system I can use "brew install libomp" to get this.

# Data

You should be able to access some data to get you started from my email. This data is restricted in eta (-2.5 to 2.5), whereas in my project I mainly focussed on  full calorimeter images.

The data that I used for the project was stored in the HDF5 file format. Gabriel (Alan's student) is probably the best person to talk to about generating these from ntuples (and obtaining ntuples in the first place...)

Once you have data, you will need to update data directories in train_network.py, as well as in both notebooks.

I reviously reweighted some .h5 files to have a fairly uniform MET histogram between 0 to ~100 GeV. If you're interested in using the CNN stuff, I'd reccomend doing this again - unfortunately I don't have the code for this anymore (but it wasn't pretty in the first place.)

# Contents

There are two main notebooks in this repo.

1. SimpleModels.ipynb - this notebook demonstrates most of the non-convolutional techniques that I used. This includes some data pre-processing, model training/evaluation, and basic feature analysis. I've included an example of using GridSearchCV at the bottom so that you can understand how hyperparameters were selected.

2. ConvolutionalModels.ipynb - this notebook demonstrates the basic plot used during the evaluation of convolutional models, as well three feature analysis techniques (weight extraction, observing activations, and gradient ascent). The gradient ascent part can be quite fiddly, see more here (original code link) https://keras.io/examples/generative/deep_dream/

To train a convolutional model, you will need to run train_network.py. To use a trained model in the notebook, you will need to update the filename to which its weights are saved.

The purpose of other files are listed:
- config.json: carries some global variables (dimensions of images, batch size etc.)
- custom_layers.py: implements a wrap function, which applies periodic boundary conditions to calorimeter images.
- data_generators.py: implements a basic DataGenerator object (used to efficiently load training data).
- deepnets.py: implementations of deep CNNs. I actually lost these, but this file contains pointers to the code that I adapted.
- etmiss_utils.py: functions for calculating etmiss from images.
- models.py: implementations of convolutional models. I didn't keep many of these, so if you're interested in these then you may have to spend some time fixing parameters in e.g. UNet.

I should mention that a lot of the CNN code is left over from Bernard's analysis, including the use of DataGenerators, most of the train_network.py file and 'BB_model'.

# Getting started with the CNNs: walkthrough

The CNN aspect of this repo is a bit messy - sorry! I'll try and walk you through it here so that it's not too nightmarish getting started.

1. Locate the directories of your .h5 data files. Update the main() function with these directories (you canlist multiple data files in "files = [...]")
2. Choose one of the CNN architectures defined in models.py. Also decide on whether you want to include tracking information. Update this information in the main loop. This code is initiated with "BB_model" architecture, with no tracking information "tracks=False"
3. Run train_network.py. If you run into issues here, make sure that there are directories named "jars" and "trained_models" in this repo directory. Once you've run the code, navigate to train_models and record the name of the trained model (I used a random model_id to help me keep track of different trained models.)
4. Start the ConvolutionalModels notebook. In the "Load data" section, update the data directory. In the "Load model" section, update the filepath of the weights file that was created in step 3. You should also make sure that the architecture specified in this section matches that of the trained model.

That should be all you need. The SimpleModels notebook is a lot more straightforward to get going with.

# Things to note

The 'simple models' were trained with 20000 events in my project, although more is probably better. The grid search was quite coarse so it might be worth redoing this. I'm getting AUC scores near 0.9 with my small local dataset, so hopefully if you try with 20000+ events you should reproduce my reported scores of 0.95+.

You should also try this with the full eta range (-5 to 5), my local dataset only contains info from -2.5 to 2.5. I used the full eta range in my report.

I had poor enough judgement to update to the latest version of tensorflow, which only supports 'channels last' (NHWC) format. This is why there is a lot of strange indexing/reshaping/expand_dims/np.newxis floating around.
