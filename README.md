# Setup & Requirements

This code was tested with Python 3.8.5 on MacOS. The package requirements for running the examples are given in requirements.txt. Install these with: pip install -r requirements.txt (run within a blank virtual environment.)

If you want to use xgboost you'll need OpenMP. On my system I can use "brew install libomp" to get this.

# Data

I should have provided a link to some example data in my email. If I didn't do this, feel free to check in with me. Due to the limited data I have leftover for testing this code, the same data were used for training, validation and testing. Obviously do rectify this in a real analysis!

The data that I used for the project was stored in the HDF5 file format. Gabriel (Alan's student) is probably the best person to talk to about generating these from ntuples (and obtaining ntuples in the first place...)

Once you have data, you will need to update data directories in train_network.py, as well as in both notebooks.

I reweighted some .h5 files to have a fairly uniform MET histograme between 0 to ~100 GeV. If you're interested in using the CNN stuff, I'd reccomend doing this again - unfortunately I don't have the code for this anymore (but it wasn't pretty in the first place.)

# Contents

There are two main notebooks in this repo. I'm afraid I've had some data loss issues, but all of the essential components of the project should be outlined in the notebooks. If there's anything missing, feel free to get in touch.

1. SimpleModels.ipynb - this notebook demonstrates most of the non-convolutional techniques that I used. This includes some data pre-processing, model training/evaluation, and basic feature analysis. I've included an example of using GridSearchCV at the bottom so that you can understand how hyperparameters were selected.

2. ConvolutionalModels.ipynb - this notebook demonstrates the basic plot used during the evaluation of convolutional models, as well three feature analysis techniques (weight extraction, observing activations, and gradient ascent). The gradient ascent part can be quite fiddly, see more here (original code link) https://keras.io/examples/generative/deep_dream/

To train a convolutional model, you will need to run train_network.py. To use a trained model in the notebook, you will need to update the filename to which its weights are saved.

The purpose of other files are listed:
- config.json: carries some global variables (dimensions of images, batch size etc.)
- custom_layers.py: implements a wrap function, which applies periodic boundary conditions to calorimeter images.
- data_generators.py: implements a basic DataGenerator object (used to efficiently load training data).
- deepnets.py: implementations of deep CNNs
- etmiss_utils.py: functions for calculating etmiss from images.
- models.py: implementations of convolutional models. I didn't keep many of these, so if you're interested in these then you may have to spend some time fixing parameters in e.g. UNet.

I should also mention that a lot of the CNN code is left over from Bernard's analysis, including the use of DataGenerators, most of the train_network.py file and 'BB_model'.

# Getting started with the CNNs: walkthrough

The CNN aspect of this repo is a bit messy - sorry! I'll try and walk you through it here so that its not too nightmarish getting started.

1. Locate the directories of your .h5 data files. Update the main() function with these directories (you canlist multiple data files in "files = [...]")
2. Choose one of the CNN architectures defined in models.py. Also decide on whether you want to include tracking information. Update this information in the main loop. This code is initiated with "BB_model" architecture, with no tracking information "tracks=False"
3. Run train_network.py. If you run into issues here, make sure that there are directories named "jars" and "trained_models" in this repo directory. Once you've run the code, navigate to train_models and record the name of the trained model (I used a random model_id to help me keep track of different trained models.)
4. Start the ConvolutionalModels notebook. In the "Load data" section, update the data directory. In the "Load model" section, update the filepath of the weights file that was created in step 3. You should also make sure that the architecture specified in this section matches that of the trained model.

That should be all you need. The SimpleModels notebook is a lot more straightforward to get going with.