# Tack forecasting

Model for forecast a tack event in sailing from sensor data. Interview task for T-DAB.


## Requirements

  - Python 3.8

The Python packages needed are listed in requirements.txt


## Setup

### Unix setup with virtualenv
  '''bash
  virtualenv venv
  source venv/bin/activate
  pip install -U pip
  deactivate
  source venv/bin/activate
  pip install -r requirements.txt
  '''


## Model explanation

The tack forecasting model consists of a VAR model for forecasting the future state of the sailing boat and a random forest for classifying whether a tack is occurring in this state.

The assumptions made in designing this model were:
  
  - Tacking is a function of the current state only. 
  - When not tacking, the state can be described by an auto-regressive process and therefore is wide-sense stationary.
  - When tacking, the state can be made wide-sense stationary.

These assumptions were verified empirically before selecting and constructing a model.


## Model evaluation

The model is composed of forecasting and classification components. The complete model can be evaluated using the eval_model.py script in the src directory. This script takes an optional argument with the path to a CSV file containing a single (labelled) sailing sequence to be used for evaluating the model accuracy. If no path is passed, a random sequence from the test dataset is chosen for evaluation. Usage:

python eval_model.py --datapath PATH/TO/DATA.csv


The classification model can similarly be evaluated (only on the test set, not custom data) by running:

python eval_classifier.py


## API usage

The run_model.py script shows how to use the API to forecast from a new datapoint (note that a history of datapoints of a user-specified length is required for forecasting).
The eval_model.py shows how this can be extended to an input stream of datapoints, as is the case in real-world application.

  
