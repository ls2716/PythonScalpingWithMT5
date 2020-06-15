# ScalpingPythonMT5

## Closed project - go to [conclusions.md](conclusions.md) to read summary and conclusions of the project.

## Table of contents / current status
- [getterLib.py](getterLib.py) - done
- [datasetLib.py](datasetLib.py) - working + improvements
- [modelLib.py](modelLib.py) - done + improvements
- [evaluateLib.py](evaluateLib.py) - done + improvements
- [testLib.py](testerLib.py) - done + improvements

## getterLib.py

This module is used for getting the data from the MT5 files folder into the working directory, putting it into pandas dataframes with understandable names and then saving the pickled dataframes into 'pickles' folder from which they can be fetched by a module downstream.

## datasetLib.py

This module is used for data preparation and data feeding. The data about timeseries is read from the 'pickles' folder and saved in Dataset object which then can be fetched to the machine learning model or testing and evaluation objects.

The data from df1m.pkl (one minute tick data) goes through following transformations on initialization:
- columns are renamed into sensible names
- appriopriate separation dates (start, validation, end) are translated into one minute dataframe indices

The module supplies functions for:
- finding prediction targets - sudden change in value with specified direction,magnitude (in terms of spread) and minutes of future lookup
- unit sample preparation with feature data containing 'close' values from past (specified by indices_offsets variable) and class label 1 or 0 specifing target event occurance
- batch preparation (whole or part of dataset)
- data resampling for rare event prediction
- dfticks dataframe index matching - function adds column to dataframe with tick data to indicate index of df1m to which it corresponds (for testing purposes)

## modelLib.py

This module is used for creating, training a deep neural network and then using it for inference on datasets. The ceated ensembles of models are saved withing 'minute_models' directory with appriopriate name correspondng to the event they are trained to predict.

## evauateLib.py

This model is used for evaluation of models based on specified metrics appriopriate for the problem of rare event prediction (e.g. precision/recall curve, confusion matrix). Evaluation is based on test set.

## testLib.py

This model is used for testing the model on historical data which is the penultimate success metric - either the model is accurate engough to generate net profit or not.
The testing is done on individual ticks to make it as if the position was evaluated in real time.
The ultimate test is when script will be joined with MT5 evaluation.
