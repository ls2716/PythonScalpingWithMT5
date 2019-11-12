# ScalpingPythonMT5

## Table of contents / current status
- [getterLib.py](getterLib.py) - done
- [datasetLib.py](datasetLib.py) - working + improvements
- [modelLib.py](modelLib.py) - done
- [evaluateLib.py](evaluateLib.py) - done + improvements
- [testerLib.py](testerLib.py) - to be done

## Next: comment all throuroughly - enough for documentation

## getterLib.py

This module is used for getting the data from the MT5 files folder into the working directory, putting it into pandas dataframes with understandable names and then saving the pickled dataframes into 'pickles' folder from which they can be fetched by a module downstream.

## datasetLib.py

This module is used for data preparation and data feeding. The data about timeseries is read from the 'pickles' folder and saved in Dataset object which then can be fetched to the machine learning model or testing and evaluation objects.

The data from df1m.pkl (one minute tick data) goes through following transformations on initialization:
- columns are renamed into sensible names
- appriopriate separation dates (start, validation, end) are translated into dataframe indices

The module supplies functions for:
- finding prediction targets - sudden change in value with specified magnitude in terms of spread and minutes of future lookup
- unit sample preparation with feature data containing 'close' values from past (specified by indices_offsets variable) and class label 1 or 0 specifing target event occurance
- batch preparation (whole or part of dataset)
- data resampling for rare event prediction
- dfticks dataframe index matching - function adds column to dataframe with tick data to indicate index of df1m to which it corresponds (for testing purposes)