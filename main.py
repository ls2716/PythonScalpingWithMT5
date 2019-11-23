"""This script combines libraries and uses them to train and test"""


import numpy as np
import pandas as pd 
import random
import matplotlib.pyplot as plt
from keras import optimizers
import matplotlib.pyplot as plt
from datasetLib import MinuteDataset
from modelLib import SclMinModel
from evaluateLib import ModelEvaluator
from testLib import ModelTester



def TrainModel(lookup, no_units_change):
   
    print("\nGetting label setting...")
    try:
        md.ReadLabelsBuy(lookup=lookup,no_units_change=no_units_change)
    except:
        print('Failed to read - Setting buy labels.')
        md.SetLabelsBuy(lookup=lookup,no_units_change=no_units_change)
        md.SaveLabelsBuy(lookup=lookup,no_units_change=no_units_change)
    print("Successfully loaded labels.")

    print("\nCreating model.")
    smm = SclMinModel(dataset=md,lookup=lookup,no_units_change=no_units_change,change_direction='buy')
    print("Successfully created model.")

    print('Training model.')
    no_models = 1
    smm.ModelTrain(model_type='locally_connected', how_many_models=no_models, epochs=10)
    print('Model trained')
    return smm

def EvaluateModel(lookup, no_units_change, smm, no_models = 1):
    print("\nTesting Evaluator initialization.")
    me = ModelEvaluator(dataset=md, model=smm, no_models=no_models, lookup=lookup,\
            no_units_change=no_units_change,change_direction='buy')
    print("Successfully created Evaluator object.")
    
    print("\nGetting label setting...")
    try:
        md.ReadLabelsBuy(lookup=lookup,no_units_change=no_units_change)
    except:
        print('Failed to read - Setting buy labels.')
        md.SetLabelsBuy(lookup=lookup,no_units_change=no_units_change)
        md.SaveLabelsBuy(lookup=lookup,no_units_change=no_units_change)
    print("Successfully loaded labels.")

    print("\nEvaluating ROC curve for the model.")
    me.ModelEvaluateRocCurve(thresholds=[])
    print("Successfully evaluated ROC curve for the model.")

    threshold = 0.3
    print("\nEvaluating confusion matrix for the model.")
    me.ModelEvaluateConfusionMatrix(threshold=threshold)
    print("Successfully evaluated confusion matrix for the model.")


def TestModel(lookup, no_units_change, smm, no_models = 1):
    print("\nTesting ModelTester initialization.")
    mt = ModelTester(dataset=md)
    print("Successfully created ModelTester object.")

    print("\nPreparing simulation. Evaluating on dataset")
    try:
        mt.ReadBuyPickles(lookup=lookup, no_units_change=no_units_change)
    except:
        md.ReadLabelsBuy(lookup=lookup, no_units_change=no_units_change)
        mt.PredictOnTest(lookup=lookup, no_units_change=no_units_change,\
            change_direction='buy', no_models=no_models)
        mt.ReadBuyPickles(lookup=lookup, no_units_change=no_units_change)
    print("Successfully prepared dataframes for simulation.")
    print("\nRunning buy simulations.")
    results = []
    for th in [ 0.7, 0.8, 0.9]:
        print('Running with threshold:',th)
        
        results.append(mt.RunSim(lookup=lookup, no_units_change=no_units_change, threshold=th))
    print("Successfully ran simulation.")
    print("Results:",results)
    return results



print('\nDataset initialization... ')
md = MinuteDataset(read_ticks=False)
print(" - Successfully initialized.")

lookups = [5,7,7,9]
no_units_changes = [5,6,7,7]

RESULTS = []
for lookup, no_units_change in zip(lookups,no_units_changes):
    # model = TrainModel(lookup=lookup, no_units_change=no_units_change)
    model = SclMinModel(dataset=md, lookup=lookup, no_units_change=no_units_change, change_direction='buy')
    model.CreateModel(model_type='locally_connected')
    EvaluateModel(lookup=lookup, no_units_change=no_units_change, smm=model)
    cur_results = TestModel(lookup=lookup, no_units_change=no_units_change, smm=model)
    RESULTS.append(cur_results)


print(RESULTS)

np.save('results.npy', np.array(RESULTS))


