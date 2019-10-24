""" Evaluator for neural network model """
import numpy as np
import pandas as pd 
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
import matplotlib.pyplot as plt
from keras import optimizers
import matplotlib.pyplot as plt
from datasetLib import MinuteDataset
from modelLib import SclMinModel

class ModelEvaluator():

    def __init__(self,dataset,model,no_models,lookup,no_units_change,change_direction):
        self.dataset = dataset
        self.model = model
        self.no_models = no_models
        self.lookup = lookup
        self.no_units_change=no_units_change
        self.change_direction=change_direction

    # Evaluating model on test dataset using precision matrix
    def ModelEvaluateConfustionMatrix(self,threshold):
        print('\tEvaluating.')
        self.dataset.GenXY(2)
        X_true,Y_true = self.dataset.X_test, self.dataset.Y_test
        Y_pred = self.model.EnsemblePredict(X_true,no_models=self.no_models)
        self.wynik = np.concatenate((Y_true,Y_pred),axis=1)
        self.wynik_int = np.around(self.wynik).astype(int)
        self.tp=0
        self.tn=0
        self.fp=0
        self.fn=0
        self.threshold = threshold
        tmp = self.wynik
        self.true_positives =[]
        self.false_positives=[]
        for i in range(tmp.shape[0]):
            if (tmp[i,0]>threshold):
                if (tmp[i,1]>threshold):
                    self.tp+=1
                    self.true_positives.append([self.dataset.test_indices[i],self.wynik[i,1]])
                else:
                    self.fn+=1
            else:
                if (tmp[i,1]>threshold):
                    self.fp+=1
                    self.false_positives.append([self.dataset.test_indices[i],self.wynik[i,1]])
                else:
                    self.tn+=1
        print('\tTrue positive:',self.tp,'True negative:',self.tn)
        print('\tFalse positive:',self.fp,'False negative',self.fn)
        print('\tPositive ratio:',(self.tp+self.fn)/float(self.fp+self.fn+self.tp+self.tn))
        print('\tPredicting ratio:',(self.tp+self.fp)/float(self.fp+self.fn+self.tp+self.tn))
        print('\tPrecision:',(self.tp)/float(self.fp+self.tp))
        self.true_positives = np.array(self.true_positives)
        self.false_positives = np.array(self.false_positives)
        plt.scatter(self.false_positives[:,0],self.false_positives[:,1],label='False')
        plt.scatter(self.true_positives[:,0],self.true_positives[:,1],label='True')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    print("\nRun as main - TESTING module")

    print('\nDataset initialization... ')
    md = MinuteDataset(read_ticks=False)
    print(" - Successfully initialized.")
   
    print("\nGetting label setting...")
    lookup = 7
    no_units_change = 3
    try:
        md.ReadLabelsBuy(lookup=lookup,no_units_change=no_units_change)
    except:
        print('Failed to read - Setting buy labels.')
        md.SetLabelsBuy(lookup=lookup,no_units_change=no_units_change)
        md.SaveLabelsBuy(lookup=lookup,no_units_change=no_units_change)
    print("Successfully loaded labels.")

    print("\nCreating model.")
    smm = SclMinModel(dataset=md,lookup=lookup,no_units_change=no_units_change,change_direction='buy')
    smm.CreateModel()
    print("Successfully created model.")

    no_models = 1
    print("\nTesting Evaluator initialization.")
    me = ModelEvaluator(dataset=md,model=smm,no_models=no_models,lookup=lookup,no_units_change=no_units_change,change_direction='buy')
    print("Successfully created Evaluator object.")
    
    threshold = 0.5
    print("\nEvaluating confuction matrix for the model.")
    me.ModelEvaluateConfustionMatrix(threshold=threshold)
    print("Successfully evaluated confusion matrix for the model.")