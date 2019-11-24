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
from copy import deepcopy

class ModelEvaluator():
    """ Object used for evaluating models for Python Scalping
        It shoulb be used in conjunction with Minutedataset and SclMinModel.
    
        USAGE:
        1. Initiate
        2. Use any evaluation function
    """

    def __init__(self,dataset,model,no_models,lookup,no_units_change,change_direction):
        """ Initialization function. The input variable are explanatory
            based on other models. the dataset is the MinuteDataset object.
            The model is the SclMinModel object.
        """

        self.dataset = dataset
        self.model = model
        self.no_models = no_models
        self.lookup = lookup
        self.no_units_change=no_units_change
        self.change_direction=change_direction
        self.wynik_done = False
        self.wynik_val_done = False

    # Evaluating model on test dataset using precision matrix
    def ModelEvaluateConfusionMatrix(self, threshold, onval=False):
        """ Evaluating confusion matrix based on threshold.
        """

        print('\tEvaluating confusion matrix.')
        # Generating test set
        if onval:
            if (not self.wynik_val_done):
                self.dataset.GenXY(2, None)
                X_true, Y_true = self.dataset.X_val, self.dataset.Y_val
                Y_pred = self.model.EnsemblePredict(X_true.reshape(self.model.input_shape), no_models=self.no_models)
                self.wynik_val = np.concatenate((Y_true, Y_pred), axis=1)
                self.wynik_val_done = True
            tmp = self.wynik_val
            print('Evaluating on validation set.')
        else:
            if (not self.wynik_done):
                self.dataset.GenXY(2,2)
                X_true, Y_true = self.dataset.X_test, self.dataset.Y_test
                Y_pred = self.model.EnsemblePredict(X_true.reshape(self.model.input_shape), no_models=self.no_models)
                self.wynik = np.concatenate((Y_true, Y_pred), axis=1)
                self.wynik_done = True
            tmp = self.wynik
            print('Evaluating on test set.')

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
            if (tmp[i,0]>=threshold):
                if (tmp[i,1]>=threshold):
                    self.tp+=1
                    self.true_positives.append([self.dataset.test_indices[i],self.wynik[i,1]])
                else:
                    self.fn+=1
            else:
                if (tmp[i,1]>=threshold):
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

    def ModelEvaluateRocCurve(self, thresholds=[], onval=False):
        """ Function which evaluates ROC curve for the model based on the threshold
        """
        
        if onval:
            if (not self.wynik_val_done):
                self.dataset.GenXY(2, None)
                X_true, Y_true = self.dataset.X_val, self.dataset.Y_val
                Y_pred = self.model.EnsemblePredict(X_true.reshape(self.model.input_shape), no_models=self.no_models)
                self.wynik_val = np.concatenate((Y_true, Y_pred), axis=1)
                self.wynik_val_done = True
            tmp = self.wynik_val
            print('Evaluating on validation set.')
        else:
            if (not self.wynik_done):
                self.dataset.GenXY(2,2)
                X_true, Y_true = self.dataset.X_test, self.dataset.Y_test
                Y_pred = self.model.EnsemblePredict(X_true.reshape(self.model.input_shape), no_models=self.no_models)
                self.wynik = np.concatenate((Y_true, Y_pred), axis=1)
                self.wynik_done = True
            tmp = self.wynik
            print('Evaluating on test set.')
        
        if (thresholds.__len__()<1):
            thresholds = np.linspace(0, 1, 20, endpoint=False)
        tprs = []
        fprs = [] 
        for threshold in thresholds:
            print("Evaluating for:", threshold)
            wynik_thresholded = deepcopy(tmp)
            wynik_thresholded[:,1] = tmp[:,1]>=threshold
            self.wynik_thresholded = wynik_thresholded.astype(int)
            print(sum(self.wynik_thresholded[:,1]))
            tp_array = wynik_thresholded[:,0] * wynik_thresholded[:,1]
            fp_array = (1 - wynik_thresholded[:,0]) * wynik_thresholded[:,1]
            fn_array = (wynik_thresholded[:,0]) * (1 - wynik_thresholded[:,1])
            tn_array = (1 - wynik_thresholded[:,0]) * (1 - wynik_thresholded[:,1])
            print(sum(tp_array), sum(fp_array))
            tpr = sum(tp_array)/(sum(tp_array)+sum(fn_array))
            fpr = sum(fp_array)/(sum(fp_array)+sum(tn_array))
            tprs.append(tpr)
            fprs.append(fpr)
            print("Evaluated for:", threshold, "tpr:", tpr, "fpr:", fpr)
        plt.plot(fprs, tprs)
        plt.show()

    def ModelEvaluatePrecisionCurve(self, thresholds=[], onval=False, plot=True):
        """ Function which precision vs classification threshold
        """
        if onval:
            if (not self.wynik_val_done):
                self.dataset.GenXY(2, None)
                X_true, Y_true = self.dataset.X_val, self.dataset.Y_val
                Y_pred = self.model.EnsemblePredict(X_true.reshape(self.model.input_shape), no_models=self.no_models)
                self.wynik_val = np.concatenate((Y_true, Y_pred), axis=1)
                self.wynik_val_done = True
            tmp = self.wynik_val
            print('Evaluating on validation set.')
        

            if (thresholds.__len__()<1):
                thresholds = np.linspace(0, 1, 50, endpoint=False)
            precisions = []
            best_th = None
            for i, threshold in enumerate(thresholds):
                print("Evaluating for:", threshold)
                wynik_thresholded = deepcopy(tmp)
                wynik_thresholded[:,1] = tmp[:,1]>=threshold
                self.wynik_thresholded = wynik_thresholded.astype(int)
                ratio = sum(self.wynik_thresholded[:,1])/self.wynik_thresholded.shape[0]
                
                tp_array = wynik_thresholded[:,0] * wynik_thresholded[:,1]
                fp_array = (1 - wynik_thresholded[:,0]) * wynik_thresholded[:,1]
                fn_array = (wynik_thresholded[:,0]) * (1 - wynik_thresholded[:,1])
                tn_array = (1 - wynik_thresholded[:,0]) * (1 - wynik_thresholded[:,1])
                precision = 0
                try:
                    precision = sum(tp_array)/(sum(tp_array)+sum(fp_array))
                except:
                    pass
                if (ratio<0.02)&(onval)&(best_th==None):
                    best_th = threshold
                    best_i = i
                    best_th_precision = precision
                    print('Best threshold:', best_th)
                print('Positives:',sum(self.wynik_thresholded[:,1]),\
                'Ratio:',sum(self.wynik_thresholded[:,1])/self.wynik_thresholded.shape[0],\
                'Precision:', precision)
                precisions.append(precision)
            if plot:
                plt.plot(thresholds, precisions)
                plt.title('Validation set precision curve.')
                plt.show()

        if (not self.wynik_done):
            self.dataset.GenXY(2,2)
            X_true, Y_true = self.dataset.X_test, self.dataset.Y_test
            Y_pred = self.model.EnsemblePredict(X_true.reshape(self.model.input_shape), no_models=self.no_models)
            self.wynik = np.concatenate((Y_true, Y_pred), axis=1)
            self.wynik_done = True
        tmp = self.wynik
        print('Evaluating on test set.')

        if (thresholds.__len__()<1):
            thresholds = np.linspace(0, 1, 50, endpoint=False)
        precisions = []
        for i, threshold in enumerate(thresholds):
            print("Evaluating for:", threshold)
            wynik_thresholded = deepcopy(tmp)
            wynik_thresholded[:,1] = tmp[:,1]>=threshold
            self.wynik_thresholded = wynik_thresholded.astype(int)
            ratio = sum(self.wynik_thresholded[:,1])/self.wynik_thresholded.shape[0]
            
            

            tp_array = wynik_thresholded[:,0] * wynik_thresholded[:,1]
            fp_array = (1 - wynik_thresholded[:,0]) * wynik_thresholded[:,1]
            fn_array = (wynik_thresholded[:,0]) * (1 - wynik_thresholded[:,1])
            tn_array = (1 - wynik_thresholded[:,0]) * (1 - wynik_thresholded[:,1])
            precision = 0
            try:
                precision = sum(tp_array)/(sum(tp_array)+sum(fp_array))
            except:
                pass
            print('Positives:',sum(self.wynik_thresholded[:,1]),\
                'Ratio:',sum(self.wynik_thresholded[:,1])/self.wynik_thresholded.shape[0],\
                'Precision:', precision)
            precisions.append(precision)
        if (onval):
            best_th_test_precision = precisions[best_i]
        if plot:
            plt.plot(thresholds, precisions)
            plt.title('Test set precision curve')
            plt.show()
        

        if onval:
            print('Best threshold:', best_th)
            print('Validation precision:', best_th_precision)
            print('Test precision:', best_th_test_precision)
            print('Expected gain:', best_th_test_precision*(self.no_units_change-1)-(1-best_th_test_precision)*2)
            return best_th
        

if __name__ == "__main__":
    print("\nRun as main - TESTING module")

    print('\nDataset initialization... ')
    md = MinuteDataset(read_ticks=False)
    print(" - Successfully initialized.")
   
    print("\nGetting label setting...")
    lookup = 5
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
    smm.CreateModel(model_type='locally_connected')
    print("Successfully created model.")

    no_models = 1
    #smm.ModelTrain(model_type='locally_connected', how_many_models=no_models, epochs=20)
    print("\nTesting Evaluator initialization.")
    me = ModelEvaluator(dataset=md, model=smm, no_models=no_models, lookup=lookup,\
            no_units_change=no_units_change,change_direction='buy')
    print("Successfully created Evaluator object.")
    

    print("\nEvaluating ROC curve for the model.")
    # me.ModelEvaluateRocCurve(thresholds=[])
    print("Successfully evaluated ROC curve for the model.")

    # print("\nEvaluating precision curve for the model.")
    # me.ModelEvaluatePrecisionCurve(thresholds=[], onval=True)
    # print("Successfully evaluated precision curve for the model.")

    print("\nEvaluating precision curve for the model on the validation set.")
    threshold = me.ModelEvaluatePrecisionCurve(thresholds=[], onval=True)
    print("Successfully evaluated precision curve for the model on the validaion set.")

    #threshold = 0.6
    print("\nEvaluating confusion matrix for the model.")
    me.ModelEvaluateConfusionMatrix(threshold=threshold)
    print("Successfully evaluated confusion matrix for the model.")

