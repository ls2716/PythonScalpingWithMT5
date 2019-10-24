"Neural network classes for scalping on minutes"
import numpy as np
import pandas as pd 
import random
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
import matplotlib.pyplot as plt
from keras import optimizers
import matplotlib.pyplot as plt
from datasetLib import MinuteDataset


class SclMinModel():

    # Model Initialization
    def __init__(self,dataset,lookup,no_units_change,change_direction):
        self.dataset = dataset
        self.lookup = lookup
        self.no_units_change=no_units_change
        self.change_direction=change_direction

    # Creating model with name of the ensemble to save separate models in
    def CreateModel(self):
        self.ensemble_name = "ensemble_"+str(self.lookup)+"_"+str(self.no_units_change)+"_"+self.change_direction
        print("\tEnsemble name:",self.ensemble_name)
        foldername = "minute_models/"+self.ensemble_name
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        self.model = Sequential()
        self.model.add(Dense(2056, activation='relu', input_dim=self.dataset.SampleShape()[0]))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.45))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(rate=0.35))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(rate=0.1))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(rate=0.1))
        self.model.add(Dense(self.dataset.SampleShape()[1]))

        adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        self.model.compile(optimizer=adam,
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])

        print("\t Printing model summary.")
        self.model.summary()

    # Training enseble of models and saving them in the ensemble folder
    def ModelTrain(self,how_many_models, epochs):
        self.hist_list=[]
        self.train_num=None
        self.val_num=None
        for i in range(0,how_many_models):
            self.CreateModel()
            print('Global epoch:',i+1)
            self.dataset.GenTrainValList(ratio=0.4,do_validation=True)
            self.dataset.GenXY(self.train_num,self.val_num)
            X_train,Y_train = self.dataset.X_train, self.dataset.Y_train
            X_val,Y_val = self.dataset.X_val, self.dataset.Y_val
            filepath='minute_models/'+self.ensemble_name+'/bestofmodel_'+str(i)+'.h5' 
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
            callbacks_list = [checkpoint,earlystop]
            hist = self.model.fit(x=X_train,y=Y_train,epochs=epochs,batch_size=512,shuffle=True,validation_data=(X_val,Y_val),callbacks=callbacks_list)
            self.hist_list.append(hist)
    
    def ModelLoad(self,model_number):
        try:
            self.model.load_weights('minute_models/'+self.ensemble_name+'/bestofmodel_'+str(model_number)+'.h5') 
            print('Successfully loaded.')
        except:
            print('Unable to load model.')

    # Ensemble predict
    def EnsemblePredict(self,X,no_models):
        print('Ensemble predict.')
        Y_pred=np.zeros(X.shape[0])
        Y_pred=Y_pred.reshape(-1,1)
        for i in range(no_models):
            self.ModelLoad(model_number=i)
            Y_pred+=self.model.predict(X)
        Y_pred=Y_pred/no_models
        return Y_pred

    # Enhancing jumped arrrays to exclude problematic samples
    def EnhanceJumped(self):
        """Function which enhances the jumped array
        deleting data which causes problems
        """
        pass


    def ShowHistory(self):
        pass

    
    
            


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

    print("\nTesting training.")
    smm.ModelTrain(how_many_models=1,epochs=10)
    print("Successfully trained and saved model.")




