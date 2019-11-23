"Neural network classes for scalping on minutes"
import numpy as np
import pandas as pd 
import random
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LocallyConnected1D, Flatten, Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
import matplotlib.pyplot as plt
from keras import optimizers
import matplotlib.pyplot as plt
from datasetLib import MinuteDataset


class SclMinModel():
    """ SclMinModel object is an object used for creating,
        training multi-layer-perceptron models and using them for inference
        for purposes of scalping with Python.

        The SclMinModel is supposed to be used together with MinuteDataset
        for training/validation data fetching.
    
    
    USAGE:
    1. Initialize the model with dataset and event parameters.
    2. Create model.
    3a. Train ensemble.
    3b. Use ensemble for inference.
    """


    # Model Initialization
    def __init__(self, dataset, lookup, no_units_change, change_direction):
        """ Class initialization function setting up the class variables.
            dataset - minute dataset variable
            lookup, no_units_change - look MinuteDataset 
            change_direction - either 'buy'/'sell' - for naming purposes only
                does not make difference for training or inference
        """
        self.dataset = dataset
        self.lookup = lookup
        self.no_units_change=no_units_change
        self.change_direction=change_direction

    # Creating model with name of the ensemble to save separate models in
    def CreateModel(self, model_type):
        """ Function which initializes the model with hard-coded architecture,
            optimizer, loss function and metrics,
            together with a folder to which save the ensemble models.

            The ensemble_name class variable is created with name of format:
            'ensemble_<lookup>_<no_units_change>_<change_direction>'

            The summary of the model is printed after initialization.
        """


        self.ensemble_name = "ensemble_"+str(self.lookup)+"_"\
                        +str(self.no_units_change)+"_"+self.change_direction
        print("\tEnsemble name:",self.ensemble_name)
        print('Model type:', model_type)
        foldername = "minute_models/"+self.ensemble_name
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        
        if (model_type=='dense_mlp'):
            self.input_shape = (-1, self.dataset.SampleShape()[0])
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
            self.model.add(Dense(self.dataset.SampleShape()[1], activation='sigmoid'))

            adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

            self.model.compile(optimizer=adam,
                loss='binary_crossentropy', #rms does not work very well with sigmoid
                metrics=['mean_absolute_error'])

        elif (model_type=='locally_connected'):
            print('building')
            self.input_shape = (-1, self.dataset.SampleShape()[0], 1)
            self.model = Sequential()

            self.model.add(LocallyConnected1D(4, 60, padding='valid', activation='relu',\
                                    input_shape=(self.dataset.SampleShape()[0],1)))
            self.model.add(Dropout(rate=0.5))
            #self.model.add(LocallyConnected1D(8, 20, padding='valid', activation='relu'))
            #self.model.add(Dropout(rate=0.5))
            self.model.add(Flatten())
            self.model.add(Dense(512, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(rate=0.45))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(rate=0.35))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(rate=0.1))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(rate=0.1))
            self.model.add(Dense(self.dataset.SampleShape()[1], activation='sigmoid'))

            adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

            self.model.compile(optimizer=adam,
                loss='binary_crossentropy', #rms does not work very well with sigmoid
                metrics=['mean_absolute_error'])

            # WORKS BETTER for 5 3
            # self.model = Sequential()
            # self.model.add(LocallyConnected1D(24, 90, padding='valid', activation='relu',\
            #                         input_shape=(self.dataset.SampleShape()[0], 1)))
            # self.model.add(Dropout(rate=0.5))
            # self.model.add(LocallyConnected1D(48, 30, padding='valid', activation='relu'))
            # self.model.add(Dropout(rate=0.5))
            # self.model.add(Flatten())
            # self.model.add(Dense(768, activation='relu'))
        print("\t Printing model summary.")
        self.model.summary()

    # Training enseble of models and saving them in the ensemble folder
    def ModelTrain(self, model_type, how_many_models, epochs):
        """ Function which based on required number of models
            and number of epochs.

            aguments:
            :param int how_many_models - number of models to be trained
            :param int epochs - number of epochs for training

            Models are trained with early stopping of patience 40 and 
            model checkpointing. The models are saved in path
            'minute_models/<ensemble_name>/' with name:
            'bestofmodel_<number_of_model>.h5'.

            The history of training is appended to class variable
            hist_list.
        """

        self.hist_list=[]
        self.train_num=None
        self.val_num=None
        for i in range(0,how_many_models):
            self.CreateModel(model_type=model_type)
            print('Global epoch:',i+1)
            self.dataset.GenTrainValList(ratio=0.4, do_validation=True)
            self.dataset.GenXY(self.train_num,self.val_num)
            X_train, Y_train = self.dataset.X_train.reshape(self.input_shape), self.dataset.Y_train
            X_val, Y_val = self.dataset.X_val.reshape(self.input_shape), self.dataset.Y_val
            filepath='minute_models/'+self.ensemble_name+'/bestofmodel_'+str(i)+'.h5' 
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,\
                         save_best_only=True, mode='min')
            earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,\
                         patience=30)
            callbacks_list = [checkpoint, earlystop]
            hist = self.model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=512,\
                                shuffle=True, validation_data=(X_val,Y_val),\
                                callbacks=callbacks_list)
            self.hist_list.append(hist)
    
    def ModelLoad(self,model_number):
        """ Function which loads each model into memory for inference.
            Not the best solution as this takes unnecessary amount of time.
            :todo: load all models at once

            arguments:
            :param: int model_number - number of model from the ensemble
                cannot be bigger than the total number of models trained
            """

        try:
            self.model.load_weights('minute_models/'+self.ensemble_name+'/bestofmodel_'+str(model_number)+'.h5') 
            print('Successfully loaded.')
        except:
            print('Unable to load model.')

    # Ensemble predict
    def EnsemblePredict(self,X,no_models):
        """ Function which performs ensemble predict for
            given feature array X.

            arguments:
            :param array<float> X - array of rows with features
            :param int no_model - number of models used for prediction
                cannot be greater than the number of trained models 
                in the ensemble
            
            :returns: array Y - array of predictions
        """

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
        """ Function which enhances the jumped array
            deleting data which causes problems
            :todo: TO BE DONE
        """
        pass


    def ShowHistory(self):
        """ Function which shows history of training for models.
            Not finished yet as no need for it was detected.
        """
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
        md.ReadLabelsBuy(lookup=lookup, no_units_change=no_units_change)
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




