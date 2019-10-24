""" Tester for Neural Network model"""
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
from datetime import datetime

class ModelTester():

    # Initialisation
    def __init__(self,dataset):
        self.dataset=dataset
        self.df1mtest = self.dataset.df1m.copy()
        self.dfticktest = self.dataset.dfticks.copy()
        self.FindStartPoint()

    # Adds the day of the week - not used
    def AddDayofWeek(self):
        print('Adding weekday info...')
        self.df1mtest['weekday'] = (self.df1mtest['high']*0).astype(int)
        weekday = np.zeros((self.df1mtest.shape[0]))-1
        for i in range(self.dataset.end_ind,(self.dataset.split_ind2+1)):
            t = self.df1mtest.iloc[i]['time']
            weekday[i] = datetime.fromisoformat(t.replace('.','-')).weekday()
        self.df1mtest['weekday'] = weekday+1

        print('Done')

    # Adds hour info - pupose for escaping weekend break
    def AddHour(self):
        print('Adding hour info...')
        self.df1mtest['hour'] = (self.df1mtest['high']*0).astype(int)
        self.df1mtest['dtime'] = (self.df1mtest['high']*0).astype(int)
        hour = np.zeros((self.df1mtest.shape[0]))
        dtime = np.zeros((self.df1mtest.shape[0]))
        times = self.df1mtest['time'].values
        for i in range(self.dataset.end_ind,(self.dataset.split_ind2+1)):
            t = times[i]
            hour[i] = int(t[-8:-6])
            dtime[i] = 100*int(t[-5:-3])+10000*int(t[-8:-6])+1000000*int(t[-11:-9])+100000000*int(t[-14:-12])
        self.df1mtest['hour'] = hour
        self.df1mtest['dtime'] = dtime
        self.dfticktest['hour'] = (self.dfticktest['minute_index']*0).astype(int)
        self.dfticktest['dtime'] = (self.dfticktest['minute_index']*0).astype(int)
        hour = np.zeros((self.dfticktest.shape[0]))
        dtime = np.zeros((self.dfticktest.shape[0]))
        times = self.dfticktest['time'].values
        for i in range(times.shape[0]):
            t = times[i]
            hour[i] = int(t[-8:-6])
            dtime[i] = int(t[-2:])+ 100*int(t[-5:-3])+10000*int(t[-8:-6])+1000000*int(t[-11:-9])+100000000*int(t[-14:-12])
        self.dfticktest['hour'] = hour
        self.dfticktest['dtime'] = dtime
        print('Done.')

    # Finds starting index in tick which corresponds to split_ind2 on minute dataframe
    def FindStartPoint(self):
        self.start_ind_tick = self.dfticktest[(self.dfticktest['minute_index']==self.dataset.split_ind2)].index[0]
        self.end_ind_tick = self.dfticktest[(self.dfticktest['minute_index']==self.dataset.end_ind)].index[0]
        print('Starting index tick: ',self.start_ind_tick,'row:',self.dfticktest.iloc[self.start_ind_tick])
        print('Starting index minute: ',self.dataset.split_ind2,'row:',self.df1mtest.iloc[self.dataset.split_ind2])
        print('Ending index tick: ',self.end_ind_tick,'row:',self.dfticktest.iloc[self.end_ind_tick])
        print('Ending index minute: ',self.dataset.end_ind,'row:',self.df1mtest.iloc[self.dataset.end_ind])

    # Assigns model
    def AssignModel(self,model):
        self.model = model

    # Reads test pickles with hour info
    def ReadPickles(self,change):
        print('Reading pickles...',end='')
        directory = 'minute_pickles/test_pickles/'
        self.dfticktest = pd.read_pickle(directory+'dfticktest'+str(change)+'.pkl')
        self.df1mtest = pd.read_pickle(directory+'df1mtest'+str(change)+'.pkl')
        print(' - Done.')

    def PredictOnTest(self,change):
        print('Predicting on test UP...',end='')
        self.dataset.GenXY(2)
        X_true,Y_true = self.dataset.X_test, self.dataset.Y_test
        self.model.CreateModel(change,'up')
        Y_pred = self.model.EnsemblePredict(X_true)
        vals = np.zeros((self.df1mtest.shape[0],1))
        vals[self.dataset.test_indices] = Y_pred
        self.df1mtest['UP'] = vals
        print(' - Done')
        # print('Predicting on test DOWN...',end='')
        # self.model.CreateModel(change,'down')
        # Y_pred = self.model.EnsemblePredict(X_true)
        # vals = np.zeros((self.df1mtest.shape[0],1))
        # vals[self.dataset.test_indices] = Y_pred
        # self.df1mtest['DOWN'] = vals
        # print(' - Done.')
        print('Saving...',end='')
        directory = 'minute_pickles/test_pickles/'
        self.dfticktest.to_pickle(directory+'dfticktest'+str(change)+'.pkl')
        self.df1mtest.to_pickle(directory+'df1mtest'+str(change)+'.pkl')


    """ Model evaluation functions"""


    # Running simulation
    def RunSim(self,change,threshold=0.92):
        print('\n\n\nRunning simulation.\n\n\n')
        self.spread=0.00015
        self.change=change
        self.verbose=False
        
        self.balance=0
        self.balances=[]
        self.no_buys=0
        self.no_sells=0
        # # self.minute_ar = self.df1mtest[['UP','DOWN']].values
        self.minute_ar = self.df1mtest[['UP']].values
        self.tick_ar = self.dfticktest.iloc[:,3:7].values
        self.unit=100
        self.bought=False
        # self.sold=False
        cur_ind = self.start_ind_tick
        
        while (cur_ind>self.end_ind_tick):
            self.minute_index = self.tick_ar[cur_ind,2]
            cur_hour = self.tick_ar[cur_ind,3]
            next_hour = self.tick_ar[cur_ind-1,3]
            self.cur_ask = self.tick_ar[cur_ind,0]
            self.cur_bid = self.tick_ar[cur_ind,1]
            self.spread =self.cur_ask-self.cur_bid
            if (cur_hour-next_hour==23):
                if self.bought:
                    self.close_buy()
                # # if self.sold:
                # #     self.close_sell()
            elif (self.minute_index!=-1):
                if (self.minute_ar[int(self.minute_index)+1,0]>threshold)&(self.bought==False):
                    self.buy()
                    self.minute_index_buy = self.minute_index
                    
                # # elif (self.minute_ar[int(self.minute_index)+1,1]>threshold)&(self.sold==False):
                # #     self.sell()
                # #     pass
            if self.bought:
                self.examine_buy()
                # # if self.sold:
                # #     self.examine_sell()
            cur_ind-=1
            # if self.no_buys>500:
            #     self.verbose=True
        print('Balance:',self.balance,'no transactions: ',self.no_buys)
        plt.plot(self.balances)
        plt.show()


    def close_buy(self):
        self.balance+=(self.stoplossbuy-self.price_bought)*self.unit
        self.bought=False
        if self.verbose:
            print('Closed buy: price_bought =',self.price_bought,'price_sold =',self.cur_bid,'balance = ',self.balance)
            print(self.df1mtest.iloc[int(self.minute_index)+1:int(self.minute_index)-1:-1,:],'\n\n\n')
            self.plot_buy()
        self.no_buys+=1
        self.balances.append(self.balance)
        

    def buy(self):
        self.spread_bought = self.spread
        self.bought=True
        self.price_bought=self.cur_ask
        self.stoplossbuy=self.cur_bid-2*self.spread
        if self.verbose:
            print('Tran: ',self.no_buys+1)
            print('Bought =',self.price_bought,'Cur_bid =',self.cur_bid,'stoploss = ',self.stoplossbuy,'spread = ',self.spread)
            print(self.df1mtest.iloc[int(self.minute_index)+1:int(self.minute_index)-1:-1,:])
    


    def examine_buy(self):
        if (self.cur_bid<self.stoplossbuy):
            self.close_buy()
            
        if (self.cur_bid>self.stoplossbuy+2.05*self.spread):
            self.stoplossbuy=self.cur_bid-2*self.spread
            if self.verbose:
                print('Adjusted stoploss',self.stoplossbuy,'Cur bid',self.cur_bid,'Spread',self.cur_ask-self.cur_bid)
        if (self.cur_bid>self.price_bought+1*self.spread):
            self.stoplossbuy = self.cur_bid
            self.close_buy()



    def plot_buy(self):
        start_plot_min = int(self.minute_index_buy)+1
        end_plot_min = start_plot_min-10
        start_plot_tick = self.dfticktest[(self.dfticktest['minute_index']==int(self.minute_index_buy)+1)].index[0]
        end_plot_tick = self.dfticktest[(self.dfticktest['minute_index']==end_plot_min)].index[0]
        #print(start_plot_tick)
        #print(end_plot_tick)
        minute_array = self.df1mtest[['open','dtime']].iloc[end_plot_min:start_plot_min,:].values
        tick_array = self.dfticktest[['bid','dtime']].iloc[end_plot_tick:start_plot_tick,:].values
        plt.scatter(np.mod(minute_array[:,1],1000000),minute_array[:,0])
        plt.scatter(np.mod(minute_array[9,1],1000000),self.price_bought,c='r')
        plt.plot(np.remainder(tick_array[:,1],1000000),tick_array[:,0],c='g')
        plt.plot(np.mod(minute_array[:,1],1000000),[minute_array[9,0]+self.spread_bought*1]*10)
        plt.plot(np.mod(minute_array[:,1],1000000),[minute_array[9,0]-self.spread_bought*1]*10)
        plt.ylim(minute_array[9,0]-self.spread_bought*5,minute_array[9,0]+self.spread_bought*5)
        #print(self.df1mtest.iloc[end_plot_min:start_plot_min,:])
        plt.show()
        

    # Evaluating model on test dataset using precision matrix
    def ModelEvaluate(self,threshold=0.5):
        print('Evaluating.')
        self.dataset.GenXY(2)
        X_true,Y_true = self.dataset.X_test, self.dataset.Y_test
        #Y_pred = self.model.predict(X_true)
        Y_pred = self.EnsemblePredict(X_true)
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
        print('True positive:',self.tp,'True negative:',self.tn)
        print('False positive:',self.fp,'False negative',self.fn)
        print('Positive ratio:',(self.tp+self.fn)/float(self.fp+self.fn+self.tp+self.tn))
        print('Predicting ratio:',(self.tp+self.fp)/float(self.fp+self.fn+self.tp+self.tn))
        print('Precision:',(self.tp)/float(self.fp+self.tp))
        self.true_positives = np.array(self.true_positives)
        self.false_positives = np.array(self.false_positives)
        plt.scatter(self.false_positives[:,0],self.false_positives[:,1],label='False')
        plt.scatter(self.true_positives[:,0],self.true_positives[:,1],label='True')
        
        plt.legend()
        plt.show()


