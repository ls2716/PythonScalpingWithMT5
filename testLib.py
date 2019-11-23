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
    def __init__(self, dataset):
        """ Initialization function which assigns dataset and model
            as well as finds the starting index in the ticks dataframe.

            Arguments:
            :param dataset - MinuteDataset object with the data
            :param model - SclMinModel object with trained model
        """
        self.dataset = dataset
        try:
            self.df1m = pd.read_pickle('./minute_pickles/test_pickles/df1m_houred.pkl')
            self.dfticks_mm = pd.read_pickle('./minute_pickles/test_pickles/dfticks_minute_matched_houred.pkl')
        except:
            self.df1m = self.dataset.df1m.copy()
            self.dfticks_mm = pd.read_pickle('minute_pickles/dfticks_minute_matched.pkl')
            self.FindStartIndex()
            self.AddHour()
        self.FindStartIndex()
        

        

    # Finds starting index in tick which corresponds to split_ind2 on minute dataframe
    def FindStartIndex(self):
        self.start_ind_tick = self.dfticks_mm[(self.dfticks_mm['minute_index']==self.dataset.index_test)].index[0]
        self.end_ind_tick = self.dfticks_mm[(self.dfticks_mm['minute_index']==self.dataset.end_index)].index[0]
        print('Starting index tick: ',self.start_ind_tick,'row:',self.dfticks_mm.iloc[self.start_ind_tick])
        print('Starting index minute: ',self.dataset.index_test,'row:',self.dfticks_mm.iloc[self.dataset.index_test])
        print('Ending index tick: ',self.end_ind_tick,'row:',self.dfticks_mm.iloc[self.end_ind_tick])
        print('Ending index minute: ',self.dataset.end_index,'row:',self.dfticks_mm.iloc[self.dataset.end_index])


    # Adds hour info - purpose for escaping weekend break
    def AddHour(self):
        print('Adding hour info...')
        self.df1m['hour'] = (self.df1m['high']*0).astype(int)
        self.df1m['dtime'] = (self.df1m['high']*0).astype(int)
        hour = np.zeros((self.df1m.shape[0]))
        dtime = np.zeros((self.df1m.shape[0]))
        times = self.df1m['time'].values
        for i in range(self.dataset.end_index,(self.dataset.index_test+1)):
            t = times[i]
            hour[i] = int(t[-8:-6])
            dtime[i] = 100*int(t[-5:-3])+10000*int(t[-8:-6])+1000000*int(t[-11:-9])+100000000*int(t[-14:-12])
        self.df1m['hour'] = hour
        self.df1m['dtime'] = dtime
        self.dfticks_mm['hour'] = (self.dfticks_mm['minute_index']*0).astype(int)
        self.dfticks_mm['dtime'] = (self.dfticks_mm['minute_index']*0).astype(int)
        hour = np.zeros((self.dfticks_mm.shape[0]))
        dtime = np.zeros((self.dfticks_mm.shape[0]))
        times = self.dfticks_mm['time'].values
        for i in range(times.shape[0]):
            t = times[i]
            hour[i] = int(t[-8:-6])
            dtime[i] = int(t[-2:])+ 100*int(t[-5:-3])+10000*int(t[-8:-6])+1000000*int(t[-11:-9])+100000000*int(t[-14:-12])
        self.dfticks_mm['hour'] = hour
        self.dfticks_mm['dtime'] = dtime
        self.df1m.to_pickle('./minute_pickles/test_pickles/df1m_houred.pkl')
        self.dfticks_mm.to_pickle('./minute_pickles/test_pickles/dfticks_minute_matched_houred.pkl')
        print('Done. Saved houred.' )


    def AssignModel(self, lookup, no_units_change, change_direction):
        print("Assigning model.")
        self.model = SclMinModel(dataset=self.dataset, lookup=lookup,\
                    no_units_change=no_units_change, change_direction=change_direction)
        self.model.CreateModel(model_type='locally_connected')



    def PredictOnTest(self, lookup, no_units_change, change_direction, no_models):
        self.AssignModel(lookup, no_units_change, change_direction)
        print('Predicting on test %s...'%(change_direction),end='')
        self.dataset.GenXY(2)
        X_true, Y_true = self.dataset.X_test, self.dataset.Y_test
        Y_pred = self.model.EnsemblePredict(X_true.reshape(self.model.input_shape), no_models=no_models)
        vals = np.zeros((self.df1m.shape[0],1))
        vals[self.dataset.test_indices] = Y_pred
        self.df1m[change_direction] = vals
        print(' - Done')
        print('Saving...',end='')
        directory = 'minute_pickles/test_pickles/'
        self.dfticks_mm.to_pickle(directory+'dfticktest_%dlookup_%dunitchange_%s'\
                                    %(lookup, no_units_change, change_direction) +'.pkl')
        self.df1m.to_pickle(directory+'df1mtest_%dlookup_%dunitchange_%s'\
                                    %(lookup, no_units_change, change_direction)+'.pkl')

    # Reads test pickles with hour info
    def ReadBuyPickles(self, lookup, no_units_change, change_direction='buy'):
        print('Reading pickles...',end='')
        directory = 'minute_pickles/test_pickles/'
        self.dfticks_mm_buy = pd.read_pickle(directory+'dfticktest_%dlookup_%dunitchange_%s'\
                                    %(lookup, no_units_change, change_direction) +'.pkl')
        self.df1m_buy = pd.read_pickle(directory+'df1mtest_%dlookup_%dunitchange_%s'\
                                    %(lookup, no_units_change, change_direction)+'.pkl')
        print(' - Done.')

    


    """ Model evaluation functions"""


    # Running simulation
    def RunSim(self, lookup, no_units_change, threshold):
        print('\nRunning simulation.\n')
        self.spread=0.00015
        self.verbose=False
        self.lookup = lookup
        self.no_units_change = no_units_change
        self.balance=0
        self.balances=[]
        self.no_buys=0
        self.no_sells=0
        self.minute_ar = self.df1m_buy[['buy']].values
        self.tick_ar = self.dfticks_mm.iloc[:,3:7].values
        self.unit=100
        self.bought=False
        self.minutes_passed = 0
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
            elif (self.minute_index!=-1):
                if (self.minute_ar[int(self.minute_index)+1,0]>threshold)&(self.bought==False):
                    self.minutes_passed+=1
                    self.buy()
                    
                    self.minute_index_buy = self.minute_index
                    # self.plot_buy()

                if self.bought:
                    self.examine_buy()

            cur_ind-=1
            # if self.no_buys>500:
            #     self.verbose=True
        print('Balance:',self.balance,'no transactions: ',self.no_buys)
        # plt.plot(self.balances)
        # plt.show()
        return (self.balance, self.no_buys)


    def close_buy(self):
        self.balance+=(self.stoplossbuy-self.price_bought)*self.unit
        self.bought=False
        if self.verbose:
            print('Closed buy: price_bought =',self.price_bought,'price_sold =',self.cur_bid,'balance = ',self.balance)
            print(self.df1m.iloc[int(self.minute_index)+1:int(self.minute_index)-1:-1,:],'\n\n')
            self.plot_buy()
        self.no_buys+=1
        self.balances.append(self.balance)
        

    def buy(self):
        self.minutes_passed = 0
        self.spread_bought = self.spread
        self.bought=True
        self.price_bought=self.cur_ask
        self.stoplossbuy=self.cur_bid-self.spread
        if self.verbose:
            print('Tran: ',self.no_buys+1)
            print('Bought =',self.price_bought,'Cur_bid =',self.cur_bid,'stoploss = ',self.stoplossbuy,'spread = ',self.spread)
            print(self.df1m.iloc[int(self.minute_index)+1:int(self.minute_index)-1:-1,:])
    


    def examine_buy(self):
        self.minutes_passed+=1
        
            
        if (self.cur_bid<self.stoplossbuy):
            #print('Sell since stoploss')
            self.stoplossbuy = self.cur_bid
            self.close_buy()
        elif (self.minutes_passed==self.lookup):
            #print('Sell since minute passed')
            self.stoplossbuy = self.cur_bid
            self.close_buy()
            
        elif (self.cur_bid>self.stoplossbuy+2.05*self.spread):
            self.stoplossbuy=self.cur_bid-2*self.spread
            if self.verbose:
                print('Adjusted stoploss',self.stoplossbuy,'Cur bid',self.cur_bid,'Spread',self.cur_ask-self.cur_bid)
        elif (self.cur_bid>self.price_bought+(self.no_units_change-1)*self.spread):
            self.stoplossbuy = self.cur_bid
            #print('Sell since expected gain achieved')
            self.close_buy()
            



    def plot_buy(self):
        start_plot_min = int(self.minute_index_buy)+1
        end_plot_min = start_plot_min-10
        start_plot_tick = self.dfticks_mm[(self.dfticks_mm['minute_index']==int(self.minute_index_buy)+1)].index[0]
        end_plot_tick = self.dfticks_mm[(self.dfticks_mm['minute_index']==end_plot_min)].index[0]
        #print(start_plot_tick)
        #print(end_plot_tick)
        minute_array = self.df1m[['open','dtime']].iloc[end_plot_min:start_plot_min,:].values
        tick_array = self.dfticks_mm[['bid','dtime']].iloc[end_plot_tick:start_plot_tick,:].values
        plt.scatter(np.mod(minute_array[:,1],1000000),minute_array[:,0])
        plt.scatter(np.mod(minute_array[9,1],1000000),self.price_bought,c='r')
        plt.plot(np.mod(minute_array[:,1],1000000),[self.stoplossbuy]*10,c='c',)
        plt.plot(np.remainder(tick_array[:,1],1000000),tick_array[:,0],c='g')
        plt.plot(np.mod(minute_array[:,1],1000000),[minute_array[9,0]+self.spread_bought*1]*10,linestyle=':')
        plt.plot(np.mod(minute_array[:,1],1000000),[minute_array[9,0]-self.spread_bought*1]*10,linestyle=':')
        plt.ylim(minute_array[9,0]-self.spread_bought*5,minute_array[9,0]+self.spread_bought*5)
        #print(self.df1m.iloc[end_plot_min:start_plot_min,:])
        plt.show()
        


if __name__ == "__main__":
    
    print("\nRun as main - TESTING module")

    print('\nDataset initialization... ')
    md = MinuteDataset(read_ticks=False)
    print(" - Successfully initialized.")
   
    

    print("\nCreating model.")
    
    print("Successfully created model.")

    lookup = 5
    no_units_change = 3
    no_models = 3
    # smm.ModelTrain(how_many_models=no_models, epochs=100)
    print("\nTesting ModelTester initialization.")
    mt = ModelTester(dataset=md)
    print("Successfully created ModelTester object.")
    

    


    print("\nEvaluating prediction.")
    md.ReadLabelsBuy(lookup=lookup, no_units_change=no_units_change)
    mt.PredictOnTest(lookup=lookup, no_units_change=no_units_change,\
       change_direction='buy', no_models=no_models)
    mt.ReadBuyPickles(lookup=lookup, no_units_change=no_units_change)
    print("Successfully evaluated prediction.")


    print("\nRunning buy simulation")
    for th in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print('Running with threshold:',th)
        mt.RunSim(lookup=lookup, no_units_change=no_units_change, threshold=th)
    print("Successfully ran simulation.")























        # # Adds the day of the week - not used
    # def AddDayofWeek(self):
    #     print('Adding weekday info...')
    #     self.df1m['weekday'] = (self.df1m['high']*0).astype(int)
    #     weekday = np.zeros((self.df1m.shape[0]))-1
    #     for i in range(self.dataset.end_index,(self.dataset.index_test+1)):
    #         t = self.df1m.iloc[i]['time']
    #         weekday[i] = datetime.fromisoformat(t.replace('.','-')).weekday()
    #     self.df1m['weekday'] = weekday+1
    #     print('Done')