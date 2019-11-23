"Dataset classes for scalping"
import numpy as np
import pandas as pd 
import os
import sys
import csv
import keras
from datetime import datetime
import pickle
import random
 

########################################################################################
# Clean and comment this
class MinuteDataset():
    """ MinuteDataset object holds data and feeds it in required form,
        either in samples for training or inferemce, or individual timesteps
        for testing purposes.

        The crucial parameters of the dataset are:

        :param pandas.Dataframe df1m, df - dataframe with 1 minute period timeseries
        :param pandas.Dataframe dfticks - datafram with tick timeseries
        :params int index_<train/val/test> - indices which
                                             separate train/val/test sets
        :param list<int> <train/val/test>_indices
                    - list with indices corresponding to sets sampling indices
        :param list<int indices_offsets - list of offsets to obtain sample indices
                                         based on reference index taken from
                                         <train/val/test> indices lists
    
        USAGE:
        1. Initialize the dataset.
        2. Read/Set(and Save) lables for each row of one minute dataframe.
        3. Generate train and validation sets with required undersamplng ratio.
        (4. Optionally match one minute and one tick indices for testinf.)
    """


    
    def __init__(self,read_ticks=True):
        """ Initialization function which reads 1 minute timeseries
           optionally tick timeseries and:
           - renames columns with understandable names
           - finds train/val/test separation dates
           - transforms these dates into 1m dataframe indices
           - sets up the sample architecture (indices offsets to gather)
           - establishes mean spread for fixing

           arguments:
            :param boolean read_ticks - if True the dataset will read
                                        tick data - used for index matching
                                        and testing putposes
        """
           
        # Reading data
        print('Reading.')
        self.df1m = pd.read_pickle(r'.\pickles\df1m.pkl')
        print('Read minute. Initialized.')
        self.read_ticks=read_ticks
        if read_ticks:
            self.dfticks = pd.read_pickle(r'.\pickles\dfticks.pkl')
            print('Read ticks. Initialized.')
        self.Rename()
        self.df = self.df1m
        # Putting minute data into array to speed calculations
        self.ar = self.df.iloc[:,1:6].values
        # Setting division dates and finding corrresponding indices
        self.SetDates()
        self.index_train = self.FindIndexDate(self.df,self.train_date)
        self.index_val = self.FindIndexDate(self.df,self.val_date)
        self.index_test = self.FindIndexDate(self.df,self.test_date)
        self.end_index = self.FindIndexDate(self.df,self.end_date)
        # Diving indices into train, validation and test sets
        self.train_indices = list(range(self.index_val,self.index_train))
        self.val_indices = list(range(self.index_test,self.index_val))
        self.test_indices = list(range(self.end_index,self.index_test))
        # Defining offsets for single feature pack for prediction
        self.indices_offsets = list(range(0,100,1))+list(range(100,300,10)) + list(range(300,1000,30)) + \
                                list (range(1000,4000,60)) + list(range(4000,16000,240)) +\
                                 list(range(16000,72000,1440))+ list(range(72000,26000,10080)) #Which points are taken for minutes and offsets
        # Mean spread
        self.mean_spread = 0.00014 #TODO: change as different for other currency pairs
   
    # Renaming column function
    def Rename(self):
        """ Renaming function for 1 minute and ticks dataframes
        """
        bar_columns = ['time','open','high','low','close','spread']
        self.df1m.columns = bar_columns
        self.df1m['spread']=self.df1m['spread']/100000
        if self.read_ticks:
            if (self.dfticks.columns.__len__()<6):
                self.dfticks.columns = ['time','ms','timesec','ask','bid']
            else:
                self.dfticks.columns = ['time','ms','timesec','ask','bid','minute_index']

    # Setting division dates
    def SetDates(self,train_date='2017.02.01 00:00:32',\
                 val_date='2019.02.16 21:07:24',\
                 test_date='2019.06.20 21:07:24',\
                 end_date='2019.08.16 21:07:24'):
        """ Function with hard coded train/val/test sets separation dates
            The (first) train_date should be so as not to incur out of range
            error when sampling the most earliest dataframe index.
            (the sample includes indices even before that one
             as indices_offsets looks into the past)
        """
        self.train_date=train_date
        self.val_date=val_date
        self.test_date=test_date
        self.end_date=end_date

    def FindIndexDate(self,df,date):
        """ Function which find index based on date string.
            returns:
            :param int - the latest index corresponding to first found date
            earlier than the one specified in the string
            :raises ValueError if not such index exists
        """
        times = df['time'].values
        for i in range(df.shape[0]):
            if times[i]<date:
                return i
        raise ValueError

########################################################################################
    # Labelling the minutes which - after end will experience jump
    # change - number of spreads in jump
    # fut - number of bars looked forward
    def SetLabelsBuy(self,lookup,no_units_change): #comment: changing to using open and not close - NO
        """ Function based on the arguments sets the labels for each index
            stating whether the specified event occured there (1) or not (0).
            The event to predict is a raise in the value by <no_unit_shange>
            units of spread in <lookup> minutes.

            arguments:
            :param int lookup - number of minutes to look into the future for value increase
            :param float no_units change - number of units of spread of required increase
        """
        
        # meanSpread is supposed ot fix as some bars have spread = 0
        mean_spread = self.mean_spread
        print("Setting labels:")
        print("\tMean spread is: meanSpread")
        end_index = lookup # We are ending when cur-lookup would be zero
        start_index = self.ar.shape[0] # Starting from last index
        current_index = start_index #Increasing by one to include the start index
        self.labels = np.zeros(self.ar.shape[0]) # Label matrix
        print('\tStart index:',start_index,'\n\tCurrent index:',current_index,'\n\tEnd index:',end_index)
        while (current_index>end_index):
            current_index=current_index-1
            y = self.ar[(current_index-lookup):current_index:1,3]#[::-1] # Getting close values - uncomment to change order
            y = y - self.ar[current_index,3] # Normalizing
            spread = self.ar[current_index,4] # Getting spread
            how_many_done = (start_index-current_index) # For tracking progress
            # Fixing zero spreads
            if spread<0.00008:
                spread=mean_spread
            difference = no_units_change*spread
            if (y.max()>difference)&(y.min()>-spread):
                self.labels[current_index]=1
            if (how_many_done%5000==0):        
                progress = int(how_many_done/(start_index-end_index)*100)
                print('|'+''.join(['#' for i in range(progress)])+''.join([' ' for i in range(100-progress)])+'| '+ \
                        str(progress)+'%  ',self.labels.sum(),self.labels.sum()/how_many_done,spread,end='\r')
            
        print('|'+''.join(['#' for i in range(100)])+'| 100%')
        print("\tLabeled time stamps: ",int(self.labels.sum()),"\n\tRatio: ",self.labels.sum()/how_many_done)
        self.df['label'] = self.labels
       
        # Saving buy labels - saves time
    def SaveLabelsBuy(self,lookup,no_units_change):
        """ Function which saves the dataframe with labeled classification
            whether the event occured or not.
            arguments:
            :param int lookup - number of minutes to look into the future for value increase
            :param float no_units change - number of units of spread of required increase
        """

        self.df.to_pickle(".\minute_pickles\df_"+str(lookup)+"lookup_"+str(no_units_change)+"unitschange_buy_labels_v0.pkl")

    # Reading buy labels
    def ReadLabelsBuy(self,lookup,no_units_change):
        """ Function which reads a one minute dataframe with labels
            specifying whether the event ocured or not.
            The dataframe is read to self.df.

            arguments:
            :param int lookup - number of minutes to look into the future for value increase
            :param float no_units change - number of units of spread of required increase
        """

        print('Reading buy labels.')
        df = pd.read_pickle(".\minute_pickles\df_"+str(lookup)+"lookup_"+str(no_units_change)+"unitschange_buy_labels_v0.pkl")
        if (self.df.shape[0]!=df.shape[0]):
            raise FileNotFoundError
        self.df=df
        self.labels=df['label'].values
        print("Labels have been read successfully.")

########################################################################################
    # Preparing single unit based on index
    def PrepareUnit(self,index):   
        """ Function which prepares a unit sample i.e.
            value history specified by indices_offsets
            and the label classifying the moment in time.

            argument:
            :param int index - index of the dataframe self.df 
        """

        indices = [index+item for item in self.indices_offsets]
        x = self.ar[indices,1:4]
        y = self.labels[index]
        x = x - x[0,2]
        x = x.reshape(1,-1)
        spread = self.ar[index,4]
        # Fixing spread
        if spread<0.00008:
            spread=self.mean_spread
        # Inserting spread as the first value
        x = np.insert(x,0,spread,axis=1)
        x = x / x.std()
        x = x.reshape(1,-1)
        y = y.reshape(1,-1)
        return x,y

    # Preparing batch
    def PrepareBatch(self,index_list):
        """ Function which prepares a batch of samples based
            on the list with samples' indices.
            note: not a minibatch but could be used as such
                  if there is a need for a generator

            argument:
            :param list<int> index_list - list of indices in the required batch
        """
        N = len(index_list) # Number of samples
        # Batch array width must fit all indices - adding 1 for spread
        batch_array_width = len(self.indices_offsets)*3+1 
        # Creating empty arrays to hold samples
        X = np.zeros((N,batch_array_width))
        Y = np.zeros((N,1))
        for i,index in enumerate(index_list):
            x,y = self.PrepareUnit(index)
            X[i,:] = x
            Y[i,:] = y
        return X,Y

        # Generating X and  Y
    def GenXY(self,no_train_samples=None,no_val_samples=None):
        """ Function which generates train/val/test sets
            from pre-defined list of train/val/test indices.
            The results are given in class variables:
            self.<X/Y>_<train/val/test>
            giving the feature label splif for each set.

            arguments:
            :param int no_train_samples (default: None) -  
                number of train samples required if less than all
                None means all samples will be taken.
            :param int no_val_samples (default: None) -  
                number of val samples required if less than all
                None means all available set samples will be taken.
        """

        # If number of samples not given we take all possible
        if no_train_samples==None:
            no_train_samples = self.train_indices.__len__()
        if no_val_samples==None:
            no_val_samples = self.val_indices.__len__()
        # Creating training and validation sets with data
        self.X_train, self.Y_train = self.PrepareBatch(self.train_indices[:no_train_samples])
        self.X_val, self.Y_val = self.PrepareBatch(self.val_indices[:no_val_samples])
        self.X_test, self.Y_test = self.PrepareBatch(self.test_indices)

    # Shuffling data
    def Shuffle(self):
        """ Function which shuffles insides of class variables
            containing train and validation indices.
        """

        random.shuffle(self.train_indices)
        random.shuffle(self.val_indices)

    # Returning shape of sample for model creation
    def SampleShape(self):
        """ Function which prints out the shape of a sample.
            Useful for dynamic declaration of ML model
            input and output dimensions.

            :returns: number of features (int), number of outputs (int)
        """

        return (self.indices_offsets.__len__()*3+1 , 1)

    # Generating train and validation list with target minority class ratio
    def GenTrainValList(self,ratio,do_validation):
        """ Function which generates train and validation
            indices list with random undersampling based on required
            ratio of positive to all samples.

            The undersampling is done on the training list and optionally
            on the validation list.

            arguments:
            :param float ratio: required ratio of positive to all samples
            :param bool do_validation: if True undersampling with same ratio
                will be performed on the validation set

            Outputs:
            list<int> self.train_indices - list containing train indices
            listint> self.val_indices - list containing validation indices
        """
        # First for train indices
        self.train_indices = list(range(self.index_val,self.index_train))
        print('Generating train and validation set indices.')
        random.shuffle((self.train_indices))
        self.train_labels = self.labels[self.train_indices]
        no_positive_train_samples = self.train_labels.sum()
        no_train_samples = self.train_indices.__len__()
        print('\tPositive samples:',no_positive_train_samples,'\n\tAll samples:',no_train_samples,'\n\tInitial ratio:',no_positive_train_samples/float(no_train_samples))
        # if ratio is required the new set will undersample negative samples
        if (ratio!=None):
            print('\tUndersampling train set.')
            target_no_negative_samples=int((1-ratio)/ratio*no_positive_train_samples)
            print("\t\t Target negative samples:",target_no_negative_samples)
            negative_indices = [self.train_indices[i] for i in np.where(np.isin(self.train_labels,[0]))[0].tolist()]
            positive_indices = [self.train_indices[i] for i in np.where(np.isin(self.train_labels,[1]))[0].tolist()]
            print("\t\t Current negative samples:",negative_indices.__len__())
            self.train_indices = positive_indices + negative_indices[:target_no_negative_samples]
            print('\tTrain set undersampled.')
        self.train_labels = self.labels[self.train_indices]    
        no_positive_train_samples = self.train_labels.sum()
        no_train_samples = self.train_indices.__len__()
        print('\tPositive samples:',no_positive_train_samples,'\n\tAll samples:',no_train_samples,'\n\tOutput ratio:',no_positive_train_samples/float(no_train_samples))
        random.shuffle(self.train_indices) # shuffling at the end

        # for validation indices
        self.val_indices = list(range(self.index_test,self.index_val))
        if do_validation:
            print('\n\tUndersampling validation set.')
            random.shuffle(self.val_indices)
            self.val_labels = self.labels[self.val_indices]
            no_positive_val_samples = self.val_labels.sum()
            no_val_samples = self.val_indices.__len__()
            print('\tPositive samples:',no_positive_val_samples,'\n\tAll samples:',no_val_samples,'\n\tInitial ratio:',no_positive_val_samples/float(no_val_samples))
            target_no_negative_samples=int((1-ratio)/ratio*no_positive_val_samples)
            print("\t\t Target negative samples:",target_no_negative_samples)
            #print(np.where(np.isin(self.train_ar,[1]))[0].tolist())
            negative_indices = [self.val_indices[i] for i in np.where(np.isin(self.val_labels,[0]))[0].tolist()]
            positive_indices = [self.val_indices[i] for i in np.where(np.isin(self.val_labels,[1]))[0].tolist()]
            print("\t\t Current negative samples:",negative_indices.__len__())
            self.val_indices = positive_indices + negative_indices[:target_no_negative_samples]
            print('\tValidation set undersampled.')
            self.val_labels = self.labels[self.val_indices]
            no_positive_val_samples = self.val_labels.sum()
            no_val_samples = self.val_indices.__len__()
            print('\tPositive samples:',no_positive_val_samples,'\n\tAll samples:',no_val_samples,'\n\tOutput ratio:',no_positive_val_samples/float(no_val_samples))
            random.shuffle(self.val_indices)



######################################################################################## 
    # Finding index based on datetime - faster for big arrays
    def FindIndexDatetime(self,dtimes,dtime,last_index):
        """ Function which finds the index in the dtimes array
            at which dtimes[i] is equal of dtime.
            The search begins from last_index as descends in the array.

            arguments:
            :param list<string> dtimes - list of datetimes in string format
            :param string dtime - datetime in string
            :param int last_index - index from which search starts

            :returns: index at which the string is positioned in dtimes array

            :raises: ValueError if dtime is not in dtimes array
        """

        for i in range(last_index-1,-1,-1):
            if dtimes[i]==dtime:
                #print("Last index %d, current index %d"%(last_index,i))
                return i
        
        print("Last index %d, current index %d"%(last_index,i))
        raise ValueError
   
    # Matching tick index to appriopriate minute index - for testing
    def MatchIndex(self,is_this_test=False):
        """ Function which matches each row in the dfticks dataframe
            to the corresponding row in the df1m dataframe.
            Matching is based on the date.

            The row is only matched if the minute on the date has
            just changed (first tick of each minute is matched).
            Other ticks get assigned '-1'. The matched indices 
            are used later during testing to decide whether to make buy or not.

            arguments:
            :param bool this_test - if True only the function will stop
                after first tick is matched and not save the result
            
            output:
            The minute matched tick dataframe is daved to 'minute_pickles'
            folder as 'dfticks_minute_matched.pkl'
        """

        # Creating an empty column with minute indices with default value -1
        self.dfticks['minute_index']=(self.dfticks['ask']*0).astype(int)-1
        self.minute_index_array = self.dfticks['minute_index'].values
        times = self.df1m['time'].values
        # Creating datetime array with the dat in for of an int
        dtimes=np.zeros((times.shape[0]))
        for i in range(0,times.shape[0]):
            dtimes[i] = int(times[i][-5:-3])+100*int(times[i][-8:-6])\
                        +10000*int(times[i][-11:-9])+1000000*int(times[i][-14:-12])\
                        +100000000*int(times[i][-16:-15])
        
        last_index=dtimes.shape[0]
        minute_array = dtimes%100
        last_minute=-1
        no_tick_stamps = self.dfticks.shape[0]
        for i in range(no_tick_stamps-1,-1,-1):
            current_time = self.dfticks['time'].iloc[i]
            current_minute = int(current_time[-5:-3])
            dtime = int(current_time[-5:-3])+100*int(current_time[-8:-6])\
                    +10000*int(current_time[-11:-9])+1000000*int(current_time[-14:-12])\
                    +100000000*int(current_time[-16:-15])
            # if minute changed - means a new entry in df1m
            if (current_minute!=last_minute):
                #print("Current minute",current_minute)
                last_minute-current_minute
                current_date = self.dfticks['time'].iloc[i]
                current_date=current_date+'1'
                corresponding_minute_index = self.FindIndexDatetime(dtimes,dtime,\
                                                                    last_index)
                self.minute_index_array[i]=corresponding_minute_index
                print('Progress %d '%((float(no_tick_stamps-i)/no_tick_stamps)*100),\
                                        '%',end='\r')
                last_index = corresponding_minute_index
                last_minute=current_minute
                if is_this_test:
                    break
                
        # Saving ticks
        self.dfticks['minute_index'] = self.minute_index_array
        if not is_this_test:
            self.dfticks.to_pickle('minute_pickles/dfticks_minute_matched.pkl')

########################################################################################
        
## FINISHED :) 

#Testing
if __name__ == "__main__":
    print("Run as main - TESTING module")

    print('\nTesting initialization... ')
    md = MinuteDataset()
    print(" - Successfully initialized.")

    # print("\nTesting label setting...")
    # lookup = 7
    # no_units_change = 4
    # try:
    #     md.ReadLabelsBuy(lookup=lookup,no_units_change=no_units_change)
    # except:
    #     print('Failed to read')
    #     md.SetLabelsBuy(lookup=lookup,no_units_change=no_units_change)
    #     md.SaveLabelsBuy(lookup=lookup,no_units_change=no_units_change)
    # print(' - Successfully created labels.')

    # print('\nTesting generating train and validation lists - underspampling.')
    # md.GenTrainValList(ratio=0.5,do_validation=True)
    # print('Successfully tested undersampling.\n')
    # print("Testing set preparation.")

    print("\tGenerating training, validation and test sets.")
    # ds.GenXY(1000,500)
    # print("\tTraining set shape:",ds.X_train.shape,ds.Y_train.shape)
    # print("\tValidation set shape:",ds.X_val.shape,ds.Y_val.shape)
    print("Successfully tested sample generation.\n")

    print("Testing index matching.")
    # md.MatchIndex(is_this_test=False)
    print("Minute index matching successfully performed.\n")
