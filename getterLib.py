import numpy as np
import pandas as pd 
import os
import sys
import csv


class Getter():
    """ Getter object which gets the currency timeseries data
        from the MT5 folder.
    """
    
    def __init__(self,data_dir="000"):
        """ Initialization - creating dataframes to hold timeseries data
            with names df<period>
        """
        self.data_dir = data_dir
        self.df1m = pd.DataFrame()
        self.df5m = pd.DataFrame()
        self.df15m = pd.DataFrame()
        self.df30m = pd.DataFrame()
        self.df1h = pd.DataFrame()
        self.df4h = pd.DataFrame()
        self.df1d = pd.DataFrame()
        self.df1w = pd.DataFrame()
        self.df1mn = pd.DataFrame()
        self.dfticks = pd.DataFrame()
        self.periods = ['1m','5m','15m','30m','1h','4h','1d','1w','1mn']  

    def CheckDir(self):
        """ Checking MT5 folder reference function -
           whether one exists or not yet set up.
        """
        if (self.data_dir=="000"):
            print("Data directory not initialized.")
            return False
        elif (os.path.isdir(self.data_dir)):
            print("Data directory correct.")
            print(self.data_dir)
            return True 
        else:
            print("Data directory does not exist.")
            print(self.data_dir)
            return False

    

    def SetDefaultDir(self):
        """ Function hard-coding directory with currency data
        """
        print('Setting Default Data Directory')
        self.data_dir = r"C:\Users\Lukasz\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files"

    def ReadData(self):
        """ Function which reads the data from csv files
           in the MT5 folder into pandas dataframes
        """
        print("Reading data:")
        print('Reading 1 minute', end='',flush=True)
        self.df1m = pd.read_csv(os.path.join(self.data_dir,'TrainData','1m.csv'),header=None)
        print(' - Read 1 minute.')
        print('Reading 5 minute', end='',flush=True)
        self.df5m = pd.read_csv(os.path.join(self.data_dir,'TrainData','5m.csv'),header=None)
        print(' - Read 5 minute.')
        print('Reading 15 minute', end='',flush=True)
        self.df15m = pd.read_csv(os.path.join(self.data_dir,'TrainData','15m.csv'),header=None)
        print(' - Read 15 minute.')
        print('Reading 30 minute', end='',flush=True)
        self.df30m = pd.read_csv(os.path.join(self.data_dir,'TrainData','30m.csv'),header=None)
        print(' - Read 30 minute.')
        print('Reading 1 hour', end='',flush=True)
        self.df1h = pd.read_csv(os.path.join(self.data_dir,'TrainData','1h.csv'),header=None)
        print(' - Read 1 hour.')
        print('Reading 4 hour', end='',flush=True)
        self.df4h = pd.read_csv(os.path.join(self.data_dir,'TrainData','4h.csv'),header=None)
        print(' - Read 4 hour.')
        print('Reading 1 day', end='',flush=True)
        self.df1d = pd.read_csv(os.path.join(self.data_dir,'TrainData','1d.csv'),header=None)
        print(' - Read 1 day.')
        print('Reading 1 week', end='',flush=True)
        self.df1w = pd.read_csv(os.path.join(self.data_dir,'TrainData','1w.csv'),header=None)
        print(' - Read 1 week.')
        print('Reading 1 month', end='',flush=True)
        self.df1mn = pd.read_csv(os.path.join(self.data_dir,'TrainData','1mn.csv'),header=None)
        print(' - Read 1 month.')
        print('Reading ticks', end='',flush=True)
        self.dfticks = pd.read_csv(os.path.join(self.data_dir,'TrainData','ticks.csv'),header=None)
        print(' - Read ticks.')
        print('All data read successfully.')
        print('Reindexing tick data.')
        self.dfticknew = self.dfticks.iloc[::-1].reset_index()
        self.dfticknew = self.dfticknew.drop(columns=['index',0])
        print('Reindexed.')

    def SaveData(self):
        """ Funcion which saves dataframes as pickles
           into 'pickles' folder.
        """
        self.df1m.to_pickle(r'.\pickles\df1m.pkl')
        self.df1m.to_pickle(r'.\minute_pickles\df1m.pkl')
        self.df5m.to_pickle(r'.\pickles\df5m.pkl')
        self.df15m.to_pickle(r'.\pickles\df15m.pkl')
        self.df30m.to_pickle(r'.\pickles\df30m.pkl')
        self.df1h.to_pickle(r'.\pickles\df1h.pkl')
        self.df4h.to_pickle(r'.\pickles\df4h.pkl')
        self.df1d.to_pickle(r'.\pickles\df1d.pkl')
        self.df1w.to_pickle(r'.\pickles\df1w.pkl')
        self.df1mn.to_pickle(r'.\pickles\df1mn.pkl')
        self.dfticknew.to_pickle(r'.\pickles\dfticks.pkl')
        self.dfticknew.to_pickle(r'.\minute_pickles\dfticks.pkl')

    def ReadSaveFormat(self):
        pass
        

if __name__ == "__main__":
    a=Getter()
    a.SetDefaultDir()
    if (a.CheckDir()):
        a.ReadData()
        a.SaveData()