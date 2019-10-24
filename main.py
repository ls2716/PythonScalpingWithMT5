"""This script combines libraries and uses them to train and test"""


import numpy as np
import pandas as pd 
import random
import matplotlib.pyplot as plt
from keras import optimizers
import matplotlib.pyplot as plt
from datasetLib import MinuteDataset
from modelLib import SclMinModel
from testLib import TestModel


# Preparation
a = MinuteDataset(read_ticks=True)
change=4
print(change)
try:
    a.ReadUpJumped(change=change)
except:
    print('Failed to read')
    a.Set01Up(change=change)
    a.SaveUpJumped(change=change)

t = TestModel(a)
m = SclMinModel(a)
t.AssignModel(m)
try:
    t.ReadPickles(change)
except:
    t.AddHour()
    t.PredictOnTest(change)
t.RunSim(change=change)