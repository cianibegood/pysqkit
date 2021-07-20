#-----------------------------------------------------------------------------
# Script to plot the result of the scans for a Ficheux-like gate between
# a fluxonium and a transmon as a function of drive frequency and 
# drive strength 
#-----------------------------------------------------------------------------

#%%
import numpy as np 
import matplotlib.pyplot as plt
import json

#%%
pippo = [1, 2, 3, 4]

with open("test.txt", "w") as fp:
    json.dump(pippo, fp)

# %%
