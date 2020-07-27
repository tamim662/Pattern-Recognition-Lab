import pandas as pd
import numpy as np

link1=input("Enter the path of csv file:")
data = pd.read_csv(link1) 
# Preview the first 5 lines of the loaded data 
# data.head()

data.drop(['1'], axis = 1, inplace = True) 
# data.head()

#data.to_csv('Extracted/test.csv',index=False)


npy=data.to_numpy()
# type(npy)
link2=input("Enter the path to create npy file:")
np.save(link2,npy)
