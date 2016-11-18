import numpy as np

data = np.genfromtxt("in.dta")
# get all features, 
X = data[:,0:-1]
# get y, last column
y = data[:,-1]

sample_cnt = X.shape[0]

