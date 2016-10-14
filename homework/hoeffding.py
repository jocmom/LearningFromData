import math
import random
import numpy as np

experiment_cnt = 100
flip_cnt = 10
coin_cnt = 1000
v_1 = []
v_rand = []
v_min = []
c_min = []

for i in range(0,experiment_cnt):
    # init 10 coin flips for 1000 fair coins
    v = [[np.random.randint(0,2) for i in range(0,flip_cnt)] for j in range(0,coin_cnt)]

    # get first, random and minimum flip
    v_1.append(sum(v[0])/flip_cnt)
    v_rand.append(sum(v[np.random.randint(0,coin_cnt)]) / flip_cnt)
    v_min.append(sum(min(v))/flip_cnt)

print(v_min)
print(v_rand)
print(v_1)
print("Average of v_min", sum(v_min)/experiment_cnt)
print("Average of v_1", sum(v_1)/experiment_cnt)
print("Average of v_rand", sum(v_rand)/experiment_cnt)

