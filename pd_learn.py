#!/usr/bin/env python

import numpy as np
import pandas as pd


"""
I named this file pandas.py and it imported pandas so it gonna import itself and the code looks like running twice
"""


np.random.seed(100)
r = np.arange(80)
r =r.reshape((8, 10))
r = r.reshape((-1, 1))
r = r.reshape((-1, 8))

# negetive number is just like python list index -1 means the last index
#r = r.reshape((-2, 4))
print(r)

print(r[1:3])
print(r[1:3, 0:3])

print(r[1: 3, 0: 1])
# 退化成一个数组
print(r[1:3, 0])

r = r.reshape(-1, 1)
print(r[0: 4])
print(r[0: 4, 0])
print(r[0, 0])
