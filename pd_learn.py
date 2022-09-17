#!/usr/bin/env python

import numpy as np
import pandas as pd


"""
I named this file pandas.py and it imported pandas so it gonna import itself and the code looks like running twice
"""


np.random.seed(100)
r = np.arange(12)
r =r.reshape((4, 3))
r = r.reshape((-1, 1))
r = r.reshape((1, -1))

# negetive number is just like python list index -1 means the last index
r = r.reshape((-2, 4))
print(r)
