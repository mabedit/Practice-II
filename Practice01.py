# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:56:12 2023

@author: DD8105
"""

import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first 5 rows of the dataset
print(df.head())

