import json
import math
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import preprocessing as pre
import classifcation as run

print(pre.X_train[0], pre.y_train[0])