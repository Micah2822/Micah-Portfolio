import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("/Users/micahokoko/Desktop/Micah-Portfolio/Fight_Prediction_Model/dataset/dataset.csv")
# punchingstats = pd.read_csv("/Users/micahokoko/Desktop/Micah-Portfolio/Fight_Prediction_Model/dataset/punchingstats.csv")


# Convert to numerical
#dataset['sex'] = dataset['sex'].map({'male': 1, 'female': 0})
#

# One hot coding
#final_data = pd.get_dummies(dataset, columns=['name', 'opposition']).dropna()

#X = final_data.drop('clean_outcome', axis=1)  # features (drop the target variable)
#Y = final_data['clean_outcome']  # target variable



dataset = dataset[dataset['clean_outcome'].notna()]
dataset = dataset[dataset['clean_outcome'] != 'Unknown']

#categorical to numerical
# dataset.loc[(dataset['clean_outcome'] == 'Win KO')|(dataset['clean_outcome'] == 'Win Other'), 'clean_outcome'] = 1
# dataset.loc[(dataset['clean_outcome'] == 'Draw'), 'clean_outcome'] = 2
# dataset.loc[(dataset['clean_outcome'] == 'Loss Other')|(dataset['clean_outcome'] == 'Loss KO'), 'clean_outcome'] = 3
dataset['clean_outcome'] = dataset['clean_outcome'].map({'win': 1, 'loss': 0})


X = dataset[['opp_last6','opp_KO ratio','Win KO','opp_winOther','Loss Other','Loss KO','Win Other','KO ratio','opp_winOther',
             'KnockedOut ratio','opp_winKO','opp_KO ratio','opp_loss','opp_lossOther','opp_win','Draw','opp_lossKO']]
X = X.fillna(0)

Y = dataset['clean_outcome']
Y = Y.fillna(0)



# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
x_train, x_, y_train, y_ = train_test_split(X, Y, test_size=0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

# Delete temporary
del x_, y_