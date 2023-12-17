from sklearn.model_selection import train_test_split
import pandas as pd

dataset = pd.read_csv("https://github.com/Micah2822/Micah-Portfolio/blob/main/Boxing%20Match%20Outcome%20Prediction/Fight_Prediction_Model/dataset/dataset.csv")

dataset = dataset[dataset['clean_outcome'].notna()]
dataset = dataset[dataset['clean_outcome'] != 'Unknown']

#categorical to numerical
dataset['clean_outcome'] = dataset['clean_outcome'].map({'win': 1, 'loss': 0})

X = dataset[['opp_last6', 'opp_KnockedOut ratio', 'opp_winOther',  'opp_winKO', 'opptotalaccuracy', 'oppavgtotalagainst',
                    'opp_KO ratio', 'opp_loss', 'opp_lossOther', 'opp_win', 'opp_lossKO', 'oppavgtotallanded',
                    'Total punch accuracy', 'Avg Total punches landed', 'Avg Total punches landed against',
                    'Win KO', 'Loss KO', 'Win Other', 'KO ratio', 'Loss Other',
                    'KnockedOut ratio', 'Draw', 'last6']]
X = X.fillna(0)

Y = dataset['clean_outcome']
Y = Y.fillna(0)

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
x_train, x_, y_train, y_ = train_test_split(X, Y, test_size=0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

# Delete temporary
del x_, y_
