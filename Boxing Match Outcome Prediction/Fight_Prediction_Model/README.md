# How To Use:


(1) Download the Fight_Prediction_Model folder 

(2) Adjust the file path in data_processing/data_select_clean.py to dataset.csv

(3) Adjust Fighter1_name and Fighter2_name variables in MakePredictions.py and run

(4) To view the model evaluation run NN_ML_Model/model_evaluation.py


# How It Works:

It predicts the outcome of a boxing match by analyzing the historical data of two fighters. 
It intelligently combines the statistics of each fighter, considering one as the primary fighter and the other as the opposition, and vice versa, to generate a comprehensive profile for each possible matchup. 
A trained neural network is used to evaluate these matchups, predicting the likelihood of each fighter winning. 
It concludes by comparing the results of both scenarios, determining the most probable winner and their chance of victory. 
This approach allows for a nuanced prediction that takes into account the strengths and weaknesses of each fighter in relation to their opponent.
