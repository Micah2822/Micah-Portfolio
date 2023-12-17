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

# Evaluation:

My boxing match outcome prediction model demonstrates commendable performance with the following key metrics:

**Accuracy: 85.48%:** This indicates the proportion of total predictions (win/loss) our model got right. An accuracy over 85% is quite robust for predictive modeling, especially in complex scenarios like boxing matches where many factors can influence the outcome.

**Precision: 84.38%:** This metric reflects the accuracy of positive predictions. In other words, when our model predicts a fighter will win, it is correct approximately 84.38% of the time.

**Recall:** 95.43%: This shows the model’s ability to correctly identify actual wins. A recall of over 90% means the model is highly effective in capturing most of the true winning cases.

**F1 Score: 89.56%:** The F1 Score is a balance between precision and recall, providing a holistic view of the model’s accuracy and robustness. An F1 Score closer to 90% is indicative of the model’s balanced performance in precision and recall.

**AUC-ROC:** 81.11%: The Area Under the Receiver Operating Characteristic Curve represents the model’s ability to distinguish between the classes (win and loss). An AUC-ROC score of over 80% is a strong indicator of the model’s discriminatory power.

In my efforts to optimize the model, I experimented with L1 and L2 regularization terms ranging from 0.01 to 0.001. Regularization helps prevent overfitting by penalizing larger weights in the model. However, experimentation with these regularization terms resulted in a slightly lower accuracy of around 83%. This suggests that the regularization terms might have been too restrictive, leading to underfitting where the model was unable to capture the complexity of the data adequately. The fine-tuning of regularization parameters is a delicate balance – too high a penalty can oversimplify the model (underfitting), while too low a penalty might not effectively prevent overfitting. The current model performance metrics suggest an optimal balance in model complexity and generalization ability without excessive regularization.

# Results:

The matchup between Terence Crawford vs Errol Spence Jr, which actually took place on 29th July 2023, was regarded as almost a 50/50 matchup with the odds slightly in favour of Terence Crawford. When inputted into the model, it predicted Terence Crawford as the winner, aligning with the actual match result. The model’s probability scores indicated a competitive match (34.09% for Crawford vs. 24.05% for Spence Jr.), slightly favoring Crawford, reflecting the real-life scenario where the match was considered evenly matched (50/50), and where Terence Crawford won by a Ninth round TKO.

Although this is a simple model, its prediction of Crawford as the winner in what was considered a balanced and unpredictable fight demonstrates its effectiveness in analyzing and interpreting complex sports data.

