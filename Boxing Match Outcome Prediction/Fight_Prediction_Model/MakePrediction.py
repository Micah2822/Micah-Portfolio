import pandas as pd
from Fight_Prediction_Model.NN_ML_Model.nn_model import model
from Fight_Prediction_Model.data_processing.data_select_clean import dataset
import tensorflow as tf


# Input Names Here
fighter1_name = "Terence Crawford"
fighter2_name = "Errol Spence Jr"


# Define the columns used as features and their corresponding 'opposition' versions
feature_columns = ['opp_last6', 'opp_KnockedOut ratio', 'opp_winOther',  'opp_winKO', 'opptotalaccuracy', 'oppavgtotalagainst',
                    'opp_KO ratio', 'opp_loss', 'opp_lossOther', 'opp_win', 'opp_lossKO', 'oppavgtotallanded',
                    'Total punch accuracy', 'Avg Total punches landed', 'Avg Total punches landed against',
                    'Win KO', 'Loss KO', 'Win Other', 'KO ratio', 'Loss Other',
                    'KnockedOut ratio', 'Draw', 'last6']

opposition_feature_mapping = {
    'opp_winKO': 'Win KO',
    'opp_winOther': 'Win Other',
    'opp_lossKO': 'Loss KO',
    'opp_lossOther': 'Loss Other',
    'opp_last6': 'last6',
    'opp_KO ratio': 'KO ratio',
    'opp_KnockedOut ratio': 'KnockedOut ratio',
    'opptotalaccuracy': 'Total punch accuracy',
    'oppavgtotalagainst': 'Avg Total punches landed against',
    'oppavgtotallanded': 'Avg Total punches landed'
}

# Function to create a new row for prediction
def create_prediction_row(fighter1, fighter2, dataset, feature_columns, opposition_feature_mapping):
    if fighter1 not in dataset['name'].values:
        raise ValueError(f"Fighter {fighter1} not found in dataset.")
    if fighter2 not in dataset['name'].values:
        raise ValueError(f"Fighter {fighter2} not found in dataset.")

    # Find the rows for fighter1 and fighter2
    row_fighter1 = dataset[dataset['name'] == fighter1].iloc[0]
    row_fighter2 = dataset[dataset['name'] == fighter2].iloc[0]

    # Create a new row with the structure of the dataset
    new_row = pd.DataFrame(columns=dataset.columns)
    new_row.loc[0, 'name'] = fighter1
    new_row.loc[0, 'opposition'] = fighter2

    # Populate the new row with features from both fighters
    for feature in feature_columns:
        if feature in opposition_feature_mapping:  # If it's an opposition feature
            mapped_feature = opposition_feature_mapping[feature]
            new_row.loc[0, feature] = row_fighter2[mapped_feature]
        else:  # If it's a regular feature
            new_row.loc[0, feature] = row_fighter1[feature]

    return new_row[feature_columns]

# Prediction function
def predict_outcome(fighter1, fighter2):
    # Create a new row for prediction
    prediction_row_f1_vs_f2 = create_prediction_row(fighter1, fighter2, dataset, feature_columns, opposition_feature_mapping)
    prediction_row_f2_vs_f1 = create_prediction_row(fighter2, fighter1, dataset, feature_columns, opposition_feature_mapping)

    prediction_features_f1_vs_f2 = prediction_row_f1_vs_f2.values.reshape(1, -1)
    prediction_features_f1_vs_f2 = tf.convert_to_tensor(prediction_features_f1_vs_f2, dtype=tf.float32)
    prediction_f1_vs_f2 = model.predict(prediction_features_f1_vs_f2)

    prediction_features_f2_vs_f1 = prediction_row_f2_vs_f1.values.reshape(1, -1)
    prediction_features_f2_vs_f1 = tf.convert_to_tensor(prediction_features_f2_vs_f1, dtype=tf.float32)
    prediction_f2_vs_f1 = model.predict(prediction_features_f2_vs_f1)

    # Determine the most likely winner
    if prediction_f1_vs_f2[0] > prediction_f2_vs_f1[0]:
        winner = fighter1
        looser = fighter2
        winning_chance = prediction_f1_vs_f2[0]
        looser_chance = prediction_f2_vs_f1[0]
    else:
        winner = fighter2
        looser = fighter1
        winning_chance = prediction_f2_vs_f1[0]
        looser_chance = prediction_f1_vs_f2[0]

    return winner, looser, winning_chance, looser_chance

try:
    winner, looser, winning_chance, looser_chance = predict_outcome(fighter1_name, fighter2_name)
    print(f"Predicted Winner: {winner} with a {winning_chance * 100}% chance to win")
    print(f"Predicted Looser: {looser} with a {looser_chance * 100}% chance to win")
except ValueError as e:
    print(e)
