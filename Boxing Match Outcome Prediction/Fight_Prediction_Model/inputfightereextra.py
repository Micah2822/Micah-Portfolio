import pandas as pd
from Fight_Prediction_Model.NN_ML_Model.nn_model import model
from Fight_Prediction_Model.data_processing.data_select_clean import final_data, dataset
import tensorflow as tf
import numpy as np

dataset['sex'] = dataset['sex'].map({'male': 1, 'female': 0})
dataset['clean_outcome'] = dataset['clean_outcome'].map({'win': 1, 'loss': 0})
new_data = dataset.dropna()


print("Please input fighters first and last name with first letter capatalised")

fighter = input("Input your chosen fighter: ")
opposition = input("Input their opposition: ")

def get_fighter_stats(encoded_name, data):
    # Retrieve stats for the fighter based on the one-hot encoded column
    fighter_data = data[data[encoded_name] == 1]
    numeric_data = fighter_data.select_dtypes(include=[np.number])
    return numeric_data.mean()

def predict_winner(name1, name2, trained_model, data):
    # Create one-hot encoded column names
    encoded_name1 = 'name_' + name1
    encoded_name2 = 'name_' + name2

    # Check if columns exist
    if encoded_name1 not in data.columns or encoded_name2 not in data.columns:
        raise ValueError("One or both fighters not found in the dataset")

    # Get fighter stats
    fighter1_stats = get_fighter_stats(encoded_name1, data)
    fighter2_stats = get_fighter_stats(encoded_name2, data)

    # Prepare the input vector
    matchup_features = pd.concat([fighter1_stats, fighter2_stats])
    matchup_features = matchup_features.astype('float32').values.reshape(1, -1)

    # Predict the outcome
    prediction = trained_model.predict(matchup_features)

    # Interpret the result
    winner = name1 if prediction >= 0.5 else name2
    return winner

# Example usage
winner = predict_winner(fighter, opposition, model, new_data)
print("Predicted winner:", winner)





import pandas as pd
from Fight_Prediction_Model.NN_ML_Model.nn_model import model
from Fight_Prediction_Model.data_processing.data_select_clean import final_data, dataset
import tensorflow as tf

dataset['sex'] = dataset['sex'].map({'male': 1, 'female': 0})
dataset['clean_outcome'] = dataset['clean_outcome'].map({'win': 1, 'loss': 0})
new_data = dataset.dropna()

print("Please input fighters first and last name with first letter capatalised")

fighter = input("Input your chosen fighter: ")
opposition = input("Input their opposition: ")

def extract_features(fighter_data):
    # If the data contains only one row per fighter
    if len(fighter_data) == 1:
        features = fighter_data.iloc[0]
    else:
        # If the data contains multiple rows, aggregate the data (e.g., by taking the mean)
        features = fighter_data.mean()

    return features

def get_fighter_stats(name, data):
    # Filter the dataset for the fighter - catch error if fighter not found
    fighter_data = data[data['name'] == name]

    # Extract features
    stats = extract_features(fighter_data)
    return stats

def combine_stats(stats1, stats2):
    # Simplest approach is to concatenate the stats
    # Ensure the order of stats is the same as was used for training the model
    combined = pd.concat([stats1, stats2], axis=0)
    return combined


def predict_winner(name1, name2, trained_model, data):
    # Retrieve fighter stats from the dataset
    fighter1_stats = get_fighter_stats(name1, data)
    fighter2_stats = get_fighter_stats(name2, data)

    # Combine stats into a single feature vector  - ie get the features from dataset instead of typing it out
    matchup_features = combine_stats(fighter1_stats, fighter2_stats)

    matchup_features = tf.convert_to_tensor(matchup_features, dtype=tf.float32)

    # Predict the outcome
    prediction = trained_model.predict([matchup_features])

    # Interpret the result
    winner = name1 if prediction >= 0.5 else name2
    return winner


# Example usage
winner = predict_winner(fighter, opposition, model, new_data)
print("Predicted winner:", winner)