#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import tensorflow as tf

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Paths
    input_path = os.path.join(script_dir, "data", "data_pca_transformed.csv")
    model_path = os.path.join(project_root, "f_Trained_Models", "Neural Network_man_best_model.h5")
    output_path = os.path.join(script_dir, "data", "predictions.csv")

    # 1. Load Data
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
    
    df = pd.read_csv(input_path)
    print(f"Loaded data with {len(df)} rows.")

    # 2. Prepare Features for NN
    # The NN expects these 13 features in a specific order:
    nn_features = [
        'home_Goalkeeper', 'home_Defenders', 'home_Midfielders', 'home_Attackers', 
        'away_Goalkeeper', 'away_Defenders', 'away_Midfielders', 'away_Attackers', 
        'attendance', 'home_club_position_before_game', 'away_club_position_before_game', 
        'home_form', 'away_form'
    ]
    
    missing_cols = [c for c in nn_features if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns for NN: {missing_cols}")
        return
        
    X = df[nn_features].fillna(0.0).values

    # 3. Load Neural Network
    if not os.path.exists(model_path):
        print(f"Error: NN model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 4. Predict
    # Predict probabilities
    probs = model.predict(X)
    
    # The model output is typically [Away, Draw, Home] based on alphabetical sort of labels 'Away', 'Draw', 'Home'
    # Class labels from training: Away (0), Draw (1), Home (2)
    classes = ['Away_Prob', 'Draw_Prob', 'Home_Prob']
    df_probs = pd.DataFrame(probs, columns=classes)
    
    # Add predicted class (highest probability)
    class_labels = ['Away', 'Draw', 'Home']
    df_probs['Prediction'] = [class_labels[i] for i in np.argmax(probs, axis=1)]

    # 5. Combine with metadata
    # Keep Date and Teams for identification
    df_final = pd.concat([df[['Date', 'Home Team', 'Away Team']], df_probs], axis=1)

    # 6. Save
    df_final.to_csv(output_path, index=False)
    print(f"Predictions complete. Saved to {output_path}")

if __name__ == "__main__":
    main()
