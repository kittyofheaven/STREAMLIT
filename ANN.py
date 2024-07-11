# import streamlit as st
# import pandas as pd
# import numpy as np
# from joblib import load
# import tensorflow
# from tensorflow.keras.models import load_model



# # Function to load scaler and predict stress level
# def predict_stress(features):
#     # Load scaler
#     scaler_filename = 'standardScalerAnova8Features.bin'
#     scaler = load(scaler_filename)
    
#     # Normalize features
#     features_df = pd.DataFrame([features], columns=[
#         'pNN50', 'LF/HF_Ratio', 'Mean_HR', 'MPF Fp1',
#         'Mean_RR', 'LBPFp1', 'SD1', 'SD2'
#     ])
#     features_scaled = scaler.transform(features_df)
    
#     # Load pretrained model
#     pretrained_model_path = 'C:\Users\RARA\OneDrive - Institut Teknologi Sepuluh Nopember\Dokumen\#1 TUGAS AKHIR RARA\02. TUGAS AKHIR\4. HASIL\4. STREAMLIT\STREAMLIT BISA\Anova8020Rank8.h5'
#     model = load_model(pretrained_model_path)
    
#     # Predict with the model
#     predictions = model.predict(features_scaled)
    
#     # Determine predicted class
#     predicted_class_index = np.argmax(predictions)
#     return predicted_class_index