import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

@st.cache
def sample_data():
    col = ['ID_code']
    for i in range(200):
        col.append('var_'+ str(i))
    id_code = np.array(['test_'+ str(j) for j in range(5)])
    val = np.around(np.random.random((5,200)), decimals=2)
    samp_data = np.concatenate((id_code.reshape(-1,1), val), axis=1)
    samp_data = pd.DataFrame(samp_data, columns=col)
    return samp_data

@st.cache
def read_data(data):
    data = pd.read_csv(data)
    return data

@st.cache
def preprocessing(df):
    for column in df.columns:
        df[column].fillna(df[column].median(), inplace=True)
    return df

@st.cache
def feature_engineering(model, df):
    #Predicting 4 class probabilities using model_2_ and adding to train set
    predict_prob = model.predict_proba(df)
    predict_prob_df = pd.DataFrame(predict_prob, columns=['C_high', 'C_low', 'W_high', 'W_low'])
    df_pred = pd.concat([df, predict_prob_df], axis=1)
    
    #Standardizing and transforming the above dataframe using PCA to obtain top 5 components 
    df_pred_std = StandardScaler().fit_transform(df_pred)
    pca = PCA(n_components = 5)
    pca_comp = pca.fit_transform(df_pred_std)
    df_pca = pd.DataFrame(data=pca_comp, columns=("Component_1", "Component_2", "Component_3", "Component_4", "Component_5",))
    
    #Adding the top 5 components from PCA to new train set
    df_new = pd.concat([df_pred, df_pca], axis=1)
    
    return df_new

@st.cache    
def prediction(model, df):
    pred = model.predict(df)
    pred = pd.DataFrame(pred, columns=['Prediction'])
    return pred

@st.cache
def export_df(df):
    return df.to_csv()

st.header('Santander Customer Transaction Prediction')
if st.checkbox('Show sample input'):
    st.write(sample_data().head())
    
data = st.file_uploader('Upload file to obtain predictions', type=['csv'])

if data is not None:
    test_data = read_data(data)
    id_code = test_data['ID_code']
    df = test_data.drop(['ID_code'], axis=1)
        
    model_FE = joblib.load('LGBM_MulClassifier.pkl')
    model_pred = joblib.load('LGBM_BiClassifierFE.pkl')
        
    if st.checkbox('Show uploaded file'):
        st.write(test_data)
            
    if st.button('Predict'):
        df = preprocessing(df)
        df_new = feature_engineering(model_FE, df)
        df_prediction = prediction(model_pred, df_new)
        df_prediction = pd.concat([id_code, df_prediction], axis=1)
        st.write(df_prediction)
        st.download_button('Download', export_df(df_prediction), key='download-csv')