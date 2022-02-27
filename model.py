#Importing required libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

@st.cache
#Function to create sample input data
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
#Function to read .csv file
def read_data(data):
    data = pd.read_csv(data)
    return data

#Function to check errors in input data
def check_data(test_data):
    dtype_lst = ['object', 'bool', 'datetime64']
    col_lst = []
    for column in test_data.columns:#check if dtype is float64 or int64
        if column == 'ID_code':
            continue
        elif test_data.dtypes[column] in dtype_lst:
            col_lst.append(column)
            
    if test_data.shape[1] != 201:#check if input data has same 201 columns
        st.error('Error: Input data mismatch. Expected 201 columns. Please refer sample input.')
        return 0

    elif 'ID_code' not in test_data.columns:#check if ID_code column is present
        st.error("Error: Input data mismatch. Column 'ID_code' does not exist. Please refer sample input.")
        return 0
    
    elif len(col_lst) !=0:
        st.error ('Error: Input data mismatch. Following column(s) are non int/float data type. Please refer sample input.')
        st.error(col_lst)
        return 0
    
    else: return 1

#Function to preprocess input data
def preprocessing(df):
    null_col = df.columns[df.isna().any()].tolist()
    if len(null_col):
        st.write('Following columns has null values and will be imputed with median value.')
        st.write(null_col)
    for column in df.columns:
        df[column].fillna(df[column].median(), inplace=True)#impute median values for missing data
        high = df[column].mean() + 3*df[column].std()
        low = df[column].mean() - 3*df[column].std()
        df[column] = np.where(df[column] > high, high,
                              np.where(df[column] < low, low, df[column]))#Capping data to lie within 3 standard deviation
    return df

@st.cache
#Function to create new features
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
#Function to predict on input data
def prediction(model, df):
    pred = model.predict(df)
    pred = pd.DataFrame(pred, columns=['Prediction'])
    return pred

@st.cache
#Function to export data
def export_df(df):
    return df.to_csv()

st.header('Santander Customer Transaction Prediction')
if st.checkbox('Show sample input'):
    st.write(sample_data().head())
    
data = st.file_uploader('Upload file to obtain predictions', type=['csv'])

if data is not None:
    st.success('Upload sucessful!')
    test_data = read_data(data)#Function call
    validated = check_data(test_data)#Function call
    
    if validated:
        if st.checkbox('Show uploaded file'):
            st.write(test_data)
        
        if st.button('Predict'):
            id_code = test_data['ID_code']
            df = test_data.drop(['ID_code'], axis=1)
                
            model_FE = joblib.load('LGBM_MulClassifier.pkl')#load model
            model_pred = joblib.load('LGBM_BiClassifierFE.pkl')#load model
        
            df = preprocessing(df)#Function call
            df_new = feature_engineering(model_FE, df)#Function call
            df_prediction = prediction(model_pred, df_new)#Function call
            df_prediction = pd.concat([id_code, df_prediction], axis=1)
            st.write(df_prediction)
            st.download_button('Download', export_df(df_prediction), key='download-csv')