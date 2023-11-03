import pickle

import numpy as np
import streamlit as st

with open('F:\\class\\Machine Learning\\project\\House Price prediction\\df3.pkl','rb') as df_files:
    df = pickle.load(df_files)

with open('F:\\class\\Machine Learning\\project\\House Price prediction\\pipe.pkl','rb') as model_files:
    pipe = pickle.load(model_files)
    
st.title("Model of Houe price prediction")

location = st.selectbox('Selec Location',df['location'].unique())

Size = st.selectbox('Select Bedrom size',df['size'].unique())

total_sqft = st.number_input('Enter Total Sqft')

bath = st.number_input("Enter Number Of Bathroom")

balcony = st.number_input("Enter Balcony Value")

if st.button("Prediction Price"):
    query = np.array([location,Size,total_sqft,bath,balcony])
    query = query.reshape(1,5)
    st.title("The prediction price of this house is" + str(int(np.exp(pipe.predict(query)[0]))))
    