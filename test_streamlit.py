import streamlit as st
import pandas as pd 
import joblib
from data_geter import *
#from stramlit_functions import *

def form_value() :
    data=get_dtypes()
    dict_element={}
    for i,j in data.items() :
        if j in ['object','int64'] :
            dict_element[i]=st.selectbox(i.capitalize(),get_unique(i))
        else :
            print(i)
            dict_element[i]=st.slider(i.capitalize(),float(get_min_max(i)[0]),float(get_min_max(i)[1]))
    return dict_element

st.title('CAR PRICE PREDICTION')

#Form construction
with st.form("my_form"):

   #st.write("Car Price Prediction Form")

   #generate each field from dataset columns
   data=form_value()

   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       data_form=pd.DataFrame.from_dict(data, orient='index').transpose()
       #df_form=pd.DataFrame.from_dict(data,orient='columns')
       st.write(data_form)

