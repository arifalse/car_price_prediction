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
with st.form("my_form"):
   st.write("Car Price Prediction Form")

   form_value()

   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       st.write("succes")






"""with st.form("my_form"):
   st.write("Car Price Prediction Form")

   slider_val = st.slider("Form slider")
   checkbox_val = st.checkbox("Form checkbox")

   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       st.write("slider", slider_val, "checkbox", checkbox_val)
"""
"""get_dtypes()

dtypes=get_dtypes()
subheader_dict={}
element_dict={}
for i in dtypes :
    title=st.title(i)
    st.selectbox
"""
#st.title('CAR PRICE PREDICTION')

#st.subheader('Menu Page')
#('yeet')

