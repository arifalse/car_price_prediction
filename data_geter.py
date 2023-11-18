import pandas as pd
import os
from scipy.special import boxcox1p
import numpy as np
import requests
import shutil
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import time

#dataset origin
df=pd.read_csv('https://github.com/arifalse/car_price_prediction/raw/main/car_price_prediction.csv')
df=df[df.Levy!='-']
df['Mileage']=df['Mileage'].str.split(' ',expand=True)[0].astype('int64')

#dataset reference columns
df_ref=pd.read_csv('https://github.com/arifalse/bootcamp_datascience_files/raw/main/dataset_columns_modelling_2.csv')

def get_dtypes() :
    dict_result={}
    for i in df.columns[2:] :
        dict_result[i]=str(df.dtypes['Price'])
        if i in ['Levy','Mileage'] :
            dict_result[i]=str('float64')
    return dict_result

def get_unique(column) :
    
    if column=='Mileage':
        ls_unique=sorted([ i.split(' ')[0] for i in df['Mileage'].unique().tolist()])
    else :    
        ls_unique=sorted(df[column].unique().tolist())
    return ls_unique

def get_min_max(column):
    ls_minmax=[float(df[column].min()),float(df[column].max())]
    return ls_minmax

#### FUNCTIONS FOR DATA PREP
def skewness_check(df,columns,plot=True) :

  #####-- Skewness Score
  # 0 means : perfectly symmetrcolcal
  # -0.5 to 0.5 means :approxcolmated symmetrcolcal
  # -0.5 to -1 OR 0.5 to 1 means :moderately skewed
  # over -1 OR over 1 means : SCREWED

  dict_holder={}

  for col in columns :

    print(col)
    print(f"Skewness score : {df[col].skew()}")
    print(f"Kurtocols score : {df[col].kurt()}")

    if plot==True :
      sns.displot(df[col])
      continue

    dict_holder[col]=df[col].kurt()

  return dict_holder

def get_column_with_high_skewness(df) :

  #calculate skewness
  dict_col_skew_value=skewness_check(df,[i for i in df if df[i].dtype!='object'],plot=False)

  #filter it
  columns_with_high_skew=[i for i,j in dict_col_skew_value.items() if j >0.7]
  print(f"---- column wit skewness over 0.7 {columns_with_high_skew}----" )

  return columns_with_high_skew

#### FUNCTION TO CHECK ENCODE COLUMN IN DATAFRAME + SAVE ITS ENCODER
def dataframe_encoder(df,columns,os) :

  #dictionary to store encoder
  encoder_col = {}

  #loop to encode each column, then store it in list encoder_col
  for col in columns :
    if col == 'Doors' :

        if 'door_encoder.pkl' not in os.listdir():

            response = requests.get('https://github.com/arifalse/bootcamp_datascience_files/raw/main/door_encoder.pkl', stream=True)
            with open('door_encoder.pkl', 'wb') as fin1:
                    shutil.copyfileobj(response.raw, fin1)
            pkl_door = open('door_encoder.pkl', 'rb')
            le_door = pickle.load(pkl_door)
            fin1.close()
            #os.remove("door_encoder.pkl")

        else :
            
            pkl_door = open('door_encoder.pkl', 'rb')
            le_door = pickle.load(pkl_door)
            #os.remove("door_encoder.pkl")


        df[col]=le_door.transform(df[col].values.tolist())
    
    elif col=='Model' :
        
        if 'model_encoder.pkl' not in os.listdir() :
            response = requests.get('https://github.com/arifalse/bootcamp_datascience_files/raw/main/model_encoder.pkl', stream=True)
            with open('model_encoder.pkl', 'wb') as fin2:
                    shutil.copyfileobj(response.raw, fin2)
            pkl_model = open('model_encoder.pkl', 'rb')
            le_model = pickle.load(pkl_model)
            fin2.close()
            #os.remove("model_encoder.pkl")
        else :
            pkl_model = open('model_encoder.pkl', 'rb')
            le_model = pickle.load(pkl_model) 

        df[col]=le_model.transform(df[col].values.tolist())
        
  return df,encoder_col

def data_preparation(dform,os):

    # Tranform | Replace '-' in levy column to nan then covert  to float then fill it with mean
    dform['Levy']=dform.Levy.replace('-',np.NaN)
    dform['Levy']=dform['Levy'].astype('float64')
    #dform['Levy']=dform.Levy.replace(np.NaN,df['Levy'].mean())

    # Tranform | lower string value in object columns
    for i in dform.columns :
        if dform[i].dtype=='object' :
            try :
                dform[i]=dform[i].str.lower()
            except :
                pass

    #turbo egine feature
    dform['Engine volume']=dform['Engine volume'].astype(str)
    dform['turbo_engine']=0
    dform['Engine volume']=dform['Engine volume'].str.split(' ',expand=True)[0]
    dform['Engine volume']=[float(i) for i in dform['Engine volume']]

    #skewness tratmen
    columns_with_high_skew=get_column_with_high_skewness(dform)
    lam = 0.20
    for col in columns_with_high_skew:
        dform[col] = boxcox1p(dform[col], lam)

    #encoder
    dform,encoder_col=dataframe_encoder(dform,['Doors','Model'],os)

    #one hot encoding
    #dform.info()
    #print(dform.columns)
    dform = pd.get_dummies(dform, columns=[i for i in df.columns if df[i].dtype=='object'])
    #print(dform.columns)

    #create scaler
    scaler_x = RobustScaler()
    scaler_x = scaler_x.fit(dform)
    ddformf=pd.DataFrame(scaler_x.transform(dform))

    #add require column
    for i in df_ref.columns:
        if i not in dform.columns :
            dform[i]=0
    dform=dform[df_ref.columns]

    #predict
    print(dform)
    if 'random_forrest_regressor.pkl' not in os.listdir() :
        response = requests.get('https://github.com/arifalse/bootcamp_datascience_files/raw/main/random_forrest_regressor.pkl', stream=True)
        with open('random_forrest_regressor.pkl', 'wb') as modelio:
                shutil.copyfileobj(response.raw, modelio)
        
        pkl_modelregression = open('random_forrest_regressor.pkl', 'rb')
        model_regression = pickle.load(pkl_modelregression)
        modelio.close()
    else :
        model_regression = pickle.load(open('random_forrest_regressor.pkl', 'rb'))
     
    predicted_price=model_regression.predict(dform)
    val=predicted_price*10000
    return '$'+str(round(val[0],2))
