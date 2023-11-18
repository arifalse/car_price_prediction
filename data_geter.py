import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import boxcox1p

url='https://github.com/arifalse/car_price_prediction/raw/main/car_price_prediction.csv'

df=pd.read_csv(r'C:\Users\arif\Downloads\car_price_prediction.csv')
df=df[df.Levy!='-']
df['Mileage']=df['Mileage'].str.split(' ',expand=True)[0].astype('int64')

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
  dict_col_skew_value=skewness_check(df,[i for i in df.drop('Price',axis=1) if df[i].dtype!='object'],plot=False)

  #filter it
  columns_with_high_skew=[i for i,j in dict_col_skew_value.items() if j >0.7]
  print(f"---- column wit skewness over 0.7 {columns_with_high_skew}----" )

  return columns_with_high_skew

#### FUNCTION TO CHECK ENCODE COLUMN IN DATAFRAME + SAVE ITS ENCODER
def dataframe_encoder(df,columns) :

  #dictionary to store encoder
  encoder_col = {}

  #loop to encode each column, then store it in list encoder_col
  for col in columns :
    lbl=LabelEncoder()
    lbl.fit(df[col].values.tolist())
    df[col]=lbl.transform(df[col].values.tolist())

    #save it to dictionary
    encoder_col[col]=lbl
    print(f"encoder for column {col} saved in encoder_col")

  return df,encoder_col

def data_preparation(dataset):

    #drop duplicates by id
    dataset.drop_duplicates('ID',inplace=True)

    # Tranform | Replace '-' in levy column to nan then covert  to float then fill it with mean
    dataset['Levy']=dataset.Levy.replace('-',np.NaN)
    dataset['Levy']=dataset['Levy'].astype('float64')
    dataset['Levy']=dataset.Levy.replace(np.NaN,dataset['Levy'].mean())

    # Tranform | lower string value in object columns
    for i in dataset.columns :
        if dataset[i].dtype=='object' :
            dataset[i]=dataset[i].str.lower()

    #removing ID columns
    dataset.drop('ID',axis=1,inplace=True)

    #change mileage to float
    dataset['Mileage']=[i.split(' ')[0] for i in dataset.Mileage]
    dataset['Mileage']=dataset['Mileage'].astype('float64')
    dataset[['Mileage']].head(2)

    #turbo egine feature
    dataset['turbo_engine']=dataset['Engine volume'].str.split(' ',expand=True)[1]
    dataset['Engine volume']=dataset['Engine volume'].str.split(' ',expand=True)[0]
    dataset['Engine volume']=[float(i) for i in dataset['Engine volume']]

    #skewness tratmen
    columns_with_high_skew=get_column_with_high_skewness(dataset)
    lam = 0.20
    for col in columns_with_high_skew:
        dataset[col] = boxcox1p(dataset[col], lam)

    #encoder
    testdf,encoder_col=dataframe_encoder(testdf,['Doors','Model'])



dataset
"""def main () :
    st.title('CAR PRICE PREDICTION')
    menu=['Menu','ML Page']
    choice=st.sidebar.selectbox('AKU',menu)

    if choice == 'Menu' :
        st.subheader('Menu Page')
        education = st.selectbox('Education', ["Below Secondary", "Bachelor's", "Master's & above"])
        Manufacture = st.selectbox('Manufacturer', ['xxxx'])
        st.radio('Gender', ['m','f'])
    elif choice == 'ML Page' :
        st.subheader('ML Page')
if __name__ == '__main__' :
    main()
""" 