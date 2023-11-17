import pandas as pd
import os

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