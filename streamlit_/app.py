# app.py
import streamlit as st
from eda import EDA
from preprocessing import Preprocessing
from ml_models import MLModels
from dl_models import DLModels

def main():
    st.title('Telco Customer Churn Analysis')
    
    # Sidebar navigation
    st.sidebar.title('목차')
    
    # Main category selection
    main_category = st.sidebar.selectbox(
        'Select Category',
        ['EDA', 'Preprocessing', 'ML', 'DL']
    )
    
    # Sub category selection based on main category
    if main_category == 'ML':
        sub_category = st.sidebar.selectbox(
            'Select ML Model',
            ['Logistic', 'gb', 'xgb', 'ensemble']
        )
    elif main_category == 'DL':
        sub_category = st.sidebar.selectbox(
            'Select DL Model',
            ['basic', 'batchnormal', 'dropout', 
             'cosineannealingwarmrestarts', 'CAWR+L2', 
             'stepLR', 'stepLR+SMOTE']
        )
    else:
        sub_category = None
    
    # Display appropriate page based on selection
    if main_category == 'EDA':
        eda = EDA()
        eda.run()
    elif main_category == 'Preprocessing':
        preprocessing = Preprocessing()
        preprocessing.run()
    elif main_category == 'ML':
        ml = MLModels()
        ml.run(sub_category)
    elif main_category == 'DL':
        dl = DLModels()
        dl.run(sub_category)

if __name__ == '__main__':
    main()