# preprocessing.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Preprocessing:
    def __init__(self):
        self.df = pd.read_csv('Telco.csv')
        
    def show_initial_info(self):
        st.write("### 데이터 기본 정보")
        st.write(f"데이터 크기: {self.df.shape}")
        
        st.write("### 결측치 정보")
        missing_data = {
            'Column': [],
            'Missing Values': []
        }
        for col in self.df.select_dtypes(include=['object', 'int', 'float']).columns:
            has_blank_or_nan = self.df[col].isna().sum() + (self.df[col] == ' ').sum()
            missing_data['Column'].append(col)
            missing_data['Missing Values'].append(has_blank_or_nan)
        
        st.table(pd.DataFrame(missing_data))
        
        st.write("### 데이터 타입 정보")
        st.write(self.df.dtypes)

    def process_total_charges(self):
        st.write("### TotalCharges 처리")
        
        # Before processing
        st.write("처리 전 TotalCharges 타입:", self.df['TotalCharges'].dtype)
        
        # Convert to numeric
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        
        # Fill missing values
        median_value = self.df['TotalCharges'].median()
        self.df['TotalCharges'] = self.df['TotalCharges'].fillna(median_value)
        
        # After processing
        st.write("처리 후 TotalCharges 타입:", self.df['TotalCharges'].dtype)
        st.write("처리 후 결측치 수:", self.df['TotalCharges'].isna().sum())

    def handle_outliers(self, whis=1.5):
        st.write("### 이상치 처리")
        
        def find_outliers_and_replace(df, whis=1.5):
            df_copy = df.copy()
            numeric_columns = df_copy.select_dtypes(include='number').columns
            
            for column_name in numeric_columns:
                # Check if binary
                unique_values = df_copy[column_name].nunique()
                if unique_values == 2:
                    continue
                
                q1, q3 = df_copy[column_name].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - iqr * whis
                upper_bound = q3 + iqr * whis
                
                # Show outliers info
                outliers = df_copy[
                    (df_copy[column_name] < lower_bound) | 
                    (df_copy[column_name] > upper_bound)
                ].shape[0]
                
                st.write(f"{column_name}의 이상치 수: {outliers}")
                
                # Replace outliers
                df_copy[column_name] = df_copy[column_name].apply(
                    lambda x: lower_bound if x < lower_bound else (
                        upper_bound if x > upper_bound else x
                    )
                )
            
            return df_copy
        
        self.df = find_outliers_and_replace(self.df, whis)

    def encode_features(self):
        st.write("### 피처 인코딩")
        
        # Label Encoding for binary variables
        st.write("#### Label Encoding (이진 변수)")
        label_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                     'PaperlessBilling', 'Churn']
        label_encoder = LabelEncoder()
        
        for col in label_cols:
            self.df[col] = label_encoder.fit_transform(self.df[col])
            st.write(f"{col} 인코딩 완료")
        
        # One-Hot Encoding for categorical variables
        st.write("#### One-Hot Encoding (다중 범주 변수)")
        onehot_cols = ['InternetService', 'Contract', 'PaymentMethod', 
                      'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                      'DeviceProtection', 'TechSupport', 'StreamingTV', 
                      'StreamingMovies']
        
        self.df = pd.get_dummies(self.df, columns=onehot_cols)
        st.write("One-Hot Encoding 완료")
        
        # Show final columns
        st.write("### 최종 컬럼 정보")
        for idx, col in enumerate(self.df.columns):
            st.write(f"Index: {idx}, Column Name: {col}")

    def save_data(self):
        output_file = "train.csv"
        self.df.to_csv(output_file, index=False)
        st.success(f"전처리된 데이터프레임이 '{output_file}'에 저장되었습니다.")

    def run(self):
        st.header('Data Preprocessing')
        
        # 전처리 단계 선택
        preprocessing_steps = st.multiselect(
            '전처리 단계 선택',
            ['기본 정보 확인', 'TotalCharges 처리', '이상치 처리', '피처 인코딩'],
            default=['기본 정보 확인']
        )
        
        if '기본 정보 확인' in preprocessing_steps:
            self.show_initial_info()
            
        if 'TotalCharges 처리' in preprocessing_steps:
            self.process_total_charges()
            
        if '이상치 처리' in preprocessing_steps:
            self.handle_outliers()
            
        if '피처 인코딩' in preprocessing_steps:
            self.encode_features()
            
        # Save data button
        if st.button('전처리 데이터 저장'):
            self.save_data()
        
        # Show original code
        with st.expander("Show Original Code"):
            with open(__file__, 'r', encoding='utf-8') as file:
                st.code(file.read())

if __name__ == "__main__":
    preprocessing = Preprocessing()
    preprocessing.run()