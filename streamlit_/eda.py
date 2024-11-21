# eda.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import chi2_contingency
import gdown

class EDA:
    def __init__(self):
        self.df = self.load_data()
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_data():
        gdown.download('https://drive.google.com/uc?id=16ApwcdLGYhU3EjphBtDsE9rbIfFMRHTM',
                      'Telco.csv', quiet=False)
        return pd.read_csv("Telco.csv")

    def show_column_info(self):
        st.subheader('컬럼 정보')
        col_df = pd.DataFrame({
            'Index': range(len(self.df.columns)),
            'Column Name': self.df.columns,
            'Data Type': self.df.dtypes.values
        })
        st.dataframe(col_df)
        
        # 기본 데이터 정보
        st.write("### 기본 데이터 정보")
        st.write(f"행 개수: {self.df.shape[0]}, 컬럼 개수: {self.df.shape[1]}")

    def show_missing_values(self):
        st.subheader('결측치 분석')
        
        # 결측치 수 계산
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Values': [self.df[col].isna().sum() + (self.df[col] == ' ').sum() 
                             for col in self.df.columns],
            'Missing Percentage': [(self.df[col].isna().sum() + (self.df[col] == ' ').sum()) / len(self.df) * 100
                                 for col in self.df.columns]
        })
        missing_df['Missing Percentage'] = missing_df['Missing Percentage'].round(2)
        st.dataframe(missing_df)

        # Missing Values Matrix
        st.subheader('결측치 Matrix')
        plt.figure(figsize=(12, 6))
        msno.matrix(self.df)
        plt.title('결측치 Matrix', size=15, pad=20)
        st.pyplot(plt.gcf())
        plt.close()

    def show_correlation_analysis(self):
        st.subheader('상관관계 분석')
        
        # 상관관계 히트맵
        corr = self.df.apply(lambda x: pd.factorize(x)[0]).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(corr, mask=mask, 
                    xticklabels=corr.columns, 
                    yticklabels=corr.columns, 
                    annot=True, 
                    fmt='.2f',
                    linewidths=.5, 
                    cmap='RdBu_r', 
                    vmin=-1, 
                    vmax=1,
                    annot_kws={'size': 8})
        
        plt.title('변수들간 상관관계 Heatmap', 
                  pad=20, 
                  size=15, 
                  fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

        # 높은 상관관계 변수들의 Pairplot
        st.subheader('높은 상관관계 변수들간의 Pairplot')
        filtered_corr = corr[abs(corr) >= 0.75]
        high_corr_vars = filtered_corr.columns[(filtered_corr != 1).any(axis=0)]
        
        if len(high_corr_vars) > 0:
            filtered_df = self.df[high_corr_vars]
            pair_plot = sns.pairplot(filtered_df)
            plt.suptitle("Highly Correlated Features Pairplot", 
                        y=1.02, 
                        size=15, 
                        fontweight='bold')
            st.pyplot(pair_plot.fig)
            plt.close()
        else:
            st.write("상관관계가 0.75이상인 변수쌍을 찾을 수 없습니다.")

    def show_chi_square_test(self):
        st.subheader('카이제곱 검정')
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Progress bar 추가
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total_combinations = len(categorical_cols) * (len(categorical_cols) - 1) // 2
        current_progress = 0
        
        for i, col1 in enumerate(categorical_cols):
            for j, col2 in enumerate(categorical_cols[i+1:], i+1):
                contingency_table = pd.crosstab(self.df[col1], self.df[col2])
                chi2, p, dof, _ = chi2_contingency(contingency_table)
                
                results.append({
                    'Variable 1': col1,
                    'Variable 2': col2,
                    'Chi2 Statistic': chi2,
                    'P-value': p,
                    'Degrees of Freedom': dof,
                    'Independent': 'Yes' if p > 0.05 else 'No'
                })
                
                current_progress += 1
                progress = current_progress / total_combinations
                progress_bar.progress(progress)
                status_text.text(f'Processing... {current_progress}/{total_combinations} combinations')

        chi2_results = pd.DataFrame(results)
        st.dataframe(chi2_results)
        
        # 시각화
        plt.figure(figsize=(12, 8))
        independence_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)
        
        for result in results:
            independence_matrix.loc[result['Variable 1'], result['Variable 2']] = result['Independent']
            independence_matrix.loc[result['Variable 2'], result['Variable 1']] = result['Independent']
        
        sns.heatmap(independence_matrix.notna(), cmap='YlOrRd', cbar=False)
        plt.title('카이제곱 검정 결과', size=15, pad=20)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    def show_numerical_analysis(self):
        st.subheader('Numerical Features 분석')
        
        # TotalCharges를 숫자형으로 변환
        df_copy = self.df.copy()
        df_copy.replace({'TotalCharges': {' ': np.nan}}, inplace=True)
        df_copy['TotalCharges'] = df_copy['TotalCharges'].astype(float)
        
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        for col in numerical_cols:
            st.write(f"#### {col}")
            
            col_stats = df_copy[col].describe()
            st.write("통계량:")
            st.dataframe(col_stats)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histogram with KDE
            sns.histplot(data=df_copy, x=col, kde=True, ax=ax1)
            ax1.set_title(f'{col} Distribution', size=12)
            ax1.set_xlabel(col, size=10)
            ax1.set_ylabel('Count', size=10)
            
            # Boxplot
            sns.boxplot(data=df_copy, y=col, ax=ax2)
            ax2.set_title(f'{col} Boxplot', size=12)
            ax2.set_ylabel(col, size=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    def show_categorical_analysis(self):
        st.subheader('Categorical Features 분석')
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            st.write(f"#### {col}")
            
            # Value counts and percentages
            value_counts = self.df[col].value_counts()
            value_percentages = self.df[col].value_counts(normalize=True) * 100
            
            counts_df = pd.DataFrame({
                'Count': value_counts,
                'Percentage': value_percentages.round(2)
            })
            st.dataframe(counts_df)
            
            # Bar plot
            plt.figure(figsize=(12, 6))
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'{col} Distribution', size=15, pad=20)
            plt.xlabel(col, size=10)
            plt.ylabel('Count', size=10)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()

    def run(self):
        st.title('데이터 탐구 및 분석')
        
        # 사이드바에서 분석 항목 선택
        analysis_options = st.sidebar.multiselect(
            'Select Analysis Types',
            ['기본 정보', '결측치 분석', '상관관계 분석', 
             '카이제곱 검정', '숫자형 변수 분석', '범주형 변수 분석'],
            default=['기본 정보']
        )
        
        if '기본 정보' in analysis_options:
            self.show_column_info()
            
        if '결측치 분석' in analysis_options:
            self.show_missing_values()
            
        if '상관관계 분석' in analysis_options:
            self.show_correlation_analysis()
            
        if '카이제곱 검정' in analysis_options:
            self.show_chi_square_test()
            
        if '숫자형 변수 분석' in analysis_options:
            self.show_numerical_analysis()
            
        if '범주형 변수 분석' in analysis_options:
            self.show_categorical_analysis()
            
        # Show code
        with st.expander("Show Code"):
            with open(__file__, 'r', encoding='utf-8') as file:
                st.code(file.read())

if __name__ == "__main__":
    eda = EDA()
    eda.run()