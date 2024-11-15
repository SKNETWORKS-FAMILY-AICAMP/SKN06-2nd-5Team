# SKN06-2nd-5Team
## 2차 프로젝트 - Telco Customer Churn / 통신사 고객 이탈 예측

 - 기간 2024.11.13 ~ 2024.11.16

## 팀 소개
  ### 팀명 : Churn Busters

  ### 팀원 👥  

| <img width="60px" src="image/cy.jpg" /> | <img width="60px" src="image/ks.jpg" /> | <img width="60px" src="image/mh.jpg" /> |  <img width="60px" src="image/hj.jpg" /> |
|:----------:|:----------:|:----------:|:----------:|
| 강채연 | 유경상 | 박미현 | 홍 준 |
|[@codus090880](https://github.com/codus090880)|[@kyungsangYu](https://github.com/kyuyounglee)|[@ppim321](https://github.com/ppim321)|[@zl-zone](https://github.com/silenc3502)|
| DL | DL | ML | ML |  


## 프로젝트 개요

### 소개
- 통신사 고객 데이터를 활용하여 고객의 이탈(Churn)예측과 고객의 특성 요소들을 시각화
- 고객의 다양한 서비스 이용 패턴을 분석하고 고객의 이탈 예측을 위한 분류예측모델을 구축

### 필요성
- 고객 유지율 향상 : 이탈 위험이 높은 고객을 사전에 파악하여 적절한 대응 전략 수립
- 수익성 개선 : 고객 이탈로 인한 매출 손실 최소화
- 맞춤형 서비스 제공 : 고객 특성에 따른 개인화된 서비스 및 프로모션 설계
- 경영 의사결정 지원 : 데이터 기반의 객관적인 고객 관리 전략 수립

### 목표
- 고객 이탈을 정확히 예측할 수 있는 머신러닝 딥러닝 모델 개발 (목표 정확도: 80% 이상)
- 이탈에 영향을 미치는 주요 요인 파악 및 인사이트 도출
- 고객 세그먼트별 이탈 위험도 분석 및 시각화
- 고객 유지를 위한 실행 가능한 전략 제안  
 이 프로젝트를 통해 팀원들은 실제 비즈니스에서 직면할 수 있는 데이터 분석을 통해
간접적으로 실무 경험을 쌓고, 고객 행동 예측 모델 개발 역량을 키울 수 있을 것으로 기대합니다.  
또한, 통신 산업의 고객 관리 전략에 대한 이해를 높이고 실무적인 문제 해결 능력을 향상시킬 수 있을 것입니다.

  
## 과정  

### 활용 데이터
- 프로젝트에 사용할 데이터는 통신사의 고객 정보와 이용중인 서비스 형태를 포함.

<br>출처 : Kaggle
<br>dtypes: float64(1), int64(2), object(18)
<br>RangeIndex: 7043 entries, 0 to 7042
<br>Data columns : total 21 columns  

| # | Column | Non-Null | Count | Dtype |
|:---:|:---:|:---:|:---:|:---:|
| 0 | customerID    | 7043 |non-null  |   object |
| 1  | gender       | 7043 |non-null  |   object |
| 2  | SeniorCitizen| 7043 |non-null  |   int64 | 
| 3  | Partner      | 7043 |non-null  |   object |
| 4  | Dependents   | 7043 |non-null  |  object |
| 5  | tenure          |7043 |non-null  |  int64  |
| 6  | PhoneService    | 7043 |non-null  |   object |
| 7  | MultipleLines    | 7043 |non-null |  object |
| 8  | InternetService   | 7043 |non-null  |   object |
| 9  | OnlineSecurity      | 7043 |non-null  |   object |
| 10  | OnlineBackup        | 7043 |non-null  |   object |
| 11  | DeviceProtection    | 7043 |non-null  |   object | 
| 12  | TechSupport         | 7043 |non-null  |   object |
| 13  | StreamingTV         | 7043 |non-null  |   object |
| 14  | StreamingMovies     | 7043 |non-null  |   object |
| 15  | Contract            | 7043 |non-null  |   object |
| 16  | PaperlessBilling    | 7043 |non-null  |   object |
| 17  | PaymentMethod       | 7043 |non-null  |   object |
| 18  | MonthlyCharges      | 7043 | non-null  |   float64 |
| 19  | TotalCharges        | 7043 | non-null  |   object |
| 20  | Churn               | 7043 | non-null  |   object |


### 데이터 컬럼 정보 확인  
- 데이터 분석을 위해 컬럼의 정보 확인
- 학습에 참여할 변수나 수치형 데이터 확인


### 데이터 전처리
#### 1. 결측치 처리
사용된 데이터에 존재한 결측치가 유일하게 하나 있는 것을 발견하였지만, 데이터를 처리하는 과정에서 해결이 되었기 때문에 따로 처리하지 않았다.
무시한 채, 인코딩 실행 후 데이터가 채워진 것을 확인.

#### 2. 범주형 데이터 처리

#### 3. 수치형 데이터 처리

#### 4. 인코딩


###  모델

#### ML
1. 교차 검증 및 모델 학습 수행


   - StratifiedKFold
   - Cross-Validation
   - pipeline
   - 결과


## CrossValid   
<img width="400px" src="image/ML_crossval_ROC.png" />

<img width="400px" src="image/ML_crossval_report.png" />   




## SMOTE
<img width="400px" src="image/ML_SMOTE_report.png" /> <img width="400px" src="image/ML_SMOTE_ROC.png" />  




## Hyper Parameter RF
<img width="400px" src="image/ML_HyperPRM_RF_report.png" /> <img width="400px" src="image/ML_HyperPRM_RF_ROC.png" />  




## Log Regression
<img width="400px" src="image/ML_LogR_report.png" />  



  
## Low Importance feature drop
<img width="400px" src="image/ML_LowIFdel_RF_report.png" />  



  
## RF
<img width="400px" src="image/ML_RF_report.png" />  



  
## XGB
<img width="400px" src="image/ML_XGB_report.png" />  



  
## Final
<img width="400px" src="image/ML_fianl2.png" />  



  
<img width="400px" src="image/ML_final1.png" />    




  
## 앙상블  
<img width="400px" src="image/ML_EMSBL_report.png" />    
 


 
2. 모델 선택
   - 랜덤포레스트(RandomForest)
   - XGBoost
   - Ensemble (RandomForest + XGBoost)  
     
- 앙상블 (RF + XGB)  
<img width="400px" src="image/ML_ESBL(RF_XGB)_report.png" />  



 3. 모델 재선택 및 결정
    - 로지스틱 회귀(Logistic Resgression)
    - Ensemble (Logistic + Gradient + XGBoost)  

- 앙상블 (L + Gradient + XGB)  
<img width="400px" src="image/ML_ESBL(LG_G_XGB)_report.png" />  


    
#### DL
1. 
2. 
3. 


## 📚 Stacks 

### Environment
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-007ACC?style=for-the-badge&logo=Visual%20Studio%20Code&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white)
![Github](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white)             

### Development
![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white) 
![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### Communication
![Discord](https://img.shields.io/badge/discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)


### 결과 요약:


### 결론:


## 한 줄 회고
