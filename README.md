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
| DB | DB | Streamlit | DB, Streamlit |

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
- 고객 이탈을 정확히 예측할 수 있는 머신러닝 딥러닝 모델 개발 (목표 정확도: 85% 이상)
- 이탈에 영향을 미치는 주요 요인 파악 및 인사이트 도출
- 고객 세그먼트별 이탈 위험도 분석 및 시각화
- 고객 유지를 위한 실행 가능한 전략 제안  
 이 프로젝트를 통해 팀원들은 실제 비즈니스에서 직면할 수 있는 데이터 분석을 통해
간접적으로 실무 경험을 쌓고, 고객 행동 예측 모델 개발 역량을 키울 수 있을 것으로 기대합니다.  
또한, 통신 산업의 고객 관리 전략에 대한 이해를 높이고 실무적인 문제 해결 능력을 향상시킬 수 있을 것입니다.

  
## 과정  

#### 활용 데이터
- 프로젝트에 사용할 데이터는 통신사의 고객 개인정보와 이용중인 서비스 형태를 포함.
  
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
| 7  | MultipleLines    | 7043 |non-null   object |
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
  
| cutomerID | object | 고객 식별자(ID)  ->   각 고객을 고유하게 구분할 수 있는 코드 |
| :---: | :---: | :--- |
| gender | object | 고객의 성별 -> "Male" 또는 "Female"로 표기 |
| SeniorCitizen | int64 | 고령자 여부 -> 1 : 고령자, 0 : 고령자가 아닌 고객 |
| Partner | object | 배우자 유무 ->  "Yes" : 배우자 있음, "No" : 배우자 없음 |
| Dependents | object | 부양가족 유무 ->  "Yes" : 부양가족 있음, "No" : 부양가족 없음 |
| tenure | int64 | 고객이 통신사와 계약을 유지한 개월 수 |
| PhoneService | object | 전화 서비스 가입 여부 ->  "Yes" : 서비스 이용, "No" : 서비스 이용 안함 |
| MultipleLines | object | 다중 회선 여부 ->  "Yes" : 다중 회선 사용, "No" : 단일 회선 사용 "No phone service"일 경우 전화 서비스를 이용하지 않는 경우 |
| InternetService | object | 인터넷 서비스 유형 ->  "DSL", "Fiber optic", 또는 "No"로 표시, 각기 다른 인터넷 서비스 종류를 나타냄 |
| OnlineSecurity | object | 온라인 보안 서비스 가입 여부 ->  "Yes" : 가입함, "No" : 가입하지 않음 "No internet service"일 경우 인터넷 서비스를 이용하지 않는 경우 |
| OnlineBackup | object | 온라인 백업 서비스 가입 여부 ->  "Yes" 또는 "No"로 나타내며, "No internet service"는 인터넷 서비스를 이용하지 않는 경우 |
| DeviceProtection | object | 기기 보호 서비스 가입 여부 ->  "Yes" 또는 "No"로 나타내며, "No internet service"는 인터넷 서비스를 이용하지 않는 경우 |
| TechSupport | object | 기술 지원 서비스 가입 여부 ->  "Yes" 또는 "No"로 나타내며, "No internet service"는 인터넷 서비스를 이용하지 않는 경우 |
| StreamingTV | object | TV 스트리밍 서비스 가입 여부 ->  "Yes" 또는 "No"로 나타내며, "No internet service"는 인터넷 서비스를 이용하지 않는 경우 |
| StreamingMovies | object | 영화 스트리밍 서비스 가입 여부 ->  "Yes" 또는 "No"로 나타내며, "No internet service"는 인터넷 서비스를 이용하지 않는 경우 |
| Contract | object | 계약 유형으 ->  "Month-to-month" (월별 계약), "One year" (1년 계약), "Two year" (2년 계약) 중 하나 |
| PaperlessBilling | object | 종이 없는 청구서 여부 ->  "Yes" : 종이 청구서 없이 온라인 청구서 사용, "No" : 종이 청구서를 사용 |
| PaymentMethod | object | 요금 납부 방식으 ->  "Electronic check" (전자 수표), "Mailed check" (우편 수표), "Bank transfer (automatic)" (자동 은행 이체), "Credit card (automatic)" (자동 신용카드 결제) 중 하나 |
| MonthlyCharges | float64 | 고객의 월간 요금 ->  매달 청구되는 금액 |
| TotalCharges | object | 고객이 전체 기간(가입기간)동안 청구된 총 금액. 월 요금과 유지 기간을 기반으로 산출 |
| Churn | object | 고객 이탈 여부 ->  "Yes" : 이탈한 고객, "No" : 현재 고객 상태를 유지 중 |   

### 데이터 가
#### 1. 결측치 처리

#### 2. 범주형 데이터 처리

#### 3. 수치형 데이터 처리

#### 4. 인코딩


### - 모델

#### ML
1. 로지스틱 회귀(Logistic Regression)
2. 랜덤포레스트(Random Forest)

#### DL


### 결과 요약:


### 결론:


## 한 줄 회고
