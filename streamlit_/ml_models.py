# ml_models.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

class MLModels:
    def __init__(self):
        self.df = pd.read_csv('train.csv')
        self.prepare_data()
        
    def prepare_data(self):
        self.X = self.df.drop(columns=['Churn', 'customerID'], errors='ignore')
        self.y = self.df['Churn']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

    def logistic_regression(self):
        st.write("### Logistic Regression Model")
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=5000,
            random_state=42
        )
        self.train_and_evaluate(model, "Logistic Regression", threshold=None)

    def gradient_boosting(self):
        st.write("### Gradient Boosting Model")
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.train_and_evaluate(model, "Gradient Boosting", threshold=None)

    def xgboost(self):
        st.write("### XGBoost Model")
        subsample = st.slider("Select subsample", 0.0, 1.0, 0.5, step=0.1)
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=9,
            learning_rate=0.01,
            subsample=subsample,
            colsample_bytree=1,
            scale_pos_weight=1,
            eval_metric='logloss',
            random_state=42
        )
        
        self.train_and_evaluate(model, "XGBoost", threshold=None)

    def ensemble(self):
        st.write("### Ensemble Model")
        lr_model = LogisticRegression(class_weight='balanced', max_iter=5000, random_state=42)
        gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
        xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=9, learning_rate=0.01, random_state=42)
        
        model = VotingClassifier(
            estimators=[('lr', lr_model), ('gb', gb_model), ('xgb', xgb_model)],
            voting='soft',
            weights=[2, 1, 2]
        )
        threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5, step=0.1)
        self.train_and_evaluate(model, "Ensemble", threshold)

    def train_and_evaluate(self, model, model_name, threshold):
        with st.spinner(f'Training {model_name}...'):
            model.fit(self.X_train, self.y_train)
            y_pred_prob = model.predict_proba(self.X_test)[:, 1]

            if threshold is None:
                threshold = 0.5
            
            y_pred = (y_pred_prob >= threshold).astype(int)
            
            
            st.write("#### Confusion Matrix:")
            st.write(confusion_matrix(self.y_test, y_pred))
            
            st.write("#### Classification Report:")
            st.text(classification_report(self.y_test, y_pred))
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)
            auc_score = roc_auc_score(self.y_test, y_pred_prob)
            
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {model_name}')
            ax.legend()
            st.pyplot(fig)

    def run(self, sub_category):
        if sub_category == 'Logistic':
            self.logistic_regression()
        elif sub_category == 'gb':
            self.gradient_boosting()
        elif sub_category == 'xgb':
            self.xgboost()
        elif sub_category == 'ensemble':
            self.ensemble()
        
        with st.expander("Show Code"):
            with open('ml_models.py', 'r', encoding='utf-8') as file:
                st.code(file.read())