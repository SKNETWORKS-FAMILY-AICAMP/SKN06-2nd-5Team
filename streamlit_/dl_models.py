# dl_models.py
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

class DLModels:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"Using device: {self.device}")
        self.prepare_data()
        self.epochs = 100
        self.lr = 0.01

    def prepare_data(self):
        try:
            df = pd.read_csv("train.csv")
            X = df.drop(columns=['Churn', 'customerID']).values
            y = df['Churn'].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify=y, test_size=0.2, random_state=0
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.train_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_train_scaled, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
                ),
                batch_size=128,
                shuffle=True,
                drop_last=True
            )
            
            self.test_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_test_scaled, dtype=torch.float32),
                    torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
                ),
                batch_size=128
            )
            st.success("Data preparation completed!")
            
        except Exception as e:
            st.error(f"Error in data preparation: {str(e)}")
            raise e

    class BasicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lr1 = nn.Linear(40, 32)
            self.lr2 = nn.Linear(32, 8)
            self.lr3 = nn.Linear(8, 1)
            self.relu = nn.ReLU()
            self.logistic = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.lr1(x))
            x = self.relu(self.lr2(x))
            return self.logistic(self.lr3(x))

    class BatchNormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lr1 = nn.Linear(40, 32)
            self.bn1 = nn.BatchNorm1d(32)
            self.lr2 = nn.Linear(32, 8)
            self.bn2 = nn.BatchNorm1d(8)
            self.lr3 = nn.Linear(8, 1)
            self.relu = nn.ReLU()
            self.logistic = nn.Sigmoid()

        def forward(self, x):
            x = self.lr1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.lr2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.lr3(x)
            return self.logistic(x)

    class DropoutModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lr1 = nn.Linear(40, 32)
            self.dropout1 = nn.Dropout(0.3)
            self.lr2 = nn.Linear(32, 8)
            self.dropout2 = nn.Dropout(0.3)
            self.lr3 = nn.Linear(8, 1)
            self.relu = nn.ReLU()
            self.logistic = nn.Sigmoid()

        def forward(self, x):
            x = self.lr1(x)
            x = self.dropout1(x)
            x = self.relu(x)
            x = self.lr2(x)
            x = self.dropout2(x)
            x = self.relu(x)
            x = self.lr3(x)
            return self.logistic(x)

    def train_model(self, model_name, model, optimizer, scheduler=None, use_smote=False, patience=10):
        try:
            st.write(f"Starting training for {model_name}...")
            loss_fn = nn.BCELoss()
            
            # Early stopping setup
            best_val_loss = float('inf')
            early_stop_counter = 0
            best_model = None
            
            if use_smote:
                st.write("Applying SMOTE...")
                X = torch.stack([x for x, _ in self.train_loader.dataset]).numpy()
                y = torch.stack([y for _, y in self.train_loader.dataset]).numpy().reshape(-1)
                
                smote = SMOTE(sampling_strategy=1.0, random_state=0)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
                train_loader = DataLoader(
                    TensorDataset(
                        torch.tensor(X_resampled, dtype=torch.float32),
                        torch.tensor(y_resampled, dtype=torch.float32).reshape(-1, 1)
                    ),
                    batch_size=128,
                    shuffle=True
                )
            else:
                train_loader = self.train_loader

            train_losses = []
            train_accuracies = []
            val_losses = []
            val_accuracies = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for epoch in range(self.epochs):
                # Training
                model.train()
                train_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    predicted = (outputs.data >= 0.5).float()
                    total += targets.size(0)
                    correct += predicted.eq(targets.data).sum().item()
                
                if scheduler:
                    scheduler.step()
                
                # Validation
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, targets in self.test_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets)
                        
                        val_loss += loss.item()
                        predicted = (outputs.data >= 0.5).float()
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets.data).sum().item()
                
                # Calculate metrics
                train_loss = train_loss / len(train_loader)
                train_acc = 100. * correct / total
                val_loss = val_loss / len(self.test_loader)
                val_acc = 100. * val_correct / val_total
                
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict().copy()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    
                if early_stop_counter >= patience:
                    st.write(f'Early stopping triggered at epoch {epoch+1}')
                    model.load_state_dict(best_model)
                    break
                
                # Update progress
                progress = (epoch + 1) / self.epochs
                progress_bar.progress(progress)
                status_text.text(f'Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}, '
                               f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, '
                               f'Val Acc: {val_acc:.2f}%')

            # Plot results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            ax1.plot(train_losses, label='Train Loss')
            ax1.plot(val_losses, label='Validation Loss')
            ax1.set_title('Loss Curves')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            ax2.plot(train_accuracies, label='Train Accuracy')
            ax2.plot(val_accuracies, label='Validation Accuracy')
            ax2.set_title('Accuracy Curves')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Final metrics
            st.write("### Final Results")
            st.write(f"Train Loss: {train_losses[-1]:.4f}")
            st.write(f"Train Accuracy: {train_accuracies[-1]:.2f}%")
            st.write(f"Validation Loss: {val_losses[-1]:.4f}")
            st.write(f"Validation Accuracy: {val_accuracies[-1]:.2f}%")
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            raise e

    def run(self, sub_category):
        st.header('Deep Learning Models')
        
        try:
            if sub_category == 'basic':
                st.subheader('Basic Model')
                model = self.BasicModel().to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=self.lr)
                self.train_model("Basic Model", model, optimizer)
                
            elif sub_category == 'batchnormal':
                st.subheader('BatchNorm Model')
                model = self.BatchNormModel().to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=self.lr)
                self.train_model("BatchNorm Model", model, optimizer)
                
            elif sub_category == 'dropout':
                st.subheader('Dropout Model')
                model = self.DropoutModel().to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=self.lr)
                self.train_model("Dropout Model", model, optimizer)
                
            elif sub_category == 'cosineannealingwarmrestarts':
                st.subheader('Cosine Annealing Warm Restarts')
                model = self.DropoutModel().to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=self.lr)
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=10, T_mult=2
                )
                self.train_model("CAWR Model", model, optimizer, scheduler)
                
            elif sub_category == 'CAWR+L2':
                st.subheader('CAWR + L2 Regularization')
                model = self.DropoutModel().to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.01)
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=10, T_mult=2
                )
                self.train_model("CAWR+L2 Model", model, optimizer, scheduler)
                
            elif sub_category == 'stepLR':
                st.subheader('Step LR')
                model = self.DropoutModel().to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=self.lr)
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=5, gamma=0.5
                )
                self.train_model("StepLR Model", model, optimizer, scheduler)
                
            elif sub_category == 'stepLR+SMOTE':
                st.subheader('Step LR + SMOTE')
                model = self.DropoutModel().to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=self.lr)
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=5, gamma=0.5
                )
                self.train_model("StepLR+SMOTE Model", model, optimizer, scheduler, use_smote=True)

            # Show code
            with st.expander("Show Code"):
                with open(__file__, 'r', encoding='utf-8') as file:
                    st.code(file.read())
                
        except Exception as e:
            st.error(f"Error in model execution: {str(e)}")
            raise e

if __name__ == "__main__":
    dl = DLModels()
    dl.run('basic')