import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib


class HotelBookingModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.transformer = None
        self.model = None
        self.feature_names = None
        self.label_map = {'Canceled': 0, 'Not_Canceled': 1}
    
    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df = df.drop(columns=['Booking_ID', 'arrival_date'])
        
        X = df.drop(['booking_status'], axis=1)
        y = df['booking_status']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        
        return X_train, X_test, y_train, y_test
    
    def handle_missing_values(self, X_train, X_test):
        num_cols = X_train.select_dtypes(include=['float', 'int']).columns.to_list()
        object_cols = X_train.select_dtypes(include='object').columns.to_list()
        
        missing_num_cols = X_train[num_cols].isnull().sum()[X_train[num_cols].isnull().sum() > 0].index.to_list()
        for col in missing_num_cols:
            median_value = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_value)
            X_test[col] = X_test[col].fillna(median_value)
        
        missing_object_cols = X_test[object_cols].isnull().sum()[X_test[object_cols].isnull().sum() > 0].index.to_list()
        for col in missing_object_cols:
            most_freq = X_train[col].value_counts().idxmax()
            X_train[col] = X_train[col].fillna(most_freq)
            X_test[col] = X_test[col].fillna(most_freq)
        
        return X_train, X_test, num_cols, object_cols
    
    def detect_outliers(self, X_train, num_cols):
        outlier_cols = []
        no_outlier_cols = []
        
        for col in num_cols:
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = X_train[(X_train[col] < lower_bound) | (X_train[col] > upper_bound)][col]
            
            if len(outliers) > 0:
                outlier_cols.append(col)
            else:
                no_outlier_cols.append(col)
                
        return outlier_cols, no_outlier_cols
    
    def preprocess_data(self, X_train, X_test, ohe_cols, scaler_cols, robust_cols):
        self.transformer = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), scaler_cols),
                ('robust', RobustScaler(), robust_cols),
                ('ohe', OneHotEncoder(handle_unknown='ignore'), ohe_cols)
            ]
        )
        
        X_train_transformed = self.transformer.fit_transform(X_train)
        X_test_transformed = self.transformer.transform(X_test)
        
        ohe_feature_names = self.transformer.named_transformers_['ohe'].get_feature_names_out(ohe_cols)
        self.feature_names = scaler_cols + robust_cols + list(ohe_feature_names)
        
        return X_train_transformed, X_test_transformed
    
    def apply_smote(self, X_train, y_train):
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        return X_train_resampled, y_train_resampled
    
    def train_model(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        base_model = XGBClassifier(random_state=self.random_state)
        
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=5, 
            scoring='accuracy', 
            verbose=1, 
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = XGBClassifier(random_state=self.random_state)
        self.model.set_params(**grid_search.best_params_)
        self.model.fit(X_train, y_train)
        
        print("Best parameters found:", grid_search.best_params_)
        print("Best cross-validation score:", grid_search.best_score_)
        
        return self.model
    
    def evaluate_model(self, X_train, y_train, X_test, y_test):
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        print('Classification Report (Training Set):\n', classification_report(y_train, y_pred_train))
        print(f"Accuracy Score (Training Set): {accuracy_score(y_train, y_pred_train):.4f}\n")
        
        print('Classification Report (Test Set):\n', classification_report(y_test, y_pred_test))
        print(f"Accuracy Score (Test Set): {accuracy_score(y_test, y_pred_test):.4f}")
        
        return {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test)
        }
    
    def save_model(self, model_path='best_model.pkl', transformer_path='transformer_data.pkl'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.transformer, transformer_path)
        print(f"Model saved to {model_path}")
        print(f"Transformer saved to {transformer_path}")
    
    def run_pipeline(self, file_path):
        # Data Splitting
        X_train, X_test, y_train, y_test = self.load_data(file_path)
        
        # Handle missing values
        X_train, X_test, num_cols, object_cols = self.handle_missing_values(X_train, X_test)
        
        # Detect outliers
        outlier_cols, no_outlier_cols = self.detect_outliers(X_train, num_cols)
        
        # Preprocess data
        X_train_transformed, X_test_transformed = self.preprocess_data(
            X_train, X_test, object_cols, no_outlier_cols, outlier_cols
        )
        
        # Mapping target variables
        y_train_mapped = y_train.map(self.label_map)
        y_test_mapped = y_test.map(self.label_map)
        
        # Apply SMOTE
        X_train_resampled, y_train_resampled = self.apply_smote(X_train_transformed, y_train_mapped)
        
        # Train model
        self.train_model(X_train_resampled, y_train_resampled)
        
        # Evaluate model
        metrics = self.evaluate_model(
            X_train_resampled, y_train_resampled, 
            X_test_transformed, y_test_mapped
        )
        
        # Save model
        self.save_model()
        
        return metrics

if __name__ == "__main__":
    hotel_model = HotelBookingModel(random_state=42)
    
    results = hotel_model.run_pipeline('Dataset_B_hotel.csv')
    
    print("\nPipeline completed successfully!")
    print(f"Final training accuracy: {results['train_accuracy']:.4f}")
    print(f"Final test accuracy: {results['test_accuracy']:.4f}")


### 2702262652 / Richard Dean Tanjaya

### Run OOP
# terminal -> python [name].py