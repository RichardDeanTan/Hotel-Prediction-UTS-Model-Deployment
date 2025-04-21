import pandas as pd
import numpy as np
import joblib

class HotelBookingInference:
    def __init__(self, model_path='best_model.pkl', transformer_path='transformer_data.pkl'):
        self.model = joblib.load(model_path)
        self.transformer = joblib.load(transformer_path)
        self.label_map = {0: 'Canceled', 1: 'Not_Canceled'}
    
    def preprocess_single_input(self, input_data):
        input_df = pd.DataFrame([input_data])
        
        input_df = input_df.drop(columns=['Booking_ID', 'arrival_date'])
        
        return self.handle_missing_values(input_df)
    
    def preprocess_csv_input(self, input_file_path):
        input_df = pd.read_csv(input_file_path)
        
        input_df = input_df.drop(columns=['Booking_ID', 'arrival_date'])
        
        return self.handle_missing_values(input_df)
    
    def handle_missing_values(self, df):
        num_cols = df.select_dtypes(include=['float', 'int']).columns.to_list()
        object_cols = df.select_dtypes(include='object').columns.to_list()
        
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
        
        for col in object_cols:
            if df[col].isnull().sum() > 0:
                most_freq = df[col].value_counts().idxmax()
                df[col] = df[col].fillna(most_freq)
        
        return df
    
    def predict(self, input_data, is_file=False):
        if is_file:
            processed_data = self.preprocess_csv_input(input_data)
        else:
            processed_data = self.preprocess_single_input(input_data)
        
        has_target = 'booking_status' in processed_data.columns
        if has_target:
            actual_status = processed_data['booking_status']
            processed_data = processed_data.drop(['booking_status'], axis=1)
        
        transformed_data = self.transformer.transform(processed_data)
        
        # Model predictions (0 / 1)
        predictions_numeric = self.model.predict(transformed_data)
        
        predictions = [self.label_map[pred] for pred in predictions_numeric]
        
        if has_target:
            accuracy = (predictions == actual_status).mean() * 100
            return predictions, accuracy
        else:
            return predictions

if __name__ == "__main__":
    predictor = HotelBookingInference(model_path='best_model.pkl', transformer_path='transformer_data.pkl')
    
    sample_input = {
        'Booking_ID': 'INN00001',
        'no_of_adults': 2,
        'no_of_children': 0,
        'no_of_weekend_nights': 0,
        'no_of_week_nights': 1,
        'type_of_meal_plan': 'Not Selected',
        'required_car_parking_space': 0,
        'room_type_reserved': 'Room_Type 1',
        'lead_time': 120,
        'arrival_year': 2024,
        'arrival_month': 7,
        'arrival_date': 4,
        'market_segment_type': 'Online',
        'repeated_guest': 0,
        'no_of_previous_cancellations': 1,
        'no_of_previous_bookings_not_canceled': 0,
        'avg_price_per_room': 180.0,
        'no_of_special_requests': 0
    }
    
    # Single Prediction
    prediction = predictor.predict(sample_input)
    print(f"Prediction for sample input: {prediction}")
    
    # CSV Prediction
    batch_predictions = predictor.predict('inference.csv', is_file=True)
    print(f"Predictions: {batch_predictions}")