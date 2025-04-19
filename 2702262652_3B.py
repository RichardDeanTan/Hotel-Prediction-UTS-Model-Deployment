import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load pickle
model = joblib.load("best_model.pkl")
transformer = joblib.load("transformer_data.pkl")

# Set page configuration
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

# Initialize state buat example selection (keep, cancel, none)
if 'example_selected' not in st.session_state:
    st.session_state.example_selected = None

# Example data buat canceled
canceled_example = {
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
    'market_segment_type': 'Online',
    'repeated_guest': 0,
    'no_of_previous_cancellations': 1,
    'no_of_previous_bookings_not_canceled': 0,
    'avg_price_per_room': 180.0,
    'no_of_special_requests': 0
}

# Example data buat not canceled
not_canceled_example = {
    'no_of_adults': 2,
    'no_of_children': 1,
    'no_of_weekend_nights': 2,
    'no_of_week_nights': 3,
    'type_of_meal_plan': 'Meal Plan 1',
    'required_car_parking_space': 1,
    'room_type_reserved': 'Room_Type 4',
    'lead_time': 30,
    'arrival_year': 2025,
    'arrival_month': 4,
    'market_segment_type': 'Offline',
    'repeated_guest': 1,
    'no_of_previous_cancellations': 0,
    'no_of_previous_bookings_not_canceled': 2,
    'avg_price_per_room': 100.0,
    'no_of_special_requests': 1
}

# Functions buat example selection
def select_cancel_example():
    st.session_state.example_selected = "cancel"
    
def select_keep_example():
    st.session_state.example_selected = "keep"
    
def reset_example():
    st.session_state.example_selected = None

# Main title
st.title("🏨 Hotel Booking Cancellation Predictor")

# Sidebar
with st.sidebar:
    st.header("About This App")
    st.write("""
    Aplikasi ini memprediksi apakah pemesanan hotel akan dibatalkan atau tidak berdasarkan berbagai fitur.
    Model ini dilatih menggunakan data historis pemesanan hotel.
    """)
    
    st.subheader("Available Features:")
    
    features_info = {
        "no_of_adults": "Jumlah orang dewasa",
        "no_of_children": "Jumlah anak kecil",
        "no_of_weekend_nights": "Jumlah malam akhir pekan (Sabtu atau Minggu) tamu menginap atau memesan untuk menginap di hotel",
        "no_of_week_nights": "Jumlah malam dalam seminggu (Senin hingga Jumat) tamu menginap atau memesan untuk menginap di hotel",
        "type_of_meal_plan": "Jenis paket makanan yang dipesan oleh pelanggan",
        "required_car_parking_space": "Apakah pelanggan membutuhkan tempat parkir mobil? (0 - Tidak, 1- Ya)",
        "room_type_reserved": "Jenis kamar yang dipesan oleh pelanggan. Nilai-nilai tersebut dienkripsi oleh INN Hotels",
        "lead_time": "Jumlah hari antara tanggal pemesanan dan tanggal kedatangan",
        "arrival_year": "Tahun tanggal kedatangan",
        "arrival_month": "Bulan tanggal kedatangan",
        "market_segment_type": "Penunjukan segmen pasar",
        "repeated_guest": "Apakah pelanggan tersebut merupakan tamu yang pernah melakukan booking dan juga menginap? (0 - Tidak, 1- Ya)",
        "no_of_previous_cancellations": "Jumlah pemesanan sebelumnya yang dibatalkan oleh pelanggan sebelum pemesanan saat ini",
        "no_of_previous_bookings_not_canceled": "Jumlah pemesanan sebelumnya yang tidak dibatalkan oleh pelanggan sebelum pemesanan saat ini",
        "avg_price_per_room": "Harga rata-rata per hari pemesanan; harga kamar bersifat dinamis. (dalam euro)",
        "no_of_special_requests": "Jumlah total permintaan khusus yang dibuat oleh pelanggan (misalnya lantai yang tinggi, pemandangan dari kamar, dan lain-lain.)"
    }
    
    for feature, description in features_info.items():
        with st.expander(feature):
            st.write(description)
    
    # Example buttons di sidebar
    st.subheader("Try Example Bookings")
    col_buttons1, col_buttons2, col_buttons3 = st.columns(3)

    with col_buttons1:
        st.button("📋 Does Cancel", key="sidebar_cancel", on_click=select_cancel_example)
    with col_buttons2:
        st.button("📋 Doesn't Cancel", key="sidebar_keep", on_click=select_keep_example)
    with col_buttons3:
        st.button("🔄 Reset Form", key="reset_form", on_click=reset_example)

if st.session_state.example_selected == "cancel":
    selected_example = canceled_example
elif st.session_state.example_selected == "keep":
    selected_example = not_canceled_example
else:
    selected_example = {
        'no_of_adults': 2,
        'no_of_children': 0,
        'no_of_weekend_nights': 0,
        'no_of_week_nights': 2,
        'type_of_meal_plan': 'Meal Plan 1',
        'required_car_parking_space': 0,
        'room_type_reserved': 'Room_Type 1',
        'lead_time': 30,
        'arrival_year': 2024,
        'arrival_month': 6,
        'market_segment_type': 'Online',
        'repeated_guest': 0,
        'no_of_previous_cancellations': 0,
        'no_of_previous_bookings_not_canceled': 0,
        'avg_price_per_room': 100.0,
        'no_of_special_requests': 0
    }

# 2 columns
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Enter Booking Details")
    
    with st.form("prediction_form"):
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        row3_col1, row3_col2 = st.columns(2)
        row4_col1, row4_col2 = st.columns(2)
        row5_col1, row5_col2 = st.columns(2)
        row6_col1, row6_col2 = st.columns(2)
        row7_col1, row7_col2 = st.columns(2)
        row8_col1, row8_col2 = st.columns(2)
        
        with row1_col1:
            adults = st.number_input("Number of Adults", 
                                     min_value=0, max_value=10, 
                                     value=selected_example['no_of_adults'])
        with row1_col2:
            children = st.number_input("Number of Children", 
                                       min_value=0, max_value=10, 
                                       value=selected_example['no_of_children'])
            
        with row2_col1:
            weekend_nights = st.number_input("Number of Weekend Nights", 
                                             min_value=0, max_value=10, 
                                             value=selected_example['no_of_weekend_nights'])
        with row2_col2:
            week_nights = st.number_input("Number of Week Nights", 
                                          min_value=0, max_value=20, 
                                          value=selected_example['no_of_week_nights'])
            
        with row3_col1:
            meal_options = ['Meal Plan 1', 'Not Selected', 'Meal Plan 2', 'Meal Plan 3']
            meal_index = meal_options.index(selected_example['type_of_meal_plan'])
            meal_plan = st.selectbox("Type of Meal Plan", 
                                     options=meal_options,
                                     index=meal_index)
        with row3_col2:
            parking = st.selectbox("Required Car Parking Space", 
                                   options=[0, 1], 
                                   index=selected_example['required_car_parking_space'],
                                   format_func=lambda x: 'No' if x == 0 else 'Yes')
            
        with row4_col1:
            room_options = ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 
                            'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']
            room_index = room_options.index(selected_example['room_type_reserved'])
            room_type = st.selectbox("Room Type Reserved", 
                                     options=room_options,
                                     index=room_index)
        with row4_col2:
            lead_time = st.number_input("Lead Time (days)", 
                                        min_value=0, max_value=365, 
                                        value=selected_example['lead_time'])
            
        with row5_col1:
            arrival_year = st.number_input("Arrival Year", 
                                          min_value=2020, max_value=2025, 
                                          value=selected_example['arrival_year'])
        with row5_col2:
            arrival_month = st.number_input("Arrival Month", 
                                           min_value=1, max_value=12, 
                                           value=selected_example['arrival_month'])
            
        with row6_col1:
            segment_options = ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary']
            segment_index = segment_options.index(selected_example['market_segment_type'])
            market_segment = st.selectbox("Market Segment Type", 
                                         options=segment_options,
                                         index=segment_index)
        with row6_col2:
            repeated_guest_val = st.selectbox("Repeated Guest", 
                                              options=[0, 1],
                                              index=selected_example['repeated_guest'],
                                              format_func=lambda x: 'No' if x == 0 else 'Yes')
            
        with row7_col1:
            prev_cancellations = st.number_input("Number of Previous Cancellations", 
                                                min_value=0, max_value=10, 
                                                value=selected_example['no_of_previous_cancellations'])
        with row7_col2:
            prev_bookings_not_canceled = st.number_input("Number of Previous Bookings Not Canceled", 
                                                        min_value=0, max_value=10, 
                                                        value=selected_example['no_of_previous_bookings_not_canceled'])
            
        with row8_col1:
            avg_price = st.slider("Average Price Per Room (in euros)", 
                                min_value=10.0, max_value=500.0, 
                                value=float(selected_example['avg_price_per_room']),
                                step=10.0)
        with row8_col2:
            special_requests = st.slider("Number of Special Requests", 
                                        min_value=0, max_value=5, 
                                        value=selected_example['no_of_special_requests'])
        
        submit_button = st.form_submit_button("Predict")

# Function buat predictions
def make_prediction(input_data):
    # DataFrame buat input data
    input_df = pd.DataFrame([input_data])
    
    # Transform pickle
    input_transformed = transformer.transform(input_df)
    
    # Prediction
    prediction = model.predict(input_transformed)
    prediction_proba = model.predict_proba(input_transformed)
    
    return prediction[0], prediction_proba[0]

with col2:
    st.header("Prediction Results")
    
    if submit_button:
        # Dictionary buat input data
        input_data = {
            'no_of_adults': adults,
            'no_of_children': children,
            'no_of_weekend_nights': weekend_nights,
            'no_of_week_nights': week_nights,
            'type_of_meal_plan': meal_plan,
            'required_car_parking_space': parking,
            'room_type_reserved': room_type,
            'lead_time': lead_time,
            'arrival_year': arrival_year,
            'arrival_month': arrival_month,
            'market_segment_type': market_segment,
            'repeated_guest': repeated_guest_val,
            'no_of_previous_cancellations': prev_cancellations,
            'no_of_previous_bookings_not_canceled': prev_bookings_not_canceled,
            'avg_price_per_room': avg_price,
            'no_of_special_requests': special_requests
        }
        
        # Make prediction
        prediction, prediction_proba = make_prediction(input_data)
        
        # Display prediction
        if prediction == 0:
            st.error("### Prediction: BOOKING WILL BE CANCELED")
            confidence = prediction_proba[0] * 100
        else:
            st.success("### Prediction: BOOKING WILL NOT BE CANCELED")
            confidence = prediction_proba[1] * 100
        
        st.write(f"Confidence: {confidence:.2f}%")
        
        # Pie chart
        fig = px.pie(
            values=[prediction_proba[0] * 100, prediction_proba[1] * 100],
            names=['Canceled', 'Not Canceled'],
            title='Prediction Confidence',
            color=['Canceled', 'Not Canceled'],
            color_discrete_map={'Canceled': '#EF553B', 'Not Canceled': '#00CC96'},
            hole=0.4
        )
        
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(
            annotations=[dict(text=f"{confidence:.1f}%", x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig)
    else:
        st.info("Please enter the booking details and click 'Predict' to get a prediction.")

# Example Tables Section
st.markdown("---")
st.header("Example Booking Scenarios")

col_ex1, col_ex2 = st.columns(2)

with col_ex1:
    st.subheader("Contoh Canceled")
    prediction, prediction_proba = make_prediction(canceled_example)
    
    # Convert ke DataFrame
    df_canceled = pd.DataFrame(canceled_example.items(), columns=['Feature', 'Value'], dtype=str)
    
    # Format values tertentu
    df_canceled['Value'] = df_canceled.apply(
        lambda row: 'No' if row['Feature'] == 'required_car_parking_space' and row['Value'] == "0" else 
                   'Yes' if row['Feature'] == 'required_car_parking_space' and row['Value'] == "1" else
                   'No' if row['Feature'] == 'repeated_guest' and row['Value'] == "0" else
                   'Yes' if row['Feature'] == 'repeated_guest' and row['Value'] == "1" else
                   row['Value'], axis=1)
    
    st.table(df_canceled)
    
    # Display prediction
    if prediction == 0:
        st.error(f"Prediction: **CANCELED** (Confidence: {prediction_proba[0]*100:.1f}%)")
    else:
        st.success(f"Prediction: **NOT CANCELED** (Confidence: {prediction_proba[1]*100:.1f}%)")
    
    # Button
    st.button("📝 Use This Example", key="use_cancel_example", on_click=select_cancel_example)

with col_ex2:
    st.subheader("Contoh Not Canceled")
    prediction, prediction_proba = make_prediction(not_canceled_example)
    
    # Convert to DataFrame
    df_not_canceled = pd.DataFrame(not_canceled_example.items(), columns=['Feature', 'Value'], dtype=str)
    
    # Format values tertentu
    df_not_canceled['Value'] = df_not_canceled.apply(
        lambda row: 'No' if row['Feature'] == 'required_car_parking_space' and row['Value'] == "0" else 
                   'Yes' if row['Feature'] == 'required_car_parking_space' and row['Value'] == "1" else
                   'No' if row['Feature'] == 'repeated_guest' and row['Value'] == "0" else
                   'Yes' if row['Feature'] == 'repeated_guest' and row['Value'] == "1" else
                   row['Value'], axis=1)
    
    st.table(df_not_canceled)
    
    # Display prediction
    if prediction == 0:
        st.error(f"Prediction: **CANCELED** (Confidence: {prediction_proba[0]*100:.1f}%)")
    else:
        st.success(f"Prediction: **NOT CANCELED** (Confidence: {prediction_proba[1]*100:.1f}%)")
    
    # Button
    st.button("📝 Use This Example", key="use_not_cancel_example", on_click=select_keep_example)

# Bottom section
st.markdown("---")
st.header("Model Information")

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.subheader("Model Details")
    st.write("""
    Aplikasi ini menggunakan model XGBoost Classifier yang dilatih pada data pemesanan hotel. Model ini memprediksi apakah pemesanan akan dibatalkan berdasarkan berbagai karakteristik booking.
    
    Model ini dilatih dengan:
    - Feature engineering dan preprocessing menggunakan scikit-learn
    - Class imbalance menggunakan SMOTE
    - Hyperparameter tuning menggunakan GridSearchCV
    
    Model telah dilakukan training pada dataset dengan total 36000 data. Hasil dari training model memperoleh akurasi train 92% dan test 88%.
    """)

with col_info2:
    st.subheader("Training Process")
    st.write("""
    Proses Training:
    1. Data preprocessing (handling missing values dan outliers)
    2. Feature scaling (StandardScaler untuk fitur non outlier, RobustScaler untuk fitur outliers)
    3. Categorical encoding (OneHotEncoder)
    4. Class Balancing dengan SMOTE
    5. Model selection (Membandingkan Random Forest and XGBoost)
    6. Hyperparameter tuning untuk memaksimalkan performance
    
    
    Model XGBoost dipilih sebagai model final karena kinerjanya yang unggul dibanding Random Forest.
    """)

st.markdown("---")
st.caption("© 2025 Hotel Booking Cancellation Predictor | Created with Streamlit |  Author: Richard Dean Tanjaya ")

### 2702262652 / Richard Dean Tanjaya