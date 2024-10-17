import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

@st.cache_data
def load_data():
    df = pd.read_csv("restaurant_customer_satisfaction.csv")
    return df

@st.cache_data
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    return feature_dict.get(val, None)

def get_value(val, my_dict):
    return my_dict.get(val, None)

# Sidebar setup
st.sidebar.image('SatisfactionPredictor.png', width=150)  # Adjust the width as needed
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction', 'Data Visualization', 'About Us', 'Contact Us'])  

if app_mode == 'Home':
    #st.title('PREDICTING CUSTOMER SATISFACTION')
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: orange;'>PREDICTING CUSTOMER SATISFACTION</h1>
    </div>
    """, unsafe_allow_html=True)
    st.image('resturant.png')

    #st.subheader("About the App")
    st.write("""
        Welcome to the **Customer Satisfaction Predictor**! 
        
        This app is designed to help restaurant owners and managers understand their customers' satisfaction levels. By using advanced machine learning models, we predict whether a customer is satisfied based on various factors, such as:

        - Age, income, and loyalty program membership
        - Frequency of visits and preferred cuisine
        - Ratings for service, food, and ambiance
        - Dining occasions and times of visit

        With this predictive tool, you can gain insights into your customers' dining experiences and improve their overall satisfaction. Whether you are a restaurant owner seeking to enhance customer loyalty or a curious customer looking to explore how different factors influence satisfaction, this app is for you!

        Start by navigating to the **Prediction** page, or explore visual insights on the **Data Visualization** page. For any inquiries, visit our **Contact Us** page.
    """)

# Prediction mode
elif app_mode == 'Prediction':
    # Display the GIF for prediction mode
    file_ = open("PREDICTING CUSTOMER SATISFACTION.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="Customer Satisfaction Prediction GIF">', unsafe_allow_html=True)
    st.subheader('Please Complete all required fields to see if you are satisfied as a customer!')
    st.sidebar.header("Information about the client:")

    # User inputs from Streamlit
    Age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=0, step=1, key="Age_input")
    Income = st.sidebar.number_input('Income', min_value=0, max_value=500000, value=0, step=100, key="Income_input")
    VisitFrequency = st.sidebar.radio('Visit Frequency', ('Daily', 'Weekly', 'Monthly', 'Rarely'), key="VisitFrequency_radio")
    PreferredCuisine = st.sidebar.radio('Preferred Cuisine', ('Chinese', 'American', 'Indian', 'Mexican', 'Italian'), key="PreferredCuisine_radio")
    TimeOfVisit = st.sidebar.radio('Time Of Visit', ['Breakfast', 'Lunch', 'Dinner'], key="TimeOfVisit_radio")
    DiningOccasion = st.sidebar.radio('Dining Occasion', ('Business', 'Casual', 'Celebration'), key="DiningOccasion_radio")
    LoyaltyProgramMember = st.sidebar.radio('Loyalty Program Member', [0, 1], key="LoyaltyProgramMember_radio")
    ServiceRating = st.sidebar.slider('Service Rating', 0, 5, 0, key='service_rating_slider')
    FoodRating = st.sidebar.slider('Food Rating', 0, 5, 0, key='food_rating_slider')
    AmbianceRating = st.sidebar.slider('Ambiance Rating', 0, 5, 0, key='ambiance_rating_slider')

    # Initialize a dictionary to store the encoded values
    encoded_input = {
        'Age': Age,
        'Income': Income,
        'LoyaltyProgramMember': LoyaltyProgramMember,
        'ServiceRating': ServiceRating,
        'FoodRating': FoodRating,
        'AmbianceRating': AmbianceRating,
        
    }

    # VisitFrequency Encoding (one-hot)
    for key in ['Daily', 'Weekly', 'Monthly', 'Rarely']:
        encoded_input[f'VisitFrequency_{key}'] = 1 if VisitFrequency == key else 0

    # PreferredCuisine Encoding (one-hot)
    for key in ['American', 'Chinese', 'Indian', 'Mexican', 'Italian']:
        encoded_input[f'PreferredCuisine_{key}'] = 1 if PreferredCuisine == key else 0

    # TimeOfVisit Encoding (one-hot)
    for key in ['Breakfast', 'Lunch', 'Dinner']:
        encoded_input[f'TimeOfVisit_{key}'] = 1 if TimeOfVisit == key else 0

    # DiningOccasion Encoding (one-hot)
    for key in ['Business', 'Casual', 'Celebration']:
        encoded_input[f'DiningOccasion_{key}'] = 1 if DiningOccasion == key else 0

    # Convert encoded_input dictionary to a DataFrame
    single_sample_df = pd.DataFrame([encoded_input])

    st.write("Encoded Input DataFrame:")
    st.write(single_sample_df)
    
    # Convert encoded_input dictionary to a DataFrame with exact column names and order
    columns_in_order = ['Age',
    'Income',
    'ServiceRating',
    'FoodRating',
    'AmbianceRating',
    'LoyaltyProgramMember',
    'VisitFrequency_Daily',
    'VisitFrequency_Monthly',
    'VisitFrequency_Rarely',
    'VisitFrequency_Weekly',
    'PreferredCuisine_American',
    'PreferredCuisine_Chinese',
    'PreferredCuisine_Indian',
    'PreferredCuisine_Italian',
    'PreferredCuisine_Mexican',
    'TimeOfVisit_Breakfast',
    'TimeOfVisit_Dinner',
    'TimeOfVisit_Lunch',
    'DiningOccasion_Business',
    'DiningOccasion_Casual',
    'DiningOccasion_Celebration']
    
    for col in columns_in_order:
        if col not in encoded_input:
            encoded_input[col] = 0  # Fill missing columns with 0
            
    # Ensure the data is ordered properly
    single_sample_df = pd.DataFrame([encoded_input], columns=columns_in_order)

    # Load the scaler using pickle
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    # Ensure columns match exactly before scaling
    if list(single_sample_df.columns) != list(scaler.feature_names_in_):
        raise ValueError(f"Feature names do not match!\nExpected: {list(scaler.feature_names_in_)}\nGot: {list(single_sample_df.columns)}")

    # Scale the input data
    single_sample_scaled = scaler.transform(single_sample_df)
    
    st.write(single_sample_scaled)

    # Load the model using pickle
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Prediction logic
    if st.button("Predict"):
        # Predict using the scaled data
        prediction = model.predict(single_sample_scaled)

        if prediction[0] == 0:
            st.error('You are not satisfied with the service.')
            file = open("bad-customers.gif", "rb")
            contents = file.read()
            data_url_no = base64.b64encode(contents).decode("utf-8")
            file.close()
            st.markdown(f'<img src="data:image/gif;base64,{data_url_no}" alt="unsatisfied customer gif">', unsafe_allow_html=True)
        else:
            st.success('Congratulations! You are a satisfied customer')
            file_ = open("happy-customers.gif", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()
            st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="satisfied customer gif">', unsafe_allow_html=True)
    