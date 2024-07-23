import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv('C:/Users/DELL/OneDrive/Desktop/crop_yield/New folder/crop_yield.csv')  # Replace 'your_dataset.csv' with the actual dataset path

# Drop the 'Crop_Year' column
data = data.drop(columns=['Crop_Year'])

# Define features and target variable
X = data.drop(columns=['Yield'])
y = data['Yield']

# Define categorical and numerical features
categorical_features = ['Crop', 'Season', 'State']
numerical_features = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor(random_state=42))])

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

# Print the R2 score for Random Forest model
st.write(f'R2 Score for Random Forest: {r2}')

# Streamlit app for predicting crop yield
import streamlit as st
import pandas as pd

st.title('Crop Yield Prediction App')

# Input fields for features
area = st.number_input('Area', value=0.0)
production = st.number_input('Production', value=0.0)
annual_rainfall = st.number_input('Annual Rainfall', value=0.0)
fertilizer = st.number_input('Fertilizer', value=0.0)
pesticide = st.number_input('Pesticide', value=0.0)
crop = st.selectbox('Select Crop', data['Crop'].unique())
season = st.selectbox('Select Season', data['Season'].unique())
state = st.selectbox('Select State', data['State'].unique())

# Prepare input features
input_features = pd.DataFrame({
    'Area': [area],
    'Production': [production],
    'Annual_Rainfall': [annual_rainfall],
    'Fertilizer': [fertilizer],
    'Pesticide': [pesticide],
    'Crop': [crop],
    'Season': [season],
    'State': [state]
})

# Predict button
if st.button('Predict'):
    # Predict using the trained model
    prediction = rf_model.predict(input_features)
    st.success(f'Predicted Crop Yield: {prediction[0]}')