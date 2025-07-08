import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser # To open the generated map in a browser, if embedding is not preferred directly

# Suppress warnings from scikit-learn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Data Simulation Functions ---
def generate_weather_data(num_entries=10):
    """Generates synthetic current weather data."""
    data = []
    locations = [
        'Haripur', 'Shantipur', 'Ramnagar', 'Krishnapur', 'Faizabad', 'Varanasi',
        'Cuttack', 'Guwahati', 'Patna', 'Surat', 'Kochi', 'Hyderabad',
        'Chennai', 'Pune', 'Bhopal', 'Jaipur'
    ]
    for i in range(num_entries):
        timestamp = datetime.now() - timedelta(minutes=i*30)
        rainfall_amount = round(random.uniform(5, 60), 2)
        water_level = round(random.uniform(5, 12), 2)
        temperature = round(random.uniform(20, 35), 1)
        humidity = round(random.uniform(60, 95), 1)
        location = random.choice(locations)
        data.append({
            'timestamp': timestamp,
            'location': location,
            'rainfall_amount_mm': rainfall_amount,
            'water_level_m': water_level,
            'temperature_c': temperature,
            'humidity_percent': humidity
        })
    return pd.DataFrame(data)

def generate_historical_flood_data(num_entries=15):
    """Generates synthetic historical flood records."""
    data = []
    locations = [
        'Haripur', 'Shantipur', 'Ramnagar', 'Krishnapur', 'Faizabad', 'Varanasi',
        'Cuttack', 'Guwahati', 'Patna', 'Surat', 'Kochi', 'Hyderabad',
        'Chennai', 'Pune', 'Bhopal', 'Jaipur'
    ]
    severities = ['Low', 'Medium', 'High']
    for i in range(num_entries):
        timestamp = datetime.now() - timedelta(days=random.randint(30, 365*5))
        rainfall_amount = round(random.uniform(40, 150), 2)
        water_level = round(random.uniform(9, 15), 2)
        location = random.choice(locations)
        severity = random.choice(severities)
        deaths = 0
        if severity == 'High':
            deaths = random.randint(1, 10)
        elif severity == 'Medium':
            deaths = random.randint(0, 3)

        data.append({
            'timestamp': timestamp,
            'location': location,
            'rainfall_amount_mm': rainfall_amount,
            'water_level_m': water_level,
            'severity': severity,
            'deaths': deaths
        })
    return pd.DataFrame(data)

# --- Streamlit UI Starts Here ---
st.set_page_config(layout="wide", page_title="Smart Flood Alert System")
st.title("Smart Flood Alert System")

# Use Streamlit's session state to persist dataframes and model across reruns
if 'current_weather_df' not in st.session_state:
    st.session_state.current_weather_df = pd.DataFrame()
if 'historical_floods_df' not in st.session_state:
    st.session_state.historical_floods_df = pd.DataFrame()
if 'model' not in st.session_state:
    st.session_state.model = None

# Create tabs for different sections
tab_simulation, tab_analysis, tab_prediction, tab_sms, tab_map = st.tabs([
    "Data Simulation", "Data Analysis", "Flood Prediction", "SMS Alert", "Map Visualization"
])

# --- Data Simulation Tab ---
with tab_simulation:
    st.header("1. Data Simulation")

    st.subheader("Current Weather Data Simulation")
    col1, col2 = st.columns([1, 2])
    with col1:
        num_current_entries = st.number_input(
            "Number of entries for Current Weather Data:",
            min_value=1,
            value=20,
            step=1,
            key="num_current_input" # Unique key for this widget
        )
        if st.button("Simulate Current Data", key="simulate_current_btn"):
            st.session_state.current_weather_df = generate_weather_data(num_entries=num_current_entries)
            st.success(f"{num_current_entries} entries of current weather data simulated.")
    with col2:
        if not st.session_state.current_weather_df.empty:
            st.dataframe(st.session_state.current_weather_df, height=250)
        else:
            st.info("No current weather data simulated yet. Click 'Simulate Current Data'.")

    st.markdown("---") # Separator

    st.subheader("Historical Flood Data Simulation")
    col3, col4 = st.columns([1, 2])
    with col3:
        num_historical_entries = st.number_input(
            "Number of entries for Historical Flood Data:",
            min_value=1,
            value=30,
            step=1,
            key="num_historical_input" # Unique key for this widget
        )
        if st.button("Simulate Historical Data", key="simulate_historical_btn"):
            st.session_state.historical_floods_df = generate_historical_flood_data(num_entries=num_historical_entries)
            st.success(f"{num_historical_entries} entries of historical flood data simulated.")
    with col4:
        if not st.session_state.historical_floods_df.empty:
            st.dataframe(st.session_state.historical_floods_df, height=250)
        else:
            st.info("No historical flood data simulated yet. Click 'Simulate Historical Data'.")

# --- Data Analysis Tab ---
with tab_analysis:
    st.header("2. Data Analysis")
    if st.button("Perform Analysis", key="perform_analysis_btn"):
        if st.session_state.historical_floods_df.empty:
            st.warning("No historical flood data available. Please simulate data first in the 'Data Simulation' tab.")
        else:
            st.subheader("Analysis Results:")
            avg_rainfall_flood = st.session_state.historical_floods_df['rainfall_amount_mm'].mean()
            avg_water_level_flood = st.session_state.historical_floods_df['water_level_m'].mean()
            st.write(f"**Average Rainfall during Historical Floods:** {avg_rainfall_flood:.2f} mm")
            st.write(f"**Average Water Level during Historical Floods:** {avg_water_level_flood:.2f} m")

            st.write("**Historical Flood Severity Counts:**")
            severity_counts = st.session_state.historical_floods_df['severity'].value_counts()
            st.dataframe(severity_counts)

            if 'deaths' in st.session_state.historical_floods_df.columns:
                total_deaths = st.session_state.historical_floods_df['deaths'].sum()
                st.write(f"**Total Recorded Deaths from Historical Floods:** {total_deaths}")
            
            st.write("**Descriptive Statistics for Historical Flood Data:**")
            st.dataframe(st.session_state.historical_floods_df[['rainfall_amount_mm', 'water_level_m', 'deaths']].describe())
            st.success("Data analysis performed successfully.")
    else:
        st.info("Click 'Perform Analysis' to see the results.")

# --- Flood Prediction Tab ---
with tab_prediction:
    st.header("3. Flood Prediction")
    if st.button("Train Model and Predict", key="train_predict_btn"):
        if st.session_state.historical_floods_df.empty or st.session_state.current_weather_df.empty:
            st.warning("Please simulate both historical and current weather data first in the 'Data Simulation' tab.")
        else:
            # Prepare historical data for training
            X_hist = st.session_state.historical_floods_df[['rainfall_amount_mm', 'water_level_m']]
            y_hist = st.session_state.historical_floods_df['severity'].apply(lambda x: 1 if x in ['Medium', 'High'] else 0) # Binary classification: Flood (1) or No Flood (0)

            if len(X_hist) < 2:
                st.error("Not enough historical data to train the model. Need at least 2 entries.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_hist, y_hist, test_size=0.3, random_state=42)

                # Train Random Forest Classifier
                st.session_state.model = RandomForestClassifier(n_estimators=100, random_state=42)
                st.session_state.model.fit(X_train, y_train)
                st.success("Random Forest Classifier trained successfully.")

                # Evaluate model
                if not X_test.empty:
                    y_pred = st.session_state.model.predict(X_test)
                    st.subheader("Model Evaluation on Test Data:")
                    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
                    st.write(f"**Precision:** {precision_score(y_test, y_pred, zero_division=0):.2f}")
                    st.write(f"**Recall:** {recall_score(y_test, y_pred, zero_division=0):.2f}")
                    st.write(f"**F1-Score:** {f1_score(y_test, y_pred, zero_division=0):.2f}")
                    st.text("Classification Report:")
                    st.code(classification_report(y_test, y_pred, zero_division=0))
                else:
                    st.info("Not enough test data to evaluate the model.")

                st.markdown("---")

                # Rule-based Prediction (using current weather data)
                st.subheader("Rule-based Flood Risk Assessment (Current Data):")
                for index, row in st.session_state.current_weather_df.iterrows():
                    location = row['location']
                    rainfall = row['rainfall_amount_mm']
                    water_level = row['water_level_m']
                    risk = "Low"
                    if rainfall > 50 or water_level > 10:
                        risk = "High"
                    elif rainfall > 30 or water_level > 8:
                        risk = "Medium"
                    st.write(f"  **{location}**: Rainfall={rainfall}mm, Water Level={water_level}m, Risk={risk}")
                
                st.markdown("---")

                # ML-based Prediction (using current weather data)
                if st.session_state.model:
                    st.subheader("ML-based Flood Risk Prediction (Current Data):")
                    X_current = st.session_state.current_weather_df[['rainfall_amount_mm', 'water_level_m']]
                    current_predictions = st.session_state.model.predict(X_current)
                    for i, pred in enumerate(current_predictions):
                        location = st.session_state.current_weather_df.iloc[i]['location']
                        risk_level = "Flood Risk" if pred == 1 else "No Flood Risk"
                        st.write(f"  **{location}**: {risk_level}")
                else:
                    st.warning("ML model not trained. Please train the model first.")
    else:
        st.info("Click 'Train Model and Predict' to see the prediction results.")


# --- SMS Alert Tab ---
with tab_sms:
    st.header("4. SMS Alert System (Conceptual)")
    st.write("This is a simulated SMS sending feature. No actual SMS will be sent.")
    
    phone_number = st.text_input("Phone Number (e.g., +1234567890):", value="+91XXXXXXXXXX", key="sms_phone_input")
    sms_message = st.text_area(
        "Message:",
        value="Flood warning: High risk in your area. Take necessary precautions.",
        key="sms_message_input"
    )

    if st.button("Send Simulated SMS Alert", key="send_sms_btn"):
        if not phone_number:
            st.error("Please enter a phone number.")
        else:
            st.info(f"Simulating SMS to: **{phone_number}**")
            st.info(f"Message: **{sms_message}**")
            st.success("SMS simulation complete. (No actual SMS sent)")

# --- Map Visualization Tab ---
with tab_map:
    st.header("5. Map Visualization")
    st.info("Click 'Generate and Open Map' to visualize simulated flood risk on an interactive map. This will open in your default web browser.")
    
    if st.button("Generate and Open Map", key="generate_map_btn"):
        if st.session_state.current_weather_df.empty:
            st.warning("No current weather data to visualize. Please simulate data first in the 'Data Simulation' tab.")
        else:
            # Create a base map centered around a general location in India
            m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

            # Simulate coordinates for locations (for demonstration purposes)
            location_coords = {
                'Haripur': [22.3511, 71.8708], 'Shantipur': [22.9964, 88.6293],
                'Ramnagar': [22.3167, 87.2167], 'Krishnapur': [22.5726, 88.3639],
                'Faizabad': [26.7725, 82.1333], 'Varanasi': [25.3176, 82.9739],
                'Cuttack': [20.4625, 85.8830], 'Guwahati': [26.1445, 91.7362],
                'Patna': [25.5941, 85.1376], 'Surat': [21.1702, 72.8311],
                'Kochi': [9.9312, 76.2673], 'Hyderabad': [17.3850, 78.4867],
                'Chennai': [13.0827, 80.2707], 'Pune': [18.5204, 73.8567],
                'Bhopal': [23.2599, 77.4126], 'Jaipur': [26.9124, 75.7873]
            }

            # Add markers based on simulated current weather data and rule-based risk
            for index, row in st.session_state.current_weather_df.iterrows():
                village = row['location']
                rainfall = row['rainfall_amount_mm']
                water_level = row['water_level_m']
                coords = location_coords.get(village, [20.5937, 78.9629]) # Default if not found

                risk = "Low"
                if rainfall > 50 or water_level > 10:
                    risk = "High"
                elif rainfall > 30 or water_level > 8:
                    risk = "Medium"
                
                marker_color = 'blue' # Default color
                icon_type = 'info-sign' # Default icon

                if risk == 'High':
                    marker_color = 'red'
                    icon_type = 'exclamation-sign'
                elif risk == 'Medium':
                    marker_color = 'orange'
                    icon_type = 'warning-sign'
                elif risk == 'Low':
                    marker_color = 'green'
                    icon_type = 'ok-sign'

                folium.Marker(
                    location=coords,
                    popup=f"<b>{village}</b><br>Rainfall: {rainfall}mm<br>Water Level: {water_level}m<br>Risk: {risk}",
                    tooltip=village,
                    icon=folium.Icon(color=marker_color, icon=icon_type)
                ).add_to(m)

            # Save the map to an HTML file
            map_file = "flood_risk_map.html"
            m.save(map_file)
            
            # Open the HTML file in the default web browser
            webbrowser.open_new_tab(map_file)
            st.success(f"Map generated and saved to **{map_file}**. Attempting to open in your browser...")
            st.warning("Note: If running on a remote server (like Streamlit Community Cloud), the browser tab will open on the server, not your local machine.")

            # Optional: Embed the map directly in Streamlit if you install streamlit-folium
            # from streamlit_folium import st_folium
            # st.subheader("Embedded Map (Requires 'streamlit-folium' library)")
            # st_folium(m, width=700, height=500)
