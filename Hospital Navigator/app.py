import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
import os
from dotenv import load_dotenv

import streamlit as st

st.title('Campus Location Maps')

# Selection box for campus
campus = st.selectbox(
    'Select the campus:',
    ('Yanqihu Campus', 'Zhongguancun Campus')
)

if campus == 'Yanqihu Campus':
    st.header('Hospitals and pharmacies within 2km of Yanqihu Campus')

    # User's location for Yanqihu Campus
    latitude = 40.407521
    longitude = 116.68452

    hospital_map = folium.Map(location=[latitude, longitude], zoom_start=12)

    # Add a marker for the user's location
    folium.Marker(
        location=[latitude, longitude],
        popup="User's Location",
        tooltip="User's Location",
        icon=folium.Icon(color='black', icon="info-sign")  # Using built-in icon
    ).add_to(hospital_map)

    df_hospitals = pd.read_excel('yanqihu_hospitals.xlsx', engine='openpyxl')
    st.write("The table below shows hospitals within 2km of you.🏥")
    st.dataframe(df_hospitals)
    # Add hospital markers
    for _, row in df_hospitals.iterrows():
        lon, lat = map(float, row['location'].split(','))
        folium.Marker(
            location=[lat, lon],
            popup=f"{row['name']}, {row['address']}",
            tooltip=row['name'],
            icon=folium.Icon(color='blue', icon="info-sign")  # Using built-in icon
        ).add_to(hospital_map)

    df_pharmacies = pd.read_excel('yanqihu_pharmacies.xlsx', engine='openpyxl')
    st.write("The table below shows pharmacies within 2km of you.💊")
    st.dataframe(df_pharmacies)
    # Add pharmacy markers
    for _, row in df_pharmacies.iterrows():
        lon, lat = map(float, row['location'].split(','))
        folium.Marker(
            location=[lat, lon],
            popup=f"{row['name']}, {row['address']}",
            tooltip=row['name'],
            icon=folium.Icon(color='purple', icon="info-sign")  # Using built-in icon
        ).add_to(hospital_map)

    # Display the map in Streamlit
    st_folium(hospital_map, width=725, height=500)

elif campus == 'Zhongguancun Campus':
    st.header('Hospitals and pharmacies within 2km of Zhongguancun Campus')

    # User's location for Zhongguancun Campus
    latitude = 39.979402
    longitude = 116.3134541

    hospital_map = folium.Map(location=[latitude, longitude], zoom_start=14)

    # Add a marker for the user's location
    folium.Marker(
        location=[latitude, longitude],
        popup="User's Location",
        tooltip="User's Location",
        icon=folium.Icon(color='black', icon="info-sign")  # Using built-in icon
    ).add_to(hospital_map)

    df_hospitals = pd.read_excel('zhongguancun_hospitals.xlsx', engine='openpyxl')
    st.write("The table below shows hospitals within 2km of you.🏥")
    st.dataframe(df_hospitals)
    # Add hospital markers
    for _, row in df_hospitals.iterrows():
        lon, lat = map(float, row['location'].split(','))
        folium.Marker(
            location=[lat, lon],
            popup=f"{row['name']}, {row['address']}",
            tooltip=row['name'],
            icon=folium.Icon(color='blue', icon="info-sign")  # Using built-in icon
        ).add_to(hospital_map)

    df_pharmacies = pd.read_excel('zhongguancun_pharmacies.xlsx', engine='openpyxl')
    st.write("The table below shows pharmacies within 2km of you.💊")
    st.dataframe(df_pharmacies)
    # Add pharmacy markers
    for _, row in df_pharmacies.iterrows():
        lon, lat = map(float, row['location'].split(','))
        folium.Marker(
            location=[lat, lon],
            popup=f"{row['name']}, {row['address']}",
            tooltip=row['name'],
            icon=folium.Icon(color='purple', icon="info-sign")  # Using built-in icon
        ).add_to(hospital_map)

    # Display the map in Streamlit
    st_folium(hospital_map, width=725, height=500)

# Explanation of color keys
st.divider()
st.write("Color Key:\n- Black🖤: Your current location📍\n- Blue💙: Hospitals🏥\n- Purple💜: Pharmacies💊")

st.divider()   


# Streamlit page title
st.title('Enter Your Location')

# Allow users to input latitude and longitude
latitude = st.number_input('Enter your latitude:(eg.39.918058)', value=0.0, format="%.6f")
longitude = st.number_input('Enter your longitude:(eg.116.397026)', value=0.0, format="%.6f")

# Display the inputted location information if valid
if latitude and longitude:
    st.write(f"Your entered location is at latitude {latitude} and longitude {longitude}")

# Load environment variables
load_dotenv()
api_key = os.getenv('api_key')  # Your Amap API key

# User's location
#latitude = 39.993015
#longitude = 116.473168

hospital_map = folium.Map(location=[latitude, longitude], zoom_start=14)

# Add a marker for the user's location
folium.Marker(
    location=[latitude, longitude],
    popup="User's Location",
    tooltip="User's Location",
    icon=folium.Icon(color='black')
).add_to(hospital_map)

# Hospital API URL
url_hospital = f'https://restapi.amap.com/v3/place/around?key={api_key}&location={longitude},{latitude}&types=090100&radius=2000&offset=20&page=1&extensions=all'
# Pharmacy API URL
url_pharmacy = f'https://restapi.amap.com/v3/place/around?key={api_key}&location={longitude},{latitude}&types=090601&radius=2000&offset=20&page=1&extensions=all'

try:
    # Make API request for hospitals
    response_hospital = requests.get(url_hospital)
    response_pharmacy = requests.get(url_pharmacy)  # API request for pharmacies
    
    if response_hospital.status_code == 200 and response_pharmacy.status_code == 200:
        data_hospital = response_hospital.json()
        data_pharmacy = response_pharmacy.json()
        
        # Process hospital data
        if 'pois' in data_hospital:
            df_hospitals = pd.DataFrame(data_hospital['pois'])
            if not df_hospitals.empty:
                df_hospitals = df_hospitals[['name', 'location', 'address']]
                st.write("The table below shows hospitals within 2km of you.🏥")
                st.dataframe(df_hospitals)
                # Add hospital markers
                for _, row in df_hospitals.iterrows():
                    lon, lat = map(float, row['location'].split(','))
                    folium.Marker(
                        location=[lat, lon],
                        popup=f"{row['name']}, {row['address']}",
                        tooltip=row['name'],
                        icon=folium.Icon(color='blue')
                    ).add_to(hospital_map)

        # Process pharmacy data
        if 'pois' in data_pharmacy:
            df_pharmacies = pd.DataFrame(data_pharmacy['pois'])
            if not df_pharmacies.empty:
                df_pharmacies = df_pharmacies[['name', 'location', 'address']]
                st.write("The table below shows pharmacies within 2km of you.💊")
                st.dataframe(df_pharmacies)
                # Add pharmacy markers
                for _, row in df_pharmacies.iterrows():
                    lon, lat = map(float, row['location'].split(','))
                    folium.Marker(
                        location=[lat, lon],
                        popup=f"{row['name']}, {row['address']}",
                        tooltip=row['name'],
                        icon=folium.Icon(color='purple')  # Different color for pharmacies
                    ).add_to(hospital_map)

        # Display the map in Streamlit
        st.title('Nearby Hospitals and Pharmacies')
        st_folium(hospital_map, width=725, height=500)
        st.write("Color Key:\n- Black🖤: Your current location📍\n- Blue💙: Hospitals🏥\n- Purple💜: Pharmacies💊")       
    else:
        st.error("Failed to retrieve data. Check the status codes.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    

    