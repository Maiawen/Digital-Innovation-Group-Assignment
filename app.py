import streamlit as st

# Dish allergen recognition part
# load the necessary libraries
import matplotlib.pyplot as plt
import seaborn

# set the style of the plots
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
seaborn.set_style("whitegrid", {'axes.grid': False})
plt.rcParams['image.cmap'] = 'gray'  # grayscale looks better
from pathlib import Path  # Path management
import numpy as np
import pandas as pd
import os
from skimage.io import imread as imread
from skimage.util import montage
from PIL import Image

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
from skimage.color import label2rgb

# image path
image_dir = Path('')

# load allergen ingredient mapping file
mapping_file = Path('clean_list.json')
alleg_df = pd.read_json(mapping_file)

# add the image path into the mapping dataset
alleg_df['image_path'] = alleg_df['image_path'].map(lambda x: image_dir / 'new' / x)
print(alleg_df['image_path'].map(lambda x: x.exists()).value_counts())
allergens = alleg_df.columns[3:].tolist()
alleg_df.sample(2)
import os
from pathlib import Path

# load the color features
color_file = Path('color_features.json')
color_feat_df = pd.read_json(color_file)
color_feat_df['image_path'] = color_feat_df['image_path'].map(lambda x: image_dir / 'new' / x)

# create a dictionary of color features
color_feat_dict = {c_row['image_path']: c_row['color_features'] for _, c_row in color_feat_df.iterrows()}
# add a new color feature column
alleg_df['color_features'] = alleg_df['image_path'].map(color_feat_dict.get)
alleg_df.sample(2)

# calculate the correlation matrix of the allergens and draw the heatmap
co_all = np.corrcoef(np.stack(alleg_df[allergens].applymap(lambda x: 1 if x > 0 else 0).values, 0).T)
fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
seaborn.heatmap(co_all, annot=True, fmt='2.1%', ax=ax1, cmap='RdBu', vmin=-1, vmax=1)
ax1.set_xticklabels(allergens, rotation=90)
ax1.set_yticklabels(allergens);

# package the allergens together
alleg_df['allergy_vec'] = alleg_df[allergens].applymap(lambda x: 1 if x > 0 else 0).values.tolist()

# split the dataset into training and validation sets
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(alleg_df.drop(columns='ingredients_list'),
                                      test_size=0.1,
                                      random_state=2019,
                                      stratify=alleg_df['allergy_vec'].map(lambda x: x[0:3]))

# reset the index of the training and validation datasets
train_df.reset_index(inplace=True)
valid_df.reset_index(inplace=True)

# show the number of training and validation images
print(train_df.shape[0], 'training images')
print(valid_df.shape[0], 'validation images')

# there are 8687 training images and 966 validation images

# The np.stack() function is used to stack the data in the DataFrame into a NumPy array.
train_x_vec = np.stack(train_df['color_features'].values, 0)
train_y_vec = np.stack(train_df['allergy_vec'], 0)

# load the necessary libraries for the model

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler  # RobustScaler
# create a pipeline for the model, it will make the model first scale the data and then use the Random Forest Regressor to train the model
# rf_pipe = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators=15))
# rf_pipe.fit(train_x_vec, train_y_vec)
# But in this file, In order to shorten the running time of the app, we saved the trained model file in advance, and read the trained model directly here


import joblib

# read trained model result
loaded_rf_pipe = joblib.load('rf_model.joblib')

# streamlit app
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
from imageio import imread


@st.cache_data
def show_model_results(_in_model, picture_number=None):
    # Ensure 'valid_df', 'valid_x_vec', 'valid_y_vec', 'train_df', 'train_x_vec', 'train_y_vec', 'allergens' are defined
    # stack the data in the DataFrame into a NumPy array.

    valid_x_vec = np.stack(valid_df['color_features'].values, 0)
    valid_y_vec = np.stack(valid_df['allergy_vec'], 0)

    x_vec = valid_x_vec
    y_vec = valid_y_vec

    # predict according to the input model
    valid_pred = _in_model.predict(x_vec)
    valid_num = picture_number

    # visualization
    fig, m_axs = plt.subplots(2, 2, figsize=(10, 10))
    all_rows = []
    ax1 = m_axs[0, 0]
    for i, c_allergen in enumerate(allergens):
        tpr, fpr, _ = roc_curve(y_vec[:, i], valid_pred[:, i])
        auc = roc_auc_score(y_vec[:, i], valid_pred[:, i])
        acc = accuracy_score(y_vec[:, i], valid_pred[:, i] > 0.5)
        ax1.plot(tpr, fpr, '.-', label='{}: AUC {:0.2f}, Accuracy: {:2.0%}'.format(c_allergen, auc, acc))
        all_rows += [{'allegen': c_allergen,
                      'prediction': valid_pred[j, i],
                      'class': 'Positive' if y_vec[j, i] > 0.5 else 'Negative'}
                     for j in range(valid_pred.shape[0])]

    # Plotting the predictions of the model versus the true labels
    d_ax = m_axs[0, 1]
    t_yp = np.mean(valid_pred, 0)
    t_y = np.mean(y_vec, 0)
    d_ax.barh(np.arange(len(allergens)) + 0.1, t_yp, alpha=0.5, label='Predicted')
    d_ax.barh(np.arange(len(allergens)) - 0.1, t_y + 0.001, alpha=0.5, label='Ground Truth')
    d_ax.set_xlim(0, 1)
    d_ax.set_yticks(range(len(allergens)))
    d_ax.set_yticklabels(allergens, rotation=0)
    d_ax.set_title('Overall')
    d_ax.legend()

    # Plotting the predictions of the model versus the true labels for the test picture

    ax1.legend()
    for (_, c_row), (c_ax, d_ax) in zip(
            valid_df.iloc[valid_num:valid_num + 1].iterrows(),
            m_axs[1:]):
        c_ax.imshow(imread(c_row['image_path']))
        c_ax.set_title(c_row['title'])
        c_ax.axis('off')
        t_yp = _in_model.predict(np.expand_dims(c_row['color_features'], 0))
        t_y = np.array(c_row['allergy_vec'])
        d_ax.barh(np.arange(len(allergens)) + 0.1, t_yp[0], alpha=0.5, label='Predicted')
        d_ax.barh(np.arange(len(allergens)) - 0.1, t_y + 0.001, alpha=0.5, label='Ground Truth')
        d_ax.set_yticks(range(len(allergens)))
        d_ax.set_yticklabels(allergens, rotation=0)
        d_ax.set_xlim(0, 1)
        d_ax.legend()

    # add the current figures into Streamlit page
    st.pyplot(fig)
    return st.write("Completed")


# Get the image paths
image_paths = valid_df['image_path'].tolist()


# Streamlit app
def recognition():
    st.write(f'<span style="font-size:20px;">We use the images from the test set as examples.</span>',
             unsafe_allow_html=True)

    image_paths = valid_df['image_path'].tolist()

    # Display the first 10 images
    num_rows = 2
    num_cols = 5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            image = imread(image_paths[index])
            axes[i, j].imshow(image)
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Image {index + 1}')

    st.pyplot(fig)

    num_images = 10

    # User interaction to select image
    st.write(f'<span style="font-size:20px;">Enter the image number you want to analyze.</span>',
             unsafe_allow_html=True)

    # User input
    choice = st.number_input(f"Range (1-{num_images}): ", min_value=1, max_value=num_images)

    # Show model results
    if st.button('Show Results'):
        show_model_results(loaded_rf_pipe, choice - 1)


# Menu-allergen database part
# load the libraries we need
import streamlit as st
import pandas as pd
import plotly.express as px  # Streamlined plotting and interactive visualization

# input our dataset

# Create a text element and let customers know the data is loading.
data = pd.read_csv('data.csv')


# set the function to filter the allgergens customers may have
# The mask is created by negating the result of checking whether the 'Allergens' column in the data contains any of the allergens joined by '|'.
# | (pipe symbol) is used for "or" logic when joining the allergens that means the union of selected allergons will be filtered.
def filter_products(allergens):
    # The ~ in front of it negates the result. This is for deleting the food data containing the allergens
    mask = ~data['Allergens'].str.contains('|'.join(allergens), case=False,
                                           na=False)  # case=False shows not considering case, and na=False handles missing values appropriately.
    filtered_data = data[mask]
    return filtered_data


import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
import os
from dotenv import load_dotenv

import streamlit as st

page = st.sidebar.radio("Page",
                        ["Home Page", "Menu-allergen database", "Dish_Allergen_Recognition", "Hospital Navigator"])

if page == "Home Page":
    st.title("TrallergyFree")
    st.image('logo.jpg')
    st.write(
        "The purpose of the TrallergyFree app is to provide a reliable and convenient solution for individuals with food allergies, ensuring their safety and peace of mind in various dining situations. The app aims to assist users in identifying allergens in dishes, particularly when they are unable to communicate effectively about their dietary restrictions, such as when traveling abroad. By leveraging advanced image recognition technology, TrallergyFree empowers users to make informed decisions about their food choices, thereby reducing the risk of allergic reactions and enhancing their overall dining experience.")


elif page == "Menu-allergen database":
    st.title("Menu-allergen database")
    # st.multiselect is a widget in the streamlit library. It is used to create a multiple selection dropdown menu.
    # It allows users to select multiple options from a given list of choices which are collected by unique

    allergens = st.multiselect("What foods are you allergic to?",
                               [
                                   "Wheat",
                                   "Nuts",
                                   "Sesame",
                                   "Soy",
                                   "Mustard",
                                   "Celery",
                                   "Dairy",
                                   "Egg",
                                   'Others',
                               ]
                               )

    # When customers select one or more allergens, our interactive page will recommend the dishes that don't contain the allergens
    if allergens:
        # Filter and display the recommendations
        recommended_products = filter_products(allergens)
        if not recommended_products.empty:  # case when there is food not containing the selected allergens
            # Group products by type and create a tab for each type
            st.write("Below is a list of allergen-free products. Please make your selection✅")
            # st.write("🔸You can choose the food you want to eat and select it in the box⬜🔸")
            product_types = recommended_products['type'].unique()  # de-weighted product types for choosing
            tabs = st.tabs([str(pt) for pt in product_types])  # show tabs function

            # create an empty dataframe for all selected products
            all_selected_products = pd.DataFrame()
            for i, tab in enumerate(tabs):  # interactive function with tabs for products
                with tab:
                    type_specific_products = recommended_products[recommended_products['type'] == product_types[i]]
                    type_specific_products[
                        'selected'] = False  # This adds a 'selected' column and initializes it to False.
                    for j, row in type_specific_products.iterrows():
                        # create columns and display the product name and allergens
                        cols = st.columns([1, 1, 1])
                        cols[0].write(row['Products'])
                        cols[1].write(row['Allergens'])
                        selected = cols[2].checkbox("Select",
                                                    key=f"select_{i}_{j}")  # set a checkbox for selection. The key is set uniquely for each row and tab
                        type_specific_products.at[
                            j, 'selected'] = selected  # Updates the 'selected' value in the type-specific product based on the checkbox state
                    selected_products = type_specific_products[type_specific_products['selected']]
                    all_selected_products = pd.concat([all_selected_products,
                                                       selected_products])  # row concatenation, the axis in pd.concat is 0 in default
            #display the nutritional information of selected products 
            if not all_selected_products.empty:  # case when customers select products
                st.divider()
                st.write("Below is the nutritional information for the selected products:")
                st.dataframe(all_selected_products[
                                 ['type', 'Products', 'Allergens', 'Total Energy (Kcal)', 'Carbohydrate (g)',
                                  'Protein (g)', 'Total Fat (g)', 'Sodium (g)', 'Sugar (g)']])
                nutritional_totals = all_selected_products[
                    ['Total Energy (Kcal)', 'Carbohydrate (g)', 'Protein (g)', 'Total Fat (g)', 'Sodium (g)',
                     'Sugar (g)']].sum()

                # Display metrics for nutritional totals in a 3x2 grid
                st.divider()
                st.write("Below are the nutritional totals for the selected products:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Energy (Kcal)", f"{nutritional_totals['Total Energy (Kcal)']} kcal")
                    st.metric("Total Fat", f"{nutritional_totals['Total Fat (g)']} g")
                with col2:
                    st.metric("Carbohydrates", f"{nutritional_totals['Carbohydrate (g)']} g")
                    st.metric("Sodium", f"{nutritional_totals['Sodium (g)']} g")
                with col3:
                    st.metric("Protein", f"{nutritional_totals['Protein (g)']} g")
                    st.metric("Sugar", f"{nutritional_totals['Sugar (g)']} g")

                st.divider()
                st.write("Below are the nutrient proportions of the selected products:")
                # Create pie charts for each nutritional metric in 2x3 grid
                # If they are multi selections, each product will contribute proportion in the pie chart
                metrics = ['Total Energy (Kcal)', 'Carbohydrate (g)', 'Protein (g)', 'Total Fat (g)', 'Sodium (g)',
                           'Sugar (g)']
                for i in range(0, len(metrics), 2):
                    cols = st.columns(2)  # Define three columns
                    for j in range(2):
                        if i + j < len(metrics):  # Check if the current index is within the metrics array
                            with cols[j]:
                                metric = metrics[i + j]
                                fig = px.pie(all_selected_products, values=metric, names='Products',
                                             title=f'Proportion of {metric}',
                                             width=300, height=300)  # Adjust width and height if necessary
                                st.plotly_chart(fig)
            else:
                st.write("No selected products in this category.")  # case when no product is selected
        else:
            st.write(
                "No products found without the specified allergens.")  # case when there is no food without selected allergens

elif page == "Dish_Allergen_Recognition":
    st.title('Dish Allergen Recognition')
    if __name__ == "__main__":
        recognition()
elif page == "Hospital Navigator":
    st.title("Hospital Navigator")
    # Selection box for campus
    campus = st.selectbox(
        'Select the campus:',
        ('Yanqihu Campus', 'Zhongguancun Campus')
    )
    # Yanqihu campus part
    if campus == 'Yanqihu Campus':
        st.header('Hospitals and pharmacies within 2km of Yanqihu Campus')

        # User's location for Yanqihu Campus
        latitude = 40.407521
        longitude = 116.68452
        # Create a map centered at the user's location
        hospital_map = folium.Map(location=[latitude, longitude], zoom_start=12)

        # Add a marker for the user's location
        folium.Marker(
            location=[latitude, longitude],
            popup="User's Location",
            tooltip="User's Location",
            icon=folium.Icon(color='black', icon="info-sign")  # Using built-in icon
        ).add_to(hospital_map)
        #import the dataset of hospitals nearby yanqihu campus
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
        #import the dataset of pharmacies nearby yanqihu campus
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
        # Create a map centered at the user's location
        hospital_map = folium.Map(location=[latitude, longitude], zoom_start=14)

        # Add a marker for the user's location
        folium.Marker(
            location=[latitude, longitude],
            popup="User's Location",
            tooltip="User's Location",
            icon=folium.Icon(color='black', icon="info-sign")  # Using built-in icon
        ).add_to(hospital_map)
        #import the dataset of hospitals nearby zhongguancun campus
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
        #import the dataset of pharmacies nearby zhongguancun campus
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
    # Create a map centered at the user's location
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