# load the libraries we need
import streamlit as st
import pandas as pd
import plotly.express as px # Streamlined plotting and interactive visualization
# set the title for our database
st.title('Menu-allergen database')
# input our dataset

# Create a text element and let customers know the data is loading.
data = pd.read_csv('data.csv')

# set the function to filter the allgergens customers may have
# The mask is created by negating the result of checking whether the 'Allergens' column in the data contains any of the allergens joined by '|'.
# | (pipe symbol) is used for "or" logic when joining the allergens that means the union of selected allergons will be filtered.
def filter_products(allergens):
    # The ~ in front of it negates the result. This is for deleting the food data containing the allergens
    mask = ~data['Allergens'].str.contains('|'.join(allergens), case=False, na=False) # case=False shows not considering case, and na=False handles missing values appropriately.
    filtered_data = data[mask]
    return filtered_data

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
    if not recommended_products.empty: # case when there is food not containing the selected allergens
        # Group products by type and create a tab for each type
        st.write("Below is a list of allergen-free products. Please make your selection✅")
        #st.write("🔸You can choose the food you want to eat and select it in the box⬜🔸")
        product_types = recommended_products['type'].unique() # de-weighted product types for choosing
        tabs = st.tabs([str(pt) for pt in product_types]) # show tabs function

        # create an empty dataframe for all selected products
        all_selected_products = pd.DataFrame()
        for i, tab in enumerate(tabs): # interactive function with tabs for products
            with tab:
                type_specific_products = recommended_products[recommended_products['type'] == product_types[i]]
                type_specific_products['selected'] = False # This adds a 'selected' column and initializes it to False.
                for j, row in type_specific_products.iterrows():
                    # create columns and display the product name and allergens
                    cols = st.columns([1, 1, 1]) 
                    cols[0].write(row['Products'])
                    cols[1].write(row['Allergens'])
                    selected = cols[2].checkbox("Select", key=f"select_{i}_{j}") # set a checkbox for selection. The key is set uniquely for each row and tab
                    type_specific_products.at[j, 'selected'] = selected # Updates the 'selected' value in the type-specific product based on the checkbox state
                selected_products = type_specific_products[type_specific_products['selected']]
                all_selected_products = pd.concat([all_selected_products, selected_products]) # row concatenation, the axis in pd.concat is 0 in default

        if not all_selected_products.empty: # case when customers select products
            st.divider()
            st.write("Below is the nutritional information for the selected products:")
            st.dataframe(all_selected_products[['type', 'Products', 'Allergens', 'Total Energy (Kcal)', 'Carbohydrate (g)', 'Protein (g)', 'Total Fat (g)', 'Sodium (g)', 'Sugar (g)']])
            nutritional_totals = all_selected_products[['Total Energy (Kcal)', 'Carbohydrate (g)', 'Protein (g)', 'Total Fat (g)', 'Sodium (g)', 'Sugar (g)']].sum()
            
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
            # Create pie charts for each nutritional metric in 3x2 grid
            # If they are multi selections, each product will contribute proportion in the pie chart
            metrics = ['Total Energy (Kcal)', 'Carbohydrate (g)', 'Protein (g)', 'Total Fat (g)', 'Sodium (g)', 'Sugar (g)']
            for i in range(0, len(metrics), 2):
                cols = st.columns(2)  # Define three columns
                for j in range(2):
                    if i + j < len(metrics):  # Check if the current index is within the metrics array
                        with cols[j]:
                            metric = metrics[i + j]
                            fig = px.pie(all_selected_products, values=metric, names='Products', title=f'Proportion of {metric}', 
                                        width=300, height=300)  # Adjust width and height if necessary
                            st.plotly_chart(fig)
        else:
            st.write("No selected products in this category.") # case when no product is selected
    else:
        st.write("No products found without the specified allergens.") # case when there is no food without selected allergens
