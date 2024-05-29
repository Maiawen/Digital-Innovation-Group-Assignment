# load the necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
# set the style of the plots
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams['image.cmap'] = 'gray' # grayscale looks better

from pathlib import Path # Path management

import numpy as np
import pandas as pd
import os
# Get the directory of the current file and set as path
path = os.getcwd()

from skimage.io import imread as imread
from skimage.util import montage
from PIL import Image
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
from skimage.color import label2rgb

# image path
image_dir = Path(path + '/feature_food/')
# load allergen ingredient mapping file
mapping_file = Path(path + '/feature_food/clean_list.json')

alleg_df = pd.read_json(mapping_file)
# add the image path into the mapping dataset
alleg_df['image_path'] = alleg_df['image_path'].map(lambda x: image_dir / 'subset' / x) 

print(alleg_df['image_path'].map(lambda x: x.exists()).value_counts())

# show the first 2 rows of the dataset
allergens = alleg_df.columns[3:].tolist()
alleg_df.sample(2)

# load the color features
color_file = Path(path + '/feature_food/color_features.json')
color_feat_df = pd.read_json(color_file)
# add the image path into the color feature dataset
color_feat_df['image_path'] = color_feat_df['image_path'].map(lambda x: image_dir / 'subset' / x) 

# create a dictionary of color features
color_feat_dict = {c_row['image_path']: c_row['color_features'] for _, c_row in color_feat_df.iterrows()}

# add a new color feature column to the mapping dataset
alleg_df['color_features'] = alleg_df['image_path'].map(color_feat_dict.get)
alleg_df.sample(2)

# calculate the correlation matrix of the allergens and draw the heatmap
co_all = np.corrcoef(np.stack(alleg_df[allergens].applymap(lambda x: 1 if x>0 else 0).values, 0).T)
fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(co_all, annot=True, fmt='2.1%', ax=ax1, cmap='RdBu', vmin=-1, vmax=1)
ax1.set_xticklabels(allergens, rotation=90)
ax1.set_yticklabels(allergens)


# package the allergens together
alleg_df['allergy_vec'] = alleg_df[allergens].applymap(lambda x: 1 if x>0 else 0).values.tolist()

# split the dataset into training and validation sets
from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(alleg_df.drop(columns='ingredients_list'), 
                                      test_size=0.1, 
                                      random_state=2019, # set the random seed
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
from sklearn.preprocessing import RobustScaler  # introduce RobustScaler

# create a pipeline for the model, it will make the model first scale the data and then use the Random Forest Regressor to train the model
rf_pipe = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators=15))
rf_pipe.fit(train_x_vec, train_y_vec)

# streamlit app
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
from imageio import imread

@st.cache
def show_model_results(in_model, picture_number=None):
    # Your data loading code here
    # Ensure 'valid_df', 'valid_x_vec', 'valid_y_vec', 'train_df', 'train_x_vec', 'train_y_vec', 'allergens' are defined

    # stack the data in the DataFrame into a NumPy array.
    valid_x_vec = np.stack(valid_df['color_features'].values, 0)
    valid_y_vec = np.stack(valid_df['allergy_vec'], 0)

    x_vec = valid_x_vec
    y_vec = valid_y_vec

    # predict according to the input model
    valid_pred = in_model.predict(x_vec)
    valid_num = picture_number
    
    # visualization
    fig, m_axs = plt.subplots(2, 2, figsize=(10, 10))
    all_rows = []
    ax1 = m_axs[0, 0]
    for i, c_allergen in enumerate(allergens):
        tpr, fpr, _ = roc_curve(y_vec[:, i], valid_pred[:, i]) # calculate the true positive rate and false positive rate
        auc = roc_auc_score(y_vec[:, i], valid_pred[:, i]) # calculate the area under the curve
        acc = accuracy_score(y_vec[:, i], valid_pred[:, i] > 0.5) # calculate the accuracy
        ax1.plot(tpr, fpr, '.-', label='{}: AUC {:0.2f}, Accuracy: {:2.0%}'.format(c_allergen, auc, acc)) # plot the ROC curve
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
    valid_df.iloc[valid_num:valid_num+1].iterrows(), 
    m_axs[1:]):
        c_ax.imshow(imread(c_row['image_path']))
        c_ax.set_title(c_row['title'])
        c_ax.axis('off')
        t_yp = in_model.predict(np.expand_dims(c_row['color_features'], 0))
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
def main():
    st.title('Model Results')
    st.write(f'<span style="font-size:20px;">This is a MVP, so we use the images from the test set as examples.</span>', unsafe_allow_html=True)

    image_paths = valid_df['image_path'].tolist()

    # Display the first 10 images
    num_rows = 2
    num_cols = 5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j + 21  # Starting from index 21
            image = imread(image_paths[index])
            axes[i, j].imshow(image)
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Image {index-20}')

    st.pyplot(fig)

    num_images = 10 

    # User interaction to select image
    st.write(f'<span style="font-size:20px;">Enter the image number you want to analyze.</span>', unsafe_allow_html=True)
    # User input
    choice = st.number_input(f"Range (1-{num_images}): ", min_value=1, max_value=num_images)
    
    # Show model results
    if st.button('Show Results'):
        show_model_results(rf_pipe, choice+20)

if __name__ == "__main__":
    main()
