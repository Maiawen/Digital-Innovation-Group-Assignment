# Digital-Innovation-Group-Assignment
This project (TravellergyFree) was a group assignment for SDC 2023 cohort IM program students in the Digital Innovation course and was completed by Huayra's group. The project contains all the datasets and code files to accomplish this project. We create a minimun viable product (MVP) for allergy travellers and serve for the need of the demand of knowing the allergens in cuisines.

The deployment of Hugging Face link: https://huggingface.co/spaces/Cruicis/TrallergyFree

The purpose of the TrallergyFree app is to provide a reliable and convenient solution for individuals with food allergies, ensuring their safety and peace of mind in various dining situations. The app aims to assist users in identifying allergens in dishes, particularly when they are unable to communicate effectively about their dietary restrictions, such as when traveling abroad. By leveraging advanced image recognition technology, TrallergyFree empowers users to make informed decisions about their food choices, thereby reducing the risk of allergic reactions and enhancing their overall dining experience.

## Main Branch
* app.py is our main file for deploying in Hugging Face which integrate three main functions of our apps using siders.
* requirements.txt is the required libraries to be used in the Hugging Face workspace.

## Dish Allergen Recognition
* app.py shows the single function of dish allergen image recognition.
* image to feature.ipynb is the code for processing image's color features into vectors.
* ingredients to allergies mapping.ipynb is the code for mapping the ingredients with common allergens.
* readme.text contains the data source link.

## Hospital Navigator
* app.py shows the single function of navigation function of hospital and pharmacies.
* yanqihu_hospitals.xlsx is the pre-saved data of the nearby hospitals.
* yanqihu_pharmacies.xlsx is the pre-saved data of the nearby pharmacies.
* zhongguancun_hospitals.xlsx is the pre-saved data of the nearby hospitals.
* zhongguancun_pharmacies.xlsx is the pre-saved data of the nearby pharmacies.

# Menu-allgergen Datadase
* app.py shows the single function of allergen free cusine recommendation.
* data.csv is the dataset we used for creating the query database.

# User Analysis
The following google colab link is an extension analysis about the user reviews' sentiment and further exploration of other future potential functions like translation, allergy chatbot. This is not covered in the group assignment but as a way of brainstorming prospecting ways to improve.
https://colab.research.google.com/drive/1nlYSAM7gMF2Rpk-G-eltGoOFNPwg6rcM?usp=sharing#scrollTo=LAVcT8D4fQA1

