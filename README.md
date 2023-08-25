# Predicting_Accidents_in_Manufacturing_Plants

Welcome to the "Predicting Accidents in Manufacturing Plants" repository! This repository contains code and resources related to the project aimed at predicting and analyzing accident severity in manufacturing plant environments using data analytics and machine learning techniques.


## Project Overview

Accidents in manufacturing plants can have serious consequences. This project focuses on leveraging data analytics and machine learning to predict accident severity in manufacturing environments. By analyzing historical accident data and implementing advanced modeling techniques, this project aims to provide insights that can help prevent and mitigate accidents.


## Data: Industrial Safety and Health Analytics Database

Industrial Safety and Health Analytics Database

About Dataset:
Welcome to the Industrial Safety and Health Analytics Database. This dataset provides valuable insights into industrial labor accidents and safety incidents from 12 different plants across 3 different countries, including one of the largest industries in Brazil and the world. The urgency to understand and prevent accidents in manufacturing plants, where employees sometimes face injuries and fatalities, drives our commitment to sharing this data with the community. By exploring this dataset, we hope to uncover new insights that can contribute to safer working environments.

Contents:
The database comprises records of accidents, where each entry represents an occurrence of an accident within the manufacturing plants. The dataset features a range of informative columns, offering details about each incident:

- Data: Timestamp or time/date information when the accident occurred.
- Countries: Anonymized information about the country where the accident took place.
- Local: Anonymized city information corresponding to the location of the manufacturing plant.
- Industry Sector: Indicates the sector to which the plant belongs.
- Accident Level: Ranges from I to VI, representing the severity of the accident (I indicates not severe, while VI indicates very severe).
- Potential Accident Level: Corresponds to the possible severity of the accident based on various factors.
- Genre: Specifies the gender of the affected person (male or female).
- Employee or Third Party: Identifies whether the injured individual is an employee or a third party.
- Critical Risk: Provides a description of the critical risk associated with the accident.
- Description: Offers a detailed account of how the accident occurred.

Context:
The dataset provides a unique opportunity to delve into real-world accident data from manufacturing plants, allowing us to analyze and address the safety concerns faced by employees and third parties. By sharing this dataset, we invite the community to collaborate, explore, and gain insights that can lead to safer industrial environments.

Download the Dataset:
You can download the dataset here: https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database?select=IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv


## Repository Contents

This repository contains the following files and directories:

- `EDA_Accidents_Industries.py`:
    -   Exploratory Data Analysis (EDA) notebook for analyzing accidents and industries data.
- `Data_Preprocessing.py`:
    -   Data preprocessing notebook for cleaning and preparing the accident data.
- `Data_Augmentation_SMOTE.py`:
    -   Notebook for applying Synthetic Minority Over-sampling Technique (SMOTE) for data augmentation.
- `Data_Augmentation_Synonym_Replacement.py`:
    -   Notebook for applying synonym replacement for data augmentation.
- `Pycaret_ML_SMOTE_equal_weights.py`:
    -   Notebook implementing machine learning models using PyCaret and SMOTE technique.
- `FSL_BiLSTM_Glove_Synonym_Replacement.py`:
    -   Notebook for Few-Shot Learning using Bi-LSTM, GloVe embeddings, and synonym replacement.
- `data/`:
    -   Directory containing the raw and preprocessed accident data.

 

## Usage
  - Clone the repository to your local machine using git clone.
  - Open and run the Jupyter notebooks to explore and analyze the code.
  - Experiment with different preprocessing techniques, augmentation methods, and machine learning models to predict accident severity.

## Thank You for Visiting!

Thank you for taking the time to explore the "Predicting Accidents in Manufacturing Plants" repository. Your interest and engagement in this project mean a lot. If you find any value, insights, or inspiration from this work, we consider it a success.

We believe that by leveraging data analytics and machine learning, we can contribute to the improvement of safety measures and accident prevention in manufacturing environments. Your feedback, suggestions, and contributions are greatly appreciated as we strive to make a positive impact.

Feel free to explore the notebooks, experiment with the code, and provide your insights. If you have any questions, ideas, or feedback, don't hesitate to share them. Your participation is a valuable part of this journey.

Once again, thank you for your interest and support!

Best regards,

Malvica Lewis
