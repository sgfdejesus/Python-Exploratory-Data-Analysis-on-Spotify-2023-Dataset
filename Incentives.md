# 🎶 Exploratory Data Analysis: Most Streamed Spotify Songs 2023 🎶

This project performs an **Exploratory Data Analysis (EDA)** on a dataset detailing popular tracks on Spotify in 2023. The goal is to analyze, visualize, and interpret data to extract insights about trends, characteristics, and factors that contribute to a song's popularity.

Dataset source: [Kaggle - Most Streamed Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023)

---

## 📈 Analysis Workflow

- **Data Loading tapos overview then Cleaning**: Load the dataset, handle missing values, and confirm data types.
- **Exploratory Analysis**: Perform descriptive statistics and initial visualizations to gain insights.
- **In-Depth Analysis**: Dive deeper into specific aspects like **temporal trends** and **platform comparisons**.
- **Insight Generation**: Summarize findings and provide recommendations.

---

``` python
# Import libraries for data manipulation, visualization, and analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Spotify 2023 dataset, ensuring proper encoding for special characters
spotify_data = pd.read_csv('spotify-2023.csv', encoding='latin1')
```

### 1. Overview of Dataset
   - **Dataset Size:** How many rows and columns?
   - **Data Types:** What are the types of each column? Any **missing values**?

``` python
# Get the dimensions of the dataset (number of rows and columns)
num_rows, num_columns = spotify_data.shape
print(f"The dataset contains {num_rows} rows and {num_columns} columns.\n")

# Display data types of each column to understand the dataset structure
print("Data types of each column:")
print(spotify_data.dtypes)
print("\n")

# Check for missing values in each column and display only columns with missing values
missing_values = spotify_data.isnull().sum()
print("Columns with missing values and their counts:")
print(missing_values[missing_values > 0])
```

Data Cleaning 
   - According to the Overview of the Dataset, it was found that some columns had incorrect data types, such as numeric columns stored as objects and missing values scattered throughout. With that, the data cleaning process was essential to ensure consistency by converting these columns to their appropriate types, handling missing values by filling them with zeros or unknown where necessary, and removing duplicate entries. This approach improves the accuracy and reliability of the dataset for further analysis.

``` python
# Define columns that should be converted to numeric data types
numeric_columns = ['artist_count', 'released_year', 'released_month', 'released_day', 'in_spotify_playlists', 
                'in_spotify_charts', 'streams', 'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists', 
                'in_deezer_charts', 'in_shazam_charts', 'bpm', 'danceability_%', 'valence_%', 
                'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']

# Define columns that should be treated as strings
string_columns = ['track_name', 'artist(s)_name', 'key', 'mode']

# Convert specified columns to numeric, coercing errors to NaN (for invalid data)
for col in numeric_columns:
    spotify_data[col] = pd.to_numeric(spotify_data[col], errors='coerce')  # Convert to numeric, invalid parsing set as NaN

# Fill missing values in numeric columns with 0 to ensure completeness
spotify_data[numeric_columns] = spotify_data[numeric_columns].fillna(0)

# Fill missing values in string columns with 'unknown' to avoid NaN values
for col in string_columns:
    spotify_data[col] = spotify_data[col].fillna('unknown')  # Fill missing values with 'unknown'

# Convert float columns (where applicable) in numeric data to integers
for col in numeric_columns:
    spotify_data[col] = spotify_data[col].astype(int)  # Convert to int

# Remove duplicate rows to maintain unique entries in the dataset
spotify_data.drop_duplicates(inplace=True)

# Check for any remaining missing values after cleaning
remaining_missing = spotify_data.isnull().sum()

# Display any remaining missing values or confirm that none remain
if remaining_missing.sum() == 0:
    print("Remaining missing values after cleaning: None")
else:
    print("Remaining missing values after cleaning:")
    print(remaining_missing[remaining_missing > 0])  # Display only columns with missing values
print("\n")

# Verify data types of each column after cleaning
print("Data types after cleaning:")
print(spotify_data.dtypes)
print("\n")

# Save the cleaned dataset to a new CSV file
cleaned_file_path = 'spotify-2023-cleaned.csv'
spotify_data.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}.")
```

### 2. Basic Descriptive Statistics
   - **Streams Statistics:** What are the **mean, median, and standard deviation** of the streams?
   - **Released Year & Artist Count:** What is the distribution? Are there any **trends** or **outliers**?

### 3. Top Performers
   - **Top Tracks:** Which track has the highest number of streams? List the **top 5 most streamed** tracks.
   - **Frequent Artists:** Who are the **top 5 most frequent artists** based on track count?

### 4. Temporal Trends
   - **Yearly Trends:** Analyze the **trend** in the number of tracks released annually.
   - **Monthly Patterns:** Does the release pattern vary by month? Which month has the most releases?

### 5. Genre and Music Characteristics
   - **Streams vs. Attributes:** Examine correlation between **streams** and musical attributes like **bpm, danceability_%, energy_%**.
   - **Attribute Relationships:** Is there a correlation between **danceability_%** and **energy_%**? How about **valence_%** and **acousticness_%**?

### 6. Platform Popularity
   - **Platform Comparison:** Compare track counts in **Spotify Playlists, Deezer Playlists**, and **Apple Playlists**. Which platform favors the most popular tracks?

### 7. Advanced Analysis
   - **Key & Mode Patterns:** Based on **streams**, are there patterns among tracks with similar **key** or **mode** (Major vs. Minor)?
   - **Frequent Artists in Playlists/Charts:** Do certain **artists** consistently appear in more playlists or charts?

---

## 📋 General Guidelines

1. **Dataset Familiarization**  
   - Inspect the dataset structure, checking for **missing values** and **data types**.
   - Conduct an initial exploration to understand the **features** available.

2. **Summary Statistics**  
   - Present key metrics such as **number of streams**, **release dates**, and **musical attributes** (e.g., BPM, danceability).

3. **Visualizations**  
   - Utilize visualizations (e.g., bar charts, histograms, scatter plots) to identify trends. Ensure **clear labeling** for interpretability.

4. **Correlation Analysis**  
   - Examine relationships between **streams** and attributes like **tempo, energy**, or **playlist appearances**.
   - Provide insights and recommendations based on these analyses.

5. **Insights and Recommendations**  
   - Summarize insights about tracks, artists, or musical trends that could help explain a track's popularity.

---


