# Python Exploratory Data Analysis on Spotify 2023 Dataset

``` python

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
spotify_data = pd.read_csv('spotify-2023.csv', encoding='latin1')
```


``` python

# Display the number of rows and columns in the dataset
num_rows, num_columns = spotify_data.shape
print(f"The dataset contains {num_rows} rows and {num_columns} columns.\n")

# Display the data types of each column
print("Data types of each column:")
print(spotify_data.dtypes)
print("\n")

# Check for missing values
missing_values = spotify_data.isnull().sum()
print("Number of missing values in each column:")
print(missing_values[missing_values > 0])  # Display only columns with missing values

```

``` python

# Fix data types for columns
# Specify columns that should be numeric but might be stored as strings, and vice versa
numeric_columns = [
    'artist_count', 
    'released_year', 
    'released_month', 
    'released_day', 
    'in_spotify_playlists', 
    'in_spotify_charts', 
    'streams', 
    'in_apple_playlists', 
    'in_apple_charts', 
    'in_deezer_playlists', 
    'in_deezer_charts', 
    'in_shazam_charts', 
    'bpm', 
    'danceability_%', 
    'valence_%', 
    'energy_%', 
    'acousticness_%', 
    'instrumentalness_%', 
    'liveness_%', 
    'speechiness_%'
]

string_columns = [
    'track_name', 
    'artist(s)_name', 
    'key', 
    'mode'
]

# Convert columns to appropriate data types
for col in numeric_columns:
    spotify_data[col] = pd.to_numeric(spotify_data[col], errors='coerce')  # Convert to numeric, invalid parsing set as NaN

# Step 2: Handle missing values
# Fill missing values for string columns: use 'unknown'
for col in string_columns:
    spotify_data[col] = spotify_data[col].fillna('unknown')  # Fill missing values with 'unknown'

# Fill missing values: use 0 for numerical columns
spotify_data[numeric_columns] = spotify_data[numeric_columns].fillna(0)

# Step 3: Convert float columns to integers
for col in numeric_columns:
    spotify_data[col] = spotify_data[col].astype(int)  # Convert to int

# Step 4: Additional Cleaning (if needed)
# Remove any duplicate rows to ensure uniqueness
spotify_data.drop_duplicates(inplace=True)

# Check for any remaining missing values after cleaning
remaining_missing = spotify_data.isnull().sum()

# Output for remaining missing values
if remaining_missing.sum() == 0:
    print("Remaining missing values after cleaning: None")
else:
    print("Remaining missing values after cleaning:")
    print(remaining_missing[remaining_missing > 0])  # Display only columns with missing values
print("\n")

# Confirm data types after cleaning
print("Data types after cleaning:")
print(spotify_data.dtypes)
print("\n")

# Save the cleaned data to a new CSV file
cleaned_file_path = 'spotify-2023-cleaned.csv'
spotify_data.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}.")

```
