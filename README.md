# 🎶 Exploratory Data Analysis: Most Streamed Spotify Songs 2023 🎶

This project performs an **Exploratory Data Analysis (EDA)** on a dataset detailing popular tracks on Spotify in 2023. The goal is to analyze, visualize, and interpret data to extract insights about trends, characteristics, and factors that contribute to a song's popularity.

Dataset source: [Kaggle - Most Streamed Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023)

---

## 🗃️ Table of Contents

- [📥 Data Loading](#-data-loading)
- [🔍 Overview of Dataset](#-overview-of-dataset)
   - Dataset Size
   - Data Types
- [🧹 Data Cleaning](#-data_cleaning)
- [📊 Basic Descriptive Statistics](#-basic-descriptive-statistics)
   - Streams Statistics
   - Released Year & Artist Count
- [🏆 Top Performers](#-top-performers)
   - Top Tracks
   - Frequent Artists
- [📅 Temporal Trends](#-temporal-trends)
   - Yearly Trends
   - Monthly Patterns
- [🎼 Genre and Music Characteristics](#-genre-and-music-characteristics)
   - Streams vs. Attributes
   - Attribute Relationships
- [🌐 Platform Popularity](#-platform-popularity)
   - Platform Comparison
- [💡 Advanced Analysis](#-advanced-analysis)
   - Key & Mode Patterns
   - Frequent Artists in Playlists/Charts
    
---

## 📈 Analysis Workflow (will change pa)

- **Data Loading, Overview, and Cleaning**: Load the dataset, handle missing values, and confirm data types.
- **Exploratory Analysis**: Perform descriptive statistics and initial visualizations to gain insights.
- **In-Depth Analysis**: Dive deeper into specific aspects like **temporal trends** and **platform comparisons**.
- **Insight Generation**: Summarize findings and provide recommendations.

---

### 📥 Data Loading

``` python
# Import necessary libraries for data manipulation, analysis, and visualization
import pandas as pd             
import numpy as np    
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Spotify dataset into a pandas DataFrame, specifying the encoding to handle special characters
spotify_data = pd.read_csv('spotify-2023.csv', encoding='latin1')

# Display the first few rows of the loaded dataset to quickly inspect its structure and contents
display(spotify_data)
```

![Untitled design](https://github.com/user-attachments/assets/31d51e71-557f-43e5-a189-5d27c1fd151f)

### 🔍 Overview of Dataset
   - **Dataset Size:** How many rows and columns?
   - **Data Types:** What are the types of each column? Any **missing values**?

``` python
# Retrieve the number of rows and columns in the dataset to understand its size
num_rows, num_columns = spotify_data.shape

# Display the dataset dimensions in a user-friendly message format
print(f"The dataset contains {num_rows} rows and {num_columns} columns.\n")

# Print the data type of each column, which helps identify the type of data stored and informs potential transformations
print("Data types of each column:")
print(spotify_data.dtypes)
print("\n")

# Check for missing values in each column and count them, giving insights into data completeness
missing_values = spotify_data.isnull().sum()

# Display columns that have missing values and their respective counts, if any
print("Columns with missing values and their counts:")
print(missing_values[missing_values > 0])
```

<img width="1241" alt="Screenshot 2024-11-07 at 12 12 31 PM" src="https://github.com/user-attachments/assets/fe80cc10-62b5-4c4a-a2d4-b4a4e67901ca">

### 🧹 Data Cleaning

``` python
# Define the columns expected to have numerical data for conversion and cleaning
numeric_columns = ['artist_count', 'released_year', 'released_month', 'released_day', 
                   'in_spotify_playlists', 'in_spotify_charts', 'streams', 
                   'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists', 
                   'in_deezer_charts', 'in_shazam_charts', 'bpm', 'danceability_%', 
                   'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 
                   'liveness_%', 'speechiness_%']

# Define the columns expected to contain string data
string_columns = ['track_name', 'artist(s)_name', 'key', 'mode']

# Convert numeric columns to numeric data types, coercing errors to NaN
spotify_data[numeric_columns] = spotify_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Fill NaN values in numeric columns with 0 to handle missing data
spotify_data[numeric_columns] = spotify_data[numeric_columns].fillna(0)

# Fill NaN values in string columns with 'unknown' to handle missing textual data
spotify_data[string_columns] = spotify_data[string_columns].fillna('unknown')

# Convert numeric columns to integers after handling missing values
spotify_data[numeric_columns] = spotify_data[numeric_columns].astype(int)

# Remove any duplicate rows from the dataset to ensure data uniqueness
spotify_data.drop_duplicates(inplace=True)

# Convert 'released_month' to numeric and map month numbers to names
spotify_data['released_month'] = pd.to_numeric(spotify_data['released_month'], errors='coerce')
month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
               5: 'May', 6: 'June', 7: 'July', 8: 'August',
               9: 'September', 10: 'October', 11: 'November', 12: 'December'}

# Map month numbers to names, defaulting to 'unknown' for invalid months
spotify_data['released_month'] = spotify_data['released_month'].map(month_names).fillna('unknown')

# Check if any missing values remain after cleaning
remaining_missing = spotify_data.isnull().sum()
if remaining_missing.sum() == 0:
    print("Remaining missing values after cleaning: None")
else:
    print("Remaining missing values after cleaning:")
    print(remaining_missing[remaining_missing > 0])

# Display the data types of each column after cleaning, confirming consistency
print("\nData types after cleaning:")
print(spotify_data.dtypes)
print("\n")

# Save the cleaned dataset to a new CSV file, confirming successful data cleaning and export
cleaned_file_path = 'spotify-2023-cleaned.csv'
spotify_data.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}.")
```

<img width="1241" alt="Screenshot 2024-11-07 at 12 12 49 PM" src="https://github.com/user-attachments/assets/0762e09a-8dba-403f-9c9c-7f68a7625294">

### 📊 Basic Descriptive Statistics
   - **Streams Statistics:** What are the **mean, median, and standard deviation** of the streams?
   - **Released Year & Artist Count:** What is the distribution? Are there any **trends** or **outliers**?

``` python
# Load the cleaned Spotify dataset
spotify_data_cleaned = pd.read_csv('spotify-2023-cleaned.csv', encoding='latin1')

# Calculate summary statistics for 'streams' column (mean, median, std)
streams_mean = spotify_data_cleaned['streams'].mean()
streams_median = spotify_data_cleaned['streams'].median()
streams_std = spotify_data_cleaned['streams'].std()

# Display the summary statistics
print("\nStreams Summary Statistics")
print(f"Mean: {streams_mean:,.2f}")
print(f"Median: {streams_median:,.2f}")
print(f"Standard Deviation: {streams_std:,.2f}\n")

# Define a pastel color for consistency in plots
pastel_color = '#B2E0E6'

# Create a 2x2 grid of subplots with high resolution for clear visualization
fig, axs = plt.subplots(2, 2, figsize=(20, 10), dpi=300)

# Plot distribution of released years using a barplot
released_year_counts = spotify_data_cleaned['released_year'].value_counts().sort_index()
sns.barplot(x=released_year_counts.index, y=released_year_counts.values, ax=axs[0, 0], color=pastel_color, edgecolor='#7a7a7a')
axs[0, 0].set_title("Distribution of Released Year", fontsize=18, fontweight='semibold', color='#000000')
axs[0, 0].set_xlabel('Released Year', fontsize=15, color='#000000')
axs[0, 0].set_ylabel('Frequency', fontsize=15, color='#000000')
axs[0, 0].set_xticks(range(len(released_year_counts.index)))
axs[0, 0].set_xticklabels(released_year_counts.index, rotation=60, ha='right')
axs[0, 0].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

# Plot distribution of artist counts using a barplot
artist_count_counts = spotify_data_cleaned['artist_count'].value_counts().sort_index()
sns.barplot(x=artist_count_counts.index, y=artist_count_counts.values, ax=axs[0, 1], color=pastel_color, edgecolor='#7a7a7a')
axs[0, 1].set_title("Distribution of Artist Count", fontsize=18, fontweight='semibold', color='#000000')
axs[0, 1].set_xlabel('Artist Count', fontsize=15, color='#000000')
axs[0, 1].set_ylabel('Frequency', fontsize=15, color='#000000')
axs[0, 1].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

# Plot boxplot for 'released_year' to identify the spread of data
sns.boxplot(data=spotify_data_cleaned, x='released_year', color=pastel_color, ax=axs[1, 0])
axs[1, 0].set_title('Box Plot of Released Year', fontsize=18, fontweight='semibold', color='#000000')
axs[1, 0].set_xlabel('Released Year', fontsize=15, color='#000000')
axs[1, 0].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

# Plot boxplot for 'artist_count' to identify the spread of data
sns.boxplot(data=spotify_data_cleaned, x='artist_count', color=pastel_color, ax=axs[1, 1])
axs[1, 1].set_title('Box Plot of Artist Count', fontsize=18, fontweight='semibold', color='#000000')
axs[1, 1].set_xlabel('Artist Count', fontsize=15, color='#000000')
axs[1, 1].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

# Adjust layout to prevent overlapping of subplots
plt.tight_layout()
plt.show()

# Function to calculate outliers based on the IQR method
def calculate_outliers(data, column):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for the column
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    
    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers as values outside the bounds
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    # Return the unique outliers as a sorted list
    return sorted(outliers[column].unique().tolist())

# Calculate outliers for 'released_year' and 'artist_count'
unique_year_outliers = calculate_outliers(spotify_data_cleaned, 'released_year')
unique_artist_outliers = calculate_outliers(spotify_data_cleaned, 'artist_count')

# Print the identified outliers for both columns
print(f'Outliers for Released Year:')
print(unique_year_outliers)
print(f'\nOutliers for Artist Count:')
print(unique_artist_outliers)
```

<img width="1241" alt="Screenshot 2024-11-07 at 12 13 01 PM" src="https://github.com/user-attachments/assets/82ea0172-b6e8-4957-a094-5dc7fd31eadd">

### 🏆 Top Performers
   - **Top Tracks:** Which track has the highest number of streams? List the **top 5 most streamed** tracks.
   - **Frequent Artists:** Who are the **top 5 most frequent artists** based on track count?

``` python
# Split the 'artist(s)_name' column by commas and explode it to get each artist as a separate row
artists_expanded = spotify_data_cleaned['artist(s)_name'].str.split(',').explode().str.strip()

# Get the top 5 most frequent artists based on the count of their appearances
top_artists = artists_expanded.value_counts().nlargest(5)

# Convert the top artists series into a DataFrame for better readability in the plot
top_artists_df = top_artists.reset_index().rename(columns={'index': 'artist(s)_name', 0: 'track_count'})

# Set up a 2-row subplot with large figure size for better visibility and high resolution
fig, axs = plt.subplots(2, 1, figsize=(20, 9.75), dpi=300)

# Plot a horizontal bar chart for the top 5 most streamed tracks
axs[0].barh(top_streamed_tracks['track_name'], top_streamed_tracks['streams'], color=pastel_color, edgecolor='black')
axs[0].set_title('Top 5 Most Streamed Tracks', fontsize=18, fontweight='semibold', color='#000000')
axs[0].set_xlabel('Number of Streams', fontsize=15, color='#000000')
axs[0].invert_yaxis() 
axs[0].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

# Add stream count labels on each bar for clarity
for i, bar in enumerate(axs[0].patches):
    axs[0].text(bar.get_width() / 1.06, bar.get_y() + bar.get_height() / 2, 
                f'{int(bar.get_width()):,}', va='center', ha='center', fontsize=12)

# Plot a horizontal bar chart for the top 5 most frequent artists based on the number of tracks
axs[1].barh(top_artists_df['artist(s)_name'], top_artists_df['count'], color=pastel_color, edgecolor='black')
axs[1].set_title('Top 5 Most Frequent Artists Based on the Number of Tracks', fontsize=18, fontweight='semibold', color='#000000')
axs[1].set_xlabel('Number of Tracks', fontsize=15, color='#000000')
axs[1].invert_yaxis()
axs[1].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

# Add track count labels on each bar for clarity
for i, bar in enumerate(axs[1].patches):
    axs[1].text(bar.get_width() - 1, bar.get_y() + bar.get_height() / 2, 
                f'{int(bar.get_width()):,}', va='center', ha='center', fontsize=12)

# Adjust layout to ensure that labels and titles are not overlapping
plt.tight_layout()
plt.show()
```

<img width="1241" alt="Screenshot 2024-11-07 at 12 13 11 PM" src="https://github.com/user-attachments/assets/2f3eaf6b-d289-43cc-82e0-c5338f08530b">

### 📅 Temporal Trends
   - **Yearly Trends:** Analyze the **trend** in the number of tracks released annually.
   - **Monthly Patterns:** Does the release pattern vary by month? Which month has the most releases?

``` python
# Create a 2-row subplot with a large figure size for clear visibility and high resolution
fig, axes = plt.subplots(2, 1, figsize=(20, 9.75), dpi=300)

# Get the count of tracks per year, sorted in chronological order
tracks_per_year = spotify_data_cleaned['released_year'].value_counts().sort_index()

# Plot the number of tracks released per year
sns.lineplot(x=tracks_per_year.index, y=tracks_per_year.values, marker='o', linewidth=2.5, markersize=8, color='#2165A8', ax=axes[0])
axes[0].set_title("Number of Tracks Released Per Year", fontsize=18, fontweight='semibold', color='#000000')
axes[0].set_xlabel("Year", fontsize=15, color='#000000')
axes[0].set_ylabel("Number of Tracks", fontsize=15, color='#000000')
axes[0].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

# Get the count of tracks released per month, ensuring the months are in the correct order
tracks_per_month = spotify_data_cleaned['released_month'].value_counts().reindex(month_names.values()).fillna(0)

# Plot the number of tracks released per month
sns.lineplot(x=tracks_per_month.index, y=tracks_per_month.values, marker='o', linewidth=2.5, markersize=8, color='#2165A8', ax=axes[1])
axes[1].set_title("Number of Tracks Released Per Month", fontsize=18, fontweight='semibold', color='#000000')
axes[1].set_xlabel("Month", fontsize=15, color='#000000')
axes[1].set_ylabel("Number of Tracks", fontsize=15, color='#000000')
axes[1].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

# Adjust layout to prevent overlapping of titles and labels
plt.tight_layout()
plt.show()
```

<img width="1241" alt="Screenshot 2024-11-07 at 12 13 18 PM" src="https://github.com/user-attachments/assets/53f86f87-9b37-4eb6-bb3c-1eb0c951e588">

### 🎼 Genre and Music Characteristics
   - **Streams vs. Attributes:** Examine correlation between **streams** and musical attributes like **bpm, danceability_%, energy_%**.
   - **Attribute Relationships:** Is there a correlation between **danceability_%** and **energy_%**? How about **valence_%** and **acousticness_%**?

``` python
# List of musical attributes to analyze correlation between
musical_attributes = ['streams', 'bpm', 'danceability_%', 'valence_%', 'energy_%', 
                      'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']

# Filtered dataset containing only the musical attributes for correlation analysis
spotify_data_filtered = spotify_data_cleaned[musical_attributes]

# Generate the correlation matrix for the selected musical attributes
correlation_matrix = spotify_data_filtered.corr()

# Define a custom blue color palette for the heatmap
blue_palette = ["#B2E0E6", "#A0D4E8", "#8CC8E1", "#78BCEB", "#66A7E3",
                "#5296D4", "#4181C6", "#3173B7", "#2165A8"]

# Create the heatmap figure with specified size and high resolution for clarity
plt.figure(figsize=(10, 4.2), dpi=300)

# Plot the heatmap using seaborn, with annotations for the correlation values
ax = sns.heatmap(correlation_matrix, annot=True, cmap=sns.color_palette(blue_palette, as_cmap=True), 
                 fmt=".4f", square=False, cbar_kws={'label': 'Correlation Coefficient'},
                 annot_kws={'size': 6.5})  

# Set the plot title and format the axes' labels for better readability
plt.title("Correlation Heatmap between Streams and Musical Attributes", fontsize=8, fontweight='semibold', color='#000000')
plt.xticks(fontsize=6.5, color='#000000')
plt.yticks(fontsize=6.5, color='#000000')

# Customize the colorbar label font size
colorbar = ax.collections[0].colorbar  
colorbar.set_label('Correlation Coefficient', fontsize=6.5)
colorbar.ax.tick_params(labelsize=6.5)

# Display the plot
plt.show()
```

<img width="1241" alt="Screenshot 2024-11-07 at 12 13 25 PM" src="https://github.com/user-attachments/assets/9c603d5f-0cc5-4fd0-9095-39026ba1c9f4">

### 🌐 Platform Popularity
   - **Platform Comparison:** Compare track counts in **Spotify Playlists, Deezer Playlists**, and **Apple Playlists**. Which platform favors the most popular tracks?

``` python
# Create a dictionary to store the sum of track counts for each platform category
platform_counts = {'Spotify Playlists': spotify_data_cleaned['in_spotify_playlists'].sum(),
                   'Spotify Charts': spotify_data_cleaned['in_spotify_charts'].sum(),
                   'Apple Playlists': spotify_data_cleaned['in_apple_playlists'].sum(),
                   'Apple Charts': spotify_data_cleaned['in_apple_charts'].sum(),
                   'Deezer Playlists': spotify_data_cleaned['in_deezer_playlists'].sum(),
                   'Deezer Charts': spotify_data_cleaned['in_deezer_charts'].sum(),
                   'Shazam Charts': spotify_data_cleaned['in_shazam_charts'].sum(),}

# Initialize a figure for the bar plot with specified size and resolution
plt.figure(figsize=(20, 4.05), dpi=300)

# Create a bar plot showing the number of tracks on each platform or chart
bars = sns.barplot(x=list(platform_counts.keys()), y=list(platform_counts.values()), color=pastel_color, edgecolor='black')

# Set the title and axis labels with formatting for readability
plt.title("Comparison of Track Counts Across Platforms", fontsize=15, fontweight='semibold', color='#000000')
plt.xlabel("Platform", fontsize=13, color='#000000')
plt.ylabel("Number of Tracks", fontsize=13, color='#000000')
plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

# Annotate each bar with the exact count above it for clarity
for bar in bars.patches:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
             f'{int(bar.get_height()):,}', 
             ha='center', va='bottom', fontsize=10, color='#000000')

# Display the plot
plt.show()
```

<img width="1241" alt="Screenshot 2024-11-07 at 12 13 31 PM" src="https://github.com/user-attachments/assets/b9ea8e76-f440-48bf-9b4a-60d1f9cb2592">

### 💡 Advanced Analysis
   - **Key & Mode Patterns:** Based on **streams**, are there patterns among tracks with similar **key** or **mode** (Major vs. Minor)?
   - **Frequent Artists in Playlists/Charts:** Do certain **artists** consistently appear in more playlists or charts?

``` python
# Filter the dataset to exclude rows where the 'key' column has 'unknown' values
filtered_spotify_data_by_key = spotify_data_cleaned[spotify_data_cleaned['key'].str.lower() != 'unknown']

# Calculate total streams by musical key, sorted by key order
total_streams_by_key = filtered_spotify_data_by_key.groupby('key')['streams'].sum().sort_index()

# Calculate total streams by musical mode, sorted by stream count in ascending order
total_streams_by_mode = spotify_data_cleaned.groupby('mode')['streams'].sum().sort_values()

# Set up a 2x2 grid of subplots for visualizing the data, with a defined figure size and resolution
fig, axes = plt.subplots(2, 2, figsize=(20, 9.65), dpi=300)

# Bar plot for total streams by key
axes[0, 0].bar(total_streams_by_key.index, total_streams_by_key.values, color=pastel_color, edgecolor='black')
axes[0, 0].set_title('Total Streams by Key', fontsize=17, fontweight='semibold')
axes[0, 0].set_xlabel('Key', fontsize=15, color='#000000')
axes[0, 0].set_ylabel('Total Streams', fontsize=15, color='#000000')
axes[0, 0].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

# Annotate each bar with the stream count, formatted with commas for readability
for i, v in enumerate(total_streams_by_key):
    axes[0, 0].text(i, v + (0.01 * v), f"{v:,.0f}", ha='center', va='bottom', fontsize=7.5)

# Bar plot for total streams by mode
axes[0, 1].bar(total_streams_by_mode.index, total_streams_by_mode.values, color=pastel_color, edgecolor='black')
axes[0, 1].set_title('Total Streams by Mode', fontsize=17, fontweight='semibold')
axes[0, 1].set_xlabel('Mode', fontsize=15, color='#000000')
axes[0, 1].set_ylabel('Total Streams', fontsize=15, color='#000000')
axes[0, 1].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

# Annotate each bar with the stream count for modes
for i, v in enumerate(total_streams_by_mode):
    axes[0, 1].text(i, v + (0.01 * v), f"{v:,.0f}", ha='center', va='bottom', fontsize=10)

# Bar plot for top 5 artists in total playlists
axes[1, 0].bar(top_artists_playlists.index, top_artists_playlists.values, color=pastel_color, edgecolor='black')
axes[1, 0].set_title('Top 5 Artists in Total Playlists', fontsize=17, fontweight='semibold')
axes[1, 0].set_xlabel('Artist', fontsize=15, color='#000000')
axes[1, 0].set_ylabel('Total Playlist Count', fontsize=15, color='#000000')
axes[1, 0].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

# Annotate each bar with the playlist count for artists
for i, v in enumerate(top_artists_playlists):
    axes[1, 0].text(i, v + (0.01 * v), f"{v:,.0f}", ha='center', va='bottom', fontsize=10)

# Bar plot for top 5 artists in total charts
axes[1, 1].bar(top_artists_charts.index, top_artists_charts.values, color=pastel_color, edgecolor='black')
axes[1, 1].set_title('Top 5 Artists in Total Charts', fontsize=17, fontweight='semibold')
axes[1, 1].set_xlabel('Artist', fontsize=15, color='#000000')
axes[1, 1].set_ylabel('Total Chart Count', fontsize=15, color='#000000')
axes[1, 1].grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

# Annotate each bar with the chart count for artists
for i, v in enumerate(top_artists_charts):
    axes[1, 1].text(i, v + (0.01 * v), f"{v:,.0f}", ha='center', va='bottom', fontsize=10)

# Adjust layout for better spacing between plots
plt.tight_layout()

# Display all plots
plt.show()
```

<img width="1241" alt="Screenshot 2024-11-07 at 12 13 38 PM" src="https://github.com/user-attachments/assets/6c5a709d-0462-4d24-8771-3d4d54b6dc1d">

---

# waley pa

## 📋 General Guidelines (hahahahha)

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
