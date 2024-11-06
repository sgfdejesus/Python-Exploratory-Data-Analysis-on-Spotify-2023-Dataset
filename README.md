# 🎶 Exploratory Data Analysis: Most Streamed Spotify Songs 2023 🎶 andami paaaa

This project performs an **Exploratory Data Analysis (EDA)** on a dataset detailing popular tracks on Spotify in 2023. The goal is to analyze, visualize, and interpret data to extract insights about trends, characteristics, and factors that contribute to a song's popularity.

Dataset source: [Kaggle - Most Streamed Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023)

---

## 📈 Analysis Workflow (will change pa)

- **Data Loading, Overview, and Cleaning**: Load the dataset, handle missing values, and confirm data types.
- **Exploratory Analysis**: Perform descriptive statistics and initial visualizations to gain insights.
- **In-Depth Analysis**: Dive deeper into specific aspects like **temporal trends** and **platform comparisons**.
- **Insight Generation**: Summarize findings and provide recommendations.

---

### Data Loading

``` python
# Import libraries for data manipulation, visualization, and analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Spotify 2023 dataset, ensuring proper encoding for special characters
spotify_data = pd.read_csv('spotify-2023.csv', encoding='latin1')
display(spotify_data)
```

<img width="1018" alt="Screenshot 2024-11-03 at 8 36 50 PM" src="https://github.com/user-attachments/assets/860b17fe-3345-425f-83ab-e70224735be3">
<img width="1032" alt="Screenshot 2024-11-03 at 8 36 56 PM" src="https://github.com/user-attachments/assets/4bd58836-9293-4fd9-9fac-f12c95da5905">

- The code begins by importing essential libraries for **data manipulation**, **visualization**, and **analysis** in Python. It imports **NumPy** as `np` for **numerical operations**, **Pandas** as `pd` for handling data in a **structured format**, **Matplotlib's** `pyplot` as `plt` for creating **static visualizations**, and **Seaborn** as `sns` for enhanced **statistical graphics**. Following the imports, the code loads the **Spotify 2023 dataset** from a **CSV file** named **'spotify-2023.csv'** into a **Pandas DataFrame** called `spotify_data`. The `encoding='latin1'` parameter is specified to ensure that any **special characters** within the dataset are read correctly, which is crucial for accurate **data representation** and **analysis**. This setup establishes a foundation for further **exploration** and **analysis** of the dataset.

### Overview of Dataset
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

<img width="1128" alt="Screenshot 2024-11-03 at 8 28 48 PM" src="https://github.com/user-attachments/assets/c387c30c-6f45-453b-a605-2f75d0810751">

- This code provides an overview of the **Spotify dataset** by examining its **dimensions**, **column data types**, and **missing values**. First, it retrieves and prints the dataset's **dimensions**, revealing **953 rows** and **24 columns**. Then, it displays the **data types** of each column, helping to understand the **structure** and **content type** of the dataset; most columns are **integers**, with a few containing **objects** (strings). Finally, it checks for any **missing values**, identifying two columns, **`in_shazam_charts`** and **`key`**, with missing entries, reporting counts of **50** and **95**, respectively. This summary offers a foundational understanding of the **dataset**, highlighting areas that may need further **cleaning** or **handling**.


### Data Cleaning

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

<img width="1128" alt="Screenshot 2024-11-03 at 9 45 19 PM" src="https://github.com/user-attachments/assets/a5d35dae-d137-4802-a5cd-b57156166161">

- This code **cleans** and **standardizes** the **Spotify dataset** to ensure **consistency** and **completeness**. It starts by defining two sets of columns for conversion: **`numeric_columns`**, which should contain **numeric data**, and **`string_columns`**, meant to store **text**. It then converts the columns in `numeric_columns` to **numeric data types**, handling any **errors** by setting invalid entries as **NaN**. **Missing values** in numeric columns are filled with **0**, while missing values in string columns are replaced with **'unknown'**. Next, any **floats** in numeric columns are converted to **integers** for uniformity. **Duplicate rows** are removed to maintain **unique entries**, and a final check confirms that there are no remaining **missing values**. Afterward, it verifies **data types** for each column post-cleaning and saves the **cleaned dataset** as a new **CSV file**, **'spotify-2023-cleaned.csv'**, for future **analysis**.

### Basic Descriptive Statistics
   - **Streams Statistics:** What are the **mean, median, and standard deviation** of the streams?
   - **Released Year & Artist Count:** What is the distribution? Are there any **trends** or **outliers**?



- The code begins by loading a **cleaned version** of the **Spotify dataset** and calculating basic **descriptive statistics** for the **'streams'** column, including the **mean**, **median**, and **standard deviation**. These statistics provide insights into the distribution of streaming counts, with the output indicating a mean of approximately **513.6 million streams**, a median of about **290.2 million**, and a substantial standard deviation of **566.8 million**, suggesting significant variability in the streaming numbers.

  Following the statistics, the code sets up a series of **visualizations** to explore the dataset further. Four **subplots** are created: the first two are **bar graphs** displaying the distribution of **released years** and **artist counts**, respectively. The visualizations show that **2022** has the highest distribution of released songs, which may indicate a **trend** of increasing music releases over the years, possibly driven by the growing popularity of **digital streaming platforms** and the ease of music distribution. The **artist count** graph reveals that most tracks feature a **single artist**, underscoring the trend towards **solo performances** in the music industry.

  The last two subplots are **box plots** for `released_year` and `artist_count`, providing a **visual summary** of the data's spread and outliers. For the `released_year`, the code identifies **151 outliers**, with years ranging from **1930 to 2016**. In contrast, for `artist_count`, there are **27 outliers**, with counts ranging from tracks featuring **4 to 8 artists**.

### Top Performers
   - **Top Tracks:** Which track has the highest number of streams? List the **top 5 most streamed** tracks.
   - **Frequent Artists:** Who are the **top 5 most frequent artists** based on track count?

- This code snippet identifies the **top performers** in the **Spotify dataset** by analyzing **streaming counts** and **artist frequency**. First, it finds the track with the highest number of streams and displays the **top five most streamed tracks**. According to the results, **"Blinding Lights"** leads with approximately **3.7 billion streams**, followed by **"Shape of You"** with about **3.56 billion streams**, **"Someone You Loved"** with approximately **2.89 billion streams**, **"Dance Monkey"** at around **2.86 billion streams**, and **"Sunflower - Spider-Man: Into the Spider-Verse"** with about **2.81 billion streams**.

   Next, the code identifies the **top five most frequent artists** based on the number of tracks present in the dataset. The results show that **Bad Bunny** has the highest count with **40 tracks**, followed closely by **Taylor Swift** with **38 tracks**. **The Weeknd** appears with **37 tracks**, while **SZA** and **Kendrick Lamar** each have **23 tracks**. This analysis highlights not only the most popular tracks based on streaming but also the artists who dominate the music landscape in terms of **track quantity**.

### Temporal Trends
   - **Yearly Trends:** Analyze the **trend** in the number of tracks released annually.
   - **Monthly Patterns:** Does the release pattern vary by month? Which month has the most releases?



### Genre and Music Characteristics
   - **Streams vs. Attributes:** Examine correlation between **streams** and musical attributes like **bpm, danceability_%, energy_%**.
   - **Attribute Relationships:** Is there a correlation between **danceability_%** and **energy_%**? How about **valence_%** and **acousticness_%**?

### Platform Popularity
   - **Platform Comparison:** Compare track counts in **Spotify Playlists, Deezer Playlists**, and **Apple Playlists**. Which platform favors the most popular tracks?

### Advanced Analysis
   - **Key & Mode Patterns:** Based on **streams**, are there patterns among tracks with similar **key** or **mode** (Major vs. Minor)?
   - **Frequent Artists in Playlists/Charts:** Do certain **artists** consistently appear in more playlists or charts?

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






