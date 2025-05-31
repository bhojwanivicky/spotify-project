# Rolling Stones Spotify Analysis

This project analyzes Rolling Stones' songs on Spotify using various data analysis and machine learning techniques. The script (`app.py`) performs exploratory data analysis (EDA), clustering, and visualization of audio features.

## Features

1. **Data Loading and Cleaning**:

   - Loads song data from `rolling_stones_spotify.csv`.
   - Handles missing values and removes duplicates.

2. **Exploratory Data Analysis (EDA)**:

   - Identifies albums with the most popular songs.
   - Visualizes feature distributions.
   - Analyzes popularity trends over time.
   - Displays a correlation heatmap of audio features.

3. **Clustering**:

   - Scales features using `StandardScaler`.
   - Reduces dimensionality with PCA for visualization.
   - Determines the optimal number of clusters using the elbow method.
   - Performs KMeans clustering and visualizes clusters in PCA space.

4. **Cluster Interpretation**:
   - Summarizes feature averages for each cluster.
   - Exports clustered data to `clustered_rolling_stones_songs.csv`.

## Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Place the `rolling_stones_spotify.csv` file in the same directory as `app.py`.
2. Run the script:

   ```bash
   python app.py
   ```

3. View the generated visualizations and cluster summary in the console.
4. Check the `clustered_rolling_stones_songs.csv` file for the exported clustered data.

## Visualizations

- **Bar Chart**: Albums with the most popular songs.
- **Histograms**: Distributions of audio features.
- **Line Plot**: Popularity trend over time.
- **Heatmap**: Correlation between audio features.
- **Scatter Plot**: PCA projection of songs and clusters.

## Clustering Details

- **Feature Scaling**: Standardized all features for clustering.
- **Dimensionality Reduction**: PCA reduced features to 2 components for visualization.
- **KMeans Clustering**: Optimal number of clusters determined using the elbow method.

## Output

- Visualizations displayed during script execution.
- Clustered data saved to `clustered_rolling_stones_songs.csv`.

## Notes

- Ensure the `rolling_stones_spotify.csv` file contains valid data with columns such as `popularity`, `acousticness`, `danceability`, etc.
- Modify the `popular_threshold` variable in the script to adjust the popularity cutoff for EDA.
