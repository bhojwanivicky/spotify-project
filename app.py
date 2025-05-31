# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 2. Load Data
df = pd.read_csv("rolling_stones_spotify.csv")
df.head()

# 3. Data Inspection & Cleaning
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Drop duplicates if any
df.drop_duplicates(inplace=True)

# Handle missing values (if any exist)
df.dropna(inplace=True)  # or use imputation if appropriate

# 4. EDA: Albums with Most Popular Songs
popular_threshold = 70
album_popularity = df[df['popularity'] > popular_threshold].groupby(
    'album').size().sort_values(ascending=False)
album_popularity.plot(kind='barh', figsize=(
    10, 8), title='Albums with Most Popular Songs')
plt.xlabel('Number of Popular Songs')
plt.show()

# 5. EDA: Feature Distributions
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness',
            'loudness', 'speechiness', 'tempo', 'valence', 'popularity']
df[features].hist(bins=20, figsize=(15, 10))
plt.suptitle('Feature Distributions')
plt.show()

# 6. Popularity Trend Over Time
df['release_date'] = pd.to_datetime(df['release_date'])
df['year'] = df['release_date'].dt.year

popularity_trend = df.groupby('year')['popularity'].mean()
plt.plot(popularity_trend.index, popularity_trend.values, marker='o')
plt.title('Average Popularity Over Time')
plt.xlabel('Year')
plt.ylabel('Popularity')
plt.grid(True)
plt.show()

# 7. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Audio Features')
plt.show()

# 8. Feature Scaling for Clustering
X = df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 9. Dimensionality Reduction (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.title('PCA Projection of Songs')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# 10. Determine Optimal Clusters (Elbow Method)
inertia = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# 11. KMeans Clustering
k_optimal = 4  # based on elbow plot
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 12. Visualize Clusters in PCA Space
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=df['cluster'], palette='tab10')
plt.title('Clusters of Songs (PCA Projection)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Cluster')
plt.show()

# 13. Cluster Interpretation
cluster_summary = df.groupby('cluster')[features].mean()
print("\nCluster Feature Averages:")
print(cluster_summary)

# Optionally: Export clustered data
df.to_csv("clustered_rolling_stones_songs.csv", index=False)
