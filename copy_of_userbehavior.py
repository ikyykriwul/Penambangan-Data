import subprocess
import sys

# Ensure scikit-learn is installed
try:
    import sklearn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
finally:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

print("KMeans imported successfully")
# Judul aplikasi
st.title('User Behavior Clustering')

# Upload file CSV
uploaded_file = st.file_uploader("Choose a file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Menampilkan data
    st.write(df)

    # Scaling data
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 'Number of Apps Installed', 'Data Usage (MB/day)']])

    # KMeans clustering
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plot Elbow Method
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    st.pyplot(plt)

    # Clustering dengan jumlah cluster yang dipilih
    n_clusters = st.slider('Select number of clusters', 1, 10, 4)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    df['Cluster'] = y_kmeans

    st.write('Cluster centroids:')
    for i in range(n_clusters):
        st.write(f'Cluster {i}: {kmeans.cluster_centers_[i]}')

    st.write('\nCluster sizes:')
    st.write(df.groupby('Cluster').size())

    st.write('\nCluster demographics:')
    st.write(df.groupby('Cluster')[['Age']].agg(['mean', 'count']))  # Only include numeric columns for 'mean'
    st.write(df.groupby('Cluster')[['Gender']].agg(['count']))  # Use 'count' for categorical columns like 'Gender'

    for cluster in range(n_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f'\nPersonalization for Cluster {cluster}:')
        st.write(f"- Avg App Usage Time: {cluster_data['App Usage Time (min/day)'].mean():.2f} min/day")
        st.write(f"- Avg Screen On Time: {cluster_data['Screen On Time (hours/day)'].mean():.2f} hours/day")
        st.write(f"- Avg Battery Drain: {cluster_data['Battery Drain (mAh/day)'].mean():.2f} mAh/day")
        st.write(f"- Avg Apps Installed: {cluster_data['Number of Apps Installed'].mean():.2f}")
        st.write(f"- Avg Data Usage: {cluster_data['Data Usage (MB/day)'].mean():.2f} MB/day")
        st.write(f"- Predominant Age: {cluster_data['Age'].mode().values[0]}")
        st.write(f"- Predominant Gender: {cluster_data['Gender'].mode().values[0]}")

# Jalankan aplikasi Streamlit dengan perintah berikut di terminal:
# streamlit run /c:/Kuliah/Semester 5/DataMining/copy_of_userbehavior.py