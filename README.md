# Data Mining GUI Application

A Python-based GUI application for performing various data mining tasks including data visualization, preprocessing, and clustering.

## Features

- **Data Import**: Supports both CSV and ARFF file formats
- **Data Visualization**:
  - Scatter plots
  - Box plots
  - Histograms
- **Data Preprocessing**:
  - Missing value handling (drop rows, fill with min/max/mean/median/mode/custom value)
  - Data normalization (Min-Max and Z-Score)
- **Clustering Algorithms**:
  - K-Means
  - K-Medoids
  - DBSCAN
  - DIANA (Divisive Analysis)
  - AGNES (Agglomerative Hierarchical Clustering)
- **Cluster Visualization**: Interactive 2D/3D plots of clustering results
- **Elbow Method**: For determining optimal number of clusters
- **Data Export**: Save processed data to CSV or Excel format

## Requirements

- Python 3.x
- Required packages:
  - tkinter
  - pandas
  - numpy
  - scipy
  - scikit-learn
  - matplotlib

## Installation

1. Clone this repository or download the source code
2. Install the required packages:
   ```bash
   pip install pandas numpy scipy scikit-learn matplotlib