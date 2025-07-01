
# Customer Segmentation Using Clustering Algorithms

## Project Documentation

### 1. Introduction

- **Problem Statement**: In today’s highly competitive market, businesses often face significant challenges in retaining customers and delivering personalized experiences due to a lack of deep insight into customer behavior. A generic "one-size-fits-all" marketing approach frequently leads to wasted resources and dissatisfied customers. Customer segmentation emerges as a powerful solution, enabling companies to group customers based on shared behaviors and characteristics, thereby facilitating tailored marketing strategies, enhanced customer satisfaction, and improved return on investment (ROI).
- **Project Goal**: The primary objective of this project is to segment customers based on their purchasing behavior using advanced clustering techniques. By identifying distinct customer groups, the project aims to support the following:
  - Design of targeted marketing campaigns.
  - Development of personalized product recommendations.
  - Enhancement of customer satisfaction and loyalty.
  - Efficient allocation of organizational resources.
- **Expected Outcome**: 
  - Clearly defined customer segments accompanied by behavioral labels (e.g., "High-Value Spenders", "Occasional Visitors").
  - Data-driven recommendations to inform targeted marketing strategies.
  - Visual representations providing insights into customer distribution and behavior patterns.

---

### 2. Dataset Overview

- **Source**: The dataset is sourced from Kaggle, titled "Customer Segmentation Dataset" (specifically, "Customer Segmentation from Online Retail").
- **Format**: CSV file.
- **Description**: Contains anonymized details of customers and their purchasing behaviors, derived from online retail transactions.
- **Schema**:
  - **Expected Columns (based on common segmentation datasets):**
    | Column Name     | Description                          |
    |-----------------|--------------------------------------|
    | `CustomerID`    | Unique identifier for each customer  |
    | `Genre`         | Gender of the customer               |
    | `Age`           | Age of the customer                  |
    | `Annual Income` | Income of the customer (likely in thousands) |
    | `Spending Score`| Score assigned based on purchasing behavior |

  - **Actual Columns (from uploaded dataset):**
    | Column Name    | Description                          |
    |----------------|--------------------------------------|
    | `InvoiceNo`    | Invoice number (transaction ID)      |
    | `StockCode`    | Product/item identifier              |
    | `Description`  | Description of the purchased item    |
    | `Quantity`     | Number of items purchased            |
    | `InvoiceDate`  | Date and time of transaction         |
    | `UnitPrice`    | Price per item                       |
    | `CustomerID`   | Unique customer identifier           |
    | `Country`      | Customer's country of residence      |

- **Statistics**:
  - **Rows**: 541,909
  - **Columns**: 8
  - **Missing Values**: `CustomerID` and `Description` contain missing entries
  - **Date Range**: Includes timestamps of transactions, suitable for recency-based analysis

---

### 3. Methodology

We are pleased to present a detailed methodology outlining the step-by-step process we followed to execute this customer segmentation project, as implemented in the Jupyter Notebook (`Customer_Segmentation.ipynb`). Each stage reflects a thoughtful approach to data handling and analysis.

#### 3.1 Importing Libraries
- **What We Did**: We began by importing the essential Python libraries required to support our analysis.
- **Code Snippet**:
  ```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  from datetime import datetime
  from sklearn.preprocessing import StandardScaler
  from sklearn.cluster import KMeans
  from sklearn.metrics import silhouette_score
  ```
- **Why It Matters**: These libraries provide the foundation for data manipulation, visualization, and clustering, forming the backbone of our analytical toolkit.

#### 3.2 Loading & Exploring Data
- **What We Did**: We loaded the dataset and conducted an initial exploration to understand its structure.
- **Code Snippet**:
  ```python
  df = pd.read_csv("Customer Segmentation.csv", encoding='ISO-8859-1')
  df.head()
  print(df.info())
  ```
- **What We Found**: The dataset comprises 541,909 rows and 8 columns, with missing values identified in `CustomerID` and `Description`. This initial review guides our subsequent cleaning and analysis steps.

#### 3.3 Data Cleaning
- **What We Did**: We meticulously cleaned the dataset by removing missing values and invalid entries to ensure data integrity.
- **Code Snippet**:
  ```python
  df.dropna(subset=['CustomerID', 'Description'], inplace=True)
  df = df[df['Quantity'] > 0]
  df = df[df['UnitPrice'] > 0]
  df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
  ```
- **Why It Matters**: This process eliminates nulls, negative quantities, and prices, while converting `InvoiceDate` to a datetime format, laying a solid foundation for time-based analysis.

#### 3.4 Feature Engineering via RFM Metrics
- **What We Did**: We engineered Recency, Frequency, and Monetary (RFM) metrics to capture key aspects of customer behavior.
- **Code Snippet**:
  ```python
  reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
  rfm = df.groupby('CustomerID').agg({
      'InvoiceDate': lambda x: (reference_date - x.max()).days,
      'InvoiceNo': 'nunique',
      'UnitPrice': lambda x: np.round(np.sum(x * df.loc[x.index, 'Quantity']), 2)
  })
  rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'UnitPrice': 'Monetary'}, inplace=True)
  print(rfm.describe())
  ```
- **What We Found**: The RFM analysis revealed averages of 92.5 days for Recency, 4.27 for Frequency, and £2,054 for Monetary, providing valuable insights into customer purchasing patterns.

#### 3.5 Data Preprocessing for Clustering
- **What We Did**: We preprocessed the RFM data by applying standardization to prepare it for clustering.
- **Code Snippet**:
  ```python
  scaler = StandardScaler()
  rfm_scaled = scaler.fit_transform(rfm)
  ```
- **Why It Matters**: Scaling ensures that Recency, Frequency, and Monetary are on a comparable scale, which is essential for the KMeans algorithm’s effectiveness.

#### 3.6 Optimal Cluster Detection
- **What We Did**: We determined the optimal number of clusters using the Elbow and Silhouette methods.
- **Code Snippet**:
  ```python
  sse = []; silhouette_scores = []; K = range(2, 11)
  for k in K:
      kmeans = KMeans(n_clusters=k, random_state=42)
      kmeans.fit(rfm_scaled)
      sse.append(kmeans.inertia_)
      silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1); plt.plot(K, sse, 'bx-'); plt.xlabel('k'); plt.ylabel('SSE'); plt.title('Elbow Method')
  plt.subplot(1, 2, 2); plt.plot(K, silhouette_scores, 'rx-'); plt.xlabel('k'); plt.ylabel('Silhouette Score'); plt.title('Silhouette Score')
  plt.tight_layout(); plt.show()
  ```
- **What We Found**: The Elbow method suggests 4 clusters as an optimal point, while the Silhouette Score peaks at 3, offering a thoughtful choice to consider.

#### 3.7 K-Means Clustering Implementation
- **What We Did**: We implemented KMeans clustering with 4 clusters, based on the Elbow method’s recommendation.
- **Code Snippet**:
  ```python
  optimal_k = 4
  kmeans = KMeans(n_clusters=optimal_k, random_state=42)
  rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
  ```
- **Why It Matters**: This step assigns each customer to a cluster, enabling us to derive meaningful segment profiles.

#### 3.8 Cluster Profiling & Insights
- **What We Did**: We profiled the clusters to gain deeper insights into their characteristics.
- **Code Snippet**:
  ```python
  cluster_summary = rfm.groupby('Cluster').agg({
      'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean', 'CustomerID': 'count'
  }).rename(columns={'CustomerID': 'Count'})
  print(cluster_summary)
  ```
- **What We Found**: The clusters exhibit varied behaviors, with some indicating recent high spenders and others showing inactive patterns—details are explored further in the results.

#### 3.9 Visualizations
- **What We Did**: We generated visualizations to illustrate the cluster distributions effectively.
- **Code Snippet**:
  ```python
  sns.pairplot(rfm.reset_index(), vars=['Recency', 'Frequency', 'Monetary'], hue='Cluster', palette='Set2')
  plt.suptitle('RFM Pair Plot by Cluster', y=1.02)
  plt.show()
  ```
- **Why It Matters**: These visual aids help us observe how clusters differ across Recency, Frequency, and Monetary, supporting informed decision-making.

---

### 4. Results

We are delighted to share the outcomes of our analysis, derived from the clustering process and accompanying visualizations:

- **Cluster Analysis**:
  - **Cluster 0 (Green)**: These customers exhibit low Recency (0-50 days) and significant Monetary values (>100,000), identifying them as loyal, high-value individuals.
  - **Cluster 1 (Orange)**: Characterized by high Recency (200-350 days) and low Monetary, these appear to be inactive or lapsed customers.
  - **Cluster 2 (Blue)**: Featuring moderate Recency, low Frequency, and mixed Monetary values, these may represent occasional buyers.
  - **Cluster 3 (Pink)**: Displaying a spread across Recency (100-300 days) with varying Monetary, this group likely includes mid-tier or occasional spenders.

- **Elbow & Silhouette Insights**:
  - The Elbow method indicates 4 clusters as an optimal choice, where the Sum of Squared Errors (SSE) begins to stabilize.
  - The Silhouette Score reaches its peak at 3 clusters, suggesting superior separation between groups.
  - Please refer to the pair plots and scatterplots within the notebook (`Customer_Segmentation.ipynb`) for a visual representation of these findings.

---

### 5. Conclusion and Implications

#### Conclusion
We have successfully segmented 4,338 customers into 4 distinct clusters using RFM metrics and KMeans clustering, a commendable achievement. Cluster 0 emerges as our loyal, high-value segment, while Cluster 1 highlights lapsed customers requiring re-engagement efforts. The Elbow method supports 4 clusters for detailed segmentation, whereas the Silhouette Score favors 3 clusters for optimal cohesion. This presents a strategic decision point to align with business objectives.

#### Implications
- **Targeted Campaigns**: We recommend focusing retention and upselling initiatives on Cluster 0, while developing reactivation strategies for Cluster 1.
- **Resource Allocation**: Prioritize resource investment in the high-value Cluster 0, while monitoring the potential growth of Cluster 3’s mid-tier spenders.
- **Decision Point**: The variance between 3 and 4 clusters necessitates alignment with business priorities—choosing between granularity (4) or cohesion (3).

---

### 7. References
- **Dataset**: "Customer Segmentation Dataset" from Kaggle.
- **Libraries**: pandas, numpy, matplotlib, seaborn, sklearn.

---
