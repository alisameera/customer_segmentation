# Customer Segmentation & CRM Strategy — RFM + K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-KMeans-orange?style=flat&logo=scikitlearn&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-CRM%20Strategy-green?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

---

## Business Problem

Generic marketing campaigns treat all customers the same — wasting budget on lapsed customers and under-investing in loyal ones. This project segments **4,338 customers** into behavioural groups using RFM analysis and K-Means clustering, then translates each segment into a **specific, actionable CRM recommendation**.

---

## Results — 4 Customer Segments

| Cluster | Label | Recency | Monetary | Count | Recommended Action |
|---------|-------|---------|----------|-------|--------------------|
| **0** | 🟢 Loyal High-Value | 0–50 days | >£100,000 | ~18% | Loyalty rewards, early access, minimal discounting |
| **1** | 🟠 Lapsed / At-Risk | 200–350 days | Low | ~23% | Win-back campaign; strong incentive; deprioritise if no response in 60 days |
| **2** | 🔵 Occasional Buyers | Moderate | Mixed | ~31% | Bundle offers, personalised recommendations |
| **3** | 🩷 Mid-Tier Spenders | 100–300 days | Varying | ~28% | Upsell campaigns, loyalty points to increase switching cost |

---

## Key Business Insight

> The top ~18% of customers (Cluster 0 — Loyal High-Value) generate a disproportionate share of revenue. Retaining this segment delivers 3× more ROI per marketing ₹/£ than acquiring new customers. CRM strategy should **prioritise retention over acquisition**.

---

## Dataset

- **Source:** Kaggle — Customer Segmentation from Online Retail
- **Size:** 541,909 raw transactions → cleaned to **4,338 unique customers**
- **Columns:** InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

---

## RFM Feature Engineering

| Metric | Description | Dataset Average |
|--------|-------------|-----------------|
| **Recency** | Days since last purchase | 92.5 days |
| **Frequency** | Number of unique invoices | 4.27 transactions |
| **Monetary** | Total spend | £2,054 |

---

## Methodology

```
541,909 raw transactions
        │
        ▼
Data Cleaning
  ├── Remove null CustomerID / Description
  ├── Filter Quantity > 0 and UnitPrice > 0
  └── Parse InvoiceDate to datetime
        │
        ▼
RFM Feature Engineering
  ├── Recency = days since last invoice
  ├── Frequency = count of unique invoices
  └── Monetary = sum(Quantity × UnitPrice)
        │
        ▼
Preprocessing
  └── StandardScaler normalisation
        │
        ▼
Optimal k Selection
  ├── Elbow Method → k=4 (SSE stabilises)
  └── Silhouette Score → peaks at k=3
        │
        ▼
K-Means Clustering (k=4 selected)
        │
        ▼
Cluster Profiling + CRM Recommendations
```

---

## Why k=4?

The Elbow method suggested k=4 (SSE stabilisation point) while Silhouette Score peaked at k=3. **k=4 was chosen** to provide more granular segmentation — the business value of distinguishing mid-tier from occasional buyers outweighs the slight drop in cluster cohesion.

---

## Visualisations

- RFM pair plot by cluster (Seaborn)
- Elbow curve (SSE vs k)
- Silhouette score curve
- Cluster scatter plots (Recency vs Monetary, Frequency vs Monetary)

---

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn (KMeans, StandardScaler, silhouette_score)` · `Matplotlib` · `Seaborn` · `Jupyter`

---

## Files

```
├── Customer_Segmentation.ipynb   ← main notebook
├── Customer Segmentation.csv     ← dataset (Kaggle)
└── README.md
```

---

## Author

**Sameera Ali** | [LinkedIn](https://www.linkedin.com/in/sameera-ali-0055252a2/) | [GitHub](https://github.com/alisameera)
