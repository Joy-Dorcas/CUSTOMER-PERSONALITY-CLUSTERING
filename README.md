# Customer Segmentation Analysis Project

## Overview

This project performs comprehensive customer segmentation analysis using machine learning clustering techniques on marketing campaign data. The analysis identifies distinct customer segments to enable targeted marketing strategies and improve business outcomes.

## Dataset

**Source**: Marketing Campaign Dataset (2,240 customers, 29 features)

**Key Features**:
- Demographics: Age, Education, Marital Status, Income
- Purchase Behavior: Spending across 6 product categories (Wines, Fruits, Meat, Fish, Sweets, Gold)
- Customer Lifecycle: Seniority, Recency, Purchase channels
- Family Structure: Children, Teenagers at home

## Project Structure

```
customer-segmentation-analysis/
│
├── data/
│   └── marketing_campaign.csv
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clustering_analysis.ipynb
│   └── 04_cluster_interpretation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── clustering_models.py
│   └── visualization.py
├── results/
│   ├── cluster_profiles.csv
│   ├── visualizations/
│   └── business_recommendations.md
└── README.md
```

## Methodology

### 1. Data Preprocessing
- **Missing Value Treatment**: Imputed Income missing values with median
- **Feature Engineering**: 
  - Created Age from Year_Birth
  - Calculated total Spending across categories
  - Computed Seniority (months since registration)
  - Simplified categorical variables (Education, Marital_Status)
- **Standardization**: Applied StandardScaler to numerical features
- **Encoding**: One-hot encoding for categorical variables

### 2. Clustering Analysis
- **Algorithm**: K-Means clustering with multiple validation techniques
- **Optimal Clusters**: 4 clusters determined through:
  - Elbow Method (WCSS)
  - Silhouette Analysis
  - Calinski-Harabasz Index
  - Davies-Bouldin Index
- **Features Used**: Age, Income, Spending, Seniority, Children, Product categories

### 3. Cluster Validation
- **Silhouette Score**: 0.536 (Good separation)
- **Statistical Significance**: ANOVA tests confirm significant differences across all features
- **Effect Sizes**: 9/11 features show large practical significance (η² > 0.14)
- **PCA Analysis**: 66% variance explained in 3D space

## Results

### Identified Customer Segments

#### Cluster 0: Budget-Conscious Families (54.0% - 1,209 customers)
- **Profile**: Low income, high number of children, minimal spending
- **Characteristics**: Price-sensitive, focus on essentials
- **Strategy**: Value pricing, family bundles, cost-effective marketing

#### Cluster 1: Premium Wine Enthusiasts (12.4% - 277 customers)
- **Profile**: High wine spending, moderate income, quality-focused
- **Characteristics**: Sophisticated taste, brand loyalty
- **Strategy**: Premium wine curation, tasting events, loyalty programs

#### Cluster 2: Balanced Middle-Income Shoppers (24.7% - 554 customers)
- **Profile**: Moderate spending across categories, diverse preferences
- **Characteristics**: Responsive to promotions, balanced purchasing
- **Strategy**: Cross-selling, seasonal campaigns, mid-range positioning

#### Cluster 3: Affluent Gourmet Food Lovers (8.9% - 200 customers)
- **Profile**: Highest spending, no children, premium food focus
- **Characteristics**: High disposable income, quality-driven
- **Strategy**: Luxury products, VIP service, exclusivity

## Key Findings

### Statistical Validation
- **100% Feature Significance**: All 11 features show statistically significant differences between clusters
- **Strong Effect Sizes**: Large practical significance in spending patterns and demographics
- **Cluster Quality**: Good separation with minimal overlap

### Business Impact
- **Revenue Distribution**: Clear differentiation in customer value across segments
- **Product Preferences**: Distinct patterns enable targeted product development
- **Marketing Efficiency**: Segment-specific strategies can improve ROI

## Technical Implementation

### Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
```

### Key Functions
- `complete_cluster_analysis()`: Comprehensive analysis with quality metrics
- `evaluate_cluster_quality()`: Multiple clustering validation methods
- `visualize_clusters()`: PCA plots, heatmaps, and business visualizations

## Business Recommendations

### Immediate Actions
1. **Segment-Specific Marketing**: Deploy targeted campaigns for each cluster
2. **Product Portfolio Optimization**: Align inventory with cluster preferences
3. **Pricing Strategy**: Implement tiered pricing based on segment value
4. **Customer Journey Mapping**: Design segment-specific experiences

### Resource Allocation
- **40%** - Cluster 3 (Affluent): High ROI, premium positioning
- **30%** - Cluster 1 (Wine): Growth potential, specialization
- **20%** - Cluster 2 (Balanced): Cross-selling opportunities
- **10%** - Cluster 0 (Budget): Retention, operational efficiency

### Success Metrics
- Revenue per segment
- Customer acquisition cost by cluster
- Cross-selling success rates
- Segment migration patterns
- Customer lifetime value by cluster

## Model Performance

| Metric | Score | Interpretation |
|--------|-------|---------------|
| Silhouette Score | 0.536 | Good cluster separation |
| Calinski-Harabasz | 2,832 | Well-separated clusters |
| Davies-Bouldin | 0.885 | Compact, distinct clusters |
| PCA Variance (2D) | 56.4% | Adequate dimensionality reduction |

## Usage Instructions

### Running the Analysis
```python
# Load and preprocess data
data = load_and_preprocess('marketing_campaign.csv')

# Perform clustering
results = complete_cluster_analysis(data)

# Generate business insights
create_cluster_profiles(results)
```

### Customization Options
- Adjust number of clusters based on business needs
- Modify feature selection for different focus areas
- Update cluster naming and interpretation for specific industries

## Limitations and Considerations

### Data Limitations
- **Temporal Snapshot**: Analysis based on single time period
- **Missing Context**: Limited demographic and behavioral variables
- **Sample Bias**: May not represent entire customer population

### Model Considerations
- **Static Segmentation**: Clusters may evolve over time
- **Feature Dependency**: Results sensitive to feature selection and scaling
- **Business Context**: Requires domain expertise for optimal interpretation

### Implementation Risks
- **Resource Allocation**: Extreme cluster size differences may create operational challenges
- **Customer Migration**: Segments may shift requiring model updates
- **Measurement Complexity**: Multi-cluster KPIs can be challenging to track

## Future Enhancements

### Advanced Analytics
- **Dynamic Clustering**: Time-series clustering for evolving segments
- **Predictive Modeling**: Customer lifetime value and churn prediction
- **Recommendation Systems**: Cluster-based product recommendations

### Additional Data Integration
- **Transaction History**: Detailed purchase patterns over time
- **External Data**: Demographic and psychographic enrichment
- **Behavioral Tracking**: Website and app interaction data

### Operational Integration
- **Real-time Scoring**: API for live customer classification
- **A/B Testing Framework**: Validate segment-specific strategies
- **Feedback Loop**: Continuous model improvement based on campaign results

## Contributors

- Data Scientist: Clustering analysis and validation
- Business Analyst: Strategic interpretation and recommendations
- Marketing Team: Campaign strategy and implementation guidance

## License

This project is for internal business use. Data and methodologies are proprietary.



**Last Updated**:22ND OF SEPT 2025
**Model Version**: 1.0
