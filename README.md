# 🚗 BMW Sales Analysis Dashboard (2010-2024)

<div align="center">

![BMW Dashboard](https://img.shields.io/badge/BMW-Sales%20Dashboard-0066B1?style=for-the-badge&logo=bmw&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.22.0-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

**Interactive Data Analytics Platform for BMW Sales Performance**
<img width="1914" height="918" alt="BMW Sales Dashboard" src="https://github.com/user-attachments/assets/354d6ffa-aaea-4372-ba09-037002b7b0ac" />

</div>

---

## 🎯 About the Project

BMW Sales Analysis Dashboard is an interactive **Streamlit**-based web application that analyzes BMW sales data from **2010-2024**. It combines modern data analytics techniques, advanced visualizations, and machine learning models.

### 🌟 Project Goals
- 📊 Visualize sales trends
- 🌍 Analyze geographical market distribution
- 🤖 Customer segmentation and price prediction
- 📈 Provide data-driven strategic insights

---

## ✨ Key Features

### 🎨 Visual Design
- ✅ Custom BMW-themed interface (gradient animated header)
- ✅ Interactive showcase with 6 BMW model images
- ✅ Dark theme and responsive KPI cards
- ✅ Custom CSS styling

### 📊 Data Analytics
- ✅ 9 different interactive visualizations
- ✅ Dynamic filtering (Year, Region, Model, Fuel Type)
- ✅ Automatic insight analysis
- ✅ 3 KPI metrics (Total Sales, Average Price, Median Mileage)
- ✅ CSV export feature

### 🤖 Machine Learning
- ✅ **K-Means Clustering** - 3D/2D customer segmentation
- ✅ **Random Forest Regressor** - Price prediction model
- ✅ Correlation Matrix - Feature relationship analysis
- ✅ Model performance metrics (R², MSE, MAE)

---

## 🛠 Technology Stack

```
🐍 Python 3.8+          📊 Pandas 2.2.2         🔢 NumPy 1.26.4
🎈 Streamlit 1.36.0     📈 Plotly 5.22.0        📉 Scipy 1.13.1
🤖 Scikit-learn 1.5.1   📏 StandardScaler       🎯 PCA
```

---

## 🚀 Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Starting the Dashboard
```bash
streamlit run app.py
```

The dashboard will automatically open at `http://localhost:8501`.

---

## 📊 Visualizations

The dashboard includes 9 different interactive visualizations:

1. **Annual Sales Trend** - Sales trends with line chart
2. **Region → Model Treemap** - Hierarchical sales distribution
3. **Price by Fuel Type** - Price analysis with box plot
4. **Price vs Mileage** - Correlation with scatter plot
5. **Best Selling Models** - Model comparison with bar chart
6. **Parallel Coordinates** - Multi-dimensional analysis
7. **Global Sales Map** - Geographical distribution
8. **Sankey Flow Diagram** - Region-Fuel-Model flow
9. **Price Distribution + KDE** - Statistical analysis

---

## 🤖 Machine Learning Models

### K-Means Clustering
- Customer segmentation (adjustable between 2-8 clusters)
- 3D and 2D visualization (dimension reduction with PCA)
- Clustering quality assessment with silhouette score

### Random Forest Regressor
- Price prediction model (200 decision trees)
- Model performance metrics (R², MSE, MAE)
- Actual vs Predicted visualization

### Correlation Matrix
- Correlation analysis of all numerical features
- Heatmap visualization

---
## 👥 Team Members and Contributions

### 🎯 Zeynep Ceren Kocaoğlu | 2021555041 | K-Means clustering, 3D/2D visualizations, Sankey diagram |

**1. Global Sales Map (Chart #7)**
- ✅ Implementation of 3 different projection types
- ✅ Coordinate mapping for 7 regions
- ✅ Interactive bubble size control
- ✅ Market share calculation algorithm
- ✅ Hover data optimization

**2. Sankey Flow Diagram (Chart #8)**
- ✅ Bidirectional flow system (Left-Right / Right-Left)
- ✅ 8 custom color definitions (fuel, transmission, sales)
- ✅ Normalize function (between 10-30)
- ✅ Percentage display toggle system
- ✅ Custom hover template

**3. Price Distribution Analysis (Chart #9)**
- ✅ Dual mode (Histogram / Histogram + KDE)
- ✅ Year-based filtering
- ✅ Adjustable bin count (10-100)
- ✅ Mean and median lines
- ✅ Statistical metric calculation (7 metrics)
- ✅ Outlier detection (IQR method)

**4. K-Means Clustering ML Model**
- ✅ 3D scatter visualization
- ✅ 2D PCA projection (optional)
- ✅ StandardScaler normalization
- ✅ Silhouette score calculation
- ✅ Automatic profile determination (4 segments)
- ✅ Expandable metric cards

**5. README Documentation**
- ✅ Project architecture documentation
- ✅ Installation and usage guide
- ✅ Technical specifications
- ✅ Team task distribution
- ✅ Visual examples and descriptions

---

### 💻 Melisa Mehenktas | 2021555044 | Scatter plot, Bar chart, Parallel coordinates, Geographic map |

**1. Dashboard Infrastructure and Design**
- ✅ Streamlit page configuration
- ✅ Custom CSS design system
  - Gradient animated header (BMW colors)
  - Sidebar gradient background
  - KPI card styles
  - Automatic insight cards
- ✅ Wide layout optimization
- ✅ Responsive design principles

**2. Data Management and Preprocessing**
- ✅ CSV reading and cache mechanism
- ✅ Data type conversions (numeric conversion)
- ✅ Missing value management (dropna strategy)
- ✅ Error handling

**3. Filter System**
- ✅ Implementation of 4 different filter types:
  - Year range slider (min-max)
  - Multiple region selection (multiselect)
  - Top 10 model selection
  - Fuel type selection
- ✅ Filter application logic (boolean masking)
- ✅ Dynamic filtered row count display

**4. Price vs Mileage Scatter (Chart #4)**
- ✅ Multi-dimensional scatter plot
- ✅ Color: Fuel type
- ✅ Size: Sales volume
- ✅ Hover data: 6 features
- ✅ Custom marker styling (black border, opacity)

**5. Best Selling Models Bar (Chart #5)**
- ✅ Horizontal bar chart
- ✅ Descending order
- ✅ Gradient coloring (Blues)
- ✅ External text display (thousand separator)

**6. Parallel Coordinates (Chart #6)**
- ✅ 3-dimensional parallel coordinates
- ✅ Sampling (1000 records)
- ✅ Viridis color scale
- ✅ Custom dimension labels

---

### 📊 Mustafa Yılmaz | 2021555071 | Data preparation, Line chart, Treemap, Box plot, Random Forest model |

**1. Dataset Preparation**
- ✅ Finding and sourcing BMW sales data (2010-2024)
- ✅ Data quality assessment
- ✅ Missing value analysis
- ✅ Data verification and validation
- ✅ Feature engineering


**2. Annual Sales Trend Chart (Chart #1)**
- ✅ Time series analysis
- ✅ Annual total calculation (groupby)
- ✅ Line chart implementation
- ✅ Custom hover template

**3. Regional Treemap (Chart #2)**
- ✅ Hierarchical data structure (Region → Model)
- ✅ 2-level treemap
- ✅ Sizing based on sales volume

**4. Price Box Plot (Chart #3)**
- ✅ Grouping by fuel type
- ✅ Outlier detection
- ✅ Quartile calculation

**5. Random Forest Price Prediction Model**
- ✅ Feature selection (4 features)
- ✅ Categorical encoding with LabelEncoder
- ✅ Train-test split (80/20)
- ✅ Random Forest Regressor training:
  - n_estimators=200
  - random_state=42
- ✅ Model evaluation:
  - R² Score
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
- ✅ Actual vs Predicted scatter plot
- ✅ Perfect prediction line (y=x)

**6. Correlation Matrix**
- ✅ Correlation calculation of all numerical features
- ✅ Heatmap visualization
- ✅ RdBu color scale (-1 to 1)
- ✅ Interpretation text

**ML Model Details:**
```python
# Random Forest Configuration
model = RandomForestRegressor(
    n_estimators=200,       # 200 decision trees
    random_state=42,        # Reproducibility
    max_depth=None,         # Unlimited depth
    min_samples_split=2,    # Default
    min_samples_leaf=1      # Default
)

# Performance Metrics
R² Score: Model explanatory power (0-1)
MSE: Mean squared error
MAE: Mean absolute error (interpretable)
```
---
