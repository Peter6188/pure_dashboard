# Bank Marketing Dataset Dashboard

A comprehensive interactive dashboard that combines all analysis results from the three Jupyter notebooks (Part1, Part2, Part3) into a single, professional web application using Dash.

## Features

### 📊 Exploratory Data Analysis (EDA)
- Dataset overview with key metrics
- Age and demographic distributions
- Target variable analysis
- Job and education breakdowns
- Financial analysis (account balance, call duration)
- Interactive visualizations with Plotly

### 🤖 Machine Learning Models
- Multiple classification algorithms:
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors
- Performance comparison across all metrics
- Feature importance analysis
- Confusion matrix visualization
- Best model identification

### 🎯 Clustering & Dimensionality Reduction
- K-Means clustering with PCA visualization
- Principal Component Analysis (PCA)
- Cluster size distribution
- Feature correlation heatmap
- Explained variance analysis

## Installation & Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   - Place your `bank-full.csv` file in the same directory as `dashboard_app.py`
   - If the file is not found, the app will generate sample data for demonstration

3. **Run the dashboard:**
   ```bash
   python dashboard_app.py
   ```

4. **Access the dashboard:**
   - Open your web browser and go to: `http://127.0.0.1:8050`

## Dashboard Structure

The dashboard is organized into three main tabs:

### Tab 1: Exploratory Data Analysis
- **Dataset Overview Cards**: Total records, features, positive cases, success rate
- **Data Distributions**: Age histogram, target variable pie chart
- **Categorical Analysis**: Job distribution, education vs target
- **Financial Analysis**: Account balance and call duration by target

### Tab 2: Machine Learning Models
- **Performance Metrics Cards**: Accuracy for each model
- **Comparison Charts**: Side-by-side performance comparison
- **Feature Importance**: Top 10 most important features from Random Forest
- **Confusion Matrix**: For the best performing model
- **Best Model Summary**: Detailed metrics for the top performer

### Tab 3: Clustering Analysis
- **Cluster Summary Cards**: Number of clusters, PCA components, variance explained
- **Cluster Visualization**: 2D scatter plot with PCA reduction
- **PCA Analysis**: Explained variance by component
- **Cluster Distribution**: Pie chart of cluster sizes
- **Correlation Heatmap**: Feature relationships

## Technical Details

### Data Processing
- Automatic handling of categorical variables with Label Encoding
- Feature scaling for distance-based algorithms
- Train/test split with stratification
- Missing value detection and handling

### Machine Learning Pipeline
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble method with 100 trees
- **K-Nearest Neighbors**: Instance-based learning with k=5
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score

### Clustering Pipeline
- **K-Means**: 3 clusters with random state for reproducibility
- **PCA**: Dimensionality reduction to 2 components
- **Visualization**: Interactive scatter plots with cluster coloring

### Styling & UX
- **Modern Design**: Gradient backgrounds and card-based layout
- **Responsive Layout**: Flexible grid system
- **Interactive Charts**: Hover effects and zoom capabilities
- **Professional Color Scheme**: Consistent purple-pink gradient theme

## File Structure
```
Pure_dashboard/
├── dashboard_app.py      # Main dashboard application
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── Part1.ipynb          # Original EDA notebook
├── Part2.ipynb          # Original ML notebook
├── Part3.ipynb          # Original clustering notebook
└── bank-full.csv        # Dataset (if available)
```

## Customization

### Adding New Visualizations
To add new charts, modify the respective render functions:
- `render_eda_content()` for EDA visualizations
- `render_ml_content()` for ML analysis
- `render_cluster_content()` for clustering analysis

### Styling Changes
Modify the CSS in the `app.index_string` section to change colors, fonts, or layout.

### Model Parameters
Adjust model parameters in the model training section:
```python
# Example: Change Random Forest parameters
models['Random Forest'] = RandomForestClassifier(
    n_estimators=200,  # Increase trees
    max_depth=10,      # Limit depth
    random_state=42
)
```

## Troubleshooting

### Common Issues
1. **Missing data file**: App will use sample data if `bank-full.csv` is not found
2. **Package conflicts**: Use a virtual environment to avoid conflicts
3. **Port already in use**: Change the port in `app.run_server(port=8051)`

### Performance Optimization
- For large datasets, consider sampling or data aggregation
- Use `debug=False` in production
- Implement caching for expensive computations

## Credits

This dashboard consolidates analysis from three separate Jupyter notebooks into a unified web application, providing an interactive and professional way to explore bank marketing data insights.
# pure_dashboard
