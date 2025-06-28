"""
Bank Marketing Dataset Dashboard
Comprehensive analysis dashboard combining EDA, Classification, and Clustering results
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, silhouette_score
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Bank Marketing Analysis Dashboard"

# Custom CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f8f9fa;
            }
            .main-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .section-header {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                font-size: 1.2em;
            }
            .card {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                margin: 1rem;
                padding: 1.5rem;
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-align: center;
                border-radius: 12px;
                padding: 1rem;
                margin: 0.5rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Load and preprocess data
try:
    df = pd.read_csv('bank-full.csv')
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
except FileNotFoundError:
    # Create sample data if file not found
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'job': np.random.choice(['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar'], n_samples),
        'marital': np.random.choice(['single', 'married', 'divorced'], n_samples),
        'education': np.random.choice(['primary', 'secondary', 'tertiary'], n_samples),
        'default': np.random.choice(['no', 'yes'], n_samples, p=[0.9, 0.1]),
        'balance': np.random.randint(-1000, 10000, n_samples),
        'housing': np.random.choice(['no', 'yes'], n_samples),
        'loan': np.random.choice(['no', 'yes'], n_samples),
        'contact': np.random.choice(['cellular', 'telephone', 'unknown'], n_samples),
        'day': np.random.randint(1, 32, n_samples),
        'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun'], n_samples),
        'duration': np.random.randint(0, 1000, n_samples),
        'campaign': np.random.randint(1, 10, n_samples),
        'pdays': np.random.randint(-1, 365, n_samples),
        'previous': np.random.randint(0, 5, n_samples),
        'poutcome': np.random.choice(['success', 'failure', 'unknown'], n_samples),
        'y': np.random.choice(['no', 'yes'], n_samples, p=[0.7, 0.3])
    })
    print("Using sample data!")

# Prepare data for machine learning
def prepare_ml_data(df):
    df_ml = df.copy()
    
    # Encode categorical variables
    le_dict = {}
    categorical_cols = df_ml.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le_dict[col] = LabelEncoder()
        df_ml[col] = le_dict[col].fit_transform(df_ml[col])
    
    # Separate features and target
    X = df_ml.drop('y', axis=1)
    y = df_ml['y']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, le_dict

# Prepare data
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, le_dict = prepare_ml_data(df)

# Train models
models = {}

# Logistic Regression
models['Logistic Regression'] = LogisticRegression(random_state=42)
models['Logistic Regression'].fit(X_train_scaled, y_train)

# Random Forest
models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
models['Random Forest'].fit(X_train, y_train)

# KNN
models['KNN'] = KNeighborsClassifier(n_neighbors=5)
models['KNN'].fit(X_train_scaled, y_train)

# Calculate model performance
model_performance = {}
roc_data = {}

for name, model in models.items():
    if name == 'Logistic Regression' or name == 'KNN':
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    model_performance[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

# Perform clustering analysis
# Elbow method analysis
k_range = range(1, 8)
sse_scores = []
silhouette_scores = []

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_train_scaled)
    sse_scores.append(kmeans_temp.inertia_)
    
    if k > 1:
        score = silhouette_score(X_train_scaled, kmeans_temp.labels_)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)

# Find optimal K
best_k = silhouette_scores[1:].index(max(silhouette_scores[1:])) + 2

# Apply optimal K-Means
kmeans = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_train_scaled)

# PCA for visualization
pca = PCA(n_components=3)  # 3D for better visualization
X_pca = pca.fit_transform(X_train_scaled)

# t-SNE for additional visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_train_scaled[:1000])  # Sample for speed

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Bank Marketing Dataset Analysis Dashboard", style={'margin': '0', 'fontSize': '2.5em'}),
        html.H3("Comprehensive EDA, Classification & Clustering Analysis", style={'margin': '10px 0 0 0', 'fontWeight': 'normal', 'opacity': '0.9'})
    ], className='main-header'),
    
    # Navigation tabs
    dcc.Tabs(id="main-tabs", value='eda-tab', children=[
        dcc.Tab(label='📊 Exploratory Data Analysis', value='eda-tab', style={'padding': '10px', 'fontSize': '16px'}),
        dcc.Tab(label='🤖 Machine Learning Models', value='ml-tab', style={'padding': '10px', 'fontSize': '16px'}),
        dcc.Tab(label='🎯 Clustering Analysis', value='cluster-tab', style={'padding': '10px', 'fontSize': '16px'}),
    ], style={'margin': '0 1rem'}),
    
    # Content area
    html.Div(id='tab-content')
])

# Callback for tab content
@app.callback(Output('tab-content', 'children'), Input('main-tabs', 'value'))
def render_content(active_tab):
    if active_tab == 'eda-tab':
        return render_eda_content()
    elif active_tab == 'ml-tab':
        return render_ml_content()
    elif active_tab == 'cluster-tab':
        return render_cluster_content()

def render_eda_content():
    # Dataset overview
    dataset_info = html.Div([
        html.Div("📋 Dataset Overview", className='section-header'),
        html.Div([
            html.Div([
                html.H3(f"{df.shape[0]:,}", style={'margin': '0', 'fontSize': '2em'}),
                html.P("Total Records", style={'margin': '5px 0'})
            ], className='metric-card', style={'flex': '1'}),
            html.Div([
                html.H3(f"{df.shape[1]}", style={'margin': '0', 'fontSize': '2em'}),
                html.P("Features", style={'margin': '5px 0'})
            ], className='metric-card', style={'flex': '1'}),
            html.Div([
                html.H3(f"{df['y'].value_counts()['yes']:.0f}", style={'margin': '0', 'fontSize': '2em'}),
                html.P("Positive Cases", style={'margin': '5px 0'})
            ], className='metric-card', style={'flex': '1'}),
            html.Div([
                html.H3(f"{(df['y'].value_counts()['yes']/len(df)*100):.1f}%", style={'margin': '0', 'fontSize': '2em'}),
                html.P("Success Rate", style={'margin': '5px 0'})
            ], className='metric-card', style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '1rem', 'margin': '1rem'})
    ])
    
    # Age distribution
    age_fig = px.histogram(df, x='age', nbins=30, title='Age Distribution',
                          color_discrete_sequence=['#667eea'])
    age_fig.add_vline(x=df['age'].mean(), line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {df['age'].mean():.1f}")
    age_fig.add_vline(x=df['age'].median(), line_dash="dash", line_color="green", 
                     annotation_text=f"Median: {df['age'].median():.1f}")
    age_fig.update_layout(showlegend=False, height=400)
    
    # Target variable distribution
    target_fig = px.pie(df, names='y', title='Target Variable Distribution (Term Deposit Subscription)',
                       color_discrete_sequence=['#f093fb', '#667eea'])
    target_fig.update_layout(height=400)
    
    # Numerical features distributions
    numerical_features = ['balance', 'duration', 'campaign', 'previous']
    numerical_figs = []
    
    for feature in numerical_features:
        fig = px.histogram(df, x=feature, nbins=30, title=f'{feature.title()} Distribution',
                          color_discrete_sequence=['#764ba2'])
        fig.add_vline(x=df[feature].mean(), line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {df[feature].mean():.1f}")
        fig.add_vline(x=df[feature].median(), line_dash="dash", line_color="green", 
                     annotation_text=f"Median: {df[feature].median():.1f}")
        fig.update_layout(height=350)
        numerical_figs.append(fig)
    
    # Job distribution
    job_counts = df['job'].value_counts()
    job_fig = px.bar(x=job_counts.index, y=job_counts.values, title='Job Distribution',
                     color_discrete_sequence=['#764ba2'])
    job_fig.update_layout(xaxis_title='Job Type', yaxis_title='Count', height=400)
    
    # Education vs Target
    edu_target = pd.crosstab(df['education'], df['y'])
    edu_fig = px.bar(edu_target, title='Education Level vs Target Variable',
                     color_discrete_sequence=['#f093fb', '#667eea'])
    edu_fig.update_layout(height=400)
    
    # Binary features analysis
    binary_features = ['default', 'housing', 'loan']
    binary_figs = []
    
    for feature in binary_features:
        counts = df[feature].value_counts()
        fig = px.bar(x=counts.index, y=counts.values, title=f'{feature.title()} Distribution',
                    color_discrete_sequence=['#f5576c'])
        # Add percentage annotations
        total = len(df)
        for i, (label, count) in enumerate(counts.items()):
            percentage = (count/total)*100
            fig.add_annotation(x=i, y=count, text=f"{count:,}<br>({percentage:.1f}%)",
                             showarrow=False, yshift=10)
        fig.update_layout(height=350)
        binary_figs.append(fig)
    
    # Box plots for numerical features by target
    box_figs = []
    for feature in ['age'] + numerical_features:
        fig = px.box(df, x='y', y=feature, title=f'{feature.title()} by Target Variable',
                    color='y', color_discrete_sequence=['#f093fb', '#667eea'])
        fig.update_layout(height=350)
        box_figs.append(fig)
    
    # Subscription rates by categorical features
    categorical_features = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    subscription_figs = []
    
    for feature in categorical_features:
        cross_tab = pd.crosstab(df[feature], df['y'], normalize='index') * 100
        fig = px.bar(x=cross_tab.index, y=cross_tab['yes'], 
                    title=f'Subscription Rate by {feature.title()}',
                    color_discrete_sequence=['#667eea'])
        fig.update_layout(xaxis_title=feature.title(), yaxis_title='Subscription Rate (%)', height=350)
        subscription_figs.append(fig)
    
    # Balance distribution by target
    balance_fig = px.box(df, x='y', y='balance', title='Account Balance Distribution by Target',
                        color='y', color_discrete_sequence=['#f093fb', '#667eea'])
    balance_fig.update_layout(height=400)
    
    # Duration vs Target
    duration_fig = px.box(df, x='y', y='duration', title='Call Duration Distribution by Target',
                         color='y', color_discrete_sequence=['#f093fb', '#667eea'])
    duration_fig.update_layout(height=400)
    
    # Month distribution
    month_counts = df['month'].value_counts()
    month_fig = px.bar(x=month_counts.index, y=month_counts.values, title='Campaign Distribution by Month',
                      color_discrete_sequence=['#764ba2'])
    month_fig.update_layout(xaxis_title='Month', yaxis_title='Count', height=400)
    
    # Contact method effectiveness
    contact_target = pd.crosstab(df['contact'], df['y'])
    contact_fig = px.bar(contact_target, title='Contact Method vs Success Rate',
                        color_discrete_sequence=['#f093fb', '#667eea'])
    contact_fig.update_layout(height=400)
    
    # Correlation analysis
    df_corr = df.copy()
    df_corr['y_encoded'] = df_corr['y'].map({'yes': 1, 'no': 0})
    numerical_cols = ['age', 'balance', 'duration', 'campaign', 'previous']
    correlations = df_corr[numerical_cols + ['y_encoded']].corr()['y_encoded'].drop('y_encoded')
    correlations = correlations.sort_values(key=abs, ascending=False)
    
    corr_fig = px.bar(x=correlations.index, y=correlations.values, 
                     title='Correlation of Numerical Features with Target',
                     color=[v if v > 0 else v for v in correlations.values],
                     color_continuous_scale=['red', 'white', 'green'])
    corr_fig.update_layout(height=400, yaxis_title='Correlation Coefficient')
    
    return html.Div([
        dataset_info,
        
        html.Div("📈 Basic Data Distributions", className='section-header'),
        html.Div([
            html.Div([dcc.Graph(figure=age_fig)], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=target_fig)], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        
        html.Div("🔢 Numerical Features Distributions", className='section-header'),
        html.Div([
            html.Div([dcc.Graph(figure=numerical_figs[0])], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=numerical_figs[1])], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        html.Div([
            html.Div([dcc.Graph(figure=numerical_figs[2])], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=numerical_figs[3])], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        
        html.Div("🏢 Categorical Features", className='section-header'),
        html.Div([
            html.Div([dcc.Graph(figure=job_fig)], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=edu_fig)], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        
        html.Div("✅ Binary Features Analysis", className='section-header'),
        html.Div([
            html.Div([dcc.Graph(figure=binary_figs[0])], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=binary_figs[1])], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=binary_figs[2])], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        
        html.Div("� Features vs Target Variable (Box Plots)", className='section-header'),
        html.Div([
            html.Div([dcc.Graph(figure=box_figs[0])], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=box_figs[1])], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=box_figs[2])], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        html.Div([
            html.Div([dcc.Graph(figure=box_figs[3])], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=box_figs[4])], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        
        html.Div("🎯 Subscription Rates by Categories", className='section-header'),
        html.Div([
            html.Div([dcc.Graph(figure=subscription_figs[0])], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=subscription_figs[1])], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=subscription_figs[2])], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        html.Div([
            html.Div([dcc.Graph(figure=subscription_figs[3])], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=subscription_figs[4])], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=subscription_figs[5])], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        
        html.Div("�💰 Financial Analysis", className='section-header'),
        html.Div([
            html.Div([dcc.Graph(figure=balance_fig)], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=duration_fig)], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        
        html.Div("📅 Temporal & Contact Analysis", className='section-header'),
        html.Div([
            html.Div([dcc.Graph(figure=month_fig)], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=contact_fig)], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        
        html.Div("🔗 Correlation Analysis", className='section-header'),
        html.Div([
            html.Div([dcc.Graph(figure=corr_fig)], className='card')
        ])
    ])

def render_ml_content():
    # Model performance comparison
    performance_df = pd.DataFrame(model_performance).T
    
    # Performance metrics chart
    metrics_fig = go.Figure()
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#667eea', '#f093fb', '#764ba2', '#f5576c']
    
    for i, metric in enumerate(metrics):
        metrics_fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=list(performance_df.index),
            y=performance_df[metric],
            marker_color=colors[i]
        ))
    
    metrics_fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        height=500,
        yaxis_title='Score'
    )
    
    # ROC Curves
    roc_fig = go.Figure()
    
    for name, data in roc_data.items():
        roc_fig.add_trace(go.Scatter(
            x=data['fpr'], 
            y=data['tpr'],
            mode='lines',
            name=f"{name} (AUC = {data['auc']:.3f})",
            line=dict(width=2)
        ))
    
    # Add diagonal reference line
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    roc_fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        showlegend=True
    )
    
    # Feature importance (Random Forest)
    rf_model = models['Random Forest']
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    importance_fig = px.bar(importance_df.tail(10), x='importance', y='feature',
                           title='Top 10 Feature Importances (Random Forest)',
                           orientation='h',
                           color_discrete_sequence=['#667eea'])
    importance_fig.update_layout(height=500)
    
    # Individual confusion matrices for all models
    cm_figs = []
    for name, model in models.items():
        if name == 'Logistic Regression' or name == 'KNN':
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        
        # Create confusion matrix with proper labels
        cm_fig = px.imshow(cm, 
                          text_auto=True, 
                          aspect="auto",
                          title=f'Confusion Matrix - {name}',
                          color_continuous_scale='Blues',
                          x=['Predicted No', 'Predicted Yes'],
                          y=['Actual No', 'Actual Yes'])
        
        # Add text annotations with percentages
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = (cm[i, j] / total) * 100
                cm_fig.add_annotation(
                    x=j, y=i,
                    text=f"{cm[i, j]}<br>({percentage:.1f}%)",
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
                )
        
        cm_fig.update_layout(height=400)
        cm_figs.append(cm_fig)
    
    # Detailed metrics table
    detailed_metrics = []
    for name, metrics in model_performance.items():
        detailed_metrics.append({
            'Model': name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}",
            'AUC': f"{roc_data[name]['auc']:.4f}"
        })
    
    metrics_table = pd.DataFrame(detailed_metrics)
    
    # Best model identification
    best_model_name = max(model_performance.keys(), key=lambda x: model_performance[x]['f1'])
    best_metrics = model_performance[best_model_name]
    
    return html.Div([
        html.Div("🤖 Machine Learning Results", className='section-header'),
        
        # Model performance cards
        html.Div([
            html.Div([
                html.H3(f"{model_performance[name]['accuracy']:.3f}", style={'margin': '0', 'fontSize': '1.8em'}),
                html.P(f"{name} Accuracy", style={'margin': '5px 0'})
            ], className='metric-card', style={'flex': '1'}) for name in models.keys()
        ], style={'display': 'flex', 'gap': '1rem', 'margin': '1rem'}),
        
        # Performance comparison and ROC curves
        html.Div([
            html.Div([dcc.Graph(figure=metrics_fig)], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=roc_fig)], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        
        # Feature importance
        html.Div([
            html.Div([dcc.Graph(figure=importance_fig)], className='card')
        ]),
        
        # Detailed metrics table
        html.Div("📊 Detailed Performance Metrics", className='section-header'),
        html.Div([
            html.Table([
                html.Thead([
                    html.Tr([html.Th(col, style={'padding': '10px', 'textAlign': 'center'}) for col in metrics_table.columns])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(metrics_table.iloc[i][col], style={'padding': '8px', 'textAlign': 'center'}) 
                        for col in metrics_table.columns
                    ]) for i in range(len(metrics_table))
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'})
        ], className='card'),
        
        # Confusion matrices
        html.Div("🎯 Confusion Matrices", className='section-header'),
        html.Div([
            html.Div([dcc.Graph(figure=cm_figs[0])], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=cm_figs[1])], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        html.Div([
            html.Div([dcc.Graph(figure=cm_figs[2])], className='card')
        ]),
        
        # Best model summary
        html.Div([
            html.H3(f"🏆 Best Performing Model: {best_model_name}"),
            html.Div([
                html.Div([
                    html.H4(f"{best_metrics['f1']:.4f}", style={'margin': '0', 'color': '#667eea'}),
                    html.P("F1-Score", style={'margin': '5px 0'})
                ], style={'textAlign': 'center', 'flex': '1'}),
                html.Div([
                    html.H4(f"{best_metrics['accuracy']:.4f}", style={'margin': '0', 'color': '#f093fb'}),
                    html.P("Accuracy", style={'margin': '5px 0'})
                ], style={'textAlign': 'center', 'flex': '1'}),
                html.Div([
                    html.H4(f"{best_metrics['precision']:.4f}", style={'margin': '0', 'color': '#764ba2'}),
                    html.P("Precision", style={'margin': '5px 0'})
                ], style={'textAlign': 'center', 'flex': '1'}),
                html.Div([
                    html.H4(f"{best_metrics['recall']:.4f}", style={'margin': '0', 'color': '#f5576c'}),
                    html.P("Recall", style={'margin': '5px 0'})
                ], style={'textAlign': 'center', 'flex': '1'}),
                html.Div([
                    html.H4(f"{roc_data[best_model_name]['auc']:.4f}", style={'margin': '0', 'color': '#667eea'}),
                    html.P("AUC", style={'margin': '5px 0'})
                ], style={'textAlign': 'center', 'flex': '1'})
            ], style={'display': 'flex', 'gap': '2rem', 'padding': '1rem'})
        ], className='card')
    ])

def render_cluster_content():
    # Elbow method plot
    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(x=list(k_range), y=sse_scores, mode='lines+markers',
                                  name='SSE', line=dict(color='#667eea', width=3),
                                  marker=dict(size=8)))
    elbow_fig.update_layout(title='Elbow Method for Optimal K',
                           xaxis_title='Number of Clusters (K)',
                           yaxis_title='Sum of Squared Errors (SSE)',
                           height=400)
    
    # Silhouette scores plot
    silhouette_fig = go.Figure()
    silhouette_fig.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers',
                                       name='Silhouette Score', line=dict(color='#f093fb', width=3),
                                       marker=dict(size=8)))
    silhouette_fig.update_layout(title='Silhouette Analysis for Optimal K',
                                xaxis_title='Number of Clusters (K)',
                                yaxis_title='Silhouette Score',
                                height=400)
    
    # 2D PCA cluster visualization
    cluster_2d_fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=cluster_labels,
                               title='K-Means Clustering Results (PCA 2D Visualization)',
                               labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
                               color_continuous_scale='viridis')
    cluster_2d_fig.update_layout(height=500)
    
    # 3D PCA cluster visualization
    cluster_3d_fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], 
                                  color=cluster_labels,
                                  title='K-Means Clustering Results (PCA 3D Visualization)',
                                  labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
                                  color_continuous_scale='viridis')
    cluster_3d_fig.update_layout(height=600)
    
    # t-SNE visualization (using subset for speed)
    cluster_labels_subset = cluster_labels[:1000]
    tsne_fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=cluster_labels_subset,
                         title='t-SNE Clustering Visualization (Sample)',
                         labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
                         color_continuous_scale='viridis')
    tsne_fig.update_layout(height=500)
    
    # PCA explained variance
    pca_variance = pca.explained_variance_ratio_
    pca_fig = px.bar(x=[f'PC{i+1}' for i in range(len(pca_variance))], y=pca_variance,
                     title='PCA Explained Variance Ratio',
                     color_discrete_sequence=['#667eea'])
    pca_fig.update_layout(height=400, yaxis_title='Explained Variance Ratio')
    
    # Cluster size distribution
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_size_fig = px.pie(values=cluster_sizes.values, 
                             names=[f'Cluster {i}' for i in cluster_sizes.index],
                             title=f'Cluster Size Distribution (K={best_k})',
                             color_discrete_sequence=['#667eea', '#f093fb', '#764ba2', '#f5576c'])
    cluster_size_fig.update_layout(height=400)
    
    # Feature correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    heatmap_fig = px.imshow(correlation_matrix,
                           title='Feature Correlation Heatmap',
                           color_continuous_scale='RdBu_r',
                           aspect="auto")
    heatmap_fig.update_layout(height=600)
    
    # Cluster characteristics analysis
    cluster_chars = []
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = np.nan
    df_with_clusters.iloc[:len(cluster_labels), df_with_clusters.columns.get_loc('cluster')] = cluster_labels
    
    for cluster_id in range(best_k):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            cluster_chars.append({
                'Cluster': f'Cluster {cluster_id}',
                'Size': len(cluster_data),
                'Avg Age': f"{cluster_data['age'].mean():.1f}",
                'Avg Balance': f"{cluster_data['balance'].mean():.0f}",
                'Avg Duration': f"{cluster_data['duration'].mean():.1f}",
                'Success Rate': f"{(cluster_data['y'] == 'yes').mean()*100:.1f}%"
            })
    
    cluster_table = pd.DataFrame(cluster_chars)
    
    return html.Div([
        html.Div("🎯 Clustering & Dimensionality Reduction", className='section-header'),
        
        # Cluster summary cards
        html.Div([
            html.Div([
                html.H3(f"{best_k}", style={'margin': '0', 'fontSize': '2em'}),
                html.P("Optimal K (Clusters)", style={'margin': '5px 0'})
            ], className='metric-card', style={'flex': '1'}),
            html.Div([
                html.H3("3", style={'margin': '0', 'fontSize': '2em'}),
                html.P("PCA Components", style={'margin': '5px 0'})
            ], className='metric-card', style={'flex': '1'}),
            html.Div([
                html.H3(f"{sum(pca_variance):.1%}", style={'margin': '0', 'fontSize': '2em'}),
                html.P("Variance Explained", style={'margin': '5px 0'})
            ], className='metric-card', style={'flex': '1'}),
            html.Div([
                html.H3(f"{max(silhouette_scores[1:]):.3f}", style={'margin': '0', 'fontSize': '2em'}),
                html.P("Best Silhouette Score", style={'margin': '5px 0'})
            ], className='metric-card', style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '1rem', 'margin': '1rem'}),
        
        # Optimal K analysis
        html.Div("📈 Optimal Cluster Number Analysis", className='section-header'),
        html.Div([
            html.Div([dcc.Graph(figure=elbow_fig)], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=silhouette_fig)], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        
        # Cluster visualizations
        html.Div("🔍 Cluster Visualizations", className='section-header'),
        html.Div([
            html.Div([dcc.Graph(figure=cluster_2d_fig)], className='card')
        ]),
        
        html.Div([
            html.Div([dcc.Graph(figure=cluster_3d_fig)], className='card')
        ]),
        
        html.Div([
            html.Div([dcc.Graph(figure=tsne_fig)], className='card', style={'flex': '1'}),
            html.Div([dcc.Graph(figure=cluster_size_fig)], className='card', style={'flex': '1'})
        ], style={'display': 'flex'}),
        
        # PCA analysis
        html.Div("📊 Principal Component Analysis", className='section-header'),
        html.Div([
            html.Div([dcc.Graph(figure=pca_fig)], className='card')
        ]),
        
        # Cluster characteristics
        html.Div("📋 Cluster Characteristics", className='section-header'),
        html.Div([
            html.Table([
                html.Thead([
                    html.Tr([html.Th(col, style={'padding': '10px', 'textAlign': 'center'}) for col in cluster_table.columns])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(cluster_table.iloc[i][col], style={'padding': '8px', 'textAlign': 'center'}) 
                        for col in cluster_table.columns
                    ]) for i in range(len(cluster_table))
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': '1px solid #ddd'})
        ], className='card'),
        
        # Correlation analysis
        html.Div("🔗 Feature Correlations", className='section-header'),
        html.Div([
            html.Div([dcc.Graph(figure=heatmap_fig)], className='card')
        ])
    ])

if __name__ == '__main__':
    print("Starting Bank Marketing Dashboard...")
    print("Dashboard will be available at: http://127.0.0.1:8050")
    app.run_server(debug=True, host='127.0.0.1', port=8050)
