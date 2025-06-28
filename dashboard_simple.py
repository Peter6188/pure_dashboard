"""
Bank Marketing Dataset Dashboard - Working Version
Comprehensive analysis dashboard combining EDA, Classification, and Clustering results
"""

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Bank Marketing Analysis Dashboard"

# Load and preprocess data
print("Loading data...")
try:
    df = pd.read_csv('bank-full.csv')
    print(f"Dataset loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print("CSV file not found, creating sample data...")
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
print("Preparing data for ML...")
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
print("Data preparation completed!")

# Train models
print("Training models...")
models = {}

# Logistic Regression
models['Logistic Regression'] = LogisticRegression(random_state=42)
models['Logistic Regression'].fit(X_train_scaled, y_train)

# Random Forest
models['Random Forest'] = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduced for speed
models['Random Forest'].fit(X_train, y_train)

# KNN
models['KNN'] = KNeighborsClassifier(n_neighbors=5)
models['KNN'].fit(X_train_scaled, y_train)

print("Models trained!")

# Calculate model performance with more detailed metrics
model_performance = {}
model_predictions = {}
model_probabilities = {}

for name, model in models.items():
    if name == 'Logistic Regression' or name == 'KNN':
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    model_performance[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    model_predictions[name] = y_pred
    model_probabilities[name] = y_pred_proba

print("Model evaluation completed!")

# Perform clustering with more detailed analysis
print("Performing clustering...")
# Determine optimal number of clusters using elbow method
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    cluster_labels_temp = kmeans_temp.fit_predict(X_train_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_train_scaled, cluster_labels_temp))

# Use 3 clusters as optimal
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_train_scaled)

# PCA for 2D and 3D visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_train_scaled)

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_train_scaled)

# t-SNE for additional dimensionality reduction
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_train_scaled[:500])  # Use smaller subset for speed
cluster_labels_tsne = cluster_labels[:500]

print("Clustering completed!")

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Bank Marketing Dataset Analysis Dashboard", 
                style={'margin': '0', 'fontSize': '2.5em', 'color': 'white'}),
        html.H3("Comprehensive EDA, Classification & Clustering Analysis", 
                style={'margin': '10px 0 0 0', 'fontWeight': 'normal', 'opacity': '0.9', 'color': 'white'})
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'color': 'white',
        'padding': '2rem',
        'textAlign': 'center',
        'marginBottom': '2rem',
        'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
    }),
    
    # Navigation tabs
    dcc.Tabs(id="main-tabs", value='eda-tab', children=[
        dcc.Tab(label='📊 Exploratory Data Analysis', value='eda-tab', 
                style={'padding': '10px', 'fontSize': '16px'}),
        dcc.Tab(label='🤖 Machine Learning Models', value='ml-tab', 
                style={'padding': '10px', 'fontSize': '16px'}),
        dcc.Tab(label='🎯 Clustering Analysis', value='cluster-tab', 
                style={'padding': '10px', 'fontSize': '16px'}),
    ], style={'margin': '0 1rem'}),
    
    # Content area
    html.Div(id='tab-content', style={'padding': '20px'})
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f8f9fa'})

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
        html.H2("📋 Dataset Overview", style={'color': '#667eea', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.H3(f"{df.shape[0]:,}", style={'margin': '0', 'fontSize': '2em', 'color': '#667eea'}),
                html.P("Total Records", style={'margin': '5px 0'})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '10px',
                'textAlign': 'center',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1',
                'margin': '10px'
            }),
            html.Div([
                html.H3(f"{df.shape[1]}", style={'margin': '0', 'fontSize': '2em', 'color': '#f093fb'}),
                html.P("Features", style={'margin': '5px 0'})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '10px',
                'textAlign': 'center',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1',
                'margin': '10px'
            }),
            html.Div([
                html.H3(f"{df['y'].value_counts()['yes']:.0f}", style={'margin': '0', 'fontSize': '2em', 'color': '#764ba2'}),
                html.P("Positive Cases", style={'margin': '5px 0'})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '10px',
                'textAlign': 'center',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1',
                'margin': '10px'
            }),
            html.Div([
                html.H3(f"{(df['y'].value_counts()['yes']/len(df)*100):.1f}%", style={'margin': '0', 'fontSize': '2em', 'color': '#f5576c'}),
                html.P("Success Rate", style={'margin': '5px 0'})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '10px',
                'textAlign': 'center',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1',
                'margin': '10px'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'})
    ])
    
    # Age distribution with mean and median
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
    
    balance_fig = px.histogram(df, x='balance', nbins=50, title='Balance Distribution',
                              color_discrete_sequence=['#764ba2'])
    balance_fig.add_vline(x=df['balance'].mean(), line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {df['balance'].mean():.0f}")
    balance_fig.add_vline(x=df['balance'].median(), line_dash="dash", line_color="green", 
                         annotation_text=f"Median: {df['balance'].median():.0f}")
    balance_fig.update_layout(height=350)
    
    duration_fig = px.histogram(df, x='duration', nbins=50, title='Call Duration Distribution',
                               color_discrete_sequence=['#f5576c'])
    duration_fig.add_vline(x=df['duration'].mean(), line_dash="dash", line_color="red", 
                          annotation_text=f"Mean: {df['duration'].mean():.0f}")
    duration_fig.add_vline(x=df['duration'].median(), line_dash="dash", line_color="green", 
                          annotation_text=f"Median: {df['duration'].median():.0f}")
    duration_fig.update_layout(height=350)
    
    campaign_fig = px.histogram(df, x='campaign', nbins=20, title='Campaign Contacts Distribution',
                               color_discrete_sequence=['#667eea'])
    campaign_fig.add_vline(x=df['campaign'].mean(), line_dash="dash", line_color="red", 
                          annotation_text=f"Mean: {df['campaign'].mean():.1f}")
    campaign_fig.update_layout(height=350)
    
    previous_fig = px.histogram(df, x='previous', nbins=15, title='Previous Contacts Distribution',
                               color_discrete_sequence=['#f093fb'])
    previous_fig.add_vline(x=df['previous'].mean(), line_dash="dash", line_color="red", 
                          annotation_text=f"Mean: {df['previous'].mean():.1f}")
    previous_fig.update_layout(height=350)
    
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
    
    default_counts = df['default'].value_counts()
    default_fig = px.bar(x=default_counts.index, y=default_counts.values, 
                        title='Credit Default Distribution',
                        color_discrete_sequence=['#f5576c'])
    total = len(df)
    for i, (label, count) in enumerate(default_counts.items()):
        percentage = (count/total)*100
        default_fig.add_annotation(x=i, y=count, text=f"{count:,}<br>({percentage:.1f}%)",
                                  showarrow=False, yshift=10)
    default_fig.update_layout(height=350)
    
    housing_counts = df['housing'].value_counts()
    housing_fig = px.bar(x=housing_counts.index, y=housing_counts.values, 
                        title='Housing Loan Distribution',
                        color_discrete_sequence=['#667eea'])
    for i, (label, count) in enumerate(housing_counts.items()):
        percentage = (count/total)*100
        housing_fig.add_annotation(x=i, y=count, text=f"{count:,}<br>({percentage:.1f}%)",
                                  showarrow=False, yshift=10)
    housing_fig.update_layout(height=350)
    
    loan_counts = df['loan'].value_counts()
    loan_fig = px.bar(x=loan_counts.index, y=loan_counts.values, 
                     title='Personal Loan Distribution',
                     color_discrete_sequence=['#764ba2'])
    for i, (label, count) in enumerate(loan_counts.items()):
        percentage = (count/total)*100
        loan_fig.add_annotation(x=i, y=count, text=f"{count:,}<br>({percentage:.1f}%)",
                               showarrow=False, yshift=10)
    loan_fig.update_layout(height=350)
    
    # Box plots for numerical features by target
    age_box_fig = px.box(df, x='y', y='age', title='Age Distribution by Target',
                        color='y', color_discrete_sequence=['#f093fb', '#667eea'])
    age_box_fig.update_layout(height=350)
    
    balance_box_fig = px.box(df, x='y', y='balance', title='Balance Distribution by Target',
                            color='y', color_discrete_sequence=['#f093fb', '#667eea'])
    balance_box_fig.update_layout(height=350)
    
    duration_box_fig = px.box(df, x='y', y='duration', title='Duration Distribution by Target',
                             color='y', color_discrete_sequence=['#f093fb', '#667eea'])
    duration_box_fig.update_layout(height=350)
    
    # Subscription rates by categorical features
    job_success = pd.crosstab(df['job'], df['y'], normalize='index') * 100
    job_success_fig = px.bar(x=job_success.index, y=job_success['yes'], 
                            title='Subscription Rate by Job Type',
                            color_discrete_sequence=['#667eea'])
    job_success_fig.update_layout(xaxis_title='Job Type', yaxis_title='Subscription Rate (%)', height=400)
    
    education_success = pd.crosstab(df['education'], df['y'], normalize='index') * 100
    education_success_fig = px.bar(x=education_success.index, y=education_success['yes'], 
                                  title='Subscription Rate by Education Level',
                                  color_discrete_sequence=['#f093fb'])
    education_success_fig.update_layout(xaxis_title='Education', yaxis_title='Subscription Rate (%)', height=350)
    
    # Monthly subscription analysis
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    if 'month' in df.columns:
        df_month = df.copy()
        df_month['month'] = pd.Categorical(df_month['month'], categories=month_order, ordered=True)
        month_success = pd.crosstab(df_month['month'], df_month['y'], normalize='index') * 100
        month_success_fig = px.bar(x=month_success.index, y=month_success['yes'], 
                                  title='Subscription Rate by Month',
                                  color_discrete_sequence=['#667eea'])
        month_success_fig.update_layout(xaxis_title='Month', yaxis_title='Subscription Rate (%)', height=400)
    else:
        month_success_fig = go.Figure()
        month_success_fig.add_annotation(text="Month data not available", xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
        month_success_fig.update_layout(title='Subscription Rate by Month', height=400)
    
    # Contact method analysis
    if 'contact' in df.columns:
        contact_success = pd.crosstab(df['contact'], df['y'], normalize='index') * 100
        contact_success_fig = px.bar(x=contact_success.index, y=contact_success['yes'], 
                                    title='Subscription Rate by Contact Method',
                                    color_discrete_sequence=['#f093fb'])
        contact_success_fig.update_layout(xaxis_title='Contact Method', yaxis_title='Subscription Rate (%)', height=350)
    else:
        contact_success_fig = go.Figure()
        contact_success_fig.add_annotation(text="Contact data not available", xref="paper", yref="paper",
                                          x=0.5, y=0.5, showarrow=False)
        contact_success_fig.update_layout(title='Subscription Rate by Contact Method', height=350)
    
    # Previous outcome analysis
    if 'poutcome' in df.columns:
        poutcome_success = pd.crosstab(df['poutcome'], df['y'], normalize='index') * 100
        poutcome_success_fig = px.bar(x=poutcome_success.index, y=poutcome_success['yes'], 
                                     title='Subscription Rate by Previous Campaign Outcome',
                                     color_discrete_sequence=['#764ba2'])
        poutcome_success_fig.update_layout(xaxis_title='Previous Outcome', yaxis_title='Subscription Rate (%)', height=350)
    else:
        poutcome_success_fig = go.Figure()
        poutcome_success_fig.add_annotation(text="Previous outcome data not available", xref="paper", yref="paper",
                                           x=0.5, y=0.5, showarrow=False)
        poutcome_success_fig.update_layout(title='Subscription Rate by Previous Campaign Outcome', height=350)
    
    # Age groups analysis
    df_temp = df.copy()
    df_temp['age_group'] = pd.cut(df_temp['age'], bins=[0, 30, 40, 50, 60, 100], 
                                 labels=['<30', '30-40', '40-50', '50-60', '60+'])
    age_group_success = pd.crosstab(df_temp['age_group'], df_temp['y'], normalize='index') * 100
    age_group_fig = px.bar(x=age_group_success.index, y=age_group_success['yes'], 
                          title='Subscription Rate by Age Group',
                          color_discrete_sequence=['#f5576c'])
    age_group_fig.update_layout(xaxis_title='Age Group', yaxis_title='Subscription Rate (%)', height=350)
    
    marital_success = pd.crosstab(df['marital'], df['y'], normalize='index') * 100
    marital_success_fig = px.bar(x=marital_success.index, y=marital_success['yes'], 
                                title='Subscription Rate by Marital Status',
                                color_discrete_sequence=['#764ba2'])
    marital_success_fig.update_layout(xaxis_title='Marital Status', yaxis_title='Subscription Rate (%)', height=350)
    
    return html.Div([
        dataset_info,
        
        html.H2("📈 Basic Data Distributions", style={'color': '#667eea', 'margin': '40px 0 20px 0'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=age_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=target_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        html.H2("🔢 Numerical Features Distributions", style={'color': '#667eea', 'margin': '40px 0 20px 0'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=balance_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=duration_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=campaign_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=previous_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        html.H2("🏢 Categorical Features", style={'color': '#667eea', 'margin': '40px 0 20px 0'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=job_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=edu_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        html.H2("✅ Binary Features Analysis", style={'color': '#667eea', 'margin': '40px 0 20px 0'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=default_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=housing_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=loan_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        html.H2("📊 Features vs Target Variable", style={'color': '#667eea', 'margin': '40px 0 20px 0'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=age_box_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=balance_box_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=duration_box_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        html.H2("🎯 Subscription Rates by Categories", style={'color': '#667eea', 'margin': '40px 0 20px 0'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=job_success_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'
            })
        ]),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=education_success_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=marital_success_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=age_group_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=contact_success_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=month_success_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=poutcome_success_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'})
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
    colors_roc = ['#667eea', '#f093fb', '#764ba2']
    
    for i, (name, model) in enumerate(models.items()):
        # Get probabilities
        y_proba = model_probabilities[name]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        roc_fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{name} (AUC = {roc_auc:.3f})',
            line=dict(color=colors_roc[i], width=2)
        ))
    
    # Add diagonal line
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', dash='dash')
    ))
    
    roc_fig.update_layout(
        title='ROC Curves for All Models',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    
    # Confusion Matrices
    confusion_figs = []
    for name, model in models.items():
        y_pred = model_predictions[name]
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations
        annotations = []
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                annotations.append(dict(
                    x=j, y=i,
                    text=f'{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)',
                    showarrow=False,
                    font=dict(color='white' if cm[i][j] > cm.max()/2 else 'black')
                ))
        
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No', 'Predicted Yes'],
            y=['Actual No', 'Actual Yes'],
            colorscale='Blues',
            showscale=True
        ))
        
        cm_fig.update_layout(
            title=f'Confusion Matrix - {name}',
            annotations=annotations,
            height=400,
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed')
        )
        
        confusion_figs.append((name, cm_fig))
    
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
    
    # Correlation matrix
    corr_matrix = X_train.corr()
    
    correlation_fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 8}
    ))
    correlation_fig.update_layout(
        title='Feature Correlation Matrix',
        height=600,
        width=800
    )
    
    # Model performance detailed table
    performance_table = []
    for name in models.keys():
        row = [name] + [f"{model_performance[name][metric]:.3f}" for metric in ['accuracy', 'precision', 'recall', 'f1']]
        performance_table.append(row)
    
    performance_table_fig = go.Figure(data=[go.Table(
        header=dict(values=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
                   fill_color='#667eea',
                   font=dict(color='white', size=12),
                   align="center"),
        cells=dict(values=list(zip(*performance_table)),
                  fill_color='#f8f9fa',
                  align="center",
                  font=dict(size=11))
    )])
    performance_table_fig.update_layout(title='Detailed Model Performance Metrics', height=300)
    
    # Best model summary
    best_model_name = max(model_performance.keys(), key=lambda x: model_performance[x]['f1'])
    
    return html.Div([
        html.H2("🤖 Machine Learning Results", style={'color': '#667eea', 'marginBottom': '20px'}),
        
        # Model performance cards
        html.Div([
            html.Div([
                html.H3(f"{model_performance[name]['accuracy']:.3f}", 
                       style={'margin': '0', 'fontSize': '1.8em', 'color': '#667eea'}),
                html.P(f"{name} Accuracy", style={'margin': '5px 0'})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '10px',
                'textAlign': 'center',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1',
                'margin': '10px'
            }) for name in models.keys()
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        # Performance comparison and ROC curves
        html.H2("📊 Model Performance Analysis", style={'color': '#667eea', 'margin': '40px 0 20px 0'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=metrics_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=roc_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        # Detailed performance table
        html.Div([
            html.Div([
                dcc.Graph(figure=performance_table_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'
            })
        ]),
        
        # Confusion matrices
        html.H2("🎯 Confusion Matrices", style={'color': '#667eea', 'margin': '40px 0 20px 0'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=cm_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }) for name, cm_fig in confusion_figs
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        # Feature analysis
        html.H2("🔍 Feature Analysis", style={'color': '#667eea', 'margin': '40px 0 20px 0'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=importance_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=correlation_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        # Best model summary
        html.Div([
            html.H3(f"🏆 Best Performing Model: {best_model_name}"),
            html.P(f"F1-Score: {model_performance[best_model_name]['f1']:.3f}"),
            html.P(f"Accuracy: {model_performance[best_model_name]['accuracy']:.3f}"),
            html.P(f"Precision: {model_performance[best_model_name]['precision']:.3f}"),
            html.P(f"Recall: {model_performance[best_model_name]['recall']:.3f}")
        ], style={
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'padding': '20px',
            'margin': '10px',
            'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'
        })
    ])

def render_cluster_content():
    # Elbow method visualization
    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers',
                                  name='Inertia', line=dict(color='#667eea')))
    elbow_fig.update_layout(title='Elbow Method for Optimal K',
                           xaxis_title='Number of Clusters (k)',
                           yaxis_title='Inertia',
                           height=400)
    
    # Silhouette score visualization
    silhouette_fig = go.Figure()
    silhouette_fig.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers',
                                       name='Silhouette Score', line=dict(color='#f093fb')))
    silhouette_fig.update_layout(title='Silhouette Score for Different K',
                                xaxis_title='Number of Clusters (k)',
                                yaxis_title='Silhouette Score',
                                height=400)
    
    # 2D PCA cluster visualization
    cluster_2d_fig = px.scatter(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], color=cluster_labels,
                               title='K-Means Clustering Results (2D PCA Visualization)',
                               labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
                               color_discrete_sequence=['#667eea', '#f093fb', '#764ba2'])
    cluster_2d_fig.update_layout(height=500)
    
    # 3D PCA cluster visualization
    cluster_3d_fig = px.scatter_3d(x=X_pca_3d[:, 0], y=X_pca_3d[:, 1], z=X_pca_3d[:, 2], 
                                  color=cluster_labels,
                                  title='K-Means Clustering Results (3D PCA Visualization)',
                                  labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
                                  color_discrete_sequence=['#667eea', '#f093fb', '#764ba2'])
    cluster_3d_fig.update_layout(height=600)
    
    # t-SNE visualization
    tsne_fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=cluster_labels_tsne,
                         title='K-Means Clustering Results (t-SNE Visualization)',
                         labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'},
                         color_discrete_sequence=['#667eea', '#f093fb', '#764ba2'])
    tsne_fig.update_layout(height=500)
    
    # PCA explained variance
    pca_variance_2d = pca_2d.explained_variance_ratio_
    pca_variance_3d = pca_3d.explained_variance_ratio_
    
    pca_variance_fig = go.Figure()
    pca_variance_fig.add_trace(go.Bar(x=['PC1', 'PC2'], y=pca_variance_2d,
                                     name='2D PCA', marker_color='#667eea'))
    pca_variance_fig.add_trace(go.Bar(x=['PC1', 'PC2', 'PC3'], y=pca_variance_3d,
                                     name='3D PCA', marker_color='#f093fb'))
    pca_variance_fig.update_layout(title='PCA Explained Variance Ratio',
                                  xaxis_title='Principal Components',
                                  yaxis_title='Explained Variance Ratio',
                                  height=400)
    
    # Cluster characteristics
    cluster_df = X_train.copy()
    cluster_df['Cluster'] = cluster_labels
    
    # Calculate cluster means for numerical features
    numerical_features = ['age', 'balance', 'duration', 'campaign', 'previous']
    cluster_means = []
    
    for cluster_id in sorted(cluster_df['Cluster'].unique()):
        cluster_data = cluster_df[cluster_df['Cluster'] == cluster_id]
        cluster_mean = {
            'Cluster': f'Cluster {cluster_id}',
            'Size': len(cluster_data),
            'Avg_Age': cluster_data['age'].mean(),
            'Avg_Balance': cluster_data['balance'].mean(),
            'Avg_Duration': cluster_data['duration'].mean(),
            'Avg_Campaign': cluster_data['campaign'].mean(),
            'Avg_Previous': cluster_data['previous'].mean()
        }
        cluster_means.append(cluster_mean)
    
    cluster_means_df = pd.DataFrame(cluster_means)
    
    # Cluster characteristics heatmap data
    heatmap_data = cluster_means_df.set_index('Cluster')[['Avg_Age', 'Avg_Balance', 'Avg_Duration', 'Avg_Campaign', 'Avg_Previous']]
    
    # Normalize for better visualization
    heatmap_data_norm = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
    
    cluster_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data_norm.values,
        x=heatmap_data_norm.columns,
        y=heatmap_data_norm.index,
        colorscale='RdYlBu_r',
        text=heatmap_data.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    cluster_heatmap.update_layout(title='Cluster Characteristics Heatmap (Normalized)',
                                 height=400)
    
    # Cluster size distribution
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_size_fig = px.pie(values=cluster_sizes.values, 
                             names=[f'Cluster {i}' for i in cluster_sizes.index],
                             title='Cluster Size Distribution',
                             color_discrete_sequence=['#667eea', '#f093fb', '#764ba2'])
    cluster_size_fig.update_layout(height=400)
    
    return html.Div([
        html.H2("🎯 Clustering & Dimensionality Reduction", style={'color': '#667eea', 'marginBottom': '20px'}),
        
        # Cluster summary cards
        html.Div([
            html.Div([
                html.H3("3", style={'margin': '0', 'fontSize': '2em', 'color': '#667eea'}),
                html.P("K-Means Clusters", style={'margin': '5px 0'})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '10px',
                'textAlign': 'center',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1',
                'margin': '10px'
            }),
            html.Div([
                html.H3(f"{silhouette_score(X_train_scaled, cluster_labels):.3f}", style={'margin': '0', 'fontSize': '2em', 'color': '#f093fb'}),
                html.P("Silhouette Score", style={'margin': '5px 0'})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '10px',
                'textAlign': 'center',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1',
                'margin': '10px'
            }),
            html.Div([
                html.H3(f"{sum(pca_variance_2d):.1%}", style={'margin': '0', 'fontSize': '2em', 'color': '#764ba2'}),
                html.P("2D PCA Variance", style={'margin': '5px 0'})
            ], style={
                'backgroundColor': 'white',
                'padding': '20px',
                'borderRadius': '10px',
                'textAlign': 'center',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1',
                'margin': '10px'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        # Optimal K determination
        html.H2("📊 Optimal Number of Clusters", style={'color': '#667eea', 'margin': '40px 0 20px 0'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=elbow_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=silhouette_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        # 2D and 3D cluster visualizations
        html.H2("🔍 Cluster Visualizations", style={'color': '#667eea', 'margin': '40px 0 20px 0'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=cluster_2d_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'
            })
        ]),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=cluster_3d_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'
            })
        ]),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=tsne_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=pca_variance_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'}),
        
        # Cluster characteristics
        html.H2("📈 Cluster Analysis", style={'color': '#667eea', 'margin': '40px 0 20px 0'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=cluster_heatmap)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            }),
            html.Div([
                dcc.Graph(figure=cluster_size_fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '10px',
                'padding': '20px',
                'margin': '10px',
                'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                'flex': '1'
            })
        ], style={'display': 'flex', 'flexWrap': 'wrap'})
    ])

if __name__ == '__main__':
    print("Starting Bank Marketing Dashboard...")
    print("Dashboard will be available at: http://127.0.0.1:8050")
    app.run_server(debug=True, host='127.0.0.1', port=8050)
