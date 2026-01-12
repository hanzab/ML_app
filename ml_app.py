import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import f_classif

# XGBoost import with error handling
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    st.warning("XGBoost not available. Some models will be disabled.")

# Page configuration
st.set_page_config(
    page_title="Malaysia USD Exchange Rate Predictor",
    page_icon="🇲🇾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class USDExchangePredictor:
    def __init__(self):
        self.df = None
        self.models = {}
        self.results_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_importance = None
        self.trained = False
        
    def load_and_process_data(self):
        """Load and process all datasets"""
        # Helper function to expand annual data to monthly
        def process_annual(df, year_col, start_year=1997, value_cols=[]):
            if df[year_col].dtype == object:
                df['Year'] = pd.to_datetime(df[year_col]).dt.year
            else:
                df['Year'] = df[year_col]

            df = df[df['Year'] >= start_year].copy()

            # Create monthly range
            min_year, max_year = df['Year'].min(), df['Year'].max()
            all_dates = []
            for yr in range(min_year, max_year + 1):
                for mn in range(1, 13):
                    all_dates.append(pd.Timestamp(year=yr, month=mn, day=1))

            template = pd.DataFrame({'date': all_dates})
            template['Year'] = template['date'].dt.year

            return pd.merge(template, df[['Year'] + value_cols], on='Year', how='left').drop(columns=['Year'])
        
        try:
            # 1. Main Dataset (MEI)
            mei = pd.read_csv('mei (1).csv')
            mei['date'] = pd.to_datetime(mei['date'])
            mei = mei[mei['date'] >= '1997-01-01'].reset_index(drop=True)

            # 2. GDP/GNI (Filter 'abs', keep gdp, gni)
            gdp_gni = pd.read_csv('gdp_gni_annual_nominal (1).csv')
            gdp_gni = gdp_gni[gdp_gni['series'] == 'abs']
            gdp_gni_p = process_annual(gdp_gni, 'date', value_cols=['gdp', 'gni'])

            # 3. Net Migration (keep Net migration)
            net_mig = pd.read_csv('Net_migration_MYS.csv')
            net_mig_p = process_annual(net_mig, 'Year', value_cols=['Net migration'])

            # 4. Monthly (indicator 'avg', keep USD)
            monthly = pd.read_csv('monthly (2).csv')
            monthly['date'] = pd.to_datetime(monthly['date'])
            monthly = monthly[(monthly['indicator'] == 'avg') & (monthly['date'] >= '1997-01-01')]
            monthly_p = monthly[['date', 'USD']]

            # 5. CPI (division 'overall', keep inflation)
            cpi = pd.read_csv('cpi_2d_annual_inflation.csv')
            cpi = cpi[cpi['division'] == 'overall']
            cpi_p = process_annual(cpi, 'date', value_cols=['inflation'])

            # 6. Final Merge
            df = mei.merge(gdp_gni_p, on='date', how='left') \
                          .merge(net_mig_p, on='date', how='left') \
                          .merge(monthly_p, on='date', how='left') \
                          .merge(cpi_p, on='date', how='left')
            
            # Process the dataframe
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            
            # Drop date column and filter out 2025
            df = df.drop(columns=['date'])
            df = df[df['year'] != 2025]
            
            # Handle missing values in USD
            df['USD'] = df['USD'].ffill()
            df = df.dropna(subset=['USD'])
            
            return df
            
        except FileNotFoundError as e:
            st.error(f"Error loading data file: {e}")
            st.info("Using sample data for demonstration.")
            return self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate sample data for demonstration"""
        # Create date range
        dates = pd.date_range(start='1997-01-01', end='2024-12-01', freq='MS')
        n = len(dates)
        
        np.random.seed(42)
        
        # Generate synthetic economic data
        data = pd.DataFrame({
            'leading': np.random.normal(100, 5, n).cumsum()/50 + 95,
            'coincident': np.random.normal(100, 5, n).cumsum()/40 + 98,
            'lagging': np.random.normal(100, 5, n).cumsum()/30 + 100,
            'leading_diffusion': np.random.uniform(40, 60, n),
            'coincident_diffusion': np.random.uniform(45, 65, n),
            'gdp': np.random.normal(1000, 50, n).cumsum()/20 + 950,
            'gni': np.random.normal(1100, 60, n).cumsum()/20 + 1050,
            'Net migration': np.random.normal(0, 10000, n),
            'inflation': np.random.normal(2.5, 0.5, n).cumsum()/50 + 2,
            'USD': np.random.normal(4.2, 0.2, n).cumsum()/100 + 4.0
        })
        
        # Add trends and seasonality
        data['USD'] += 0.1 * np.sin(2 * np.pi * np.arange(n) / 12)
        data['USD'] += 0.01 * np.arange(n) / n  # Slight upward trend
        
        # Add year and month
        data['year'] = [d.year for d in dates]
        data['month'] = [d.month for d in dates]
        
        return data
    
    def prepare_features(self, feature_selection='manual'):
        """Prepare features for modeling"""
        X = self.df[['leading', 'coincident', 'lagging', 'leading_diffusion', 
                    'coincident_diffusion', 'gdp', 'gni', 'Net migration', 
                    'inflation', 'year', 'month']]
        y = self.df['USD']
        
        if feature_selection == 'correlation':
            # Correlation-based feature selection
            corr = X.corrwith(y)
            selected_features = corr[abs(corr) >= 0.1].index
            X_selected = X[selected_features]
            
        elif feature_selection == 'anova':
            # ANOVA-based feature selection
            anova_results = []
            for feature in X.columns:
                F, p = f_classif(X[[feature]], y)
                anova_results.append({'Feature': feature, 'F-value': F[0], 'p-value': p[0]})
            anova_results = pd.DataFrame(anova_results).sort_values(by='p-value')
            selected_features = anova_results[anova_results['p-value'] < 0.05]['Feature'].tolist()
            X_selected = X[selected_features]
            
        else:  # 'manual' - default from your code
            # Manual feature selection (default from your code)
            X_selected = self.df[['gdp', 'gni', 'lagging', 'coincident', 'leading', 'year']]
        
        return X_selected, y
    
    def train_models(self, X_train, y_train, X_test, y_test, use_xgboost_tuning=True):
        """Train all machine learning models"""
        results = {}
        training_times = {}
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scaler = scaler
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        # 1. Decision Tree
        start_time = time.time()
        dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42)
        dt.fit(X_train_scaled, y_train)
        training_times['Decision Tree'] = time.time() - start_time
        results['Decision Tree'] = {
            'model': dt,
            'predictions': dt.predict(X_test_scaled)
        }
        
        # 2. Linear Regression
        start_time = time.time()
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        training_times['Linear Regression'] = time.time() - start_time
        results['Linear Regression'] = {
            'model': lr,
            'predictions': lr.predict(X_test_scaled)
        }
        
        # 3. Random Forest
        start_time = time.time()
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train_scaled, y_train)
        training_times['Random Forest'] = time.time() - start_time
        results['Random Forest'] = {
            'model': rf,
            'predictions': rf.predict(X_test_scaled)
        }
        
        # 4. Gradient Boosting
        start_time = time.time()
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
        gb.fit(X_train_scaled, y_train)
        training_times['Gradient Boosting'] = time.time() - start_time
        results['Gradient Boosting'] = {
            'model': gb,
            'predictions': gb.predict(X_test_scaled)
        }
        
        # 5. SVR
        start_time = time.time()
        svr = SVR(kernel='rbf', C=1, epsilon=0.1)
        svr.fit(X_train_scaled, y_train)
        training_times['SVR'] = time.time() - start_time
        results['SVR'] = {
            'model': svr,
            'predictions': svr.predict(X_test_scaled)
        }
        
        # 6. XGBoost
        if XGB_AVAILABLE:
            if use_xgboost_tuning:
                # XGBoost with GridSearchCV
                start_time = time.time()
                xgb_model = xgb.XGBRegressor(random_state=42)
                param_grid_xgb = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'gamma': [0, 0.1, 0.2],
                    'min_child_weight': [1, 5, 10]
                }
                grid_search_xgb = GridSearchCV(
                    estimator=xgb_model, 
                    param_grid=param_grid_xgb, 
                    cv=5, 
                    n_jobs=-1, 
                    scoring='neg_mean_squared_error'
                )
                grid_search_xgb.fit(X_train_scaled, y_train)
                best_xgb = grid_search_xgb.best_estimator_
                training_times['XGBoost'] = time.time() - start_time
                results['XGBoost'] = {
                    'model': best_xgb,
                    'predictions': best_xgb.predict(X_test_scaled),
                    'best_params': grid_search_xgb.best_params_
                }
                self.best_model = best_xgb
            else:
                # Simple XGBoost without tuning
                start_time = time.time()
                xgb_model = xgb.XGBRegressor(random_state=42)
                xgb_model.fit(X_train_scaled, y_train)
                training_times['XGBoost'] = time.time() - start_time
                results['XGBoost'] = {
                    'model': xgb_model,
                    'predictions': xgb_model.predict(X_test_scaled)
                }
                self.best_model = xgb_model
        else:
            st.warning("XGBoost not available. Skipping XGBoost model.")
        
        # Store feature importance from Random Forest
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            self.feature_importance = importance_df
        
        self.models = results
        self.trained = True
        
        return results, training_times
    
    def evaluate_models(self, models, y_test):
        """Evaluate all models and return results dataframe"""
        results = []
        
        for name, model_info in models.items():
            pred = model_info['predictions']
            
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R² Score': r2
            })
        
        results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)
        self.results_df = results_df
        
        if not results_df.empty:
            self.best_model_name = results_df.iloc[0]['Model']
        
        return results_df
    
    def forecast_future(self, forecast_months=12):
        """Forecast future USD values"""
        if self.best_model is None:
            return None
        
        # Get last available data
        last_row = self.df.iloc[-1]
        last_year = last_row['year']
        last_month = last_row['month']
        
        # Create future dates
        future_dates = []
        current_year = last_year
        current_month = last_month
        
        for _ in range(forecast_months):
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
            future_dates.append(f"{current_year}-{current_month:02d}")
        
        # Create future dataframe with same features
        future_data = []
        for i in range(forecast_months):
            # Use the last values as baseline, with small trends
            row = last_row.copy()
            row['year'] = current_year
            row['month'] = current_month
            
            # Add small trends for economic indicators
            growth_factor = 1 + (0.005 * (i+1))  # 0.5% growth per period
            row['gdp'] = row['gdp'] * growth_factor
            row['gni'] = row['gni'] * growth_factor
            row['leading'] = row['leading'] * (1 + np.random.uniform(-0.005, 0.01))
            row['coincident'] = row['coincident'] * (1 + np.random.uniform(-0.005, 0.01))
            row['lagging'] = row['lagging'] * (1 + np.random.uniform(-0.005, 0.01))
            future_data.append(row)
        
        future_df = pd.DataFrame(future_data)
        
        # Prepare features (using the same features as training)
        # Get the columns used during training
        if hasattr(self, 'X_train') and self.X_train is not None:
            # We need to know which features were used
            # Since we don't have the original column names after scaling,
            # we'll use the manual feature selection
            X_future = future_df[['gdp', 'gni', 'lagging', 'coincident', 'leading', 'year']]
        else:
            # Default to manual features
            X_future = future_df[['gdp', 'gni', 'lagging', 'coincident', 'leading', 'year']]
        
        X_future_scaled = self.scaler.transform(X_future)
        
        # Make predictions
        forecasts = self.best_model.predict(X_future_scaled)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Year-Month': future_dates,
            'Forecasted_USD': forecasts
        })
        
        return forecast_df

def main():
    st.markdown('<h1 class="main-header">🇲🇾 Malaysia USD Exchange Rate Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Predicting USD/MYR with Economic Indicators & Migration Data</h3>', unsafe_allow_html=True)
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = USDExchangePredictor()
    
    predictor = st.session_state.predictor
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Flag_of_Malaysia.svg/800px-Flag_of_Malaysia.svg.png", 
                width=150)
        st.title("Navigation")
        
        page = st.selectbox(
            "Select Page:",
            ["🏠 Overview", "📊 Data Exploration", "🔍 Feature Analysis", 
             "🤖 Model Training", "📈 Predictions & Forecast"]
        )
        
        st.markdown("---")
        st.subheader("Settings")
        
        # Feature selection method
        feature_selection = st.selectbox(
            "Feature Selection Method:",
            ["manual", "correlation", "anova"],
            index=0,
            help="Manual: Use predefined features\nCorrelation: Features with |correlation| ≥ 0.1\nANOVA: Features with p-value < 0.05"
        )
        
        # Test size
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        
        # XGBoost tuning
        use_xgboost_tuning = st.checkbox("Use XGBoost Hyperparameter Tuning", value=True)
        
        st.markdown("---")
        st.info("""
        **Course:** BSD3523 Machine Learning
        **University:** UMPSA Gambang
        **Team:** 
        - Muhammad Danial Bin Issham
        - Ain Mardhiah Binti Abdul Hamid
        - Haizatul Syifa Binti Mansor
        - Hamizan Nasri Bin Zulkairi
        - Siti Nurul Insyirah Binti Mohd Fauzi
        """)
    
    # Load data if not already loaded
    if predictor.df is None:
        with st.spinner("Loading and processing data..."):
            df = predictor.load_and_process_data()
            predictor.df = df
    
    # Page routing
    if page == "🏠 Overview":
        show_overview_page(predictor)
    elif page == "📊 Data Exploration":
        show_data_exploration_page(predictor)
    elif page == "🔍 Feature Analysis":
        show_feature_analysis_page(predictor, feature_selection)
    elif page == "🤖 Model Training":
        show_model_training_page(predictor, feature_selection, test_size, use_xgboost_tuning)
    elif page == "📈 Predictions & Forecast":
        show_predictions_page(predictor)

def show_overview_page(predictor):
    """Display overview page"""
    st.markdown("""
    ## Project Overview
    
    This project aims to predict the USD/MYR exchange rate using various economic indicators,
    including GDP, GNI, migration trends, and leading economic indicators.
    
    ### **Data Sources:**
    1. **MEI (Main Economic Indicators)** - Leading, Coincident, Lagging indices
    2. **GDP/GNI Annual Data** - Gross Domestic Product and Gross National Income
    3. **Net Migration Statistics** - Migration trends in Malaysia
    4. **Monthly Exchange Rates** - USD/MYR exchange rates
    5. **CPI Inflation Data** - Consumer Price Index inflation rates
    """)
    
    # Display dataset information
    if predictor.df is not None:
        # Key metrics
        st.subheader("📈 Key Economic Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_usd = predictor.df['USD'].iloc[-1] if not predictor.df.empty else 0
            st.metric("Current USD/MYR", f"{latest_usd:.3f}")
        
        with col2:
            avg_usd = predictor.df['USD'].mean() if not predictor.df.empty else 0
            st.metric("Average USD/MYR", f"{avg_usd:.3f}")
        
        with col3:
            gdp_growth = ((predictor.df['gdp'].iloc[-1] - predictor.df['gdp'].iloc[0]) / predictor.df['gdp'].iloc[0] * 100) if not predictor.df.empty else 0
            st.metric("GDP Growth", f"{gdp_growth:.1f}%")
        
        with col4:
            avg_migration = predictor.df['Net migration'].mean() if not predictor.df.empty else 0
            st.metric("Avg Net Migration", f"{avg_migration:,.0f}")
        
        # Dataset information
        st.subheader("📊 Dataset Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Total Samples:** {len(predictor.df)}")
            st.info(f"**Features:** {len(predictor.df.columns)}")
            st.info(f"**Time Period:** {predictor.df['year'].min()} - {predictor.df['year'].max()}")
        
        with col2:
            st.info(f"**Missing Values:** {predictor.df.isnull().sum().sum()}")
            st.info(f"**Duplicate Rows:** {predictor.df.duplicated().sum()}")
        
        # Data preview
        with st.expander("View Dataset Preview"):
            st.dataframe(predictor.df.head(10), use_container_width=True)
        
        # Statistical summary
        with st.expander("View Statistical Summary"):
            st.dataframe(predictor.df.describe(), use_container_width=True)
        
        # Time series visualization
        st.subheader("📈 Economic Trends")
        
        if 'year' in predictor.df.columns and 'month' in predictor.df.columns:
            # Create date column for plotting
            plot_df = predictor.df.copy()
            plot_df['YearMonth'] = plot_df['year'].astype(str) + '-' + plot_df['month'].astype(str).str.zfill(2)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('USD Exchange Rate', 'GDP vs GNI',
                               'Economic Indicators', 'Migration & Inflation'),
                vertical_spacing=0.15
            )
            
            # USD Exchange Rate
            fig.add_trace(
                go.Scatter(x=plot_df['YearMonth'], y=plot_df['USD'], name='USD/MYR',
                          line=dict(color='#FF6B6B')),
                row=1, col=1
            )
            
            # GDP vs GNI
            fig.add_trace(
                go.Scatter(x=plot_df['YearMonth'], y=plot_df['gdp'], name='GDP',
                          line=dict(color='#4ECDC4')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=plot_df['YearMonth'], y=plot_df['gni'], name='GNI',
                          line=dict(color='#45B7D1')),
                row=1, col=2
            )
            
            # Economic Indicators
            fig.add_trace(
                go.Scatter(x=plot_df['YearMonth'], y=plot_df['leading'], name='Leading',
                          line=dict(color='#FECA57')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=plot_df['YearMonth'], y=plot_df['coincident'], name='Coincident',
                          line=dict(color='#FF9FF3')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=plot_df['YearMonth'], y=plot_df['lagging'], name='Lagging',
                          line=dict(color='#54A0FF')),
                row=2, col=1
            )
            
            # Migration & Inflation (dual y-axis)
            fig.add_trace(
                go.Scatter(x=plot_df['YearMonth'], y=plot_df['Net migration'], name='Net Migration',
                          line=dict(color='#2ecc71')),
                row=2, col=2
            )
            
            fig.update_layout(height=700, showlegend=True, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

def show_data_exploration_page(predictor):
    """Display data exploration page"""
    st.subheader("📊 Data Exploration")
    
    if predictor.df is None:
        st.warning("No data loaded. Please check your data files.")
        return
    
    # Distribution plots
    st.markdown("### 📈 Distribution of Numerical Variables")
    
    numerical_columns = predictor.df.select_dtypes(include=["int64", "float64"]).columns
    
    # Let user select which variables to plot
    selected_columns = st.multiselect(
        "Select variables for distribution analysis:",
        numerical_columns.tolist(),
        default=['USD', 'gdp', 'gni', 'Net migration', 'inflation']
    )
    
    if selected_columns:
        n_cols = 2
        n_rows = int(np.ceil(len(selected_columns) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten()
        
        for i, col in enumerate(selected_columns):
            if i < len(axes):
                # Histogram with KDE
                sns.histplot(predictor.df[col], kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(f"{col} | Skewness: {predictor.df[col].skew():.2f}")
                axes[i].set_xlabel("")
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Boxplots for outlier detection
    st.markdown("### 📦 Outlier Detection (Box Plots)")
    
    boxplot_cols = st.multiselect(
        "Select variables for box plots:",
        numerical_columns.tolist(),
        default=['USD', 'gdp', 'Net migration', 'inflation', 'leading', 'coincident']
    )
    
    if boxplot_cols:
        n_cols = 3
        n_rows = int(np.ceil(len(boxplot_cols) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = axes.flatten()
        
        for i, col in enumerate(boxplot_cols):
            if i < len(axes):
                sns.boxplot(x=predictor.df[col], ax=axes[i], color='lightcoral')
                axes[i].set_title(f"Boxplot of {col}")
                axes[i].set_xlabel("")
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # USD analysis by month
    st.markdown("### 📅 USD Analysis by Month")
    
    if 'month' in predictor.df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Boxplot by month
        sns.boxplot(x='month', y='USD', data=predictor.df, ax=ax1, palette='viridis')
        ax1.set_title("USD Distribution by Month")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("USD")
        
        # Line plot over time
        plot_df = predictor.df.copy()
        plot_df = plot_df.sort_values(['year', 'month'])
        
        # Create a sequential index for x-axis
        plot_df['time_index'] = range(len(plot_df))
        
        ax2.plot(plot_df['time_index'], plot_df['USD'], color='darkblue', linewidth=2)
        ax2.set_title("USD Trends Over Time")
        ax2.set_xlabel("Time Index")
        ax2.set_ylabel("USD")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

def show_feature_analysis_page(predictor, feature_selection):
    """Display feature analysis page"""
    st.subheader("🔍 Feature Analysis")
    
    if predictor.df is None:
        st.warning("No data loaded. Please check your data files.")
        return
    
    # Prepare features
    X_selected, y = predictor.prepare_features(feature_selection)
    
    # Display selected features
    st.markdown(f"### 🎯 Selected Features ({feature_selection} method)")
    st.info(f"**Number of features:** {X_selected.shape[1]}")
    st.write("**Selected features:**", list(X_selected.columns))
    
    # Show the features
    with st.expander("View Feature Data"):
        st.dataframe(X_selected.head(), use_container_width=True)
    
    # Correlation Analysis
    st.markdown("### 🔗 Correlation Analysis")
    
    # Combine features and target for correlation matrix
    corr_df = pd.concat([X_selected, y], axis=1)
    corr_matrix = corr_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                center=0, ax=ax, linewidths=0.5, square=True)
    ax.set_title("Correlation Matrix (Features & Target)")
    st.pyplot(fig)
    
    # Correlation with target
    st.markdown("### 📊 Correlation with USD")
    
    corr_with_target = X_selected.corrwith(y).sort_values(ascending=False)
    
    fig = go.Figure(go.Bar(
        x=corr_with_target.values,
        y=corr_with_target.index,
        orientation='h',
        marker_color=['#3498db' if x > 0 else '#e74c3c' for x in corr_with_target.values]
    ))
    
    fig.update_layout(
        title="Feature Correlation with USD Exchange Rate",
        xaxis_title="Correlation Coefficient",
        yaxis_title="Feature",
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def show_model_training_page(predictor, feature_selection, test_size, use_xgboost_tuning):
    """Display model training page"""
    st.subheader("🤖 Model Training")
    
    if predictor.df is None:
        st.warning("No data loaded. Please check your data files.")
        return
    
    # Prepare features
    X_selected, y = predictor.prepare_features(feature_selection)
    
    # Display training configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Training samples:** {int(len(X_selected) * (1 - test_size/100))}")
        st.info(f"**Test samples:** {int(len(X_selected) * (test_size/100))}")
    
    with col2:
        st.info(f"**Features:** {X_selected.shape[1]}")
        st.info(f"**Feature selection:** {feature_selection}")
    
    with col3:
        st.info(f"**XGBoost tuning:** {'Enabled' if use_xgboost_tuning else 'Disabled'}")
        if not XGB_AVAILABLE:
            st.warning("XGBoost not available")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size/100, random_state=42
    )
    
    # Train models button
    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        with st.spinner("Training models... This may take a few minutes."):
            # Train models
            models, training_times = predictor.train_models(
                X_train, y_train, X_test, y_test, use_xgboost_tuning
            )
            
            # Evaluate models
            results_df = predictor.evaluate_models(models, y_test)
            
            # Store in session state
            st.session_state.models_trained = True
            st.session_state.results_df = results_df
            
            st.success("✅ Models trained successfully!")
            
            # Show completion message
            if predictor.results_df is not None and not predictor.results_df.empty:
                best_model = predictor.results_df.iloc[0]['Model']
                best_r2 = predictor.results_df.iloc[0]['R² Score']
                st.success(f"Best Model: **{best_model}** with R² Score: **{best_r2:.4f}**")
    
    # Display results if available
    if predictor.results_df is not None and not predictor.results_df.empty:
        st.subheader("📊 Model Performance Comparison")
        
        # Display results table
        st.dataframe(
            predictor.results_df.style.format({
                'RMSE': '{:.4f}',
                'MAE': '{:.4f}',
                'R² Score': '{:.4f}'
            }).background_gradient(subset=['R² Score'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Visual comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score bar chart
            fig = go.Figure(go.Bar(
                x=predictor.results_df['Model'],
                y=predictor.results_df['R² Score'],
                marker_color=['#2ecc71' if x > 0.8 else '#f39c12' if x > 0.6 else '#e74c3c' 
                             for x in predictor.results_df['R² Score']],
                text=[f'{val:.3f}' for val in predictor.results_df['R² Score']],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Model Performance (R² Score)",
                xaxis_title="Model",
                yaxis_title="R² Score",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # RMSE and MAE comparison
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=predictor.results_df['Model'],
                y=predictor.results_df['RMSE'],
                name='RMSE',
                marker_color='#3498db'
            ))
            
            fig.add_trace(go.Bar(
                x=predictor.results_df['Model'],
                y=predictor.results_df['MAE'],
                name='MAE',
                marker_color='#e74c3c'
            ))
            
            fig.update_layout(
                title="Error Metrics Comparison",
                xaxis_title="Model",
                yaxis_title="Error Value",
                barmode='group',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        if predictor.feature_importance is not None:
            st.subheader("🔍 Feature Importance (Random Forest)")
            
            fig = go.Figure(go.Bar(
                x=predictor.feature_importance['Importance'],
                y=predictor.feature_importance['Feature'],
                orientation='h',
                marker_color='#9b59b6'
            ))
            
            fig.update_layout(
                title="Feature Importance Scores",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Actual vs Predicted comparison
        st.subheader("📈 Actual vs Predicted Comparison")
        
        if predictor.models and hasattr(predictor, 'y_test') and predictor.y_test is not None:
            # Select model for visualization
            selected_model = st.selectbox(
                "Select model for detailed visualization:",
                list(predictor.models.keys())
            )
            
            if selected_model and selected_model in predictor.models:
                model_info = predictor.models[selected_model]
                predictions = model_info['predictions']
                
                # Create comparison plot
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Actual vs Predicted', 'Residual Analysis'),
                    column_widths=[0.6, 0.4]
                )
                
                # Scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=predictor.y_test.values,
                        y=predictions,
                        mode='markers',
                        name='Predictions',
                        marker=dict(size=8, color='#e74c3c', opacity=0.6)
                    ),
                    row=1, col=1
                )
                
                # Add perfect prediction line
                min_val = min(predictor.y_test.min(), predictions.min())
                max_val = max(predictor.y_test.max(), predictions.max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='#2c3e50', dash='dash')
                    ),
                    row=1, col=1
                )
                
                # Residuals
                residuals = predictions - predictor.y_test.values
                fig.add_trace(
                    go.Scatter(
                        x=predictions,
                        y=residuals,
                        mode='markers',
                        name='Residuals',
                        marker=dict(size=8, color='#3498db', opacity=0.6)
                    ),
                    row=1, col=2
                )
                
                # Add zero line for residuals
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[0, 0],
                        mode='lines',
                        name='Zero Residual',
                        line=dict(color='#2c3e50', dash='dash')
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=500,
                    showlegend=True,
                    template='plotly_white'
                )
                
                fig.update_xaxes(title_text="Actual Values", row=1, col=1)
                fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
                fig.update_xaxes(title_text="Predicted Values", row=1, col=2)
                fig.update_yaxes(title_text="Residuals", row=1, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display model metrics
                rmse = np.sqrt(mean_squared_error(predictor.y_test, predictions))
                mae = mean_absolute_error(predictor.y_test, predictions)
                r2 = r2_score(predictor.y_test, predictions)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"{rmse:.4f}")
                col2.metric("MAE", f"{mae:.4f}")
                col3.metric("R² Score", f"{r2:.4f}")
    
    elif not hasattr(predictor, 'results_df') or predictor.results_df is None:
        st.info("👈 Click the 'Train All Models' button to start training.")
    else:
        st.warning("No models trained yet. Please click the 'Train All Models' button.")

def show_predictions_page(predictor):
    """Display predictions and forecast page"""
    st.subheader("📈 Predictions & Future Forecast")
    
    if not hasattr(predictor, 'trained') or not predictor.trained:
        st.warning("⚠️ Please train models first in the 'Model Training' section.")
        st.info("Go to the Model Training page and click 'Train All Models'.")
        return
    
    # Create tabs for different prediction types
    tab1, tab2 = st.tabs(["📊 Model Predictions Comparison", "🔮 Future Forecast"])
    
    with tab1:
        # Model predictions comparison
        if predictor.models and hasattr(predictor, 'y_test') and predictor.y_test is not None:
            # Get predictions from top models
            top_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Decision Tree']
            available_models = [m for m in top_models if m in predictor.models]
            
            if available_models:
                # Create comparison plot
                fig = go.Figure()
                
                # Add actual values
                fig.add_trace(go.Scatter(
                    x=np.arange(len(predictor.y_test)),
                    y=predictor.y_test.values,
                    mode='lines',
                    name='Actual USD',
                    line=dict(color='#2c3e50', width=3)
                ))
                
                # Add predictions for each model
                colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
                line_styles = ['solid', 'dash', 'dot', 'dashdot']
                
                for i, model_name in enumerate(available_models):
                    if model_name in predictor.models:
                        predictions = predictor.models[model_name]['predictions']
                        
                        fig.add_trace(go.Scatter(
                            x=np.arange(len(predictions)),
                            y=predictions,
                            mode='lines',
                            name=f'{model_name} Predictions',
                            line=dict(color=colors[i % len(colors)], 
                                     width=2, 
                                     dash=line_styles[i % len(line_styles)])
                        ))
                
                fig.update_layout(
                    title='Comparison of Actual vs. Predicted USD by Top Models',
                    xaxis_title='Sample Index',
                    yaxis_title='USD Exchange Rate',
                    template='plotly_white',
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display performance metrics
                st.subheader("📊 Model Performance Metrics")
                
                if predictor.results_df is not None:
                    st.dataframe(
                        predictor.results_df.style.format({
                            'RMSE': '{:.4f}',
                            'MAE': '{:.4f}',
                            'R² Score': '{:.4f}'
                        }).background_gradient(subset=['R² Score'], cmap='RdYlGn'),
                        use_container_width=True
                    )
        else:
            st.warning("No trained models available. Please train models first.")
    
    with tab2:
        # Future forecast
        st.markdown("### 🔮 Future USD Exchange Rate Forecast")
        
        if predictor.best_model is None:
            st.warning("No best model available. Please train models first.")
            return
        
        # Forecast configuration
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_months = st.slider(
                "Number of months to forecast:",
                min_value=1,
                max_value=36,
                value=12,
                help="Select how many months into the future to forecast"
            )
        
        with col2:
            confidence_level = st.slider(
                "Confidence interval (%):",
                min_value=80,
                max_value=95,
                value=90,
                help="Confidence interval for the forecast"
            )
        
        # Generate forecast button
        if st.button("Generate Forecast", type="primary", key="forecast_btn"):
            with st.spinner("Generating forecast..."):
                forecast_df = predictor.forecast_future(forecast_months)
                
                if forecast_df is not None and not forecast_df.empty:
                    st.success("✅ Forecast generated successfully!")
                    
                    # Display forecast table
                    st.subheader("📋 Forecast Results")
                    
                    # Format the forecast values
                    display_df = forecast_df.copy()
                    display_df['Forecasted_USD'] = display_df['Forecasted_USD'].round(4)
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Create forecast visualization
                    fig = go.Figure()
                    
                    # Historical data (last 24 months)
                    historical_df = predictor.df.copy()
                    historical_df = historical_df.sort_values(['year', 'month'])
                    historical_df = historical_df.tail(24)  # Last 2 years
                    
                    # Create date for historical data
                    historical_dates = []
                    for _, row in historical_df.iterrows():
                        historical_dates.append(f"{int(row['year'])}-{int(row['month']):02d}")
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=historical_dates,
                        y=historical_df['USD'],
                        mode='lines+markers',
                        name='Historical USD',
                        line=dict(color='#3498db', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Year-Month'],
                        y=forecast_df['Forecasted_USD'],
                        mode='lines+markers',
                        name='Forecasted USD',
                        line=dict(color='#e74c3c', width=3, dash='dash'),
                        marker=dict(size=6)
                    ))
                    
                    # Add confidence interval
                    std_dev = forecast_df['Forecasted_USD'].std()
                    upper_bound = forecast_df['Forecasted_USD'] + (std_dev * (confidence_level/100))
                    lower_bound = forecast_df['Forecasted_USD'] - (std_dev * (confidence_level/100))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Year-Month'].tolist() + forecast_df['Year-Month'].tolist()[::-1],
                        y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(231, 76, 60, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{confidence_level}% Confidence Interval'
                    ))
                    
                    fig.update_layout(
                        title=f'{forecast_months}-Month USD Exchange Rate Forecast',
                        xaxis_title='Year-Month',
                        yaxis_title='USD/MYR Exchange Rate',
                        template='plotly_white',
                        height=500,
                        xaxis=dict(
                            tickangle=45
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast statistics
                    st.subheader("📊 Forecast Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_forecast = forecast_df['Forecasted_USD'].mean()
                        st.metric("Average Forecast", f"{avg_forecast:.4f}")
                    
                    with col2:
                        min_forecast = forecast_df['Forecasted_USD'].min()
                        st.metric("Minimum Forecast", f"{min_forecast:.4f}")
                    
                    with col3:
                        max_forecast = forecast_df['Forecasted_USD'].max()
                        st.metric("Maximum Forecast", f"{max_forecast:.4f}")
                    
                    with col4:
                        last_historical = historical_df['USD'].iloc[-1]
                        last_forecast = forecast_df['Forecasted_USD'].iloc[-1]
                        change = ((last_forecast - last_historical) / last_historical) * 100
                        st.metric("Final Change", f"{change:.2f}%")
                    
                    # Forecast interpretation
                    st.subheader("📝 Forecast Interpretation")
                    
                    # Determine trend
                    first_forecast = forecast_df['Forecasted_USD'].iloc[0]
                    last_forecast = forecast_df['Forecasted_USD'].iloc[-1]
                    trend = last_forecast - first_forecast
                    
                    if trend > 0.1:
                        st.success(f"📈 **Strong Appreciation Trend**: Forecast shows an increase of {trend:.3f} over {forecast_months} months")
                        st.info("This suggests strengthening of the Malaysian Ringgit against USD.")
                    elif trend > 0:
                        st.info(f"📈 **Moderate Appreciation**: Forecast shows a slight increase of {trend:.3f}")
                        st.info("The Ringgit is expected to strengthen slightly against USD.")
                    elif trend < -0.1:
                        st.warning(f"📉 **Strong Depreciation Trend**: Forecast shows a decrease of {abs(trend):.3f}")
                        st.warning("The Ringgit is expected to weaken significantly against USD.")
                    else:
                        st.info(f"➡️ **Stable Exchange Rate**: Forecast shows minimal change ({trend:.3f})")
                        st.info("The USD/MYR exchange rate is expected to remain relatively stable.")
                    
                    # Download forecast
                    st.subheader("📥 Download Forecast")
                    
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast as CSV",
                        data=csv,
                        file_name="usd_forecast.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Failed to generate forecast. Please check if models are trained properly.")
        else:
            st.info("Click 'Generate Forecast' button to create future predictions.")

if __name__ == "__main__":
    main()