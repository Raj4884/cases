import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os # NEW: Added for path handling and robustness
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Set page config
st.set_page_config(
    page_title="NayaySetu:Court Case Prioritization AI System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (kept for professional appearance)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin: 1.5rem 0;
    }
    .explanation-box {
        background: #f8f9fa;
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚖️ NayaySetu</h1>', unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.title("🔧 Configuration Panel")

# Model selection
model_choice = st.sidebar.selectbox(
    "Select ML Model",
    ["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression", "Ensemble"]
)

# Priority weights configuration
st.sidebar.subheader("Priority Weights (Rule-Based Prioritization)")
w_pending = st.sidebar.slider("Pending Days Weight", 0.0, 1.0, 0.35, 0.05)
w_hearing = st.sidebar.slider("Hearing Count Weight", 0.0, 1.0, 0.20, 0.05)
w_severity = st.sidebar.slider("Case Severity Weight", 0.0, 1.0, 0.25, 0.05)
w_disposal = st.sidebar.slider("Disposal Days Weight", 0.0, 1.0, 0.10, 0.05)
w_bail = st.sidebar.slider("Bail Status Weight", 0.0, 1.0, 0.10, 0.05)


# --- EMERGENCY FALLBACK DATA ---
# Used only if the local file read fails.
def create_emergency_dataframe():
    data = {
        'CNR_NUMBER': ['E_001', 'E_002', 'E_003', 'E_004', 'E_005'],
        'CASE_NUMBER': ['CC/1', 'CC/2', 'CC/3', 'CC/4', 'CC/5'],
        'CIVIL_CRIMINAL': ['Criminal', 'Civil', 'Criminal', 'Family', 'Commercial'],
        'PENDING_DAYS': [1500, 300, 90, 800, 450],
        'HEARING_COUNT': [30, 5, 2, 12, 8],
        'DISPOSAL_DAYS': [500, 150, 40, 350, 180],
        'Mapped_Bail': [1, 0, 1, 0, 0],
        'COURT_NAME': ['High Court', 'District Court', 'District Court', 'High Court', 'Supreme Court'],
        'JUDGE_NAME': ['J1', 'J2', 'J3', 'J4', 'J5'],
    }
    return pd.DataFrame(data)


# --- DATA LOADING SECTION (Modified as requested) ---

st.markdown('<h2 class="sub-header">📊 Data Loading & Preview</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Court Cases Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
else:
    # Default dataset path (replace with your actual CSV)
    default_path = "E:\data\combined_cases.csv"
    try:
        # NOTE: This path MUST exist on the machine running the Streamlit server.
        df = pd.read_csv(default_path)
        st.info(f"📋 No file uploaded. Using default dataset from local path: `{default_path}`")
    except FileNotFoundError:
        df = create_emergency_dataframe()
        st.error(f"❌ File Not Found at `{default_path}`. Using emergency fallback data.")
    except Exception as e:
        df = create_emergency_dataframe()
        st.error(f"❌ Error loading file from `{default_path}` ({e}). Using emergency fallback data.")

st.subheader("Dataset Preview")
st.dataframe(df.head())


# Data preprocessing
@st.cache_data
def preprocess_data(df):
    """Enhanced data preprocessing with feature engineering"""
    df_processed = df.copy()
    
    # Handle numeric columns
    numeric_cols = ['PENDING_DAYS', 'HEARING_COUNT', 'DISPOSAL_DAYS', 'Mapped_Bail']
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
    
    # Encode categorical columns and map severity
    if 'CIVIL_CRIMINAL' in df_processed.columns:
        le_case_type = LabelEncoder()
        df_processed['CASE_TYPE_ENCODED'] = le_case_type.fit_transform(df_processed['CIVIL_CRIMINAL'].astype(str))
        
        # NOTE: This severity map relies on case types being present
        severity_map = {'Criminal': 3, 'Family': 2, 'Commercial': 2, 'Labor': 2, 'Civil': 1, 'Unknown': 1}
        df_processed['CASE_SEVERITY'] = df_processed['CIVIL_CRIMINAL'].map(severity_map).fillna(1)
    else:
        df_processed['CASE_TYPE_ENCODED'] = 1
        df_processed['CASE_SEVERITY'] = 1
    
    # Feature engineering
    df_processed['URGENCY_FACTOR'] = (df_processed['PENDING_DAYS'] / 30) * df_processed['CASE_SEVERITY']
    df_processed['COMPLEXITY_SCORE'] = df_processed['HEARING_COUNT'] * df_processed['CASE_SEVERITY']
    df_processed['DELAY_RISK'] = np.where(df_processed['PENDING_DAYS'] > 180, 1, 0)
    
    # Normalize features for ML
    scaler = StandardScaler()
    normalized_features = ['PENDING_DAYS', 'HEARING_COUNT', 'DISPOSAL_DAYS']
    for feature in normalized_features:
        if feature in df_processed.columns:
            # Check for zero variance before scaling
            if df_processed[feature].nunique() > 1:
                 df_processed[f'{feature}_NORM'] = scaler.fit_transform(df_processed[[feature]]).flatten()
            else:
                 df_processed[f'{feature}_NORM'] = 0.0

    return df_processed

df_processed = preprocess_data(df)

# Display dataset info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Cases", len(df_processed))
with col2:
    st.metric("Features", len(df_processed.columns))
with col3:
    avg_pending = df_processed['PENDING_DAYS'].mean()
    st.metric("Avg Pending Days", f"{avg_pending:.0f}")

# Dataset preview with tabs
tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📈 Statistics", "🔍 Data Quality"])

with tab1:
    st.dataframe(df_processed.head(10), use_container_width=True)

with tab2:
    st.write("**Statistical Summary**")
    st.dataframe(df_processed.describe(), use_container_width=True)

with tab3:
    missing_data = df_processed.isnull().sum()
    if missing_data.sum() > 0:
        st.write("**Missing Data Analysis**")
        st.bar_chart(missing_data[missing_data > 0])
    else:
        st.success("✅ No missing data found")

# Priority calculation function
def calculate_priority_score(df, weights):
    """Calculate priority scores with given weights (Rule-Based Prioritization)"""
    
    # Check if necessary columns exist and calculate normalization factors safely
    max_pending = df['PENDING_DAYS'].max() if 'PENDING_DAYS' in df.columns and df['PENDING_DAYS'].max() > 0 else 1
    max_hearing = df['HEARING_COUNT'].max() if 'HEARING_COUNT' in df.columns and df['HEARING_COUNT'].max() > 0 else 1
    max_disposal = df['DISPOSAL_DAYS'].max() if 'DISPOSAL_DAYS' in df.columns and df['DISPOSAL_DAYS'].max() > 0 else 1
    max_severity = df['CASE_SEVERITY'].max() if 'CASE_SEVERITY' in df.columns and df['CASE_SEVERITY'].max() > 0 else 1

    # Normalize features
    df['PENDING_NORM'] = df['PENDING_DAYS'] / max_pending
    df['HEARING_NORM'] = df['HEARING_COUNT'] / max_hearing
    df['DISPOSAL_NORM'] = df['DISPOSAL_DAYS'] / max_disposal
    df['SEVERITY_NORM'] = df['CASE_SEVERITY'] / max_severity
    df['BAIL_NORM'] = df['Mapped_Bail'] if 'Mapped_Bail' in df.columns else 0
    
    # Calculate weighted priority score
    df['PRIORITY_SCORE'] = (
        weights['pending'] * df['PENDING_NORM'] +
        weights['hearing'] * df['HEARING_NORM'] +
        weights['severity'] * df['SEVERITY_NORM'] +
        weights['disposal'] * df['DISPOSAL_NORM'] +
        weights['bail'] * df['BAIL_NORM']
    ) * 100
    
    return df

# Calculate priority scores
weights = {
    'pending': w_pending,
    'hearing': w_hearing,
    'severity': w_severity,
    'disposal': w_disposal,
    'bail': w_bail
}

df_processed = calculate_priority_score(df_processed, weights)

# Assign priority levels
def assign_priority_levels(df, high_percentile=75, low_percentile=25):
    """Assign priority levels based on percentiles"""
    if len(df) < 4:
        # Cannot calculate meaningful percentiles with too few rows
        df['PRIORITY_LEVEL'] = 'Medium'
        return df, 0, 0
        
    high_threshold = np.percentile(df['PRIORITY_SCORE'], high_percentile)
    low_threshold = np.percentile(df['PRIORITY_SCORE'], low_percentile)
    
    def get_priority(score):
        if score >= high_threshold:
            return "High"
        elif score <= low_threshold:
            return "Low"
        else:
            return "Medium"
    
    df['PRIORITY_LEVEL'] = df['PRIORITY_SCORE'].apply(get_priority)
    return df, high_threshold, low_threshold

df_processed, high_threshold, low_threshold = assign_priority_levels(df_processed)

# --- Remaining application logic ---

# Priority distribution visualization
st.markdown('<h2 class="sub-header">📊 Priority Distribution Analysis</h2>', unsafe_allow_html=True)

if len(df_processed) > 1:
    col1, col2 = st.columns(2)

    with col1:
        # Priority level distribution
        priority_counts = df_processed['PRIORITY_LEVEL'].value_counts()
        fig_pie = px.pie(
            values=priority_counts.values,
            names=priority_counts.index,
            title="Case Priority Distribution",
            color_discrete_map={'High': '#ff4444', 'Medium': '#ff9800', 'Low': '#4caf50'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Priority score distribution
        fig_hist = px.histogram(
            df_processed,
            x='PRIORITY_SCORE',
            color='PRIORITY_LEVEL',
            title="Priority Score Distribution",
            color_discrete_map={'High': '#ff4444', 'Medium': '#ff9800', 'Low': '#4caf50'}
        )
        if high_threshold > 0:
            fig_hist.add_vline(x=high_threshold, line_dash="dash", line_color="red", annotation_text="High Threshold")
            fig_hist.add_vline(x=low_threshold, line_dash="dash", line_color="green", annotation_text="Low Threshold")
        st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.warning("Not enough data points to generate meaningful distribution charts.")


# Machine Learning Models
st.markdown('<h2 class="sub-header">🤖 Machine Learning Model Training</h2>', unsafe_allow_html=True)

# Prepare features for ML
feature_columns = ['PENDING_DAYS', 'HEARING_COUNT', 'DISPOSAL_DAYS', 'Mapped_Bail', 'CASE_SEVERITY', 'URGENCY_FACTOR', 'COMPLEXITY_SCORE']
available_features = [col for col in feature_columns if col in df_processed.columns]

if len(df_processed) >= 20 and len(available_features) > 0 and df_processed['PRIORITY_LEVEL'].nunique() > 1:

    X = df_processed[available_features]
    y = df_processed['PRIORITY_LEVEL']

    @st.cache_data
    def train_models(X, y):
        """Train multiple ML models and return their performance"""
        # Ensure we can split the data
        if len(X) * 0.2 < 1 or y.nunique() < 2:
            return None, None, None, None, None, None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        trained_models = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'actual': y_test,
                'model': model
            }
            trained_models[name] = model
        
        return results, trained_models, X_train, X_test, y_train, y_test

    # Train models
    with st.spinner("Training machine learning models..."):
        model_results, trained_models, X_train, X_test, y_train, y_test = train_models(X, y)

    if model_results:
        # Model performance comparison
        st.subheader("🏆 Model Performance Comparison")

        performance_df = pd.DataFrame({
            'Model': list(model_results.keys()),
            'Accuracy': [results['accuracy'] for results in model_results.values()]
        }).sort_values('Accuracy', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            fig_performance = px.bar(
                performance_df,
                x='Model',
                y='Accuracy',
                title="Model Accuracy Comparison",
                color='Accuracy',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_performance, use_container_width=True)

        with col2:
            st.write("**Detailed Performance Metrics**")
            for model_name, results in model_results.items():
                with st.expander(f"{model_name} - Accuracy: {results['accuracy']:.3f}"):
                    st.text(classification_report(results['actual'], results['predictions']))

        # Select best model for predictions
        selected_model = trained_models[model_choice] if model_choice != "Ensemble" else trained_models[performance_df.iloc[0]['Model']]

        # Generate predictions
        df_processed['PREDICTED_PRIORITY'] = selected_model.predict(X)
        df_processed['PREDICTION_CONFIDENCE'] = selected_model.predict_proba(X).max(axis=1)
    
    else:
        # Fallback if training failed (e.g., too few samples for stratified split)
        st.warning("ML Model training skipped: Data is too small or unbalanced for training. Showing Rule-Based Priority only.")
        df_processed['PREDICTED_PRIORITY'] = df_processed['PRIORITY_LEVEL']
        df_processed['PREDICTION_CONFIDENCE'] = 1.0 
else:
    st.warning("ML Model training skipped: Need at least 20 rows and two priority classes to train models effectively.")
    df_processed['PREDICTED_PRIORITY'] = df_processed['PRIORITY_LEVEL']
    df_processed['PREDICTION_CONFIDENCE'] = 1.0


# Advanced Explainable AI
st.markdown('<h2 class="sub-header">🔍 Advanced Explainable AI</h2>', unsafe_allow_html=True)

def generate_detailed_explanation(row):
    """Generate detailed explanation for case prioritization"""
    explanations = []
    risk_factors = []
    
    # Case severity analysis
    severity_map = {1: "Low severity (Civil)", 2: "Medium severity", 3: "High severity (Criminal)"}
    case_severity_text = severity_map.get(row.get('CASE_SEVERITY', 1), "Unknown severity")
    explanations.append(f"**Case Type**: {case_severity_text}")
    
    # Pending days analysis
    pending_days = row.get('PENDING_DAYS', 0)
    if pending_days > 365:
        risk_factors.append(f"⚠️ Extremely delayed case ({int(pending_days)} days pending)")
        explanations.append(f"**Critical Delay**: Case has been pending for {int(pending_days)} days (>1 year)")
    elif pending_days > 180:
        risk_factors.append(f"📅 Significantly delayed ({int(pending_days)} days pending)")
        explanations.append(f"**Significant Delay**: Case pending for {int(pending_days)} days")
    
    # Bail status
    if row.get('Mapped_Bail') == 1:
        risk_factors.append("⚖️ Bail-related case")
        explanations.append("**Bail Matter**: Requires immediate attention for bail decisions")
    
    # Priority score interpretation
    priority_percentile = (df_processed['PRIORITY_SCORE'] <= row['PRIORITY_SCORE']).mean() * 100
    explanations.append(f"**Priority Score**: {row['PRIORITY_SCORE']:.1f}/100 (Top {100-priority_percentile:.1f}%)")
    
    # Prediction confidence
    confidence_text = "High" if row['PREDICTION_CONFIDENCE'] > 0.8 else "Medium" if row['PREDICTION_CONFIDENCE'] > 0.6 else "Low"
    explanations.append(f"**Model Confidence**: {confidence_text} ({row['PREDICTION_CONFIDENCE']:.2%})")
    
    return {
        'detailed_explanations': explanations,
        'risk_factors': risk_factors,
        'priority_level': row['PREDICTED_PRIORITY'],
        'priority_score': row['PRIORITY_SCORE']
    }

# Top prioritized cases with detailed explanations
st.markdown('<h2 class="sub-header">🔝 Top Prioritized Cases with Detailed Analysis</h2>', unsafe_allow_html=True)

top_cases = df_processed.nlargest(20, 'PRIORITY_SCORE')

# Display cases with detailed explanations
for idx, (_, case) in enumerate(top_cases.iterrows()):
    explanation = generate_detailed_explanation(case)
    
    with st.expander(f"🏛️ Case #{idx+1}: {case.get('CNR_NUMBER', 'N/A')} - {explanation['priority_level']} Priority"):
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.write("**Case Details:**")
            st.write(f"- **CNR Number**: {case.get('CNR_NUMBER', 'N/A')}")
            st.write(f"- **Case Number**: {case.get('CASE_NUMBER', 'N/A')}")
            st.write(f"- **Case Type**: {case.get('CIVIL_CRIMINAL', 'N/A')}")
            st.write(f"- **Court**: {case.get('COURT_NAME', 'N/A')}")
            st.write(f"- **Judge**: {case.get('JUDGE_NAME', 'N/A')}")
        
        with col2:
            st.write("**Priority Analysis:**")
            for exp in explanation['detailed_explanations']:
                st.write(f"- {exp}")
            
            if explanation['risk_factors']:
                st.write("**⚠️ Risk Factors:**")
                for risk in explanation['risk_factors']:
                    st.write(f"  - {risk}")

# Dashboard summary
st.markdown('<h2 class="sub-header">📋 Dashboard Summary</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

high_priority_count = len(df_processed[df_processed['PRIORITY_LEVEL'] == 'High'])

with col1:
    st.metric("High Priority Cases", high_priority_count, delta=f"{high_priority_count/len(df_processed)*100:.1f}%")

with col2:
    avg_confidence = df_processed['PREDICTION_CONFIDENCE'].mean() if 'PREDICTION_CONFIDENCE' in df_processed.columns else 0
    st.metric("Avg Model Confidence", f"{avg_confidence:.1%}")

with col3:
    critical_cases = len(df_processed[df_processed.get('PENDING_DAYS', pd.Series([0])) > 365])
    st.metric("Critical Delayed Cases", critical_cases)

with col4:
    best_model_accuracy = performance_df.iloc[0]['Accuracy'] if 'performance_df' in locals() and not performance_df.empty else 0.0
    st.metric("Best Model Accuracy", f"{best_model_accuracy:.1%}")

# Export functionality
st.markdown('<h2 class="sub-header">💾 Export Results</h2>', unsafe_allow_html=True)

export_columns = ['CNR_NUMBER', 'CASE_NUMBER', 'CIVIL_CRIMINAL', 'PRIORITY_LEVEL', 'PRIORITY_SCORE', 'PREDICTION_CONFIDENCE', 'PENDING_DAYS', 'HEARING_COUNT']
export_df = df_processed[[col for col in export_columns if col in df_processed.columns]].copy()

csv = export_df.to_csv(index=False)
st.download_button(
    label="📥 Download Prioritized Cases (CSV)",
    data=csv,
    file_name="prioritized_court_cases.csv",
    mime="text/csv"
)

# Sidebar - Model insights
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Quick Insights")
st.sidebar.write(f"**Total Cases**: {len(df_processed):,}")
st.sidebar.write(f"**High Priority**: {high_priority_count} ({high_priority_count/len(df_processed)*100:.1f}%)")
st.sidebar.write(f"**Avg Pending**: {df_processed['PENDING_DAYS'].mean():.0f} days")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Adjust the priority weights in the sidebar to see how it affects case prioritization!")