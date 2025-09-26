# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

st.set_page_config(page_title="Court Case Prioritization AI", layout="wide")
st.title("Court Case Prioritization AI/ML System")

st.markdown("""
This app prioritizes court cases into **High, Medium, and Low** priority using AI/ML.
It also provides **explanations** for why a case is prioritized.
""")

# --- Step 1: Load Dataset ---
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
else:
    # Default dataset path (replace with your actual CSV)
    default_path = "E:\data\combined_cases.csv"
    df = pd.read_csv(default_path)
    st.info(f"No file uploaded. Using default dataset: {default_path}")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --- Step 2: Preprocess Data ---
numeric_cols = ['PENDING_DAYS', 'HEARING_COUNT', 'DISPOSAL_DAYS', 'Mapped_Bail']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Encode categorical columns
if 'CIVIL_CRIMINAL' in df.columns:
    le = LabelEncoder()
    df['CIVIL_CRIMINAL_ENC'] = le.fit_transform(df['CIVIL_CRIMINAL'].astype(str))
else:
    df['CIVIL_CRIMINAL_ENC'] = 1

df['CASE_SEVERITY'] = df['CIVIL_CRIMINAL_ENC']

# --- Step 3: Generate Priority Scores ---
df['PENDING_NORM'] = df['PENDING_DAYS'] / df['PENDING_DAYS'].max()
df['HEARING_NORM'] = df['HEARING_COUNT'] / df['HEARING_COUNT'].max()
df['DISPOSAL_NORM'] = df['DISPOSAL_DAYS'] / df['DISPOSAL_DAYS'].max()
df['SEVERITY_NORM'] = df['CASE_SEVERITY'] / df['CASE_SEVERITY'].max()
df['BAIL_NORM'] = df['Mapped_Bail']

# Weighted priority score
w_pending, w_hearing, w_severity, w_disposal, w_bail = 0.35, 0.2, 0.25, 0.1, 0.1
df['PRIORITY_SCORE'] = (
    w_pending*df['PENDING_NORM'] +
    w_hearing*df['HEARING_NORM'] +
    w_severity*df['SEVERITY_NORM'] +
    w_disposal*df['DISPOSAL_NORM'] +
    w_bail*df['BAIL_NORM']
) * 100

# --- Step 4: Assign High/Medium/Low Priority ---
high_thres = df['PRIORITY_SCORE'].quantile(0.75)
low_thres = df['PRIORITY_SCORE'].quantile(0.25)

def priority_label(score):
    if score >= high_thres:
        return "High"
    elif score <= low_thres:
        return "Low"
    else:
        return "Medium"

df['PRIORITY_LEVEL'] = df['PRIORITY_SCORE'].apply(priority_label)

# --- Step 5: Display Priority Distribution ---
st.subheader("Priority Level Distribution")
st.bar_chart(df['PRIORITY_LEVEL'].value_counts())

# --- Step 6: ML Model Training (Predict Priority) ---
features = ['PENDING_DAYS','HEARING_COUNT','DISPOSAL_DAYS','Mapped_Bail','CASE_SEVERITY']
target = 'PRIORITY_LEVEL'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.subheader("ML Model Performance")
st.text(classification_report(y_test, y_pred))

df['PREDICTED_PRIORITY'] = clf.predict(X)

# --- Step 7: Explainable AI ---
def explain_case(row):
    reasons = []
    if row['CASE_SEVERITY'] > 2: reasons.append("Serious Criminal case")
    elif row['CASE_SEVERITY'] == 2: reasons.append("Moderate case")
    else: reasons.append("Civil case")
    if row['PENDING_DAYS'] > 180: reasons.append(f"Pending for {int(row['PENDING_DAYS'])} days")
    if row['HEARING_COUNT'] > 3: reasons.append(f"{int(row['HEARING_COUNT'])} hearings")
    if row['Mapped_Bail'] > 0: reasons.append("Bail pending")
    if row['DISPOSAL_DAYS'] > 180: reasons.append(f"Approaching disposal deadline ({int(row['DISPOSAL_DAYS'])} days)")
    reasons.append(f"Assigned Priority: {row['PREDICTED_PRIORITY']}")
    return "; ".join(reasons)

df['EXPLANATION'] = df.apply(explain_case, axis=1)

# --- Step 8: Top 10 Cases ---
st.subheader("Top 10 Prioritized Cases")
top_10 = df.sort_values(by='PRIORITY_SCORE', ascending=False).head(10)
st.dataframe(top_10[['CNR_NUMBER','CASE_NUMBER','PREDICTED_PRIORITY','EXPLANATION']])

# --- Step 9: Visualization ---
st.subheader("Priority Score Distribution by Case Type")
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='CIVIL_CRIMINAL', y='PRIORITY_SCORE', palette="Set2")
st.pyplot(plt)
