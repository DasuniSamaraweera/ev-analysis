import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

def show_classification_page(df):
    st.header("🚗 CAFV Eligibility Prediction Tool")

    # -----------------------------
    # Data Prep
    # -----------------------------
    if "Clean Alternative Fuel Vehicle (CAFV) Eligibility" not in df.columns:
        st.warning("Dataset missing CAFV Eligibility column.")
        return

    df = df[[
        "Model Year",
        "Make",
        "Electric Vehicle Type",
        "Electric Range",
        "Clean Alternative Fuel Vehicle (CAFV) Eligibility"
    ]].copy()

    df = df.rename(columns={
        "Clean Alternative Fuel Vehicle (CAFV) Eligibility": "CAFV Eligibility"
    })
    df = df.dropna(subset=["CAFV Eligibility"])

    # Handle missing numeric values
    df["Electric Range"] = df["Electric Range"].fillna(df["Electric Range"].median())

    # ✅ FIXED: Encode target properly
    def encode_cafv(x: str) -> int:
        x = x.lower()
        if "clean alternative fuel vehicle eligible" in x:
            return 1   # Eligible
        else:
            return 0   # Not eligible or unknown

    df["CAFV Eligibility"] = df["CAFV Eligibility"].astype(str).apply(encode_cafv)

    # Encode categorical features
    le_make = LabelEncoder()
    le_type = LabelEncoder()
    df["Make"] = le_make.fit_transform(df["Make"].astype(str))
    df["Electric Vehicle Type"] = le_type.fit_transform(df["Electric Vehicle Type"].astype(str))

    # Scale numeric features
    scaler = StandardScaler()
    df[["Model Year", "Electric Range"]] = scaler.fit_transform(df[["Model Year", "Electric Range"]])

    # Train/test split
    X = df.drop("CAFV Eligibility", axis=1)
    y = df["CAFV Eligibility"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Train Model (Random Forest)
    # -----------------------------
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # -----------------------------
    # User Input Prediction
    # -----------------------------
    st.subheader("🔍 Check a Vehicle's CAFV Eligibility")

    # Collect user input
    model_year = st.number_input("Model Year", min_value=1990, max_value=2025, value=2022)
    make_input = st.selectbox("Make", options=list(le_make.classes_))
    ev_type_input = st.selectbox("Electric Vehicle Type", options=list(le_type.classes_))
    electric_range = st.number_input("Electric Range (miles)", min_value=0, max_value=600, value=100)

    # Encode input
    make_encoded = le_make.transform([make_input])[0]
    type_encoded = le_type.transform([ev_type_input])[0]

    # Scale numeric features
    scaled_features = scaler.transform([[model_year, electric_range]])
    model_year_scaled, electric_range_scaled = scaled_features[0]

    # Create input dataframe
    input_df = pd.DataFrame([{
        "Model Year": model_year_scaled,
        "Make": make_encoded,
        "Electric Vehicle Type": type_encoded,
        "Electric Range": electric_range_scaled
    }])

    # Prediction
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # Show result
    if prediction == 1:
        st.success(f"✅ This vehicle IS likely CAFV eligible (probability: {prob:.2f})")
    else:
        st.error(f"❌ This vehicle is NOT likely CAFV eligible (probability: {prob:.2f})")
