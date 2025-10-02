# ⚡ EV Analytics Dashboard

An interactive Streamlit web app for exploring, forecasting, and visualizing electric vehicle (EV) registration data. This dashboard supports data-driven planning for EV infrastructure, policy analysis, and trend forecasting.

## 🚀 Features

### 🔍 1. **EV Clustering & Infrastructure Planning**
- Cluster EVs by **Electric Range** and **Location**
- Visualize clusters on a **scatter plot** and an **interactive map**
- Designed to help identify ideal locations for charging stations

### 📈 2. **Trends & Forecasting**
- Filter EV data by Make, Model, Year, and Type
- Use **ARIMA** for time-series forecasting of EV adoption trends
- Predict **Electric Range** using a Random Forest Regressor
- Visualize future EV range trends by type and year

### 🧠 3. **CAFV Eligibility Prediction**
- Predict whether a vehicle qualifies as a **Clean Alternative Fuel Vehicle (CAFV)**
- Real-time input prediction using a trained **Random Forest Classifier**
- Supports categorical inputs (Make, Type) and numeric inputs (Year, Range)

---

## 📊 Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Web app framework |
| [pandas](https://pandas.pydata.org) | Data manipulation |
| [scikit-learn](https://scikit-learn.org) | Machine learning models |
| [statsmodels](https://www.statsmodels.org/) | Time series forecasting (ARIMA) |
| [pydeck](https://deckgl.readthedocs.io/en/latest/) | Interactive map visualization |
| [matplotlib](https://matplotlib.org) | Data visualization |

---



