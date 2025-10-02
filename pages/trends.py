import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def show_trends_page(df):
    st.header("📈 EV Trends & Forecasting 🚗")

    # ----------------------------------
    # Sidebar Filters
    # ----------------------------------
    st.sidebar.header("Filters")

    def safe_default(seq, n):
        return list(seq[:n]) if len(seq) >= n else list(seq)

    ev_types_all = sorted(df["Electric Vehicle Type"].dropna().unique().tolist()) if "Electric Vehicle Type" in df.columns else []
    makes_all   = sorted(df["Make"].dropna().unique().tolist()) if "Make" in df.columns else []
    models_all  = sorted(df["Model"].dropna().unique().tolist()) if "Model" in df.columns else []

    sel_types  = st.sidebar.multiselect("🚙Electric Vehicle Type", ev_types_all, default=safe_default(ev_types_all, 2))
    sel_makes  = st.sidebar.multiselect("🚙Make", makes_all, default=safe_default(makes_all, 5))
    sel_models = st.sidebar.multiselect("🚙Model (optional)", models_all)

    min_year = int(np.nanmin(df["Model Year"])) if df["Model Year"].notna().any() else 2000
    max_year = int(np.nanmax(df["Model Year"])) if df["Model Year"].notna().any() else 2025
    yr_range = st.sidebar.slider("🚙Model Year range", min_value=min_year, max_value=max_year,
                                 value=(min_year, max_year), step=1)

    # Apply filters
    filtered = df.copy()
    if sel_types:
        filtered = filtered[filtered["Electric Vehicle Type"].isin(sel_types)]
    if sel_makes:
        filtered = filtered[filtered["Make"].isin(sel_makes)]
    if sel_models:
        filtered = filtered[filtered["Model"].isin(sel_models)]
    filtered = filtered[(filtered["Model Year"] >= yr_range[0]) & (filtered["Model Year"] <= yr_range[1])]

    # ----------------------------------
    # Forecast with ARIMA
    # ----------------------------------
    st.subheader("⚡ Forecast: EV Counts by Model Year")

    def build_modelyear_series(sub: pd.DataFrame):
        yrs = pd.to_numeric(sub["Model Year"], errors="coerce")
        counts = (
            sub.loc[yrs.notna()]
            .groupby(yrs).size()
            .rename("count")
            .rename_axis("year")
            .reset_index()
        )
        if counts.empty:
            return None, None
        counts["year"] = counts["year"].astype(int)
        y = pd.Series(counts["count"].values,
                      index=pd.to_datetime(counts["year"].astype(str) + "-12-31")
                     ).asfreq("YE").interpolate()
        return counts, y

    counts, y = build_modelyear_series(filtered)
    steps_ahead = st.number_input("Forecast steps (years)", min_value=1, max_value=10, value=5, step=1)

    if (counts is None) or (y is None) or (len(y) < 6):
        st.info("Not enough data to fit ARIMA on the current filter. Try widening the filters.")
    else:
        try:
            model = ARIMA(y, order=(1, 1, 1)).fit()
            fc = model.forecast(steps=int(steps_ahead))

            fig = plt.figure(figsize=(8, 5))
            plt.plot(y.index.year, y.values, label="Observed")
            plt.plot([ix.year for ix in fc.index], fc.values, "o--", label=f"Forecast (+{steps_ahead}y)")
            plt.title("EV count by Model Year (filtered) with ARIMA forecast")
            plt.xlabel("Model Year")
            plt.ylabel("Count")
            plt.legend()
            st.pyplot(fig)
        except Exception as ex:
            st.warning(f"ARIMA failed: {ex}")

    # ----------------------------------
    # Regression for Electric Range
    # ----------------------------------
    st.subheader("⚡ Predict Electric Range")
    feat_cols = [c for c in ["Model Year", "Electric Vehicle Type", "Make", "Base MSRP"] if c in df.columns]

    @st.cache_resource
    def train_regressor(df_in: pd.DataFrame, feat_cols: list):
        data = df_in.dropna(subset=["Electric Range"]).copy()
        if len(data) < 500:
            return None, None, None

        X = data[feat_cols].copy()
        y = data["Electric Range"].values

        cat_cols = [c for c in feat_cols if X[c].dtype == "object"]
        num_cols = [c for c in feat_cols if c not in cat_cols]

        pre = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_cols),
                ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                  ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
            ]
        )
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        pipe = Pipeline([("pre", pre), ("rf", model)])

        max_rows = 60000
        if len(X) > max_rows:
            ix = np.random.RandomState(42).choice(len(X), size=max_rows, replace=False)
            X = X.iloc[ix]; y = y[ix]

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)

        mae = float(np.mean(np.abs(y_te - preds)))
        rmse = float(np.sqrt(np.mean((y_te - preds) ** 2)))
        r2 = float(1 - np.var(y_te - preds) / np.var(y_te)) if np.var(y_te) > 0 else float("nan")

        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "n_train": int(len(X_tr)), "n_test": int(len(X_te))}
        return pipe, metrics, (X_tr, X_te, y_tr, y_te)

    pipe, metrics, _ = train_regressor(filtered if len(filtered) > 2000 else df, feat_cols)

    if pipe is None:
        st.info("Not enough data to train the regressor. Relax your filters.")
    else:
        st.markdown("**Try a custom configuration**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            year_in = st.number_input("Model Year", min_value=min_year, max_value=max_year + 5,
                                      value=min(max_year, 2024), step=1)
        with col2:
            type_in = st.selectbox("Electric Vehicle Type", ev_types_all if ev_types_all else ["BEV", "PHEV"])
        with col3:
            fallback_make_list = makes_all if makes_all else ["Generic"]
            make_in = st.selectbox("Make", fallback_make_list)
        with col4:
            if "Base MSRP" in df.columns and make_in in df["Make"].values:
                msrp_default = int(np.nanmedian(df.loc[df["Make"] == make_in, "Base MSRP"].values))
                msrp_default = msrp_default if not np.isnan(msrp_default) else 50000
            else:
                msrp_default = 50000
            msrp_in = st.number_input("Base MSRP (USD)", min_value=0, max_value=300000, value=int(msrp_default), step=500)

        X_new = pd.DataFrame([{
            "Model Year": year_in,
            "Electric Vehicle Type": type_in,
            "Make": make_in,
            "Base MSRP": msrp_in if "Base MSRP" in df.columns else np.nan
        }])
        pred = float(pipe.predict(X_new)[0])
        st.success(f"**Predicted Electric Range:** ~ {pred:.1f} miles")

        # Trend lines
        trend_years = list(range(yr_range[0], min(yr_range[1] + 2, max_year + 2)))
        selected_types_for_trend = sel_types if sel_types else (ev_types_all if ev_types_all else ["BEV", "PHEV"])

        trend_rows = []
        for ev_t in selected_types_for_trend:
            subset = filtered[filtered["Electric Vehicle Type"] == ev_t] if len(filtered) > 0 else df[df["Electric Vehicle Type"] == ev_t]
            if "Make" in df.columns and len(subset) > 0 and subset["Make"].notna().any():
                common_make = subset["Make"].value_counts().idxmax()
            else:
                common_make = make_in
            med_price = float(subset["Base MSRP"].median()) if ("Base MSRP" in subset.columns) else (
                float(df["Base MSRP"].median()) if "Base MSRP" in df.columns else np.nan
            )
            for yr in trend_years:
                trend_rows.append({"Model Year": yr, "Electric Vehicle Type": ev_t, "Make": common_make, "Base MSRP": med_price})

        trend_df = pd.DataFrame(trend_rows)
        try:
            needed_cols = [c for c in ["Model Year", "Electric Vehicle Type", "Make", "Base MSRP"] if c in feat_cols]
            trend_df["predicted_range"] = pipe.predict(trend_df[needed_cols])
            fig2 = plt.figure(figsize=(8, 5))
            for ev_t in sorted(trend_df["Electric Vehicle Type"].unique()):
                sub = trend_df[trend_df["Electric Vehicle Type"] == ev_t]
                sub = sub.groupby("Model Year")["predicted_range"].mean().reset_index()
                plt.plot(sub["Model Year"], sub["predicted_range"], label=ev_t)
            plt.title("Predicted Electric Range vs Model Year (by EV Type)")
            plt.xlabel("Model Year"); plt.ylabel("Predicted Range (miles)")
            plt.legend()
            st.pyplot(fig2)
        except Exception as ex:
            st.warning(f"Trend plotting failed: {ex}")

    # ----------------------------------
    # Quick EDA Tables
    # ----------------------------------
    with st.expander("Show quick EDA tables"):
        yrs_num = pd.to_numeric(filtered["Model Year"], errors="coerce")
        year_counts = (
            filtered.loc[yrs_num.notna()]
            .groupby(yrs_num).size()
            .rename("count")
            .rename_axis("year")
            .reset_index()
            .astype({"year": int})
            .sort_values("year")
        )
        st.dataframe(year_counts)

        if "Electric Vehicle Type" in filtered.columns:
            ev_type_counts = (
                filtered["Electric Vehicle Type"]
                .value_counts(dropna=False)
                .rename_axis("EV Type")
                .reset_index(name="count")
            )
            st.dataframe(ev_type_counts)

        if "Make" in filtered.columns:
            make_counts = (
                filtered["Make"]
                .value_counts(dropna=False)
                .head(15)
                .rename_axis("Make")
                .reset_index(name="count")
            )
            st.dataframe(make_counts)

        if "Model" in filtered.columns:
            model_counts = (
                filtered["Model"]
                .value_counts(dropna=False)
                .head(15)
                .rename_axis("Model")
                .reset_index(name="count")
            )
            st.dataframe(model_counts)
