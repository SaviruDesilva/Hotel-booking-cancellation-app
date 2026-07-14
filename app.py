"""
Hotel Booking Cancellation Analysis — single-file Streamlit app.

Covers, in one page with tabs:
  - Overview
  - EDA
  - Feature Selection
  - Modelling & Prediction

Just drop this file (app.py), requirements.txt, and the model/ folder into
your existing GitHub repo (alongside your notebooks) and deploy on
Streamlit Community Cloud with main file = app.py.
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(page_title="Hotel Booking Cancellations", page_icon="🏨", layout="wide")

RED, BLUE = "#B22222", "#5dade2"

# =============================================================================
# DATA SOURCE
# =============================================================================
# The original notebooks pull the data from Kaggle via `kagglehub`, which
# needs Kaggle API credentials that Streamlit Cloud won't have by default.
# This mirror is the identical dataset (same 119,390 rows/columns), hosted
# publicly, so the app deploys with zero secrets required.
DATA_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "master/data/2020/2020-02-11/hotels.csv"
)

COUNTRY_NAME_MAP = {
    'PRT': 'Portugal', 'GBR': 'United Kingdom', 'USA': 'United States', 'ESP': 'Spain',
    'IRL': 'Ireland', 'FRA': 'France', 'ROU': 'Romania', 'NOR': 'Norway',
    'OMN': 'Oman', 'ARG': 'Argentina', 'POL': 'Poland', 'DEU': 'Germany',
    'BEL': 'Belgium', 'CHE': 'Switzerland', 'CN': 'China', 'GRC': 'Greece',
    'ITA': 'Italy', 'NLD': 'Netherlands', 'DNK': 'Denmark', 'RUS': 'Russia',
    'SWE': 'Sweden', 'AUS': 'Australia', 'EST': 'Estonia', 'CZE': 'Czech Republic',
    'BRA': 'Brazil', 'FIN': 'Finland', 'MOZ': 'Mozambique', 'BWA': 'Botswana',
    'LUX': 'Luxembourg', 'SVN': 'Slovenia', 'ALB': 'Albania', 'IND': 'India',
    'CHN': 'China', 'MEX': 'Mexico', 'MAR': 'Morocco', 'UKR': 'Ukraine',
    'SMR': 'San Marino', 'LVA': 'Latvia', 'PRI': 'Puerto Rico', 'SRB': 'Serbia',
    'CHL': 'Chile', 'AUT': 'Austria', 'BLR': 'Belarus', 'LTU': 'Lithuania',
    'TUR': 'Turkey', 'ZAF': 'South Africa', 'AGO': 'Angola', 'ISR': 'Israel',
    'CYM': 'Cayman Islands', 'ZMB': 'Zambia', 'CPV': 'Cape Verde', 'ZWE': 'Zimbabwe',
    'DZA': 'Algeria', 'KOR': 'South Korea', 'CRI': 'Costa Rica', 'HUN': 'Hungary',
    'ARE': 'United Arab Emirates', 'TUN': 'Tunisia', 'JAM': 'Jamaica', 'HRV': 'Croatia',
    'HKG': 'Hong Kong', 'IRN': 'Iran', 'GEO': 'Georgia', 'AND': 'Andorra',
    'GIB': 'Gibraltar', 'URY': 'Uruguay', 'JEY': 'Jersey', 'CAF': 'Central African Republic',
    'CYP': 'Cyprus', 'COL': 'Colombia', 'GGY': 'Guernsey', 'KWT': 'Kuwait',
    'NGA': 'Nigeria', 'MDV': 'Maldives', 'VEN': 'Venezuela', 'SVK': 'Slovakia',
    'FJI': 'Fiji', 'KAZ': 'Kazakhstan', 'PAK': 'Pakistan', 'IDN': 'Indonesia',
    'LBN': 'Lebanon', 'PHL': 'Philippines', 'SEN': 'Senegal', 'SYC': 'Seychelles',
    'AZE': 'Azerbaijan', 'BHR': 'Bahrain', 'NZL': 'New Zealand', 'THA': 'Thailand',
    'DOM': 'Dominican Republic', 'MKD': 'North Macedonia', 'MYS': 'Malaysia', 'ARM': 'Armenia',
    'JPN': 'Japan', 'LKA': 'Sri Lanka', 'CUB': 'Cuba', 'CMR': 'Cameroon',
    'BIH': 'Bosnia and Herzegovina', 'MUS': 'Mauritius', 'COM': 'Comoros', 'SUR': 'Suriname',
    'UGA': 'Uganda', 'BGR': 'Bulgaria', 'CIV': 'Ivory Coast', 'JOR': 'Jordan',
    'SYR': 'Syria', 'SGP': 'Singapore', 'BDI': 'Burundi', 'SAU': 'Saudi Arabia',
    'VNM': 'Vietnam', 'PLW': 'Palau', 'QAT': 'Qatar', 'EGY': 'Egypt',
    'PER': 'Peru', 'MLT': 'Malta', 'MWI': 'Malawi', 'ECU': 'Ecuador',
    'MDG': 'Madagascar', 'ISL': 'Iceland', 'UZB': 'Uzbekistan', 'NPL': 'Nepal',
    'BHS': 'Bahamas', 'MAC': 'Macau', 'TGO': 'Togo', 'TWN': 'Taiwan',
    'DJI': 'Djibouti', 'STP': 'Sao Tome and Principe', 'KNA': 'Saint Kitts and Nevis', 'ETH': 'Ethiopia',
    'IRQ': 'Iraq', 'HND': 'Honduras', 'RWA': 'Rwanda', 'KHM': 'Cambodia',
    'MCO': 'Monaco', 'BGD': 'Bangladesh', 'IMN': 'Isle of Man', 'TJK': 'Tajikistan',
    'NIC': 'Nicaragua', 'BEN': 'Benin', 'VGB': 'British Virgin Islands', 'TZA': 'Tanzania',
    'GAB': 'Gabon', 'GHA': 'Ghana', 'TMP': 'East Timor', 'GLP': 'Guadeloupe',
    'KEN': 'Kenya', 'LIE': 'Liechtenstein', 'GNB': 'Guinea-Bissau', 'MNE': 'Montenegro',
    'UMI': 'United States Minor Outlying Islands', 'MYT': 'Mayotte', 'FRO': 'Faroe Islands', 'MMR': 'Myanmar',
    'PAN': 'Panama', 'BFA': 'Burkina Faso', 'LBY': 'Libya', 'MLI': 'Mali',
    'NAM': 'Namibia', 'BOL': 'Bolivia', 'PRY': 'Paraguay', 'BRB': 'Barbados',
    'ABW': 'Aruba', 'AIA': 'Anguilla', 'SLV': 'El Salvador', 'DMA': 'Dominica',
    'PYF': 'French Polynesia', 'GUY': 'Guyana', 'LCA': 'Saint Lucia', 'ATA': 'Antarctica',
    'GTM': 'Guatemala', 'ASM': 'American Samoa', 'MRT': 'Mauritania', 'NCL': 'New Caledonia',
    'KIR': 'Kiribati', 'SDN': 'Sudan', 'ATF': 'French Southern and Antarctic Lands', 'SLE': 'Sierra Leone',
    'LAO': 'Laos',
}

MONTH_ORDER = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

NUMERIC_FEATURES = [
    'lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
    'adults', 'previous_cancellations', 'previous_bookings_not_canceled',
    'adr', 'required_car_parking_spaces', 'total_of_special_requests',
    'country_frequency', 'total_nights',
]
CATEGORICAL_FEATURES = [
    'hotel', 'arrival_date_month', 'meal', 'market_segment',
    'distribution_channel', 'is_repeated_guest', 'reserved_room_type',
    'deposit_type', 'customer_type', 'kids',
]
FINAL_FEATURES = NUMERIC_FEATURES[:6] + ['reserved_room_type'] + NUMERIC_FEATURES[6:9] + \
    ['country_frequency', 'total_nights', 'kids']  # placeholder, real order set below
FINAL_FEATURES = [
    'hotel', 'lead_time', 'arrival_date_month', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'adults', 'meal', 'market_segment',
    'distribution_channel', 'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'reserved_room_type', 'deposit_type',
    'customer_type', 'adr', 'required_car_parking_spaces',
    'total_of_special_requests', 'country_frequency', 'total_nights', 'kids',
]

# Results copied from the Modelling notebook's printed output (retraining
# all 8 models with CV + randomized search on every app load isn't practical
# on Streamlit Cloud's free tier)
MODEL_COMPARISON = pd.DataFrame([
    ["Logistic Regression (No Reg)", "Baseline", 0.8022, 0.8068, 0.6146, 0.6977, 0.8028, 0.8060, 0.6167, 0.6988],
    ["Ridge Logistic (L2)", "Baseline", 0.8023, 0.8065, 0.6153, 0.6980, 0.8023, 0.8058, 0.6153, 0.6978],
    ["Lasso Logistic (L1)", "Baseline", 0.8022, 0.8069, 0.6146, 0.6977, 0.8028, 0.8071, 0.6153, 0.6983],
    ["Elastic Net Logistic", "Baseline", 0.8023, 0.8064, 0.6154, 0.6980, 0.8022, 0.8059, 0.6151, 0.6977],
    ["Random Forest", "Baseline", 0.8796, 0.8697, 0.7948, 0.8306, 0.8611, 0.8614, 0.7455, 0.7993],
    ["XGBoost", "Baseline", 0.8632, 0.8410, 0.7790, 0.8088, 0.8548, 0.8340, 0.7597, 0.7951],
    ["LightGBM", "Baseline", 0.8567, 0.8391, 0.7601, 0.7977, 0.8515, 0.8350, 0.7475, 0.7888],
    ["CatBoost", "Baseline", 0.8648, 0.8463, 0.7772, 0.8103, 0.8632, 0.8486, 0.7681, 0.8064],
    ["Logistic Regression (No Reg)", "Tuned", 0.8022, 0.8068, 0.6146, 0.6977, 0.8028, 0.8060, 0.6167, 0.6988],
    ["Ridge Logistic (L2)", "Tuned", 0.8023, 0.8060, 0.6159, 0.6982, 0.8027, 0.8063, 0.6161, 0.6985],
    ["Lasso Logistic (L1)", "Tuned", 0.8024, 0.8075, 0.6146, 0.6979, 0.8024, 0.8064, 0.6151, 0.6978],
    ["Elastic Net Logistic", "Tuned", 0.8023, 0.8064, 0.6154, 0.6980, 0.8022, 0.8055, 0.6155, 0.6978],
    ["Random Forest", "Tuned", 0.8813, 0.8711, 0.7987, 0.8333, 0.8630, 0.8614, 0.7515, 0.8027],
    ["XGBoost", "Tuned", 0.8618, 0.8430, 0.7715, 0.8057, 0.8567, 0.8390, 0.7595, 0.7973],
    ["LightGBM", "Tuned", 0.8657, 0.8448, 0.7822, 0.8123, 0.8584, 0.8374, 0.7673, 0.8009],
    ["CatBoost", "Tuned", 0.8692, 0.8504, 0.7861, 0.8170, 0.8608, 0.8444, 0.7661, 0.8033],
], columns=["Model", "Stage", "CV Accuracy", "CV Precision", "CV Recall", "CV F1",
            "Test Accuracy", "Test Precision", "Test Recall", "Test F1"])

BEST_MODEL_METRICS = {
    "Test Accuracy": 0.8632, "Test Precision": 0.8486, "Test Recall": 0.7681,
    "Test F1": 0.8064, "ROC AUC": 0.9364,
}

FEATURE_IMPORTANCE = pd.DataFrame({
    "Feature": ["deposit_type", "country_frequency", "required_car_parking_spaces",
                "market_segment", "lead_time", "previous_cancellations", "customer_type",
                "adr", "total_of_special_requests", "arrival_date_week_number",
                "total_nights", "arrival_date_month", "hotel", "meal",
                "arrival_date_day_of_month", "reserved_room_type", "adults",
                "previous_bookings_not_canceled", "distribution_channel",
                "is_repeated_guest", "kids"],
    "Importance": [16.4065, 13.2914, 13.0137, 11.4600, 8.5208, 7.0946, 6.3215, 5.2190,
                   3.5753, 2.9505, 1.7985, 1.7894, 1.6312, 1.6182, 1.4529, 0.9706,
                   0.9108, 0.8468, 0.5661, 0.3405, 0.2219],
})


# =============================================================================
# DATA PIPELINE (mirrors the notebooks' cleaning + feature engineering)
# =============================================================================
@st.cache_data(show_spinner="Downloading dataset...")
def load_raw_data():
    return pd.read_csv(DATA_URL)


@st.cache_data(show_spinner="Cleaning data...")
def clean_data(df_raw):
    df = df_raw.copy()
    df = df.drop(columns=['agent', 'company', 'reservation_status_date', 'reservation_status'])
    df = df.dropna()
    df['children'] = df['children'].astype(int)

    cat_cols = ['hotel', 'arrival_date_month', 'arrival_date_year', 'meal', 'country',
                'market_segment', 'distribution_channel', 'reserved_room_type',
                'assigned_room_type', 'deposit_type', 'customer_type', 'is_repeated_guest']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    df = df[df['distribution_channel'] != 'Undefined']
    df['distribution_channel'] = df['distribution_channel'].cat.remove_unused_categories()

    df['country'] = df['country'].map(COUNTRY_NAME_MAP).astype('category')
    df['meal'] = df['meal'].astype(object).replace('Undefined', 'SC').astype('category')

    counts = df['reserved_room_type'].value_counts()
    rare = counts[counts < 10].index
    df['reserved_room_type'] = df['reserved_room_type'].astype(object).replace(rare, 'Other').astype('category')

    df = df[df['adr'] >= 0].reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Splitting data...")
def split_data(df):
    X = df.drop(columns='is_canceled')
    y = df['is_canceled']
    groups = pd.factorize(X.apply(tuple, axis=1))[0]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))
    return (X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True),
            y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True))


def engineer_features(X, country_freq):
    X = X.copy()
    X['country_frequency'] = X['country'].map(country_freq).fillna(0)
    X['total_nights'] = X['stays_in_weekend_nights'] + X['stays_in_week_nights']
    X['kids'] = ((X['children'] > 0) | (X['babies'] > 0)).astype('category')
    return X[FINAL_FEATURES]


@st.cache_data(show_spinner="Preparing model-ready data...")
def get_model_ready_data():
    df = clean_data(load_raw_data())
    X_train_raw, X_test_raw, y_train, y_test = split_data(df)
    country_freq = X_train_raw['country'].value_counts()
    X_train = engineer_features(X_train_raw, country_freq)
    X_test = engineer_features(X_test_raw, country_freq)
    return X_train, X_test, y_train, y_test, country_freq


@st.cache_resource(show_spinner="Loading model...")
def load_pipeline():
    return joblib.load("model/catboost_pipeline.joblib")


# =============================================================================
# HEADER
# =============================================================================
st.title("🏨 Hotel Booking Cancellation Analysis")
st.markdown(
    "End-to-end **EDA → Feature Selection → Modelling** walkthrough on the "
    "[Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) dataset "
    "(119,390 bookings, City & Resort hotels, 2015–2017)."
)

X_train, X_test, y_train, y_test, country_freq = get_model_ready_data()
eda_df = X_train.copy()
eda_df["is_canceled"] = y_train.values
eda_df["booking_status"] = eda_df["is_canceled"].map({0: "Not Cancelled", 1: "Cancelled"})
eda_df["arrival_date_month"] = pd.Categorical(eda_df["arrival_date_month"], categories=MONTH_ORDER, ordered=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Bookings analyzed", f"{len(X_train) + len(X_test):,}")
c2.metric("Cancellation rate", f"{pd.concat([y_train, y_test]).mean():.1%}")
c3.metric("Best model", "CatBoost")
c4.metric("Test F1 score", f"{BEST_MODEL_METRICS['Test F1']:.1%}")

tab_overview, tab_eda, tab_fs, tab_model = st.tabs(
    ["🏠 Overview", "📊 EDA", "🎯 Feature Selection", "🤖 Modelling & Prediction"]
)

# =============================================================================
# OVERVIEW TAB
# =============================================================================
with tab_overview:
    st.subheader("Project workflow")
    o1, o2, o3 = st.columns(3)
    with o1:
        st.markdown("**📊 EDA**")
        st.write("Cleaning, outlier checks, and visual exploration of cancellation drivers.")
    with o2:
        st.markdown("**🎯 Feature Selection**")
        st.write("VIF, correlation, permutation importance & mutual information to pick 21 final features.")
    with o3:
        st.markdown("**🤖 Modelling**")
        st.write("8 models (linear + tree-based), baseline vs. tuned, compared on CV + test set.")

    st.subheader("Headline result")
    r1, r2 = st.columns([1, 1.3])
    with r1:
        st.markdown(
            f"""
Best model: **CatBoost (baseline)**

| Metric | Test set |
|---|---|
| Accuracy | **{BEST_MODEL_METRICS['Test Accuracy']:.1%}** |
| Precision | **{BEST_MODEL_METRICS['Test Precision']:.1%}** |
| Recall | **{BEST_MODEL_METRICS['Test Recall']:.1%}** |
| F1 score | **{BEST_MODEL_METRICS['Test F1']:.1%}** |
| ROC-AUC | **{BEST_MODEL_METRICS['ROC AUC']:.1%}** |

Strongest signals: **deposit type**, **guest's home-country booking frequency**,
**required car-parking spaces**, and **market segment**.
"""
        )
    with r2:
        top = FEATURE_IMPORTANCE.head(8).sort_values("Importance")
        fig = px.bar(top, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Reds",
                     title="Top 8 features by importance")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=360, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Data source: Antonio, Almeida & Nunes (2019), *Hotel Booking Demand Datasets*, "
        "mirrored via the TidyTuesday project."
    )

# =============================================================================
# EDA TAB
# =============================================================================
with tab_eda:
    st.subheader("Overall booking outcome")
    e1, e2 = st.columns([1, 2])
    with e1:
        counts = eda_df["booking_status"].value_counts().reset_index()
        counts.columns = ["booking_status", "count"]
        fig = px.pie(counts, names="booking_status", values="count", hole=0.45,
                     color="booking_status", color_discrete_map={"Not Cancelled": BLUE, "Cancelled": RED})
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, height=340, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    with e2:
        hotel_stats = eda_df.groupby("hotel", observed=True)["is_canceled"].mean().mul(100).reset_index(name="rate")
        fig = px.bar(hotel_stats, x="hotel", y="rate", color="hotel", text_auto=".1f",
                     color_discrete_sequence=[BLUE, RED], title="Cancellation rate by hotel type",
                     labels={"rate": "Cancellation rate (%)"})
        fig.update_layout(showlegend=False, height=340, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Seasonality")
    monthly = (eda_df.groupby("arrival_date_month", observed=True)
               .agg(bookings=("is_canceled", "count"), rate=("is_canceled", "mean"))
               .reset_index().sort_values("arrival_date_month"))
    monthly["rate"] *= 100
    fig = go.Figure()
    fig.add_bar(x=monthly["arrival_date_month"], y=monthly["bookings"], name="Bookings", marker_color=BLUE)
    fig.add_scatter(x=monthly["arrival_date_month"], y=monthly["rate"], name="Cancellation rate (%)",
                     marker_color=RED, yaxis="y2", mode="lines+markers")
    fig.update_layout(title="Monthly bookings vs. cancellation rate", yaxis=dict(title="Bookings"),
                       yaxis2=dict(title="Cancellation rate (%)", overlaying="y", side="right"),
                       legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                       height=420, margin=dict(t=60, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cancellation rate by booking characteristics")
    tabs = st.tabs(["Market segment", "Deposit type", "Customer type", "Repeat guest / Kids"])

    def rate_bar(col, title):
        s = (eda_df.groupby(col, observed=True)
             .agg(total=("is_canceled", "count"), rate=("is_canceled", "mean")).reset_index())
        s["rate"] *= 100
        s = s.sort_values("rate", ascending=False)
        fig = px.bar(s, x=col, y="rate", color="rate", color_continuous_scale="Reds",
                     text_auto=".1f", title=title, labels={"rate": "Cancellation rate (%)"})
        fig.update_layout(coloraxis_showscale=False, height=400, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[0]:
        rate_bar("market_segment", "Cancellation rate by market segment")
    with tabs[1]:
        rate_bar("deposit_type", "Cancellation rate by deposit type")
    with tabs[2]:
        rate_bar("customer_type", "Cancellation rate by customer type")
    with tabs[3]:
        f1, f2 = st.columns(2)
        with f1:
            rate_bar("is_repeated_guest", "Repeat vs. new guest")
        with f2:
            rate_bar("kids", "Travelling with kids vs. without")

    st.subheader("ADR by booking outcome")
    adr_cap = eda_df["adr"].quantile(0.95)
    fig = px.box(eda_df[eda_df["adr"] <= adr_cap], x="booking_status", y="adr", color="booking_status",
                 color_discrete_map={"Not Cancelled": BLUE, "Cancelled": RED},
                 labels={"adr": "Average Daily Rate", "booking_status": ""})
    fig.update_layout(showlegend=False, height=400, margin=dict(t=20, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation between numerical features")
    corr = eda_df[NUMERIC_FEATURES + ["is_canceled"]].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=500, margin=dict(t=20, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# FEATURE SELECTION TAB
# =============================================================================
with tab_fs:
    st.subheader("Multicollinearity check (VIF)")
    with st.spinner("Computing VIF..."):
        X_num = X_train[NUMERIC_FEATURES]
        vif_df = pd.DataFrame({
            "Feature": X_num.columns,
            "VIF": [variance_inflation_factor(X_num.values, i) for i in range(X_num.shape[1])],
        }).sort_values("VIF", ascending=False).reset_index(drop=True)

    v1, v2 = st.columns([1.3, 1])
    with v1:
        fig = px.bar(vif_df.sort_values("VIF"), x="VIF", y="Feature", orientation="h",
                     color="VIF", color_continuous_scale="Reds", text_auto=".2f", title="VIF by numeric feature")
        fig.add_vline(x=5, line_dash="dash", line_color="orange")
        fig.add_vline(x=10, line_dash="dash", line_color="red")
        fig.update_layout(coloraxis_showscale=False, height=400, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    with v2:
        st.dataframe(vif_df.style.format({"VIF": "{:.2f}"}), use_container_width=True, height=400)
    st.info("All numeric features sit below VIF = 10, so none were dropped purely for multicollinearity.", icon="✅")

    st.subheader("Random Forest permutation importance")
    st.caption("Computed on a sampled Random Forest fit for feature-ranking purposes (not the final model).")

    @st.cache_data(show_spinner="Fitting Random Forest + computing permutation importance (~30s)...")
    def get_permutation_importance():
        sample = X_train.sample(min(8000, len(X_train)), random_state=42)
        y_sample = y_train.loc[sample.index]
        pre = ColumnTransformer([
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])
        X_enc = pre.fit_transform(sample)
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10,
                                     random_state=42, n_jobs=-1)
        rf.fit(X_enc, y_sample)
        perm = permutation_importance(rf, X_enc, y_sample, n_repeats=3, random_state=42, n_jobs=1)
        names = pre.get_feature_names_out()
        imp = pd.DataFrame({"Feature": names, "Importance": perm.importances_mean})

        def orig(f):
            if f.startswith("num__"):
                return f.replace("num__", "")
            if f.startswith("cat__"):
                f = f.replace("cat__", "")
                for c in CATEGORICAL_FEATURES:
                    if f.startswith(c + "_"):
                        return c
            return f

        imp["Original Feature"] = imp["Feature"].apply(orig)
        return imp.groupby("Original Feature", as_index=False)["Importance"].sum().sort_values(
            "Importance", ascending=False).reset_index(drop=True)

    perm_df = get_permutation_importance()
    fig = px.bar(perm_df.sort_values("Importance"), x="Importance", y="Original Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues", title="Permutation importance")
    fig.update_layout(coloraxis_showscale=False, height=500, margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mutual information with cancellation outcome")

    @st.cache_data(show_spinner="Computing mutual information...")
    def get_mutual_info():
        X = X_train.copy()
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].cat.codes
        mask = X.columns.isin(CATEGORICAL_FEATURES)
        mi = mutual_info_classif(X, y_train, discrete_features=mask, random_state=42)
        return pd.Series(mi, index=X.columns).sort_values(ascending=False)

    mi_series = get_mutual_info()
    mi_df = mi_series.reset_index()
    mi_df.columns = ["Feature", "MI Score"]
    fig = px.bar(mi_df.sort_values("MI Score"), x="MI Score", y="Feature", orientation="h",
                 color="MI Score", color_continuous_scale="Blues", title="Mutual information score")
    fig.update_layout(coloraxis_showscale=False, height=500, margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MODELLING & PREDICTION TAB
# =============================================================================
with tab_model:
    st.subheader("Model comparison")
    stage = st.radio("Show:", ["Baseline", "Tuned", "Both"], horizontal=True, index=2)
    table = MODEL_COMPARISON if stage == "Both" else MODEL_COMPARISON[MODEL_COMPARISON["Stage"] == stage]
    sorted_table = table.sort_values("Test F1", ascending=False).reset_index(drop=True)
    st.dataframe(
        sorted_table.style.format({c: "{:.4f}" for c in sorted_table.columns if c not in ("Model", "Stage")})
        .background_gradient(subset=["Test F1"], cmap="Reds"),
        use_container_width=True, height=350,
    )

    st.subheader("🏆 Best model: CatBoost (baseline)")
    m = BEST_MODEL_METRICS
    mc = st.columns(5)
    mc[0].metric("Accuracy", f"{m['Test Accuracy']:.1%}")
    mc[1].metric("Precision", f"{m['Test Precision']:.1%}")
    mc[2].metric("Recall", f"{m['Test Recall']:.1%}")
    mc[3].metric("F1", f"{m['Test F1']:.1%}")
    mc[4].metric("ROC-AUC", f"{m['ROC AUC']:.1%}")

    pipeline = load_pipeline()
    d1, d2 = st.columns(2)
    with d1:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=RED, width=3), name=f"CatBoost (AUC={auc:.3f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="gray", dash="dash"), name="Random")
        fig.update_layout(title="ROC curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                           height=380, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Reds",
                         x=["Not Cancelled", "Cancelled"], y=["Not Cancelled", "Cancelled"],
                         labels=dict(x="Predicted", y="Actual"), title="Confusion matrix")
        fig.update_layout(height=380, margin=dict(t=40, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature importance")
    fi = FEATURE_IMPORTANCE.sort_values("Importance")
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h", color="Importance",
                 color_continuous_scale="Reds", title="Aggregated CatBoost feature importance")
    fig.update_layout(coloraxis_showscale=False, height=500, margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🔮 Try it: cancellation-risk calculator")

    country_options = country_freq.sort_values(ascending=False).index.tolist()
    meal_labels = {"BB": "Bed & Breakfast", "HB": "Half Board", "FB": "Full Board", "SC": "Self Catering"}

    with st.form("booking_form"):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        hotel = r1c1.selectbox("Hotel type", ["City Hotel", "Resort Hotel"])
        lead_time = r1c2.number_input("Lead time (days)", 0, 709, 68)
        total_nights = r1c3.number_input("Total nights", 0, 56, 3)
        adults = r1c4.number_input("Adults", 0, 10, 2)

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        arrival_month = r2c1.selectbox("Arrival month", MONTH_ORDER, index=6)
        arrival_week = r2c2.slider("Arrival week #", 1, 53, 28)
        arrival_day = r2c3.slider("Arrival day of month", 1, 31, 16)
        kids = r2c4.checkbox("Travelling with kids")

        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        market_segment = r3c1.selectbox("Market segment",
                                         ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups",
                                          "Complementary", "Aviation"])
        distribution_channel = r3c2.selectbox("Distribution channel", ["TA/TO", "Direct", "Corporate", "GDS"])
        customer_type = r3c3.selectbox("Customer type", ["Transient", "Transient-Party", "Contract", "Group"])
        is_repeated_guest = r3c4.selectbox("Repeated guest?", ["No", "Yes"])

        r4c1, r4c2, r4c3, r4c4 = st.columns(4)
        previous_cancellations = r4c1.number_input("Previous cancellations", 0, 25, 0)
        previous_bookings_not_canceled = r4c2.number_input("Previous bookings honoured", 0, 72, 0)
        country = r4c3.selectbox("Guest's country", country_options,
                                  index=country_options.index("United Kingdom") if "United Kingdom" in country_options else 0)
        meal = r4c4.selectbox("Meal plan", list(meal_labels.keys()), format_func=lambda k: meal_labels[k])

        r5c1, r5c2, r5c3, r5c4 = st.columns(4)
        reserved_room_type = r5c1.selectbox("Reserved room type", ["A", "B", "C", "D", "E", "F", "G", "H", "Other"])
        deposit_type = r5c2.selectbox("Deposit type", ["No Deposit", "Refundable", "Non Refund"])
        adr = r5c3.number_input("ADR", 0.0, 2000.0, 95.0, step=5.0)
        required_car_parking_spaces = r5c4.number_input("Parking spaces required", 0, 8, 0)

        total_of_special_requests = st.slider("Total special requests", 0, 5, 0)
        submitted = st.form_submit_button("Predict cancellation risk", type="primary", use_container_width=True)

    if submitted:
        booking = pd.DataFrame([{
            "hotel": hotel, "lead_time": lead_time, "arrival_date_month": arrival_month,
            "arrival_date_week_number": arrival_week, "arrival_date_day_of_month": arrival_day,
            "adults": adults, "meal": meal, "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "is_repeated_guest": 1 if is_repeated_guest == "Yes" else 0,
            "previous_cancellations": previous_cancellations,
            "previous_bookings_not_canceled": previous_bookings_not_canceled,
            "reserved_room_type": reserved_room_type, "deposit_type": deposit_type,
            "customer_type": customer_type, "adr": adr,
            "required_car_parking_spaces": required_car_parking_spaces,
            "total_of_special_requests": total_of_special_requests,
            "country_frequency": int(country_freq.get(country, 0)),
            "total_nights": total_nights, "kids": bool(kids),
        }])[FINAL_FEATURES]

        prob = float(pipeline.predict_proba(booking)[0, 1])
        risk_label, risk_color = ("High risk", "#B22222") if prob >= 0.6 else \
            ("Medium risk", "#e67e22") if prob >= 0.35 else ("Low risk", "#27ae60")

        pc1, pc2 = st.columns([1, 2])
        with pc1:
            st.metric("Predicted cancellation probability", f"{prob:.1%}")
            st.markdown(f"<h3 style='color:{risk_color}'>{risk_label}</h3>", unsafe_allow_html=True)
        with pc2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100, number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": risk_color},
                       "steps": [{"range": [0, 35], "color": "#e8f8f0"},
                                 {"range": [35, 60], "color": "#fdf2e3"},
                                 {"range": [60, 100], "color": "#fbeaea"}]},
            ))
            fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
