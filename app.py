import io
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Smart Inventory Prediction - Customer Intelligence App",
    page_icon="📦",
    layout="wide",
)


SURVEY_COLUMNS = [
    "State",
    "City_Type",
    "Store_Type",
    "Store_Count",
    "SKU_Count",
    "Inventory_Management",
    "Ordering_Method",
    "Stockout_Frequency",
    "Overstock_Frequency",
    "Unsold_Inventory_Percentage",
    "Supplier_Count",
    "Supplier_Lead_Time",
    "Top_Selling_Category",
    "Demand_Driver",
    "Peak_Season_Demand_Change",
    "Combination_Purchase_Frequency",
    "Common_Product_Combinations",
    "Digital_Sales_Data_Years",
    "Decision_Maker",
    "AI_Comfort_Level",
    "Most_Valuable_Feature",
    "Monthly_Inventory_Purchase_Value",
    "Monthly_Software_Budget",
    "Adoption_Motivation",
    "Adoption_Likelihood",
]

MODEL_FEATURES = SURVEY_COLUMNS[:-1]
TARGET_CLASS = "Adoption_Likelihood"
TARGET_BINARY = "Interested_Binary"
TARGET_BUDGET_NUM = "Software_Budget_Midpoint"
TARGET_PURCHASE_NUM = "Inventory_Purchase_Midpoint"
CLUSTER_COL = "Customer_Persona"
LEAD_SCORE_COL = "Lead_Priority"


def budget_to_midpoint(value: str) -> float:
    mapping = {
        "Less than ₹1000": 750,
        "₹1000–₹3000": 2000,
        "₹3000–₹7000": 5000,
        "₹7000–₹15000": 11000,
        "More than ₹15000": 20000,
    }
    return mapping.get(value, np.nan)



def purchase_to_midpoint(value: str) -> float:
    mapping = {
        "Less than ₹1 lakh": 75000,
        "₹1–5 lakh": 300000,
        "₹5–10 lakh": 750000,
        "₹10–50 lakh": 3000000,
        "More than ₹50 lakh": 6000000,
    }
    return mapping.get(value, np.nan)



def adoption_to_binary(value: str) -> int:
    return int(value in ["Very likely", "Likely"])



def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_BINARY] = df[TARGET_CLASS].apply(adoption_to_binary)
    df[TARGET_BUDGET_NUM] = df["Monthly_Software_Budget"].apply(budget_to_midpoint)
    df[TARGET_PURCHASE_NUM] = df["Monthly_Inventory_Purchase_Value"].apply(purchase_to_midpoint)
    return df



def choose(rng: np.random.Generator, options: List[str], probs: List[float]) -> str:
    return str(rng.choice(options, p=probs))


@st.cache_data(show_spinner=False)
def generate_synthetic_dataset(n_rows: int = 2200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    states = [
        "Maharashtra", "Karnataka", "Telangana", "Tamil Nadu", "Delhi/NCR",
        "Gujarat", "West Bengal", "Uttar Pradesh", "Rajasthan", "Other"
    ]
    state_probs = [0.15, 0.11, 0.1, 0.1, 0.1, 0.08, 0.07, 0.1, 0.05, 0.14]

    city_types = ["Tier 1 city", "Tier 2 city", "Tier 3 city / small town", "Rural"]
    city_probs = [0.4, 0.35, 0.2, 0.05]

    store_types = [
        "Grocery / Supermarket", "Pharmacy / Medical Store", "Apparel / Fashion",
        "Electronics", "Beauty / Cosmetics", "General Store / Kirana", "Other"
    ]
    store_probs = [0.31, 0.16, 0.18, 0.1, 0.1, 0.12, 0.03]

    rows = []
    for _ in range(n_rows):
        state = choose(rng, states, state_probs)
        city = choose(rng, city_types, city_probs)
        store_type = choose(rng, store_types, store_probs)

        # Store count conditional on city/store type
        if store_type in ["General Store / Kirana", "Other"]:
            store_count = choose(rng, ["1 store", "2–5 stores", "6–10 stores", "11–25 stores", "25+ stores"], [0.72, 0.2, 0.05, 0.02, 0.01])
        elif city == "Tier 1 city":
            store_count = choose(rng, ["1 store", "2–5 stores", "6–10 stores", "11–25 stores", "25+ stores"], [0.32, 0.36, 0.18, 0.1, 0.04])
        else:
            store_count = choose(rng, ["1 store", "2–5 stores", "6–10 stores", "11–25 stores", "25+ stores"], [0.5, 0.31, 0.12, 0.05, 0.02])

        sku_map = {
            "1 store": ["Less than 500", "500–2000", "2000–5000", "5000–10000", "More than 10000"],
            "2–5 stores": ["Less than 500", "500–2000", "2000–5000", "5000–10000", "More than 10000"],
            "6–10 stores": ["Less than 500", "500–2000", "2000–5000", "5000–10000", "More than 10000"],
            "11–25 stores": ["Less than 500", "500–2000", "2000–5000", "5000–10000", "More than 10000"],
            "25+ stores": ["Less than 500", "500–2000", "2000–5000", "5000–10000", "More than 10000"],
        }
        sku_probs = {
            "1 store": [0.42, 0.42, 0.11, 0.04, 0.01],
            "2–5 stores": [0.14, 0.46, 0.26, 0.11, 0.03],
            "6–10 stores": [0.05, 0.22, 0.4, 0.24, 0.09],
            "11–25 stores": [0.02, 0.12, 0.3, 0.35, 0.21],
            "25+ stores": [0.01, 0.06, 0.18, 0.35, 0.4],
        }
        sku_count = choose(rng, sku_map[store_count], sku_probs[store_count])

        if city in ["Tier 1 city", "Tier 2 city"] and store_count in ["6–10 stores", "11–25 stores", "25+ stores"]:
            inventory_mgmt = choose(rng, [
                "Manual notebook / paper records", "Excel / spreadsheets", "POS or billing software",
                "ERP or inventory management software", "Combination of multiple tools"
            ], [0.03, 0.12, 0.3, 0.27, 0.28])
        elif city == "Rural":
            inventory_mgmt = choose(rng, [
                "Manual notebook / paper records", "Excel / spreadsheets", "POS or billing software",
                "ERP or inventory management software", "Combination of multiple tools"
            ], [0.42, 0.26, 0.22, 0.03, 0.07])
        else:
            inventory_mgmt = choose(rng, [
                "Manual notebook / paper records", "Excel / spreadsheets", "POS or billing software",
                "ERP or inventory management software", "Combination of multiple tools"
            ], [0.14, 0.29, 0.28, 0.11, 0.18])

        if inventory_mgmt in ["ERP or inventory management software", "POS or billing software"]:
            ordering_method = choose(rng, [
                "Past experience / intuition", "Supplier recommendations", "Sales reports / historical data",
                "Software analytics", "Combination of methods"
            ], [0.08, 0.12, 0.35, 0.25, 0.2])
        else:
            ordering_method = choose(rng, [
                "Past experience / intuition", "Supplier recommendations", "Sales reports / historical data",
                "Software analytics", "Combination of methods"
            ], [0.38, 0.2, 0.18, 0.04, 0.2])

        # pain levels vary by retail type and maturity
        stockout = choose(rng, ["Daily", "Weekly", "Monthly", "Rarely", "Almost never"],
                          [0.08, 0.28, 0.36, 0.22, 0.06] if store_type in ["Grocery / Supermarket", "Pharmacy / Medical Store"]
                          else [0.05, 0.18, 0.34, 0.31, 0.12])
        overstock = choose(rng, ["Very frequently", "Occasionally", "Rarely", "Never"],
                           [0.14, 0.45, 0.33, 0.08] if store_type in ["Apparel / Fashion", "Beauty / Cosmetics"]
                           else [0.08, 0.39, 0.41, 0.12])

        unsold = choose(rng, ["Less than 5%", "5–10%", "10–20%", "20–30%", "More than 30%"],
                        [0.22, 0.34, 0.28, 0.11, 0.05] if overstock in ["Rarely", "Never"]
                        else [0.08, 0.24, 0.37, 0.2, 0.11])

        if store_count in ["11–25 stores", "25+ stores"]:
            supplier_count = choose(rng, ["1–3 suppliers", "4–10 suppliers", "10–20 suppliers", "More than 20 suppliers"], [0.03, 0.18, 0.41, 0.38])
        elif store_count == "6–10 stores":
            supplier_count = choose(rng, ["1–3 suppliers", "4–10 suppliers", "10–20 suppliers", "More than 20 suppliers"], [0.08, 0.42, 0.34, 0.16])
        else:
            supplier_count = choose(rng, ["1–3 suppliers", "4–10 suppliers", "10–20 suppliers", "More than 20 suppliers"], [0.38, 0.42, 0.16, 0.04])

        lead_time = choose(rng, ["Same day", "1–2 days", "3–5 days", "1 week", "Highly variable"],
                           [0.05, 0.26, 0.36, 0.19, 0.14] if supplier_count in ["10–20 suppliers", "More than 20 suppliers"]
                           else [0.09, 0.36, 0.33, 0.14, 0.08])

        category_map = {
            "Grocery / Supermarket": "FMCG / grocery items",
            "Pharmacy / Medical Store": "Medicines / healthcare products",
            "Apparel / Fashion": "Apparel / clothing",
            "Electronics": "Electronics",
            "Beauty / Cosmetics": "Beauty / cosmetics",
            "General Store / Kirana": "FMCG / grocery items",
            "Other": choose(rng, ["FMCG / grocery items", "Medicines / healthcare products", "Apparel / clothing", "Electronics", "Beauty / cosmetics"], [0.28, 0.18, 0.22, 0.16, 0.16]),
        }
        top_category = category_map[store_type]

        if store_type in ["Grocery / Supermarket", "Apparel / Fashion", "Beauty / Cosmetics"]:
            demand_driver = choose(rng, ["Festivals", "Seasonal weather", "Promotions / discounts", "Social media trends", "Regular repeat customers"], [0.25, 0.14, 0.26, 0.09, 0.26])
        elif store_type == "Pharmacy / Medical Store":
            demand_driver = choose(rng, ["Festivals", "Seasonal weather", "Promotions / discounts", "Social media trends", "Regular repeat customers"], [0.03, 0.34, 0.05, 0.01, 0.57])
        else:
            demand_driver = choose(rng, ["Festivals", "Seasonal weather", "Promotions / discounts", "Social media trends", "Regular repeat customers"], [0.14, 0.16, 0.24, 0.12, 0.34])

        peak_change = choose(rng, ["Increases significantly", "Slight increase", "No change", "Difficult to predict"],
                             [0.37, 0.39, 0.11, 0.13] if demand_driver in ["Festivals", "Promotions / discounts"]
                             else [0.16, 0.42, 0.21, 0.21])

        combo_freq = choose(rng, ["Yes frequently", "Sometimes", "Rarely", "Never"], [0.33, 0.45, 0.18, 0.04])
        combos_map = {
            "FMCG / grocery items": ["Rice + cooking oil", "Snacks + beverages", "Other"],
            "Medicines / healthcare products": ["Medicines + health supplements", "Other"],
            "Apparel / clothing": ["Saree + blouse", "Other"],
            "Electronics": ["Other"],
            "Beauty / cosmetics": ["Shampoo + conditioner", "Other"],
        }
        combo_options = combos_map[top_category]
        if len(combo_options) == 1:
            common_combo = combo_options[0]
        else:
            probs = [0.62, 0.28, 0.10] if len(combo_options) == 3 else [0.8, 0.2]
            common_combo = choose(rng, combo_options, probs)

        if inventory_mgmt == "Manual notebook / paper records":
            digital_years = choose(rng, ["No digital data", "Less than 1 year", "1–3 years", "More than 3 years"], [0.72, 0.17, 0.08, 0.03])
        elif inventory_mgmt == "Excel / spreadsheets":
            digital_years = choose(rng, ["No digital data", "Less than 1 year", "1–3 years", "More than 3 years"], [0.08, 0.23, 0.42, 0.27])
        else:
            digital_years = choose(rng, ["No digital data", "Less than 1 year", "1–3 years", "More than 3 years"], [0.02, 0.12, 0.36, 0.5])

        if store_count == "1 store":
            decision_maker = choose(rng, ["Store owner", "Operations manager", "IT team", "Head office / franchise owner", "Distributor recommendation"], [0.76, 0.12, 0.01, 0.05, 0.06])
        elif store_count in ["2–5 stores", "6–10 stores"]:
            decision_maker = choose(rng, ["Store owner", "Operations manager", "IT team", "Head office / franchise owner", "Distributor recommendation"], [0.37, 0.31, 0.06, 0.21, 0.05])
        else:
            decision_maker = choose(rng, ["Store owner", "Operations manager", "IT team", "Head office / franchise owner", "Distributor recommendation"], [0.08, 0.28, 0.18, 0.44, 0.02])

        if city == "Tier 1 city" and digital_years in ["1–3 years", "More than 3 years"]:
            ai_comfort = choose(rng, ["Very comfortable", "Somewhat comfortable", "Neutral", "Not comfortable"], [0.31, 0.43, 0.17, 0.09])
        elif digital_years == "No digital data":
            ai_comfort = choose(rng, ["Very comfortable", "Somewhat comfortable", "Neutral", "Not comfortable"], [0.02, 0.14, 0.33, 0.51])
        else:
            ai_comfort = choose(rng, ["Very comfortable", "Somewhat comfortable", "Neutral", "Not comfortable"], [0.11, 0.36, 0.31, 0.22])

        if store_type in ["Grocery / Supermarket", "Pharmacy / Medical Store"]:
            valuable_feature = choose(rng, ["Demand forecasting", "Automatic reorder suggestions", "Inventory alerts", "Supplier performance insights", "Promotion demand prediction"], [0.28, 0.29, 0.2, 0.14, 0.09])
        else:
            valuable_feature = choose(rng, ["Demand forecasting", "Automatic reorder suggestions", "Inventory alerts", "Supplier performance insights", "Promotion demand prediction"], [0.24, 0.22, 0.16, 0.11, 0.27])

        if store_count == "1 store" and city != "Tier 1 city":
            monthly_purchase = choose(rng, ["Less than ₹1 lakh", "₹1–5 lakh", "₹5–10 lakh", "₹10–50 lakh", "More than ₹50 lakh"], [0.38, 0.46, 0.11, 0.04, 0.01])
        elif store_count in ["2–5 stores", "6–10 stores"]:
            monthly_purchase = choose(rng, ["Less than ₹1 lakh", "₹1–5 lakh", "₹5–10 lakh", "₹10–50 lakh", "More than ₹50 lakh"], [0.09, 0.38, 0.29, 0.2, 0.04])
        else:
            monthly_purchase = choose(rng, ["Less than ₹1 lakh", "₹1–5 lakh", "₹5–10 lakh", "₹10–50 lakh", "More than ₹50 lakh"], [0.01, 0.1, 0.21, 0.46, 0.22])

        # base monthly software budget from purchase + digital maturity
        if monthly_purchase == "Less than ₹1 lakh":
            budget = choose(rng, ["Less than ₹1000", "₹1000–₹3000", "₹3000–₹7000", "₹7000–₹15000", "More than ₹15000"], [0.62, 0.28, 0.08, 0.015, 0.005])
        elif monthly_purchase == "₹1–5 lakh":
            budget = choose(rng, ["Less than ₹1000", "₹1000–₹3000", "₹3000–₹7000", "₹7000–₹15000", "More than ₹15000"], [0.18, 0.44, 0.25, 0.10, 0.03])
        elif monthly_purchase == "₹5–10 lakh":
            budget = choose(rng, ["Less than ₹1000", "₹1000–₹3000", "₹3000–₹7000", "₹7000–₹15000", "More than ₹15000"], [0.05, 0.23, 0.38, 0.25, 0.09])
        elif monthly_purchase == "₹10–50 lakh":
            budget = choose(rng, ["Less than ₹1000", "₹1000–₹3000", "₹3000–₹7000", "₹7000–₹15000", "More than ₹15000"], [0.02, 0.1, 0.24, 0.41, 0.23])
        else:
            budget = choose(rng, ["Less than ₹1000", "₹1000–₹3000", "₹3000–₹7000", "₹7000–₹15000", "More than ₹15000"], [0.005, 0.04, 0.16, 0.42, 0.375])

        motivations = ["Reduced stockouts", "Lower inventory cost", "Increased revenue", "Automated purchasing", "Better supplier management"]
        if stockout in ["Daily", "Weekly"]:
            adoption_motivation = choose(rng, motivations, [0.42, 0.18, 0.18, 0.14, 0.08])
        elif overstock == "Very frequently":
            adoption_motivation = choose(rng, motivations, [0.16, 0.38, 0.12, 0.16, 0.18])
        elif supplier_count in ["10–20 suppliers", "More than 20 suppliers"]:
            adoption_motivation = choose(rng, motivations, [0.16, 0.18, 0.12, 0.19, 0.35])
        else:
            adoption_motivation = choose(rng, motivations, [0.24, 0.23, 0.21, 0.17, 0.15])

        # adoption score
        score = 0
        score += {"Very comfortable": 3, "Somewhat comfortable": 2, "Neutral": 1, "Not comfortable": 0}[ai_comfort]
        score += {"Daily": 3, "Weekly": 2, "Monthly": 1, "Rarely": 0, "Almost never": 0}[stockout]
        score += {"Very frequently": 2, "Occasionally": 1, "Rarely": 0, "Never": 0}[overstock]
        score += {"No digital data": 0, "Less than 1 year": 1, "1–3 years": 2, "More than 3 years": 3}[digital_years]
        score += {"Less than ₹1000": 0, "₹1000–₹3000": 1, "₹3000–₹7000": 2, "₹7000–₹15000": 3, "More than ₹15000": 4}[budget]
        score += {"1 store": 0, "2–5 stores": 1, "6–10 stores": 2, "11–25 stores": 3, "25+ stores": 4}[store_count]
        if store_type in ["Grocery / Supermarket", "Pharmacy / Medical Store"]:
            score += 1
        if valuable_feature in ["Demand forecasting", "Automatic reorder suggestions"]:
            score += 1
        score += int(rng.normal(0, 1.2))

        if score >= 12:
            adoption = choose(rng, ["Very likely", "Likely"], [0.63, 0.37])
        elif score >= 9:
            adoption = choose(rng, ["Likely", "Maybe"], [0.56, 0.44])
        elif score >= 6:
            adoption = choose(rng, ["Maybe", "Unlikely"], [0.65, 0.35])
        elif score >= 3:
            adoption = choose(rng, ["Unlikely", "Definitely not"], [0.62, 0.38])
        else:
            adoption = choose(rng, ["Definitely not", "Unlikely"], [0.7, 0.3])

        rows.append({
            "State": state,
            "City_Type": city,
            "Store_Type": store_type,
            "Store_Count": store_count,
            "SKU_Count": sku_count,
            "Inventory_Management": inventory_mgmt,
            "Ordering_Method": ordering_method,
            "Stockout_Frequency": stockout,
            "Overstock_Frequency": overstock,
            "Unsold_Inventory_Percentage": unsold,
            "Supplier_Count": supplier_count,
            "Supplier_Lead_Time": lead_time,
            "Top_Selling_Category": top_category,
            "Demand_Driver": demand_driver,
            "Peak_Season_Demand_Change": peak_change,
            "Combination_Purchase_Frequency": combo_freq,
            "Common_Product_Combinations": common_combo,
            "Digital_Sales_Data_Years": digital_years,
            "Decision_Maker": decision_maker,
            "AI_Comfort_Level": ai_comfort,
            "Most_Valuable_Feature": valuable_feature,
            "Monthly_Inventory_Purchase_Value": monthly_purchase,
            "Monthly_Software_Budget": budget,
            "Adoption_Motivation": adoption_motivation,
            "Adoption_Likelihood": adoption,
        })

    df = pd.DataFrame(rows)

    # Inject noise
    noise_idx = rng.choice(df.index, size=max(1, int(0.07 * len(df))), replace=False)
    for idx in noise_idx:
        col = rng.choice(["Inventory_Management", "AI_Comfort_Level", "Monthly_Software_Budget", "Supplier_Lead_Time"])
        if col == "Inventory_Management":
            df.at[idx, col] = choose(rng, [
                "Manual notebook / paper records", "Excel / spreadsheets", "POS or billing software",
                "ERP or inventory management software", "Combination of multiple tools"
            ], [0.2, 0.2, 0.2, 0.2, 0.2])
        elif col == "AI_Comfort_Level":
            df.at[idx, col] = choose(rng, ["Very comfortable", "Somewhat comfortable", "Neutral", "Not comfortable"], [0.25, 0.25, 0.25, 0.25])
        elif col == "Monthly_Software_Budget":
            df.at[idx, col] = choose(rng, ["Less than ₹1000", "₹1000–₹3000", "₹3000–₹7000", "₹7000–₹15000", "More than ₹15000"], [0.2, 0.2, 0.2, 0.2, 0.2])
        else:
            df.at[idx, col] = choose(rng, ["Same day", "1–2 days", "3–5 days", "1 week", "Highly variable"], [0.2, 0.2, 0.2, 0.2, 0.2])

    # Inject outliers
    outlier_idx = rng.choice(df.index, size=max(1, int(0.025 * len(df))), replace=False)
    for idx in outlier_idx:
        df.at[idx, "Store_Count"] = "25+ stores"
        df.at[idx, "SKU_Count"] = "More than 10000"
        df.at[idx, "Monthly_Inventory_Purchase_Value"] = "More than ₹50 lakh"
        df.at[idx, "Monthly_Software_Budget"] = "More than ₹15000"
        df.at[idx, "Digital_Sales_Data_Years"] = "More than 3 years"
        df.at[idx, "AI_Comfort_Level"] = "Very comfortable"

    # Missing values
    for col in df.columns:
        missing_idx = rng.choice(df.index, size=max(1, int(0.03 * len(df))), replace=False)
        df.loc[missing_idx, col] = np.where(rng.random(len(missing_idx)) < 0.5, df.loc[missing_idx, col], np.nan)

    return add_derived_columns(df)



def build_preprocessor(columns: List[str]) -> ColumnTransformer:
    categorical_features = columns
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        ]
    )


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame) -> Dict[str, object]:
    model_df = df.copy()
    X = model_df[MODEL_FEATURES]
    y_class = model_df[TARGET_BINARY]
    y_reg = model_df[TARGET_BUDGET_NUM]

    preprocessor = build_preprocessor(MODEL_FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.25, random_state=42, stratify=y_class
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
        ]
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    class_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    # Feature importance from trained RF
    ohe = clf.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    feature_names = ohe.get_feature_names_out(MODEL_FEATURES)
    importances = clf.named_steps["model"].feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(20)

    # Regression
    reg_df = model_df.dropna(subset=[TARGET_BUDGET_NUM])
    Xr = reg_df[MODEL_FEATURES]
    yr = reg_df[TARGET_BUDGET_NUM]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.25, random_state=42)

    reg = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(MODEL_FEATURES)),
            ("model", RandomForestRegressor(n_estimators=250, random_state=42)),
        ]
    )
    reg.fit(Xr_train, yr_train)
    yr_pred = reg.predict(Xr_test)

    regression_metrics = {
        "mae": mean_absolute_error(yr_test, yr_pred),
        "rmse": math.sqrt(mean_squared_error(yr_test, yr_pred)),
        "r2": r2_score(yr_test, yr_pred),
        "y_test": yr_test,
        "y_pred": yr_pred,
    }

    # Clustering on selected transformed features
    cluster_features = [
        "City_Type", "Store_Type", "Store_Count", "SKU_Count", "Inventory_Management",
        "Stockout_Frequency", "Overstock_Frequency", "Supplier_Count", "Supplier_Lead_Time",
        "Digital_Sales_Data_Years", "AI_Comfort_Level", "Monthly_Software_Budget",
    ]
    cluster_preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cluster_features,
            )
        ]
    )
    X_cluster = cluster_preprocessor.fit_transform(model_df[cluster_features])
    scaler = StandardScaler(with_mean=False)
    X_cluster_scaled = scaler.fit_transform(X_cluster)

    best_k = None
    best_score = -1
    best_model = None
    for k in range(3, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_cluster_scaled)
        score = silhouette_score(X_cluster_scaled, labels)
        if score > best_score:
            best_k = k
            best_score = score
            best_model = km

    cluster_labels = best_model.predict(X_cluster_scaled)
    persona_names = {
        rank: name for rank, name in enumerate([
            "Growth-ready retailers",
            "Traditional low-tech retailers",
            "High-pain high-potential adopters",
            "Large organized chains",
            "Price-sensitive digital explorers",
            "Operationally complex enterprises",
        ][:best_k])
    }
    persona_series = pd.Series(cluster_labels).map(persona_names)

    model_df = model_df.copy()
    model_df[CLUSTER_COL] = persona_series.values

    pca = PCA(n_components=2, random_state=42)
    pca_points = pca.fit_transform(X_cluster_scaled.toarray() if hasattr(X_cluster_scaled, "toarray") else X_cluster_scaled)
    pca_df = pd.DataFrame({
        "PC1": pca_points[:, 0],
        "PC2": pca_points[:, 1],
        CLUSTER_COL: model_df[CLUSTER_COL],
        "Store_Type": model_df["Store_Type"],
        TARGET_CLASS: model_df[TARGET_CLASS],
    })

    # Build lead priority rules
    scored = model_df.copy()
    scored["Adoption_Probability"] = clf.predict_proba(scored[MODEL_FEATURES])[:, 1]
    scored["Predicted_Budget"] = reg.predict(scored[MODEL_FEATURES])
    scored[LEAD_SCORE_COL] = np.select(
        [
            (scored["Adoption_Probability"] >= 0.75) & (scored["Predicted_Budget"] >= 5000),
            (scored["Adoption_Probability"] >= 0.5),
        ],
        ["High Priority", "Medium Priority"],
        default="Low Priority",
    )

    return {
        "classifier": clf,
        "class_metrics": class_metrics,
        "feature_importance": feature_importance_df,
        "regressor": reg,
        "regression_metrics": regression_metrics,
        "cluster_preprocessor": cluster_preprocessor,
        "cluster_scaler": scaler,
        "cluster_model": best_model,
        "cluster_persona_names": persona_names,
        "cluster_features": cluster_features,
        "clustered_df": scored,
        "pca_df": pca_df,
        "best_k": best_k,
        "best_silhouette": best_score,
    }



def predict_persona(df_new: pd.DataFrame, assets: Dict[str, object]) -> pd.Series:
    X_new = assets["cluster_preprocessor"].transform(df_new[assets["cluster_features"]])
    X_new_scaled = assets["cluster_scaler"].transform(X_new)
    labels = assets["cluster_model"].predict(X_new_scaled)
    return pd.Series(labels).map(assets["cluster_persona_names"])



def recommend_action(prob: float, budget: float, persona: str) -> str:
    if prob >= 0.75 and budget >= 7000:
        return f"Prioritize sales demo and Growth/Enterprise pitch for {persona}."
    if prob >= 0.6:
        return f"Run ROI-led nurture campaign and offer pilot/trial for {persona}."
    if prob >= 0.4:
        return f"Use educational marketing, webinar content, and light follow-up for {persona}."
    return f"Keep in low-cost awareness funnel; revisit later for {persona}."



def score_new_customers(df_new: pd.DataFrame, assets: Dict[str, object]) -> pd.DataFrame:
    scored = df_new.copy()
    scored["Predicted_Interest_Probability"] = assets["classifier"].predict_proba(scored[MODEL_FEATURES])[:, 1]
    scored["Predicted_Interest_Label"] = np.where(
        scored["Predicted_Interest_Probability"] >= 0.5, "Interested", "Not Interested"
    )
    scored["Predicted_Monthly_Software_Budget"] = assets["regressor"].predict(scored[MODEL_FEATURES]).round(0)
    scored[CLUSTER_COL] = predict_persona(scored, assets)
    scored[LEAD_SCORE_COL] = np.select(
        [
            (scored["Predicted_Interest_Probability"] >= 0.75) & (scored["Predicted_Monthly_Software_Budget"] >= 5000),
            (scored["Predicted_Interest_Probability"] >= 0.5),
        ],
        ["High Priority", "Medium Priority"],
        default="Low Priority",
    )
    scored["Recommended_Action"] = scored.apply(
        lambda r: recommend_action(
            float(r["Predicted_Interest_Probability"]),
            float(r["Predicted_Monthly_Software_Budget"]),
            str(r[CLUSTER_COL]),
        ),
        axis=1,
    )
    return scored



def make_downloadable_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")



def render_metric_row(metrics: Dict[str, float]) -> None:
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    cols[1].metric("Precision", f"{metrics['precision']:.3f}")
    cols[2].metric("Recall", f"{metrics['recall']:.3f}")
    cols[3].metric("F1-score", f"{metrics['f1']:.3f}")
    cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")



def parse_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload a CSV or XLSX file.")



def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    missing = [col for col in required_columns if col not in df.columns]
    return (len(missing) == 0, missing)



def get_association_rules(df: pd.DataFrame) -> pd.DataFrame:
    association_fields = [
        "Store_Type",
        "Inventory_Management",
        "Stockout_Frequency",
        "Overstock_Frequency",
        "Demand_Driver",
        "Most_Valuable_Feature",
        "Adoption_Motivation",
        "Common_Product_Combinations",
        "Adoption_Likelihood",
    ]
    transactions = []
    assoc_df = df[association_fields].fillna("Missing")
    for _, row in assoc_df.iterrows():
        transactions.append([f"{col}: {row[col]}" for col in association_fields])

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent = apriori(trans_df, min_support=0.05, use_colnames=True)
    if frequent.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    rules = association_rules(frequent, metric="confidence", min_threshold=0.55)
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    return rules.sort_values(["lift", "confidence"], ascending=False).head(20)


st.title("📦 Smart Inventory Prediction - Retail Customer Intelligence App")
st.markdown(
    "Analyze current respondents, identify the best customer segments, train predictive models, "
    "discover association rules, and score future leads for targeted marketing."
)

with st.sidebar:
    st.header("Data controls")
    source_mode = st.radio("Choose dataset source", ["Use built-in synthetic dataset", "Upload your own full dataset"])
    if source_mode == "Use built-in synthetic dataset":
        n_rows = st.slider("Synthetic respondent count", min_value=2000, max_value=5000, value=2200, step=100)
        seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42)
        dataset = generate_synthetic_dataset(n_rows=n_rows, seed=int(seed))
    else:
        upload = st.file_uploader("Upload full survey dataset (CSV/XLSX)", type=["csv", "xlsx"])
        if upload is not None:
            uploaded_df = parse_upload(upload)
            ok, missing_cols = validate_columns(uploaded_df, SURVEY_COLUMNS)
            if not ok:
                st.error(f"Uploaded dataset is missing required columns: {missing_cols}")
                st.stop()
            dataset = add_derived_columns(uploaded_df[SURVEY_COLUMNS].copy())
        else:
            st.info("Upload a dataset to continue. Using built-in synthetic data for now.")
            dataset = generate_synthetic_dataset(n_rows=2200, seed=42)

    st.download_button(
        label="Download current dataset as CSV",
        data=make_downloadable_csv(dataset),
        file_name="synthetic_retail_customer_survey.csv",
        mime="text/csv",
    )

assets = train_models(dataset)
clustered_df = assets["clustered_df"]
rules_df = get_association_rules(clustered_df)


# top-line metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Respondents", f"{len(clustered_df):,}")
m2.metric("Interested share", f"{clustered_df[TARGET_BINARY].mean() * 100:.1f}%")
m3.metric("Avg predicted budget", f"₹{clustered_df['Predicted_Budget'].mean():,.0f}")
m4.metric("Best cluster k", assets["best_k"])


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Descriptive",
    "Diagnostic",
    "Classification",
    "Clustering",
    "Association Rules",
    "Regression & Prescriptive",
    "New Customer Scoring",
])

with tab1:
    st.subheader("Descriptive analytics")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(clustered_df, x="Store_Type", color=TARGET_CLASS, barmode="group", title="Store type distribution by adoption likelihood")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(clustered_df, x="City_Type", color="Store_Type", title="City-tier mix of respondents")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        budget_dist = clustered_df["Monthly_Software_Budget"].value_counts().reset_index()
        budget_dist.columns = ["Budget", "Count"]
        fig = px.bar(budget_dist, x="Budget", y="Count", title="Monthly software budget distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.histogram(clustered_df, x="Stockout_Frequency", color="Store_Type", barmode="group", title="Stockout frequency by store type")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(clustered_df.head(20), use_container_width=True)

with tab2:
    st.subheader("Diagnostic analytics")
    st.markdown("Understand why adoption differs across retailers.")

    cross = pd.crosstab(clustered_df["Store_Type"], clustered_df[TARGET_CLASS], normalize="index").round(3)
    st.write("**Adoption likelihood share within each store type**")
    st.dataframe(cross, use_container_width=True)

    diag1, diag2 = st.columns(2)
    with diag1:
        pivot = (
            clustered_df.groupby(["AI_Comfort_Level", TARGET_CLASS]).size().reset_index(name="Count")
        )
        fig = px.bar(pivot, x="AI_Comfort_Level", y="Count", color=TARGET_CLASS, barmode="group", title="AI comfort vs adoption")
        st.plotly_chart(fig, use_container_width=True)
    with diag2:
        fig = px.box(clustered_df, x="Store_Type", y="Predicted_Budget", color=TARGET_CLASS, title="Predicted budget by store type")
        st.plotly_chart(fig, use_container_width=True)

    heat_df = pd.crosstab(clustered_df["Stockout_Frequency"], clustered_df["Adoption_Motivation"])
    fig = px.imshow(heat_df, text_auto=True, aspect="auto", title="Diagnostic heatmap: stockouts vs adoption motivation")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Predictive analytics - classification")
    render_metric_row(assets["class_metrics"])

    col_a, col_b = st.columns(2)
    with col_a:
        cm = assets["class_metrics"]["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        fig = px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion matrix")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        y_test = assets["class_metrics"]["y_test"]
        y_proba = assets["class_metrics"]["y_proba"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline"))
        roc_fig.update_layout(title="ROC curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(roc_fig, use_container_width=True)

    fig = px.bar(assets["feature_importance"], x="Importance", y="Feature", orientation="h", title="Top feature importances")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Customer segmentation - clustering")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.scatter(
            assets["pca_df"],
            x="PC1", y="PC2", color=CLUSTER_COL, symbol="Store_Type",
            title=f"PCA view of customer personas (silhouette={assets['best_silhouette']:.3f})",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        persona_summary = clustered_df.groupby(CLUSTER_COL).agg(
            Respondents=(CLUSTER_COL, "size"),
            Avg_Interest_Probability=("Adoption_Probability", "mean"),
            Avg_Predicted_Budget=("Predicted_Budget", "mean"),
        ).sort_values("Avg_Interest_Probability", ascending=False)
        st.dataframe(persona_summary, use_container_width=True)

    st.write("**Persona profiling**")
    profile = clustered_df.groupby([CLUSTER_COL, "Store_Type"]).size().reset_index(name="Count")
    fig = px.bar(profile, x=CLUSTER_COL, y="Count", color="Store_Type", title="Store-type mix within each persona")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Association rule mining")
    st.markdown("Rules ranked with **confidence** and **lift** to reveal co-occurring needs and preferences.")
    if rules_df.empty:
        st.warning("No strong rules found with the current support/confidence thresholds.")
    else:
        st.dataframe(rules_df, use_container_width=True)
        fig = px.scatter(rules_df, x="confidence", y="lift", size="support", hover_data=["antecedents", "consequents"], title="Association rules by confidence and lift")
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("Regression and prescriptive analytics")
    regm = assets["regression_metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"₹{regm['mae']:,.0f}")
    c2.metric("RMSE", f"₹{regm['rmse']:,.0f}")
    c3.metric("R²", f"{regm['r2']:.3f}")

    compare_df = pd.DataFrame({"Actual": regm["y_test"], "Predicted": regm["y_pred"]})
    fig = px.scatter(compare_df, x="Actual", y="Predicted", trendline="ols", title="Actual vs predicted software budget")
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Prescriptive targeting recommendations**")
    priority_summary = clustered_df.groupby([LEAD_SCORE_COL, "Store_Type"]).size().reset_index(name="Count")
    fig = px.bar(priority_summary, x=LEAD_SCORE_COL, y="Count", color="Store_Type", title="Lead priority by store type")
    st.plotly_chart(fig, use_container_width=True)

    top_targets = clustered_df.groupby(["Store_Type", "City_Type"]).agg(
        respondents=("Store_Type", "size"),
        avg_interest=("Adoption_Probability", "mean"),
        avg_budget=("Predicted_Budget", "mean"),
    ).reset_index()
    top_targets = top_targets.sort_values(["avg_interest", "avg_budget"], ascending=False).head(10)
    st.dataframe(top_targets, use_container_width=True)

    st.info(
        "Founder recommendation: prioritize high-probability grocery and pharmacy retailers in Tier 1/Tier 2 cities, "
        "especially those with frequent stockouts, 2–25 stores, and at least 1 year of digital sales data."
    )

with tab7:
    st.subheader("Upload new would-be customers and score them")
    st.markdown(
        "Upload a CSV/XLSX with the same 25 survey columns. The app will predict interest, budget, persona, and suggested action."
    )
    new_file = st.file_uploader("Upload new customer data", type=["csv", "xlsx"], key="new_customer_upload")

    template_df = pd.DataFrame(columns=SURVEY_COLUMNS)
    st.download_button(
        label="Download blank upload template",
        data=make_downloadable_csv(template_df),
        file_name="new_customer_template.csv",
        mime="text/csv",
    )

    if new_file is not None:
        new_df = parse_upload(new_file)
        ok, missing_cols = validate_columns(new_df, SURVEY_COLUMNS)
        if not ok:
            st.error(f"Uploaded new-customer file is missing required columns: {missing_cols}")
        else:
            scored_new = score_new_customers(add_derived_columns(new_df[SURVEY_COLUMNS].copy()), assets)
            st.success("Scoring completed.")
            st.dataframe(scored_new, use_container_width=True)
            st.download_button(
                label="Download scored new customers",
                data=make_downloadable_csv(scored_new),
                file_name="scored_new_customers.csv",
                mime="text/csv",
            )
