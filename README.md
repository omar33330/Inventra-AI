# Smart Inventory Prediction - Streamlit App

This repository contains a single-file Streamlit app for:
- descriptive, diagnostic, predictive, and prescriptive analysis
- classification using accuracy, precision, recall, F1-score, ROC curve, and feature importance
- clustering-based customer persona segmentation
- association rule mining using support, confidence, and lift
- regression for monthly software budget prediction
- scoring newly uploaded would-be customers for marketing prioritization

## Files
- `app.py` - main Streamlit application
- `requirements.txt` - Python dependencies for Streamlit Community Cloud
- `synthetic_retail_customer_survey.csv` - sample survey dataset
- `new_customer_template.csv` - template for future lead uploads

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload all files in this folder to the root of a public GitHub repository.
2. Log in to Streamlit Community Cloud.
3. Create a new app and set `app.py` as the entrypoint.
4. Keep `requirements.txt` in the repository root.

## Upload format for new customers
The upload file must contain the same 25 survey columns used in the sample CSV.
