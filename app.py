import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler, OrdinalEncoder

import xgboost as xgb
import lightgbm as lgbm

st.title('Telco Customer Churn Prediction')

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df_churn = pd.read_csv(uploaded_file)

    # st.subheader("Uploaded Data:")
    # st.write(df_churn)

    st.header('Churn Data Overview')
    st.write('Data Dimension: ' + str(df_churn.shape[0]) + ' rows and ' + str(df_churn.shape[1]) + ' columns.')
    st.dataframe(df_churn)

    frac = st.slider('Select Train-Test Split of the data:-', 0, 99, 80)
    # st.write("I'm ", frac, 'years old')
    frac_per = frac/100
    df_train=df_churn.sample(frac=frac_per,random_state=200)
    df_test=df_churn.drop(df_train.index)

    st.write(df_train)
    st.write(df_test)

    st.sidebar.markdown('## Predict Customer Churn Rate')
    
    classifier_name = st.sidebar.selectbox(
        'Select a Classifier',
        ('XGBoost', 'CatBoost', 'LightGBM')
    )
    def get_classifier(clf_name):
        if clf_name == 'XGBoost':
            clf = xgb.XGBClassifier()  # init model
            clf.load_model("models/model_xgb.json")
        elif clf_name == 'CatBoost':
        # clf = CatBoostClassifier()  # parameters not required.
            clf.load_model('models/model_catboost')
        else:
            # clf = lgbm.LGBMClassifier()
            # clf = joblib.load("models/model_lgbm.pkl")
            clf = lgbm.Booster(model_file='models/model_lgbm.txt')
        return clf
    
    # clf = get_classifier(classifier_name)

    def get_transformed_data(test_data=None):
        X = df_train.drop('Churn', axis=1)

        if test_data is None:
            test_data = df_test.copy()
        
        X_test = test_data.drop("Churn", axis=1)
        y_test = test_data['Churn'].values

        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        cat_cols = list(set(X.columns)) -set(X._get_numeric_data().columns())

        ordinal_encoder = OrdinalEncoder()
        X[cat_cols] = ordinal_encoder.fit_transform(X[cat_cols])
        X_test[cat_cols] = ordinal_encoder.transform(X_test[cat_cols])

        transformer = RobustScaler()
        X[num_cols] = transformer.fit_tranform(X[num_cols])
        X_test[num_cols] = transformer.transform(X_test[num_cols])

        del X
        return X_test, y_test

st.sidebar.markdown("## User Input") 

def binning_feature(feature, value):
    bins = np.linspace(min(df_churn[feature]), max(df_churn[feature]), 4)
    if bins[0] <= value <= bins[1]:
        return 'Low'
    elif bins[1] < value <= bins[2]:
        return 'Medium'
    else:
        return 'High'
    
def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    senior_citizen = st.sidebar.selectbox('Senior Citizen', ('Yes', 'No'))
    partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
    phone_service = st.sidebar.selectbox('Phone Service', ('Yes', 'No', 'No phone service'))
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ('Yes', 'No'))
    internet_service_type = st.sidebar.selectbox('Internet Service Type', ('DSL', 'Fiber optic', 'No'))
    online_security = st.sidebar.selectbox('Online Security', ('Yes', 'No', 'No internet service'))
    online_backup = st.sidebar.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
    device_protection = st.sidebar.selectbox('Device Protection', ('Yes', 'No', 'No internet service'))
    tech_support = st.sidebar.selectbox('Tech Support', ('Yes', 'No', 'No internet service'))
    streaming_tv = st.sidebar.selectbox('Streaming TV', ('Yes', 'No', 'No internet service'))
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ('Yes', 'No', 'No internet service'))
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))

    payment_method = st.sidebar.selectbox('PaymentMethod', (
        'Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))

    # tenure filter
    unique_tenure_values = df_churn.tenure.unique()
    min_value, max_value = min(unique_tenure_values), max(unique_tenure_values)

    # tenure slider
    tenure = st.sidebar.slider("Tenure", int(min_value), int(max_value), int(min_value), 1)

    # MonthlyCharges filter
    unique_monthly_charges_values = df_churn.MonthlyCharges.unique()
    min_value, max_value = min(unique_monthly_charges_values), max(unique_monthly_charges_values)

    # MonthlyCharges slider
    monthly_charges = st.sidebar.slider("Monthly Charges", min_value, max_value, float(min_value))

    min_value_total = monthly_charges * tenure
    max_value_total = (monthly_charges * tenure) + 100

    st.sidebar.markdown("**`TotalCharges`** = `MonthlyCharges` * `Tenure` + `Extra Cost ( ~100 )`")

    # TotalCharges slider
    total_charges = st.sidebar.slider("Total Charges", min_value_total, max_value_total)

    # Churn filter
    data = {'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen.lower() == 'yes' else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service_type],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'tenure-binned': binning_feature('tenure', 7),
            'MonthlyCharges-binned': binning_feature('MonthlyCharges', monthly_charges),
            'TotalCharges-binned': binning_feature('TotalCharges', total_charges)
            }

    features = pd.DataFrame(data)

    return features
input_df = user_input_features()

num_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = input_df.select_dtypes(include=['object']).columns

X = df_train.drop("Churn", axis=1)
user_df = input_df.copy()