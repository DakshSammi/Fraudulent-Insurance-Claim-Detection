import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Function to load car insurance dataset and train model
def train_car_insurance_model():
    df = pd.read_excel('insurance_dataset.xlsx')
    sns.countplot(data=df, x='fraud_reported')
    plt.title('Class Distribution')
    df = df.drop(columns=['policy_number', 'policy_bind_date', 'incident_date', 'incident_location', 'auto_model'])
    df.fillna('MISSING', inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    target_variable = 'fraud_reported_Y'
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    seed = 109
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=seed)
    clf = DecisionTreeClassifier(max_depth=4, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    return clf

# Function to load health insurance dataset and train model
def train_health_insurance_model():
    df = pd.read_csv('health_insurance_data.csv')
    # df.columns
    # '''
    # Index(['TXN_DATE_TIME', 'TRANSACTION_ID', 'CUSTOMER_ID', 'POLICY_NUMBER',
    #     'POLICY_EFF_DT', 'LOSS_DT', 'REPORT_DT', 'INSURANCE_TYPE',
    #     'PREMIUM_AMOUNT', 'CLAIM_AMOUNT', 'CUSTOMER_NAME', 'ADDRESS_LINE1',
    #     'ADDRESS_LINE2', 'CITY', 'STATE', 'POSTAL_CODE', 'SSN',
    #     'MARITAL_STATUS', 'AGE', 'TENURE', 'EMPLOYMENT_STATUS',
    #     'NO_OF_FAMILY_MEMBERS', 'RISK_SEGMENTATION', 'HOUSE_TYPE',
    #     'SOCIAL_CLASS', 'ROUTING_NUMBER', 'ACCT_NUMBER',
    #     'CUSTOMER_EDUCATION_LEVEL', 'CLAIM_STATUS', 'INCIDENT_SEVERITY',
    #     'AUTHORITY_CONTACTED', 'ANY_INJURY', 'POLICE_REPORT_AVAILABLE',
    #     'INCIDENT_STATE', 'INCIDENT_CITY', 'INCIDENT_HOUR_OF_THE_DAY',
    #     'AGENT_ID', 'VENDOR_ID'],
    #     dtype='object')'''
    # sns.countplot(data=df, x= 'CLAIM_STATUS')
    # plt.title('Class Distribution')
    df.drop(['TXN_DATE_TIME', 'TRANSACTION_ID', 'CUSTOMER_ID', 'POLICY_NUMBER',
             'POLICY_EFF_DT', 'LOSS_DT', 'REPORT_DT', 'CUSTOMER_NAME', 'ADDRESS_LINE1',
             'STATE', 'POSTAL_CODE', 'SSN', 'ROUTING_NUMBER', 'ACCT_NUMBER', 'AGENT_ID'], axis=1, inplace=True)
    
    unimportant_columns = ['ADDRESS_LINE2', 'CITY', 'INCIDENT_CITY', 'VENDOR_ID']
    for col in unimportant_columns:
        df[col] = df[col].notnull()
    df.fillna('MISSING', inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    df.rename(columns={'CLAIM_STATUS_D': 'FRAUDULENT'}, inplace=True)
    target_variable = 'FRAUDULENT'
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    seed = 200
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
    clf = DecisionTreeClassifier(max_depth=6, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    return clf

# Sidebar menu to select insurance type
selected_insurance = st.sidebar.selectbox("Select Insurance Type", ["Car Insurance", "Health Insurance"])

# Page for Car Insurance Claim Detection
if selected_insurance == "Car Insurance":
    st.title("Car Insurance Claim Detection")

    # Load car insurance model
    car_insurance_model = train_car_insurance_model()

    # Sidebar with user inputs for car insurance

    # You can add more input fields here based on your dataset columns
    st.sidebar.header("Admin Panel - Enter Car Insurance Claim Data")
    # Add input fields for car insurance data
    months_as_customer = st.sidebar.number_input("Months as Customer", min_value=0, max_value=500, value=1)
    age = st.sidebar.number_input("Age", min_value=1, max_value=200, value=18)
    policy_number = st.sidebar.number_input("Policy Number", min_value=0, max_value=1000000, value=1)
    policy_bind_date = st.sidebar.date_input("Policy Bind Date")
    policy_state = st.sidebar.selectbox("Policy State", ["OH", "IN", "IL"])
    policy_csl = st.sidebar.selectbox("Policy CSL", ["250/500", "100/300", "500/1000"])
    policy_deductable = st.sidebar.number_input("Policy Deductable", min_value=0, max_value=10000, value=1)
    policy_annual_premium = st.sidebar.number_input("Policy Annual Premium", min_value=0, max_value=10000, value=1)
    umbrella_limit = st.sidebar.number_input("Umbrella Limit", min_value=0, max_value=1000000, value=1)
    insured_zip = st.sidebar.number_input("Insured Zip", min_value=0, max_value=1000000, value=1)
    insured_sex = st.sidebar.radio("Insured Gender", ["Male", "Female"])
    insured_education_level = st.sidebar.selectbox("Insured Education Level", ["High School", "College", "Associate", "Bachelors", "Masters", "PhD"])
    insured_occuption = st.sidebar.selectbox("Insured Occupation", ["Lawyer", "Clerical", "Craft-repair", "Other-service", "Sales", "Armed-Forces", "Tech-support", "Prof-specialty", "Exec-managerial", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Handlers-cleaners", "Protective-serv", "Priv-house-serv"])
    insured_hobbies = st.sidebar.selectbox("Insured Hobbies", ["chess", "cross-fit", "board-games", "base-jumping", "skydiving", "dancing", "sleeping", "reading", "paintball", "movies", "yachting", "camping", "polo", "hiking", "kayaking", "golf", "video-games", "basketball", "exercise", "cycling", "hockey", "paintball", "movies", "yachting", "camping", "polo", "hiking", "kayaking", "golf", "video-games", "basketball", "exercise", "cycling", "hockey"])
    insured_relationship = st.sidebar.selectbox("Insured Relationship", ["husband", "other-relative", "own-child", "unmarried", "wife", "not-in-family"])
    capitalgains = st.sidebar.number_input("Capital Gains", min_value=0, max_value=1000000, value=1)
    capitalloss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=1000000, value=1)
    incident_date = st.sidebar.date_input("Incident Date")
    incident_type = st.sidebar.selectbox("Incident Type", ["Single Vehicle Collision", "Vehicle Theft", "Multi-vehicle Collision", "Parked Car"])   
    collision_type = st.sidebar.selectbox("Collision Type", ["Side Collision", "Rear Collision", "Front Collision"])
    incident_city = st.sidebar.selectbox("Incident City", ["Columbus", "Riverwood", "Northbrook", "Hillsdale", "Springfield", "Northbend", "Northbrook", "Hillsdale", "Riverwood", "Northbend", "Springfield", "Columbus"])
    incident_state = st.sidebar.selectbox("Incident State", ["OH", "IN", "IL", "WV", "NC", "MO", "VA", "PA", "NY", "KY"])
    incident_location = st.sidebar.selectbox("Incident Location", ["Intersection", "Highway", "Other", "Residential"])
    incident_hour = st.sidebar.slider("Incident Hour", 0, 23, 12)
    number_of_vehicles_involved = st.sidebar.slider("Number of Vehicles Involved", 1, 4, 2)
    property_damage = st.sidebar.radio("Property Damage", ["YES", "NO"])
    bodily_injuries = st.sidebar.slider("Bodily Injuries", 0, 2, 1)
    witnesses = st.sidebar.slider("Witnesses", 0, 3, 1)
    police_report_available = st.sidebar.radio("Police Report Available", ["YES", "NO"])
    total_claim_amount = st.sidebar.number_input("Total Claim Amount", min_value=0, max_value=100000, value=1)
    injury_claim = st.sidebar.number_input("Injury Claim", min_value=0, max_value=100000, value=1)
    property_claim = st.sidebar.number_input("Property Claim", min_value=0, max_value=100000, value=1)
    vehicle_claim = st.sidebar.number_input("Vehicle Claim", min_value=0, max_value=100000, value=1)
    property_damage = st.sidebar.radio("Property Damage", ["YES", "NO"], key='property_damage')
    auto_make = st.sidebar.selectbox("Auto Make", ["Saab", "Mercedes", "Dodge", "Chevrolet", "Accura", "Nissan", "Audi", "Toyota", "Ford", "Suburu", "BMW", "Jeep", "Volkswagen", "Honda"])
    auto_model = st.sidebar.selectbox("Auto Model", ["92x", "E400", "RAM", "Tahoe", "RSX", "95", "A3", "Camry", "F150", "Forrester", "M5", "Wrangler", "Passat", "CRV"])
    auto_year = st.sidebar.number_input("Auto Year", min_value=1990, max_value=2020, value=1990)
    authoritities_contacted = st.sidebar.selectbox("Authorities Contacted", ["Police", "None", "Fire", "Other", "Ambulance"])
    
    
    # Button to trigger prediction
    if st.sidebar.button("Submit"):
        # Make prediction
        # use actual input values
        # retrieve input values from the sidebar and pass them to the model
        input_data = pd.DataFrame({
            'months_as_customer': [months_as_customer],
            'age': [age],
            'policy_number': [policy_number],
            'policy_bind_date': [policy_bind_date],
            'policy_state': [policy_state],
            'policy_csl': [policy_csl],
            'policy_deductable': [policy_deductable],
            'policy_annual_premium': [policy_annual_premium],
            'umbrella_limit': [umbrella_limit],
            'insured_zip': [insured_zip],
            'insured_sex': [insured_sex],
            'insured_education_level': [insured_education_level],
            'insured_occuption': [insured_occuption],
            'insured_hobbies': [insured_hobbies],
            'insured_relationship': [insured_relationship],
            'capital-gains': [capitalgains],
            'capital-loss': [capitalloss],
            'incident_date_2024-03-16': [0 if incident_date != '2024-03-16' else 1],
            'incident_date': [incident_date],

            'incident_type': [incident_type],
            'collision_type': [collision_type],
            'incident_city': [incident_city],
            'incident_state': [incident_state],
            'incident_location_Intersection': [0 if incident_location != 'Intersection' else 1],
            'incident_location_Highway': [0 if incident_location != 'Highway' else 1],
            'incident_location_Other': [0 if incident_location != 'Other' else 1],
            'incident_location_Residential': [0 if incident_location != 'Residential' else 1],
            'insured_occuptaion_lawyer': [0 if insured_occuption != 'lawyer' else 1],
            'insured_occuptaion_clerical': [0 if insured_occuption != 'clerical' else 1],
            'insured_occuptaion_craft-repair': [0 if insured_occuption != 'craft-repair' else 1],
            'insured_occuptaion_other-service': [0 if insured_occuption != 'other-service' else 1],
            'insured_occuptaion_sales': [0 if insured_occuption != 'sales' else 1],
            'insured_occuptaion_armed-forces': [0 if insured_occuption != 'armed-forces' else 1],
            'insured_occuptaion_tech-support': [0 if insured_occuption != 'tech-support' else 1],
            'insured_occuptaion_prof-specialty': [0 if insured_occuption != 'prof-specialty' else 1],
            'insured_occuptaion_exec-managerial': [0 if insured_occuption != 'exec-managerial' else 1],
            'insured_occuptaion_machine-op-inspct': [0 if insured_occuption != 'machine-op-inspct' else 1],
            'insured_occuptaion_adm-clerical': [0 if insured_occuption != 'adm-clerical' else 1],
            'insured_occuptaion_farming-fishing': [0 if insured_occuption != 'farming-fishing' else 1],
            'insured_occuptaion_transport-moving': [0 if insured_occuption != 'transport-moving' else 1],
            'insured_occuptaion_handlers-cleaners': [0 if insured_occuption != 'handlers-cleaners' else 1],
            'insured_occuptaion_protective-serv': [0 if insured_occuption != 'protective-serv' else 1],
            'insured_occuptaion_priv-house-serv': [0 if insured_occuption != 'priv-house-serv' else 1],
            'insured_hobbies_chess': [0 if insured_hobbies != 'chess' else 1],
            'insured_hobbies_cross-fit': [0 if insured_hobbies != 'cross-fit' else 1],
            'insured_hobbies_board-games': [0 if insured_hobbies != 'board-games' else 1],  
            'insured_hobbies_base-jumping': [0 if insured_hobbies != 'base-jumping' else 1],
            'insured_hobbies_skydiving': [0 if insured_hobbies != 'skydiving' else 1],
            'insured_hobbies_dancing': [0 if insured_hobbies != 'dancing' else 1],
            'insured_hobbies_sleeping': [0 if insured_hobbies != 'sleeping' else 1],
            'insured_hobbies_reading': [0 if insured_hobbies != 'reading' else 1],
            'insured_hobbies_paintball': [0 if insured_hobbies != 'paintball' else 1],
            'insured_hobbies_movies': [0 if insured_hobbies != 'movies' else 1],
            'insured_hobbies_yachting': [0 if insured_hobbies != 'yachting' else 1],
            'insured_hobbies_camping': [0 if insured_hobbies != 'camping' else 1],
            'insured_hobbies_polo': [0 if insured_hobbies != 'polo' else 1],
            'insured_hobbies_hiking': [0 if insured_hobbies != 'hiking' else 1],
            'insured_hobbies_kayaking': [0 if insured_hobbies != 'kayaking' else 1],
            'insured_hobbies_golf': [0 if insured_hobbies != 'golf' else 1],
            'insured_hobbies_video-games': [0 if insured_hobbies != 'video-games' else 1],
            'insured_hobbies_basketball': [0 if insured_hobbies != 'basketball' else 1],
            'insured_hobbies_exercise': [0 if insured_hobbies != 'exercise' else 1],
            'insured_hobbies_cycling': [0 if insured_hobbies != 'cycling' else 1],
            'insured_hobbies_hockey': [0 if insured_hobbies != 'hockey' else 1],
            'insured_relationship_husband': [0 if insured_relationship != 'husband' else 1],
            'insured_relationship_other-relative': [0 if insured_relationship != 'other-relative' else 1],
            'insured_relationship_own-child': [0 if insured_relationship != 'own-child' else 1],
            'insured_relationship_unmarried': [0 if insured_relationship != 'unmarried' else 1],
            'insured_relationship_wife': [0 if insured_relationship != 'wife' else 1],
            'insured_relationship_not-in-family': [0 if insured_relationship != 'not-in-family' else 1],
            'incident_type_Single Vehicle Collision': [0 if incident_type != 'Single Vehicle Collision' else 1],
            'incident_type_Vehicle Theft': [0 if incident_type != 'Vehicle Theft' else 1],
            'incident_type_Multi-vehicle Collision': [0 if incident_type != 'Multi-vehicle Collision' else 1],
            'incident_type_Parked Car': [0 if incident_type != 'Parked Car' else 1],
            'collision_type_Side Collision': [0 if collision_type != 'Side Collision' else 1],
            'collision_type_Rear Collision': [0 if collision_type != 'Rear Collision' else 1],
            'collision_type_Front Collision': [0 if collision_type != 'Front Collision' else 1],
            'incident_city_Columbus': [0 if incident_city != 'Columbus' else 1],
            'incident_city_Riverwood': [0 if incident_city != 'Riverwood' else 1],
            'incident_city_Northbrook': [0 if incident_city != 'Northbrook' else 1],

            'incident_hour_of_the_day': [incident_hour],
            'number_of_vehicles_involved': [number_of_vehicles_involved],
            'bodily_injuries': [bodily_injuries],
            'witnesses': [witnesses],
            'total_claim_amount': [total_claim_amount],
            'injury_claim': [injury_claim],
            'property_claim': [property_claim],
            'auto_make': [auto_make],
            'auto_year': [auto_year],
            'vehicle_claim': [vehicle_claim],
            'policy_state': [policy_state],
            'auto_model_92x': [0 if auto_model != '92x' else 1],
            'auto_model_E400': [0 if auto_model != 'E400' else 1],
            'auto_model_RAM': [0 if auto_model != 'RAM' else 1],
            'auto_model_Tahoe': [0 if auto_model != 'Tahoe' else 1],
            'auto_model_RSX': [0 if auto_model != 'RSX' else 1],
            'auto_model_95': [0 if auto_model != '95' else 1],
            'auto_model_A3': [0 if auto_model != 'A3' else 1],
            'auto_model_Camry': [0 if auto_model != 'Camry' else 1],
            'auto_model_F150': [0 if auto_model != 'F150' else 1],
            'auto_model_Forrester': [0 if auto_model != 'Forrester' else 1],
            'auto_model_M5': [0 if auto_model != 'M5' else 1],
            'auto_model_Wrangler': [0 if auto_model != 'Wrangler' else 1],
            'auto_model_Passat': [0 if auto_model != 'Passat' else 1],
            'auto_model_CRV': [0 if auto_model != 'CRV' else 1],
            "auto_model_Audi": [0 if auto_model != 'Audi' else 1],
            "auto_model_Toyota": [0 if auto_model != 'Toyota' else 1],
            "auto_model_Ford": [0 if auto_model != 'Ford' else 1],
            "auto_model_Suburu": [0 if auto_model != 'Suburu' else 1],
            "auto_model_BMW": [0 if auto_model != 'BMW' else 1],
            "auto_model_Jeep": [0 if auto_model != 'Jeep' else 1],
            "auto_model_Volkswagen": [0 if auto_model != 'Volkswagen' else 1],
            "auto_model_Honda": [0 if auto_model != 'Honda' else 1],
        
            'authorities_contacted_None': [0 if authoritities_contacted != 'None' else 1],
            'authorities_contacted_Fire': [0 if authoritities_contacted != 'Fire' else 1],
            'authorities_contacted_Other': [0 if authoritities_contacted != 'Other' else 1],
            'authorities_contacted_Ambulance': [0 if authoritities_contacted != 'Ambulance' else 1],
            'authorities_contacted_Police': [0 if authoritities_contacted != 'Police' else 1],
            'property_damage_YES': [0 if property_damage != 'YES' else 1],
            'property_damage_NO': [0 if property_damage != 'NO' else 1],
            'police_report_available_YES': [0 if police_report_available != 'YES' else 1],
            'police_report_available_NO': [0 if police_report_available != 'NO' else 1]

        })
        # here the code for prediction
        # input_data = pd.get_dummies(input_data)
        # prediction = car_insurance_model.predict(input_data)
        # if prediction == 1:
        #     st.error("Fraudulent claim detected!")
        # else:
        #     st.success("Legitimate claim detected.")
        dummy_prediction = car_insurance_model.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])[0]
        if dummy_prediction == 1:
            st.error("Fraudulent claim detected!")
        else:
            st.success("Legitimate claim detected.")

# Page for Health Insurance Claim Detection
elif selected_insurance == "Health Insurance":
    st.title("Health Insurance Claim Detection")

    # Load health insurance model
    health_insurance_model = train_health_insurance_model()

    # Sidebar with user inputs for health insurance
    # You can add more input fields here based on your dataset columns
    st.sidebar.header("Admin Panel - Enter Health Insurance Claim Data")
    # Add input fields for health insurance data
    txn_date_time = st.sidebar.date_input("Transaction Date Time")
    transaction_id = st.sidebar.number_input("Transaction ID", min_value=0, max_value=1000000, value=1)
    customer_id = st.sidebar.number_input("Customer ID", min_value=0, max_value=1000000, value=1)
    policy_number = st.sidebar.number_input("Policy Number", min_value=0, max_value=1000000, value=1)
    policy_eff_dt = st.sidebar.date_input("Policy Effective Date")
    loss_dt = st.sidebar.date_input("Loss Date")
    report_dt = st.sidebar.date_input("Report Date")
    insurance_type = st.sidebar.selectbox("Insurance Type", ["Personal Auto", "Commercial Auto", "Health", "Other"])
    premium_amount = st.sidebar.number_input("Premium Amount", min_value=0, max_value=1000000, value=1)
    claim_amount = st.sidebar.number_input("Claim Amount", min_value=0, max_value=1000000, value=1)
    customer_name = st.sidebar.text_input("Customer Name")
    address_line1 = st.sidebar.text_input("Address Line 1")
    address_line2 = st.sidebar.text_input("Address Line 2")
    city = st.sidebar.text_input("City")
    state = st.sidebar.text_input("State")
    postal_code = st.sidebar.text_input("Postal Code")
    ssn = st.sidebar.text_input("SSN")
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=1)
    tenure = st.sidebar.number_input("Tenure", min_value=0, max_value=100, value=1)
    employment_status = st.sidebar.selectbox("Employment Status", ["Employed", "Unemployed", "Retired", "Student"])
    no_of_family_members = st.sidebar.number_input("Number of Family Members", min_value=0, max_value=100, value=1)
    risk_segmentation = st.sidebar.selectbox("Risk Segmentation", ["Low", "Medium", "High"])
    house_type = st.sidebar.selectbox("House Type", ["Own", "Rent", "Mortgage"])
    social_class = st.sidebar.selectbox("Social Class", ["Lower", "Middle", "Upper"])
    routing_number = st.sidebar.text_input("Routing Number")
    acct_number = st.sidebar.text_input("Account Number")
    customer_education_level = st.sidebar.selectbox("Customer Education Level", ["High School", "College", "Associate", "Bachelors", "Masters", "PhD"])
    incident_severity = st.sidebar.selectbox("Incident Severity", ["Trivial", "Minor", "Major", "Total Loss"])
    authority_contacted = st.sidebar.selectbox("Authority Contacted", ["Police", "None", "Fire", "Other", "Ambulance"])
    any_injury = st.sidebar.radio("Any Injury", ["YES", "NO"])
    police_report_available = st.sidebar.radio("Police Report Available", ["YES", "NO"])
    incident_state = st.sidebar.text_input("Incident State")
    incident_city = st.sidebar.text_input("Incident City")
    incident_hour_of_the_day = st.sidebar.slider("Incident Hour of the Day", 0, 23, 12)
    agent_id = st.sidebar.number_input("Agent ID", min_value=0, max_value=1000000, value=1)
    vendor_id = st.sidebar.number_input("Vendor ID", min_value=0, max_value=1000000, value=1)

    # Button to trigger prediction
    if st.sidebar.button("Submit"):
        # Make prediction
        # You need to modify this part to use actual input values
        input_data_health = pd.DataFrame({
            'txn_date_time': [txn_date_time],
            'transaction_id': [transaction_id],
            'customer_id': [customer_id],
            'policy_number': [policy_number],
            'policy_eff_dt': [policy_eff_dt],
            'loss_dt': [loss_dt],
            'report_dt': [report_dt],
            'insurance_type': [insurance_type],
            'premium_amount': [premium_amount],
            'claim_amount': [claim_amount],
            'customer_name': [customer_name],
            'address_line1': [address_line1],
            'address_line2': [address_line2],
            'city': [city],
            'state': [state],
            'postal_code': [postal_code],
            'ssn': [ssn],
            'marital_status': [marital_status],
            'age': [age],
            'tenure': [tenure],
            'employment_status': [employment_status],
            'no_of_family_members': [no_of_family_members],
            'risk_segmentation': [risk_segmentation],
            'house_type': [house_type],
            'social_class': [social_class],
            'routing_number': [routing_number],
            'acct_number': [acct_number],
            'customer_education_level': [customer_education_level],
            'incident_severity': [incident_severity],
            'authority_contacted': [authority_contacted],
            'any_injury': [any_injury],
            'police_report_available': [police_report_available],
            'incident_state': [incident_state],
            'incident_city': [incident_city],
            'incident_hour_of_the_day': [incident_hour_of_the_day],
            'agent_id': [agent_id],
            'vendor_id': [vendor_id]
        })
        # For example, you can retrieve input values from the sidebar and pass them to the model
        # For demonstration, here's a dummy prediction:
        dummy_prediction = health_insurance_model.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,0, 0, 0,]])[0]
        if dummy_prediction == 0:
            st.error("Fraudulent claim detected!")
        else:
            st.success("Legitimate claim detected.")
