import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

def train_model():
    # Veriyi yükleme
    train = pd.read_csv("./data/fraudTrain.csv")
    test = pd.read_csv("./data/fraudTest.csv")

    # Veri ön işleme
    train['dob'] = pd.to_datetime(train['dob'])
    test['dob'] = pd.to_datetime(test['dob'])

    train['age'] = (pd.to_datetime('today') - train['dob']).dt.days // 365
    test['age'] = (pd.to_datetime('today') - test['dob']).dt.days // 365
    train = train.drop(columns=['dob'])
    test = test.drop(columns=['dob'])

    train['trans_date_trans_time'] = pd.to_datetime(train['trans_date_trans_time'])
    test['trans_date_trans_time'] = pd.to_datetime(test['trans_date_trans_time'])

    train['hour'] = train['trans_date_trans_time'].dt.hour
    train['day_of_week'] = train['trans_date_trans_time'].dt.dayofweek
    train['month'] = train['trans_date_trans_time'].dt.month
    train['year'] = train['trans_date_trans_time'].dt.year
    train['day'] = train['trans_date_trans_time'].dt.day

    test['hour'] = test['trans_date_trans_time'].dt.hour
    test['day_of_week'] = test['trans_date_trans_time'].dt.dayofweek
    test['month'] = test['trans_date_trans_time'].dt.month
    test['year'] = test['trans_date_trans_time'].dt.year
    test['day'] = test['trans_date_trans_time'].dt.day

    # Özellikleri belirleme
    categorical_features = ['cc_num', 'merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 'job']
    numeric_features = ['amt', 'city_pop', 'lat', 'long', 'unix_time', 'merch_lat', 'merch_long', 'hour', 'day_of_week', 'month', 'year', 'day', 'age']

    y_train = train['is_fraud']
    X_train = train.drop(columns=['is_fraud', 'trans_date_trans_time', 'trans_num'])
    y_test = test['is_fraud']
    X_test = test.drop(columns=['is_fraud', 'trans_date_trans_time', 'trans_num'])

    


    # Modeli eğitme
    catboost_model = CatBoostClassifier(iterations=1, depth=10, learning_rate=0.05, random_seed=42, cat_features=categorical_features)
    catboost_model.fit(X_train, y_train)

    # Performans ölçümleri
    train_metrics = classification_report(y_train, catboost_model.predict(X_train), output_dict=True)
    test_roc_auc = roc_auc_score(y_test, catboost_model.predict_proba(X_test)[:, 1])

    return catboost_model, X_train, X_test, y_train, y_test, train_metrics, test_roc_auc

