import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import plotly.graph_objects as go
import plotly.express as px

# Veri yükleme
def load_data():
    train_data = pd.read_csv('data/fraudTrain.csv')
    test_data = pd.read_csv('data/fraudTest.csv')
    return train_data, test_data

# Veri ön işleme
def preprocess_data(data):
    # Tarih sütununu dönüştürme (örneğin 'trans_date_trans_time' gibi bir sütun varsa)
    if 'trans_date_trans_time' in data.columns:
        data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
        data['transaction_hour'] = data['trans_date_trans_time'].dt.hour
        data['transaction_day'] = data['trans_date_trans_time'].dt.day
        data['transaction_month'] = data['trans_date_trans_time'].dt.month
        data = data.drop(['trans_date_trans_time'], axis=1)

    # Kategorik sütunları sayısal değerlere dönüştürme
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category').cat.codes

    return data

# ROC Curve oluşturma
def create_roc_curve(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
    fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    return fig.to_html(full_html=False)

# Feature Importances oluşturma
def create_feature_importances(model, feature_names):
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importances')
    return fig.to_html(full_html=False)

# Eğitim metriklerini tabloya dönüştürme
def metrics_to_table(metrics):
    table_rows = ""
    for label, stats in metrics.items():
        if isinstance(stats, dict):
            row = f"<tr><td>{label}</td><td>{stats['precision']:.2f}</td><td>{stats['recall']:.2f}</td><td>{stats['f1-score']:.2f}</td><td>{stats['support']}</td></tr>"
            table_rows += row
    return table_rows

# Modeli eğit
def train_model(iterations, depth, learning_rate, random_seed):
    train_data, test_data = load_data()

    # Veri ön işleme
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    # Özellik ve hedef değişkenleri ayırma
    X_train = train_data.drop('is_fraud', axis=1)
    y_train = train_data['is_fraud']
    X_test = test_data.drop('is_fraud', axis=1)
    y_test = test_data['is_fraud']

    # Model oluşturma
    model = RandomForestClassifier(
        n_estimators=iterations,
        max_depth=depth,
        random_state=random_seed
    )
    model.fit(X_train, y_train)

    # Tahminler ve metrikler
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    train_metrics = classification_report(y_test, y_pred, output_dict=True)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Görselleri oluşturma
    roc_html = create_roc_curve(y_test, y_pred_proba)
    feature_html = create_feature_importances(model, X_train.columns)
    train_metrics_table = metrics_to_table(train_metrics)

    # Model bilgileri
    model_id = f"model_{iterations}_{depth}_{learning_rate}_{random_seed}"
    model_info = {
        'model': model,
        'train_metrics_table': train_metrics_table,
        'test_roc_auc': test_roc_auc,
        'roc_html': roc_html,
        'feature_html': feature_html,
    }

    return model_id, model_info
