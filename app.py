import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from flask import Flask, render_template
from sklearn.metrics import roc_curve

from model_training import train_model

app = Flask(__name__)

# Veri yükleme
train_data = pd.read_csv('data/fraudTrain.csv')
test_data = pd.read_csv('data/fraudTest.csv')

@app.template_filter('zip')
def zip_filter(*args):
    return zip(*args)

# Zip fonksiyonunu global olarak kaydetme
app.jinja_env.globals['zip'] = zip


# Veri özetleme fonksiyonları
def get_data_summary():
    summary = {}

    # Eğitim verisi bilgisi
    train_info = {
        "kolonlar": train_data.columns.tolist(),
        "veri_tipleri": train_data.dtypes.astype(str).tolist(),
        "dolu_hücre_sayısı": train_data.count().tolist()
    }
    summary['egitim_verisi_bilgisi'] = train_info

    # Test verisi bilgisi
    test_info = {
        "kolonlar": test_data.columns.tolist(),
        "veri_tipleri": test_data.dtypes.astype(str).tolist(),
        "dolu_hücre_sayısı": test_data.count().tolist()
    }
    summary['test_verisi_bilgisi'] = test_info

    # Eğitim verisi başlıkları
    summary['egitim_verisi_basliki'] = train_data.head().to_dict(orient='records')

    # Eksik değerler
    summary['eksik_degerler_egitim'] = train_data.isnull().sum().to_dict()
    summary['eksik_degerler_test'] = test_data.isnull().sum().to_dict()

    # Eğitim verisi tanımı
    summary['egitim_verisi_tanimi'] = train_data.describe().to_dict()

    return summary

# Görselleştirme oluşturma
def generate_visualizations():
    visualizations = []

    # Hedef değişkenin dağılımı
    fig1 = px.histogram(train_data, x="is_fraud", title="Sahte İşlemlerin Dağılımı", labels={"is_fraud": "Sahte mi?"})
    visualizations.append(fig1.to_html(full_html=False))

    # İşlem tutarlarının sahteciliğe göre dağılımı
    fig2 = px.box(train_data, x="is_fraud", y="amt", title="İşlem Tutarı ve Sahtecilik İlişkisi", labels={"is_fraud": "Sahte mi?", "amt": "İşlem Tutarı"})
    visualizations.append(fig2.to_html(full_html=False))

    # Cinsiyete göre dağılım
    fig3 = px.histogram(train_data, x="gender", color="is_fraud", barmode="group", title="Cinsiyete Göre Sahtecilik Dağılımı", labels={"gender": "Cinsiyet", "is_fraud": "Sahte mi?"})
    visualizations.append(fig3.to_html(full_html=False))

    # İşlem kategorisine göre dağılım
    fig4 = px.histogram(train_data, x="category", color="is_fraud", barmode="group", title="Kategoriye Göre Sahtecilik Dağılımı", labels={"category": "Kategori", "is_fraud": "Sahte mi?"})
    visualizations.append(fig4.to_html(full_html=False))

    # Sahte işlemlerin saatlik dağılımı
    train_data['trans_hour'] = pd.to_datetime(train_data['trans_date_trans_time']).dt.hour
    fig5 = px.histogram(train_data, x="trans_hour", color="is_fraud", barmode="group", title="Sahte İşlemlerin Saatlik Dağılımı", labels={"trans_hour": "Saat", "is_fraud": "Sahte mi?"})
    visualizations.append(fig5.to_html(full_html=False))

    # Gün bazında sahte işlemlerin dağılımı
    train_data['trans_day'] = pd.to_datetime(train_data['trans_date_trans_time']).dt.dayofweek
    fig6 = px.histogram(train_data, x="trans_day", color="is_fraud", barmode="group", title="Gün Bazında Sahte İşlemlerin Dağılımı", labels={"trans_day": "Gün", "is_fraud": "Sahte mi?"})
    visualizations.append(fig6.to_html(full_html=False))

    # Coğrafi verilerin dağılımı
    fig7 = px.scatter(train_data, x="long", y="lat", color="is_fraud", title="Coğrafi Dağılım ve Sahtecilik", labels={"long": "Boylam", "lat": "Enlem", "is_fraud": "Sahte mi?"})
    visualizations.append(fig7.to_html(full_html=False))

    return visualizations





def generate_additional_visualizations():
    additional_visualizations = []
    # İşlem tarihi ve saati sütununu datetime formatına dönüştürün
    if 'trans_date_trans_time' in train_data.columns:
        train_data['trans_date_trans_time'] = pd.to_datetime(train_data['trans_date_trans_time'])
        # 'trans_date' sütununu oluşturun
        train_data['trans_date'] = train_data['trans_date_trans_time'].dt.date
    else:
        raise KeyError("'trans_date_trans_time' sütunu bulunamadı.")

    # İşlem sıklığı analizi
    transaction_counts = train_data.groupby(['trans_date', 'is_fraud']).size().unstack()
    fig1 = px.line(transaction_counts, title="Zaman İçinde İşlem Sıklığı", labels={"value": "İşlem Sayısı", "trans_date": "Tarih"})
    additional_visualizations.append(fig1.to_html(full_html=False))
    
    # Diğer grafikler aynı sırayla eklenir...

    return additional_visualizations

@app.route('/')
def index():
    data_summary = get_data_summary()
    visualizations = generate_visualizations()
    additional_visualizations = generate_additional_visualizations()
    all_visualizations = visualizations + additional_visualizations

    # Görsel isimlerini tanımla
    visualization_names = [
        "Sahte İşlemlerin Dağılımı",
        "İşlem Tutarı ve Sahtecilik İlişkisi",
        "Cinsiyete Göre Sahtecilik Dağılımı",
        "Kategoriye Göre Sahtecilik Dağılımı",
        "Sahte İşlemlerin Saatlik Dağılımı",
        "Gün Bazında Sahte İşlemlerin Dağılımı",
        "Coğrafi Dağılım ve Sahtecilik",
        "Zaman İçinde İşlem Sıklığı",
        "Zaman İçinde Sahte İşlem Sıklığı",
        "Son İşlemden Bu Yana Geçen Süre ve Sahtecilik İlişkisi",
        "Kategoriye Göre İşlem Tutarı",
        "Yaş ve Cinsiyet Dağılımı",
        "Şehir Nüfusu ve İşlem Tutarı Karşılaştırması",
        "Kart Numarasına Göre İşlem Sıklığı",
        "Sahtecilik ve İşlem Zamanı Dağılımı"
    ]

    return render_template('index.html', 
                           data_summary=data_summary, 
                           visualization_names=visualization_names,
                           enumerate=enumerate)


# Modeli eğitme ve verileri hazırlama
catboost_model, X_train, X_test, y_train, y_test, train_metrics, test_roc_auc = train_model()

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

# Eğitim metriklerini formatlama
def format_metrics(metrics):
    return json.dumps(metrics, indent=4)

# Metrikleri tabloya dönüştürme
def metrics_to_table(metrics):
    table_rows = ""
    for label, stats in metrics.items():
        if isinstance(stats, dict):
            row = f"<tr><td>{label}</td><td>{stats['precision']:.2f}</td><td>{stats['recall']:.2f}</td><td>{stats['f1-score']:.2f}</td><td>{stats['support']}</td></tr>"
            table_rows += row
    return table_rows

@app.route('/model-info')
def model_info():
    # Görselleri ve metrikleri oluştur
    roc_html = create_roc_curve(y_test, catboost_model.predict_proba(X_test)[:, 1])
    feature_html = create_feature_importances(catboost_model, X_train.columns)
    train_metrics_json = format_metrics(train_metrics)
    train_metrics_table = metrics_to_table(train_metrics)

    return render_template(
        'modelInfo.html',
        test_roc_auc=test_roc_auc,
        roc_html=roc_html,
        feature_html=feature_html,
        train_metrics_json=train_metrics_json,
        train_metrics_table=train_metrics_table
    )


@app.route('/visualization/<int:vis_id>')
def serve_visualization(vis_id):
    visualizations = generate_visualizations() + generate_additional_visualizations()
    if 0 <= vis_id < len(visualizations):
        return visualizations[vis_id]
    return "Visualization not found", 404

if __name__ == '__main__':
    app.run(debug=True)
