import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

from sklearn.metrics import roc_curve
# Veri yükleme
train_data = pd.read_csv('data/fraudTrain.csv')
test_data = pd.read_csv('data/fraudTest.csv')


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
    
        # Zaman içinde sahte işlem sıklığı
    fraud_counts = train_data[train_data['is_fraud'] == 1].groupby('trans_date').size()
    fig8 = px.line(fraud_counts, title="Zaman İçinde Sahte İşlem Sıklığı", labels={"value": "Sahte İşlem Sayısı", "trans_date": "Tarih"})
    additional_visualizations.append(fig8.to_html(full_html=False))

    # Son işlemden bu yana geçen süre ve sahtecilik ilişkisi
    train_data['time_diff'] = train_data['unix_time'].diff().fillna(0)
    fig9 = px.box(train_data, x="is_fraud", y="time_diff", title="Son İşlemden Bu Yana Geçen Süre ve Sahtecilik İlişkisi", labels={"is_fraud": "Sahte mi?", "time_diff": "Geçen Süre (saniye)"})
    additional_visualizations.append(fig9.to_html(full_html=False))

    # Kategoriye göre işlem tutarı
    fig10 = px.box(train_data, x="category", y="amt", title="Kategoriye Göre İşlem Tutarı", labels={"category": "Kategori", "amt": "İşlem Tutarı"})
    additional_visualizations.append(fig10.to_html(full_html=False))

    # Yaş ve cinsiyet dağılımı
    train_data['age'] = (pd.to_datetime('today') - pd.to_datetime(train_data['dob'])).dt.days // 365
    fig11 = px.scatter(train_data, x="age", y="gender", color="is_fraud", title="Yaş ve Cinsiyet Dağılımı", labels={"age": "Yaş", "gender": "Cinsiyet", "is_fraud": "Sahte mi?"})
    additional_visualizations.append(fig11.to_html(full_html=False))

    # Şehir nüfusu ve işlem tutarı karşılaştırması
    fig12 = px.scatter(train_data, x="city_pop", y="amt", color="is_fraud", title="Şehir Nüfusu ve İşlem Tutarı Karşılaştırması", labels={"city_pop": "Şehir Nüfusu", "amt": "İşlem Tutarı", "is_fraud": "Sahte mi?"})
    additional_visualizations.append(fig12.to_html(full_html=False))

    # Kart numarasına göre işlem sıklığı
    card_counts = train_data.groupby('cc_num').size().reset_index(name="count")
    fig13 = px.bar(card_counts, x="cc_num", y="count", title="Kart Numarasına Göre İşlem Sıklığı", labels={"cc_num": "Kart Numarası", "count": "İşlem Sayısı"})
    additional_visualizations.append(fig13.to_html(full_html=False))

    # Sahtecilik ve işlem zamanı dağılımı
    fig14 = px.density_heatmap(train_data, x="trans_day", y="trans_hour", z="is_fraud", title="Sahtecilik ve İşlem Zamanı Dağılımı", labels={"trans_day": "Gün", "trans_hour": "Saat", "is_fraud": "Sahtecilik"})
    additional_visualizations.append(fig14.to_html(full_html=False))



    return additional_visualizations


# Modeli eğitme ve verileri hazırlama
#catboost_model, X_train, X_test, y_train, y_test, train_metrics, test_roc_auc = train_model()

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
