from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

app = Flask(__name__)

# Veri yükleme
train_data = pd.read_csv('data/fraudTrain.csv')
test_data = pd.read_csv('data/fraudTest.csv')

@app.template_filter('zip')
def zip_filter(*args):
    return zip(*args)


# Veri özetleme fonksiyonları
def get_data_summary():
    summary = {}

    # Train data info
    train_info = {
        "columns": train_data.columns.tolist(),
        "dtypes": train_data.dtypes.astype(str).tolist(),
        "non_null_counts": train_data.count().tolist()
    }
    summary['train_info'] = train_info

    # Test data info
    test_info = {
        "columns": test_data.columns.tolist(),
        "dtypes": test_data.dtypes.astype(str).tolist(),
        "non_null_counts": test_data.count().tolist()
    }
    summary['test_info'] = test_info

    # Train data head
    summary['train_head'] = train_data.head().to_dict(orient='records')

    # Missing values
    summary['missing_train'] = train_data.isnull().sum().to_dict()
    summary['missing_test'] = test_data.isnull().sum().to_dict()

    # Train data describe
    summary['train_describe'] = train_data.describe().to_dict()

    return summary

# Görselleştirme oluşturma
def generate_visualizations():
    visualizations = []

    # Hedef değişkenin dağılımı
    fig1 = px.histogram(train_data, x="is_fraud", title="Distribution of Fraudulent Transactions", labels={"is_fraud": "Is Fraud"})
    visualizations.append(fig1.to_html(full_html=False))

    # İşlem tutarlarının dolandırıcılık durumuna göre dağılımı
    fig2 = px.box(train_data, x="is_fraud", y="amt", title="Transaction Amount vs Fraud", labels={"is_fraud": "Is Fraud", "amt": "Transaction Amount"})
    visualizations.append(fig2.to_html(full_html=False))

    # Cinsiyete göre dağılım
    fig3 = px.histogram(train_data, x="gender", color="is_fraud", barmode="group", title="Distribution of Gender by Fraud", labels={"gender": "Gender", "is_fraud": "Is Fraud"})
    visualizations.append(fig3.to_html(full_html=False))

    # İşlem kategorisine göre dağılım
    fig4 = px.histogram(train_data, x="category", color="is_fraud", barmode="group", title="Distribution of Categories by Fraud", labels={"category": "Category", "is_fraud": "Is Fraud"})
    visualizations.append(fig4.to_html(full_html=False))

    # Sahte işlemlerin saatlik dağılımı
    train_data['trans_hour'] = pd.to_datetime(train_data['trans_date_trans_time']).dt.hour
    fig5 = px.histogram(train_data, x="trans_hour", color="is_fraud", barmode="group", title="Sahte İşlemlerin Saatlik Dağılımı", labels={"trans_hour": "Hour", "is_fraud": "Is Fraud"})
    visualizations.append(fig5.to_html(full_html=False))

    # Gün bazında sahte işlemlerin dağılımı
    train_data['trans_day'] = pd.to_datetime(train_data['trans_date_trans_time']).dt.dayofweek
    fig6 = px.histogram(train_data, x="trans_day", color="is_fraud", barmode="group", title="Day-wise Distribution of Fraudulent Transactions", labels={"trans_day": "Day", "is_fraud": "Is Fraud"})
    visualizations.append(fig6.to_html(full_html=False))

    # Coğrafi verilerin dağılımı
    fig7 = px.scatter(train_data, x="long", y="lat", color="is_fraud", title="Geographical Distribution of Transactions and Fraud", labels={"long": "Longitude", "lat": "Latitude", "is_fraud": "Is Fraud"})
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

    # İşlem Sıklığı Analizi
    transaction_counts = train_data.groupby(['trans_date', 'is_fraud']).size().unstack()
    fig1 = px.line(transaction_counts, title="Zaman İçinde İşlem Sıklığı", labels={"value": "İşlem Sayısı", "trans_date": "Tarih"})
    fig1.update_layout(xaxis_title="Tarih", yaxis_title="İşlem Sayısı", legend_title="Sahte mi?")
    additional_visualizations.append(fig1.to_html(full_html=False))

    # Zaman İçinde Sahte İşlem Sıklığı
    fraud_data = train_data[train_data['is_fraud'] == 1]
    fraud_transaction_counts = fraud_data.groupby('trans_date').size()
    fig2 = px.line(fraud_transaction_counts, title="Zaman İçinde Sahte İşlem Sıklığı", labels={"value": "Sahte İşlem Sayısı", "trans_date": "Tarih"})
    fig2.update_layout(xaxis_title="Tarih", yaxis_title="Sahte İşlem Sayısı")
    additional_visualizations.append(fig2.to_html(full_html=False))

    # Son İşlemden Bu Yana Geçen Süre ve Sahtecilik İlişkisi
    train_data['time_since_last_transaction'] = train_data.groupby('cc_num')['unix_time'].diff()
    fig3 = px.box(train_data, x='is_fraud', y='time_since_last_transaction', title="Son İşlemden Bu Yana Geçen Süre ve Sahtecilik İlişkisi", 
                  labels={"is_fraud": "Sahte mi?", "time_since_last_transaction": "Son İşlemden Bu Yana Geçen Süre (saniye)"})
    additional_visualizations.append(fig3.to_html(full_html=False))

    # Kategoriye Göre İşlem Tutarı
    fig4 = px.box(train_data, x='category', y='amt', color='is_fraud', title="Transaction Amount by Category and Fraud", 
                  labels={"category": "Kategori", "amt": "İşlem Tutarı", "is_fraud": "Sahte mi?"})
    fig4.update_xaxes(tickangle=45)
    additional_visualizations.append(fig4.to_html(full_html=False))

    # Yaş ve Cinsiyet Analizi
    train_data['dob'] = pd.to_datetime(train_data['dob'])
    train_data['age'] = (train_data['trans_date_trans_time'] - train_data['dob']).dt.days // 365
    fig5 = px.box(train_data, x='gender', y='age', color='is_fraud', title="Age and Gender Distribution by Fraud", 
                  labels={"gender": "Cinsiyet", "age": "Yaş", "is_fraud": "Sahte mi?"})
    additional_visualizations.append(fig5.to_html(full_html=False))

    # İşlem Tutarı ve Şehir Nüfusu Karşılaştırması
    fig6 = px.scatter(train_data, x='city_pop', y='amt', color='is_fraud', title="Transaction Amount vs. City Population and Fraud", 
                      labels={"city_pop": "Şehir Nüfusu", "amt": "İşlem Tutarı", "is_fraud": "Sahte mi?"})
    additional_visualizations.append(fig6.to_html(full_html=False))

    # Kart Numarasına Göre İşlem Sıklığı
    card_transaction_counts = train_data['cc_num'].value_counts()
    fig7 = px.histogram(card_transaction_counts, nbins=20, title="Transaction Frequency by Card Number", 
                        labels={"value": "Transaction Count", "index": "Card Number"})
    additional_visualizations.append(fig7.to_html(full_html=False))

    # İşlem Zamanı ve Sahtecilik Korelasyonu
    train_data['trans_hour'] = pd.to_datetime(train_data['trans_date_trans_time']).dt.hour
    fig8 = px.histogram(train_data, x='trans_hour', color='is_fraud', title="Sahteciliğe Göre İşlem Saati Dağılımı", 
                        labels={"trans_hour": "Saat", "count": "İşlem Sayısı", "is_fraud": "Sahte mi?"}, barmode='stack')
    additional_visualizations.append(fig8.to_html(full_html=False))

    return additional_visualizations
@app.route('/')
def index():
    data_summary = get_data_summary()
    visualizations = generate_visualizations()
    additional_visualizations = generate_additional_visualizations()
    visualizations.extend(additional_visualizations)  # Yeni görselleri ekleyin
    return render_template('index.html', data_summary=data_summary, visualizations=visualizations)


if __name__ == '__main__':
    app.run(debug=True)
