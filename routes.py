# routes.py
from flask import render_template, request, redirect, url_for
from app import app
from model_training import train_model
from model_storage import save_model, load_model, list_saved_models
from visualization import generate_visualizations, generate_additional_visualizations, get_data_summary

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

# Model formu sayfası
@app.route('/model-form', methods=['GET', 'POST'])
def model_form():
    if request.method == 'POST':
        # Kullanıcı tarafından girilen değerleri al
        iterations = int(request.form['iterations'])
        depth = int(request.form['depth'])
        learning_rate = float(request.form['learning_rate'])
        random_seed = int(request.form['random_seed'])

        # Modeli eğit ve kaydet
        model_id, model_info = train_model(iterations, depth, learning_rate, random_seed)
        save_model(model_id, model_info)

        # Model bilgi sayfasına yönlendir
        return redirect(url_for('model_info', model_id=model_id))

    return render_template('modelForm.html')

# Model bilgi sayfası
@app.route('/model-info/<model_id>')
def model_info(model_id):
    # Modeli yükle
    model_info = load_model(model_id)
    if not model_info:
        return "Model bulunamadı.", 404

    # Model bilgilerini sayfaya gönder
    return render_template(
        'modelInfo.html',
        test_roc_auc=model_info['test_roc_auc'],
        roc_html=model_info['roc_html'],
        feature_html=model_info['feature_html'],
        train_metrics_table=model_info['train_metrics_table']
    )

# Kaydedilmiş modellerin listesi
@app.route('/model-list')
def model_list():
    models = list_saved_models()
    return render_template('modelList.html', models=models)




@app.route('/visualization/<int:vis_id>')
def serve_visualization(vis_id):
    visualizations = generate_visualizations() + generate_additional_visualizations()
    if 0 <= vis_id < len(visualizations):
        return visualizations[vis_id]
    return f"Visualization {vis_id} bulunamadı. Toplam görselleştirme sayısı: {len(visualizations)}", 404
