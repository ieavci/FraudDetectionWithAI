
<h1>Fraud Detection AI Project</h1>
    <hr>

  <h2>Overview (Genel Bakış)</h2>
    <p><strong>Fraud Detection AI</strong> is a machine learning-based solution designed to detect fraudulent credit card transactions. This project employs multiple algorithms, including CatBoost, LSTM, Logistic Regression, Random Forest, XGBoost, LightGBM, and Parzen Window, to identify fraudulent activities within a highly imbalanced dataset. The system focuses on optimizing CatBoost's hyperparameters and offers a user-friendly web interface for interactive model performance analysis.</p>

  <p><strong>Sahtecilik Tespiti Yapay Zeka Projesi</strong>, kredi kartı işlemlerinde sahtecilik tespiti yapmak için geliştirilmiş bir makine öğrenimi tabanlı çözümdür. Bu proje, dengesiz bir veri seti üzerinde sahtecilik tespiti için CatBoost, LSTM, Lojistik Regresyon, Random Forest, XGBoost, LightGBM ve Parzen Pencereleme gibi çeşitli algoritmalar kullanmaktadır. Sistem, CatBoost algoritmasının hiperparametrelerini optimize etmeye odaklanır ve etkileşimli model performansı analizi için kullanıcı dostu bir web arayüzü sunar.</p>

  <hr>

   <h2>Installation (Kurulum)</h2>
   <p>To install the project on your computer, follow these steps:</p>

  <p><strong>Projeyi bilgisayarınıza kurmak için şu adımları takip edin:</strong></p>

   <ol>
        <li>Clone the repository:
            <pre><code>git clone https://github.com/ieavci/FraudDetectionWithAI.git</code></pre>
        </li>
        <li>Navigate to the project directory:
            <pre><code>cd FraudDetectionWithAI</code></pre>
        </li>
        <li>Install the required dependencies:
            <pre><code>pip install -r requirements.txt</code></pre>
        </li>
        <li>Download the data from Kaggle: <a href="https://www.kaggle.com/datasets/kartik2112/fraud-detection" target="_blank">Download Fraud Detection Data</a></li>
        <li>Update the file paths for data in <code>model_training.py</code> and <code>visualization.py</code>.</li>
        <li>Start the project by running:
            <pre><code>python app.py</code></pre>
        </li>
        <li>Open the website in your browser:
            <pre><code>http://127.0.0.1:5000</code></pre>
            Or use your own configured port.
        </li>
    </ol>

   <hr>

  <h2>Project Structure (Proje Yapısı)</h2>
    <ul>
        <li><strong>model_training.py</strong>: Model training and evaluation logic.</li>
        <li><strong>visualization.py</strong>: Data visualization for analysis and results.</li>
        <li><strong>app.py</strong>: Web interface for interacting with the models.</li>
        <li><strong>requirements.txt</strong>: Project dependencies.</li>
        <li><strong>data/</strong>: Directory for storing datasets.</li>
    </ul>

   <hr>

   <h2>Technologies Used (Kullanılan Teknolojiler)</h2>
    <ul>
        <li><strong>Machine Learning Models</strong>: CatBoost</li>
        <li><strong>Programming Language</strong>: Python</li>
        <li><strong>Web Framework</strong>: Flask</li>
        <li><strong>Data Visualization</strong>: Matplotlib, Plotly</li>
    </ul>

   <hr>

   <h2>Results (Sonuçlar)</h2>
    <ul>
        <li>The CatBoost algorithm outperforms others with a high ROC-AUC score (0.99569), especially in handling imbalanced datasets.</li>
        <li>Insights show that fraudulent activities often occur at unusual times and specific vendor categories.</li>
    </ul>

   <hr>

   <h2>License (Lisans)</h2>
    <p>This project is licensed under the MIT License.</p>
