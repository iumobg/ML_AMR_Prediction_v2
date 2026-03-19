# Proje Mimarisi (Project Architecture)

Bu belge, *E. coli* genomik verilerinde Antimikrobiyal Direnç (AMR) tahminini amaçlayan makine öğrenmesi boru hattının (pipeline) genel mimarisini, dosya görevlerini, matematiksel altyapısını ve model konfigürasyonlarını detaylandırır.

---

## 1. Genel Sistem Mimarisi (Pipeline Akışı)

Proje, referanssız (alignment-free) genomik k-mer analizi ile ikili (binary) bir sınıflandırma problemi çözer. Sistem uçtan uca şu adımlardan oluşur:

1. **Veri Girişi ve Doğrulama (Data Ingestion & Validation):** Ham `.fna` genom montajları ve etiketleri (direnç/duyarlılık) içeren meta veri (`.csv`) yüklenir. Hedef antibiyotik için yeterli örneklem olup olmadığı istatistiksel eşikler üzerinden test edilir.
2. **K-mer Çıkarımı (Feature Extraction):** Geçerli genomlar KMC sayacından geçirilerek her bir bakteri için k=31 uzunluğunda alt dizilimler (k-mer) çıkarılır ve veritabanlarına (`.kmc_pre`, `.kmc_suf`) dönüştürülür.
3. **Öznitelik Mühendisliği ve Matris Oluşturma (Feature Engineering):** KMC çıktıları belleğe sığmayacak kadar büyük olduğu için seyrek (sparse) matrislere (`scipy.sparse.csr_matrix`) dönüştürülür. Çok nadir görülen (Örn: < 10) veya çok yaygın görülen k-mer'ler filtrelenerek gürültü azaltılır ve matris parçalar (chunk) halinde `.npz` formatında kaydedilir.
4. **Hiperparametre Optimizasyonu (Optimization):** Matris parçaları içinden "kolay, orta, zor" örnekleri temsil edecek stratejik bir alt küme seçilir ve Optuna (Bayesian Optimization) ile XGBoost hiperparametreleri aranır.
5. **Model Eğitimi (Training):** Optuna'nın bulduğu en iyi parametreler kullanılarak XGBoost algoritması tüm matris parçacıkları üzerinde artımlı (incremental) olarak eğitilir.
6. **Değerlendirme (Evaluation):** Model önceden ayrılmış (hold-out) test setinde test edilir. Eşik değerleri kalibre edilerek performans metrikleri hesaplanır.
7. **Açıklanabilirlik (Explainability):** Tahmin gücü en yüksek olan (Gain metriğine göre) en iyi *N* k-mer belirlenir ve biyolojik doğrulama için `.fasta` ve `.csv` olarak dışa aktarılır.

---

## 2. Dosya Bazlı Analiz

Projenin temel kod tabanı `scripts/` dizininde sıralı bir şekilde yer almaktadır:

* **`01_data_validation.py`**: Veri kalite kontrolü yapar. Veri setindeki (özellikle azınlık sınıfı) dengesizlikleri kontrol eder ve makine öğrenmesi için asgari istatistiksel koşulların (Stratified Hold-out geçerliliği) sağlanıp sağlanmadığını belirler.
* **`02_kmer_extraction.py`**: Genom dosyalarından (`.fna`) referanssız k-mer haritaları çıkarılması için KMC aracını çağırır. İşlemleri paralelleştirerek genomları indeksler.
* **`03_matrix_construction.py`**: Çıkarılan k-merleri tüm veri seti için global bir özellik sözlüğüne (`features.txt`) dönüştürür. RAM taşmasını önlemek için veriyi 100 genomluk `chunk` (parça) dosyaları halinde `CSR` seyrek matrislerine (`.npz`) çevirir.
* **`04_optimization.py`**: Optuna kütüphanesi ile hiperparametre aramasını (Hyperparameter Tuning) yürütür. Doğru bir doğrulama yapmak için "Stratified Linspace Chunking" tekniği ile veri setinin dağılımını temsil eden örneklem parçalarını seçer ve modeli çoklu denemelerle (trials) optimize eder.
* **`05_model_training.py`**: Optimizasyon adımından gelen en iyi parametre konfigürasyonunu (ör. `config_ciprofloxacin.yaml`) yükler. Stratified train/test ayrımını koruyarak XGBoost modelini tüm eğitim parçaları üzerinde artımlı (incremental) olarak eğitir ve modeli kaydeder.
* **`06_evaluation.py`**: Kaydedilen modelin performansını test parçaları üzerinde ölçer. Karmaşıklık matrisini oluşturur ve ROC-AUC, Accuracy, F1-Score, MCC gibi temel sınıflandırma metriklerini raporlar.
* **`07_explainability.py`**: XGBoost'un ağaçlarındaki bölünmeleri inceleyerek en yüksek "Gain" (Kazanç) değerine sahip k-merleri (özellikleri) bulur. Bu k-merleri biyolojik araştırma amaçlı dışa aktarır (Örn: `top_50_features_cipro_final.fasta`).
* **`config/config_ciprofloxacin.yaml`**: Hedef antibiyotik için Optuna tarafından bulunan en iyi hiperparametreleri (learning_rate, max_depth vb.) ve train/test bölüm bilgilerini (chunk index) sabitler. Tam tekrarlanabilirliği (reproducibility) sağlar.

---

## 3. Matematiksel ve İstatistiksel Temeller

### 3.1 Sınıf Dengesizliği Eşiği (Dynamic Imbalance Ratio)
Makine öğrenmesi modelleri azınlık sınıflarında yetersiz (overfitting) kalabilir. Projede, "Dirençli" ve "Duyarlı" sınıfların asgari örneklem kabulü, veri miktarının ($N_{toplam}$) büyüklüğüne göre dinamik olarak hesaplanmıştır:

$$ P(S\imath n\imath f) = \left( \frac{\min(N_{diren\mbox{ç}li}, N_{duyarl\imath})}{N_{toplam}} \right) \times 100 $$

Küçük veri setleri azınlık sınıfında daha yüksek toleransa ihtiyaç duyarken ($N_{toplam} < 200 \implies 40.0\%$), büyük ağaç modelleriyle çalışılan yüksek örneklemlerde bu sınır düşer ($N_{toplam} \geq 2000 \implies 2.0\%$). Ayrıca test setinde tutarlı bir karmaşıklık matrisi elde etmek için mutlaka $\min(N_{diren\mbox{ç}li}, N_{duyarl\imath}) \geq 40$ koşulu aranmaktadır.

### 3.2 Maliyet ve Kayıp Fonksiyonu (Objective Loss Function)
XGBoost ile yapılan eğitimin temeli, ikili (binary) sınıflandırma ağaçları oluşturmaya dayanır. Ağaçların optimize etmek istediği maliyet fonksiyonu, "Log-Loss" veya Lojistik Kayıp (Binary Logistic Loss) formülüdür. Her bir $i$ örneği için gerçek etiket $y_i \in \{0, 1\}$ ve tahmin edilen dirençlilik olasılığı $p_i \in (0, 1)$ olmak üzere hedef küçültülmek istenen denklem:

$$ \mathcal{L}(y, p) = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] $$

### 3.3 Özellik Önemi (Feature Importance - Gain)
XGBoost, bir özelliğin (k-mer) bölünmede ne kadar kullanışlı olduğunu "Gain" (Kazanç) metriğiyle hesaplar (`07_explainability.py`). Bir özelliğin sınıflandırma ağacına katkısı, mevcut düğümü (node) sol ($L$) ve sağ ($R$) tarafa ayırdığında varyansta yarattığı azalmadır.
Gradiyentlerin ($G$) ve Hessiyenlerin ($H$) ağırlığı dikkate alındığında, yaprak düğümün bölünmesinden elde edilen Kazanç (Gain) hesaplama formülü:

$$ \text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma $$

Burada:
- $G_L, G_R$: Sol ve sağ alt düğümlerdeki kayıp fonksiyonunun birinci türevi (Gradiyent).
- $H_L, H_R$: Sol ve sağ alt düğümlerdeki kayıp fonksiyonunun ikinci türevi (Hessiyen).
- $\lambda$: L2 Düzenlileştirme parametresi (Aşırı öğrenmeyi önler).
- $\gamma$: Minimum kayıp azalma (Minimum loss reduction) eşiği.

---

## 4. Hiperparametre ve Model Konfigürasyonu

Modelin aşırı öğrenmesini (overfitting) engellemek ve tahmin başarısını maksimize etmek için Bayesian Optimizasyonu kapsamında **TPE (Tree-structured Parzen Estimator)** algoritması ile Optuna üzerinden arama yapılmıştır (`04_optimization.py`). Seçilen konfigürasyon (`config/config_ciprofloxacin.yaml`) içerisindeki temel parametreler ve sistemdeki işlevleri aşağıdaki gibidir:

* **`n_estimators` (223):** Kurulacak toplam karar ağacı sayısıdır. Çok düşük olması az öğrenmeye (underfitting), çok yüksek olması aşırı öğrenmeye (overfitting) iter. Algoritma `eval_metric: 'auc'` izleyerek bu sayıda durmayı seçmiştir.
* **`max_depth` (5):** Her bir karar ağacının inebileceği maksimum derinliktir. Yüksek boyutlu genomik verilerde (17 Milyon özellik), derinliğin 5'te sınırlanması, ağaçların çok özgül detayları ezberlemesinden ziyade genel örüntüleri (k-mer kompozisyonları) yakalamasını sağlamıştır.
* **`learning_rate` ($\sim$0.032):** Öğrenme oranı (shrinkage). Her bir yeni ağacın toplam modele yapacağı katkıyı belirler. Düşük tutularak (0.032), sistem daha sağlam (robust) ancak daha yavaş yakınsamıştır.
* **`min_child_weight` (7):** Bir yaprağın bölünebilmesi için o yaprakta gereken minimum veri ağırlığı (Hessiyen toplamı) eşiğidir. Değerin 7 olması, modeli çok nadir görülen spesifik genomlara göre bölünmekten koruyarak genelleyici yapmıştır.
* **`subsample` ($\sim$0.673) ve `colsample_bytree` ($\sim$0.693):** Ağaç oluşturulurken her seferinde toplam örneklerin %67'sinin ve toplam k-mer özelliklerinin %69'unun seçileceğini ifade eder. Bu rassallık (stochastic gradient boosting), gürültülü (noisy) özelliklere veya aykırı değerlere olan bağımlılığı doğrudan azaltır.
* **`gamma` ($\sim$2.11):** Bir ağaç dalının split (bölünme) işlemi yapması için Loss fonksiyonunda asgari yapması gereken iyileştirme miktarıdır. Bu sayede sadece önemli genetik belirteçler için dallanma yapılır.
* **`scale_pos_weight` ($\sim$1.051):** Pozitif/Negatif sınıf ağırlıklandırıcı. Dengesizlikleri kompanse edebilmesi adına algoritma tarafından hafifçe pozitif sınıfa lehine kaydırılmıştır.

---

## 5. Sonuç ve Metrikler

Tahmin sonuçlarının test veri kümeleri üzerinde hesaplanmasında (`06_evaluation.py`) şu sınıflandırma metrikleri kullanılır:
(*Kullanılan Değişkenler: TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative*)

* **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Farklı karar eşiklerinde (thresholds) Doğru Pozitif Oranı ($TPR = \frac{TP}{TP + FN}$) ile Yanlış Pozitif Oranı ($FPR = \frac{FP}{FP + TN}$) arasındaki eğrinin altında kalan alandır. Modelin dirençli ile duyarlı örnekleri istatistiksel ayırma gücünü en iyi yansıtan ana metriktir.
* **Accuracy (Doğruluk):** Toplam doğru sınıflandırılan genomların test setindeki tüm genomlara oranıdır.
  $$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} $$
* **Balanced Accuracy (Dengeli Doğruluk):** Sınıf dengesizliklerinden etkilenmemesi için Duyarlılık (Recall) ve Özgüllük (Specificity) değerlerinin ortalamasını alır.
  $$ Balanced Accuracy = \frac{ \frac{TP}{TP+FN} + \frac{TN}{TN+FP} }{2} $$
* **F1-Score:** Yanlış Pozitif (FP) ve Yanlış Negatif (FN) hataları arasında bir denge kurmak için Precision ve Recall'un Harmonik ortalamasını alır.
  $$ F1\_Score = 2 \times \frac{Precision \times Recall}{Precision + Recall} = \frac{2 \times TP}{2 \times TP + FP + FN} $$
* **MCC (Matthews Correlation Coefficient):** Sınıf boyutları çok farklı olsa dahi $-1$ ile $+1$ arasında bir korelasyon vererek modelin tahminlerinin rastgelelikten ne kadar uzaklaştığını istatistiksel olarak kesin sunar.
  $$ MCC = \frac{(TP \cdot TN) - (FP \cdot FN)}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}} $$