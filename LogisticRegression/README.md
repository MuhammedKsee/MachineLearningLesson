# Lojistik Regresyon ile Diyabet Tahmini

Bu proje, makine öğrenmesi algoritmalarından biri olan Lojistik Regresyon'un Python kullanılarak uygulanmasını ve Pima Indians Diabetes veri seti üzerinde diyabet tahmini yapılmasını içermektedir.

## Proje Hakkında

Bu projede, lojistik regresyon algoritmasını sıfırdan Python ile uyguladık ve Pima Indians Diabetes veri seti üzerinde test ettik. Projede:

- Lojistik Regresyon sınıfının sıfırdan gerçeklenmesi
- Gradyan iniş optimizasyonu
- Binary Cross Entropy kayıp fonksiyonu
- Pima Indians Diabetes veri setinin analizi ve ön işlenmesi
- Model eğitimi ve değerlendirilmesi
- Öznitelik önemi analizi
- Kullanıcı girişine göre diyabet risk tahmini

konuları ele alınmaktadır.

## Veri Seti

Projede kullanılan Pima Indians Diabetes veri seti, Arizona'daki Pima Kızılderili toplumunda yaşayan en az 21 yaşındaki kadınlar için diabetes mellitus tanı bilgilerini içerir. Veri seti:

- 768 örnek
- 8 öznitelik (hamilelik sayısı, glikoz seviyesi, kan basıncı vb.)
- 1 hedef değişken (diyabet durumu: 0 veya 1)

içermektedir.

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki kütüphanelere ihtiyaç vardır:

```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

## Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/MuhammedKsee/MachineLearningLesson.git
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

Projeyi çalıştırmak için:

```bash
python example.py
```

Diyabet risk tahmini için, `example.py` dosyasının sonundaki `predict_diabetes_risk()` fonksiyonunu etkinleştirin ve kişisel bilgilerinizi girin.

## Çıktılar

Program çalıştırıldığında şu görseller oluşturulur:

1. `diabetes_loss_history.png` - Eğitim sırasında kayıp fonksiyonunun değişimi
2. `diabetes_feature_importance.png` - Özniteliklerin önem sırasına göre sıralanması
3. `diabetes_threshold_analysis.png` - Farklı olasılık eşik değerlerine göre model performansı

## Lojistik Regresyon Sınıfı

`SimpleLogisticRegression` sınıfı şu özelliklere sahiptir:

- Vektörel işlemler ile optimize edilmiş implementasyon
- Ayarlanabilir öğrenme hızı ve iterasyon sayısı
- Sigmoid aktivasyon fonksiyonu
- Kayıp fonksiyonu izleme
- Olasılık ve ikili sınıflandırma tahminleri

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.