# Makine Öğrenmesi Projesi

Bu depo, makine öğrenmesi ve veri bilimi çalışmaları için hazırlanmış bir proje tabanıdır.

## Kurulum

Gerekli bağımlılıkları yüklemek için:

```bash
pip install -r requirements.txt
```

## Kullanılan Teknolojiler

Bu projede aşağıdaki kütüphaneler kullanılmaktadır:

- **NumPy & Pandas**: Veri manipülasyonu ve analizi
- **Scikit-learn**: Klasik makine öğrenmesi algoritmaları
- **TensorFlow & Keras**: Derin öğrenme
- **PyTorch**: Derin öğrenme
- **Matplotlib & Seaborn**: Veri görselleştirme
- **XGBoost**: Gradyan artırma modelleri

## Örnekler

### Veri Yükleme

```python
import pandas as pd

# Veri setini yükle
data = pd.read_csv('data/raw/ornek_veri.csv')

# Veri setini incele
print(data.head())
print(data.info())
print(data.describe())
```

### Model Eğitimi

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Performansı değerlendir
print(f"Doğruluk: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
```


## İletişim

Proje Sahibi - [E-posta adresiniz](mailto:kosemuhammet545@gmail.com)
