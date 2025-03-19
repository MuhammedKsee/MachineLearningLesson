# Numpy kütüphanesini içe aktar
import numpy as np
# Pandas kütüphanesini içe aktar - CSV okumak için
import pandas as pd
# Görselleştirme için matplotlib kütüphanesini içe aktar
import matplotlib.pyplot as plt

# Çoklu doğrusal regresyon için sınıf tanımla
class MultiLinearRegression:
    # Başlangıç metodu
    def __init__(self):
        # Katsayı ve sabit terim değişkenlerini None olarak başlat
        self.coefficients = None
        self.intercept = None

    # Model eğitim metodu
    def fit(self, X, y):
        # X'i numpy array'e çevir
        X_arr = np.array(X)
        # y'yi numpy array'e çevir
        y_arr = np.array(y)
        
        # X'e birler sütunu ekle (intercept için)
        X_with_ones = np.column_stack((np.ones(len(X_arr)), X_arr))
        
        # Normal denklemi çöz: beta = (X^T X)^(-1) X^T y
        # Tekil matrisler için pseudoinverse kullanılır (Moore-Penrose)
        beta = np.dot(np.linalg.pinv(X_with_ones), y_arr)
        
        # İlk beta değerini sabit terim olarak kaydet
        self.intercept = beta[0]
        # Geri kalan beta değerlerini katsayılar olarak kaydet
        self.coefficients = beta[1:]
        
        # Katsayıları ekrana yazdır
        print("Katsayılar:", self.coefficients)
        # Sabit terimi ekrana yazdır
        print("Sabit terim:", self.intercept)

    # Tahmin metodu
    def predict(self, X):
        # X'i numpy array'e çevir
        X_arr = np.array(X)
        # Tahmin formülünü uygula: y = intercept + X * coefficients
        return self.intercept + np.dot(X_arr, self.coefficients)
    
    # R-kare skorunu hesapla
    def score(self, X, y):
        # Tahminleri yap
        y_pred = self.predict(X)
        # Toplam kareler toplamı (total sum of squares)
        ss_tot = np.sum((y - np.mean(y))**2)
        # Hata kareler toplamı (residual sum of squares)
        ss_res = np.sum((y - y_pred)**2)
        # R-kare hesapla
        r2 = 1 - (ss_res / ss_tot)
        return r2

# CSV dosyasını oku
data = pd.read_csv('LinearReg/multipleLinReg/random_dataset.csv')

# Veri setini incele
print("Veri seti boyutu:", data.shape)
print("İlk 5 satır:")
print(data.head())

# Özellikler ve hedef değişkeni ayır
X = data[['SquareMeters', 'Age']].values
y = data['Price'].values

# Veriyi eğitim ve test kümelerine ayır
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğit
model = MultiLinearRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = model.predict(X_test)

# Modelin performansını değerlendir
r2_score = model.score(X_test, y_test)
print(f"R-kare skoru: {r2_score:.4f}")

# Ortalama Mutlak Hata (MAE)
mae = np.mean(np.abs(y_test - y_pred))
print(f"Ortalama Mutlak Hata: {mae:.2f}")

# Gerçek değerler ve tahminler arasındaki ilişkiyi görselleştir
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Gerçek Fiyat')
plt.ylabel('Tahmin Edilen Fiyat')
plt.title('Gerçek Fiyat vs Tahmin Edilen Fiyat')
plt.legend()
plt.savefig('LinearReg/multipleLinReg/price_prediction.png')

# Kullanıcıdan yeni veri girişi al
def predict_price():
    try:
        square_meters = float(input("Ev büyüklüğünü girin (m²): "))
        age = float(input("Evin yaşını girin: "))
        
        # Tahmin yap
        predicted_price = model.predict([[square_meters, age]])[0]
        print(f"\nTahmin edilen fiyat: {predicted_price:.2f} TL")
    except ValueError:
        print("Lütfen geçerli bir sayı girin!")

# Kullanıcı tahmini yapmayı dene
print("\nYeni bir evin fiyatını tahmin edelim:")
predict_price()