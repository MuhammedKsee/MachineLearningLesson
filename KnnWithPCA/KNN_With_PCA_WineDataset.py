import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Veri setini yükleme
df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)

# Sütun isimlerini belirle
df_wine.columns = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 
                   'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# Veri setini inceleyelim
print("Veri seti boyutu:", df_wine.shape)
print("Sınıf dağılımı:\n", df_wine['Class'].value_counts())
print("İlk 5 satır:\n", df_wine.head())

# Özellikler ve hedef değişkeni ayırma
X = df_wine.iloc[:, 1:].values
y = df_wine.iloc[:, 0].values

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi ölçeklendirme (normalleştirme)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Farklı k değerleri için doğruluk oranlarını test edelim
k_values = range(1, 20, 2)
accuracies = []

for k in k_values:
    # Scikit-learn'ün KNN modelini oluştur ve eğit
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    # Test seti üzerinde tahmin yap
    y_pred = knn.predict(X_test_scaled)
    
    # Doğruluk oranını hesapla
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"k = {k}, Doğruluk: {accuracy:.4f}")

# En iyi k değeri ile modeli yeniden eğit
best_k = k_values[np.argmax(accuracies)]
print(f"En iyi k değeri: {best_k}, Doğruluk: {max(accuracies):.4f}")

# En iyi modeli oluştur
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
y_pred = best_knn.predict(X_test_scaled)

# Sonuçları görüntüle
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# K değerleri ve doğruluk grafiği
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', markersize=8, linewidth=2)
plt.title('K Değerine Göre Doğruluk Oranı (Scikit-learn KNN)')
plt.xlabel('K Değeri')
plt.ylabel('Doğruluk')
plt.grid(True)
plt.xticks(k_values)
plt.savefig('sklearn_knn_accuracy_plot.png')
plt.show()

# PCA analizi
print("\n----- PCA Analizi -----")

# Tüm veri seti üzerinde PCA analizi
X_scaled = scaler.fit_transform(X)

# 4 bileşenli PCA
pca = PCA(n_components=4)
X_pca_4 = pca.fit_transform(X_scaled)
print("Orijinal veri şekli:", X_scaled.shape)
print("4 bileşenli PCA sonrası veri şekli:", X_pca_4.shape)

# Varyans açıklama oranları
print("\nAçıklanan varyans:", pca.explained_variance_)
print("\nAçıklanan varyans oranı:", pca.explained_variance_ratio_)
print("\nKümülatif açıklanan varyans oranı:", np.cumsum(pca.explained_variance_ratio_))

# 2 bileşenli PCA
pca = PCA(n_components=2)
X_pca_2 = pca.fit_transform(X_scaled)
print("\n2 bileşenli PCA sonrası veri şekli:", X_pca_2.shape)

# 2 boyutlu görselleştirme
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], 
                     c=y, edgecolor="none", alpha=0.7,
                     cmap=plt.cm.get_cmap("Spectral", 10))
plt.colorbar(scatter, label='Şarap Sınıfı')
plt.title('Wine Veri Setinin PCA ile 2 Boyutlu Görselleştirilmesi')
plt.xlabel("Birinci Bileşen")
plt.ylabel("İkinci Bileşen")
plt.savefig('wine_pca_2d.png')
plt.show()

# Optimum bileşen sayısını belirlemek için analiz
pca = PCA().fit(X_scaled)
plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.axhline(0.90, c="r", linestyle='--', label='%90 Varyans Açıklama Seviyesi')
plt.grid(True)
plt.xlabel("Bileşen Sayısı")
plt.ylabel("Kümülatif Açıklanan Varyans")
plt.title("PCA Bileşen Sayısına Göre Açıklanan Varyans")
plt.legend()
plt.savefig('wine_pca_components.png')
plt.show()