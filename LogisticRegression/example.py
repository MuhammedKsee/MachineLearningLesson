import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from simpleLogisticReg import SimpleLogisticRegression

# Pima Indians Diabetes veri setini yükle
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=column_names)

print("Veri seti boyutu:", data.shape)
print("\nİlk 5 satır:")
print(data.head())

# Eksik verileri kontrol et ve işle
print("\nÖznitelik istatistikleri:")
print(data.describe())

# Eksik değerleri (0 olarak girilmiş) ortalama değerlerle değiştir
for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    data[column] = data[column].replace(0, np.NaN)
    mean_value = data[column].mean(skipna=True)
    data[column] = data[column].fillna(mean_value)

# Özellikleri ve hedefi ayır
X = data.iloc[:, :-1].values  # Tüm özellikler
y = data.iloc[:, -1].values   # Outcome sütunu

# Verileri normalize et
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Veriyi eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelimizi oluştur ve eğit
model = SimpleLogisticRegression(learning_rate=0.01, num_iterations=3000)
model.fit(X_train, y_train)

# Test veri seti üzerinde tahmin yap
y_pred = model.predict(X_test)

# Sonuçları değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Doğruluğu: {accuracy:.4f}")
print("\nKarmaşıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Kayıp fonksiyonu geçmişini görselleştir
plt.figure(figsize=(10, 6))
plt.plot(range(0, model.num_iterations, 100), model.loss_history)
plt.title('Diyabet Tahmini: Kayıp Fonksiyonunun İterasyon Sayısına Göre Değişimi')
plt.xlabel('İterasyon')
plt.ylabel('Kayıp')
plt.grid(True)
plt.savefig('diabetes_loss_history.png')

# Öznitelik önemini görselleştir (ağırlıklara göre)
plt.figure(figsize=(12, 6))
feature_importance = np.abs(model.weights.flatten())
sorted_idx = np.argsort(feature_importance)
plt.barh(np.array(column_names[:-1])[sorted_idx], feature_importance[sorted_idx])
plt.title('Öznitelik Önemi (Mutlak Ağırlık Değerleri)')
plt.xlabel('Mutlak Ağırlık Değeri')
plt.tight_layout()
plt.savefig('diabetes_feature_importance.png')

# Model performansını farklı eşik değerlerine göre inceleyebiliriz
thresholds = np.arange(0.1, 1.0, 0.1)
accuracies = []

for threshold in thresholds:
    y_pred = model.predict(X_test, threshold=threshold)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies, marker='o')
plt.title('Farklı Eşik Değerlerine Göre Model Doğruluğu')
plt.xlabel('Eşik Değeri')
plt.ylabel('Doğruluk')
plt.grid(True)
plt.savefig('diabetes_threshold_analysis.png')

print("\nGörselleştirmeler kaydedildi.")

# Diyabet riski tahmin aracı (Kullanıcı girişi)
def predict_diabetes_risk():
    print("\n=== Diyabet Risk Tahmini ===")
    user_input = {}
    user_input['Pregnancies'] = float(input("Hamilelik sayısı: "))
    user_input['Glucose'] = float(input("Glikoz seviyesi (mg/dL): "))
    user_input['BloodPressure'] = float(input("Kan basıncı (mm Hg): "))
    user_input['SkinThickness'] = float(input("Cilt kalınlığı (mm): "))
    user_input['Insulin'] = float(input("İnsülin (mu U/ml): "))
    user_input['BMI'] = float(input("Vücut kitle indeksi: "))
    user_input['DiabetesPedigreeFunction'] = float(input("Diyabet soy ağacı fonksiyonu: "))
    user_input['Age'] = float(input("Yaş: "))
    
    # Kullanıcı girişini aynı şekilde ölçeklendirme
    user_df = pd.DataFrame([user_input])
    user_values = scaler.transform(user_df)
    
    # Olasılık tahmini
    probability = model.predict_proba(user_values)[0][0]
    prediction = model.predict(user_values)[0][0]
    
    print(f"\nDiyabet olma olasılığı: {probability:.2f}")
    print(f"Tahmin: {'Diyabet' if prediction else 'Diyabet Değil'}")

# İsteğe bağlı olarak risk tahmini yapabilirsiniz
# predict_diabetes_risk() 