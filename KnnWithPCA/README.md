# Wine Veri Seti KNN ve PCA Analizi

Bu proje, UCI Machine Learning Repository'den alınan Wine veri seti üzerinde K-En Yakın Komşu (KNN) sınıflandırma algoritması ve Temel Bileşen Analizi (PCA) yöntemlerini uygulayan bir çalışmadır.

## Veri Seti

Wine veri seti, üç farklı şarap türünün kimyasal analizlerini içerir. 13 özellik ve 178 örnek içeren bu veri seti, sınıflandırma algoritmaları için yaygın olarak kullanılan bir veri setidir.

Veri setindeki özellikler:
1. Alkol
2. Malik asit
3. Kül
4. Külün alkalinliği
5. Magnezyum
6. Toplam fenoller
7. Flavanoidler
8. Flavanoid olmayan fenoller
9. Proantosiyanidinler
10. Renk yoğunluğu
11. Renk tonu
12. OD280/OD315 seyreltilmiş şaraplar
13. Prolin

## Proje İçeriği

Bu projede aşağıdaki analizler yapılmıştır:

1. **KNN Sınıflandırması**:
   - Farklı k değerleri (1-19 arasında) için model performansı karşılaştırması
   - En iyi k değerinin belirlenmesi
   - Sınıflandırma performansı raporu (kesinlik, duyarlılık, F1-skoru)

2. **PCA (Temel Bileşen Analizi)**:
   - Boyut indirgeme 
   - Veri görselleştirme (2 boyutlu)
   - Açıklanan varyans analizi
   - Optimum bileşen sayısının belirlenmesi

## Grafikler ve Çıktılar

Proje çalıştırıldığında aşağıdaki çıktılar oluşturulur:

1. `sklearn_knn_accuracy_plot.png`: Farklı k değerleri için doğruluk oranlarını gösteren grafik
2. `wine_pca_2d.png`: Veri setinin 2 boyutlu PCA görselleştirmesi
3. `wine_pca_components.png`: PCA bileşen sayısına göre açıklanan varyans grafiği

## Kullanım

Projeyi çalıştırmak için:

```bash
python KNN_With_PCA_WineDataset.py
```

## Gereksinimler

Projeyi çalıştırmak için gereken kütüphaneler `requirements.txt` dosyasında belirtilmiştir. Kurulum için:

```bash
pip install -r requirements.txt
``` 