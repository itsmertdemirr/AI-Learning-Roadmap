# 🚀 Kapsamlı Proje: Hiperparametre Laboratuvarı

## 🎯 Hedef

Bu bölümde öğrendiğiniz tüm kavramları (aktivasyon fonksiyonları, ağırlık
başlatma, düzenlileştirme, dropout, batch normalization, öğrenme oranı,
parti boyutu) tek bir sistematik deney çerçevesinde birleştirerek, el
yazısı rakam sınıflandırma probleminde **en iyi performansı veren
hiperparametre kombinasyonunu** bilimsel bir şekilde bulun.

## 📋 Gereksinimler

1. **Veri Hazırlığı**
   - `sklearn.datasets.load_digits()` veri setini kullanın
   - Eğitim/doğrulama/test olarak üçe bölün (örn. %70/%15/%15)
   - Özellikleri standartlaştırın (`StandardScaler`)

2. **Deney Matrisi**
   En az aşağıdaki hiperparametreleri sistematik olarak değiştirin:
   - Mimari (`hidden_layer_sizes`): en az 3 farklı derinlik/genişlik kombinasyonu
   - Aktivasyon fonksiyonu: `relu` ve `tanh`
   - Düzenlileştirme gücü (`alpha`): en az 3 farklı değer
   - Öğrenme oranı (`learning_rate_init`): en az 2 farklı değer

3. **Ölçüm ve Kayıt**
   Her kombinasyon için şunları kaydedin:
   - Eğitim doğruluğu
   - Doğrulama doğruluğu
   - Yakınsama için gereken epok sayısı
   - Aşırı öğrenme göstergesi (eğitim - doğrulama doğruluk farkı)

4. **Analiz**
   - Sonuçları bir `pandas.DataFrame` tablosunda düzenleyin
   - En iyi 3 kombinasyonu doğrulama doğruluğuna göre sıralayın
   - En az 2 görselleştirme oluşturun (örn. alpha'ya karşı doğruluk grafiği,
     mimari derinliğine karşı eğitim süresi grafiği)

5. **Nihai Değerlendirme**
   - Seçtiğiniz "en iyi" modeli test setinde (yalnızca bir kez!) değerlendirin
   - Karışıklık matrisini görselleştirin
   - Hangi rakamların birbirine en çok karıştığını analiz edin

## 📐 Değerlendirme Kriterleri

| Kriter | Açıklama |
|--------|----------|
| Sistematik deney tasarımı | Hiperparametreler döngülerle sistematik olarak test edilmeli, elle tek tek denenmemeli |
| Doğru veri ayrımı | Test seti YALNIZCA son değerlendirmede kullanılmalı |
| Aşırı öğrenme analizi | Eğitim/doğrulama farkı açıkça tartışılmalı |
| Kod kalitesi | Type hint, docstring, PEP 8 uyumu |
| Görselleştirme | En az 2 anlamlı grafik |
| Yazılı sonuç | En iyi modelin NEDEN en iyi olduğuna dair kısa bir analiz |

## 💡 İpuçları

- `itertools.product()` kullanarak hiperparametre kombinasyonlarının
  kartezyen çarpımını otomatik oluşturabilirsiniz.
- Her deneyi aynı `random_state` ile çalıştırarak sonuçların
  karşılaştırılabilir olmasını sağlayın.
- Çok fazla kombinasyon deneme eğiliminde olabilirsiniz — kaliteli bir
  analiz, çok sayıda yüzeysel denemeden daha değerlidir.

## 📤 Teslim Edilecekler

- `hyperparameter_lab.py` veya `hyperparameter_lab.ipynb` (kodun tamamı)
- Sonuç tablosu (CSV veya markdown tablo olarak)
- Kısa bir yazılı sonuç (README veya notebook içinde markdown hücresi olarak)

---

Bu proje, gerçek bir makine öğrenmesi mühendisinin günlük işinin küçük bir
modelidir: bir modeli "bir kere eğitip bitirmek" değil, sistematik
deneylerle en iyi yapılandırmayı bilimsel olarak bulmaktır.
