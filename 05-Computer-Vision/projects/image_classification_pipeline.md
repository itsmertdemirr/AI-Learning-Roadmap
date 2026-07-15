# 🚀 Kapsamlı Proje: Görüntü Sınıflandırma Hattı Optimizasyonu

## 🎯 Hedef

Bu bölümde öğrendiğiniz tüm teknikleri (ön işleme, evrişim tabanlı özellik
çıkarma, veri artırma, elle tasarlanmış özellikler) sistematik olarak
değerlendirerek, el yazısı rakam sınıflandırma probleminde **en yüksek
performansı veren pipeline'ı bilimsel bir şekilde bulun.**

## 📋 Gereksinimler

1. **Temel (Baseline) Model**
   - `sklearn.datasets.load_digits()` ile ham piksellerle eğitilen bir
     temel model kurun ve doğruluğunu kaydedin.

2. **Özellik Mühendisliği Deneyleri**
   En az 3 farklı özellik çıkarma stratejisini karşılaştırın:
   - Ham pikseller (temel)
   - Sobel kenar özellikleri (`08_handcrafted_features_vs_raw_pixels.py`'den)
   - HOG özellikleri (`skimage.feature.hog`)
   - (İsteğe bağlı) Rastgele başlatılmış CNN özellik haritaları
     (`07_conv_layer_forward_pass.py`'den)

3. **Veri Artırma Deneyleri**
   En az 3 farklı artırma seviyesini test edin:
   - Artırma yok
   - Orta seviye artırma (`n_augmentations=2`)
   - Yoğun artırma (`n_augmentations=5`)

4. **Kombinasyon Matrisi**
   Özellik stratejisi × artırma seviyesi kombinasyonlarının TAMAMINI
   sistematik olarak test edin (en az 3×3 = 9 kombinasyon) ve sonuçları
   bir tabloda özetleyin.

5. **Hata Analizi**
   En iyi modelin karışıklık matrisini oluşturup:
   - Hangi rakamların en çok birbirine karıştığını belirleyin
   - Yanlış sınıflandırılan 5 örneği görselleştirin ve neden yanlış
     sınıflandırıldığını tartışın

## 📐 Değerlendirme Kriterleri

| Kriter | Açıklama |
|--------|----------|
| Sistematik deney tasarımı | Tüm kombinasyonlar döngülerle test edilmeli |
| Doğru veri ayrımı | Test seti yalnızca son değerlendirmede kullanılmalı |
| Görselleştirme | En az 3 anlamlı grafik/görsel (karışıklık matrisi dahil) |
| Kod kalitesi | Type hint, docstring, PEP 8 uyumu |
| Yazılı analiz | Hangi kombinasyonun neden en iyi sonucu verdiğine dair açıklama |

## 💡 İpuçları

- Her deneyi aynı `random_state` ile çalıştırarak sonuçların
  karşılaştırılabilir olmasını sağlayın.
- Veri artırma SADECE eğitim setine uygulanmalı, test setine ASLA
  uygulanmamalıdır (aksi halde test seti "sızıntısı" / veri sızıntısı
  olur).
- Çalışma süresini kısaltmak için önce küçük bir alt kümede (örn. 200
  örnek) hızlı bir prototip test edin, sonra tam veri setiyle çalıştırın.

## 📤 Teslim Edilecekler

- `image_classification_pipeline.py` veya `.ipynb` (kodun tamamı)
- Sonuç matrisi (9+ kombinasyon, doğruluk değerleriyle)
- En iyi modelin karışıklık matrisi görseli
- Kısa bir yazılı sonuç (en iyi kombinasyon ve nedenleri)

---

Bu proje, "daha karmaşık her zaman daha iyidir" varsayımını sorgulamanızı
sağlar — bazen basit bir ham piksel modeli, karmaşık özellik
mühendisliğiyle neredeyse aynı performansı gösterebilir. Bunu varsaymak
yerine ÖLÇMEK, gerçek bir bilgisayarlı görü mühendisinin işidir.
