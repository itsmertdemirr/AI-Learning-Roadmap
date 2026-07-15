# 📝 Bölüm 5 Alıştırmaları — Bilgisayarlı Görü

Alıştırmalar üç zorluk seviyesine ayrılmıştır.

---

## 🟢 Başlangıç Seviyesi

### Alıştırma 1 — Kanal Görselleştirme
`01_image_as_pixels.py`'yi genişleterek, RGB görüntüsünün R, G ve B
kanallarını AYRI AYRI gri tonlamalı görüntüler olarak görselleştirin.
Her kanalın görüntünün farklı yönlerini nasıl vurguladığını
gözlemleyin.

### Alıştırma 2 — Kendi Çekirdeğinizi Tasarlayın
`02_convolution_from_scratch.py`'deki `convolve2d` fonksiyonunu
kullanarak, sadece dikey çizgileri vurgulayan 3x3'lük özel bir çekirdek
tasarlayın (ipucu: sol ve sağ sütunları zıt işaretli yapın).

### Alıştırma 3 — Havuzlama Karşılaştırması
`03_pooling_operations.py`'deki küçük 4x4'lük örnek matrisi elle (kağıt
üzerinde) 2x2 pencerelerle hem max hem average pooling uygulayarak
hesaplayın, ardından kodla doğrulayın.

---

## 🟡 Orta Seviye

### Alıştırma 4 — Canny ile Karşılaştırma
`04_edge_detection_sobel.py`'ye `skimage.feature.canny()` fonksiyonunu
ekleyip Sobel sonucuyla yan yana görselleştirin. Hangi yöntemin daha
"temiz" kenarlar ürettiğini tartışın.

### Alıştırma 5 — ImageNet Normalizasyonu
`05_image_preprocessing_pipeline.py`'ye, ImageNet'in standart
normalizasyon değerlerini (`mean=[0.485, 0.456, 0.406]`,
`std=[0.229, 0.224, 0.225]`) RGB kanallarına ayrı ayrı uygulayan bir
fonksiyon ekleyin.

### Alıştırma 6 — Artırma Etkisini Ölçme
`06_data_augmentation.py`'deki `augment_dataset` fonksiyonunu
kullanarak `08_handcrafted_features_vs_raw_pixels.py`'deki eğitim
setini genişletin ve doğruluğun nasıl değiştiğini ölçün.

### Alıştırma 7 — Padding Ekleme
`07_conv_layer_forward_pass.py`'deki `ConvLayer` sınıfına, çıktı
boyutunu girdiyle AYNI tutan "same padding" desteği ekleyin (ipucu:
`scipy.signal.convolve2d`'nin `mode="same"` parametresini kullanın).

---

## 🔴 İleri Seviye

### Alıştırma 8 — HOG Özellikleri
`08_handcrafted_features_vs_raw_pixels.py`'ye üçüncü bir yaklaşım
olarak `skimage.feature.hog()` özelliklerini ekleyin ve üç yaklaşımı
(ham piksel, Sobel, HOG) karşılaştırın.

### Alıştırma 9 — Çoklu Ölçekli Tespit
`09_sliding_window_object_detection.py`'yi, şablonu 2-3 farklı ölçekte
(örn. %80, %100, %120 boyutunda) yeniden boyutlandırıp her ölçekte
arama yaparak farklı boyutlardaki nesneleri tespit edebilecek şekilde
genişletin ("image pyramid" tekniği).

### Alıştırma 10 — Tam Bir CNN Sınıflandırma Hattı
`07_conv_layer_forward_pass.py`'deki `ConvLayer`'ı kullanarak, TÜM
`digits` veri setindeki her görüntü için özellik haritaları çıkarın
(filtreler rastgele kalabilir), düzleştirin ve bunları
`10_image_classification_project.py`'deki `MLPClassifier`'a besleyin.
Bu "yarı-CNN" yaklaşımının doğruluğunu, ham piksellerle eğitilen temel
modelle karşılaştırın.

---

## 🎯 Mini Proje — "Kendi Görüntü Sınıflandırma Hattınız"

`10_image_classification_project.py`'yi temel alarak:

1. En az 2 farklı özellik çıkarma yöntemi karşılaştırın (ham piksel,
   Sobel, HOG'dan en az ikisi)
2. En az 2 farklı veri artırma seviyesi test edin (`n_augmentations=0,
   2, 5`)
3. Sonuçları bir tabloda özetleyin
4. En iyi kombinasyonun karışıklık matrisini analiz edip hangi
   rakamların en çok karıştığını ve olası nedenlerini yazın

Detaylar için [`projects/`](../projects/) klasörüne bakın.
