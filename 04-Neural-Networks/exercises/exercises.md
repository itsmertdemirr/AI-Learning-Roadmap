# 📝 Bölüm 4 Alıştırmaları — Sinir Ağları Derin İnceleme

Alıştırmalar üç zorluk seviyesine ayrılmıştır. Sırayla ilerlemeniz önerilir
ama deneyiminize göre istediğiniz seviyeden başlayabilirsiniz.

---

## 🟢 Başlangıç Seviyesi

### Alıştırma 1 — Manuel Nöron Hesabı
`01_single_neuron_math.py`'deki `explain_neuron` fonksiyonunu kullanarak,
girdiler `[2.0, -1.0]`, ağırlıklar `[0.5, 0.5]` ve bias `-0.3` için elle
(kağıt üzerinde) z ve sigmoid çıktısını hesaplayın, ardından kodla
doğrulayın.

### Alıştırma 2 — Aktivasyon Fonksiyonu Seçimi
Aşağıdaki senaryoların her biri için hangi aktivasyon fonksiyonunu
seçerdiniz ve neden? (a) İkili sınıflandırmanın çıktı katmanı, (b) çok
sınıflı sınıflandırmanın çıktı katmanı, (c) modern bir derin ağın gizli
katmanları.

### Alıştırma 3 — Öğrenme Oranı Deneyi
`09_learning_rate_effects.py`'yi `lr=0.5`, `lr=0.05` ve `lr=1.5` ile
çalıştırıp her birinin kaç adımda (veya ıraksayıp ıraksamadığını)
gözlemleyin. Sonuçları bir tabloya yazın.

---

## 🟡 Orta Seviye

### Alıştırma 4 — Ölü ReLU Problemi
`03_weight_initialization.py`'yi genişleterek, tüm ağırlıkların BÜYÜK
NEGATİF değerlerle başlatıldığı bir senaryo ekleyin. ReLU aktivasyonuyla
birlikte kullanıldığında "ölü nöron" oranını (aktivasyonu her zaman sıfır
olan nöron yüzdesini) hesaplayın.

### Alıştırma 5 — L1 vs L2 Katsayı Karşılaştırması
`05_l1_l2_regularization.py`'de, eğitilmiş L1 ve L2 modellerinin
katsayılarını (`model.coef_`) yan yana yazdırın. L1'in hangi katsayıları
tam olarak sıfıra ittiğini gözlemleyin ve bunun neden "otomatik özellik
seçimi" olarak adlandırıldığını açıklayın.

### Alıştırma 6 — Dropout Oranının Etkisi
`06_dropout_from_scratch.py`'yi `keep_prob` değerlerini `[0.9, 0.7, 0.5, 0.3]`
olacak şekilde bir döngüde çalıştırın ve her birinde kaç nöronun ortalama
olarak kapatıldığını raporlayın.

### Alıştırma 7 — Parti Boyutu ile Öğrenme Oranı İlişkisi
`08_mini_batch_gradient_descent.py`'de parti boyutunu 4 kat artırdığınızda
(örn. 16'dan 64'e), öğrenme oranını da orantılı artırarak ("linear
scaling rule") benzer bir yakınsama hızı elde edip edemeyeceğinizi test
edin.

---

## 🔴 İleri Seviye

### Alıştırma 8 — BatchNorm'u Derin Ağa Entegre Etme
`04_vanishing_gradient_demo.py`'deki derin sigmoid ağına, her katmandan
sonra `07_batch_normalization.py`'deki `BatchNorm` sınıfını ekleyin.
Gradyan büyüklüklerinin katmanlar boyunca nasıl değiştiğini
BatchNorm'suz versiyonla karşılaştırın.

### Alıştırma 9 — Sıfırdan Momentum Ekleme
`08_mini_batch_gradient_descent.py`'deki `gradient_step` fonksiyonuna
momentum ekleyin (ipucu: önceki gradyanların üstel hareketli
ortalamasını tutan bir `velocity` değişkeni kullanın:
`velocity = beta * velocity + (1 - beta) * gradient`). SGD'nin
salınımının nasıl azaldığını gözlemleyin.

### Alıştırma 10 — Kapsamlı MLP Ayarlaması
`10_mlp_digit_classifier.py`'de aşağıdaki hiperparametre kombinasyonlarını
sistematik olarak test edin ve en iyi test doğruluğunu bulun:
- `hidden_layer_sizes`: `(32,)`, `(64, 32)`, `(128, 64, 32)`
- `alpha`: `1e-5`, `1e-4`, `1e-2`
- `activation`: `"relu"`, `"tanh"`

Sonuçları bir tabloya yazıp en iyi kombinasyonu açıklayın (aşırı öğrenme
belirtisi var mı? early stopping ne zaman devreye giriyor?).

---

## 🎯 Mini Proje — "Hiperparametre Laboratuvarı"

`10_mlp_digit_classifier.py`'yi temel alarak, aşağıdaki gereksinimleri
karşılayan kapsamlı bir deney betiği yazın:

1. En az 3 farklı mimari (`hidden_layer_sizes`) test edin
2. En az 3 farklı düzenlileştirme gücü (`alpha`) test edin
3. Her kombinasyon için eğitim/test doğruluğunu ve kullanılan epok
   sayısını kaydedin
4. Sonuçları bir tablo halinde yazdırın (pandas DataFrame önerilir)
5. En iyi modeli seçip nedenini (sadece en yüksek doğruluk değil, aşırı
   öğrenme/yetersiz öğrenme dengesini de göz önünde bulundurarak) yazılı
   olarak açıklayın

Bu proje kapsamlı bir "kapanış projesi" niteliğindedir — detaylar için
[`projects/`](../projects/) klasörüne bakın.
