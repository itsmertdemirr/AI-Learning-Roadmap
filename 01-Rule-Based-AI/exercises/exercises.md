# 📝 Bölüm 1 Alıştırmaları — Kural Tabanlı Yapay Zeka

Bunları sırayla çalışın. Her biri bölümden bir kavramın üzerine inşa edilir.

---

### Alıştırma 1 — Hava Durumu Asistanını Genişletin
`01_weather_assistant.py` dosyasına yüksek UV endeksi (`uv_index >= 8`) için
güneş kremi önerisiyle uyaran yeni bir kural ekleyin.

**İpucu:** `WeatherReading`'e yeni bir alan ve `get_weather_advice`'a yeni bir kural bloğu ekleyin.

---

### Alıştırma 2 — Oran Sınırlı Kilit Açma Süresi Ekleyin
`02_login_system.py` dosyasında kilitli hesaplar sonsuza kadar kilitli kalır.
Belirli sayıda "tık" (bir sayaçla simüle edin) geçtikten sonra hesabın
otomatik olarak kilidinin açılması için bir kural ekleyin.

---

### Alıştırma 3 — İfade Hesap Makinesi
`03_calculator.py`'yi, operatör önceliğine uyarak `"12 + 5 * 2"` gibi tam bir
dizeyi ayrıştırıp değerlendirecek şekilde genişletin (toplamadan önce çarpma).

**İpucu:** `eval()` kullanmanıza gerek yok — dizeyi bölüp iki geçişte kurallar
uygulamayı deneyin (önce `*` ve `/`, sonra `+` ve `-`).

---

### Alıştırma 4 — Güven Puanlaması
`04_medical_symptom_checker.py`'de `diagnose()` fonksiyonunu, tam bir alt küme
gerektirmek yerine kısmi eşleşmeler için bile bir güven yüzdesi döndürecek
şekilde değiştirin: `(eşleşen semptomlar / kuraldaki toplam semptom) * 100`.

---

### Alıştırma 5 — Yeni Bir Restoran Kategorisi Ekleyin
`05_restaurant_recommendation.py`'ye kendi puanlama kurallarına sahip bir
"Sokak Lezzetleri" kategorisi ekleyin (ipucu: `budget="low"` ve
`mood="casual"` için iyi puan almalı).

---

### Alıştırma 6 — Ağırlıklı Notlandırma
`06_student_grading.py`'yi, `grade_student()` fonksiyonunun bir bileşen puanı
sözlüğü kabul edecek şekilde değiştirin, örn.
`{"homework": 90, "midterm": 75, "final": 85}` ağırlıklarla `%20`, `%30`,
`%50`, ağırlıklı ortalamayı hesaplayıp ardından mevcut not kurallarını
uygulayın.

---

### Alıştırma 7 — Yaya Butonu
`07_traffic_light_controller.py`'ye, mevcut durumdan bağımsız olarak ışığı
tek bir geçişte `RED` (kırmızı) yapmaya zorlayan bir
`request_pedestrian_crossing()` metodu ekleyin.

---

### Alıştırma 8 — İşlem Kaydı
`08_atm_simulation.py`'yi, her başarılı çekim/yatırma işleminin
`Account` üzerindeki bir `transactions: list[str]` listesine eklenecek
şekilde değiştirin ve `main()`'in sonunda mini bir ekstre yazdırın.

---

### Alıştırma 9 — Kural Önceliği
`09_smart_home_automation.py`'de, güvenlik uyarısı kuralı (Kural 3),
değerlendirildiği sıradan bağımsız olarak her zaman İLK yazdırılmalıdır.
Kural önceliklerini destekleyecek şekilde `evaluate_rules()`'ı yeniden
düzenleyin.

---

### Alıştırma 10 — Sohbet Botu için Bağlam Belleği
`10_rule_based_chatbot.py`'yi, kullanıcı "adım X" dediğinde botun X'i
hatırlaması ve sonraki yanıtlarda kullanması için değiştirin
(örn. "Tekrar merhaba, X!").

---

## 🎯 Mini Proje — "Kişisel Kural Tabanlı Asistan"

Yukarıdaki örneklerden en az **üçünü** tek bir komut satırı programında
birleştirin:

- Kullanıcıyı selamlayın (sohbet botu kuralları)
- Hava durumunu sorun ve tavsiye verin (hava durumu asistanı kuralları)
- Ne yemek istediklerini sorun ve bir restoran kategorisi önerin

Gereksinimler:
- Tek bir `.py` dosyası olmalı
- Fonksiyonlar kullanılmalı (satır içi tekrarlanan mantık olmamalı)
- Geçersiz girdiyi zarifçe ele almalı (çökme olmamalı)
- Birleştirilmiş mantık genelinde en az 5 kural içermeli

Bu proje kasıtlı olarak açık uçludur — tek bir doğru çözüm yoktur. Bu
bölümdeki örnekler gibi temiz, kural tabanlı mantık yazmaya odaklanın.
