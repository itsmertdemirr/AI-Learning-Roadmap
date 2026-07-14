# ✅ Bölüm 4 Quiz — Cevap Anahtarı

1. (1) Girdiler ile ağırlıkların eleman bazında çarpımı, (2) bu çarpımların
   toplanması, (3) toplama bias eklenmesi (z = ağırlıklı toplam + b), (4)
   z'nin bir aktivasyon fonksiyonundan geçirilmesi (a = aktivasyon(z)).

2. Girdi çok büyük (pozitif veya negatif) olduğunda, sigmoid ve tanh'ın
   çıktısı düz bir bölgeye (sigmoid için ~0 veya ~1, tanh için ~-1 veya ~1)
   "yapışır" ve türevleri neredeyse sıfıra iner — bu da geri yayılım
   sırasında gradyanların çok küçülmesine yol açar.

3. ReLU'nun türevi, pozitif girdiler için tam olarak 1'dir (sigmoid'in
   maksimum türevi olan 0.25'in aksine) — bu, derin ağlarda gradyanların
   katmanlar arasında çarpılırken küçülmesini büyük ölçüde önler ve
   hesaplama olarak da çok daha ucuzdur (basit bir eşikleme işlemi).

4. Bir nöronun ağırlıkları öyle bir hale gelir ki, hemen hemen tüm
   girdiler için z değeri negatif olur; ReLU bu durumda her zaman 0
   çıktısı verir ve türevi de 0 olduğu için bu nöron artık ASLA
   güncellenmez — kalıcı olarak "ölür".

5. Leaky ReLU, negatif girdiler için tam sıfır yerine küçük bir eğim
   (örn. 0.01 * x) verir; bu, gradyanın tamamen sıfırlanmasını önler ve
   ölü nöronların "canlanma" şansı olmasını sağlar.

6. Softmax, çok sınıflı sınıflandırmanın çıktı katmanında kullanılır. Bir
   vektördeki ham skorları (logits), toplamı tam olarak 1 olan ve her
   biri 0 ile 1 arasında olan bir olasılık dağılımına dönüştürür.

7. Xavier/Glorot başlatma, sigmoid ve tanh gibi doygunlaşabilen (sıfır
   merkezli, sınırlı çıktılı) aktivasyon fonksiyonları için tasarlanmıştır.

8. He başlatma, ReLU (ve varyantları) için tasarlanmıştır. Xavier'den
   farkı, ölçeklemenin sadece `n_in`'e (Xavier ise `n_in + n_out`'a)
   dayanmasıdır — çünkü ReLU girdilerin yaklaşık yarısını sıfırlar ve He
   başlatması bu kaybı telafi edecek şekilde varyansı iki katına çıkarır.

9. Geri yayılımda gradyan, zincir kuralı gereği her katmanda bir
   önceki katmanın türeviyle ÇARPILIR. Sığ bir ağda bu çarpım az sayıda
   kez tekrarlanır, ama derin bir ağda (örn. 10+ katman) küçük sayılar
   (örn. 0.25'in altındaki sigmoid türevleri) tekrar tekrar çarpılınca
   sonuç katlanarak küçülür (0.25^10 ≈ 0.000001).

10. L1 cezası ağırlıkların MUTLAK DEĞERLERİNİN toplamını (sum(|w|))
    cezalandırırken, L2 cezası ağırlıkların KARELERİNİN toplamını
    (sum(w²)) cezalandırır. L1'in geometrik şekli (elmas) optimum
    noktanın eksenler üzerinde olmasını teşvik ederken, L2'nin şekli
    (çember) bunu teşvik etmez.

11. L1 düzenlileştirme, önemsiz özelliklere karşılık gelen ağırlıkları
    TAM OLARAK SIFIRA iter (sadece küçültmekle kalmaz). Sıfır ağırlıklı
    özellikler modelden fiilen çıkarılmış olur — bu da L1'i bir özellik
    seçim yöntemi haline getirir.

12. Eğitim sırasında Dropout, her ileri geçişte rastgele bir nöron alt
    kümesini geçici olarak kapatır (çıktılarını sıfırlar), bu da
    nöronların birbirine aşırı bağımlı hale gelmesini önler. Test/çıkarım
    zamanında dropout tamamen kapatılır — ağın tüm nöronları aktif çalışır.

13. Bu ölçekleme ("inverted dropout"), eğitim sırasında aktivasyonların
    beklenen toplam büyüklüğünü korumak içindir. Kapatılmayan nöronları
    1/keep_prob ile büyüterek, kapatılan nöronların kaybının telafi
    edilmesi sağlanır — böylece test zamanında AYRICA bir ölçekleme
    yapmaya gerek kalmaz (aktivasyonlar zaten doğru ölçektedir).

14. Eğitim modunda BatchNorm, o anki mini-partinin KENDİ ortalama ve
    varyansını kullanarak normalize eder. Test modunda ise, eğitim
    boyunca biriktirilen KOŞAN (running) ortalama ve varyans kullanılır
    — çünkü test zamanında (özellikle tek bir örnek için) "mini-parti
    istatistiği" hesaplamak anlamsız veya kararsız olurdu.

15. `gamma` (ölçek) ve `beta` (kaydırma), ağın normalizasyonu tamamen
    veya kısmen "geri almasına" izin veren öğrenilebilir parametrelerdir.
    Bu, BatchNorm'un sadece sabit bir ön işleme adımı değil, öğrenilebilir
    ve esnek bir katman olmasını sağlar.

16. Toplu Gradyan İnişi, her adımda TÜM veri setini kullandığı için
    kararlı ama yavaş gradyan tahminleri üretir (epok başına tek
    güncelleme). SGD ise her adımda tek bir örnek kullandığı için çok
    daha sık ama gürültülü güncellemeler yapar. Temel değiş tokuş:
    kararlılık ile güncelleme sıklığı/hızı arasındadır.

17. Mini-Parti Gradyan İnişi, toplu GD'nin kararlılığı ile SGD'nin
    güncelleme sıklığı arasında pratik bir denge sağlar; ayrıca modern
    donanımın (GPU) parti içindeki paralel hesaplamalardan verimli
    şekilde faydalanmasına olanak tanır.

18. Çok küçük bir öğrenme oranı, eğitimin aşırı yavaş ilerlemesine (veya
    makul sürede hiç yakınsamamasına) yol açar. Çok büyük bir öğrenme
    oranı ise her adımda minimumun "üzerinden atlamaya" ve sonunda kayıp
    değerinin küçülmek yerine büyüyerek ıraksamasına (diverge) yol açar.

19. `early_stopping=True`, eğitim verisinin bir kısmını doğrulama
    (validation) seti olarak ayırır ve doğrulama kaybı belirli sayıda
    epok boyunca iyileşmezse eğitimi otomatik olarak durdurur. Bu, modelin
    eğitim verisini ezberlemeye (aşırı öğrenme) başladığı noktayı
    geçmeden eğitimi sonlandırarak aşırı öğrenmeyi önlemeye yardımcı olur.

20. Bu, klasik bir aşırı öğrenme (overfitting) belirtisidir. Uygulanabilecek
    teknikler: (a) L1/L2 düzenlileştirme ekleyerek/artırarak ağırlıkların
    büyümesini cezalandırmak, (b) Dropout ekleyerek nöronların birbirine
    aşırı bağımlı hale gelmesini önlemek, (c) `early_stopping` kullanarak
    doğrulama kaybı kötüleşmeye başladığında eğitimi durdurmak, (d) model
    mimarisini basitleştirmek (daha az katman/nöron), (e) daha fazla
    eğitim verisi toplamak veya veri artırma (augmentation) uygulamak.
