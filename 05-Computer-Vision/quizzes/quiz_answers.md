# ✅ Bölüm 5 Quiz — Cevap Anahtarı

1. Bir görüntü, bir bilgisayar için (yükseklik, genişlik, kanal sayısı)
   şeklinde bir sayı dizisidir (tensör). Her hücre, o pikselin
   parlaklığını (gri tonlamada) veya renk yoğunluğunu (renkli
   görüntülerde) temsil eden bir sayıdır.

2. Gri tonlamalı bir görüntü 2 boyutludur: `(yükseklik, genişlik)`.
   Renkli bir görüntü 3 boyutludur: `(yükseklik, genişlik, 3)` —
   üçüncü boyut R (kırmızı), G (yeşil), B (mavi) kanallarını temsil eder.

3. 2D evrişim, küçük bir çekirdeği (filtre) görüntü üzerinde kaydırarak
   (sliding window), her konumda çekirdek ile altındaki görüntü
   bölgesinin eleman bazında çarpımını alıp bu çarpımları toplayarak
   tek bir çıktı değeri üretir. Bu işlem, görüntünün her konumunda
   tekrarlanarak bir "özellik haritası" (feature map) oluşturulur.

4. `5 - 3 + 1 = 3` → çıktı 3x3 boyutundadır. Genel formül:
   `çıktı_boyutu = girdi_boyutu - çekirdek_boyutu + 1` (padding olmadan).

5. Bu çekirdekler, komşu piksellere verilen AĞIRLIKLARLA birbirinden
   ayrılır: bulanıklaştırma tüm komşulara eşit pozitif ağırlık verip
   ortalama alır (yumuşatma); keskinleştirme merkez pikseli güçlü
   pozitif, komşuları negatif ağırlıklandırıp kontrastı artırır; kenar
   tespiti merkez ile komşular arasındaki FARKI vurgulayarak ani
   parlaklık değişimlerini (kenarları) ortaya çıkarır.

6. Max Pooling her pencerede EN BÜYÜK değeri tutar — en "baskın"
   özelliği korur, genellikle sınıflandırma görevlerinde tercih edilir.
   Average Pooling her pencerenin ORTALAMASINI alır — daha yumuşak bir
   küçültme sağlar, bazen son katmanlarda (Global Average Pooling)
   kullanılır.

7. Bir nesne veya örüntü görüntüde birkaç piksel kaysa bile, havuzlama
   sonrası çıktının büyük ölçüde AYNI KALMASI anlamına gelir. Bu, ağın
   "kedi kulağı, görüntünün tam olarak şu pikselinde olmalı" gibi aşırı
   katı bir kural öğrenmek yerine, "kedi kulağı BİR YERDE var" gibi daha
   genelleştirilebilir bir örüntü öğrenmesini sağlar.

8. Sobel operatörü, bir görüntüdeki parlaklık DEĞİŞİM HIZINI (gradyanı)
   ölçer. Gx, yatay yöndeki değişimi (dolayısıyla DİKEY kenarları); Gy,
   dikey yöndeki değişimi (dolayısıyla YATAY kenarları) tespit eder.

9. Gradyan büyüklüğü `sqrt(Gx² + Gy²)` formülüyle hesaplanır ve bir
   pikseldeki TOPLAM parlaklık değişim hızını (yöne bakılmaksızın)
   temsil eder — yüksek değerler kenarları, düşük değerler düz
   (homojen) bölgeleri gösterir.

10. Örnekler: (a) yeniden boyutlandırma (tüm görüntüleri sabit boyuta
    getirme), (b) gri tonlama/renk kanalı düzenlemesi, (c) piksel
    değerlerini normalize etme/standartlaştırma ([0,1] veya ortalama-0
    std-1), (d) kırpma (crop). (Herhangi 3'ü.)

11. Veri artırma, modele aynı nesnenin farklı varyasyonlarını (döndürülmüş,
    çevrilmiş, farklı ışıklandırılmış) yapay olarak gösterir; bu, modelin
    eğitim verisindeki spesifik piksel düzenlerini EZBERLEMEK yerine,
    nesnenin GENEL örüntüsünü öğrenmesini teşvik eder — dolayısıyla
    görülmemiş yeni verilere daha iyi genelleme yapar.

12. El yazısı rakamlarda dikey çevirme, bir rakamı BAŞKA (veya geçersiz)
    bir rakama dönüştürebilir (örn. ters çevrilmiş bir "6", bir "9"a
    benzer hale gelebilir veya hiçbir rakama benzemeyebilir) — bu,
    modele yanlış etiketli veri öğretmek anlamına gelir.

13. Evrişim (Convolution) → ReLU (Aktivasyon) → Havuzlama (Pooling).
    Bu üçlü blok, derinlik boyunca tekrarlanır.

14. Flatten işlemi, evrişimli/havuzlama katmanlarının 2 boyutlu (veya
    3 boyutlu, kanallarla birlikte) uzamsal çıktısını, tam bağlantılı
    (MLP tarzı) katmanların işleyebileceği 1 boyutlu bir vektöre
    dönüştürmek için, genellikle mimarinin SONUNA doğru (sınıflandırma
    katmanlarından hemen önce) yapılır.

15. Elle tasarlanmış özellikler (Sobel, HOG gibi), bir İNSAN UZMANIN
    hangi örüntülerin önemli olduğuna önceden karar vermesini ve bunu
    matematiksel olarak kodlamasını gerektirir. CNN'lerin öğrendiği
    özellikler ise eğitim verisinden OTOMATİK olarak, geri yayılım
    yoluyla keşfedilir — insan müdahalesi gerekmez ve genellikle daha
    zengin, probleme özgü örüntüler yakalayabilir.

16. AlexNet, elle tasarlanmış özellik çıkarma yöntemlerine dayanan
    önceki yaklaşımları BÜYÜK bir farkla (hata oranını neredeyse yarı
    yarıya düşürerek) geride bıraktı ve derin öğrenmenin bilgisayarlı
    görüde elle tasarlanmış yöntemlerden üstün olabileceğini ilk kez
    çarpıcı bir şekilde kanıtladı — bu, alanın CNN'lere yönelmesini
    tetikledi.

17. Hesaplama açısından çok pahalıdır: her pencere boyutu ve her konum
    için ayrı bir kontrol/tahmin yapılması gerekir, bu da özellikle
    büyük görüntülerde veya birden fazla nesne boyutu aranırken çok
    yavaş olabilir.

18. Non-Max Suppression, birbirine çok yakın (aynı nesneyi işaret eden)
    birden fazla tespiti, en yüksek güven skoruna sahip TEK bir tespide
    indirger. Bu gereklidir çünkü kaydırmalı pencere/modern dedektörler,
    aynı nesne için genellikle örtüşen birden fazla aday tespit üretir.

19. Bu, CNN'lerin HİYERARŞİK öğrenme yapısından kaynaklanır: her katman,
    bir önceki katmanın çıktısı üzerine inşa edilir. İlk katmanlar ham
    piksellere yakın olduğu için sadece basit yerel örüntüleri (kenar,
    renk geçişi) öğrenebilir; derin katmanlar ise önceki katmanların
    çıktılarını BİRLEŞTİREREK (örn. birkaç kenar bir araya gelip bir
    göz oluşturur) gittikçe daha soyut ve karmaşık örüntüleri temsil
    edebilir.

20. Uygulanabilecek teknikler: (a) veri artırma (data augmentation)
    ekleyerek eğitim setini yapay olarak çeşitlendirmek, (b) ön işleme
    adımlarını (normalizasyon, yeniden boyutlandırma) gözden geçirmek,
    (c) elle tasarlanmış özellikler (Sobel/HOG) eklemek veya CNN
    mimarisini derinleştirmek, (d) Bölüm 4'teki düzenlileştirme
    tekniklerini (dropout, L2, early stopping) uygulamak, (e) daha
    fazla/daha çeşitli eğitim verisi toplamak.
