# ✅ Bölüm 1 Quiz — Cevap Anahtarı

1. **Kural Tabanlı Yapay Zeka, çıktı üretmek için açık, insan tarafından
   yazılmış EĞER-O HALDE kurallarına dayanır**; Makine Öğrenmesi ise
   kuralların kendisiyle açıkça programlanmadan verilerden istatistiksel
   olarak desenleri *öğrenir*.

2. Bir **Bilgi Tabanı** (gerçekler + kurallar) ve bir **Çıkarım Motoru**
   (sonuçlara ulaşmak için kuralları gerçeklere uygulayan mantık).

3. Bilgi Tabanı, bir alan hakkındaki uzman bilgisini kodlayan yapılandırılmış
   gerçekler ve EĞER-O HALDE kuralları koleksiyonudur.

4. Çıkarım Motoru, sonuçlara veya eylemlere karar vermek için mevcut
   gerçekleri/girdiyi Bilgi Tabanı'na karşı değerlendiren bileşendir.

5. **İleri zincirleme** bilinen gerçeklerden başlayıp bir sonuca ulaşmak
   için eşleşen kuralları tetikler (örn. "sıcaklık -3°C" → "bir mont
   öner"). **Geri zincirleme** bir hedeften/hipotezden başlayıp gerçeklerin
   bunu destekleyip desteklemediğini kontrol eder (örn. "bu bir soğuk
   algınlığı olabilir mi?" → ateş, öksürük ve boğaz ağrısı var mı diye
   kontrol et).

6. Örnek: **MYCIN** (1970'ler, Stanford) — bakteriyel enfeksiyonları teşhis
   eden ve antibiyotik öneren bir uzman sistem. Diğer geçerli bir cevap:
   kimyasal analiz için kullanılan **DENDRAL**.

7. Avantajlar: (a) tamamen açıklanabilir/şeffaf — hangi kuralın
   tetiklendiğini tam olarak izleyebilirsiniz; (b) eğitim verisi
   gerekmez, kurallar yazılır yazılmaz hemen çalışır; (c) deterministik
   ve öngörülebilir davranış. (Herhangi ikisi.)

8. Dezavantajlar: (a) kurallar binlere ulaştıkça iyi ölçeklenmez (kurallar
   çelişebilir veya yönetilemez hale gelebilir); (b) bir insan kural
   yazarının açıkça öngörmediği desenleri/nüansları ele alamaz; (c) alan
   her değiştiğinde manuel güncellemeler gerektirir. (Herhangi ikisi.)

9. Çelişkili kurallar tutarsız veya kafa karıştırıcı çıktı üretir. Bu,
   **kural öncelikleri/sıralaması**, **çatışma çözüm stratejileri**
   (örn. en spesifik kural kazanır) veya kuralları (örnekte `elif` ile
   yapıldığı gibi) birbirini dışlayacak şekilde yeniden yapılandırarak
   çözülebilir.

10. Kural tabanlı bir sohbet botu yalnızca kendisine açıkça verilen
    anahtar kelimeleri/desenleri eşleştirir — dil, bağlam veya anlam
    hakkında gerçek bir anlayışı yoktur. LLM tabanlı bir sohbet botu ise,
    daha önce hiç görmediği girdiler için bile bağlamsal olarak uygun
    yanıtlar üretmek üzere devasa metin verileri üzerinde eğitilmiş
    istatistiksel bir model kullanır.

11. Bu kontrol olmadan, "negatif bir çekim" gizlenmiş bir yatırma olarak
    istismar edilebilir veya `0`'lık bir işlem hâlâ işlem sürecini
    tetikleyebilir — girdiyi doğrulamak, anlamsız veya istismar
    edilebilir durum değişikliklerini önler.

12. Her durumu bir sonraki duruma eşleyen bir **sözlük (hash map)** — bir
    Sonlu Durum Makinesi'nin geçiş tablosunun doğrudan kod temsili.

13. **Doğru.** İster iç içe `if/else` olarak, ister bir `dict` araması
    olarak, isterse gerçek bir `DecisionTreeClassifier` nesnesi olarak
    yazılmış olsun, mantık verilerden öğrenilmek yerine bir insan
    tarafından elle yazılmışsa, kural tabanlıdır.

14. Vergi hukuku kesin, deterministiktir ve %100 açıklanabilir/denetlenebilir
    olmalıdır — kural tabanlı bir sistem, belirli bir girdinin her zaman
    tam olarak yasal açıdan doğru çıktıyı üretmesini garanti eder; ML'in
    olasılıksal doğası bunu garanti edemez.

15. Geri zincirleme bir **hedeften (hipotezden)** başlar; ileri zincirleme
    **gerçeklerden (bilinen verilerden)** başlar.

16. Kodlanmış kurallar her güncelleme için kod değişikliği ve yeniden
    dağıtım gerektirir, kuralları programcı olmayanların (örn. alan
    uzmanlarının) bakımını yapması daha zordur ve kaynak kodu doğrudan
    düzenlerken hata oluşturma riskini artırır.

17. Kural sırası dikkate alınmazsa, sonraki bir kural, önceki bir kuralın
    amaçlanan eylemini istemeden geçersiz kılabilir veya onunla çelişebilir
    (örn. bir "güvenlik uyarısı: ışıkları aç" kuralından sonra tetiklenen
    bir "enerji tasarrufu için ışıkları kapat" kuralı tehlikeli olurdu).

18. Kural sayısı çok büyüdükçe (yüzlerce/binlerce), kurallar çelişmeye
    başlayabilir, kapsamlı test yapmak zorlaşabilir ve sistem yavaş ve
    bakımı zor hale gelir — buna genellikle "kural patlaması" denir.

19. Örnekler: spam filtrelerinin katı engelleme listeleri, elektronik
    tablo doğrulama mantığı, GPS/trafik işareti sistemleri, sigorta
    poliçe uygunluk kontrolleri, havayolu ücret kural motorları veya
    temel siber güvenlik güvenlik duvarı kuralları.

20. Kural Tabanlı Yapay Zeka, "yapay zeka"nın basitçe karar veren bir
    sistem anlamına geldiği temel fikrini öğretir — istatistiksel
    öğrenmenin karmaşıklığını eklemeden önce bilgi temsili, karar
    mantığı ve çıkarım gibi kavramlar için sezgi oluşturur; bu da Makine
    Öğrenmesi'nin motivasyonlarını ve faydalarını takdir etmeyi çok daha
    kolay hale getirir.
