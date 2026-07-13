# 🤝 Katkıda Bulunma Rehberi

AI-Learning-Roadmap'e katkıda bulunduğunuz için teşekkürler! Bu depo, gerçek bir açık kaynak kursu gibi topluluk katkılarıyla büyümek üzere tasarlandı.

## Nasıl Katkıda Bulunabilirsiniz?

- 🐛 **Hata bildirimi:** Kodda bir hata mı buldunuz? Bir Issue açın.
- 📝 **İçerik ekleme:** 🚧 işaretli bir bölümü tamamlamak ister misiniz? Aşağıdaki standarda bakın.
- 🌍 **Çeviri:** Başka bir dile çeviri eklemek isterseniz yeni bir Issue ile başlayın.
- ✅ **Alıştırma/quiz ekleme:** Mevcut bölümlere ek alıştırma veya quiz sorusu ekleyebilirsiniz.
- 🎨 **Diyagram/görsel:** Mermaid diyagramları veya açıklayıcı görseller her zaman değerlidir.

## İçerik Standardı

Her bölüm klasörü şu yapıyı takip etmelidir (bkz. `01-Rule-Based-AI/` referans örneği):

```
XX-Bolum-Adi/
├── README.md      ← Giriş, Öğrenme Hedefleri, Teori, Diyagramlar, Örnek Tablosu, Özet, Kaynaklar
├── examples/        ← 10-20 çalıştırılabilir Python dosyası
├── exercises/        ← exercises.md (Başlangıç/Orta/İleri seviyeli)
├── solutions/        ← alıştırma çözümleri
├── quizzes/          ← quiz.md + quiz_answers.md (10-20 soru)
├── projects/         ← en az 1 kapsamlı mini proje
├── notebooks/        ← .ipynb sürümü
├── datasets/          ← veri seti veya indirme talimatı
├── images/            ← diyagram/ekran görüntüsü
└── resources/         ← ek okuma listesi
```

## Kod Standartları

- Python 3.10+ hedefleyin
- [PEP 8](https://peps.python.org/pep-0008/) stiline uyun
- Her fonksiyon için **type hint** ve **docstring** kullanın
- Karmaşık mantığı satır içi yorumlarla açıklayın
- Kodu göndermeden önce çalıştırıp test edin

```python
def calculate_area(radius: float) -> float:
    """Bir dairenin alanını hesaplar.

    Args:
        radius: Dairenin yarıçapı (pozitif olmalı).

    Returns:
        Dairenin alanı.
    """
    if radius <= 0:
        raise ValueError("Yarıçap pozitif olmalıdır.")
    return 3.14159 * radius ** 2
```

## Pull Request Süreci

1. Depoyu fork'layın ve bir özellik dalı (branch) oluşturun: `git checkout -b bolum-04-icerik`
2. Değişikliklerinizi yapın ve yerel olarak test edin
3. Varsa `pytest` ile testleri çalıştırın
4. Açıklayıcı bir commit mesajıyla gönderin: `git commit -m "Bölüm 4: ağırlık başlatma örnekleri eklendi"`
5. Bir Pull Request açın ve ne eklediğinizi/değiştirdiğinizi kısaca açıklayın

## Davranış Kuralları

Bu projeye katkıda bulunarak [Davranış Kuralları](CODE_OF_CONDUCT.md)'na uymayı kabul etmiş olursunuz.

## Sorularınız mı var?

Bir Issue açmaktan çekinmeyin — yeni katkıda bulunanlara yardımcı olmaktan memnuniyet duyarız.
