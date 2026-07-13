# Bölüm 04 — Sinir Ağları Derin İnceleme

[⬅ Önceki: Derin Öğrenme](../03-Deep-Learning/README.md) | [⬅ Yol Haritası](../README.md) | [➡ Sonraki: Bilgisayarlı Görü](../05-Computer-Vision/README.md)

---

| 🎯 Zorluk | ⏱️ Tahmini Süre | 📋 Ön Koşullar | 🏆 Kazanımlar |
|---|---|---|---|
| Orta–İleri | 6–8 saat | Bölüm 3 (Derin Öğrenme temelleri) | Ağırlık başlatma, düzenlileştirme, hiperparametre ayarı, PyTorch/TensorFlow ile MLP |

## 📖 Durum: 🚧 Yapı Hazır — İçerik Ekleniyor

Bu bölüm, [Bölüm 1](../01-Rule-Based-AI/README.md) ile aynı "altın standart" yapıyı izleyecek şekilde planlanmıştır:
Giriş → Öğrenme Hedefleri → Teori (Mermaid diyagramlarıyla) →
10-20 Python Örneği → Başlangıç/Orta/İleri Alıştırmalar → Çözümler → Quiz →
Notebook → Kapsamlı Proje(ler) → Özet → Kaynaklar.

Katkılarınızı bekliyoruz — nasıl katkıda bulunacağınızı öğrenmek için ana dizindeki [CONTRIBUTING.md](../CONTRIBUTING.md) dosyasına bakın.

## 🎯 Planlanan Konular

- [ ] Bir nöronun tam matematiksel detaylarla anatomisi
- [ ] Ağırlık başlatma stratejileri (Xavier, He)
- [ ] Kaybolan / patlayan gradyanlar
- [ ] Düzenlileştirme (Regularization): Dropout, L1/L2, Batch Normalization
- [ ] Hiperparametre ayarı (öğrenme oranı, parti boyutu, epok sayısı)
- [ ] PyTorch/TensorFlow ile çok katmanlı algılayıcılar inşa etmek

## 🚀 Planlanan Kapsamlı Proje(ler)

- [ ] MNIST el yazısı rakam sınıflandırıcı (MLP ile)
- [ ] Hiperparametre arama deneyi (grid/random search)

## 📁 Klasör Yapısı (standart — tüm bölümlerde aynı)

```
04-Neural-Networks/
├── README.md          ← tam bölüm içeriği (teori + diyagramlar + tablo)
├── examples/            ← 10-20 çalıştırılabilir, PEP 8 uyumlu Python örneği
├── exercises/            ← başlangıç/orta/ileri seviye alıştırmalar
├── solutions/            ← alıştırma çözümleri
├── quizzes/              ← quiz.md + quiz_answers.md
├── projects/             ← kapsamlı mini/gerçek dünya projeleri
├── notebooks/            ← etkileşimli Jupyter Notebook (.ipynb) sürümleri
├── datasets/              ← örnek veri setleri veya indirme talimatları
├── images/                ← diyagramlar, ekran görüntüleri, GIF'ler
└── resources/             ← ek okumalar, kopya kağıtları, makale linkleri
```

---

[⬅ Önceki: Derin Öğrenme](../03-Deep-Learning/README.md) | [⬅ Yol Haritası](../README.md) | [➡ Sonraki: Bilgisayarlı Görü](../05-Computer-Vision/README.md)
