"""
06 - Sıfırdan Dropout Uygulaması
====================================

Dropout, eğitim sırasında her ileri geçişte rastgele bir nöron alt
kümesini geçici olarak "kapatarak" nöronların birbirine aşırı bağımlı
hale gelmesini (co-adaptation) önleyen popüler bir düzenlileştirme
tekniğidir. Bu örnek, "Inverted Dropout" yöntemini (PyTorch/TensorFlow'un
kullandığı standart yaklaşım) sıfırdan uygular.

Gereksinimler: numpy
"""

import numpy as np


def dropout_forward(
    activations: np.ndarray, keep_prob: float, training: bool, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Bir aktivasyon katmanına Inverted Dropout uygular.

    Args:
        activations: Katmanın çıktısı, şekil (batch_size, n_units).
        keep_prob: Bir nöronun AKTİF KALMA olasılığı (örn. 0.8 = %20 kapatma).
        training: True ise dropout uygulanır; False ise (test/çıkarım
            zamanı) aktivasyonlar hiç değiştirilmeden döner.
        seed: Tekrarlanabilirlik için rastgelelik tohumu.

    Returns:
        (çıktı, maske) ikilisi. Maske, geri yayılımda aynı nöronları
        kapatmak için saklanır.
    """
    if not (0 < keep_prob <= 1):
        raise ValueError("keep_prob (0, 1] aralığında olmalıdır")

    if not training:
        # Test zamanında dropout uygulanmaz -- tüm nöronlar aktiftir.
        return activations, np.ones_like(activations)

    rng = np.random.default_rng(seed)
    # Her nöron keep_prob olasılıkla 1 (açık), (1-keep_prob) olasılıkla 0 (kapalı)
    mask = (rng.uniform(size=activations.shape) < keep_prob).astype(float)

    # "Inverted" kısmı: kapatılmayan nöronları 1/keep_prob ile ölçekleyerek
    # beklenen toplam aktivasyon büyüklüğünü koruruz. Bu sayede test
    # zamanında EK bir ölçekleme yapmaya gerek kalmaz.
    output = (activations * mask) / keep_prob
    return output, mask


def main() -> None:
    rng = np.random.default_rng(0)
    activations = rng.uniform(0.5, 1.5, size=(1, 10))
    print(f"Orijinal aktivasyonlar:        {activations.round(3)}")

    # Eğitim modunda, %30 dropout oranıyla (keep_prob=0.7)
    train_output, mask = dropout_forward(activations, keep_prob=0.7, training=True, seed=1)
    print(f"Eğitim modu (dropout aktif):   {train_output.round(3)}")
    print(f"Kullanılan maske:               {mask}")
    print(f"Kapatılan nöron sayısı:          {int((mask == 0).sum())}/10")

    # Test modunda dropout uygulanmaz
    test_output, _ = dropout_forward(activations, keep_prob=0.7, training=False)
    print(f"\nTest modu (dropout kapalı):    {test_output.round(3)}")
    print("(test modunda aktivasyonlar hiç değişmez -- orijinaliyle aynı)")

    # Beklenen değerin korunduğunu doğrula
    n_trials = 10000
    total = np.zeros_like(activations)
    for i in range(n_trials):
        out, _ = dropout_forward(activations, keep_prob=0.7, training=True, seed=i)
        total += out
    average = total / n_trials
    print(f"\n{n_trials} denemenin ortalaması: {average.round(3)}")
    print(f"Orijinal değerler:              {activations.round(3)}")
    print("(Bu ikisi birbirine çok yakın olmalı -- 'inverted' ölçekleme")
    print(" beklenen aktivasyon büyüklüğünü koruduğu için)")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Dropout, her eğitim adımında farklı bir "alt ağ" eğitiyormuş gibi davranır
-- bu da modelin tek bir nörona veya nöron grubuna aşırı güvenmesini
engeller ve daha sağlam (robust), daha az aşırı öğrenen bir model üretir.
Test zamanında dropout KAPALIDIR; tüm ağ tam kapasiteyle çalışır.

İyileştirme Fikirleri
-----------------------
1. keep_prob'u 0.5'e düşürüp modelin ne kadar daha "gürültülü" hale
   geldiğini gözlemleyin.
2. Bu fonksiyonu Bölüm 3'teki sıfırdan sinir ağına entegre edin ve XOR
   probleminde eğitim/test kaybını karşılaştırın.
3. DropConnect (nöronlar yerine ayrı ayrı bağlantıları kapatan bir
   varyant) hakkında araştırma yapın.
"""
