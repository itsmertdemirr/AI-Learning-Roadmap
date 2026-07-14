"""
01 - Tek Bir Nöronun Matematiği
===================================

Bölüm 3'te bir sinir ağını bir bütün olarak eğittik. Burada bir adım
geri gidip TEK BİR nöronun matematiğini -- ağırlıklı toplam, yanlılık
(bias) ve aktivasyon -- her adımı elle takip edebileceğiniz şekilde
inceliyoruz.

Gereksinimler: numpy
"""

import numpy as np


def neuron_forward(
    inputs: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[float, float]:
    """Tek bir nöronun ileri yayılımını adım adım hesaplar.

    Args:
        inputs: Girdi vektörü, örn. [x1, x2, x3].
        weights: Her girdiye karşılık gelen ağırlık vektörü.
        bias: Skaler yanlılık terimi.

    Returns:
        (z, a) ikilisi: z = ham ağırlıklı toplam, a = aktivasyondan sonraki çıktı.
    """
    if inputs.shape != weights.shape:
        raise ValueError("inputs ve weights aynı boyutta olmalıdır")

    z = float(np.dot(inputs, weights) + bias)
    a = 1 / (1 + np.exp(-z))  # sigmoid aktivasyonu
    return z, a


def explain_neuron(inputs: np.ndarray, weights: np.ndarray, bias: float) -> None:
    """Hesaplamanın her adımını okunabilir şekilde yazdırır."""
    print(f"Girdiler (x):    {inputs}")
    print(f"Ağırlıklar (w):  {weights}")
    print(f"Yanlılık (b):    {bias}")

    products = inputs * weights
    print(f"\nAdım 1 — Eleman bazında çarpım (x_i * w_i): {products}")

    weighted_sum = products.sum()
    print(f"Adım 2 — Toplam:                            {weighted_sum:.4f}")

    z = weighted_sum + bias
    print(f"Adım 3 — Yanlılık eklenir (z = toplam + b):  {z:.4f}")

    a = 1 / (1 + np.exp(-z))
    print(f"Adım 4 — Sigmoid aktivasyonu (a = σ(z)):     {a:.4f}")


def main() -> None:
    inputs = np.array([1.0, 0.5, -1.5])
    weights = np.array([0.4, -0.6, 0.3])
    bias = 0.1

    explain_neuron(inputs, weights, bias)

    z, a = neuron_forward(inputs, weights, bias)
    print(f"\nSonuç: z={z:.4f}, aktivasyon_sonrası={a:.4f}")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Bir "katman", paralel çalışan birçok nörondan oluşur (her biri kendi w ve b
değerlerine sahiptir); bir "ağ" ise art arda dizilmiş katmanlardan oluşur.
Karmaşık gibi görünen büyük sinir ağları bile, bu 4 adımın milyonlarca kez
tekrarından ibarettir.

İyileştirme Fikirleri
-----------------------
1. ReLU ve tanh aktivasyonları için de explain_neuron benzeri bir fonksiyon yazın.
2. Aynı girdiler için farklı ağırlık başlangıç değerlerinin çıktıyı nasıl
   değiştirdiğini gözlemleyin.
3. Bias'ı sıfır yaparak etkisinin ne olduğunu inceleyin.
"""
