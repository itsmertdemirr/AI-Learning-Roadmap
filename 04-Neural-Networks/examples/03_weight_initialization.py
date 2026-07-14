"""
03 - Ağırlık Başlatma Stratejileri (Xavier / He)
====================================================

Ağırlıkları yanlış başlatmak, bir ağın hiç öğrenememesine yol açabilir.
Bu örnek üç stratejiyi karşılaştırır: naif (çok büyük) başlatma,
Xavier/Glorot başlatma (sigmoid/tanh için) ve He başlatma (ReLU için).
Her stratejinin katmanlar arasında aktivasyon varyansını nasıl
koruduğunu (veya bozduğunu) gösterir.

Gereksinimler: numpy
"""

import numpy as np


def naive_init(n_in: int, n_out: int, seed: int) -> np.ndarray:
    """Kötü örnek: sabit, ölçeklenmemiş büyük varyanslı başlatma."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n_in, n_out))  # ölçeklenmemiş, çok büyük


def xavier_init(n_in: int, n_out: int, seed: int) -> np.ndarray:
    """Xavier/Glorot başlatma: sigmoid/tanh aktivasyonları için tasarlanmıştır.
    Varyans = 2 / (n_in + n_out) olacak şekilde ölçeklenir."""
    rng = np.random.default_rng(seed)
    limit = np.sqrt(6 / (n_in + n_out))
    return rng.uniform(-limit, limit, (n_in, n_out))


def he_init(n_in: int, n_out: int, seed: int) -> np.ndarray:
    """He başlatma: ReLU aktivasyonları için tasarlanmıştır.
    Varyans = 2 / n_in olacak şekilde ölçeklenir (ReLU'nun yarı girdiyi
    sıfırlamasını telafi eder)."""
    rng = np.random.default_rng(seed)
    std = np.sqrt(2 / n_in)
    return rng.normal(0, std, (n_in, n_out))


def simulate_forward_pass(
    init_fn, activation_fn, n_layers: int = 6, layer_size: int = 100, seed: int = 42
) -> list[float]:
    """n_layers derinliğinde bir ağda, her katmandan sonra aktivasyonların
    standart sapmasını izler. Sağlıklı bir başlatma bu değeri katmanlar
    boyunca kabaca sabit tutar; kötü bir başlatma onu sıfıra çökertir
    veya patlatır."""
    rng = np.random.default_rng(seed)
    activations = rng.normal(0, 1, (1, layer_size))  # başlangıç girdisi
    stds = [float(activations.std())]

    for layer in range(n_layers):
        weights = init_fn(layer_size, layer_size, seed + layer)
        z = activations @ weights
        activations = activation_fn(z)
        stds.append(float(activations.std()))

    return stds


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def main() -> None:
    print("Katman katman aktivasyon standart sapması (0 = tüm nöronlar 'ölü')\n")

    print("Naif başlatma + tanh aktivasyonu:")
    stds = simulate_forward_pass(naive_init, tanh)
    print("  ", [f"{s:.4f}" for s in stds])
    print("  -> Değerler ~1'e doygunlaşıyor (tanh'ın düz bölgesine sıkışıyor);")
    print("     bu, geri yayılımda gradyanların neredeyse sıfırlanmasına yol açar\n")

    print("Xavier başlatma + tanh aktivasyonu:")
    stds = simulate_forward_pass(xavier_init, tanh)
    print("  ", [f"{s:.4f}" for s in stds])
    print("  -> Varyans katmanlar boyunca makul şekilde korunuyor\n")

    print("He başlatma + ReLU aktivasyonu:")
    stds = simulate_forward_pass(he_init, relu)
    print("  ", [f"{s:.4f}" for s in stds])
    print("  -> ReLU ile birlikte varyans stabil kalıyor\n")

    print("Naif başlatma + ReLU aktivasyonu (uyumsuz kombinasyon):")
    stds = simulate_forward_pass(naive_init, relu)
    print("  ", [f"{s:.4f}" for s in stds])
    print("  -> Kontrolsüz büyüme veya çökme riski")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Kural şudur: sigmoid/tanh kullanıyorsanız Xavier/Glorot, ReLU (ve
varyantları) kullanıyorsanız He başlatmasını tercih edin. Modern
framework'ler (PyTorch, Keras) bunu sizin için otomatik yapar, ama
NEDEN yaptığını anlamak, eğitiminiz beklenmedik şekilde ıraksadığında
(diverge) hata ayıklamanıza yardımcı olur.

İyileştirme Fikirleri
-----------------------
1. Katman sayısını (n_layers) 20'ye çıkarın ve naif başlatmanın ne kadar
   hızlı bozulduğunu gözlemleyin.
2. LeCun başlatmasını (SELU aktivasyonu için kullanılır) ekleyin.
3. Aktivasyonların ortalamasını (mean) da izleyerek "ölü ReLU" oranını hesaplayın.
"""
