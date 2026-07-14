"""
02 - Aktivasyon Fonksiyonları Karşılaştırması
=================================================

Sigmoid, Tanh, ReLU, Leaky ReLU ve Softmax aktivasyon fonksiyonlarını
ve türevlerini uygular, ardından farklı girdi aralıklarındaki
davranışlarını karşılaştırır.

Gereksinimler: numpy, matplotlib
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Çıktıyı (0, 1) aralığına sıkıştırır. Kaybolan gradyana eğilimlidir."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    """Çıktıyı (-1, 1) aralığına sıkıştırır. Sigmoid'den daha iyi, sıfır merkezlidir."""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


def relu(x: np.ndarray) -> np.ndarray:
    """Negatifleri sıfırlar, pozitifleri olduğu gibi bırakır. Hızlı ve yaygın."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """ReLU'nun 'ölü nöron' sorununu, negatifler için küçük bir eğim vererek çözer."""
    return np.where(x > 0, x, alpha * x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Bir vektörü, toplamı 1 olan olasılıklara dönüştürür (çok sınıflı çıktı katmanı)."""
    shifted = x - np.max(x)  # sayısal kararlılık için
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def main() -> None:
    x = np.linspace(-6, 6, 200)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(x, sigmoid(x), label="Sigmoid")
    axes[0, 0].plot(x, sigmoid_derivative(x), "--", label="Sigmoid Türevi")
    axes[0, 0].set_title("Sigmoid")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(x, tanh(x), label="Tanh")
    axes[0, 1].plot(x, tanh_derivative(x), "--", label="Tanh Türevi")
    axes[0, 1].set_title("Tanh")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(x, relu(x), label="ReLU")
    axes[1, 0].plot(x, relu_derivative(x), "--", label="ReLU Türevi")
    axes[1, 0].set_title("ReLU")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(x, relu(x), label="ReLU", alpha=0.5)
    axes[1, 1].plot(x, leaky_relu(x), label="Leaky ReLU")
    axes[1, 1].set_title("ReLU vs. Leaky ReLU")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("activation_functions_comparison.png", dpi=100)
    print("Grafik activation_functions_comparison.png dosyasına kaydedildi.")

    # Softmax örneği
    logits = np.array([2.0, 1.0, 0.1])
    probabilities = softmax(logits)
    print("\nSoftmax örneği:")
    print(f"  Ham skorlar (logits): {logits}")
    print(f"  Olasılıklar:          {probabilities.round(4)}")
    print(f"  Toplam:               {probabilities.sum():.4f}  (her zaman 1.0 olmalı)")

    # x=10 civarında sigmoid'in ne kadar "doygun" (gradyanın neredeyse sıfır) olduğunu göster
    print("\nKaybolan gradyan gösterimi (x=10):")
    print(f"  Sigmoid'in türevi x=10'da: {sigmoid_derivative(np.array([10.0]))[0]:.6f}  (neredeyse sıfır!)")
    print(f"  ReLU'nun türevi x=10'da:   {relu_derivative(np.array([10.0]))[0]:.6f}  (tam 1, sorun yok)")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Sigmoid ve Tanh, girdi büyük (pozitif veya negatif) olduğunda "doygunlaşır"
-- türevleri neredeyse sıfıra iner. Derin bir ağda bu, geri yayılım
sırasında gradyanların katman katman küçülüp kaybolmasına yol açar
(Örnek 04'te bunu göreceğiz). ReLU bu sorunu büyük ölçüde önler ve bu
yüzden modern ağlarda varsayılan tercih haline geldi.

İyileştirme Fikirleri
-----------------------
1. ELU ve Swish/SiLU aktivasyonlarını ekleyin ve karşılaştırın.
2. "Ölü ReLU" problemini gösteren bir deney tasarlayın (tüm ağırlıklar
   negatif olduğunda ne olur?).
3. Her aktivasyon fonksiyonunun hangi görev için (gizli katman mı, çıktı
   katmanı mı) daha uygun olduğunu bir tabloya dökün.
"""
