"""
04 - Kaybolan Gradyan Problemi Gösterimi
============================================

Derin bir ağda (çok katmanlı), sigmoid aktivasyonu kullanıldığında geri
yayılım sırasında gradyanların katman katman nasıl küçüldüğünü (ve
sonunda pratik olarak sıfırlandığını) doğrudan sayısal olarak gösterir.
Aynı derinlikteki bir ReLU ağıyla karşılaştırır.

Gereksinimler: numpy
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    return a * (1 - a)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative(a: np.ndarray) -> np.ndarray:
    return (a > 0).astype(float)


def build_deep_network(
    n_layers: int, layer_size: int, seed: int, weight_std: float = 0.5
) -> list[np.ndarray]:
    """n_layers katmanlı ağırlık matrislerinden oluşan bir ağ üretir.

    weight_std, ağırlıkların ölçeğini kontrol eder: sigmoid ağı için sabit
    (naif) bir ölçek kullanılır (doygunlaşmayı/kaybolmayı göstermek için),
    ReLU ağı için ise He-tarzı sqrt(2/n) ölçeği kullanılır (adil bir
    karşılaştırma için -- aksi halde ReLU da patlar).
    """
    rng = np.random.default_rng(seed)
    return [rng.normal(0, weight_std, (layer_size, layer_size)) for _ in range(n_layers)]


def measure_gradient_magnitude(
    weights: list[np.ndarray], activation_fn, derivative_fn, seed: int = 0
) -> list[float]:
    """Sondan başa doğru (geri yayılım gibi) gradyan büyüklüğünün nasıl
    değiştiğini simüle eder ve her katmandaki ortalama mutlak gradyanı döndürür."""
    rng = np.random.default_rng(seed)
    layer_size = weights[0].shape[0]

    # İleri geçiş: her katmandaki aktivasyonu sakla
    activations = [rng.normal(0, 1, (1, layer_size))]
    for w in weights:
        z = activations[-1] @ w
        activations.append(activation_fn(z))

    # Geri geçiş: çıktıdan girdiye doğru gradyan büyüklüğünü izle
    grad = np.ones((1, layer_size))  # çıktı katmanındaki başlangıç gradyanı
    magnitudes = [float(np.abs(grad).mean())]

    for i in reversed(range(len(weights))):
        grad = (grad * derivative_fn(activations[i + 1])) @ weights[i].T
        magnitudes.append(float(np.abs(grad).mean()))

    magnitudes.reverse()  # girdiden çıktıya doğru sıraya çevir
    return magnitudes


def main() -> None:
    n_layers = 10
    layer_size = 50

    print(f"{n_layers} katmanlı bir ağda, girdi katmanından çıktı katmanına\n"
          f"gradyan büyüklüğünün nasıl değiştiğini karşılaştıralım:\n")

    sigmoid_weights = build_deep_network(n_layers, layer_size, seed=1, weight_std=0.5)
    sigmoid_grads = measure_gradient_magnitude(sigmoid_weights, sigmoid, sigmoid_derivative)

    he_std = np.sqrt(2 / layer_size)  # He başlatma ölçeği -- ReLU için doğru seçim
    relu_weights = build_deep_network(n_layers, layer_size, seed=1, weight_std=he_std)
    relu_grads = measure_gradient_magnitude(relu_weights, relu, relu_derivative)

    print("Katman  | Sigmoid Gradyanı | ReLU Gradyanı")
    print("-" * 48)
    for i, (sg, rg) in enumerate(zip(sigmoid_grads, relu_grads)):
        print(f"{i:>6}  | {sg:>16.8f} | {rg:>13.6f}")

    print("\nSigmoid ağında girdi katmanındaki gradyan, çıktı katmanındaki")
    print(f"gradyanın SADECE %{(sigmoid_grads[0] / sigmoid_grads[-1] + 1e-12) * 100:.6f}'i kadardır.")
    print("Bu, girdiye yakın katmanların pratik olarak HİÇ öğrenemediği anlamına gelir!")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Bu, 1990'larda-2000'lerin başında derin ağların eğitilmesini neredeyse
imkansız hale getiren klasik "kaybolan gradyan" problemidir. Sigmoid'in
türevinin maksimum değeri 0.25'tir -- bunu 10 kez zincir kuralıyla çarpınca
(0.25^10 ≈ 0.00000095) gradyan pratik olarak sıfırlanır. ReLU'nun türevi
ya 0 ya da tam 1 olduğu için bu çarpma problemi büyük ölçüde ortadan
kalkar. Bu, ReLU'nun neden derin ağlarda standart hale geldiğinin
matematiksel kanıtıdır.

İyileştirme Fikirleri
-----------------------
1. Katman sayısını 20'ye çıkarıp farkın nasıl büyüdüğünü gözlemleyin.
2. Batch Normalization eklemenin (Örnek 07) bu problemi nasıl azalttığını inceleyin.
3. "Patlayan gradyan" (exploding gradient) senaryosunu -- büyük ağırlıklarla -- simüle edin.
"""
