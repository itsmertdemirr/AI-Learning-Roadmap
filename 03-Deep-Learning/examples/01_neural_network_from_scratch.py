"""
01 - Sıfırdan Sinir Ağı (Sadece NumPy)
==========================================

Klasik XOR problemi üzerinde manuel ileri yayılım, geri yayılım ve gradyan
inişi ile eğitilen minimal 2 katmanlı (1 gizli katmanlı) bir sinir ağı
uygular -- XOR, tek katmanlı bir algılayıcının (perceptron) çözemeyeceği
kadar ünlü şekilde doğrusal olarak ayrılamaz bir problemdir; gizli
katmanların / "derin" öğrenmenin neden önemli olduğu tam olarak budur.

Gereksinimler: sadece numpy (framework yok) -- böylece her gradyanı
görebilirsiniz.
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Aktivasyon fonksiyonu: herhangi bir gerçek sayıyı (0, 1) aralığına sıkıştırır."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(activated: np.ndarray) -> np.ndarray:
    """Sigmoid'in türevi, KENDİ çıktısı cinsinden ifade edilir (yaygın bir hile)."""
    return activated * (1 - activated)


class SimpleNeuralNetwork:
    """Minimal bir ileri beslemeli ağ: girdi -> gizli (sigmoid) -> çıktı (sigmoid)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 1):
        rng = np.random.default_rng(seed)
        # Ağırlık matrisleri, küçük rastgele başlangıç
        self.w1 = rng.normal(0, 1, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = rng.normal(0, 1, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """İleri yayılım: X için ağın tahminini hesaplar."""
        self.z1 = X @ self.w1 + self.b1
        self.a1 = sigmoid(self.z1)          # gizli katman aktivasyonları
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = sigmoid(self.z2)          # çıktı tahmini
        return self.a2

    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray, lr: float) -> None:
        """Geri yayılım: gradyanları hesaplar ve ağırlıkları gradyan inişi ile günceller."""
        n = X.shape[0]

        # Çıktı katmanındaki hata
        error_output = output - y                                   # dKayıp/dÇıktı
        delta_output = error_output * sigmoid_derivative(output)     # zincir kuralı

        # Gizli katmana geri yayılan hata
        error_hidden = delta_output @ self.w2.T
        delta_hidden = error_hidden * sigmoid_derivative(self.a1)

        # Gradyan inişi güncellemeleri (parti üzerinden ortalama gradyan)
        self.w2 -= lr * (self.a1.T @ delta_output) / n
        self.b2 -= lr * np.mean(delta_output, axis=0, keepdims=True)
        self.w1 -= lr * (X.T @ delta_hidden) / n
        self.b1 -= lr * np.mean(delta_hidden, axis=0, keepdims=True)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10000, lr: float = 0.5) -> None:
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)   # Ortalama Kare Hata (MSE) kaybı
            self.backward(X, y, output, lr)

            if epoch % 2000 == 0:
                print(f"Epok {epoch:5d} | Kayıp: {loss:.4f}")


def main() -> None:
    # XOR doğruluk tablosu: doğrusal olarak ayrılamaz -> gizli katman gerektirir
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    net = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    net.train(X, y, epochs=10000, lr=0.5)

    print("\nNihai tahminler:")
    predictions = net.forward(X)
    for inputs, pred, true in zip(X, predictions, y):
        print(f"  Girdi: {inputs} -> Tahmin: {pred[0]:.3f} (yuvarlanmış: {round(pred[0])}) | Gerçek: {true[0]}")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Bu ağda SIFIR el yazısı EĞER-O HALDE kuralı ve SIFIR scikit-learn büyüsü
var -- her sayı, bir matris çarpımı, bir aktivasyon fonksiyonu ve kaybı
azaltmak için ağırlıkları ayarlayan gradyan inişi ile üretiliyor. Bu,
TensorFlow/PyTorch'un devasa ölçekte sizin için tam olarak yaptığı şeydir.
"""
