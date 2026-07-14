"""
08 - Parti (Batch) Boyutunun Etkisi: Toplu / Mini-Parti / Stokastik
=======================================================================

Aynı problemi üç farklı gradyan inişi stratejisiyle eğitir:
  1. Toplu (Batch) Gradyan İnişi -- her adımda TÜM veriyi kullanır
  2. Mini-Parti Gradyan İnişi -- her adımda küçük bir alt küme kullanır (en yaygın)
  3. Stokastik Gradyan İnişi (SGD) -- her adımda TEK bir örnek kullanır

Her stratejinin yakınsama hızını ve kararlılığını (gürültü seviyesini)
karşılaştırır.

Gereksinimler: numpy
"""

import numpy as np


def generate_regression_data(n_samples: int = 200, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-5, 5, (n_samples, 1))
    true_w, true_b = 2.5, -1.0
    y = true_w * X.ravel() + true_b + rng.normal(0, 1, n_samples)
    return X, y


def compute_loss(w: float, b: float, X: np.ndarray, y: np.ndarray) -> float:
    predictions = w * X.ravel() + b
    return float(np.mean((predictions - y) ** 2))


def gradient_step(w: float, b: float, X_batch: np.ndarray, y_batch: np.ndarray, lr: float) -> tuple[float, float]:
    n = len(X_batch)
    predictions = w * X_batch.ravel() + b
    error = predictions - y_batch

    grad_w = (2 / n) * np.sum(error * X_batch.ravel())
    grad_b = (2 / n) * np.sum(error)

    return w - lr * grad_w, b - lr * grad_b


def train(
    X: np.ndarray, y: np.ndarray, batch_size: int, epochs: int, lr: float, seed: int = 0
) -> list[float]:
    """Verilen parti boyutuyla eğitir ve her epoktaki TAM VERİ SETİ
    üzerindeki kaybı kaydeder (adil karşılaştırma için)."""
    rng = np.random.default_rng(seed)
    w, b = 0.0, 0.0
    n = len(X)
    losses = []

    for _ in range(epochs):
        indices = rng.permutation(n)
        X_shuffled, y_shuffled = X[indices], y[indices]

        for start in range(0, n, batch_size):
            X_batch = X_shuffled[start:start + batch_size]
            y_batch = y_shuffled[start:start + batch_size]
            w, b = gradient_step(w, b, X_batch, y_batch, lr)

        losses.append(compute_loss(w, b, X, y))

    return losses


def main() -> None:
    X, y = generate_regression_data()
    epochs = 20

    strategies = {
        "Toplu (Batch, boyut=200)": 200,
        "Mini-Parti (boyut=16)": 16,
        "Stokastik (SGD, boyut=1)": 1,
    }

    print(f"{'Strateji':<28}{'Başlangıç Kaybı':<18}{'Bitiş Kaybı':<15}{'Epok Başına Güncelleme'}")
    print("-" * 85)
    for name, batch_size in strategies.items():
        losses = train(X, y, batch_size=batch_size, epochs=epochs, lr=0.01)
        updates_per_epoch = len(X) // batch_size
        print(f"{name:<28}{losses[0]:<18.4f}{losses[-1]:<15.4f}{updates_per_epoch}")

    print("\nGözlem:")
    print("- Toplu GD: en az gürültülü ama her epokta yalnızca 1 güncelleme yapar (yavaş)")
    print("- SGD: en çok güncellemeyi yapar ama her adımı gürültülüdür")
    print("- Mini-Parti: ikisi arasında bir denge sağlar -- pratikte en çok kullanılan yöntem")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Parti boyutu, hız/kararlılık dengesini doğrudan etkileyen bir
hiperparametredir:
  - Küçük parti boyutu -> daha gürültülü ama daha sık güncelleme, bazen
    yerel minimumlardan "sıçrayarak" kaçmaya yardımcı olur
  - Büyük parti boyutu -> daha kararlı gradyan tahmini ama daha yavaş,
    daha fazla bellek gerektirir
Modern derin öğrenmede tipik olarak 32-256 arası mini-parti boyutları
kullanılır -- bu, GPU paralelleştirmesinden faydalanırken gürültüyü de
makul seviyede tutar.

İyileştirme Fikirleri
-----------------------
1. Farklı parti boyutlarıyla (4, 32, 64, 128) deney yapıp bir grafik çizin.
2. Öğrenme oranını (lr) parti boyutuna göre ayarlamanın (ölçeklemenin)
   etkisini inceleyin.
3. Momentum ekleyerek SGD'nin gürültüsünü nasıl azaltabileceğinizi görün.
"""
