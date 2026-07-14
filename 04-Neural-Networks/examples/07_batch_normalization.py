"""
07 - Sıfırdan Batch Normalization
=====================================

Batch Normalization (BatchNorm), her mini-partideki (mini-batch)
aktivasyonları normalize ederek (ortalama 0, varyans 1 yapıp, ardından
öğrenilebilir bir ölçek ve kaydırma uygulayarak) eğitimi hızlandırır ve
kararlılığını artırır. Bu, kaybolan/patlayan gradyan problemine karşı
güçlü bir savunmadır (Örnek 04 ile karşılaştırın).

Gereksinimler: numpy
"""

import numpy as np


class BatchNorm:
    """Basitleştirilmiş, tek bir katman için Batch Normalization uygulaması."""

    def __init__(self, n_features: int, epsilon: float = 1e-5, momentum: float = 0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        # Öğrenilebilir parametreler: gamma (ölçek) ve beta (kaydırma)
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        # Çıkarım (test) zamanı için koşan (running) istatistikler
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """x şekli: (batch_size, n_features)."""
        if training:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)

            # Normalize et: ortalama 0, varyans 1 yap
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)

            # Koşan istatistikleri güncelle (test zamanında kullanılacak)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            # Test zamanında, mini-parti istatistikleri yerine EĞİTİM
            # boyunca biriktirilen koşan istatistikler kullanılır.
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # Öğrenilebilir ölçek ve kaydırma uygula -- ağın gerekirse
        # normalize etmeyi "geri almasına" izin verir
        return self.gamma * x_normalized + self.beta


def main() -> None:
    rng = np.random.default_rng(42)

    # Kasıtlı olarak KÖTÜ ölçekli bir mini-parti üret (farklı özellikler
    # çok farklı büyüklüklerde -- gerçek dünyada sık karşılaşılan bir durum)
    batch = np.column_stack([
        rng.normal(100, 50, 20),   # büyük ölçekli özellik
        rng.normal(0, 0.01, 20),   # çok küçük ölçekli özellik
        rng.normal(-5, 2, 20),     # orta ölçekli özellik
    ])

    print("BatchNorm ÖNCESİ istatistikler (özellik başına):")
    print(f"  Ortalama: {batch.mean(axis=0).round(3)}")
    print(f"  Std:      {batch.std(axis=0).round(3)}")

    bn = BatchNorm(n_features=3)
    normalized = bn.forward(batch, training=True)

    print("\nBatchNorm SONRASI istatistikler (özellik başına):")
    print(f"  Ortalama: {normalized.mean(axis=0).round(3)}   (hedef: ~0)")
    print(f"  Std:      {normalized.std(axis=0).round(3)}   (hedef: ~1)")

    print("\nTüm özellikler artık benzer ölçekte -- bu, gradyan inişinin")
    print("çok daha kararlı ve hızlı yakınsamasını (converge) sağlar.")

    # Test zamanı davranışını göster (koşan istatistiklerle)
    test_sample = np.array([[110, 0.005, -4]])
    test_output = bn.forward(test_sample, training=False)
    print(f"\nTest zamanı çıkarımı (koşan istatistiklerle): {test_output.round(3)}")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
BatchNorm'un iki farklı davranış modu vardır:
  - EĞİTİM: her mini-partinin KENDİ ortalama/varyansını kullanır
  - TEST: eğitim boyunca biriktirilen KOŞAN (running) ortalama/varyansı kullanır
Bu ayrım kritiktir -- test zamanında tek bir örnek için "mini-parti
istatistiği" hesaplamak anlamsız olurdu. `gamma` ve `beta` parametreleri,
ağın gerekirse normalizasyonu kısmen veya tamamen "geri almasını" sağlar;
bu da BatchNorm'u sadece bir ön işleme adımı değil, öğrenilebilir bir
katman yapar.

İyileştirme Fikirleri
-----------------------
1. Bu BatchNorm katmanını Örnek 04'teki derin ağa ekleyip gradyan
   büyüklüğünün nasıl değiştiğini gözlemleyin.
2. Layer Normalization'ı (mini-parti yerine tek örnek üzerinde normalize
   eden, Transformer'larda yaygın kullanılan varyant) araştırın.
3. `momentum` değerini değiştirip koşan istatistiklerin ne kadar hızlı
   güncellendiğini gözlemleyin.
"""
