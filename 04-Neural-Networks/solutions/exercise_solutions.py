"""
Bölüm 4 — Alıştırma Çözümleri (Seçili Alıştırmalar)
========================================================

exercises/exercises.md dosyasındaki hesaplama gerektiren alıştırmaların
referans çözümleri. Önce kendiniz denemeden bakmayın!
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))


# ---------------------------------------------------------------------
# Çözüm 1 — Manuel nöron hesabı
# ---------------------------------------------------------------------
def solution_1() -> tuple[float, float]:
    inputs = np.array([2.0, -1.0])
    weights = np.array([0.5, 0.5])
    bias = -0.3

    z = float(np.dot(inputs, weights) + bias)  # (2*0.5) + (-1*0.5) - 0.3 = 0.2
    a = 1 / (1 + np.exp(-z))
    return z, a


# ---------------------------------------------------------------------
# Çözüm 4 — Ölü ReLU problemi
# ---------------------------------------------------------------------
def solution_4(layer_size: int = 100, seed: int = 42) -> float:
    """Büyük negatif bir bias ile başlatılan bir ReLU katmanında 'ölü
    nöron' (aktivasyonu her zaman sıfır olan nöron) oranını hesaplar.

    Not: Sadece ağırlıkları negatif yapmak yetmez -- girdi de değişken
    işaretli olduğunda z'nin ortalaması yine sıfıra yakın kalabilir.
    Nöronları güvenilir şekilde 'öldürmek' için büyük bir negatif bias
    ekliyoruz; bu, gerçek dünyada kötü bir başlatma + kötü bir öğrenme
    oranı güncellemesinin bir araya gelmesiyle de ortaya çıkabilir.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (1, layer_size))
    weights = rng.normal(0, 1, (layer_size, layer_size))
    large_negative_bias = -20.0  # z'yi neredeyse her zaman negatif yapacak kadar büyük

    z = x @ weights + large_negative_bias
    activations = np.maximum(0, z)  # ReLU

    dead_ratio = float((activations == 0).mean())
    return dead_ratio


# ---------------------------------------------------------------------
# Çözüm 6 — Dropout oranının etkisi
# ---------------------------------------------------------------------
def solution_6() -> dict[float, float]:
    from importlib import import_module

    dropout_module = import_module("06_dropout_from_scratch")

    rng = np.random.default_rng(0)
    activations = rng.uniform(0.5, 1.5, size=(1, 20))

    results = {}
    for keep_prob in [0.9, 0.7, 0.5, 0.3]:
        _, mask = dropout_module.dropout_forward(activations, keep_prob, training=True, seed=1)
        dropped_ratio = float((mask == 0).mean())
        results[keep_prob] = dropped_ratio
    return results


def main() -> None:
    z, a = solution_1()
    print(f"Çözüm 1: z={z:.4f}, sigmoid(z)={a:.4f}")

    dead_ratio = solution_4()
    print(f"Çözüm 4: Kötü başlatılmış ReLU katmanında ölü nöron oranı: %{dead_ratio * 100:.1f}")

    results = solution_6()
    print("Çözüm 6: keep_prob -> gerçekte kapatılan oran")
    for keep_prob, dropped in results.items():
        print(f"  keep_prob={keep_prob} -> %{dropped * 100:.0f} kapatıldı (beklenen: %{(1 - keep_prob) * 100:.0f})")


if __name__ == "__main__":
    main()
