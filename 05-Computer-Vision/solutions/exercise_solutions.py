"""
Bölüm 5 — Alıştırma Çözümleri (Seçili Alıştırmalar)
========================================================

exercises/exercises.md dosyasındaki hesaplama gerektiren alıştırmaların
referans çözümleri. Önce kendiniz denemeden bakmayın!
"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))


# ---------------------------------------------------------------------
# Çözüm 2 — Dikey çizgileri vurgulayan özel çekirdek
# ---------------------------------------------------------------------
def solution_2_vertical_line_kernel() -> np.ndarray:
    """Sol ve sağ sütunları zıt işaretli yaparak SADECE dikey kenarları
    vurgulayan bir çekirdek. (Bu aslında Sobel-X çekirdeğiyle aynı fikre
    dayanır ama daha basittir.)"""
    return np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ])


# ---------------------------------------------------------------------
# Çözüm 3 — Havuzlama karşılaştırması (elle hesaplama doğrulaması)
# ---------------------------------------------------------------------
def solution_3_pooling_by_hand() -> tuple[np.ndarray, np.ndarray]:
    from importlib import import_module

    pooling_module = import_module("03_pooling_operations")

    matrix = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [7, 2, 8, 3],
        [1, 9, 4, 5],
    ], dtype=float)

    # Elle hesaplama (doğrulama için):
    # Sol üst pencere [[1,3],[5,6]]     -> max=6, avg=3.75
    # Sağ üst pencere [[2,4],[1,2]]     -> max=4, avg=2.25
    # Sol alt pencere [[7,2],[1,9]]     -> max=9, avg=4.75
    # Sağ alt pencere [[8,3],[4,5]]     -> max=8, avg=5.0
    computed_max = pooling_module.max_pool2d(matrix, pool_size=2)
    computed_avg = pooling_module.average_pool2d(matrix, pool_size=2)
    return computed_max, computed_avg


# ---------------------------------------------------------------------
# Çözüm 5 — ImageNet normalizasyonu
# ---------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def solution_5_imagenet_normalize(image_rgb_0_1: np.ndarray) -> np.ndarray:
    """RGB kanallarını ImageNet istatistikleriyle ayrı ayrı normalize eder.
    Girdi: (H, W, 3) şeklinde, [0, 1] aralığında bir görüntü."""
    if image_rgb_0_1.shape[-1] != 3:
        raise ValueError("image_rgb_0_1 son eksende 3 kanal (RGB) içermelidir")
    return (image_rgb_0_1 - IMAGENET_MEAN) / IMAGENET_STD


# ---------------------------------------------------------------------
# Çözüm 7 — "Same padding" destekli evrişim
# ---------------------------------------------------------------------
def solution_7_conv_with_same_padding(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """scipy'nin mode='same' seçeneğiyle, çıktı boyutunu girdiyle aynı
    tutan bir evrişim uygular (kenarlara otomatik sıfır dolgu eklenir)."""
    return convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)


def main() -> None:
    print("Çözüm 2 — Dikey çizgi çekirdeği:")
    print(solution_2_vertical_line_kernel())

    print("\nÇözüm 3 — Havuzlama elle hesaplama doğrulaması:")
    expected_max, expected_avg = solution_3_pooling_by_hand()
    print(f"  Beklenen max pooling:\n{expected_max}")
    print(f"  Beklenen average pooling:\n{expected_avg}")

    print("\nÇözüm 5 — ImageNet normalizasyonu:")
    fake_pixel = np.array([[[0.5, 0.5, 0.5]]])  # 1x1 piksellik test görüntüsü
    normalized = solution_5_imagenet_normalize(fake_pixel)
    print(f"  Girdi piksel: {fake_pixel.flatten()}")
    print(f"  Normalize edilmiş: {normalized.flatten().round(4)}")

    print("\nÇözüm 7 — Same padding ile evrişim:")
    test_image = np.arange(25, dtype=float).reshape(5, 5)
    kernel = np.ones((3, 3)) / 9
    result = solution_7_conv_with_same_padding(test_image, kernel)
    print(f"  Girdi boyutu:  {test_image.shape}")
    print(f"  Çıktı boyutu:  {result.shape}  (girdiyle aynı boyut -- 'same padding' sayesinde)")


if __name__ == "__main__":
    main()
