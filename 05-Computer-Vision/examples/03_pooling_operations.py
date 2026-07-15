"""
03 - Sıfırdan Havuzlama (Pooling) İşlemleri
===============================================

Max Pooling ve Average Pooling'i sıfırdan uygular. Havuzlama, bir
özellik haritasının boyutunu küçültürken (hesaplama maliyetini azaltır)
en önemli bilgiyi korumaya çalışır -- CNN mimarilerinde evrişim
katmanları arasına yerleştirilen standart bir bileşendir.

Gereksinimler: numpy, matplotlib, scikit-image
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, data


def max_pool2d(feature_map: np.ndarray, pool_size: int = 2, stride: int | None = None) -> np.ndarray:
    """Bir özellik haritasına max pooling uygular: her pencerede en
    büyük değeri tutar (en 'baskın' özelliği korur)."""
    stride = stride or pool_size
    h, w = feature_map.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1

    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = feature_map[
                i * stride: i * stride + pool_size,
                j * stride: j * stride + pool_size,
            ]
            output[i, j] = region.max()
    return output


def average_pool2d(feature_map: np.ndarray, pool_size: int = 2, stride: int | None = None) -> np.ndarray:
    """Bir özellik haritasına average pooling uygular: her pencerenin
    ortalamasını alır (daha yumuşak, daha az agresif küçültme)."""
    stride = stride or pool_size
    h, w = feature_map.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1

    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = feature_map[
                i * stride: i * stride + pool_size,
                j * stride: j * stride + pool_size,
            ]
            output[i, j] = region.mean()
    return output


def main() -> None:
    image = color.rgb2gray(data.astronaut())[::2, ::2]
    print(f"Girdi boyutu: {image.shape}")

    max_pooled = max_pool2d(image, pool_size=4)
    avg_pooled = average_pool2d(image, pool_size=4)

    print(f"Max Pooling (4x4) sonrası boyut:     {max_pooled.shape}")
    print(f"Average Pooling (4x4) sonrası boyut: {avg_pooled.shape}")
    print(f"Boyut küçülme oranı: {image.size / max_pooled.size:.0f}x")

    # Küçük, elle takip edilebilir bir örnek
    print("\n--- Küçük Elle Takip Edilebilir Örnek ---")
    small_map = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [7, 2, 8, 3],
        [1, 9, 4, 5],
    ], dtype=float)
    print(f"Girdi (4x4):\n{small_map}")
    print(f"\nMax Pooling (2x2, stride=2):\n{max_pool2d(small_map, pool_size=2)}")
    print(f"\nAverage Pooling (2x2, stride=2):\n{average_pool2d(small_map, pool_size=2)}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Orijinal {image.shape}")
    axes[1].imshow(max_pooled, cmap="gray")
    axes[1].set_title(f"Max Pool {max_pooled.shape}")
    axes[2].imshow(avg_pooled, cmap="gray")
    axes[2].set_title(f"Avg Pool {avg_pooled.shape}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("pooling_comparison.png", dpi=100)
    print("\nGörsel pooling_comparison.png dosyasına kaydedildi.")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Havuzlama, bir CNN'e KÜÇÜK KONUM DEĞİŞİKLİKLERİNE karşı dayanıklılık
(translation invariance) kazandırır: bir kenar görüntüde 1-2 piksel
kaysa bile, max pooling sonrası çıktı büyük ölçüde aynı kalır. Ayrıca
her katmandan sonra özellik haritasının boyutunu küçülterek hem
hesaplama maliyetini azaltır hem de ağın "daha büyük resmi" görmesini
sağlar (her sonraki katman, girdi görüntüsünün daha geniş bir bölgesini
"görür").

İyileştirme Fikirleri
-----------------------
1. `stride`'ı `pool_size`'dan farklı yaparak (örn. pool_size=3, stride=1)
   "örtüşen" (overlapping) havuzlamayı deneyin.
2. Global Average Pooling'i uygulayın (tüm özellik haritasını TEK bir
   sayıya indirger -- modern CNN'lerde son katmanlarda yaygın kullanılır).
3. Max pooling'in gürültüye karşı average pooling'den nasıl farklı tepki
   verdiğini, görüntüye rastgele "tuz-biber" gürültüsü ekleyerek test edin.
"""
