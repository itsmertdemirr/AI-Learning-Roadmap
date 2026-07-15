"""
01 - Görüntülerin Piksel Tensörü Olarak Temsili
===================================================

Bir bilgisayarın "gördüğü" bir görüntünün aslında sadece bir sayı
dizisi (tensör) olduğunu gösterir. Gri tonlamalı ve renkli görüntülerin
nasıl temsil edildiğini, piksel değerlerine nasıl erişileceğini ve temel
görüntü istatistiklerinin nasıl hesaplanacağını inceler.

Gereksinimler: numpy, matplotlib, scikit-image
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, data


def describe_image(image: np.ndarray, name: str) -> None:
    """Bir görüntü dizisinin temel özelliklerini yazdırır."""
    print(f"--- {name} ---")
    print(f"  Şekil (shape):     {image.shape}")
    print(f"  Veri tipi (dtype): {image.dtype}")
    print(f"  Min/Max değer:     {image.min()} / {image.max()}")
    if image.ndim == 2:
        print("  Kanal sayısı:      1 (gri tonlama)")
    else:
        print(f"  Kanal sayısı:      {image.shape[2]} (renkli, genellikle R-G-B)")
    print()


def main() -> None:
    # scikit-image'in yerleşik örnek görüntülerini kullan (indirme gerekmez)
    color_image = data.astronaut()          # (yükseklik, genişlik, 3) -- RGB
    gray_image = color.rgb2gray(color_image)  # (yükseklik, genişlik) -- tek kanal

    describe_image(color_image, "Renkli Görüntü (RGB)")
    describe_image(gray_image, "Gri Tonlamalı Görüntü")

    # Tek bir pikselin değerine erişim
    y, x = 100, 150
    pixel_rgb = color_image[y, x]
    pixel_gray = gray_image[y, x]
    print(f"(satır={y}, sütun={x}) konumundaki piksel:")
    print(f"  RGB değeri:   {pixel_rgb}  (R={pixel_rgb[0]}, G={pixel_rgb[1]}, B={pixel_rgb[2]})")
    print(f"  Gri değeri:   {pixel_gray:.4f}  (0.0=siyah, 1.0=beyaz)")

    # Görüntünün bir bölgesini (region of interest) kırp
    crop = color_image[50:150, 50:150]
    print(f"\nKırpılmış bölge şekli: {crop.shape}")

    # Görselleştir
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(color_image)
    axes[0].set_title("Orijinal (RGB)")
    axes[1].imshow(gray_image, cmap="gray")
    axes[1].set_title("Gri Tonlama")
    axes[2].imshow(crop)
    axes[2].set_title("Kırpılmış Bölge")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("image_representation.png", dpi=100)
    print("\nGörsel image_representation.png dosyasına kaydedildi.")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Bir görüntü, bir sinir ağı için sadece 3 boyutlu bir sayı dizisidir:
(yükseklik, genişlik, kanal). Bir CNN'in "gördüğü" şey piksellerin
kendisi değil, bu sayıların matematiksel örüntüleridir. Bu bölümdeki her
teknik -- evrişim, havuzlama, veri artırma -- bu ham sayı dizisi üzerinde
çalışır.

İyileştirme Fikirleri
-----------------------
1. RGB kanallarını ayrı ayrı görselleştirin (sadece kırmızı kanal, sadece
   yeşil kanal vb.) ve her birinin görüntünün farklı yönlerini nasıl
   kodladığını gözlemleyin.
2. Görüntüyü 0-255 tam sayı aralığından 0.0-1.0 kayan nokta aralığına
   normalize edin (sinir ağları için standart bir ön işleme adımıdır).
3. `data` modülündeki diğer örnek görüntüleri (`data.coffee()`,
   `data.camera()` vb.) inceleyin.
"""
