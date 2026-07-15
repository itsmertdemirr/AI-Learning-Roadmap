"""
02 - Sıfırdan 2D Evrişim (Convolution)
==========================================

CNN'lerin kalbindeki matematiksel işlemi -- 2D evrişimi -- hiçbir
framework kullanmadan, sadece kayan pencere (sliding window) ve
eleman bazında çarpım/toplam ile sıfırdan uygular. Ardından klasik
bulanıklaştırma (blur), keskinleştirme (sharpen) ve kenar tespiti
(edge detection) çekirdeklerini (kernel) dener.

Gereksinimler: numpy, matplotlib, scikit-image
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, data


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2 boyutlu bir gri tonlamalı görüntüye bir evrişim çekirdeği uygular.

    Args:
        image: (yükseklik, genişlik) şeklinde gri tonlamalı görüntü.
        kernel: (kh, kw) şeklinde küçük bir kare/dikdörtgen çekirdek (filtre).

    Returns:
        Evrişim uygulanmış çıktı görüntüsü ("valid" mod -- kenarlar küçülür).
    """
    if image.ndim != 2:
        raise ValueError("image 2 boyutlu (gri tonlamalı) olmalıdır")

    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1
    output = np.zeros((out_h, out_w))

    # Çekirdeği görüntü üzerinde kaydır; her konumda eleman bazında
    # çarpıp topla ("kayan pencere" işlemi -- CNN'lerin temel taşı)
    for i in range(out_h):
        for j in range(out_w):
            region = image[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)

    return output


# --- Klasik görüntü işleme çekirdekleri (kernel'ler) --------------------
BLUR_KERNEL = np.ones((3, 3)) / 9.0  # komşu piksellerin ortalaması

SHARPEN_KERNEL = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
])

EDGE_DETECT_KERNEL = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1],
])


def main() -> None:
    image = color.rgb2gray(data.astronaut())
    # Hesaplama süresini kısaltmak için görüntüyü küçült
    image = image[::2, ::2]

    print(f"Girdi görüntüsü boyutu: {image.shape}")

    blurred = convolve2d(image, BLUR_KERNEL)
    sharpened = convolve2d(image, SHARPEN_KERNEL)
    edges = convolve2d(image, EDGE_DETECT_KERNEL)

    print(f"Bulanıklaştırılmış çıktı boyutu:  {blurred.shape}  (3x3 çekirdek -> her kenardan 1 piksel kaybı)")
    print(f"Kenar tespiti çıktı boyutu:       {edges.shape}")
    print(f"Kenar tespiti değer aralığı:      [{edges.min():.3f}, {edges.max():.3f}]")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Orijinal")
    axes[1].imshow(blurred, cmap="gray")
    axes[1].set_title("Bulanıklaştırma (Blur)")
    axes[2].imshow(sharpened, cmap="gray")
    axes[2].set_title("Keskinleştirme (Sharpen)")
    axes[3].imshow(edges, cmap="gray")
    axes[3].set_title("Kenar Tespiti")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("convolution_kernels.png", dpi=100)
    print("\nGörsel convolution_kernels.png dosyasına kaydedildi.")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Bir CNN, bu tam olarak yaptığımız işlemi yapar -- TEK FARKLA: çekirdek
değerlerini (-1, 5, -1 gibi sayıları) biz elle seçmiyoruz, ağ bunları
eğitim sırasında OTOMATİK olarak öğreniyor. "Evrişimli katman", aslında
her biri farklı bir örüntüyü (kenar, köşe, doku) tespit etmek üzere
öğrenilen onlarca/yüzlerce böyle çekirdeğin bir koleksiyonudur.

İyileştirme Fikirleri
-----------------------
1. Farklı boyutlarda çekirdekler (5x5, 7x7) deneyip çıktı boyutunun nasıl
   değiştiğini gözlemleyin.
2. "Padding" (kenarlara sıfır ekleyerek çıktı boyutunu korumak) ekleyin.
3. "Stride" (çekirdeği her seferinde 1'den fazla piksel kaydırmak)
   parametresi ekleyin.
4. Bu fonksiyonun çok yavaş olduğunu fark edeceksiniz (saf Python
   döngüleri) -- `scipy.signal.convolve2d` ile hızını karşılaştırın.
"""
