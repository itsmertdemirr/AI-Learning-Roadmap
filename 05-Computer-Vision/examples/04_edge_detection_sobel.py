"""
04 - Sobel Kenar Tespiti
============================

Klasik Sobel operatörünü kullanarak bir görüntüdeki yatay ve dikey
kenarları tespit eder, ardından bunları birleştirerek gradyan
büyüklüğü (gradient magnitude) haritasını hesaplar. `scipy.signal.convolve2d`
kullanarak Örnek 02'deki manuel evrişimden çok daha hızlı çalışır.

Gereksinimler: numpy, scipy, matplotlib, scikit-image
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from skimage import data

# Sobel çekirdekleri: biri yatay (dikey kenarları bulur), biri dikey
# (yatay kenarları bulur) gradyanı yaklaşık olarak hesaplar
SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
])

SOBEL_Y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
])


def detect_edges(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sobel operatörüyle kenarları tespit eder.

    Returns:
        (gx, gy, magnitude): x yönündeki gradyan, y yönündeki gradyan ve
        toplam gradyan büyüklüğü (kenar haritası).
    """
    gx = convolve2d(image, SOBEL_X, mode="same", boundary="symm")
    gy = convolve2d(image, SOBEL_Y, mode="same", boundary="symm")
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return gx, gy, magnitude


def main() -> None:
    image = data.camera().astype(float) / 255.0  # zaten gri tonlamalı bir örnek görüntü

    gx, gy, magnitude = detect_edges(image)

    # Görselleştirme için gradyan büyüklüğünü 0-1 aralığına normalize et
    magnitude_normalized = magnitude / magnitude.max()

    print(f"Görüntü boyutu:              {image.shape}")
    print(f"Yatay gradyan (Gx) aralığı:  [{gx.min():.2f}, {gx.max():.2f}]")
    print(f"Dikey gradyan (Gy) aralığı:  [{gy.min():.2f}, {gy.max():.2f}]")
    print(f"Gradyan büyüklüğü aralığı:   [{magnitude.min():.2f}, {magnitude.max():.2f}]")

    # Bir eşik değeri uygulayarak "ikili" (binary) bir kenar haritası oluştur
    threshold = magnitude_normalized.mean() + magnitude_normalized.std()
    binary_edges = magnitude_normalized > threshold
    edge_pixel_ratio = binary_edges.mean()
    print(f"\nEşik değeri: {threshold:.3f}")
    print(f"Kenar olarak işaretlenen piksel oranı: %{edge_pixel_ratio * 100:.1f}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Orijinal")
    axes[0, 1].imshow(gx, cmap="gray")
    axes[0, 1].set_title("Yatay Gradyan (Gx)\ndikey kenarları vurgular")
    axes[0, 2].imshow(gy, cmap="gray")
    axes[0, 2].set_title("Dikey Gradyan (Gy)\nyatay kenarları vurgular")
    axes[1, 0].imshow(magnitude_normalized, cmap="gray")
    axes[1, 0].set_title("Gradyan Büyüklüğü\n(tüm kenarlar)")
    axes[1, 1].imshow(binary_edges, cmap="gray")
    axes[1, 1].set_title(f"Eşiklenmiş Kenarlar\n(eşik={threshold:.2f})")
    axes[1, 2].axis("off")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("sobel_edge_detection.png", dpi=100)
    print("\nGörsel sobel_edge_detection.png dosyasına kaydedildi.")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Sobel operatörü, her pikseldeki parlaklık DEĞİŞİM HIZINI (gradyanı)
hesaplar -- kenarlar, parlaklığın hızla değiştiği yerlerdir. Bu, tamamen
elle tasarlanmış (hand-crafted) bir özellik çıkarma yöntemidir; 2012
öncesinde bilgisayarlı görü büyük ölçüde bu tür elle tasarlanmış
filtrelere dayanıyordu. Modern CNN'ler ise Sobel'e BENZER kenar
dedektörlerini kendileri, veriden otomatik olarak öğrenir -- ilk
evrişim katmanlarını görselleştirdiğinizde genellikle Sobel'e çok
benzeyen çekirdekler bulursunuz!

İyileştirme Fikirleri
-----------------------
1. Canny kenar dedektörünü (`skimage.feature.canny`) deneyip Sobel ile karşılaştırın.
2. Farklı eşik değerleriyle (threshold) ikili kenar haritasının nasıl değiştiğini gözlemleyin.
3. Gürültülü bir görüntüde (rastgele gürültü ekleyerek) Sobel'in ne kadar
   "gürültüye duyarlı" olduğunu test edin -- neden önce bulanıklaştırma
   (Örnek 02) uygulamanın yaygın bir pratik olduğunu düşünün.
"""
