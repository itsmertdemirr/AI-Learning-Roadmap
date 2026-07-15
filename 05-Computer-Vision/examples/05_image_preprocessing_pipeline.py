"""
05 - Görüntü Ön İşleme Hattı (Pipeline)
============================================

Herhangi bir CNN'e görüntü beslemeden önce yapılması gereken standart ön
işleme adımlarını gösterir: yeniden boyutlandırma, gri tonlamaya
çevirme, normalizasyon ve piksel değeri ölçekleme. Bu adımların
atlanması, eğitimi yavaşlatabilir veya modelin hiç yakınsamamasına
neden olabilir (Bölüm 4'teki ağırlık başlatma/BatchNorm derslerini
hatırlayın -- burada da benzer bir "ölçek tutarlılığı" prensibi geçerli).

Gereksinimler: numpy, scikit-image
"""

import numpy as np
from skimage import color, data, transform


def resize_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Görüntüyü hedef (yükseklik, genişlik) boyutuna yeniden boyutlandırır.

    CNN'ler SABİT boyutlu girdi bekler (tam bağlantılı katmanlar
    öncesindeki katman sayısı, girdi boyutuna bağlıdır), bu yüzden farklı
    boyutlardaki tüm görüntüler ortak bir boyuta getirilmelidir.
    """
    return transform.resize(image, target_size, anti_aliasing=True)


def normalize_min_max(image: np.ndarray) -> np.ndarray:
    """Piksel değerlerini [0, 1] aralığına ölçekler."""
    img_min, img_max = image.min(), image.max()
    if img_max == img_min:
        return np.zeros_like(image)
    return (image - img_min) / (img_max - img_min)


def standardize(image: np.ndarray, mean: float | None = None, std: float | None = None) -> np.ndarray:
    """Görüntüyü ortalama 0, standart sapma 1 olacak şekilde standartlaştırır
    (Bölüm 4'teki BatchNorm ile aynı matematiksel prensip -- burada tüm
    veri setine değil, tek bir görüntüye/veri setine uygulanır).
    """
    mean = image.mean() if mean is None else mean
    std = image.std() if std is None else std
    if std == 0:
        return np.zeros_like(image)
    return (image - mean) / std


def preprocess_pipeline(
    image: np.ndarray, target_size: tuple[int, int] = (64, 64), grayscale: bool = True
) -> np.ndarray:
    """Ham bir görüntüyü, bir CNN'e beslenmeye hazır hale getiren tam
    ön işleme hattı."""
    if grayscale and image.ndim == 3:
        image = color.rgb2gray(image)

    resized = resize_image(image, target_size)
    normalized = normalize_min_max(resized)
    return normalized


def main() -> None:
    raw_image = data.astronaut()
    print(f"Ham görüntü:              boyut={raw_image.shape}, dtype={raw_image.dtype}, "
          f"aralık=[{raw_image.min()}, {raw_image.max()}]")

    processed = preprocess_pipeline(raw_image, target_size=(64, 64), grayscale=True)
    print(f"İşlenmiş görüntü:          boyut={processed.shape}, dtype={processed.dtype}, "
          f"aralık=[{processed.min():.3f}, {processed.max():.3f}]")

    # Standartlaştırmayı ayrıca göster
    standardized = standardize(processed)
    print(f"\nStandartlaştırılmış:       ortalama={standardized.mean():.6f}, std={standardized.std():.6f}")
    print("(ortalama ~0, std ~1 olmalı -- Bölüm 4'teki BatchNorm prensibiyle aynı)")

    # Bir "toplu iş" (batch) senaryosu: farklı boyutlardaki birden fazla
    # görüntüyü ortak boyuta getirme
    print("\n--- Toplu İşleme Senaryosu ---")
    fake_dataset = [
        data.astronaut(),                      # (512, 512, 3)
        data.astronaut()[:300, :400],           # (300, 400, 3) -- farklı boyut
        data.astronaut()[100:, 100:],           # (412, 412, 3) -- farklı boyut
    ]
    processed_batch = np.array([
        preprocess_pipeline(img, target_size=(32, 32)) for img in fake_dataset
    ])
    print(f"Farklı boyutlardaki {len(fake_dataset)} görüntü, tek bir toplu")
    print(f"iş (batch) dizisine dönüştürüldü: {processed_batch.shape}")
    print("(artık hepsi aynı boyutta -- bir modele toplu olarak beslenebilir)")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
"Garbage in, garbage out" (çöp girdi, çöp çıktı) prensibi bilgisayarlı
görüde özellikle geçerlidir. Tutarsız boyutlar, ölçeklenmemiş piksel
değerleri veya standartlaştırılmamış veriler, en iyi CNN mimarisini bile
kötü performans göstermeye zorlayabilir. Bu ön işleme adımları, gerçek
dünya bilgisayarlı görü projelerinde model mimarisi kadar önemlidir.

İyileştirme Fikirleri
-----------------------
1. ImageNet'in standart normalizasyon değerlerini (mean=[0.485, 0.456,
   0.406], std=[0.229, 0.224, 0.225], RGB kanalları için) uygulayan bir
   fonksiyon yazın.
2. "Center crop" (görüntünün ortasından sabit boyutlu bir bölge kırpma)
   fonksiyonu ekleyin.
3. Bu hattı, Örnek 08'deki (el yazısı rakam) veri setine uygulayarak
   ön işlemenin sınıflandırma doğruluğunu nasıl etkilediğini test edin.
"""
