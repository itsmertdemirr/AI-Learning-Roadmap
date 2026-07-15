"""
06 - Veri Artırma (Data Augmentation) Teknikleri
=====================================================

Sınırlı bir eğitim veri setini, mevcut görüntülere rastgele
dönüşümler (çevirme, döndürme, parlaklık değişimi, gürültü ekleme)
uygulayarak yapay olarak çeşitlendirir. Bu, aşırı öğrenmeyi azaltmanın
(Bölüm 4'teki Dropout/L2'ye benzer bir amaçla) ve modelin gerçek dünya
varyasyonlarına karşı daha dayanıklı olmasını sağlamanın en etkili
yollarından biridir.

Gereksinimler: numpy, matplotlib, scikit-image
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, transform


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """Görüntüyü yatay olarak aynalar (soldan sağa)."""
    return image[:, ::-1]


def vertical_flip(image: np.ndarray) -> np.ndarray:
    """Görüntüyü dikey olarak aynalar (üstten alta)."""
    return image[::-1, :]


def rotate(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Görüntüyü belirtilen açı kadar döndürür."""
    return transform.rotate(image, angle_degrees, mode="edge")


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Parlaklığı ölçekler ve [0, 1] aralığında kalmasını sağlar
    (factor > 1 daha parlak, factor < 1 daha karanlık yapar)."""
    return np.clip(image * factor, 0, 1)


def add_gaussian_noise(image: np.ndarray, std: float = 0.05, seed: int = 0) -> np.ndarray:
    """Görüntüye Gauss gürültüsü ekler (gerçek kamera/sensör gürültüsünü simüle eder)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, std, image.shape)
    return np.clip(image + noise, 0, 1)


def random_crop(image: np.ndarray, crop_size: tuple[int, int], seed: int = 0) -> np.ndarray:
    """Görüntüden rastgele bir bölge kırpar (nesnenin farklı konumlarda
    görülmesini simüle eder)."""
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    ch, cw = crop_size
    if ch > h or cw > w:
        raise ValueError("crop_size, görüntü boyutundan büyük olamaz")
    top = rng.integers(0, h - ch + 1)
    left = rng.integers(0, w - cw + 1)
    return image[top:top + ch, left:left + cw]


def augment_dataset(image: np.ndarray, n_augmentations: int = 5, seed: int = 42) -> list[np.ndarray]:
    """Tek bir görüntüden, farklı dönüşümler uygulayarak birden fazla
    yapay eğitim örneği üretir -- veri artırmanın pratikteki kullanımı."""
    rng = np.random.default_rng(seed)
    augmented = []

    transforms = [
        lambda img: horizontal_flip(img),
        lambda img: rotate(img, rng.uniform(-20, 20)),
        lambda img: adjust_brightness(img, rng.uniform(0.7, 1.3)),
        lambda img: add_gaussian_noise(img, std=0.03, seed=int(rng.integers(0, 10000))),
    ]

    for _ in range(n_augmentations):
        transform_fn = transforms[rng.integers(0, len(transforms))]
        augmented.append(transform_fn(image))

    return augmented


def main() -> None:
    image = data.astronaut().astype(float) / 255.0
    image = transform.resize(image, (128, 128), anti_aliasing=True)

    flipped = horizontal_flip(image)
    rotated = rotate(image, 15)
    brightened = adjust_brightness(image, 1.4)
    darkened = adjust_brightness(image, 0.6)
    noisy = add_gaussian_noise(image, std=0.05)
    cropped = random_crop(image, (96, 96), seed=1)

    print(f"Orijinal boyut:            {image.shape}")
    print(f"Kırpılmış boyut:           {cropped.shape}  (farklı boyut -- modele beslemeden önce yeniden boyutlandırılmalı)")
    print(f"Parlaklaştırılmış aralık:  [{brightened.min():.3f}, {brightened.max():.3f}]")
    print(f"Gürültülü görüntü aralığı: [{noisy.min():.3f}, {noisy.max():.3f}]")

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    titles_images = [
        ("Orijinal", image), ("Yatay Çevirme", flipped),
        ("15° Döndürme", rotated), ("Parlaklaştırma", brightened),
        ("Karartma", darkened), ("Gauss Gürültüsü", noisy),
        ("Rastgele Kırpma", cropped),
    ]
    for ax, (title, img) in zip(axes.flat, titles_images):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    axes.flat[len(titles_images)].axis("off")  # kullanılmayan son alt grafiği gizle
    plt.tight_layout()
    plt.savefig("data_augmentation_examples.png", dpi=100)
    print("\nGörsel data_augmentation_examples.png dosyasına kaydedildi.")

    augmented_set = augment_dataset(image, n_augmentations=8)
    print(f"\n1 orijinal görüntüden {len(augmented_set)} yapay eğitim örneği üretildi.")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Veri artırma, modele "bir kedi, ters çevrilse de, biraz döndürülse de,
farklı ışıklandırmada da hâlâ bir kedidir" öğretmenin bedava bir
yoludur. Gerçek dünyada etiketli veri toplamak pahalıdır; veri artırma,
mevcut veriden daha fazla "öğrenme sinyali" sıkarak bunu kısmen telafi
eder. Modern framework'ler (PyTorch'un `torchvision.transforms`'u,
Keras'ın `ImageDataGenerator`'ı) bu dönüşümleri eğitim sırasında
OTOMATİK ve RASTGELE uygular -- her epokta model, aynı görüntünün biraz
farklı bir versiyonunu görür.

İyileştirme Fikirleri
-----------------------
1. CutMix veya Mixup gibi daha gelişmiş artırma tekniklerini araştırın
   (iki görüntüyü karıştırarak yeni eğitim örnekleri oluşturur).
2. Hangi artırma tekniklerinin HANGİ problemler için uygunsuzOLABİLECEĞİNİ
   düşünün (örn. el yazısı rakam tanımada dikey çevirme mantıklı mı?).
3. Bu fonksiyonları Örnek 08'deki (el yazısı rakam sınıflandırıcı) veri
   setine uygulayıp doğruluk üzerindeki etkisini ölçün.
"""
