"""
07 - Bir CNN Katmanının Tam İleri Geçişi (Conv + ReLU + Pool)
==================================================================

Önceki örneklerde evrişimi (Örnek 02) ve havuzlamayı (Örnek 03) ayrı
ayrı inceledik. Bu örnek, gerçek bir CNN'in bir katman bloğunu -- birden
fazla öğrenilebilir filtre + ReLU aktivasyonu + max pooling -- uçtan uca
birleştirip, girdi görüntüsünden çıktıya kadar boyutların nasıl
değiştiğini izler.

Gereksinimler: numpy, scipy
"""

import numpy as np
from scipy.signal import convolve2d


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def max_pool2d(feature_map: np.ndarray, pool_size: int = 2) -> np.ndarray:
    h, w = feature_map.shape
    out_h, out_w = h // pool_size, w // pool_size
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = feature_map[
                i * pool_size:(i + 1) * pool_size,
                j * pool_size:(j + 1) * pool_size,
            ]
            output[i, j] = region.max()
    return output


class ConvLayer:
    """Birden fazla öğrenilebilir filtreye sahip, basitleştirilmiş bir
    evrişimli katman. Gerçek bir CNN'de bu filtreler eğitim sırasında
    geri yayılımla öğrenilir; burada rastgele başlatılmış olarak
    bırakıyoruz (bu bölümün odağı MİMARİ, eğitim değil -- eğitim
    mekaniği için Bölüm 3-4'e bakın)."""

    def __init__(self, n_filters: int, filter_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He-tarzı ölçekleme (Bölüm 4) -- ReLU ile birlikte kullanılacağı için
        scale = np.sqrt(2 / (filter_size * filter_size))
        self.filters = rng.normal(0, scale, (n_filters, filter_size, filter_size))
        self.n_filters = n_filters

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Her filtreyi görüntüye uygular ve bir "özellik haritaları
        yığını" (feature map stack) üretir.

        Returns:
            Şekil (n_filters, out_h, out_w) olan bir dizi -- her dilim,
            bir filtrenin tespit ettiği örüntüyü gösteren bir haritadır.
        """
        feature_maps = []
        for filter_kernel in self.filters:
            conv_output = convolve2d(image, filter_kernel, mode="valid")
            activated = relu(conv_output)
            feature_maps.append(activated)
        return np.stack(feature_maps)


def main() -> None:
    # Basit bir sentetik "görüntü": bir kare görüntüde bir çapraz çizgi deseni
    image = np.zeros((28, 28))
    np.fill_diagonal(image, 1.0)
    image[:, 14] = 0.7  # dikey bir çizgi ekle
    image[14, :] = 0.7  # yatay bir çizgi ekle

    print(f"Girdi görüntü boyutu: {image.shape}")

    # --- Katman 1: Conv (8 filtre, 3x3) -> ReLU -> MaxPool ---
    conv1 = ConvLayer(n_filters=8, filter_size=3, seed=1)
    feature_maps_1 = conv1.forward(image)
    print(f"\nKatman 1 -- Evrişim sonrası (8 filtre, 3x3):  {feature_maps_1.shape}")
    print("  (28x28 girdi, 3x3 filtre -> 26x26 çıktı; 8 farklı örüntü haritası)")

    pooled_1 = np.stack([max_pool2d(fm, pool_size=2) for fm in feature_maps_1])
    print(f"Katman 1 -- MaxPool (2x2) sonrası:             {pooled_1.shape}")

    # --- Katman 2: İkinci bir Conv+ReLU+Pool bloğu, ilk katmanın çıktısı üzerinde ---
    # Not: gerçek bir CNN'de bu, her biri TÜM önceki katmandaki tüm
    # özellik haritalarını "gören" filtrelerle yapılır. Burada
    # basitleştirmek için ilk özellik haritasını örnek alıyoruz.
    conv2 = ConvLayer(n_filters=16, filter_size=3, seed=2)
    feature_maps_2 = conv2.forward(pooled_1[0])
    print(f"\nKatman 2 -- Evrişim sonrası (16 filtre, 3x3): {feature_maps_2.shape}")

    pooled_2 = np.stack([max_pool2d(fm, pool_size=2) for fm in feature_maps_2])
    print(f"Katman 2 -- MaxPool (2x2) sonrası:             {pooled_2.shape}")

    # --- Son: "Düzleştirme" (Flatten) -- tam bağlantılı katmana geçiş için ---
    flattened = pooled_2.flatten()
    print(f"\nDüzleştirilmiş (Flatten) vektör boyutu: {flattened.shape}")
    print("(Bu vektör, artık Bölüm 4'teki gibi bir MLP/tam bağlantılı")
    print(" katmana beslenip sınıflandırma yapılabilir)")

    print(f"\nÖzet: {image.size} pikselden başlayarak, iki Conv+Pool")
    print(f"bloğu sonunda {flattened.size} boyutlu bir özellik vektörüne indirgendi.")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Gerçek bir CNN mimarisi (LeNet, AlexNet, ResNet, hepsi) bu deseni tekrar
tekrar uygular: Evrişim (örüntüleri tespit et) -> ReLU (doğrusal olmama
ekle) -> Havuzlama (boyutu küçült, konum değişikliklerine dayanıklılık
kazandır). Her katman bloğu, bir öncekinden DAHA SOYUT örüntüler
öğrenir: ilk katmanlar kenar/köşe gibi basit örüntüleri, sonraki
katmanlar bunları birleştirerek göz/kulak gibi daha karmaşık örüntüleri,
en derin katmanlar ise "kedi yüzü" gibi tüm nesneleri temsil eden
özellikleri öğrenir. Son olarak "Flatten" işlemi, 2 boyutlu uzamsal
bilgiyi Bölüm 4'teki MLP'lerin anlayabileceği 1 boyutlu bir vektöre
dönüştürür.

İyileştirme Fikirleri
-----------------------
1. `padding="same"` mantığını uygulayarak (kenarlara sıfır ekleyerek)
   özellik haritalarının boyutunu her evrişim katmanında AYNI tutmayı
   deneyin.
2. Filtre sayısını (n_filters) artırıp azaltarak düzleştirilmiş
   vektörün boyutunun nasıl değiştiğini gözlemleyin.
3. Bu mimariyi gerçek bir CNN kütüphanesiyle (PyTorch `nn.Conv2d`,
   Keras `Conv2D`) karşılaştırın -- API'lerin bu adımları nasıl
   soyutladığını inceleyin.
"""
