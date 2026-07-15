# 📊 Bölüm 5 Veri Setleri

Bu bölümdeki örnekler `scikit-image`'in yerleşik örnek görüntülerini
(`skimage.data.astronaut()`, `skimage.data.camera()` — indirme gerekmez)
ve `scikit-learn`'in yerleşik `load_digits()` veri setini kullanır.

## Daha Büyük Veri Setleriyle Denemek İçin

| Veri Seti | Nasıl Erişilir |
|---|---|
| **CIFAR-10** (10 sınıf, 60.000 renkli görüntü) | `tensorflow.keras.datasets.cifar10` veya `torchvision.datasets.CIFAR10` |
| **MNIST** (tam boyutlu, 70.000 rakam) | `sklearn.datasets.fetch_openml('mnist_784')` |
| **ImageNet** (1000+ sınıf) | [image-net.org](https://www.image-net.org/) (kayıt gerektirir) |
| **Kendi görüntüleriniz** | `skimage.io.imread('dosya_yolu.jpg')` ile herhangi bir yerel görüntüyü yükleyebilirsiniz |
