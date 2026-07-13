# 📊 Bölüm 2 Veri Setleri

Bu bölümdeki örnekler öncelikle **scikit-learn'e yerleşik veri setlerini**
kullanır (indirme gerekmez) veya sentetik olarak üretilir. Kendi
projelerinizde denemek isteyebileceğiniz ek klasik veri setleri:

| Veri Seti | Kullanım Alanı | Nasıl Erişilir |
|-----------|------------------|------------------|
| **Iris** | Çiçek türü sınıflandırma (Örnek 1'de kullanılıyor) | `sklearn.datasets.load_iris()` — yerleşik, indirme gerektirmez |
| **Titanic** | Hayatta kalma tahmini (ikili sınıflandırma) | [Kaggle Titanic](https://www.kaggle.com/c/titanic/data) veya `seaborn.load_dataset('titanic')` |
| **MNIST** | El yazısı rakam tanıma (Bölüm 4-5'te kullanılacak) | `sklearn.datasets.fetch_openml('mnist_784')` veya `tensorflow.keras.datasets.mnist` |
| **CIFAR-10** | Renkli görüntü sınıflandırma (Bölüm 5'te kullanılacak) | `tensorflow.keras.datasets.cifar10` veya `torchvision.datasets.CIFAR10` |
| **IMDB** | Film yorumu duygu analizi (Bölüm 6'da kullanılacak) | `tensorflow.keras.datasets.imdb` veya [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| **Spam Detection** | E-posta/SMS spam sınıflandırma (Bölüm 6'da kullanılacak) | [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |

## Hızlı Örnek: Titanic Veri Setini Yükleme

```python
import seaborn as sns

titanic = sns.load_dataset("titanic")
print(titanic.head())
print(titanic.isnull().sum())  # eksik değerleri kontrol et
```

## Not

Büyük veri setlerini (MNIST, CIFAR-10 gibi) doğrudan bu depoya **eklemiyoruz**
— bunun yerine ilgili kütüphanelerin yerleşik indirme fonksiyonlarını
kullanıyoruz. Bu, depo boyutunu küçük tutar ve her zaman güncel/doğrulanmış
veriye erişmenizi sağlar.
