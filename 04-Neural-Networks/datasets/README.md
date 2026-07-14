# 📊 Bölüm 4 Veri Setleri

Bu bölümdeki tüm örnekler ya sentetik olarak üretilen veri kullanır ya da
scikit-learn'ün yerleşik `load_digits()` veri setini (1797 adet 8x8 piksel
el yazısı rakam) kullanır — ayrıca bir indirme gerekmez.

```python
from sklearn.datasets import load_digits
digits = load_digits()
```

Tam MNIST (28x28 piksel, 70.000 örnek) ile denemek isterseniz:

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
```
