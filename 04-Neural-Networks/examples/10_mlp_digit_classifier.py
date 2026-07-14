"""
10 - Çok Katmanlı Algılayıcı (MLP) ile El Yazısı Rakam Sınıflandırma
========================================================================

Bu bölümde öğrendiğimiz her şeyi (aktivasyonlar, ağırlık başlatma,
düzenlileştirme, mini-parti eğitimi) bir araya getiren, scikit-learn'in
`digits` veri setinde (8x8 piksel el yazısı rakamlar, MNIST'in küçük bir
kuzeni) çalışan gerçek bir çok katmanlı algılayıcı (MLP) sınıflandırıcı.

Gereksinimler: scikit-learn, numpy, matplotlib
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def main() -> None:
    # 1. Veriyi yükle: 1797 adet 8x8 piksel el yazısı rakam görüntüsü
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"Veri seti boyutu: {X.shape[0]} örnek, {X.shape[1]} özellik (8x8 piksel)")
    print(f"Sınıflar: {sorted(set(y))}\n")

    # 2. Eğitim/test bölmesi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Özellikleri standartlaştır -- sinir ağları ölçeklenmiş girdilerle
    # çok daha iyi ve hızlı eğitilir (Örnek 07'deki BatchNorm'un manuel
    # bir ön işleme versiyonu gibi düşünün)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Bu bölümün kavramlarını bir araya getiren MLP:
    #    - iki gizli katman (mimari derinlik)
    #    - ReLU aktivasyonu (Örnek 02, 04)
    #    - Adam optimize edici (Örnek 08-09'daki gradyan inişinin gelişmiş hali)
    #    - L2 düzenlileştirme (Örnek 05, alpha parametresiyle)
    #    - early_stopping (aşırı öğrenmeye karşı ek bir koruma)
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,                 # L2 düzenlileştirme gücü
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # 5. Değerlendir
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Doğruluğu: {accuracy:.2%}")
    print(f"Kullanılan epok sayısı (early stopping ile durdu): {model.n_iter_}\n")
    print("Sınıflandırma Raporu:")
    print(classification_report(y_test, predictions))

    # 6. Karışıklık matrisini görselleştir
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, cmap="Blues")
    ax.set_title(f"Karışıklık Matrisi (Doğruluk: {accuracy:.1%})")
    plt.tight_layout()
    plt.savefig("mlp_confusion_matrix.png", dpi=100)
    print("Karışıklık matrisi mlp_confusion_matrix.png dosyasına kaydedildi.")

    # 7. Yanlış sınıflandırılan birkaç örneği incele
    wrong_indices = np.where(predictions != y_test)[0]
    print(f"\nToplam {len(wrong_indices)} yanlış sınıflandırma bulundu (test setinin %{len(wrong_indices)/len(y_test)*100:.1f}'i).")
    if len(wrong_indices) > 0:
        idx = wrong_indices[0]
        print(f"Örnek: gerçek={y_test[idx]}, tahmin={predictions[idx]}")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Bu tek dosya, bölümün tüm kavramlarını gerçek bir uygulamada birleştirir:
özellik ölçekleme, ReLU aktivasyonu, Adam optimize edici (gelişmiş
gradyan inişi), L2 düzenlileştirme ve erken durdurma (early stopping,
dropout'a benzer bir aşırı öğrenme önleyici). scikit-learn bunların
hepsini `MLPClassifier` içinde bizim için birleştiriyor -- ama artık
perde arkasında ne olduğunu tam olarak biliyorsunuz.

İyileştirme Fikirleri
-----------------------
1. `hidden_layer_sizes`'ı değiştirip (örn. (128,) veya (32, 16, 8)) doğruluğun nasıl değiştiğini gözlemleyin.
2. `alpha` (L2 düzenlileştirme) değerini artırıp azaltarak etkisini inceleyin.
3. `early_stopping=False` yapıp modelin aşırı öğrenmeye başladığı noktayı gözlemleyin.
4. Bu örneği tam MNIST veri setiyle (28x28 piksel, `fetch_openml('mnist_784')`) tekrarlayın.
"""
