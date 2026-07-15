"""
10 - Uçtan Uca Görüntü Sınıflandırma Projesi
================================================

Bu bölümün tüm kavramlarını (ön işleme, evrişim tabanlı özellik
çıkarma, veri artırma) birleştirerek el yazısı rakam sınıflandırmada
mümkün olan en iyi performansı elde etmeye çalışan kapsamlı bir
pipeline. Ham piksellerle eğitilen temel (baseline) bir modeli, evrişim
tabanlı özellik çıkarma + veri artırma kullanan gelişmiş bir modelle
karşılaştırır.

Gereksinimler: numpy, scipy, scikit-learn, matplotlib
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


def extract_features(image_8x8: np.ndarray) -> np.ndarray:
    """Ham piksel + Sobel kenar özelliklerini birleştiren özellik vektörü."""
    gx = convolve2d(image_8x8, SOBEL_X, mode="same", boundary="symm")
    gy = convolve2d(image_8x8, SOBEL_Y, mode="same", boundary="symm")
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return np.concatenate([image_8x8.flatten(), magnitude.flatten()])


def augment_image(image_8x8: np.ndarray, seed: int) -> np.ndarray:
    """Küçük bir rastgele parlaklık/gürültü varyasyonu uygular
    (Örnek 06'daki tekniklerin küçük ölçekli bir versiyonu)."""
    rng = np.random.default_rng(seed)
    factor = rng.uniform(0.85, 1.15)
    noisy = image_8x8 * factor + rng.normal(0, 0.3, image_8x8.shape)
    return np.clip(noisy, 0, 16)  # digits veri seti 0-16 aralığında


def build_augmented_dataset(
    images: np.ndarray, labels: np.ndarray, n_augmentations: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Eğitim setini, her görüntü için n_augmentations adet yapay
    varyasyon ekleyerek genişletir."""
    rng = np.random.default_rng(seed)
    augmented_images = [images]
    augmented_labels = [labels]

    for i in range(n_augmentations):
        aug_seed_base = int(rng.integers(0, 1_000_000))
        augmented = np.array([
            augment_image(img, seed=aug_seed_base + idx) for idx, img in enumerate(images)
        ])
        augmented_images.append(augmented)
        augmented_labels.append(labels)

    return np.concatenate(augmented_images), np.concatenate(augmented_labels)


def main() -> None:
    digits = load_digits()
    X_images, y = digits.images, digits.target

    X_train_img, X_test_img, y_train, y_test = train_test_split(
        X_images, y, test_size=0.2, random_state=42, stratify=y
    )

    print("=" * 70)
    print("MODEL A: Temel Model (ham pikseller, veri artırma yok)")
    print("=" * 70)
    X_train_a = X_train_img.reshape(len(X_train_img), -1)
    X_test_a = X_test_img.reshape(len(X_test_img), -1)

    scaler_a = StandardScaler()
    X_train_a_scaled = scaler_a.fit_transform(X_train_a)
    X_test_a_scaled = scaler_a.transform(X_test_a)

    model_a = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                             random_state=42, early_stopping=True)
    model_a.fit(X_train_a_scaled, y_train)
    predictions_a = model_a.predict(X_test_a_scaled)
    accuracy_a = accuracy_score(y_test, predictions_a)
    print(f"Test Doğruluğu: {accuracy_a:.2%}\n")

    print("=" * 70)
    print("MODEL B: Gelişmiş Model (Sobel özellikleri + veri artırma)")
    print("=" * 70)
    X_train_aug, y_train_aug = build_augmented_dataset(X_train_img, y_train, n_augmentations=2)
    print(f"Veri artırma sonrası eğitim seti boyutu: {len(X_train_img)} -> {len(X_train_aug)}")

    X_train_b = np.array([extract_features(img) for img in X_train_aug])
    X_test_b = np.array([extract_features(img) for img in X_test_img])

    scaler_b = StandardScaler()
    X_train_b_scaled = scaler_b.fit_transform(X_train_b)
    X_test_b_scaled = scaler_b.transform(X_test_b)

    model_b = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                             random_state=42, early_stopping=True, alpha=1e-3)
    model_b.fit(X_train_b_scaled, y_train_aug)
    predictions_b = model_b.predict(X_test_b_scaled)
    accuracy_b = accuracy_score(y_test, predictions_b)
    print(f"Test Doğruluğu: {accuracy_b:.2%}\n")

    print("=" * 70)
    print("KARŞILAŞTIRMA")
    print("=" * 70)
    print(f"Model A (temel):                {accuracy_a:.2%}")
    print(f"Model B (özellik + artırma):     {accuracy_b:.2%}")
    print(f"Fark:                            {(accuracy_b - accuracy_a) * 100:+.2f} puan\n")

    print("En iyi modelin (Model B) sınıflandırma raporu:")
    best_predictions = predictions_b if accuracy_b >= accuracy_a else predictions_a
    print(classification_report(y_test, best_predictions))

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, best_predictions, ax=ax, cmap="Blues")
    ax.set_title(f"Karışıklık Matrisi (En İyi Model, Doğruluk: {max(accuracy_a, accuracy_b):.1%})")
    plt.tight_layout()
    plt.savefig("final_confusion_matrix.png", dpi=100)
    print("\nKarışıklık matrisi final_confusion_matrix.png dosyasına kaydedildi.")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Bu proje, bilgisayarlı görüde tipik bir mühendislik iş akışını gösterir:
temel (baseline) bir modelle başla, ardından ön işleme, özellik
mühendisliği ve veri artırma gibi teknikleri sistematik olarak ekleyip
etkilerini ÖLÇEREK ilerle. Gerçek dünyada bu iyileştirmeler her zaman
garanti değildir -- bazen elle tasarlanmış özellikler yardımcı olur,
bazen sadece gürültü ekler. Bunu varsaymak yerine ÖLÇMEK, gerçek bir
makine öğrenmesi mühendisinin günlük işidir.

İyileştirme Fikirleri
-----------------------
1. `n_augmentations` değerini artırıp azaltarak etkisini ölçün.
2. Model B'nin hangi rakamlarda Model A'dan daha iyi/kötü performans
   gösterdiğini karışıklık matrisinden analiz edin.
3. Bu projeyi tam MNIST veri setiyle (`fetch_openml('mnist_784')`)
   tekrarlayıp sonuçların büyük veri setinde nasıl değiştiğini gözlemleyin.
"""
