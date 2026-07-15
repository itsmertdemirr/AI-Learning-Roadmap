"""
08 - Elle Tasarlanmış Özellikler vs. Ham Pikseller
=======================================================

CNN'ler icat edilmeden önce, bilgisayarlı görü büyük ölçüde ELLE
TASARLANMIŞ özellik çıkarma yöntemlerine dayanıyordu: bir uzman, hangi
görüntü örüntülerinin önemli olduğuna karar verip bunları elle kodlardı
(Örnek 04'teki Sobel gibi). Bu örnek, ham piksel değerleriyle eğitilen
bir sınıflandırıcıyı, Sobel tabanlı elle tasarlanmış özelliklerle
eğitilen bir sınıflandırıcıyla karşılaştırır.

Gereksinimler: numpy, scipy, scikit-learn
"""

import numpy as np
from scipy.signal import convolve2d
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


def extract_sobel_features(image_8x8: np.ndarray) -> np.ndarray:
    """8x8'lik bir rakam görüntüsünden elle tasarlanmış özellikler çıkarır:
    Sobel gradyan büyüklüğü istatistikleri (ortalama, std, maksimum) +
    ham piksel değerlerinin kendisi."""
    gx = convolve2d(image_8x8, SOBEL_X, mode="same", boundary="symm")
    gy = convolve2d(image_8x8, SOBEL_Y, mode="same", boundary="symm")
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    features = np.concatenate([
        image_8x8.flatten(),           # ham pikseller (64 özellik)
        magnitude.flatten(),            # kenar büyüklüğü haritası (64 özellik)
        [magnitude.mean(), magnitude.std(), magnitude.max()],  # özet istatistikler (3 özellik)
    ])
    return features


def main() -> None:
    digits = load_digits()
    X_images = digits.images  # şekil: (n_örnek, 8, 8)
    y = digits.target

    X_train_img, X_test_img, y_train, y_test = train_test_split(
        X_images, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Yaklaşım 1: Ham piksel değerleri ---
    X_train_raw = X_train_img.reshape(len(X_train_img), -1)
    X_test_raw = X_test_img.reshape(len(X_test_img), -1)

    # --- Yaklaşım 2: Sobel tabanlı elle tasarlanmış özellikler ---
    X_train_handcrafted = np.array([extract_sobel_features(img) for img in X_train_img])
    X_test_handcrafted = np.array([extract_sobel_features(img) for img in X_test_img])

    print(f"Ham piksel özellik boyutu:              {X_train_raw.shape[1]}")
    print(f"Elle tasarlanmış özellik boyutu:        {X_train_handcrafted.shape[1]}\n")

    results = {}
    for name, X_train, X_test in [
        ("Ham Pikseller", X_train_raw, X_test_raw),
        ("Elle Tasarlanmış (Sobel) Özellikler", X_train_handcrafted, X_test_handcrafted),
    ]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=300, random_state=42, early_stopping=True
        )
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
        print(f"{name:<40} Test Doğruluğu: {accuracy:.2%}")

    print("\nGözlem: Sobel tabanlı özellikler, sınıflandırıcıya kenar")
    print("bilgisini doğrudan sunarak yardımcı olabilir -- ama bu özellikleri")
    print("TASARLAMAK için bir insan uzmanın Sobel operatörünü, hangi")
    print("istatistiklerin faydalı olacağını vb. önceden bilmesi gerekiyordu.")
    print("Bir CNN ise (Örnek 07), BENZER kenar dedektörlerini VERİDEN")
    print("OTOMATİK olarak öğrenir -- insan uzmanlığına ihtiyaç duymadan.")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Bu, derin öğrenmenin bilgisayarlı görüde neden devrim yarattığını
gösteren tarihsel olarak önemli bir karşılaştırmadır. 2012'den önce
(AlexNet'in ImageNet yarışmasını kazanmasından önce), araştırmacılar
SIFT, HOG, Sobel gibi elle tasarlanmış özellik çıkarıcılar üzerinde
yıllarını harcıyordu. CNN'ler bu süreci OTOMATİKLEŞTİRDİ: ağa "hangi
özelliklerin önemli olduğunu SEN bul" deme imkanı sağladı. Bu örnekte
küçük ölçekte gördüğünüz fark (elle tasarlanmış Sobel özellikleri), büyük
ölçekte (gerçek CNN'lerin öğrendiği binlerce özellik) çok daha
belirgindir.

İyileştirme Fikirleri
-----------------------
1. HOG (Histogram of Oriented Gradients) özelliklerini
   (`skimage.feature.hog`) ekleyip üçüncü bir yaklaşım olarak karşılaştırın.
2. Farklı `hidden_layer_sizes` değerleriyle her iki yaklaşımın da
   performansının nasıl değiştiğini test edin.
3. Bu deneyi, Örnek 07'deki gerçek (rastgele başlatılmış olsa da) CNN
   özellik haritalarını sınıflandırıcıya besleyerek üçüncü bir "yarı-CNN"
   yaklaşımı olarak genişletin.
"""
