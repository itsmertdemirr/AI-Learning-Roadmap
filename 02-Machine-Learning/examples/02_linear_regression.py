"""
02 - Doğrusal Regresyon (Makine Öğrenmesi)
==============================================

scikit-learn'in LinearRegression'ını kullanarak tek bir özellikten
(metrekare cinsinden büyüklük) sürekli bir değeri (ev fiyatı) tahmin eder,
ardından uydurulmuş doğruyu matplotlib ile görselleştirir.

Gereksinimler: scikit-learn, numpy, matplotlib
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # başsız (headless) çalışması için etkileşimsiz backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def generate_synthetic_housing_data(n_samples: int = 50, seed: int = 42):
    """Gerçekçi gürültüye sahip sentetik bir (büyüklük_m2 -> fiyat) veri seti oluşturur."""
    rng = np.random.default_rng(seed)
    size_m2 = rng.uniform(40, 200, n_samples)
    noise = rng.normal(0, 15000, n_samples)
    price = size_m2 * 1200 + 20000 + noise  # kabaca doğrusal bir ilişki
    return size_m2.reshape(-1, 1), price


def main() -> None:
    X, y = generate_synthetic_housing_data()

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    print(f"Öğrenilen formül: fiyat = {model.coef_[0]:.2f} * büyüklük_m2 + {model.intercept_:.2f}")
    print(f"Ortalama Kare Hata (MSE): {mse:,.2f}")
    print(f"R² Skoru: {r2:.3f}  (1.0'a ne kadar yakınsa uyum o kadar iyi)")

    # Yeni bir örneği tahmin et
    new_size = np.array([[120]])
    predicted_price = model.predict(new_size)[0]
    print(f"\n120 m²'lik bir ev için tahmini fiyat: {predicted_price:,.2f}")

    # Grafik
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, alpha=0.6, label="Gerçek veri")
    plt.plot(X, predictions, color="red", linewidth=2, label="Öğrenilen regresyon doğrusu")
    plt.xlabel("Büyüklük (m²)")
    plt.ylabel("Fiyat")
    plt.title("Doğrusal Regresyon: Ev Fiyatı vs. Büyüklük")
    plt.legend()
    plt.tight_layout()
    plt.savefig("linear_regression_output.png")
    print("\nGrafik linear_regression_output.png dosyasına kaydedildi")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Doğrusal Regresyon, tahminler ile gerçek değerler arasındaki toplam kare
hatayı EN AZA İNDİREN doğruyu bulur -- bu "bir kayıp fonksiyonunu en aza
indirme" fikri, Bölüm 3'te (Derin Öğrenme) gradyan inişi olarak her yerde
tekrar karşımıza çıkar.
"""
