"""
05 - L1 ve L2 Düzenlileştirme (Regularization)
==================================================

Aşırı öğrenmeyi (overfitting) azaltmak için ağırlıkların büyümesini
cezalandıran L1 (Lasso) ve L2 (Ridge) düzenlileştirmeyi, gürültülü
sentetik bir veri seti üzerinde scikit-learn ile karşılaştırır.

Gereksinimler: numpy, scikit-learn
"""

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def generate_noisy_data(n_samples: int = 40, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Az sayıda örnek + gürültü ile aşırı öğrenmeye YATKIN bir veri seti üretir."""
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(-3, 3, n_samples)).reshape(-1, 1)
    y = 0.5 * X.ravel() ** 2 - X.ravel() + rng.normal(0, 3, n_samples)
    return X, y


def main() -> None:
    X, y = generate_noisy_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Kasıtlı olarak yüksek dereceli polinom özellikler üret -> aşırı öğrenmeye zorla
    poly = PolynomialFeatures(degree=12)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Farklı ölçeklerdeki polinom terimlerini standartlaştır -> daha kararlı
    # optimizasyon ve daha anlamlı bir alpha karşılaştırması sağlar
    scaler = StandardScaler()
    X_train_poly = scaler.fit_transform(X_train_poly)
    X_test_poly = scaler.transform(X_test_poly)

    models = {
        "Düzenlileştirmesiz (Aşırı Öğrenmeye Açık)": LinearRegression(),
        "L2 (Ridge, alpha=5.0)": Ridge(alpha=5.0),
        "L1 (Lasso, alpha=0.05)": Lasso(alpha=0.05, max_iter=20000),
    }

    print(f"{'Model':<42}{'Eğitim MSE':<15}{'Test MSE':<15}{'Sıfır Olmayan Katsayı'}")
    print("-" * 95)
    for name, model in models.items():
        model.fit(X_train_poly, y_train)
        train_pred = model.predict(X_train_poly)
        test_pred = model.predict(X_test_poly)

        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        nonzero_coefs = int(np.sum(np.abs(model.coef_) > 1e-6))

        print(f"{name:<42}{train_mse:<15.2f}{test_mse:<15.2f}{nonzero_coefs}")

    print("\nGözlem: Düzenlileştirmesiz model eğitim verisinde çok düşük hataya sahip")
    print("olabilir ama test verisinde kötü genelleyebilir -- bu aşırı öğrenmenin")
    print("klasik belirtisidir. L1, bazı katsayıları tam olarak SIFIRA iterek")
    print("otomatik özellik seçimi de yapar; L2 katsayıları küçültür ama nadiren")
    print("tam sıfıra indirir.")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
L1 ve L2, kayıp fonksiyonuna ağırlıkların büyüklüğüne bağlı bir ceza terimi
ekler:
  L1 cezası:  alpha * sum(|w_i|)       -> bazı ağırlıkları tam sıfır yapar (seyreklik)
  L2 cezası:  alpha * sum(w_i^2)       -> tüm ağırlıkları küçültür (daha pürüzsüz)
Sinir ağlarında bu, "weight decay" olarak optimize ediciye doğrudan
gömülü şekilde de uygulanabilir (örn. PyTorch'ta `weight_decay` parametresi).

İyileştirme Fikirleri
-----------------------
1. alpha değerini artırıp azaltarak yetersiz/aşırı öğrenme dengesini gözlemleyin.
2. Elastic Net (L1 + L2 karışımı) ekleyin ve üçünü karşılaştırın.
3. Bu tekniği bir sinir ağının ağırlıklarına manuel olarak (Bölüm 3'teki
   sıfırdan ağa) uygulayın.
"""
