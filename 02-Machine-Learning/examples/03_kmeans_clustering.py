"""
03 - K-Means Kümeleme (Gözetimsiz Öğrenme)
==============================================

Etiketlenmemiş müşteri verilerini K-Means kullanarak kümeler halinde
gruplayarak Gözetimsiz Öğrenmeyi gösterir: "doğru bir cevap" etiketi
yoktur -- algoritma verideki yapıyı kendi başına keşfeder.

Gereksinimler: scikit-learn, numpy
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def generate_customer_data(seed: int = 7):
    """3 doğal gruba sahip sentetik (yıllık_harcama, aylık_ziyaret) verisi."""
    rng = np.random.default_rng(seed)
    budget_shoppers = rng.normal(loc=[200, 2], scale=[40, 0.5], size=(30, 2))
    regular_shoppers = rng.normal(loc=[800, 6], scale=[80, 1], size=(30, 2))
    premium_shoppers = rng.normal(loc=[2500, 10], scale=[200, 1.5], size=(30, 2))
    return np.vstack([budget_shoppers, regular_shoppers, premium_shoppers])


def main() -> None:
    X = generate_customer_data()

    # Her iki sütun da mesafeye eşit katkıda bulunsun diye özellikleri standartlaştır
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # k=3 ile K-Means (algoritmaya kaç grup bulacağını biz söylüyoruz)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    print("Küme atama sayıları:")
    for cluster_id in sorted(set(cluster_labels)):
        count = int(np.sum(cluster_labels == cluster_id))
        avg_spend = X[cluster_labels == cluster_id, 0].mean()
        avg_visits = X[cluster_labels == cluster_id, 1].mean()
        print(f"  Küme {cluster_id}: {count} müşteri | "
              f"ort. harcama={avg_spend:.0f}, ort. ziyaret/ay={avg_visits:.1f}")

    print("\nAnahtar Fikir: modele 'bütçe/düzenli/premium' ne demek")
    print("olduğunu asla söylemedik -- bunu tamamen sayılardan kendisi keşfetti.")


if __name__ == "__main__":
    main()
