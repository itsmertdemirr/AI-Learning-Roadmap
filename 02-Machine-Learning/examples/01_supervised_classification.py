"""
01 - Gözetimli Sınıflandırma (Makine Öğrenmesi)
===================================================

Tam gözetimli öğrenme iş akışını göstermek için klasik Iris veri seti
üzerinde bir Karar Ağacı sınıflandırıcısı eğitir: böl -> eğit -> değerlendir.
Bunu Bölüm 1'in EL İLE YAZILMIŞ karar ağaçlarıyla karşılaştırın -- burada
ağaç dalları bir insan tarafından yazılmak yerine verilerden ÖĞRENİLİR.

Gereksinimler: scikit-learn, numpy  (pip install scikit-learn numpy)
"""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    # 1. Veri setini yükle: özellikler (X) ve etiketler (y)
    data = load_iris()
    X, y = data.data, data.target

    # 2. Eğitim setine (desenleri öğrenir) ve test setine (genellemeyi değerlendirir) böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Eğit: model, eğitim verisinden desenleri ÖĞRENİR
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # 4. Görülmemiş test verisi üzerinde tahmin yap
    predictions = model.predict(X_test)

    # 5. Değerlendir
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Doğruluğu: {accuracy:.2%}\n")
    print("Sınıflandırma Raporu:")
    print(classification_report(y_test, predictions, target_names=data.target_names))

    # 6. Modelin ne öğrendiğini incele (özellik önemi)
    print("Özellik önemleri (hangi özellikler en çok önemliydi):")
    for name, importance in zip(data.feature_names, model.feature_importances_):
        print(f"  {name:<20}: {importance:.3f}")


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Dikkat edin, "petal_length > 2.5 ise" gibi bir şeyi kendimiz hiç yazmadık --
DecisionTreeClassifier, eğitim verisi üzerindeki sınıflandırma hatasını
en aza indirerek bu tür eşikleri otomatik olarak keşfetti. Bölüm 1'den
Bölüm 2'ye geçişteki temel değişim budur.
"""
