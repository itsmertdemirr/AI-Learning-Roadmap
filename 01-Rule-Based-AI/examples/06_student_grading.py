"""
06 - Öğrenci Notlandırma Sistemi (Kural Tabanlı Yapay Zeka)
===============================================================

Sayısal bir notu harf notuna ve kişiselleştirilmiş geri bildirime çeviren
kural tabanlı bir karar ağacı. Bir "karar ağacının" doğrudan iç içe
EĞER-O HALDE mantığı olarak nasıl uygulanabileceğini gösterir (ML eğitimi
gerekmez).
"""


def grade_student(score: float) -> tuple[str, str]:
    """
    Sayısal notu (harf notu, geri bildirim) ikilisine eşleyen EĞER-O HALDE
    karar ağacı. Aralık dışı notlar için ValueError fırlatır.
    """
    if not (0 <= score <= 100):
        raise ValueError("not 0 ile 100 arasında olmalıdır")

    if score >= 90:
        return "A", "Olağanüstü bir çalışma! Böyle devam edin."
    elif score >= 80:
        return "B", "Harika iş, sadece biraz daha cila gerekiyor."
    elif score >= 70:
        return "C", "Sağlam bir anlayış, zayıf konuları gözden geçirin."
    elif score >= 60:
        return "D", "Geçtiniz, ama daha derin bir tekrar şiddetle önerilir."
    else:
        return "F", "Ek çalışma ve destek gerekiyor. Pes etmeyin!"


def class_summary(scores: dict[str, float]) -> None:
    """{öğrenci_adı: not} sözlüğü için biçimlendirilmiş bir karne yazdırır."""
    print(f"{'Öğrenci':<12}{'Not':<8}{'Harf':<8}Geri Bildirim")
    print("-" * 60)
    for name, score in scores.items():
        letter, feedback = grade_student(score)
        print(f"{name:<12}{score:<8}{letter:<8}{feedback}")


def main() -> None:
    scores = {
        "Ayşe": 95,
        "Mert": 82,
        "Elif": 71,
        "Can": 58,
        "Deniz": 40,
    }
    class_summary(scores)

    # Kural tabanlı sınıf istatistikleri
    average = sum(scores.values()) / len(scores)
    passing = sum(1 for s in scores.values() if s >= 60)
    print(f"\nSınıf ortalaması: {average:.1f}")
    print(f"Geçen öğrenciler: {passing}/{len(scores)}")


if __name__ == "__main__":
    main()

"""
İyileştirme Fikirleri
-----------------------
1. Ağırlıklı notlandırmayı destekleyin (ödev %20, sınavlar %80).
2. Eğri/normalizasyon kuralları ekleyin.
3. Karneyi bir CSV veya PDF olarak dışa aktarın.
"""
