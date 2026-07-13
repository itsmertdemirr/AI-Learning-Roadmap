"""
05 - Restoran Önerisi Sistemi (Kural Tabanlı Yapay Zeka)
============================================================

Ağırlıklı EĞER-O HALDE kuralları kullanarak kullanıcı tercihlerine göre
bir restoran kategorisi önerir. Basit kural puanlamasının, hiçbir öğrenme
olmadan nasıl "öneri" davranışını simüle edebileceğini gösterir.
"""

from dataclasses import dataclass


@dataclass
class Preferences:
    budget: str          # "low", "medium", "high"
    mood: str             # "casual", "romantic", "family"
    wants_vegetarian: bool
    party_size: int


CATEGORIES = ["Fast Food", "Günlük Restoran", "Şık Restoran", "Vejetaryen Kafe", "Aile Büfesi"]


def score_category(category: str, prefs: Preferences) -> int:
    """Bir kategorinin tercihlere ne kadar uyduğuna dair kural tabanlı bir puan hesaplar."""
    score = 0

    # Kural: bütçe eşleşmesi
    if category == "Fast Food" and prefs.budget == "low":
        score += 3
    if category == "Şık Restoran" and prefs.budget == "high":
        score += 3
    if category == "Günlük Restoran" and prefs.budget == "medium":
        score += 2

    # Kural: ruh hali eşleşmesi
    if category == "Şık Restoran" and prefs.mood == "romantic":
        score += 3
    if category == "Aile Büfesi" and prefs.mood == "family":
        score += 3
    if category == "Günlük Restoran" and prefs.mood == "casual":
        score += 2

    # Kural: diyet gereksinimi
    if prefs.wants_vegetarian and category == "Vejetaryen Kafe":
        score += 4
    if prefs.wants_vegetarian and category in {"Fast Food", "Şık Restoran"}:
        score -= 1  # küçük ceza; iyi vejetaryen seçenekleri olma ihtimali daha düşük

    # Kural: grup büyüklüğü
    if prefs.party_size >= 6 and category == "Aile Büfesi":
        score += 2
    if prefs.party_size <= 2 and category == "Şık Restoran":
        score += 1

    return score


def recommend(prefs: Preferences) -> list[tuple[str, int]]:
    """Tüm kategorileri puana göre azalan sırada sıralar."""
    scored = [(cat, score_category(cat, prefs)) for cat in CATEGORIES]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


def main() -> None:
    prefs = Preferences(budget="high", mood="romantic", wants_vegetarian=False, party_size=2)
    print(f"Tercihler: {prefs}\n")

    for category, score in recommend(prefs):
        print(f"  {category:<20} puan = {score}")

    top_choice = recommend(prefs)[0][0]
    print(f"\n🍽️  Önerilen: {top_choice}")


if __name__ == "__main__":
    main()

"""
İyileştirme Fikirleri
-----------------------
1. Programcı olmayanların da ayarlayabilmesi için kuralları/ağırlıkları bir yapılandırma dosyasından yükleyin.
2. Kuralları gerçek verilerle birleştirmek için gerçek restoran verisi (puanlar, mesafe) ekleyin.
3. Bunu Bölüm 2'deki gerçek bir ML tabanlı öneri sistemiyle karşılaştırarak farkı görün!
"""
