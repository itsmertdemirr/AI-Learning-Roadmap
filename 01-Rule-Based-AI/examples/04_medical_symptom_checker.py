"""
04 - Tıbbi Semptom Kontrolcüsü (Uzman Sistem)
=================================================

Bir "Uzman Sistemin" klasik bir örneği -- insan uzman bilgisini (bir
doktorun sezgilerini) EĞER-O HALDE kuralları olarak kodlayan ve GERİ
ZİNCİRLEME yapan bir kural tabanlı yapay zeka: bir hedeften ("bu durum ne
olabilir?") başlayıp hangi kuralların koşullarının hastanın semptomları
tarafından karşılandığını kontrol eder.

UYARI: Bu basitleştirilmiş bir eğitim örneğidir, gerçek bir tıbbi tavsiye
DEĞİLDİR.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    """Tek bir uzman sistem kuralı: gerekli semptomlar -> olası durum."""
    required_symptoms: frozenset[str]
    condition: str
    advice: str


KNOWLEDGE_BASE: list[Rule] = [
    Rule(
        required_symptoms=frozenset({"fever", "cough", "sore_throat"}),
        condition="Soğuk Algınlığı / Grip",
        advice="Dinlenin, bol sıvı için ve vücut ısınızı takip edin. Kötüleşirse doktora görünün.",
    ),
    Rule(
        required_symptoms=frozenset({"fever", "rash", "joint_pain"}),
        condition="Olası Viral Enfeksiyon",
        advice="Değerlendirme için derhal tıbbi yardım alın.",
    ),
    Rule(
        required_symptoms=frozenset({"chest_pain", "shortness_of_breath"}),
        condition="Olası Kalp veya Solunum Sorunu",
        advice="⚠️ Derhal acil tıbbi yardım alın.",
    ),
    Rule(
        required_symptoms=frozenset({"headache", "nausea", "sensitivity_to_light"}),
        condition="Olası Migren",
        advice="Karanlık, sessiz bir odada dinlenin. Sık tekrarlıyorsa doktora danışın.",
    ),
]


def diagnose(symptoms: set[str]) -> list[Rule]:
    """
    Geri zincirleme tarzı kontrol: her kural (aday teşhis) için, gerekli
    tüm semptomların mevcut olup olmadığını doğrular ("bu hipotez kanıtlar
    tarafından destekleniyor mu?").

    Gerekli semptom kümesi, hastanın bildirdiği semptomların bir alt kümesi
    olan her kuralı döndürür.
    """
    if not symptoms:
        raise ValueError("En az bir semptom belirtilmelidir.")

    matches = [
        rule for rule in KNOWLEDGE_BASE
        if rule.required_symptoms.issubset(symptoms)
    ]
    return matches


def main() -> None:
    patient_symptoms = {"fever", "cough", "sore_throat", "fatigue"}
    print(f"Hasta semptomları: {patient_symptoms}\n")

    results = diagnose(patient_symptoms)
    if not results:
        print("Bilgi tabanında eşleşen bir durum bulunamadı. Bir doktora danışın.")
    else:
        for rule in results:
            print(f"🩺 Olası durum: {rule.condition}")
            print(f"   Tavsiye: {rule.advice}\n")


if __name__ == "__main__":
    main()

"""
İyileştirme Fikirleri
-----------------------
1. Güven puanları ekleyin (örn. 3/3 semptom eşleşmesi = yüksek güven).
2. Tam alt küme yerine minimum eşikli kısmi eşleşmelere izin verin.
3. Hangi kuralın neden tetiklendiğini gösteren gerçek bir "açıklama"
   özelliği ekleyin (1970'lerin MYCIN gibi gerçek uzman sistemlerinin
   yaptığı tam olarak buydu).
"""
