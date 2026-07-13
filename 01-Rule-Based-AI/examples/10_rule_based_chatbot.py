"""
10 - Kural Tabanlı Sohbet Botu
==================================

1966'nın klasik ELIZA programı tarzında, hiçbir öğrenme olmadan anahtar
kelime eşleştirme EĞER-O HALDE kuralları kullanan bir sohbet botu.
Bölüm 6'ya (NLP) köprü görevi görür; orada bunu gerçek dil anlayışıyla
yeniden inşa edeceğiz.
"""

import random

# Sıralı (anahtar kelimeler, olası yanıtlar) listesi.
# (Liste sırasına göre) İLK eşleşen kural kazanır -> bu, basit bir
# "önceliğe dayalı" ileri zincirlemeyi modeller.
RULES: list[tuple[list[str], list[str]]] = [
    (["hello", "hi", "hey", "merhaba", "selam"], ["Merhaba! Bugün nasılsın?", "Selam!"]),
    (["bye", "goodbye", "görüşürüz", "hoşça kal"], ["Hoşça kal! Kendine iyi bak.", "Görüşmek üzere!"]),
    (["sad", "unhappy", "depressed", "üzgün", "mutsuz"], ["Bunu duyduğuma üzüldüm. Konuşmak ister misin?"]),
    (["happy", "great", "good", "mutlu", "harika", "iyi"], ["Bunu duymak harika!"]),
    (["name", "isim", "adın"], ["Ben öğrenme amaçlı yapılmış basit bir kural tabanlı sohbet botuyum."]),
    (["help", "yardım"], ["Selamlaşmalara, duygulara ve temel sorulara yanıt verebilirim. 'merhaba' veya 'nasılsın' diyerek dene."]),
]

DEFAULT_RESPONSES = [
    "Tam olarak anladığımdan emin değilim. Bunu yeniden ifade edebilir misin?",
    "İlginç -- biraz daha anlat.",
    "Hmm, henüz bunun için bir kuralım yok.",
]


def get_response(user_input: str) -> str:
    """
    EĞER girdi bir kurala bağlı herhangi bir anahtar kelime içeriyorsa
    O HALDE o kuraldan (rastgele seçilmiş) bir yanıt döndür.
    DEĞİLSE varsayılan bir yanıta geri dön.
    """
    text = user_input.lower()

    for keywords, responses in RULES:
        if any(keyword in text for keyword in keywords):
            return random.choice(responses)

    return random.choice(DEFAULT_RESPONSES)


def main() -> None:
    conversation = [
        "Merhaba!",
        "Bugün biraz üzgün hissediyorum.",
        "Senin adın ne?",
        "Bana yardım edebilir misin?",
        "Mars'ta hava nasıl?",
        "Hoşça kal!",
    ]

    for message in conversation:
        print(f"Kullanıcı: {message}")
        print(f"Bot     : {get_response(message)}\n")


if __name__ == "__main__":
    main()

"""
İyileştirme Fikirleri
-----------------------
1. Kullanıcının girdisinin bir kısmını yakalayıp yeniden kullanmak için
   düzenli ifade (regex) tabanlı kurallar ekleyin (klasik ELIZA numarası:
   "Ben X'im" -> "Neden X'sin?").
2. Birden çok tur boyunca konuşma bağlamını (durumu) takip edin.
3. Bunu Bölüm 6'daki gerçek bir NLP tabanlı sohbet botuyla karşılaştırın --
   bu botun SADECE anahtar kelime eşleştirmesi yaptığını, gerçek bir
   anlayışa sahip olmadığını fark edin!
"""
