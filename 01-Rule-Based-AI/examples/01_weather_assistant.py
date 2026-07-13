"""
01 - Hava Durumu Asistanı (Kural Tabanlı Yapay Zeka)
=======================================================

Sıcaklık ve hava koşullarına göre kıyafet/aktivite önerisi veren basit bir
EĞER-O HALDE kural tabanlı sistem.

Gösterilen kavram: küçük bir (koşul -> öneri) kuralları bilgi tabanı
üzerinde ileri zincirleme.

Çalıştırmak için:
    python 01_weather_assistant.py
"""

from dataclasses import dataclass


@dataclass
class WeatherReading:
    """Tek bir hava durumu gözlemini tutar."""
    temperature_c: float   # Santigrat cinsinden sıcaklık
    is_raining: bool
    wind_speed_kmh: float


def get_weather_advice(reading: WeatherReading) -> list[str]:
    """
    Bir hava durumu okumasına EĞER-O HALDE kuralları bilgi tabanını
    uygular ve okunabilir önerilerden oluşan bir liste döndürür.

    Bu "ileri zincirleme"dir: bilinen GERÇEKLERDEN (okuma) başlayıp
    koşulu eşleşen her kuralı tetikleriz.
    """
    if not isinstance(reading, WeatherReading):
        raise TypeError("reading bir WeatherReading nesnesi olmalıdır")

    advice: list[str] = []

    # --- Kural Tabanı ----------------------------------------------------
    # Kural 1: Dondurucu sıcaklıklar
    if reading.temperature_c <= 0:
        advice.append("🥶 Hava dondurucu. Kalın mont, eldiven ve bere giyin.")
    # Kural 2: Soğuk sıcaklıklar
    elif reading.temperature_c <= 15:
        advice.append("🧥 Hava soğuk. Ceket önerilir.")
    # Kural 3: Ilıman sıcaklıklar
    elif reading.temperature_c <= 25:
        advice.append("🙂 Hava ılıman. İnce bir katman yeterli.")
    # Kural 4: Sıcak sıcaklıklar
    else:
        advice.append("🥵 Hava sıcak. Hafif kıyafet giyin ve bol su için.")

    # Kural 5: Yağmur kıyafet önerisine ekleme yapar
    if reading.is_raining:
        advice.append("☔ Yağmur yağıyor. Şemsiye alın.")

    # Kural 6: Güçlü rüzgâr uyarısı
    if reading.wind_speed_kmh >= 40:
        advice.append("💨 Rüzgâr kuvvetli. Dışarıdaki gevşek nesneleri sabitleyin.")

    # Kural 7: Mükemmel dışarı çıkma koşulları
    if (10 <= reading.temperature_c <= 25) and not reading.is_raining and reading.wind_speed_kmh < 20:
        advice.append("🚶 Yürüyüş veya dış mekân aktivitesi için harika bir gün!")

    return advice


def main() -> None:
    sample_readings = [
        WeatherReading(temperature_c=-3, is_raining=False, wind_speed_kmh=10),
        WeatherReading(temperature_c=18, is_raining=True, wind_speed_kmh=15),
        WeatherReading(temperature_c=32, is_raining=False, wind_speed_kmh=5),
        WeatherReading(temperature_c=20, is_raining=False, wind_speed_kmh=45),
    ]

    for i, reading in enumerate(sample_readings, start=1):
        print(f"\n--- Okuma {i}: {reading} ---")
        for line in get_weather_advice(reading):
            print(" ", line)


if __name__ == "__main__":
    main()

"""
Beklenen Çıktı (kısaltılmış)
------------------------------
--- Okuma 1: WeatherReading(temperature_c=-3, ...) ---
  🥶 Hava dondurucu. Kalın mont, eldiven ve bere giyin.

--- Okuma 2: WeatherReading(temperature_c=18, ...) ---
  🧥 Hava soğuk. Ceket önerilir.
  ☔ Yağmur yağıyor. Şemsiye alın.
...

İyileştirme Fikirleri
-----------------------
1. Nem ve UV endeksi kuralları ekleyin.
2. Kuralları koda gömmek yerine bir JSON/YAML yapılandırmasından yükleyin.
3. Çelişen kuralların temiz şekilde çözülmesi için bir öncelik sistemi ekleyin.
4. Bunu, kuralları dinamik olarak kabul eden sınıf tabanlı bir
   "InferenceEngine" (Çıkarım Motoru) haline getirin.
"""
