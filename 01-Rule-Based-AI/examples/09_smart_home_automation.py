"""
09 - Akıllı Ev Otomasyonu (Kural Tabanlı Yapay Zeka)
========================================================

Işıkları, termostatı ve güvenliği kontrol etmek için EĞER-O HALDE
kurallarını kullanarak sensör okumalarına tepki veren bir akıllı ev
merkezini modeller -- gerçek "akıllı ev" ürünlerinin herhangi bir ML
eklenmeden önce perde arkasında nasıl çalıştığının tam olarak aynısı.
"""

from dataclasses import dataclass


@dataclass
class SensorData:
    time_of_day: str        # "morning", "day", "evening", "night"
    motion_detected: bool
    temperature_c: float
    door_open: bool


def evaluate_rules(sensors: SensorData) -> list[str]:
    """Mevcut sensör anlık görüntüsüne karşı her otomasyon kuralını çalıştırır."""
    actions: list[str] = []

    # Kural 1: zamana + harekete göre aydınlatma
    if sensors.motion_detected and sensors.time_of_day in {"evening", "night"}:
        actions.append("💡 Işıkları AÇ (karanlıkta hareket algılandı).")
    elif not sensors.motion_detected and sensors.time_of_day in {"evening", "night"}:
        actions.append("💡 Enerji tasarrufu için ışıkları KAPALI tut (hareket yok).")

    # Kural 2: termostat kontrolü
    if sensors.temperature_c < 18:
        actions.append("🔥 Isıtmayı AÇ (konfor eşiğinin altında).")
    elif sensors.temperature_c > 26:
        actions.append("❄️ Klimayı AÇ (konfor eşiğinin üstünde).")
    else:
        actions.append("🌡️ Sıcaklık konforlu; HVAC boşta bekliyor.")

    # Kural 3: güvenlik uyarısı
    if sensors.door_open and sensors.time_of_day == "night" and not sensors.motion_detected:
        actions.append("🚨 UYARI: Gece kapı açık ve tanınan bir hareket deseni yok!")

    return actions


def main() -> None:
    scenarios = [
        SensorData(time_of_day="night", motion_detected=True, temperature_c=15, door_open=False),
        SensorData(time_of_day="night", motion_detected=False, temperature_c=21, door_open=True),
        SensorData(time_of_day="day", motion_detected=True, temperature_c=29, door_open=False),
    ]

    for i, sensors in enumerate(scenarios, start=1):
        print(f"\nSenaryo {i}: {sensors}")
        for action in evaluate_rules(sensors):
            print("  ", action)


if __name__ == "__main__":
    main()

"""
İyileştirme Fikirleri
-----------------------
1. Kural önceliği ekleyin (güvenlik uyarıları enerji tasarrufu kurallarını geçersiz kılmalıdır).
2. Zamanlama kuralları ekleyin (örn. "hareketten bağımsız olarak 01:00-06:00 arası hep KAPALI").
3. Denetim için her tetiklenen kuralı bir zaman damgasıyla kaydedin.
"""
