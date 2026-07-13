"""
Bölüm 1 — Alıştırma Çözümleri
=================================

exercises/exercises.md dosyasındaki 10 alıştırmanın referans çözümleri.
Önce kendiniz denemeden buraya bakmayın — çözüm görmek öğrenmeyi
kısaltır!

Her çözüm bağımsız çalıştırılabilir bir fonksiyon olarak yazılmıştır.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------
# Çözüm 1 — Hava Durumu Asistanına UV endeksi kuralı ekleme
# ---------------------------------------------------------------------
@dataclass
class WeatherReadingV2:
    temperature_c: float
    is_raining: bool
    wind_speed_kmh: float
    uv_index: int = 0


def get_weather_advice_v2(reading: WeatherReadingV2) -> list[str]:
    advice: list[str] = []
    if reading.temperature_c <= 0:
        advice.append("🥶 Hava dondurucu. Kalın mont giyin.")
    elif reading.temperature_c <= 25:
        advice.append("🧥 Ilıman/soğuk hava. Uygun bir katman giyin.")
    else:
        advice.append("🥵 Hava sıcak. Hafif kıyafet giyin.")

    if reading.uv_index >= 8:
        advice.append("🧴 UV endeksi yüksek! Güneş kremi sürün ve gölgede kalın.")
    return advice


# ---------------------------------------------------------------------
# Çözüm 2 — Zamana dayalı kilit açma
# ---------------------------------------------------------------------
@dataclass
class LoginStateV2:
    failed_attempts: dict[str, int] = field(default_factory=dict)
    locked_until_tick: dict[str, int] = field(default_factory=dict)
    LOCK_DURATION_TICKS: int = 5

    def is_locked(self, username: str, current_tick: int) -> bool:
        unlock_tick = self.locked_until_tick.get(username)
        if unlock_tick is None:
            return False
        if current_tick >= unlock_tick:
            del self.locked_until_tick[username]  # süre doldu, kilidi kaldır
            return False
        return True

    def register_failure(self, username: str, current_tick: int) -> None:
        self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
        if self.failed_attempts[username] >= 3:
            self.locked_until_tick[username] = current_tick + self.LOCK_DURATION_TICKS
            self.failed_attempts[username] = 0


# ---------------------------------------------------------------------
# Çözüm 3 — Operatör önceliğine sahip ifade hesap makinesi
# ---------------------------------------------------------------------
def evaluate_expression(expression: str) -> float:
    """'12 + 5 * 2' gibi bir ifadeyi çarpma/bölmeyi toplama/çıkarmadan
    önce uygulayarak değerlendirir. eval() KULLANILMAZ."""
    tokens = expression.split()

    # 1. geçiş: çarpma ve bölmeyi çöz
    pass1: list[str] = [tokens[0]]
    i = 1
    while i < len(tokens):
        op, num = tokens[i], float(tokens[i + 1])
        if op in ("*", "/"):
            prev = float(pass1.pop())
            pass1.append(str(prev * num if op == "*" else prev / num))
        else:
            pass1.append(op)
            pass1.append(str(num))
        i += 2

    # 2. geçiş: toplama ve çıkarmayı soldan sağa uygula
    result = float(pass1[0])
    i = 1
    while i < len(pass1):
        op, num = pass1[i], float(pass1[i + 1])
        result = result + num if op == "+" else result - num
        i += 2

    return result


# ---------------------------------------------------------------------
# Çözüm 4 — Güven puanlı semptom kontrolcüsü
# ---------------------------------------------------------------------
def diagnose_with_confidence(
    patient_symptoms: set[str], required_symptoms: set[str], condition: str
) -> tuple[str, float]:
    matched = patient_symptoms & required_symptoms
    confidence = (len(matched) / len(required_symptoms)) * 100
    return condition, round(confidence, 1)


# ---------------------------------------------------------------------
# Çözüm 6 — Ağırlıklı notlandırma
# ---------------------------------------------------------------------
def weighted_grade(components: dict[str, float]) -> float:
    weights = {"homework": 0.20, "midterm": 0.30, "final": 0.50}
    return sum(components[k] * weights[k] for k in weights)


def main() -> None:
    print("Çözüm 1:", get_weather_advice_v2(WeatherReadingV2(30, False, 5, uv_index=9)))
    print("Çözüm 3:", "12 + 5 * 2 =", evaluate_expression("12 + 5 * 2"))
    print("Çözüm 4:", diagnose_with_confidence(
        {"fever", "cough"}, {"fever", "cough", "sore_throat"}, "Soğuk Algınlığı"
    ))
    print("Çözüm 6:", "Ağırlıklı not =", weighted_grade(
        {"homework": 90, "midterm": 75, "final": 85}
    ))


if __name__ == "__main__":
    main()
