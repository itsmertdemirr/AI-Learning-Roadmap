"""
07 - Trafik Işığı Kontrolcüsü (Kural Tabanlı Sonlu Durum Makinesi)
======================================================================

Kural tabanlı yapay zeka genellikle bir SONLU DURUM MAKİNESİ biçimini
alır: durumlar arasında geçiş için EĞER-O HALDE kurallarına sahip sabit
bir durum kümesi. Bu, birçok gömülü/endüstriyel kontrol sisteminin tam
olarak çalışma şeklidir.
"""

import time
from enum import Enum, auto


class LightState(Enum):
    RED = auto()
    GREEN = auto()
    YELLOW = auto()


# Geçiş kuralları: durum -> sonraki durum
TRANSITIONS: dict[LightState, LightState] = {
    LightState.RED: LightState.GREEN,
    LightState.GREEN: LightState.YELLOW,
    LightState.YELLOW: LightState.RED,
}

# Her durumun ne kadar süreceği (saniye) -- başka bir basit kural tablosu
DURATIONS: dict[LightState, float] = {
    LightState.RED: 4,
    LightState.GREEN: 3,
    LightState.YELLOW: 1,
}


class TrafficLightController:
    """Mevcut durumu ve geçiş kural motorunu kapsüller."""

    def __init__(self) -> None:
        self.state: LightState = LightState.RED

    def next_state(self) -> LightState:
        """Mevcut durum için geçiş kuralını uygular."""
        self.state = TRANSITIONS[self.state]
        return self.state

    def duration(self) -> float:
        return DURATIONS[self.state]


def main(simulate_real_time: bool = False, cycles: int = 6) -> None:
    controller = TrafficLightController()
    print(f"Başlangıç durumu: {controller.state.name}")

    for _ in range(cycles):
        wait_time = controller.duration()
        print(f"  -> {controller.state.name} durumunda {wait_time}sn kal")
        if simulate_real_time:
            time.sleep(wait_time)
        controller.next_state()


if __name__ == "__main__":
    # simulate_real_time=False, böylece örnek öğrenme amaçlı anında çalışır
    main(simulate_real_time=False)

"""
İyileştirme Fikirleri
-----------------------
1. Normal döngüyü kesen bir yaya geçiş butonu kuralı ekleyin.
2. Sensör tabanlı kurallar ekleyin (trafik sensörü kuyruk algılarsa YEŞİL süresini uzatın).
3. Senkronize iki kontrolcüye sahip bir kavşağı modelleyin (K-G ve D-B).
"""
