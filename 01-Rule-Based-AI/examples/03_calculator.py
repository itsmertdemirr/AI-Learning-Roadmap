"""
03 - Kural Tabanlı Hesap Makinesi
====================================

Bir "hesap makinesinin" aslında bir operatör sembolünü bir davranışa
eşleyen bir dizi deterministik EĞER-O HALDE kuralı olduğunu gösterir.
Bu, gerçek yazılımlarda her yerde (yönlendiriciler, komut işleyiciler vb.)
kullanılan kural tabanlı dağıtım (dispatch) desenlerinin en basit örneğidir.
"""

from typing import Callable


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    # Kural: sıfıra bölme tanımsızdır -> açık bir hata fırlat
    if b == 0:
        raise ZeroDivisionError("Sıfıra bölünemez.")
    return a / b


# "Bilgi tabanı": operatör sembolü -> kural (fonksiyon) eşlemesi
OPERATIONS: dict[str, Callable[[float, float], float]] = {
    "+": add,
    "-": subtract,
    "*": multiply,
    "/": divide,
}


def calculate(a: float, operator: str, b: float) -> float:
    """
    EĞER operatör bilinen bir kuralla eşleşiyorsa O HALDE uygula,
    DEĞİLSE bir hata fırlat (eşleşen kural bulunamadı).
    """
    if operator not in OPERATIONS:
        raise ValueError(f"Bilinmeyen operatör '{operator}'. Desteklenenler: {list(OPERATIONS)}")
    return OPERATIONS[operator](a, b)


def main() -> None:
    test_cases = [
        (5, "+", 3),
        (10, "-", 4),
        (6, "*", 7),
        (20, "/", 5),
        (1, "/", 0),
    ]

    for a, op, b in test_cases:
        try:
            result = calculate(a, op, b)
            print(f"{a} {op} {b} = {result}")
        except ZeroDivisionError as e:
            print(f"{a} {op} {b} -> Hata: {e}")


if __name__ == "__main__":
    main()

"""
İyileştirme Fikirleri
-----------------------
1. Operatör önceliğini ve tam ifade dizelerini destekleyin ("2 + 3 * 4").
2. Üs alma (**) ve mod (%) kuralları ekleyin.
3. Gerçek bir kullanıcı arayüzünden gelen sayısal olmayan girdiler için doğrulama ekleyin.
"""
