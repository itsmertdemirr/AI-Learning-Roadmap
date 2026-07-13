"""
08 - ATM Menü Simülasyonu (Kural Tabanlı Yapay Zeka)
========================================================

PIN doğrulaması, bakiye kontrolleri, para çekme limitleri ve para yatırma
için EĞER-O HALDE kuralları kullanarak bir ATM'yi simüle eder. DEĞİŞKEN
DURUMU (hesap bakiyesi) güvenli bir şekilde yöneten kural tabanlı bir
sistemi gösterir.
"""

from dataclasses import dataclass

DAILY_WITHDRAWAL_LIMIT = 1000
CORRECT_PIN = "1234"


@dataclass
class Account:
    balance: float
    withdrawn_today: float = 0.0


def verify_pin(entered_pin: str) -> bool:
    """Kural: PIN, saklanan PIN ile tam olarak eşleşmelidir (gerçek hayatta asla düz metin saklamayın!)."""
    return entered_pin == CORRECT_PIN


def withdraw(account: Account, amount: float) -> str:
    """
    Kurallar:
      1. Miktar pozitif olmalıdır
      2. Miktar, kalan günlük limiti aşmamalıdır
      3. Miktar, hesap bakiyesini aşmamalıdır
    """
    if amount <= 0:
        return "❌ Çekim miktarı pozitif olmalıdır."

    remaining_limit = DAILY_WITHDRAWAL_LIMIT - account.withdrawn_today
    if amount > remaining_limit:
        return f"❌ Günlük çekim limiti aşıldı. Bugün en fazla {remaining_limit} daha çekebilirsiniz."

    if amount > account.balance:
        return "❌ Yetersiz bakiye."

    account.balance -= amount
    account.withdrawn_today += amount
    return f"✅ {amount} verildi. Yeni bakiye: {account.balance}"


def deposit(account: Account, amount: float) -> str:
    """Kural: yatırılan miktar pozitif olmalıdır."""
    if amount <= 0:
        return "❌ Yatırma miktarı pozitif olmalıdır."
    account.balance += amount
    return f"✅ {amount} yatırıldı. Yeni bakiye: {account.balance}"


def main() -> None:
    account = Account(balance=500)

    print("PIN kontrolü ('9999'):", "✅ doğru" if verify_pin("9999") else "❌ yanlış")
    print("PIN kontrolü ('1234'):", "✅ doğru" if verify_pin("1234") else "❌ yanlış")

    print(withdraw(account, 200))
    print(withdraw(account, 900))   # kalan bakiyeyi/limiti aşıyor
    print(deposit(account, 100))
    print(withdraw(account, -5))


if __name__ == "__main__":
    main()

"""
İyileştirme Fikirleri
-----------------------
1. 3 yanlış denemeden sonra bir PIN kilitleme kuralı ekleyin (örnek 02'deki deseni tekrar kullanın!).
2. Hesap durumunu bir dosyaya/veritabanına kaydedin.
3. İşlem geçmişi kaydı ekleyin.
"""
