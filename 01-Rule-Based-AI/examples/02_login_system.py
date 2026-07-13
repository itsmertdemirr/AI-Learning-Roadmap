"""
02 - Giriş Sistemi (Kural Tabanlı Yapay Zeka)
=================================================

Kural tabanlı bir kimlik doğrulama/yetkilendirme akışını gösterir:
EĞER-O HALDE kuralları, bir giriş denemesinin izin verilip verilmeyeceğine
ve kullanıcının hangi rol tabanlı izinleri alacağına karar verir.

Gösterilen kavram: bilgi tabanı + basit durum (başarısız deneme sayısı)
birlikte kullanılır; kuralların sadece mevcut girdiye değil geçmişe de
bağlı olabileceğini gösterir.
"""

from dataclasses import dataclass, field

# --- "Veritabanı" (kullanıcıların bellek içi bilgi tabanı) --------------
USERS_DB = {
    "admin": {"password": "admin123", "role": "admin"},
    "mert": {"password": "securepass", "role": "user"},
}

MAX_FAILED_ATTEMPTS = 3


@dataclass
class LoginState:
    """Kilitleme kurallarını desteklemek için kullanıcı başına giriş denemelerini izler."""
    failed_attempts: dict[str, int] = field(default_factory=dict)
    locked_users: set[str] = field(default_factory=set)


def attempt_login(state: LoginState, username: str, password: str) -> str:
    """
    Kimlik doğrulama kurallarını uygular ve okunabilir bir sonuç döndürür.

    Kurallar:
      1. EĞER kullanıcı kilitliyse O HALDE hemen reddet.
      2. EĞER kullanıcı adı veritabanında yoksa O HALDE reddet ("bilinmeyen kullanıcı").
      3. EĞER şifre uyuşmuyorsa O HALDE başarısızlık sayısını artır;
         EĞER başarısızlıklar MAX_FAILED_ATTEMPTS'e ulaşırsa O HALDE hesabı kilitle.
      4. EĞER kimlik bilgileri eşleşiyorsa O HALDE role göre erişim ver.
    """
    if not username or not password:
        raise ValueError("kullanıcı adı ve şifre boş olamaz")

    # Kural 1: hesap kilitlenmesi
    if username in state.locked_users:
        return f"⛔ '{username}' hesabı çok fazla başarısız denemeden dolayı kilitli."

    # Kural 2: bilinmeyen kullanıcı
    if username not in USERS_DB:
        return "❌ Giriş başarısız: bilinmeyen kullanıcı adı."

    # Kural 3: şifre kontrolü
    if USERS_DB[username]["password"] != password:
        state.failed_attempts[username] = state.failed_attempts.get(username, 0) + 1
        remaining = MAX_FAILED_ATTEMPTS - state.failed_attempts[username]

        if state.failed_attempts[username] >= MAX_FAILED_ATTEMPTS:
            state.locked_users.add(username)
            return f"⛔ Çok fazla başarısız deneme. '{username}' hesabı artık kilitli."

        return f"❌ Yanlış şifre. {remaining} deneme hakkınız kaldı."

    # Kural 4: başarılı giriş -> role göre mesaj
    state.failed_attempts[username] = 0  # başarıda sıfırla
    role = USERS_DB[username]["role"]
    if role == "admin":
        return f"✅ Hoş geldiniz, {username}! Yönetici paneli açıldı."
    return f"✅ Hoş geldiniz, {username}! Normal kullanıcı olarak giriş yaptınız."


def main() -> None:
    state = LoginState()
    attempts = [
        ("mert", "wrongpass"),
        ("mert", "wrongpass"),
        ("mert", "securepass"),   # yine de sorun yok, deneme sayısı her yanlışta artıyor...
        ("ghost", "whatever"),
        ("admin", "admin123"),
    ]

    for username, password in attempts:
        print(f"Giriş denemesi -> kullanıcı='{username}' -> {attempt_login(state, username, password)}")


if __name__ == "__main__":
    main()

"""
İyileştirme Fikirleri
-----------------------
1. Şifreleri düz metin karşılaştırmak yerine hash'leyin (üretimde asla böyle yapmayın!).
2. N dakika sonra otomatik kilit açan zamana dayalı bir kilitleme kuralı ekleyin.
3. Ek bir aşama olarak çok faktörlü kimlik doğrulama ekleyin.
4. Kural eşiklerini (MAX_FAILED_ATTEMPTS) bir yapılandırma dosyasına taşıyın.
"""
