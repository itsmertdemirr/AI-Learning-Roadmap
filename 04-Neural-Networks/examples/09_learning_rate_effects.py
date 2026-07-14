"""
09 - Öğrenme Oranının (Learning Rate) Etkisi
================================================

Basit bir kayıp yüzeyinde (parabol) çok küçük, iyi ayarlanmış ve çok
büyük öğrenme oranlarının gradyan inişinin davranışını nasıl
değiştirdiğini gösterir: yavaş yakınsama, düzgün yakınsama ve ıraksama
(diverge).

Gereksinimler: numpy
"""



def loss_function(x: float) -> float:
    """Basit bir parabol: minimumu x=3'te olan bir kayıp yüzeyi."""
    return (x - 3) ** 2


def gradient(x: float) -> float:
    """loss_function'ın türevi: d/dx (x-3)^2 = 2(x-3)."""
    return 2 * (x - 3)


def gradient_descent(start_x: float, lr: float, steps: int) -> list[float]:
    """Gradyan inişini uygular ve her adımdaki x değerini kaydeder."""
    x = start_x
    history = [x]
    for _ in range(steps):
        x = x - lr * gradient(x)
        history.append(x)
        if abs(x) > 1e6:  # ıraksama durumunda erken durdur
            break
    return history


def describe_run(name: str, history: list[float]) -> None:
    final_x = history[-1]
    final_loss = loss_function(final_x)
    status = "IRAKSADI (patladı!)" if final_loss > loss_function(history[0]) else "yakınsadı"
    print(f"{name}")
    print(f"  Adım sayısı: {len(history) - 1}")
    print(f"  Son x değeri: {final_x:.4f}  (hedef: 3.0)")
    print(f"  Son kayıp:    {final_loss:.6f}  ({status})")
    print(f"  İlk 5 adım:   {[round(v, 3) for v in history[:5]]}\n")


def main() -> None:
    start_x = 0.0
    steps = 20

    print("Kayıp fonksiyonu: (x - 3)^2  |  Minimum: x=3, kayıp=0\n")

    describe_run("1. Çok Küçük Öğrenme Oranı (lr=0.01) — yavaş ilerler",
                 gradient_descent(start_x, lr=0.01, steps=steps))

    describe_run("2. İyi Ayarlanmış Öğrenme Oranı (lr=0.3) — hızlı ve düzgün yakınsar",
                 gradient_descent(start_x, lr=0.3, steps=steps))

    describe_run("3. Sınırda Öğrenme Oranı (lr=0.9) — salınarak yakınsar",
                 gradient_descent(start_x, lr=0.9, steps=steps))

    describe_run("4. Çok Büyük Öğrenme Oranı (lr=1.1) — IRAKSAR",
                 gradient_descent(start_x, lr=1.1, steps=steps))


if __name__ == "__main__":
    main()

"""
Anahtar Fikir
--------------
Öğrenme oranı (lr), muhtemelen bir sinir ağını eğitirken ayarlayacağınız
EN ÖNEMLİ hiperparametredir:
  - Çok küçük  -> eğitim çok yavaş ilerler, pratikte hiç bitmeyebilir
  - İyi ayarlı -> hızlı ve düzgün bir şekilde minimuma yakınsar
  - Çok büyük  -> her adımda minimumun "üzerinden atlar", sonunda ıraksar
Bu yüzden modern optimize ediciler (Adam gibi) öğrenme oranını eğitim
sırasında OTOMATİK olarak ayarlar -- ama sabit bir lr kullanıyorsanız bile
neyi ayarladığınızı anlamak hata ayıklamada kritik önem taşır.

İyileştirme Fikirleri
-----------------------
1. "Öğrenme oranı programlaması" (learning rate scheduling) ekleyin: lr'yi
   her N adımda bir azaltın (örn. lr = lr * 0.9).
2. Momentum ekleyerek salınımın (lr=0.9 durumu) nasıl azaldığını gözlemleyin.
3. Bu deneyi Bölüm 3'teki sıfırdan sinir ağına uygulayıp farklı lr
   değerleriyle XOR eğitimi süresini karşılaştırın.
"""
