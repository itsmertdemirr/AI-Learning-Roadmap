# Güvenlik Politikası

## Desteklenen Sürümler

Bu depo eğitim amaçlı bir kaynaktır ve sürekli geliştirilmektedir.
Güvenlik güncellemeleri yalnızca `main` dalındaki en güncel içerik için
sağlanır.

| Sürüm | Destekleniyor mu? |
|-------|---------------------|
| main (en güncel) | ✅ |
| Eski etiketli sürümler | ❌ |

## Bir Güvenlik Açığı Bildirme

Bu depodaki örnek kodlarda bir güvenlik açığı (örneğin güvensiz bağımlılık,
kötü amaçlı kod enjeksiyonuna açık bir örnek, ya da hassas veri sızıntısına
yol açabilecek bir desen) bulursanız:

1. **Lütfen genel bir Issue AÇMAYIN.**
2. Bunun yerine proje yöneticileriyle özel olarak iletişime geçin (GitHub
   üzerinden "Security" sekmesini kullanarak özel bir danışmanlık raporu
   oluşturabilirsiniz).
3. Sorunu yeniden oluşturma adımlarını ve potansiyel etkisini açıklayın.

## Eğitim Amaçlı Kod Hakkında Not

Bu depodaki bazı örnekler (özellikle `01-Rule-Based-AI/examples/02_login_system.py`
ve `08_atm_simulation.py` gibi) **kasıtlı olarak basitleştirilmiştir** ve
gerçek üretim sistemlerinde kullanılmamalıdır. Örneğin:

- Şifreler düz metin olarak karşılaştırılır (gerçek sistemlerde her zaman
  `bcrypt`, `argon2` gibi güvenli hash algoritmaları kullanılmalıdır)
- Girdi doğrulaması eğitim amaçlı minimal tutulmuştur
- Kimlik doğrulama/yetkilendirme akışları basitleştirilmiştir

Bu tasarım kararları her ilgili dosyanın docstring'inde açıkça belirtilir.

## Bağımlılık Güvenliği

`Resources/requirements.txt` dosyasındaki bağımlılıklar GitHub Actions
üzerinden düzenli olarak (bkz. `.github/workflows/ci.yml`) otomatik test
edilir. Bilinen güvenlik açıklarını önlemek için bağımlılıkları güncel
tutmaya çalışıyoruz.
