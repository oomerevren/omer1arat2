# Submission Day Checklist

## A. Config kontrolü

- [ ] Doğru family seçildi mi? A / B / C
- [ ] `configs/kaggle/final/<family>.yaml` kullanılıyor mu?
- [ ] `final_mode: true` var mı?
- [ ] `frozen: true` var mı?
- [ ] `shared_final_base.yaml` referansı var mı?
- [ ] Experimental veya legacy config kullanılmadığından emin misin?

## B. Artefact kontrolü

- [ ] `reports/final/final_artifact_manifest.json` mevcut mu?
- [ ] Family metadata mevcut mu?
- [ ] Model artefact path’leri mevcut mu veya eksiklik bilinçli not edilmiş mi?
- [ ] Blend weights toplamı 1 mi?
- [ ] Threshold metadata ile config threshold uyumlu mu?
- [ ] OOF rapor path’leri erişilebilir mi?

## C. Submission üretim kontrolü

- [ ] `data/submission_pairs.csv` doğru official dosya mı?
- [ ] `artifacts/final/submissions/<family>_submission.csv` doğru dosya mı?
- [ ] Dosya adı family ile uyumlu mu?
- [ ] Aynı gün yanlış/eski submission üretilmedi mi?
- [ ] Fallback/legacy script kullanılmadı mı?

## D. Validator gate

- [ ] Submission validator geçti mi?
- [ ] Kolonlar sadece `id,prediction` mi?
- [ ] Satır sayısı `submission_pairs.csv` ile aynı mı?
- [ ] Id seti birebir aynı mı?
- [ ] Id sırası bozulmamış mı?
- [ ] Prediction sadece 0/1 mi?
- [ ] Null prediction yok mu?
- [ ] Positive rate aşırı uçta değil mi?

## E. Registry / LB notu

- [ ] Submission registry’ye işlendi mi?
- [ ] Public LB not alanı hazır mı?
- [ ] Public skor girilecekse `analyze_leaderboard_correlation.py --add-public-entry` kullanılacak mı?
- [ ] Public artışı OOF/class0 ile çelişiyor mu?

## F. Son karar

- [ ] Family A varsayılan olarak korunuyor mu?
- [ ] Family B’ye geçiş class0/private-risk gerekçeli mi?
- [ ] Family C yalnızca risk flags düşükse mi deneniyor?
- [ ] `PUBLIC_UP_OOF_DOWN` varsa submission durduruldu mu?
- [ ] `PUBLIC_UP_CLASS0_DOWN` varsa submission durduruldu mu?

## En kritik 5 adım

1. Final family config gerçekten `configs/kaggle/final/` altında mı?
2. Manifest ve config checksum kayıtları güncel mi?
3. Submission validator geçti mi?
4. Registry kaydı oluştu mu?
5. Public LB’ye göre karar verirken OOF/class0/risk flag önceliği korundu mu?
