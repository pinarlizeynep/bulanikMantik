# Bulanık Mantık Dönem Projesi — Çalışma Planı

**Referans makale:** D. Soni, N. Kumar, "Fuzzy logic-based computation offloading technique in fog computing,"
*Concurrency and Computation: Practice and Experience*, vol. 36, no. 20, e8198, 2024. doi:10.1002/cpe.8198

**Hedef bitiş:** 15 Kasım 2026 (resmi teslim: 16 Kasım 2026)
**Commit kuralı:** Öğrencinin talimatıyla commitleri Claude, öğrencinin git kimliğiyle atar (ortak yazar satırı eklenmez). 1. ve 2. gün 8 Kasım'da yapıldı; kalan günler 9 Kasım'dan itibaren.

---

## 1. Makalenin özeti (Checkpoint 1)

| Başlık | İçerik |
|---|---|
| Problem | IoT cihazlarından gelen iş akışı (workflow) görevlerinin edge / fog / cloud katmanlarından hangisine aktarılacağı (offloading) kararı; belirsiz ve hızlı değişen koşullar altında |
| Amaç | Makespan, enerji ve maliyeti (QoS) birlikte iyileştirmek |
| Yöntem | Mamdani tipi bulanık çıkarım sistemi (FIS) |
| Girdiler (5) | İş listesi boyutu `jl` (Small/Medium/Large), VM kullanımı `vt` (Poor/Medium/Heavy), ağ gecikmesi `nl` (Low/Average/High), bant genişliği `bw` (Low/Average/High), deadline `dl` (Low/Medium/High) |
| Çıktı | Karar `O` ∈ [0, 200]: < 60 → Edge, 60–140 → Fog, > 140 → Cloud |
| Üyelik fonksiyonları | Üçgen + sol/sağ açık omuz (L-R shoulder); Alg.5'te yamuk da geçiyor |
| Kural tabanı | 3^5 = 243 olası kural; QoS tercihine göre algoritma seçen 16 kural (Tablo 2) |
| Çıkarım / durulaştırma | min (AND), max toplama (MAM), ağırlık merkezi (COG) |
| Algoritmalar | Alg.1 ana kontrolcü → Alg.2 (maliyet→cloud), Alg.3 (enerji→fog), Alg.4 (zaman→edge), Alg.5 (çoklu QoS, hibrit) |
| Matematiksel model | Yerel/uzak çalıştırma süresi, Shannon tabanlı yükleme süresi, enerji, maliyet (Denk. 1–20) |
| Deney ortamı | FogWorkflowSim (Java), MONTAGE workflow, MIN-MIN zamanlama, 20–300 iş, deadline=15 |
| Baseline | Khan vd. 2023 — fog–cloud arası iki yönlü (two-way) "klasik hibrit" offloading |
| Sonuçlar | Makespan −%7.51, enerji −%4.63, maliyet −%13.60 (Şekil 4); katmana göre iş dağılımı (Tablo 4) |

### Makalenin sınırlılıkları (eleştirel değerlendirme için)
- Üyelik fonksiyonu parametreleri yalnızca şekilde verilmiş, sayısal değer yok.
- 243 kuralın tamamı verilmemiş; Tablo 2'deki 16 kural QoS tercihleri üzerinden tanımlı, 5 girdili FIS ile bağlantısı belirsiz.
- Hem scikit-fuzzy hem jFuzzyLogic kullanıldığı söyleniyor (tutarsız).
- Kod ve veri seti paylaşılmamış; baseline başka bir simülatörden (SFogSim) alınmış → karşılaştırmanın adilliği tartışmalı.
- İstatistiksel anlamlılık, tekrar sayısı, varyans raporlanmamış.
- Güvenlik/gizlilik ele alınmamış (yazarlar da belirtiyor).

---

## 2. Yeniden gerçekleştirme kapsamı (MRR önerisi)

FogWorkflowSim Java tabanlı ve makalenin tam kurulumu paylaşılmadığı için **Python'da hafif bir simülatör** yazılacak:

1. **Bulanık çıkarım motoru** (`src/fuzzy/`): 5 girdi, 1 çıktı, üçgen/omuz MF, Mamdani min–max, COG. Kendi motorumuz + scikit-fuzzy ile çapraz doğrulama.
2. **Ortam modeli** (`src/env/`): Tablo 3'teki Cloud / Fog / Edge düğüm parametreleri (MIPS, bant genişliği, güç, maliyet).
3. **İş akışı üretici** (`src/workflow/`): Montage benzeri DAG (20–300 iş).
4. **Zamanlayıcı** (`src/scheduler/`): MIN-MIN.
5. **Politikalar**: Baseline (klasik iki yönlü fog–cloud), Önerilen (bulanık edge/fog/cloud).
6. **Metrikler**: makespan, enerji, maliyet, katman dağılımı, karar süresi.
7. **Deneyler**: Şekil 4a–c (iş sayısına göre makespan/enerji/maliyet) ve Tablo 4 (QoS tercihine göre dağılım) yeniden üretilecek.

---

## 3. Özgün katkı adayları (Checkpoint 4)

| # | Katkı | Gerekçe | Ölçüm |
|---|---|---|---|
| A | **Interval Type-2 bulanık offloading + gürültü/robustness analizi** (önerilen) | Makalenin motivasyonu "belirsizlik"; Type-1 MF'ler ölçüm gürültüsünü modellemez | Gürültü seviyesine göre makespan/enerji/maliyet, karar kararlılığı, çalışma süresi |
| B | Mamdani vs Sugeno + durulaştırma yöntemleri karşılaştırması | Makale "düşük karmaşıklık" iddia ediyor ama ölçmüyor | QoS metrikleri + karar başına süre |
| C | Kural tabanı / MF parametre optimizasyonu (GA/PSO) | MF'ler elle seçilmiş | QoS iyileşmesi, ablation |

**Karar (8 Kas):** **A** ana katkı, **B** küçük ek ablation olarak.

## Alınan kararlar
- Özgün katkı: Interval Type-2 + gürültü/robustness (+ Mamdani/Sugeno ablation)
- Dil: Rapor ve sunum Türkçe; kod, yorumlar ve README İngilizce
- Ortam: python.org Python 3.12 + proje içi venv (`.venv`), `requirements.txt`
- GitHub: repo öğrencide mevcut; commitleri öğrenci atar

---

## 4. Günlük takvim

| Gün | Tarih | İş | Checkpoint | Önerilen commit(ler) |
|---|---|---|---|---|
| 1 | 8 Kas (Paz) | Yönerge + makale analizi, plan, repo iskeleti | CP1 | `Initial project structure`, `Add paper analysis and project plan` |
| 2 | 8 Kas (Paz) | Bulanık motor (MF, kurallar, Mamdani, COG) + testler | CP2 | `Implement fuzzy inference engine and paper FIS`, `Add membership function plots, FIS analysis and daily logs` |
| 3 | 9 Kas (Pzt) | Ortam modeli, workflow üretici, MIN-MIN, baseline politika | CP3 | `Fix review findings in fuzzy engine and pin dependencies`, `Add fog/cloud/edge environment model, MONTAGE workflow and MIN-MIN scheduler`, `Add baseline two-way offloading` ✅ |
| 4 | 10 Kas (Sal) | Önerilen bulanık politika, Şekil 4 + Tablo 4 yeniden üretimi | CP3 | `Count idle power only for used nodes and expose layer load`, `Add fuzzy offloading policy with QoS algorithm selection`, `Reproduce Fig.4 and Table 4` ✅ |
| 5 | 11 Kas (Çar) | Özgün katkı: Type-2 FIS + gürültü deneyleri | CP4–5 | `Add interval type-2 FIS`, `Add robustness experiments` |
| 6 | 12 Kas (Per) | Ablation (Sugeno/durulaştırma), tablolar/grafikler, literatür taraması | CP5 | `Add ablation study`, `Add result tables and figures` |
| 7 | 13 Kas (Cum) | Teknik rapor (8–12 sayfa, IEEE), README | — | `Add technical report draft`, `Write README` |
| 8 | 14 Kas (Cmt) | Sunum (15–20 dk), canlı demo | FINAL | `Add presentation` |
| — | 15 Kas (Paz) | Son kontroller, PDF'ler, temiz klondan tekrar üretilebilirlik testi | — | `Final report and cleanup` |
| — | 16 Kas | Classroom'a yükleme (resmi teslim) | — | — |

---

## 5. Repo yapısı (yönerge Bölüm 9)

```
fuzzy-logic-project/
├── README.md, LICENSE, requirements.txt, .gitignore
├── src/            (fuzzy/, env/, workflow/, scheduler/, policies/)
├── data/           (üretilen workflow'lar, seed'ler)
├── experiments/baseline/, experiments/proposed/
├── results/figures/, results/tables/
├── demo/           (canlı demo arayüzü)
├── presentation/presentation.pdf
├── report/report.pdf
└── references/README.md
```

Not: Referans makale PDF'i telif nedeniyle repoya eklenmemeli (`.gitignore`), yalnızca DOI ile atıf yapılmalı.
