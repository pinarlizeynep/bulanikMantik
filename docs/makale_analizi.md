# Referans Makale Analizi (Checkpoint 1)

**Makale:** D. Soni, N. Kumar, "Fuzzy logic-based computation offloading technique in fog computing,"
*Concurrency and Computation: Practice and Experience*, 36(20), e8198, 2024. doi:10.1002/cpe.8198

---

## 1. Problem ve motivasyon
- IoT cihazları kaynak kısıtlıdır (işlemci, enerji, bant genişliği). Ağır işleri yerelde çalıştırmak hem yavaş hem enerji tüketicidir.
- İşler **edge**, **fog** veya **cloud** katmanına aktarılabilir (offloading). Her katmanın gecikme, işlem gücü, enerji ve maliyet profili farklıdır:
  - Edge: düşük gecikme, düşük işlem gücü, ücretsiz
  - Fog: orta düzey
  - Cloud: yüksek işlem gücü, yüksek gecikme, en pahalı
- Karar ortamı belirsiz ve dinamik: gelen istekler önceden bilinmez, ağ durumu ve VM yükü sürekli değişir.
- Literatürde biyolojiden esinlenen (bio-inspired) optimizasyon yöntemleri baskın; bu yöntemler parametre ayarı ister, stokastiktir ve hesaplama maliyetleri yüksektir. Bulanık mantık ise **düşük karmaşıklıkta**, dilsel kurallarla ve belirsizliğe toleranslı karar verir. Makalenin bulanık mantığı seçme gerekçesi budur.

## 2. Amaç ve katkılar
1. Heterojen işleri edge/fog/cloud arasında dağıtan, düşük karmaşıklıklı bulanık offloading algoritması
2. Orkestratör katmanı (fog broker) içeren hibrit cloud–fog–edge mimarisi
3. Kullanıcının seçtiği QoS önceliklerine (zaman / enerji / maliyet) göre dinamik karar
4. FogWorkflowSim ile simülasyon ve klasik hibrit yöntemle karşılaştırma

## 3. Matematiksel model (Bölüm 2.1)
| Büyüklük | Denklem |
|---|---|
| Yerel çalışma süresi | `T_L = C_n / V_L` (1) |
| Yerel enerji | `E_L = T_L · P_L` (3) |
| Yerel maliyet | `C_l = T_L · C_i · CCT` (6) |
| Yükleme süresi (Shannon) | `T_up = D_n / (W · log2(1 + P_up·Gain/N))`, `Gain = d^-4` (9) |
| Uzak çalışma süresi | `T_exe = C_n / V_node` (10) |
| Offloading süresi | `T_off = T_up + T_exe` (8, 11) |
| Makespan | `CT_last − ST_first` (12) |
| Yükleme enerjisi | `E_up = T_up · P_up` (13) |
| Offloading maliyeti | `C_up = T_up · C_i · CCT` (15) |
| Offloading koşulu | `T_off < T_L`, `E_up < E_L`, `C_off < C_L` (16–18) |

Görev toplam işlem süresi: `T_tot = T_transmission + T_exec + T_queue + T_response` (22).
Kuyrukta bekleyen işler deadline'a göre artan sırayla dizilir. Bulanık girdiler min–max normalizasyonundan geçer (23).

## 4. Bulanık çıkarım sistemi
**Tip:** Mamdani. AND için min, birleştirme (aggregation) için max, durulaştırma için ağırlık merkezi (COG, Denk. 21).

### 4.1 Üyelik fonksiyonları (Şekil 3'ten okunan değerler)
Üçgen MF: `tri(a, b, c)`. Omuz (yamuk) MF: `trap(a, b, c, d)`.

| Değişken | Evren | Terim 1 | Terim 2 | Terim 3 |
|---|---|---|---|---|
| İş listesi boyutu `jl` | [0, 40] | Small `trap(0,0,8,16)` | Medium `tri(8,24,40)` | Large `trap(32,38,40,40)` |
| VM kullanımı `vt` | [0, 200] | Poor `trap(0,0,40,80)` | Medium `tri(40,120,200)` | Heavy `trap(160,180,200,200)` |
| Ağ gecikmesi `nl` | [0, 30] | Low `trap(0,0,4,8)` | Average `tri(4,17,30)` | High `trap(20,28,30,30)` |
| Bant genişliği `bw` | [0, 20] | Low `trap(0,0,4,8)` | Average `tri(4,12,20)` | High `trap(16,18,20,20)` |
| Deadline `dl` | [0, 50] | Low `trap(0,0,10,20)` | Medium `tri(10,30,50)` | High `trap(40,48,50,50)` |
| **Çıktı** `O` | [0, 200] | Edge `tri(0,40,80)` | Fog `tri(40,100,160)` | Cloud `tri(120,160,200)` |

**Karar eşikleri:** `C < 60` ise Edge, `60 ≤ C ≤ 140` ise Fog, `C > 140` ise Cloud.

**Gözlemler:**
- MF'ler asimetrik. "Medium" terimleri evrenin büyük kısmını kaplıyor, "High/Large" terimleri ise dar. Bu yüzden orta değerlerde sistem ağırlıklı olarak Fog'a yöneliyor.
- Çıktı MF'lerinin kesişim noktaları (Edge/Fog ≈ 64, Fog/Cloud ≈ 137) eşiklerle (60/140) yaklaşık olarak örtüşüyor.
- Grafikler matplotlib/scikit-fuzzy üslubunda çizilmiş. Makalenin metninde ise hem scikit-fuzzy hem jFuzzyLogic kullanıldığı yazıyor; bu bir tutarsızlık.

### 4.2 Kural tabanı
- 5 girdi × 3 terim = **3⁵ = 243** olası kural (tam liste verilmemiş).
- Örnek kurallar:
  - R1: jl Small ∧ vt Poor ∧ nl High ∧ dl Low ∧ bw Low → **Edge**
  - R2: jl Small ∧ vt Heavy ∧ nl Low ∧ dl Medium ∧ bw Low → **Fog**
  - R3: jl Large ∧ vt Heavy ∧ nl Average ∧ dl High ∧ bw High → **Cloud**
- **Tablo 2:** Kullanıcının zaman/enerji/maliyet tercih derecelerine (L/M/H) göre hangi algoritmanın (A2–A5) seçileceğini belirleyen 16 kural. Bu tablo ikinci bir karar katmanıdır: hangi offloading stratejisinin çalışacağını seçer.

### 4.3 Algoritmalar
| Algoritma | Ne zaman çalışır | Hedef katman |
|---|---|---|
| Alg. 1 | Ana kontrolcü; kullanıcı seçimini (ST, SE, SC) okur | — |
| Alg. 2 | Yalnız maliyet öncelikli | Cloud |
| Alg. 3 | Yalnız enerji öncelikli | Fog |
| Alg. 4 | Yalnız zaman öncelikli | Edge |
| Alg. 5 | Birden fazla QoS önceliği | Hibrit: FIS çıktısına (C_p) ve λ eşiklerine göre edge/fog/cloud |

Makalede verilen karmaşıklık: O(n × |DeviceList| × |RS|).

## 5. Deneysel yapı
| Öğe | Değer |
|---|---|
| Simülatör | FogWorkflowSim (CloudSim/WorkflowSim tabanlı, Java) |
| Workflow | MONTAGE (varsayılan); ayrıca CyberShake, Epigenomics, LIGO, SIPHT destekleniyor |
| Zamanlama | MIN-MIN |
| İş sayısı | 20, 40, 60, 80, 100, 200, 300 |
| Deadline | 15 birim |
| Donanım | i7-3770, 24 GB RAM, Windows 10 |
| Baseline | Khan vd. (2023): fog–cloud arası iki yönlü (two-way) offloading, "klasik hibrit". SFogSim ile üretilmiş, SJF/LJF/FCFS kullanıyor |

**Tablo 3 — düğüm parametreleri:**
| Düğüm | MIPS | UpBw | DownBw | RAM | Çalışma gücü (mW) | Boşta güç (mW) | Maliyet |
|---|---|---|---|---|---|---|---|
| Cloud | 1600 | 300 | 500 | 40960 | 1600 | 1300 | 0.96 $ görev + 0.02 $ iletişim, 0.001 $/MIPS |
| Fog | 1300 | 500 | 800 | 4096 | 800 | 40 | 0.48 $ + 0.01 $ iletişim |
| Edge | 1000 | 800 | 1000 | 1024 | 700 | 30 | 0 |

Ayrıca: Cloudlet length=400000, fileSize=outputSize=300. VM: mips=1000, ram=512, bw=1000.

## 6. Sonuçlar
- **Şekil 4A–C:** Ortalama makespan, enerji ve maliyet, iş sayısı 20'den 300'e çıkarken her iki yöntemde de artıyor; önerilen yöntem her noktada biraz daha düşük. Grafiklerden okunan değerler `data/reference_paper_values.csv` dosyasında.
- **Toplam iyileşme:** makespan %7.51, enerji %4.63, maliyet %13.60.
- **Tablo 4:** QoS tercihine göre iş dağılımı:
  - Maliyet öncelikliyken işler Cloud'a kayıyor; maliyet 181 $'a kadar düşüyor.
  - Enerji öncelikliyken Edge'e kayıyor; enerji 44 J'e kadar düşüyor.
  - Zaman öncelikliyken Fog ağırlıklı; makespan 669 ms'ye kadar düşüyor.

## 7. Eleştirel değerlendirme
**Güçlü yönler**
- Düşük karmaşıklık; açıklanabilir, dilsel kurallar
- Kullanıcı tercihine duyarlı (zaman/enerji/maliyet) çok amaçlı karar
- Üç katmanlı hibrit mimari, gerçekçi bir senaryo

**Sınırlılıklar ve tutarsızlıklar**
1. **Tekrar üretilebilirlik zayıf:** Kod, veri ve tam kural tabanı (243 kural) paylaşılmamış. MF değerleri yalnızca şekillerden okunabiliyor.
2. **İki karar katmanı arasındaki bağ belirsiz:** 5 girdili FIS ile Tablo 2'deki 3 girdili (T/E/C) seçim kurallarının nasıl birleştiği açık değil.
3. **Baseline adilliği şüpheli:** Baseline sonuçları başka bir çalışmadan ve başka bir simülatörden (SFogSim) alınmış gibi görünüyor; aynı ortamda yeniden koşulduğu açık değil.
4. **İstatistik eksik:** Tekrar sayısı, standart sapma ve anlamlılık testi raporlanmamış ("several consecutive iterations").
5. **Sezgiye aykırı eşleşmeler:** Tablo 2'de maliyet önceliği → Cloud (Alg. 2). Oysa Tablo 3'e göre Cloud en pahalı katman. Tablo 4'teki "Cost" satırlarında da Cloud oranı arttıkça maliyet düşüyor; bu, Tablo 3'teki birim fiyatlarla çelişiyor.
6. **Küçük tutarsızlıklar:**
   - Şekil 4E'deki çubuklar (~7.2 / ~4.4 / ~13.3) metindeki yüzdelerle tam örtüşmüyor.
   - Denklem numarası atıfları kaymış: metin (19)–(20) yerine (17)–(18) diyor.
   - Alg. 4 "FogDeviceList" üzerinde Edge düğümü arıyor.
7. **Belirsizliğin modellenmesi:** Motivasyon "belirsizlik" olduğu halde Type-1 MF kullanılmış. Ölçüm gürültüsüne karşı dayanıklılık test edilmemiş. **Bu boşluk özgün katkımızın gerekçesidir.**
8. Güvenlik/gizlilik ve iş göçü (migration) kapsam dışı; yazarlar da bunu kabul ediyor.

## 8. Yeniden gerçekleştirme için çıkarımlar
- Şekil 3'teki MF'ler birebir uygulanacak (Bölüm 4.1).
- 243 kuralın tamamı verilmediği için, makaledeki 3 örnek kurala uyan ve fiziksel sezgiye dayanan bir kural üretme yöntemi kullanılacak. Örneğin:
  - Yük ve iş boyutu büyüdükçe → Cloud
  - Gecikme yüksek ve deadline sıkı → Edge
  - Bant genişliği düşük → Edge/Fog

  Bu yöntem raporda açıkça belgelenecek.
- FogWorkflowSim yerine, Tablo 3 parametrelerini kullanan Python tabanlı ayrık olay simülatörü yazılacak (MONTAGE benzeri DAG + MIN-MIN).
- Baseline (klasik iki yönlü fog–cloud offloading) **aynı simülatörde** yeniden koşulacak. Böylece makaledeki baseline adilliği sorunu da giderilmiş olacak.
- Deneyler birden fazla tohumla (seed) tekrarlanacak; ortalama ± standart sapma raporlanacak.
