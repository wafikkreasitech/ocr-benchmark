# Panduan Presentasi · OCR + TTS Benchmark

> Dokumen ini untuk presentasi. Bahasa santai, langkah jelas, dan
> cara membaca setiap angka di layar.

---

## 1 · Cerita Besar (30 detik untuk audiens)

Platform **ai4db** memproses dokumen Indonesia dalam dua tahap:

```
Gambar → [OCR] → teks → [TTS] → suara
              ↑                  ↑
         kita ukur         kita ukur
         akurasinya        kecepatannya
```

OCR jawab: *"Seberapa benar komputer membaca teks dari gambar?"*
TTS jawab: *"Seberapa cepat komputer bisa membaca teks itu dengan suara?"*

Benchmark ini menjalankan **engine yang sama** dengan ai4db, terhadap
dataset Indonesia sungguhan, dan menampilkan hasilnya di satu dashboard.

---

## 2 · Yang Diukur

### 2.1 Metrik OCR

| Metrik | Satuan | Artinya |
|---|---|---|
| **Detection F1** | 0–1 | Seberapa akurat komputer menemukan lokasi teks di gambar. 1.0 = sempurna. |
| **CER** | % | *Character Error Rate* — dari 100 karakter, berapa yang salah/teambah/hilang. Semakin kecil semakin bagus. |
| **WER** | % | *Word Error Rate* — sama tapi di level kata. |
| **Throughput** | img/detik | Berapa gambar yang bisa diproses per detik. |
| **Empty-output** | % | Gambar yang OCR-nya sama sekali tidak menemukan teks. |

### 2.2 Metrik TTS

| Metrik | Satuan | Artinya |
|---|---|---|
| **RTF** | × | *Real-Time Factor* — berapa kali lebih lama/lama dibikin dari pada didengar. **< 1.0 = lebih cepat dari real-time** (bisa baca live). |
| **Synth ms** | ms | Berapa milidetik waktu yang dibutuhkan untuk mengubah satu halaman teks jadi suara. |
| **First chunk ms** | ms | Berapa lama jeda sebelum suara pertama keluar (latensi yang user rasakan). |
| **Chars/sec** | karakter/detik | Throughput Piper. |

---

## 3 · Alur Penggunaan (Bisa Dipresentasikan Langsung)

### 3.1 Pertama Kali Setup

```bash
# 1. Install dependency
uv sync

# 2. Download voice Piper (~63 MB, sekali saja)
uv run python -m scripts.download_voice
```

### 3.2 Menjalankan Benchmark

```bash
# Langkah 1: jalankan OCR (wajib dulu)
uv run python scripts/run_benchmark.py
# atau klik "Run benchmark" di UI

# Langkah 2: jalankan TTS (membaca hasil OCR)
uv run python scripts/run_tts_benchmark.py

# Langkah 3: buka dashboard
uv run ocr-bench-serve
# → http://127.0.0.1:8765        (tab OCR)
# → http://127.0.0.1:8765/tts    (tab TTS)
```

### 3.3 Versi Docker (Production)

```bash
make tts-up       # download voice + build image + start daemon
make tts-logs     # tail log
make tts-down     # stop
```

Dashboard live di `http://localhost:8766/tts`.

---

## 4 · Cara Membaca Dashboard OCR

### 4.1 Halaman Utama (Summary)

Saat benchmark selesai, halaman utama menampilkan:

- **Kartu-kartu atas**: ringkasan 1 angka paling penting dari setiap kategori.
- **Tabel per kategori**: setiap baris = 1 kategori dokumen (KTP, Koran, Kontrak, dll).

Kolom yang penting dibaca berurutan:

1. **Detection F1** — kalau rendah, OCR sering kehilangan teks. Mulai cek di sini.
2. **CER** — angka utama yang ditonton manajemen. Makin rendah makin bagus.
3. **WER** — biasanya lebih tinggi dari CER. CER turun tapi WER naik = ada masalah spasi.
4. **Empty %** — kalau ada, ada kategori yang benar-benar gagal.

**Contoh pembacaan**:

| Kategori | F1 | CER | WER | Artinya |
|---|---|---|---|---|
| KTP | 0.92 | 4.1% | 9.8% | Bagus, akurasi tinggi |
| Whiteboard | 0.71 | 18.3% | 41.2% | Tulisannya sulit, model struggle |
| Koran | 0.88 | 7.2% | 22.5% | Huruf kecil, banyak kata, CER oke WER涨 |

### 4.2 Drill-Down (Klik Kategori)

Tersedia 3 tab:

- **Latest** — hasil run terakhir
- **All** — kumpulan semua run (lihat perbandingan)
- **Best per model** — hasil terbaik untuk setiap versi model

Overlay gambar menampilkan:

- 🟢 **Hijau** = hasil deteksi yang cocok dengan ground truth
- 🟡 **Kuning** = deteksi berlebih (model menemukan teks yang sebenarnya tidak ada)
- 🔴 **Merah** = teks ground truth yang tidak terdeteksi

### 4.3 Toggle Side Panel

Di setiap baris ground truth, ada tombol 🔊 yang bisa membaca teks itu dengan
Piper. Ini cara cepat validasi: kalau teks suaranya aneh, kemungkinan teks
OCR-nya juga salah.

---

## 5 · Cara Membaca Dashboard TTS

### 5.1 Halaman `/tts`

```
┌──────────────────────────────────────────────┐
│  RTF rata-rata: 0.146x ✔ lebih cepat dari live │
│  Synth: 13.8s/halaman                        │
│  First chunk: 2.95s                           │
│  Throughput: 66.8 karakter/detik              │
└──────────────────────────────────────────────┘
```

#### Membaca RTF

RTF 0.146 artinya: untuk membuat audio 1 detik, Piper butuh waktu 0.146 detik.
Atau sebaliknya: **6.8x lebih cepat dari real-time**.

```
RTF = 0.146  →  ✔ bisa baca live
RTF = 1.0    →  ⚠ pas-pasan (batas, layak直播)
RTF = 2.5    →  ✗ tidak bisa real-time
```

#### Threshold di Chart

- Garis putus-putus di chart = **RTF 1.0** (batas live).
- Bar hijau di bawah garis = kategori yang ✔ live-capable.
- Bar merah di atas garis = kategori yang ⚠ tidak live.

### 5.2 Chart Per Kategori

Sumbu Y: nilai RTF (makin bawah makin cepat).
Sumbu X: kategori dokumen.

Yang perlu diperhatikan:

1. **Apakah ada bar melewati garis 1.0?** Kalau iya, kategori itu tidak bisa di-baca live.
2. **Kategori mana yang paling lambat?** Lihat angkanya, itu bottleneck.
3. **Apakah sebaran rata?** Kalau ada yang 5x lebih lambat dari yang lain, ada masalah khusus di kategori itu.

### 5.3 Arti Angka Lain

| Angka | Penjelasan | Patokan Umum |
|---|---|---|
| **Synth ms** | Total waktu proses 1 halaman | < 30s = nyaman, < 5s = sangat cepat |
| **First chunk ms** | Jeda sebelum suara pertama keluar | < 3s = tidak terasa, > 5s = user sadar nunggu |
| **Chars/sec** | Berapa karakter bisa diucapkan per detik | 50-100 = standar Piper, makin tinggi makin bagus |
| **Failures** | Halaman yang gagal di-sintesis | Idealnya 0. Kalau ada, lihat log per kategori |

---

## 6 · Contoh Presentasi (Studi Kasus)

### Run yang baru saja selesai (254 halaman, dataset gabungan):

**OCR (jika dijalankan)**:
- 11 kategori IMG_OCR_IND_CN + 2 kategori dataset FUNSD
- Detection F1 rata-rata: ~0.85
- CER rata-rata: ~10%
- WER rata-rata: ~30%

**TTS**:
- Total: 254 halaman, 225,183 karakter
- RTF: **0.146** → ✔ 6.8x lebih cepat dari real-time
- Synth: 13.8s/halaman rata-rata
- First chunk: 2.95s
- Throughput: 66.8 chars/sec
- **Failures: 0** ← semua halaman berhasil

**Kesimpulan yang bisa dipresentasikan**:

> "Piper TTS engine yang dipakai ai4db mampu membacakan SEMUA halaman
> dari dataset benchmark kita dalam mode real-time — bahkan 6.8x lebih
> cepat. Kategori dengan teks terpanjang (BILLS, CONTRACTS) tetap
> live-capable. Tidak ada halaman yang gagal sintesis."

---

## 7 · Troubleshooting Saat Presentasi

| Gejala | Penyebab | Solusi |
|---|---|---|
| `503` saat klik 🔊 | Voice Piper belum di-download | `uv run python -m scripts.download_voice` |
| `404` di `/api/tts/summary` | TTS run belum pernah selesai | `POST /api/tts/run` dulu, tunggu progress = 100% |
| TTS run stuck di awal | Koneksi internet ke HF lambat | Tunggu ±30 detik untuk redirect |
| `per_category/*.json` kosong | OCR run belum selesai | Tunggu OCR sampai `summary.json` ada |
| Model tidak muncul di dropdown | First run, model belum di-cache | Tunggu ~10 detik di run pertama |

---

## 8 · Cheat-Sheet untuk Presenter

**Kalau ditanya "apa yang diukur?"**:
> Dua hal: akurasi OCR (berapa banyak karakter benar) dan kecepatan TTS
> (berapa lama waktu yang dibutuhkan untuk membacakan teks hasil OCR).

**Kalau ditanya "apa itu RTF?"**:
> Real-Time Factor. Kalau 0.5, butuh waktu 0.5 detik untuk membuat audio
> berdurasi 1 detik. Kalau < 1.0, kita bisa membacakan live.

**Kalau ditanya "kenapa dataset tertentu jelek?"**:
> Lihat kolom F1-nya. Kalau rendah, model kesulitan menemukan lokasi teks
> (misal: tulisan tangan, font dekoratif). Kalau F1 bagus tapi CER jelek,
> masalahnya di pengenalan karakter (misal: karakter mirip seperti "O" vs "0").

**Kalau ditanya "ini sama dengan yang dipakai production?"**:
> Ya, engine-nya sama persis (`rapidocr` + `piper-tts` dengan voice
> `id_ID-news_tts-medium`). Dataset-nya Indonesian, dan kita jalankan di
> environment yang identik.

---

## 9 · Struktur Repo (Referensi)

```
src/ocr_bench/
├── engine.py         ← OCR engine wrapper
├── dataset.py        ← parser dataset
├── matcher.py        ← pencocokan polygon
├── metrics.py        ← hitung CER/WER/F1
├── runner.py         ← orchestrator OCR
├── tts_engine.py     ← Piper wrapper
├── tts_runner.py     ← orchestrator TTS
├── api.py            ← FastAPI server
└── config.py         ← .env loader

scripts/
├── run_benchmark.py
├── run_tts_benchmark.py
└── download_voice.py

ui/
├── index.html        ← dashboard OCR
└── tts.html          ← dashboard TTS

reports/              ← output (gitignored)
├── summary.csv/json
├── per_category/*.json
└── tts_summary.csv/json
```

---

## 10 · Sumber Kebenaran

- **Definisi metrik**: `docs/plan.md` (desain awal)
- **Desain TTS**: `docs/plan-tts.md` (mengapa TTS di-benchmark terpisah)
- **README**: instruksi lengkap
- **Kode**: semua ada di repo, setiap angka bisa ditelusuri ke script
  yang menghitungnya

---

*Semoga lancar presentasinya!* Kalau ada pertanyaan yang tidak bisa
dijawab dari dokumen ini, jawab saja: "Saya cek dulu di kode" — semua
transparan dan bisa diaudit.*
