# Cancer Gene Community Detection

Aplikasi web interaktif untuk **Overlapping Community Detection (OCD)** pada **Heterogeneous Information Network (HIN)** gen kanker. Pengguna dapat mengeksplorasi struktur komunitas gen, gen overlapping (pleiotropic), dan korelasi antar gen berdasarkan embedding Metapath2Vec.

---

## Tentang Aplikasi

Aplikasi ini memvisualisasikan hasil deteksi komunitas tumpang tindih pada jaringan gen kanker. Pipeline yang digunakan:

```
Data Mentah (Heterogeneous):
   Drug ─── targets ─── Protein/Gen ─── interacts ─── Protein/Gen
            (Drug-Target)              (PPI)

           ↓ Metapath2Vec (Heterogeneous-aware Embedding)

   Embedding Vektor 64-dim per node

           ↓ Top-K Cosine Similarity (K=10)

   Weighted Homogeneous Network (Protein only)

           ↓ BigCLAM (Faithful Yang & Leskovec 2013)

   Komunitas Tumpang Tindih (Overlapping Communities)
```

**Output utama:** Setiap gen kanker dapat menjadi anggota lebih dari satu komunitas (pleiotropic), mencerminkan peran multifungsi gen dalam berbagai jalur pathway kanker.

---

## Fitur Aplikasi

Aplikasi memiliki **5 tab utama** yang dapat diakses melalui antarmuka Streamlit:

### Tab 1: Visualisasi Per Komunitas

Menampilkan grid subplot di mana **setiap panel merepresentasikan satu komunitas**.

- Background: semua node/edge dalam abu-abu tipis (sebagai konteks)
- Highlight: node dan edge dari komunitas yang sedang dilihat
- Ukuran node berbeda untuk gen biasa vs overlapping
- Layout konsisten antar panel (posisi gen tidak berubah)
- Update otomatis ketika nilai K diubah di sidebar

**Tujuan:** Melihat distribusi komunitas dalam konteks graf lengkap.

### Tab 2: Daftar Gen Per Komunitas

Menampilkan **semua gen anggota tiap komunitas** dalam dua mode:

**Mode Card (rinci):**
- Setiap komunitas ditampilkan dalam expander
- Default: semua expanded
- Menampilkan daftar gen lengkap per komunitas

**Mode Tabel (ringkas):**
- DataFrame interaktif dengan kolom: Komunitas, Jumlah Gen, Overlapping, Daftar Gen
- Dapat di-sort dan di-search
- Tombol download CSV untuk ekspor

**Tujuan:** Eksplorasi komposisi gen tiap komunitas.

### Tab 3: Gen Overlapping (Pleiotropic Genes)

Menampilkan **gen yang menjadi anggota lebih dari satu komunitas**.

Kolom yang ditampilkan:
- Entrez ID
- Gene Symbol
- Daftar komunitas (mis. "K3, K7")
- Jumlah komunitas

Diurutkan berdasarkan jumlah komunitas (gen paling pleiotropic di atas).

**Tujuan:** Identifikasi gen multifungsi yang berperan di banyak jalur biologis sekaligus.

### Tab 4: Kurva Optimasi K

Menampilkan **tren metrik kualitas vs jumlah komunitas (K)** dari hasil eksperimen K=1 hingga K=30.

Tiga metrik yang dapat dipilih:
- **Overlap Modularity** (Newman-Girvan, semakin tinggi semakin baik)
- **Internal Density** (kepadatan internal, semakin tinggi semakin baik)
- **Conductance** (separasi antar komunitas, semakin rendah semakin baik)

**Tujuan:** Memahami pemilihan K optimal dan trade-off antar metrik.

### Tab 5: Gene Explorer

Fitur **eksplorasi gen individual** dengan auto-generated insight biologis.

**Cara pakai:**
1. Pilih gen dari dropdown (atau ketik untuk auto-filter)
2. Aplikasi menampilkan:

**Section A — Info Dasar:**
- Gene Symbol dan Entrez ID
- Jumlah komunitas yang dimasuki
- Status: Pleiotropic atau Single Community

**Section B — Co-Members:**
- Daftar gen lain dalam komunitas yang sama
- Expandable per komunitas (untuk gen pleiotropic)

**Section C — Top 10 Gen Paling Mirip:**
- Berdasarkan cosine similarity dari embedding Metapath2Vec
- Tabel dengan kolom: Gene Symbol, Cosine Similarity, Komunitas Sama, Komunitas
- Cross-reference: kolom "Komunitas Sama" menunjukkan apakah gen serupa juga di komunitas yang sama

**Section D — Insight Biologis:**
- Pathway dari 3 database (KEGG, GO Biological Process, Reactome) yang mengandung gen ini
- Top 3 pathway per database, di-filter hanya yang mengandung gen target
- Adjusted P-value untuk tiap pathway

**Section E — Ringkasan Insight:**
- Natural language summary tentang gen
- Mengandung: status pleiotropic, peran fungsional, kandidat gen partner

**Tujuan:** Demonstrasi bahwa output BigCLAM **bermakna biologis** untuk gen spesifik.

---

## Sidebar — Konfigurasi K

Aplikasi memiliki dua kontrol K (jumlah komunitas) yang **sinkron**:

- **Slider:** geser untuk browse cepat (1 hingga 30)
- **Number Input:** ketik nilai presisi atau klik panah atas/bawah

Kedua kontrol terhubung via `st.session_state` — mengubah salah satu akan otomatis update yang lain. Nilai default = K optimal hasil training.

Sidebar juga menampilkan:
- K Optimal hasil training (info)
- K yang sedang aktif (success indicator)

---

## Metric Cards

Di bagian atas main panel, ditampilkan **4 metric cards** yang update real-time saat K diubah:

| Metrik | Arah |
|--------|------|
| Overlap Modularity | Higher = better |
| Internal Density | Higher = better |
| Conductance | Lower = better |
| Node Overlapping | Jumlah gen pleiotropic |

---

## Cara Menggunakan

### Use Case 1: Eksplorasi Komunitas

1. Buka aplikasi
2. Cek metric cards di atas untuk overview kualitas komunitas
3. **Tab 1**: lihat distribusi visual semua komunitas
4. **Tab 2**: baca daftar gen anggota tiap komunitas
5. **Tab 3**: identifikasi gen pleiotropic

### Use Case 2: Bandingkan Hasil dengan K Berbeda

1. Geser slider K di sidebar (atau ketik nilai)
2. Semua tab otomatis update
3. Bandingkan jumlah komunitas, gen overlapping, dan distribusi
4. Cek Tab 4 untuk lihat trend metrik vs K

### Use Case 3: Eksplorasi Gen Spesifik (Recommended)

1. Buka **Tab 5: Gene Explorer**
2. Cari gen yang Anda kenal (mis. `TP53`, `BRCA1`, `PIK3CA`)
3. Lihat komunitas yang dimasuki gen tersebut
4. Cek gen-gen yang mirip secara embedding
5. Baca insight biologis dari KEGG, GO, dan Reactome
6. Validasi: apakah pathway yang muncul sesuai dengan literature?

---

## Metodologi Pipeline

### Tahap 1: Konstruksi Heterogeneous Information Network (HIN)

Dua tipe node:
- **Protein** (gen kanker): 290 node, Entrez ID
- **Drug** (senyawa): ~150 node, PubChem CID

Dua tipe edge:
- **Drug-Target**: hubungan obat menarget protein
- **PPI**: interaksi antar protein

### Tahap 2: Metapath Random Walks

Empat strategi metapath untuk capture struktur heterogen:
- **PP**: Protein-Protein (struktur fungsional langsung)
- **PDP**: Protein-Drug-Protein (protein yang berbagi obat)
- **DPD**: Drug-Protein-Drug (obat dengan target yang sama)
- **DPPD**: Drug-Protein-Protein-Drug (jalur panjang)

### Tahap 3: Metapath2Vec Embedding

- Algoritma: Word2Vec Skip-gram
- Dimensi embedding: 32 (dipilih via ablation study)
- Window size: 5
- Epochs: 30
- Output: vektor 32-dim per node

### Tahap 4: Transformasi ke Homogeneous Network

- Hitung cosine similarity antar embedding protein
- **Top-K Cosine Similarity (K=10)**: tiap protein terhubung ke 10 protein paling mirip
- Hasil: Weighted Homogeneous Network (protein-only, weighted)

### Tahap 5: Overlapping Community Detection dengan BigCLAM

- Algoritma: BigCLAM (Yang & Leskovec 2013) — implementasi pure-Python faithful
- Inisialisasi: Conductance-based seed selection
- Optimisasi: Projected gradient ascent per node
- Output: Affiliation matrix F → komunitas tumpang tindih

### Tahap 6: Validasi

**Validasi Statistik:**
- Hyperparameter ablation study (4 parameter)
- Multi-criteria ranking (Modularity + Density + Conductance)

**Validasi Biologis (Multi-Database):**
- KEGG Pathways (reference)
- GO Biological Process (independen)
- Reactome (independen)
- Konsistensi enrichment di database independen membuktikan hasil **bukan circular reasoning**.

---

## Data Sources

| Data | Sumber | Konten |
|------|--------|--------|
| KEGG Cancer Genes | KEGG Pathways in Cancer | 325 gen kanker (Gene Symbols) |
| Drug-Target | Decagon (SNAP) | Drug-Target interactions |
| Protein-Protein Interactions | Decagon (SNAP) | PPI network |
| Gene Mapping | MyGene.info | Symbol → Entrez ID |

---

## Technical Stack

| Komponen | Library |
|----------|---------|
| Embedding | gensim (Word2Vec) |
| Graph Processing | networkx |
| Community Detection | Custom Python (BigCLAM faithful) |
| Evaluation | cdlib |
| Similarity | scikit-learn (cosine_similarity) |
| Enrichment | gseapy (Enrichr API) |
| UI | Streamlit |
| Visualization | matplotlib, seaborn |
| Data | pandas, numpy |

---

## Reproducibility

Semua hasil dapat direproduksi dengan:
- Fixed random seed: `SEED = 42`
- `np.random.RandomState(seed)` lokal di BigCLAM
- `random.Random(seed)` lokal di metapath walks
- Word2Vec: `workers=1` + `seed=42`
- File hasil disimpan dalam pickle untuk konsistensi UI

---

## References

1. **Yang, J., & Leskovec, J. (2013).** Overlapping community detection at scale: A nonnegative matrix factorization approach. *Proceedings of WSDM 2013*.

2. **Dong, Y., Chawla, N. V., & Swami, A. (2017).** Metapath2vec: Scalable representation learning for heterogeneous networks. *Proceedings of KDD 2017*, 135-144.

3. **Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013).** Efficient estimation of word representations in vector space. *arXiv:1301.3781*.

4. **Nicosia, V., Mangioni, G., Carchiolo, V., & Malgeri, M. (2009).** Extending the definition of modularity to directed graphs with overlapping communities. *Journal of Statistical Mechanics: Theory and Experiment*, 2009(03), P03024.

5. **Lancichinetti, A., & Fortunato, S. (2009).** Community detection algorithms: A comparative analysis. *Physical Review E, 80*(5), 056117.

6. **Demšar, J. (2006).** Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research, 7*, 1-30.

7. **Chen, E. Y., et al. (2013).** Enrichr: interactive and collaborative HTML5 gene list enrichment analysis tool. *BMC Bioinformatics, 14*, 128.

---

## Project Information
- **Domain:** Bioinformatika, Network Science
- **Use Case:** Identifikasi komunitas gen kanker dan gen pleiotropic
- **Output:** Web app interaktif untuk eksplorasi hasil komunitas
