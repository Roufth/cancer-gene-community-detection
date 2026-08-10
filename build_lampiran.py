import pickle, pandas as pd

BASE = r'c:/Users/Lenovo/Documents/Skripsi'
PKL  = BASE + r'/HIN Apps/ocd_model_artifacts.pkl'
CSV  = BASE + r'/HIN Apps/biological_validation_multidb.csv'
OUT  = BASE + r'/Lampiran_Data_K12.xlsx'

d = pickle.load(open(PKL, 'rb'))
community_list = d['community_list']
gene_mapping   = d['gene_mapping']       # entrez(str) -> symbol
membership_map = d['membership_map']
prot_compounds = d['protein_compounds']
K = d['best_k']

# ---- Lampiran 1: Gene Symbol -> Entrez ID ----
df_gene = pd.DataFrame(
    [{'Gene Symbol': sym, 'Entrez ID': int(eid)}
     for eid, sym in gene_mapping.items() if str(eid).isdigit()]
).sort_values('Gene Symbol').reset_index(drop=True)

# ---- Lampiran 2: Daftar gen anggota tiap komunitas ----
rows = []
for i, members in enumerate(community_list):
    for n in members:
        n = str(n)
        rows.append({
            'Komunitas': f'K{i+1}',
            'Entrez ID': n,
            'Gene Symbol': gene_mapping.get(n, n),
            'Status': 'Overlapping' if len(membership_map.get(n, [])) > 1 else 'Tunggal',
        })
df_members = pd.DataFrame(rows)

# ---- Lampiran 3: Hasil enrichment lengkap (signifikan, p-adj < 0.05) ----
df_enr = pd.read_csv(CSV)
df_sig = df_enr[df_enr['Adjusted P-value'] < 0.05].copy()
df_sig['Komunitas'] = df_sig['Komunitas'].astype(str).str.replace('Komunitas_', 'K', regex=False)
df_sig = df_sig[['Komunitas', 'Gene_set', 'Term', 'Overlap', 'Adjusted P-value', 'Genes']]
df_sig = df_sig.rename(columns={'Gene_set': 'Database', 'Adjusted P-value': 'p-adj'})
df_sig = df_sig.sort_values(['Komunitas', 'Database', 'p-adj'])

# ---- Lampiran 4: PubChem CID -> Nama (+ kategori manual) ----
INFO = {
 271:('Calcium Cation','Ion/kofaktor'), 564:('6-Aminohexanoic Acid','Non-kanker'),
 1046:('Pyrazinamide','Non-kanker'), 1983:('Acetaminophen','Non-kanker'),
 2244:('Aspirin','Non-kanker'), 2375:('Bicalutamide','Obat kanker'),
 2662:('Celecoxib','Non-kanker'), 2708:('Chlorambucil','Obat kanker'),
 2973:('Deferoxamine','Non-kanker'), 3032:('Arthrotec','Non-kanker'),
 3059:('Diflunisal','Non-kanker'), 3117:('Disulfiram','Non-kanker'),
 3278:('Ethacrynic Acid','Non-kanker'), 3308:('Etodolac','Non-kanker'),
 3339:('Fenofibrate','Non-kanker'), 3385:('5-Fluorouracil','Obat kanker'),
 3394:('Flurbiprofen','Non-kanker'), 3397:('Flutamide','Obat kanker'),
 3672:('Ibuprofen','Non-kanker'), 3715:('Indomethacin','Non-kanker'),
 3825:('Ketoprofen','Non-kanker'), 3826:('Ketorolac','Non-kanker'),
 4044:('Mefenamic Acid','Non-kanker'), 4212:('Mitoxantrone','Obat kanker'),
 4829:('Pioglitazone','Non-kanker'), 5035:('Raloxifene','Obat kanker (SERM/pencegahan)'),
 5090:('Rofecoxib','Non-kanker'), 5291:('Imatinib','Obat kanker'),
 5381:('Tazarotene','Non-kanker'), 5717:('Zafirlukast','Non-kanker'),
 60164:('Adapalene','Non-kanker'), 65999:('Telmisartan','Non-kanker'),
 68740:('Zoledronic Acid','Non-kanker (suportif kanker tulang)'),
 82146:('Bexarotene','Obat kanker'), 104741:('Fulvestrant','Obat kanker'),
 119607:('Valdecoxib','Non-kanker'), 123631:('Gefitinib','Obat kanker'),
 176870:('Erlotinib','Obat kanker'), 216239:('Sorafenib','Obat kanker (ginjal/hati)'),
 3062316:('Dasatinib','Obat kanker'), 5329102:('Sunitinib','Obat kanker (ginjal)'),
}
def cid_num(c):
    s = str(c); return int(s[3:]) if s.startswith('CID') else int(s)
used = set()
for v in prot_compounds.values():
    for c in v:
        used.add(cid_num(c))
df_cid = pd.DataFrame(
    [{'PubChem CID': n, 'Nama Senyawa (PubChem Title)': INFO.get(n, (str(n), ''))[0],
      'Kategori (klasifikasi manual)': INFO.get(n, ('', 'perlu verifikasi'))[1]}
     for n in sorted(used)]
)

with pd.ExcelWriter(OUT) as w:
    df_gene.to_excel(w,    sheet_name='L1 Gene Symbol-Entrez', index=False)
    df_members.to_excel(w, sheet_name='L2 Gen per Komunitas',  index=False)
    df_sig.to_excel(w,     sheet_name='L3 Enrichment Lengkap',  index=False)
    df_cid.to_excel(w,     sheet_name='L4 PubChem CID-Nama',    index=False)

print('Tersimpan:', OUT)
print(f'L1 Gene-Entrez     : {len(df_gene)} baris')
print(f'L2 Gen/komunitas   : {len(df_members)} baris')
print(f'L3 Enrichment sig. : {len(df_sig)} baris (p-adj<0.05)')
print(f'L4 PubChem CID     : {len(df_cid)} baris')
