
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='OCD - Gen Kanker', layout='wide')
st.title('Overlapping Community Detection — Jaringan Gen Kanker')
st.caption('Metapath2Vec + BigCLAM | Skripsi Bioinformatika')

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_enrichment(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

artifacts          = load_artifacts('ocd_model_artifacts.pkl')
enrichment_results = load_enrichment('enrichment_results.pkl')
all_k     = sorted(artifacts['all_k_artifacts'].keys())
k_min, k_max = min(all_k), max(all_k)

# ============================================================
# SYNCED SLIDER + NUMBER INPUT
# ============================================================
if 'k_value' not in st.session_state:
    st.session_state.k_value = int(artifacts['best_k'])

def _sync_from_slider(): st.session_state.k_value = st.session_state.k_slider
def _sync_from_input():  st.session_state.k_value = st.session_state.k_input

st.sidebar.header('Konfigurasi')
st.sidebar.info(f'K Optimal: **K = {artifacts["best_k"]}**')
st.sidebar.slider('Geser K:', min_value=k_min, max_value=k_max,
                  value=st.session_state.k_value, key='k_slider',
                  on_change=_sync_from_slider)
st.sidebar.number_input('Atau ketik K:', min_value=k_min, max_value=k_max,
                        value=st.session_state.k_value, step=1, key='k_input',
                        on_change=_sync_from_input)
selected_k = st.session_state.k_value
st.sidebar.success(f'**K aktif: {selected_k}**')

# ============================================================
# DATA AMBIL UNTUK K TERPILIH
# ============================================================
data            = artifacts['all_k_artifacts'][selected_k]
gene_mapping    = artifacts['gene_mapping']
G_weighted      = artifacts['G_weighted']
pos             = artifacts['precomputed_pos']
embeddings      = artifacts['embeddings']
node_ids        = artifacts['node_ids']
community_list  = data['community_list']
membership_map  = data['membership_map']

# Reverse mapping: symbol → entrez
symbol_to_entrez = {sym: eid for eid, sym in gene_mapping.items() if sym and sym != 'Unknown'}

# ============================================================
# METRIC CARDS
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric('Overlap Modularity', f'{data["overlap_modularity"]:.4f}')
c2.metric('Internal Density',   f'{data["internal_density"]:.4f}')
c3.metric('Conductance',        f'{data["conductance"]:.4f}')
c4.metric('Node Overlapping',   data['num_overlap'])

# ============================================================
# HELPER: Render per-community grid (sama seperti sebelumnya)
# ============================================================
def render_per_community(G, communities, membership, k_val, cols=4):
    n_coms = len(communities)
    if n_coms == 0: return None
    rows = (n_coms + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).flatten()
    member_count = {n: len(c) for n, c in membership.items()}
    cmap = cm.get_cmap('tab20', max(n_coms, 1))
    for i, members in enumerate(communities):
        ax = axes[i]
        members_in_pos = [n for n in members if n in pos]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.05, edge_color='gray', width=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=15, node_color='lightgray', alpha=0.3)
        subgraph = G.subgraph(members_in_pos)
        weights = [subgraph[u][v].get('weight', 1) for u, v in subgraph.edges()]
        if weights:
            nx.draw_networkx_edges(subgraph, pos, ax=ax, width=[(w * 1.5) for w in weights],
                                   alpha=0.4, edge_color='royalblue')
        sizes = [90 if member_count.get(m, 1) > 1 else 35 for m in members_in_pos]
        nx.draw_networkx_nodes(G, pos, nodelist=members_in_pos, ax=ax,
                               node_size=sizes, node_color=[cmap(i)], alpha=0.9)
        n_ov = sum(1 for m in members_in_pos if member_count.get(m, 1) > 1)
        ax.set_title(f'Komunitas {i+1}\n{len(members_in_pos)} gen ({n_ov} overlap)',
                     fontsize=10, fontweight='bold')
        ax.axis('off')
    for j in range(n_coms, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f'Distribusi Komunitas (K={k_val})', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig

# ============================================================
# TABS UTAMA — sekarang 5 TAB (tambah Gene Explorer)
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    'Visualisasi Per Komunitas',
    'Daftar Gen Per Komunitas',
    'Gen Overlapping',
    'Kurva Optimasi K',
    'GENE EXPLORER',
])

# --- TAB 1: Per-Community Visualization ---
with tab1:
    st.subheader(f'Grid Visualisasi (K={selected_k})')
    st.caption('Setiap panel = 1 komunitas. Node besar = overlapping.')
    with st.spinner('Rendering...'):
        fig = render_per_community(G_weighted, community_list, membership_map, selected_k)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)

# --- TAB 2: Daftar Gen Per Komunitas ---
with tab2:
    st.subheader(f'Anggota Gen Tiap Komunitas ({len(community_list)} komunitas)')
    view_mode = st.radio('Mode tampilan:', ['Card (rinci)', 'Tabel (ringkas)'],
                         horizontal=True, key='view_mode_tab2')
    if view_mode == 'Tabel (ringkas)':
        table_rows = []
        for i, members in enumerate(community_list):
            symbols = sorted([gene_mapping.get(str(n), n) for n in members])
            n_overlap = sum(1 for m in members if len(membership_map.get(m, [])) > 1)
            table_rows.append({
                'Komunitas': f'K{i+1}', 'Jumlah Gen': len(symbols),
                'Overlapping': n_overlap, 'Daftar Gen': ', '.join(symbols),
            })
        df_communities = pd.DataFrame(table_rows)
        st.dataframe(df_communities, use_container_width=True, hide_index=True,
                     column_config={'Daftar Gen': st.column_config.TextColumn(width='large')})
        st.download_button('Download CSV', df_communities.to_csv(index=False),
                           file_name=f'komunitas_K{selected_k}.csv', mime='text/csv')
    else:
        for i, members in enumerate(community_list):
            symbols = sorted([gene_mapping.get(str(n), n) for n in members])
            n_overlap = sum(1 for m in members if len(membership_map.get(m, [])) > 1)
            with st.expander(f'Komunitas {i+1} — {len(symbols)} gen ({n_overlap} overlap)',
                             expanded=True):
                st.write(', '.join(symbols))

# --- TAB 3: Overlapping ---
with tab3:
    st.subheader('Gen Pleiotropic (Anggota >1 Komunitas)')
    overlap_rows = []
    for node, comm_idx in membership_map.items():
        if len(comm_idx) > 1:
            overlap_rows.append({
                'Entrez ID': node,
                'Gene Symbol': gene_mapping.get(str(node), 'Unknown'),
                'Komunitas': ', '.join([f'K{c+1}' for c in comm_idx]),
                'Jumlah': len(comm_idx),
            })
    if overlap_rows:
        df_ov = pd.DataFrame(overlap_rows).sort_values('Jumlah', ascending=False)
        st.dataframe(df_ov, use_container_width=True, hide_index=True)
    else:
        st.info('Tidak ada gen overlapping pada K ini.')

# --- TAB 4: Kurva K ---
with tab4:
    st.subheader('Tren Metrik vs K')
    df_k = artifacts['df_k_results']
    metric_choice = st.selectbox('Pilih metrik:',
        ['Overlap_Modularity', 'Internal_Density', 'Conductance'])
    st.line_chart(df_k.set_index('K')[metric_choice])

# --- TAB 5: GENE EXPLORER (FITUR BARU) ---
with tab5:
    st.subheader('🔍 Gene Explorer — Eksplorasi Relasi Gen')
    st.caption('Pilih gen untuk melihat: komunitas, gen serupa, dan insight biologis.')

    # === 1. Pilih gen ===
    valid_symbols = sorted({gene_mapping.get(str(n), '') for n in node_ids
                            if gene_mapping.get(str(n), '') and gene_mapping.get(str(n), '') != 'Unknown'})
    if not valid_symbols:
        st.warning('Tidak ada gen valid di artifacts.')
    else:
        # Opsi search: typeahead via selectbox (Streamlit auto-filter saat ketik)
        selected_gene = st.selectbox(
            'Pilih gen (ketik untuk search):',
            valid_symbols,
            index=0,
            help='Ketik nama gen untuk filter otomatis'
        )

        # Find entrez_id
        target_entrez = None
        for eid, sym in gene_mapping.items():
            if sym == selected_gene:
                # Pastikan ada di node_ids saat ini
                if str(eid) in [str(n) for n in node_ids]:
                    target_entrez = str(eid)
                    break

        if target_entrez is None:
            st.warning(f'Gen {selected_gene} tidak ditemukan di komunitas K={selected_k}.')
        else:
            # === 2. Info dasar ===
            st.markdown(f'## {selected_gene}')
            st.caption(f'Entrez ID: {target_entrez}')

            comm_idx = membership_map.get(target_entrez, [])
            colA, colB = st.columns(2)
            colA.metric('Jumlah komunitas', len(comm_idx),
                        delta='Pleiotropic' if len(comm_idx) > 1 else 'Single')
            colB.metric('Status', 'Overlapping' if len(comm_idx) > 1 else 'Non-overlapping')

            # === 3. Co-members per komunitas ===
            st.markdown('### Gen Sekomunitas (Co-members)')
            for c_idx in comm_idx:
                members = community_list[c_idx]
                co_symbols = sorted([gene_mapping.get(str(n), n) for n in members
                                     if str(n) != target_entrez])
                with st.expander(f'Komunitas {c_idx + 1} — {len(co_symbols)} gen lain', expanded=True):
                    st.write(', '.join(co_symbols))

            # === 4. Top similar genes (cosine similarity dari embedding) ===
            st.markdown('### Top 10 Gen Paling Mirip (by Metapath2Vec Embedding)')
            node_ids_str = [str(n) for n in node_ids]
            if target_entrez in node_ids_str:
                idx = node_ids_str.index(target_entrez)
                sims = cosine_similarity([embeddings[idx]], embeddings)[0]
                top_idx = np.argsort(sims)[::-1][1:11]   # top 10 exclude self
                top_data = []
                for ti in top_idx:
                    similar_eid = str(node_ids[ti])
                    similar_sym = gene_mapping.get(similar_eid, similar_eid)
                    similar_comms = membership_map.get(similar_eid, [])
                    same_com = '✓ Ya' if any(c in comm_idx for c in similar_comms) else '✗ Tidak'
                    top_data.append({
                        'Gene Symbol': similar_sym,
                        'Entrez ID': similar_eid,
                        'Cosine Similarity': f'{sims[ti]:.4f}',
                        'Komunitas Sama': same_com,
                        'Komunitas': ', '.join([f'K{c+1}' for c in similar_comms]) or '-',
                    })
                st.dataframe(pd.DataFrame(top_data), use_container_width=True, hide_index=True)

            # === 5. INSIGHT BIOLOGIS (jika enrichment tersedia) ===
            st.markdown('### 💡 Insight Biologis')
            if enrichment_results is None:
                st.info('Enrichment results belum di-generate. Jalankan Cell 25 di notebook.')
            else:
                insight_found = False
                for c_idx in comm_idx:
                    com_id = c_idx + 1
                    if com_id not in enrichment_results:
                        continue
                    df_enr = enrichment_results[com_id]
                    st.markdown(f'**Komunitas {com_id} — Top Pathway:**')
                    for db_name, db_label in [
                        ('KEGG_2021_Human', 'KEGG'),
                        ('GO_Biological_Process_2023', 'GO BP'),
                        ('Reactome_2022', 'Reactome'),
                    ]:
                        df_db = df_enr[df_enr['Gene_set'] == db_name].copy()
                        # Filter pathway yang MENGANDUNG gen ini
                        df_with_gene = df_db[df_db['Genes'].astype(str).str.contains(
                            selected_gene, na=False, case=False)]
                        df_with_gene = df_with_gene.nsmallest(3, 'Adjusted P-value')
                        if not df_with_gene.empty:
                            insight_found = True
                            st.markdown(f'  - **{db_label}:**')
                            for _, row in df_with_gene.iterrows():
                                p_val = row['Adjusted P-value']
                                term  = row['Term']
                                st.markdown(f'    • {term} (p_adj = {p_val:.2e})')

                if not insight_found:
                    st.info(f'{selected_gene} tidak muncul di pathway signifikan untuk komunitasnya.')
                else:
                    # === 6. AUTO-INSIGHT SUMMARY ===
                    st.markdown('---')
                    st.markdown('#### 📝 Ringkasan Insight')
                    n_comms     = len(comm_idx)
                    n_neighbors = len(top_data) if 'top_data' in dir() else 10
                    summary = (
                        f'Gen **{selected_gene}** teridentifikasi sebagai anggota '
                        f'**{n_comms} komunitas** {"(pleiotropic — multifungsi)" if n_comms > 1 else "(single)"}. '
                        f'Gen ini berkorespondensi dengan pathway-pathway di atas yang '
                        f'menggambarkan peran fungsional spesifiknya. '
                        f'Gen-gen yang paling mirip (by embedding) menunjukkan kandidat '
                        f'gen yang berperan dalam jalur biologis serupa.'
                    )
                    st.success(summary)
