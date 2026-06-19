
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull
from matplotlib.lines import Line2D

st.set_page_config(page_title='OCD - Gen Kanker', layout='wide')
st.title('Overlapping Community Detection — Jaringan Gen Kanker')
st.caption('Metapath2Vec + BigCLAM')

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
    cmap = plt.get_cmap('tab20', max(n_coms, 1))
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


def render_overlay_communities(G, communities, membership, k_val):
    """Overlay semua komunitas dalam 1 plot dengan Convex Hull."""
    fig, ax = plt.subplots(figsize=(14, 11))
    n_coms = len(communities)
    if n_coms == 0:
        return fig
    cmap = plt.get_cmap('tab20', max(n_coms, 1))
    colors = [cmap(i) for i in range(n_coms)]
    member_count = {n: len(c) for n, c in membership.items()}

    def get_node_size(node):
        c = member_count.get(node, 1)
        return 220 + (c * 280) if c > 1 else 90

    # Background edges
    weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=[(w * 2) for w in weights],
                           alpha=0.15, edge_color='royalblue')

    # Gambar tiap komunitas dengan Convex Hull
    for i, members in enumerate(communities):
        members_in_pos = [n for n in members if n in pos]
        if not members_in_pos:
            continue
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=members_in_pos,
                               node_color=[colors[i]],
                               node_size=[get_node_size(m) for m in members_in_pos],
                               alpha=0.7, edgecolors='white', linewidths=0.5)
        # Label entrez ID kecil
        nx.draw_networkx_labels(G, pos, ax=ax,
                                labels={m: m for m in members_in_pos},
                                font_size=6, font_color='black')
        # Convex Hull untuk highlight area komunitas
        if len(members_in_pos) >= 3:
            points = np.array([pos[n] for n in members_in_pos])
            try:
                hull = ConvexHull(points)
                ax.fill(points[hull.vertices, 0], points[hull.vertices, 1],
                        color=colors[i], alpha=0.06,
                        edgecolor=colors[i], linestyle='--', linewidth=1.0)
            except Exception:
                pass

    # Highlight overlapping nodes dengan border hitam
    overlap_nodes = [n for n, c in membership.items() if len(c) > 1 and n in pos]
    if overlap_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=overlap_nodes,
                               node_color='tomato',
                               node_size=[get_node_size(n) for n in overlap_nodes],
                               edgecolors='black', linewidths=1.5, alpha=0.85)

    # Legend
    legend_elems = [
        Line2D([0], [0], marker='', color='w', label='Status Node:'),
        Line2D([0], [0], marker='o', color='w', label='Biasa (1 komunitas)',
               markerfacecolor='lightgray', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Overlapping (>=2 komunitas)',
               markerfacecolor='tomato', markersize=12, markeredgecolor='black'),
        Line2D([0], [0], marker='', color='w', label=''),
        Line2D([0], [0], marker='', color='w', label=f'Komunitas (K={k_val}):'),
    ]
    for i in range(n_coms):
        legend_elems.append(Line2D([0], [0], marker='s', color='w',
                                   label=f'Komunitas {i + 1}',
                                   markerfacecolor=colors[i], markersize=10, alpha=0.7))
    ax.legend(handles=legend_elems, loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=9, frameon=True, borderpad=1.0, labelspacing=1.0)

    ax.set_title(
        f'Overlay Komunitas (K={k_val}) — Convex Hull per Komunitas\n'
        f'Nodes: {G.number_of_nodes()} | Overlapping: {len(overlap_nodes)}',
        fontsize=14, fontweight='bold', pad=14
    )
    ax.axis('off')
    plt.tight_layout()
    return fig


# ============================================================
# TABS UTAMA
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    'Visualisasi Komunitas',
    'Daftar Gen Per Komunitas',
    'Gen Overlapping',
    'Kurva Optimasi K',
    'Gene Explorer',
])

# --- TAB 1: Visualisasi Komunitas (toggle: Overlay vs Grid) ---
with tab1:
    st.subheader(f'Visualisasi Komunitas (K={selected_k})')
    viz_mode = st.radio(
        'Mode visualisasi:',
        ['Overlay (semua komunitas dalam 1 plot)', 'Grid (1 panel per komunitas)'],
        horizontal=True, key='viz_mode_tab1',
        help='Overlay = lihat distribusi & overlap antar komunitas. Grid = lihat tiap komunitas terisolasi.'
    )

    if viz_mode.startswith('Overlay'):
        st.caption('Tiap komunitas diberi warna berbeda dengan Convex Hull. Node tomato = overlapping (anggota ≥2 komunitas).')
        with st.spinner('Rendering overlay...'):
            fig = render_overlay_communities(G_weighted, community_list, membership_map, selected_k)
        st.pyplot(fig, use_container_width=True)
    else:
        st.caption('Setiap panel = 1 komunitas. Node besar = overlapping.')
        with st.spinner('Rendering grid...'):
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
                    # === 6. RINGKASAN INSIGHT (HANYA FAKTA DARI DATA) ===
                    st.markdown('---')
                    st.markdown('#### Ringkasan Insight')

                    n_comms = len(comm_idx)

                    # Kumpulkan top 3 pathway lintas database (faktual dari Enrichr)
                    top_pathways_facts = []
                    for c_idx in comm_idx:
                        com_id = c_idx + 1
                        if com_id not in enrichment_results:
                            continue
                        df_enr = enrichment_results[com_id]
                        for db_name, db_label in [
                            ('KEGG_2021_Human', 'KEGG'),
                            ('GO_Biological_Process_2023', 'GO BP'),
                            ('Reactome_2022', 'Reactome'),
                        ]:
                            df_with_gene = df_enr[
                                (df_enr['Gene_set'] == db_name) &
                                (df_enr['Genes'].astype(str).str.contains(selected_gene, na=False, case=False))
                            ]
                            top = df_with_gene.nsmallest(1, 'Adjusted P-value')
                            if not top.empty:
                                row = top.iloc[0]
                                top_pathways_facts.append({
                                    'community': com_id,
                                    'database' : db_label,
                                    'term'     : str(row['Term'])[:60],
                                    'p_adj'    : float(row['Adjusted P-value']),
                                })

                    # Co-members sekomunitas (faktual dari membership_map)
                    same_com_partners = []
                    if 'top_data' in dir():
                        same_com_partners = [d['Gene Symbol'] for d in top_data
                                             if d.get('Komunitas Sama', '').startswith('✓')]

                    # Susun fakta-fakta
                    fact_lines = []

                    # Fakta 1: Status komunitas
                    if n_comms > 1:
                        fact_lines.append(
                            f'- **Status komunitas:** Anggota **{n_comms} komunitas** '
                            f'(K{", K".join(str(c + 1) for c in comm_idx)}) — '
                            f'tergolong **pleiotropic** dalam graf ini.'
                        )
                    else:
                        fact_lines.append(
                            f'- **Status komunitas:** Anggota tunggal **Komunitas {comm_idx[0] + 1}** — '
                            f'tidak pleiotropic dalam graf ini.'
                        )

                    # Fakta 2: Top pathway per komunitas (best p-value)
                    if top_pathways_facts:
                        unique_terms = {}
                        for pf in top_pathways_facts:
                            key = pf['term']
                            if key not in unique_terms or pf['p_adj'] < unique_terms[key]['p_adj']:
                                unique_terms[key] = pf
                        top3 = sorted(unique_terms.values(), key=lambda x: x['p_adj'])[:3]
                        pathway_str = '; '.join([f'*{p["term"]}* ({p["database"]}, p={p["p_adj"]:.2e})'
                                                 for p in top3])
                        fact_lines.append(
                            f'- **Pathway signifikan teratas yang mengandung {selected_gene}:** {pathway_str}.'
                        )

                    # Fakta 3: Top similar genes (sekomunitas)
                    if same_com_partners:
                        partner_list = ', '.join(same_com_partners[:5])
                        fact_lines.append(
                            f'- **Gen sekomunitas dengan embedding similarity tertinggi:** {partner_list}.'
                        )

                    # Fakta 4: Catatan interpretasi (tanpa klaim eksternal)
                    if n_comms > 1:
                        fact_lines.append(
                            f'- **Catatan:** Pleiotropic gene seperti {selected_gene} dapat berperan '
                            f'menghubungkan beberapa komunitas/pathway yang berbeda dalam jaringan ini.'
                        )

                    # Render
                    summary_md = '\n'.join(fact_lines)
                    st.markdown(summary_md)
