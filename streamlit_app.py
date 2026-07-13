
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


def _hetero_size(nd, mc):
    c = mc.get(str(nd), mc.get(nd, 1))
    return 200 + c * 300 if c > 1 else 120


def _cid_label(c):
    """Format id senyawa 'CID000005291' -> PubChem CID '5291'."""
    s = str(c)
    if s.startswith('CID'):
        return s[3:].lstrip('0') or '0'
    return s


def render_hetero_single(comm_idx, communities, membership, prot_compounds):
    """1 komunitas + overlay senyawa. Style: tab20 (warna komunitas) + Convex Hull."""
    proteins = [str(p) for p in communities[comm_idx]]
    comp_set = set()
    for p in proteins:
        comp_set.update(prot_compounds.get(p, []))
    total_comp = len(comp_set)
    multi = {c for c in comp_set
             if sum(1 for p in proteins if c in prot_compounds.get(p, [])) >= 2}

    H = nx.Graph()
    H.add_nodes_from(proteins)
    H.add_nodes_from(comp_set)
    for i in range(len(proteins)):
        for j in range(i + 1, len(proteins)):
            a, b = proteins[i], proteins[j]
            if G_weighted.has_edge(a, b):
                H.add_edge(a, b, etype='pp')
    for c in comp_set:
        for p in proteins:
            if c in prot_compounds.get(p, []):
                H.add_edge(c, p, etype='cpi')

    hpos = nx.spring_layout(H, seed=42, k=0.6, iterations=120)
    K = len(communities)
    comm_color = plt.get_cmap('tab20', K)(comm_idx)
    mc = {n: len(c) for n, c in membership.items()}

    fig, ax = plt.subplots(figsize=(12, 9))
    pp  = [(u, v) for u, v, d in H.edges(data=True) if d['etype'] == 'pp']
    cpi = [(u, v) for u, v, d in H.edges(data=True) if d['etype'] == 'cpi']
    nx.draw_networkx_edges(H, hpos, edgelist=pp, edge_color='royalblue', width=1.3, alpha=0.5, ax=ax)
    nx.draw_networkx_edges(H, hpos, edgelist=cpi, edge_color='#c084fc',
                           style='dashed', width=0.9, alpha=0.6, ax=ax)

    p_over = [p for p in proteins if len(membership.get(str(p), membership.get(p, []))) > 1]
    nx.draw_networkx_nodes(H, hpos, nodelist=proteins, node_color=[comm_color],
                           node_size=[_hetero_size(p, mc) for p in proteins], alpha=0.75,
                           edgecolors='white', linewidths=0.5, ax=ax)
    if len(proteins) >= 3:
        pts = np.array([hpos[n] for n in proteins])
        try:
            hull = ConvexHull(pts)
            ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color=comm_color,
                    alpha=0.05, edgecolor=comm_color, linestyle='--')
        except Exception:
            pass
    if p_over:
        nx.draw_networkx_nodes(H, hpos, nodelist=p_over, node_color='tomato',
                               node_size=[_hetero_size(p, mc) for p in p_over],
                               edgecolors='black', linewidths=1.5, ax=ax)
    d_multi = [c for c in comp_set if c in multi]
    d_plain = [c for c in comp_set if c not in multi]
    nx.draw_networkx_nodes(H, hpos, nodelist=d_plain, node_color='#374151', node_shape='s',
                           node_size=130, alpha=0.85, ax=ax)
    nx.draw_networkx_nodes(H, hpos, nodelist=d_multi, node_color='#f59e0b', node_shape='s',
                           node_size=210, edgecolors='black', linewidths=0.6, ax=ax)
    nx.draw_networkx_labels(H, hpos, labels={p: gene_mapping.get(str(p), p) for p in proteins},
                            font_size=8, font_color='black', ax=ax)
    nx.draw_networkx_labels(H, hpos, labels={c: _cid_label(c) for c in comp_set},
                            font_size=6, font_color='#111111', ax=ax,
                            bbox=dict(boxstyle='round,pad=0.12', facecolor='#fde047',
                                      edgecolor='none', alpha=0.9))

    leg = [
        Line2D([0], [0], marker='o', color='w', label=f'Protein (Komunitas {comm_idx + 1})',
               markerfacecolor=comm_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Protein overlapping',
               markerfacecolor='tomato', markersize=11, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='Senyawa',
               markerfacecolor='#374151', markersize=9),
        Line2D([0], [0], marker='s', color='w', label='Senyawa multi-target',
               markerfacecolor='#f59e0b', markersize=11, markeredgecolor='black'),
    ]
    ax.legend(handles=leg, loc='upper right', fontsize=9, frameon=True)
    ax.set_title(f'Komunitas {comm_idx + 1} + Overlay Senyawa — '
                 f'{len(proteins)} protein + {total_comp} senyawa',
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig, total_comp, [gene_mapping.get(str(p), p) for p in p_over], len(multi)


def render_hetero_combined(communities, membership, prot_compounds):
    """Gabungan SEMUA komunitas + overlay senyawa. Style CELL 17 (tab20 + Convex Hull),
    layout protein konsisten dari precomputed_pos; posisi senyawa = centroid target."""
    K = len(communities)
    prot_set = set(str(n) for n in G_weighted.nodes())

    comp_targets = {}
    for p, cs in prot_compounds.items():
        for c in cs:
            comp_targets.setdefault(c, []).append(str(p))
    comp_pos = {}
    for c, tgts in comp_targets.items():
        pts = [pos[p] for p in tgts if p in pos]
        if pts:
            comp_pos[c] = (float(np.mean([q[0] for q in pts])),
                           float(np.mean([q[1] for q in pts])))
    comp_all = set(comp_pos.keys())
    pos_all = {**pos, **comp_pos}
    multi = {c for c in comp_all if sum(1 for p in comp_targets[c] if p in prot_set) >= 2}

    cmap = plt.get_cmap('tab20', K)
    colors = [cmap(i) for i in range(K)]
    mc = {n: len(c) for n, c in membership.items()}

    fig, ax = plt.subplots(figsize=(16, 13))
    w = [G_weighted[u][v].get('weight', 1) for u, v in G_weighted.edges()]
    nx.draw_networkx_edges(G_weighted, pos, ax=ax, width=[(x * 2) for x in w],
                           alpha=0.10, edge_color='royalblue')
    cpi_edges = [(c, p) for c in comp_all for p in comp_targets[c] if p in prot_set]
    Hcpi = nx.Graph()
    Hcpi.add_edges_from(cpi_edges)
    nx.draw_networkx_edges(Hcpi, pos_all, ax=ax, edgelist=cpi_edges,
                           edge_color='#c084fc', style='dashed', width=0.4, alpha=0.18)

    for i, members in enumerate(communities):
        mp = [str(n) for n in members if str(n) in pos]
        if not mp:
            continue
        nx.draw_networkx_nodes(G_weighted, pos, ax=ax, nodelist=mp, node_color=[colors[i]],
                               node_size=[_hetero_size(m, mc) for m in mp], alpha=0.7,
                               edgecolors='white', linewidths=0.5)
        if len(mp) >= 3:
            pts = np.array([pos[n] for n in mp])
            try:
                hull = ConvexHull(pts)
                ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color=colors[i],
                        alpha=0.05, edgecolor=colors[i], linestyle='--')
            except Exception:
                pass
    # Label Entrez ID untuk semua node protein
    nx.draw_networkx_labels(G_weighted, pos, ax=ax,
                            labels={str(n): str(n) for n in prot_set if str(n) in pos},
                            font_size=6, font_color='black')

    ov = [str(n) for n, c in membership.items() if len(c) > 1 and str(n) in pos]
    if ov:
        nx.draw_networkx_nodes(G_weighted, pos, ax=ax, nodelist=ov, node_color='tomato',
                               node_size=[_hetero_size(n, mc) for n in ov],
                               edgecolors='black', linewidths=1.5)
    d_multi = [c for c in comp_all if c in multi]
    d_plain = [c for c in comp_all if c not in multi]
    nx.draw_networkx_nodes(Hcpi, pos_all, ax=ax, nodelist=d_plain, node_color='#374151',
                           node_shape='s', node_size=50, alpha=0.8)
    nx.draw_networkx_nodes(Hcpi, pos_all, ax=ax, nodelist=d_multi, node_color='#f59e0b',
                           node_shape='s', node_size=90, edgecolors='black', linewidths=0.5)
    nx.draw_networkx_labels(Hcpi, pos_all, ax=ax,
                            labels={c: _cid_label(c) for c in comp_all},
                            font_size=5, font_color='#111111',
                            bbox=dict(boxstyle='round,pad=0.1', facecolor='#fde047',
                                      edgecolor='none', alpha=0.9))

    leg = [
        Line2D([0], [0], marker='o', color='w', label='Protein overlapping',
               markerfacecolor='tomato', markersize=11, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', label='Senyawa',
               markerfacecolor='#374151', markersize=8),
        Line2D([0], [0], marker='s', color='w', label='Senyawa multi-target',
               markerfacecolor='#f59e0b', markersize=10, markeredgecolor='black'),
    ]
    ax.legend(handles=leg, loc='upper right', fontsize=9, frameon=True)
    ax.set_title(f'Graf Heterogen Gabungan (K={K}) — {len(prot_set)} protein + '
                 f'{len(comp_all)} senyawa | Overlapping: {len(ov)} | Multi-target: {len(multi)}',
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig, len(prot_set), len(comp_all), len(ov), len(multi)


# ============================================================
# TABS UTAMA
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'Visualisasi Komunitas',
    'Daftar Gen Per Komunitas',
    'Gen Overlapping',
    'Kurva Optimasi K',
    'Gene Explorer',
    'Graf Heterogen',
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

# --- TAB 6: Graf Heterogen per Komunitas (overlay senyawa) ---
with tab6:
    st.subheader(f'Graf Heterogen per Komunitas (K={selected_k})')
    protein_compounds = artifacts.get('protein_compounds')
    if not protein_compounds:
        st.warning(
            'Data senyawa (`protein_compounds`) belum tersedia di artifacts. '
            'Jalankan ulang pipeline dengan CELL 20 versi baru, buat ulang slim pickle, '
            'lalu re-deploy agar fitur ini aktif.'
        )
    else:
        st.caption(
            'Overlay node senyawa (relasi *drug-target* asli di HIN) pada komunitas hasil BigCLAM. '
            'Senyawa merupakan overlay dari data CPI, **bukan** anggota komunitas hasil deteksi. '
            'Senyawa multi-target (oranye) menarget lebih dari satu protein.'
        )
        mode = st.radio('Mode:', ['1 Komunitas', 'Gabungan Semua Komunitas'],
                        horizontal=True, key='hetero_mode')

        if mode == '1 Komunitas':
            comm_options = [f'Komunitas {i + 1}' for i in range(len(community_list))]
            sel_comm = st.selectbox('Pilih komunitas:', comm_options, key='hetero_comm')
            cidx = comm_options.index(sel_comm)
            with st.spinner('Merender graf heterogen...'):
                fig, total_c, over_syms, n_multi = render_hetero_single(
                    cidx, community_list, membership_map, protein_compounds)
            st.pyplot(fig, use_container_width=True)
            m1, m2 = st.columns(2)
            m1.metric('Total senyawa target', total_c)
            m2.metric('Senyawa multi-target', n_multi)
            if over_syms:
                st.markdown(f'**Protein overlapping di komunitas ini:** {", ".join(over_syms)}')
        else:
            with st.spinner('Merender graf gabungan...'):
                fig, n_prot, n_comp, n_ov, n_multi = render_hetero_combined(
                    community_list, membership_map, protein_compounds)
            st.pyplot(fig, use_container_width=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Protein', n_prot)
            m2.metric('Senyawa', n_comp)
            m3.metric('Overlapping', n_ov)
            m4.metric('Multi-target', n_multi)
