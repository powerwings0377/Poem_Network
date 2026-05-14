# generate_figure2_english.py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def create_network_visualization():
    print("Generating Figure 2: Tang Poetry Semantic Network Visualization...")

    # 1. Load data
    print("1. Loading data...")
    edges_df = pd.read_csv('cleaned_poetry_pairs.csv')
    community_df = pd.read_csv('community_assignment.csv')
    themes_df = pd.read_csv('community_themes.csv')

    print(f"Edge data: {len(edges_df)} edges")
    print(f"Community assignment: {len(community_df)} nodes")
    print(f"Community themes: {len(themes_df)} communities")

    # 2. Create network graph
    print("\n2. Creating network graph...")
    G = nx.Graph()

    # Create phrase to community mapping
    community_dict = {}
    for _, row in community_df.iterrows():
        phrase = str(row['phrase']).strip()
        comm_id = int(row['community'])
        community_dict[phrase] = comm_id
        G.add_node(phrase, community=comm_id)

    print(f"Added {len(G.nodes())} nodes")

    # Add edges
    edge_count = 0
    for _, row in edges_df.iterrows():
        source = str(row['词1']).strip()
        target = str(row['词2']).strip()
        weight = row['NPMI值']

        if source in community_dict and target in community_dict:
            G.add_edge(source, target, weight=weight)
            edge_count += 1

    print(f"Added {edge_count} edges")
    print(f"Final network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 3. Load actual community information
    community_sizes = {}
    community_labels = {}

    # Load from themes_df
    print("\n3. Loading community theme information...")
    for _, row in themes_df.iterrows():
        # Check column names
        if 'community_id' in themes_df.columns:
            comm_id = int(row['community_id'])
        elif 'community' in themes_df.columns:
            comm_id = int(row['community'])
        else:
            comm_id = int(row.name)

        size = int(row['size'])
        theme_label_chinese = row['theme_label']

        # Translate Chinese theme labels to English
        theme_translation = {
            '花春风年': 'Spring Scene & Boudoir Sentiment',
            '秋月风山': 'Autumn Journey & Homesickness',
            '人白相年': 'Life Experience & Career Aspiration',
            '山水人年': 'Reclusive Landscape & Zen Philosophy',
            '千山天万': 'Court Majesty & Historical Narrative',
            '风秋山客': 'Autumn Journey & Homesickness'
        }

        theme_label = theme_translation.get(theme_label_chinese, f'Community {comm_id}')

        community_sizes[comm_id] = size
        community_labels[comm_id] = {
            'label': theme_label,
            'chinese_label': theme_label_chinese,
            'size': size
        }

        print(f"  Community {comm_id}: {theme_label} ({size} nodes)")

    # 4. Calculate node sizes (based on degree centrality)
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1

    node_sizes = []
    for node in G.nodes():
        deg = degrees.get(node, 1)
        size = 200 + (deg / max_degree) * 1800
        node_sizes.append(size)

    # 5. Prepare node colors (by community)
    community_colors = {
        0: '#E74C3C',  # Red - Spring Scene
        1: '#3498DB',  # Blue - Autumn Journey
        2: '#2ECC71',  # Green - Life Experience
        3: '#9B59B6'  # Purple - Reclusive Landscape
    }

    node_colors = []
    for node in G.nodes():
        comm_id = G.nodes[node].get('community', 0)
        color = community_colors.get(comm_id, '#95A5A6')
        node_colors.append(color)

    # 6. Create layout
    print("\n4. Calculating network layout...")

    # Use spring layout with important nodes
    important_nodes = [node for node, deg in degrees.items() if deg > 30]
    if len(important_nodes) > 100:
        important_nodes = important_nodes[:100]

    if important_nodes:
        G_important = G.subgraph(important_nodes)
        pos_important = nx.spring_layout(G_important, k=2.0, iterations=100, seed=42)

        pos = {}
        for node in G.nodes():
            if node in pos_important:
                pos[node] = pos_important[node]
            else:
                neighbors = list(G.neighbors(node))
                positioned_neighbors = [n for n in neighbors if n in pos_important]
                if positioned_neighbors:
                    avg_x = np.mean([pos_important[n][0] for n in positioned_neighbors])
                    avg_y = np.mean([pos_important[n][1] for n in positioned_neighbors])
                    pos[node] = (avg_x + np.random.uniform(-0.3, 0.3),
                                 avg_y + np.random.uniform(-0.3, 0.3))
                else:
                    pos[node] = (np.random.uniform(-2, 2), np.random.uniform(-2, 2))
    else:
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    # 7. Draw network
    print("5. Drawing network visualization...")
    fig, ax = plt.subplots(figsize=(16, 14))

    # Draw edges
    print("  Drawing edges...")
    edges_batch = list(G.edges())[:8000]
    edge_weights = [G[u][v]['weight'] for u, v in edges_batch]

    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        if max_weight > min_weight:
            normalized_weights = [(w - min_weight) / (max_weight - min_weight)
                                  for w in edge_weights]
        else:
            normalized_weights = [0.5] * len(edge_weights)

        for (u, v), alpha in zip(edges_batch, normalized_weights):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                                   alpha=alpha * 0.3 + 0.05,
                                   width=alpha * 0.8 + 0.2,
                                   edge_color='gray',
                                   ax=ax)

    # Draw nodes
    print("  Drawing nodes...")
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.85,
                           edgecolors='white',
                           linewidths=1.0,
                           ax=ax)

    # Label important nodes
    important_nodes_labels = []
    for node, degree in degrees.items():
        if degree > 80:
            important_nodes_labels.append(node)

    print(f"  Labeling {len(important_nodes_labels)} important nodes")

    if important_nodes_labels:
        labels = {node: node for node in important_nodes_labels}
        nx.draw_networkx_labels(G, pos,
                                labels=labels,
                                font_size=10,
                                font_weight='bold',
                                ax=ax)

    # 8. Add legend
    from matplotlib.patches import Patch

    legend_elements = []
    for comm_id in sorted(community_colors.keys()):
        if comm_id in community_labels:
            info = community_labels[comm_id]
            label = f"Community {comm_id}: {info['label']}\n({info['size']} nodes)"
        else:
            label = f"Community {comm_id} ({community_sizes.get(comm_id, 0)} nodes)"

        legend_elements.append(
            Patch(facecolor=community_colors[comm_id],
                  edgecolor='white',
                  label=label)
        )

    ax.legend(handles=legend_elements,
              loc='upper right',
              fontsize=9,
              frameon=True,
              fancybox=True,
              shadow=True,
              bbox_to_anchor=(1.0, 1.0))

    # 9. Add title and statistics
    plt.title(
        'Figure 2: Semantic Network of Tang Poetry Binary Phrases\n(Node color represents semantic community, size represents degree centrality)',
        fontsize=16,
        fontweight='bold',
        pad=20)

    # Network statistics
    stats_text = f"""Network Statistics:
• Number of nodes: {G.number_of_nodes()}
• Number of edges: {G.number_of_edges()}
• Network density: {nx.density(G):.4f}
• Average degree: {np.mean(list(degrees.values())):.1f}
• Modularity: 0.183"""

    plt.text(0.02, 0.02, stats_text,
             transform=ax.transAxes,
             fontsize=11,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.axis('off')

    # 10. Save figure
    plt.tight_layout()
    output_png = 'figure2_english.png'
    output_pdf = 'figure2_english.pdf'

    plt.savefig(output_png,
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.savefig(output_pdf,
                bbox_inches='tight',
                facecolor='white')

    plt.show()
    print(f"\n✓ Figure 2 generated:")
    print(f"  {output_png}")
    print(f"  {output_pdf}")

    return G, pos, community_labels


if __name__ == '__main__':
    try:
        G, pos, community_labels = create_network_visualization()
    except Exception as e:
        print(f"\n✗ Error generating figure: {e}")
        import traceback

        traceback.print_exc()