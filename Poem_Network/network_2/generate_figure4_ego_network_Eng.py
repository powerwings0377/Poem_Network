# generate_figure4_ego_network_english.py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def generate_yangliu_ego_network_english():
    """Generate Ego-Network of 'Yangliu' (Willow)"""
    print("Generating Figure 4: Ego-Network of Core Phrase 'Yangliu'...")

    # 1. Load data
    edges_df = pd.read_csv('cleaned_poetry_pairs.csv')
    community_df = pd.read_csv('community_assignment.csv')
    themes_df = pd.read_csv('community_themes.csv')

    print(f"Data loaded: {len(edges_df)} edges, {len(community_df)} nodes")

    # 2. Create network graph
    G = nx.Graph()

    # Create phrase to community mapping
    community_dict = {}
    for _, row in community_df.iterrows():
        phrase = str(row['phrase']).strip()
        comm_id = int(row['community'])
        community_dict[phrase] = comm_id
        G.add_node(phrase, community=comm_id)

    # Add edges
    edge_count = 0
    for _, row in edges_df.iterrows():
        source = str(row['词1']).strip()
        target = str(row['词2']).strip()
        weight = row['NPMI值']

        if source in community_dict and target in community_dict:
            G.add_edge(source, target, weight=weight)
            edge_count += 1

    print(f"Network constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 3. Check if '杨柳' exists in network
    central_node = "杨柳"
    if central_node not in G.nodes():
        print(f"Warning: '{central_node}' not found in network!")
        # Find nodes containing "杨" or "柳"
        similar_nodes = [n for n in G.nodes() if "杨" in n or "柳" in n]
        if similar_nodes:
            print(f"Similar nodes found: {similar_nodes}")
            central_node = similar_nodes[0]
            print(f"Using '{central_node}' as central node")
        else:
            # Use highest degree node as alternative
            degrees = dict(G.degree())
            top_node = max(degrees.items(), key=lambda x: x[1])[0]
            print(f"No similar nodes found, using highest degree node '{top_node}'")
            central_node = top_node

    # 4. Extract ego-network (1-hop neighbors)
    print(f"\nExtracting ego-network of '{central_node}'...")
    ego = nx.ego_graph(G, central_node, radius=1)
    print(f"Ego-network: {ego.number_of_nodes()} nodes, {ego.number_of_edges()} edges")

    # 5. Prepare visualization parameters
    # Community color mapping (consistent with Figure 2)
    community_colors = {
        0: '#E74C3C',  # Red - Spring Scene
        1: '#3498DB',  # Blue - Autumn Journey
        2: '#2ECC71',  # Green - Life Experience
        3: '#9B59B6'  # Purple - Reclusive Landscape
    }

    # Community English labels
    community_labels_english = {
        0: "Spring Scene & Boudoir Sentiment",
        1: "Autumn Journey & Homesickness",
        2: "Life Experience & Career Aspiration",
        3: "Reclusive Landscape & Zen Philosophy"
    }

    # Node colors and sizes
    node_colors = []
    node_sizes = []
    node_border_colors = []
    node_border_widths = []

    for node in ego.nodes():
        if node == central_node:
            # Central node: gold, large size, thick border
            node_colors.append('gold')
            node_sizes.append(2800)
            node_border_colors.append('darkorange')
            node_border_widths.append(3.0)
        else:
            # Neighbor nodes: colored by community
            comm_id = G.nodes[node].get('community', 0)
            color = community_colors.get(comm_id, '#95A5A6')
            node_colors.append(color)
            node_border_colors.append('white')
            node_border_widths.append(1.5)

            # Node size based on NPMI weight with central node
            if central_node in ego and node in ego[central_node]:
                weight = ego[central_node][node]['weight']
                size = 1000 + weight * 1200  # Higher NPMI = larger node
            else:
                size = 800
            node_sizes.append(size)

    # 6. Create layout
    print("Calculating network layout...")

    # Use spring layout with central node fixed at center
    pos = nx.spring_layout(ego, seed=42, k=1.5, iterations=150)

    # Ensure central node at center position
    pos[central_node] = (0, 0)

    # Adjust neighbor positions for better visualization
    for node in ego.nodes():
        if node != central_node:
            if node in pos:
                x, y = pos[node]
                # Push nodes outward slightly
                distance = np.sqrt(x ** 2 + y ** 2)
                if distance > 0:
                    scale = 1.2  # Expansion factor
                    pos[node] = (x * scale, y * scale)

    # 7. Draw network
    print("Drawing network visualization...")
    plt.figure(figsize=(15, 13))

    # 7.1 Draw edges (categorized by NPMI weight)
    print("  Drawing edges...")

    # Categorize edges by weight
    strong_edges = []  # NPMI > 0.4
    medium_edges = []  # 0.3 < NPMI ≤ 0.4
    weak_edges = []  # NPMI ≤ 0.3

    for u, v, data in ego.edges(data=True):
        weight = data.get('weight', 0.3)
        if weight > 0.4:
            strong_edges.append((u, v, weight))
        elif weight > 0.3:
            medium_edges.append((u, v, weight))
        else:
            weak_edges.append((u, v, weight))

    # Draw weak edges first (as background)
    if weak_edges:
        weak_edgelist = [(u, v) for u, v, w in weak_edges]
        nx.draw_networkx_edges(ego, pos,
                               edgelist=weak_edgelist,
                               width=0.8,
                               alpha=0.3,
                               edge_color='lightgray',
                               style='dashed')

    # Draw medium strength edges
    if medium_edges:
        medium_edgelist = [(u, v) for u, v, w in medium_edges]
        nx.draw_networkx_edges(ego, pos,
                               edgelist=medium_edgelist,
                               width=1.5,
                               alpha=0.5,
                               edge_color='gray')

    # Draw strong edges (most significant)
    if strong_edges:
        strong_edgelist = [(u, v) for u, v, w in strong_edges]
        nx.draw_networkx_edges(ego, pos,
                               edgelist=strong_edgelist,
                               width=2.5,
                               alpha=0.7,
                               edge_color='darkblue')

    # 7.2 Draw nodes
    print("  Drawing nodes...")
    nx.draw_networkx_nodes(ego, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9,
                           edgecolors=node_border_colors,
                           linewidths=node_border_widths)

    # 7.3 Draw node labels
    print("  Drawing node labels...")

    # Add labels for all nodes, central node more prominent
    for node in ego.nodes():
        if node == central_node:
            # Central node: larger font, bold, special color
            nx.draw_networkx_labels(ego, pos,
                                    labels={node: node},
                                    font_size=14,
                                    font_weight='bold',
                                    font_color='darkred')
        else:
            # Neighbor nodes: normal labels
            nx.draw_networkx_labels(ego, pos,
                                    labels={node: node},
                                    font_size=10,
                                    font_weight='normal')

    # 7.4 Draw edge weight labels (only strong associations)
    print("  Drawing edge weight labels...")
    edge_labels = {}
    for u, v, data in ego.edges(data=True):
        weight = data.get('weight', 0)
        if weight > 0.35:  # Only label strong associations
            edge_labels[(u, v)] = f'{weight:.2f}'

    nx.draw_networkx_edge_labels(ego, pos,
                                 edge_labels=edge_labels,
                                 font_size=9,
                                 font_weight='bold',
                                 font_color='darkgreen',
                                 label_pos=0.5,
                                 bbox=dict(alpha=0))

    # 8. Add legend
    print("Adding legend...")
    from matplotlib.lines import Line2D

    # Create custom legend handles
    legend_elements = [
        Patch(facecolor='gold', edgecolor='darkorange', linewidth=3,
              label=f'Central Node: "{central_node}" (Willow)'),
        Line2D([0], [0], color='darkblue', linewidth=2.5,
               label='Strong Association (NPMI > 0.4)', alpha=0.7),
        Line2D([0], [0], color='gray', linewidth=1.5,
               label='Medium Association (0.3 < NPMI ≤ 0.4)', alpha=0.5),
        Line2D([0], [0], color='lightgray', linewidth=0.8, linestyle='dashed',
               label='Weak Association (NPMI ≤ 0.3)', alpha=0.3),
    ]

    # Add community color legend
    for comm_id in sorted(community_colors.keys()):
        if comm_id in community_labels_english:
            label = f"Community {comm_id}: {community_labels_english[comm_id]}"
        else:
            label = f"Community {comm_id}"

        legend_elements.append(
            Patch(facecolor=community_colors[comm_id],
                  edgecolor='white',
                  label=label)
        )

    plt.legend(handles=legend_elements,
               loc='upper right',
               fontsize=9,
               frameon=True,
               fancybox=True,
               shadow=True,
               bbox_to_anchor=(1.0, 1.0),
               title="Legend",
               title_fontsize=10)

    # 9. Add title and statistics
    plt.title(
        f"Figure 4: Ego-Network of Core Phrase '{central_node}' (Willow)\n(Edge labels show NPMI association strength, node color indicates semantic community)",
        fontsize=16,
        fontweight='bold',
        pad=25)

    # Statistics panel
    # Calculate community distribution of neighbors
    neighbor_comm_dist = {}
    for neighbor in ego.neighbors(central_node):
        comm_id = G.nodes[neighbor].get('community', 0)
        neighbor_comm_dist[comm_id] = neighbor_comm_dist.get(comm_id, 0) + 1

    # Calculate average NPMI with neighbors
    neighbor_weights = [G[central_node][n]['weight'] for n in ego.neighbors(central_node)]
    avg_npmi = np.mean(neighbor_weights) if neighbor_weights else 0

    # Find strongest associations
    strongest_links = []
    for neighbor in ego.neighbors(central_node):
        weight = G[central_node][neighbor]['weight']
        comm_id = G.nodes[neighbor].get('community', 0)
        strongest_links.append((neighbor, weight, comm_id))

    strongest_links.sort(key=lambda x: x[1], reverse=True)
    top_links = strongest_links[:3]  # Top 3 strongest associations

    stats_text = f"""Network Statistics:
• Central node degree: {ego.degree(central_node)}
• Direct neighbors: {ego.number_of_nodes() - 1}
• Average association strength: {avg_npmi:.3f}
• Cross-community connections: {len(neighbor_comm_dist)}

Neighbor Community Distribution:"""

    for comm_id, count in sorted(neighbor_comm_dist.items()):
        comm_name = community_labels_english.get(comm_id, f"Community {comm_id}")
        stats_text += f"\n  • {comm_name}: {count} nodes"

    if top_links:
        stats_text += f"\n\nTop 3 Strongest Associations:"
        for neighbor, weight, comm_id in top_links:
            comm_name = community_labels_english.get(comm_id, f"Community {comm_id}")
            stats_text += f"\n  • {neighbor}: NPMI={weight:.3f} ({comm_name})"

    plt.text(0.02, 0.02, stats_text,
             transform=plt.gca().transAxes,
             fontsize=9.5,
             verticalalignment='bottom',
             horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9,
                       edgecolor='orange', linewidth=1.5))

    plt.axis('off')

    # 10. Save figure
    print("Saving figure...")
    output_png = 'figure4_ego_network_yangliu_english.png'
    output_pdf = 'figure4_ego_network_yangliu_english.pdf'

    plt.tight_layout()
    plt.savefig(output_png,
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.savefig(output_pdf,
                bbox_inches='tight',
                facecolor='white')

    plt.show()

    print(f"\n✓ Figure 4 generated:")
    print(f"  {output_png}")
    print(f"  {output_pdf}")

    # 11. Generate detailed analysis report
    print("\n" + "=" * 60)
    print(f"Detailed Analysis Report: Ego-Network of '{central_node}' (Willow)")
    print("=" * 60)

    print(f"\n1. Basic Network Properties")
    print(f"   • Central phrase: '{central_node}' (Willow)")
    print(f"   • Degree in full network: {G.degree(central_node)}")

    # Find rank in full network
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    rank = next(i for i, (node, _) in enumerate(sorted_nodes) if node == central_node) + 1
    print(f"   • Rank in full network: {rank}/{len(G.nodes())}")
    print(f"   • Direct neighbors: {list(ego.neighbors(central_node))}")

    print(f"\n2. Semantic Community Connectivity")
    print(f"   Central node connects {len(neighbor_comm_dist)} semantic communities:")
    total_neighbors = len(list(ego.neighbors(central_node)))
    for comm_id, count in sorted(neighbor_comm_dist.items()):
        comm_name = community_labels_english.get(comm_id, f"Community {comm_id}")
        percentage = count / total_neighbors * 100
        print(f"   • {comm_name}: {count} neighbors ({percentage:.1f}%)")

    print(f"\n3. Association Strength Analysis")
    print(f"   • Average NPMI association strength: {avg_npmi:.3f}")
    print(f"   • Association strength distribution:")
    print(f"     - Strong associations (NPMI > 0.4): {len([w for w in neighbor_weights if w > 0.4])}")
    print(f"     - Medium associations (0.3-0.4): {len([w for w in neighbor_weights if 0.3 <= w <= 0.4])}")
    print(f"     - Weak associations (NPMI < 0.3): {len([w for w in neighbor_weights if w < 0.3])}")

    print(f"\n4. Network Role Assessment")
    if len(neighbor_comm_dist) >= 3:
        print(f"   ✓ '{central_node}' functions as a BRIDGE NODE")
        print(f"     Connects multiple semantic communities, indicating multi-dimensional semantic functions")
    elif len(neighbor_comm_dist) == 2:
        print(f"   • '{central_node}' serves as an INTERMEDIARY NODE")
        print(f"     Primarily establishes semantic links between two communities")
    else:
        print(f"   • '{central_node}' operates mainly within a single community")
        print(f"     Plays a core role within that specific theme")

    # Calculate bridge coefficient
    if len(neighbor_comm_dist) > 1:
        bridge_score = len(neighbor_comm_dist) / len(list(ego.neighbors(central_node)))
        print(f"\n5. Bridge Coefficient Evaluation")
        print(f"   • Bridge coefficient: {bridge_score:.3f} (range 0-1, higher indicates stronger bridge function)")
        if bridge_score > 0.4:
            print(f"   → '{central_node}' exhibits significant bridge function in Tang poetry imagery system")

    return ego, pos


if __name__ == '__main__':
    try:
        print("=" * 60)
        print("Tang Poetry Ego-Network Generation System (English Version)")
        print("=" * 60)

        ego, pos = generate_yangliu_ego_network_english()

        print("\n" + "=" * 60)
        print("Figure generation completed!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nError: File not found - {e.filename}")
        print("Please ensure the following files are in current directory:")
        print("1. cleaned_poetry_pairs.csv")
        print("2. community_assignment.csv")
        print("3. community_themes.csv")

    except Exception as e:
        print(f"\nError generating figure: {e}")
        import traceback

        traceback.print_exc()