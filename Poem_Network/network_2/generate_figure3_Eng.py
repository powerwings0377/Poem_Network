# generate_figure3_english.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_community_heatmap():
    print("Generating Figure 3: Inter-Community Association Heatmap...")

    # 1. Load data
    edges_df = pd.read_csv('cleaned_poetry_pairs.csv')
    community_df = pd.read_csv('community_assignment.csv')
    themes_df = pd.read_csv('community_themes.csv')

    # Create phrase to community mapping
    phrase_to_community = {}
    for _, row in community_df.iterrows():
        phrase = str(row['phrase']).strip()
        comm_id = int(row['community'])
        phrase_to_community[phrase] = comm_id

    # 2. Load community information with English translation
    community_info = {}
    for _, row in themes_df.iterrows():
        # Get community ID
        if 'community_id' in themes_df.columns:
            comm_id = int(row['community_id'])
        elif 'community' in themes_df.columns:
            comm_id = int(row['community'])
        else:
            comm_id = int(row.name)

        theme_label_chinese = str(row['theme_label'])

        # Translate to English
        theme_translation = {
            '花春风年': 'Spring Scene & Boudoir Sentiment',
            '秋月风山': 'Autumn Journey & Homesickness',
            '人白相年': 'Life Experience & Career Aspiration',
            '山水人年': 'Reclusive Landscape & Zen Philosophy',
            '千山天万': 'Court Majesty & Historical Narrative',
            '风秋山客': 'Autumn Journey & Homesickness'
        }

        theme_label = theme_translation.get(theme_label_chinese, f'Community {comm_id}')

        size = int(row['size']) if 'size' in themes_df.columns else 0

        community_info[comm_id] = {
            'label': theme_label,
            'chinese_label': theme_label_chinese,
            'size': size
        }

    print("\nCommunity Information:")
    for comm_id, info in sorted(community_info.items()):
        print(f"  Community {comm_id}: {info['label']} ({info['size']} nodes)")

    # 3. Calculate average NPMI between communities
    communities = sorted(set(phrase_to_community.values()))
    n_communities = len(communities)

    print(f"\nDetected {n_communities} communities: {communities}")

    # Initialize matrices
    heatmap_matrix = np.zeros((n_communities, n_communities))
    count_matrix = np.zeros((n_communities, n_communities), dtype=int)

    # Process edges
    print("Calculating inter-community associations...")
    edge_count = 0
    for _, row in edges_df.iterrows():
        source = str(row['词1']).strip()
        target = str(row['词2']).strip()
        weight = row['NPMI值']

        if source in phrase_to_community and target in phrase_to_community:
            comm1 = phrase_to_community[source]
            comm2 = phrase_to_community[target]

            idx1 = communities.index(comm1)
            idx2 = communities.index(comm2)

            heatmap_matrix[idx1, idx2] += weight
            heatmap_matrix[idx2, idx1] += weight
            count_matrix[idx1, idx2] += 1
            count_matrix[idx2, idx1] += 1
            edge_count += 1

    print(f"  Processed {edge_count} edges")

    # Calculate averages
    for i in range(n_communities):
        for j in range(n_communities):
            if count_matrix[i, j] > 0:
                heatmap_matrix[i, j] = heatmap_matrix[i, j] / count_matrix[i, j]
            else:
                heatmap_matrix[i, j] = 0

    # 4. Create heatmap
    print("\nGenerating heatmap...")
    plt.figure(figsize=(12, 10))

    # Create axis labels
    x_labels = []
    y_labels = []
    for comm_id in communities:
        info = community_info.get(comm_id, {})
        label = info.get('label', f'Community {comm_id}')
        x_labels.append(f"C{comm_id}\n{label}")
        y_labels.append(f"C{comm_id}\n{label}")

    # Create heatmap
    ax = sns.heatmap(heatmap_matrix,
                     annot=True,
                     fmt=".3f",
                     cmap='YlOrRd',
                     linewidths=1,
                     linecolor='white',
                     square=True,
                     xticklabels=x_labels,
                     yticklabels=y_labels,
                     cbar_kws={'label': 'Average NPMI Association Strength',
                               'shrink': 0.8})

    # Adjust labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

    # Add edge count annotations
    for i in range(n_communities):
        for j in range(n_communities):
            if count_matrix[i, j] > 0:
                ax.text(j + 0.5, i + 0.65,
                        f'n={count_matrix[i, j]}',
                        ha='center',
                        va='top',
                        fontsize=8,
                        color='darkblue',
                        fontweight='bold')

    # 5. Add title
    plt.title('Figure 3: Inter-Community Association Strength Heatmap\n(C=Community, n=number of connecting edges)',
              fontsize=14,
              fontweight='bold',
              pad=20)

    plt.tight_layout()

    # 6. Save figure
    output_png = 'figure3_english.png'
    output_pdf = 'figure3_english.pdf'

    plt.savefig(output_png,
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.savefig(output_pdf,
                bbox_inches='tight',
                facecolor='white')

    plt.show()

    # 7. Print detailed statistics
    print("\n=== Detailed Inter-Community Association Statistics ===")
    print(f"{'From':^20} {'To':^20} {'Avg NPMI':^12} {'Edges':^10} {'Proportion':^12}")
    print("=" * 74)

    for i, comm1 in enumerate(communities):
        for j, comm2 in enumerate(communities[i:], i):
            if count_matrix[i, j] > 0:
                label1 = f"C{comm1}"
                label2 = f"C{comm2}"
                avg_npmi = heatmap_matrix[i, j]
                edges = count_matrix[i, j]
                proportion = edges / len(edges_df) * 100

                print(f"{label1:^20} {label2:^20} {avg_npmi:^12.4f} {edges:^10} {proportion:^10.2f}%")

    print(f"\n✓ Figure 3 generated:")
    print(f"  {output_png}")
    print(f"  {output_pdf}")

    return heatmap_matrix, count_matrix, community_info


if __name__ == '__main__':
    try:
        heatmap_matrix, count_matrix, community_info = create_community_heatmap()
    except Exception as e:
        print(f"\n✗ Error generating figure: {e}")
        import traceback

        traceback.print_exc()