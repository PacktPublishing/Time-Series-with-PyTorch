# Extracted from chapter15_TS_Clustering.qmd
# Do not edit the source .qmd file directly.

#| label: setup-libraries
#| message: false
#| echo: false
#| eval: true

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import seaborn as sns

from datetime import datetime
from pathlib import Path
from typing import Union, Sequence
from IPython.display import display, HTML

palette = ["#000000", "#0072B2", "#D55E00","#009E73","#CC79A7", "#56B4E9","#E69F00"]
custom_palette = palette  # alias used in later plotting code

class CFG:
    data_folder = Path.cwd().parent / "data"
    img_dim1 = 8
    img_dim2 = 4

def display_html_table(
    df: pd.DataFrame,
    n_rows: int = 3,
    cols: Union[int, Sequence[str], None] = None
) -> None:
    """
    Display a DataFrame with HTML in IPython.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to display.
    n_rows : int, optional (default=3)
        Number of top rows to show (uses df.head(n_rows)).
    cols : int, sequence of str, or None, optional
        - If None: show all columns.
        - If int: show the first `cols` columns.
        - If list of str: show only the specified column names.

    Examples
    --------
    display_html_table(df, n_rows=3, cols=6)
    display_html_table(df, n_rows=3, cols=['sales','price'])
    """
    sub = df.head(n_rows)
    
    if isinstance(cols, int):
        sub = sub.iloc[:, :cols]
    elif isinstance(cols, (list, tuple)):
        sub = sub.loc[:, cols]
    
    html = sub.to_html(classes="table table-striped", border=0)
    display(HTML(html))

# ----------------------------------------------------------------------

#| label: tbl-iris-load
#| tbl-cap: "Table 15.1: First three rows of the Iris dataset with the four measurement columns and species label."
#| message: false
#| echo: true
#| eval: true

from sklearn import datasets

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

display_html_table(df, n_rows=3, cols=6)

# ----------------------------------------------------------------------

#| label: fig-iris-pairplot
#| fig-cap: "Figure 15.1: Pairplot of the four Iris measurements coloured by species, showing *setosa* as linearly separable and some overlap between *versicolor* and *virginica*."
#| message: false
#| echo: false
#| eval: true

column_list2 = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]

with sns.axes_style("white"):
    eda1 = sns.pairplot(
        df,
        hue="species",          
        diag_kind="kde",
        palette=palette,
        markers=["o", "s", "D"],
        vars=column_list2,
        height=2                
    )

handles, labels = eda1._legend_data.values(), eda1._legend_data.keys()
eda1._legend.remove()
eda1.fig.legend(
    handles=handles,
    labels=labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.45, 1.03)
)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: fig-iris-boxplot
#| fig-cap: "Figure 15.2: Boxplots of the four Iris measurements by species, showing that petal length and width separate the three species more cleanly than the sepal measurements."
#| message: false
#| echo: false
#| eval: true

melted = pd.melt(
    df,
    id_vars="species",
    value_vars=column_list2,
    var_name="variable",
    value_name="value"
)

plt.figure(figsize=(8, 4))
dodge_amount = 2  
box_width    = 0.8   

ax = sns.boxplot(
    x="variable", y="value", hue="species", data=melted, palette=palette,
    width=box_width, dodge=dodge_amount, linewidth=0.8, fliersize=0, 
)

sns.swarmplot(
    x="variable", y="value", hue="species", data=melted, palette=palette,
    size=2, dodge=dodge_amount, edgecolor="w", linewidth=0.15, alpha=0.7,
    legend=False, 
)

ax.set_xlabel("")
ax.set_ylabel("Measurement (cm)", fontsize=12, fontweight="bold")
ax.tick_params(axis="x", labelsize=12, rotation=15)
ax.tick_params(axis="y", labelsize=12)

leg = ax.legend(title="", loc="upper right", fontsize=14, frameon=False)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: hopkins-analysis
#| message: false
#| echo: true
#| eval: true

from sklearn.neighbors import NearestNeighbors

def hopkins(X, frac=0.1, random_state=None):
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    m = int(frac * n)
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)

    real_idx = rng.choice(n, m, replace=False)
    real = X[real_idx]

    mins, maxs = X.min(axis=0), X.max(axis=0)
    uniform = rng.uniform(mins, maxs, size=(m, d))

    u_dist = nbrs.kneighbors(uniform, return_distance=True)[0][:, 1]
    w_dist = nbrs.kneighbors(real,    return_distance=True)[0][:, 1]

    return u_dist.sum() / (u_dist.sum() + w_dist.sum())


X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values

h2 = hopkins(X, frac=0.1, random_state=43526)
print(f"Hopkins (fast): {h2:.4f}")

# ----------------------------------------------------------------------

#| label: elbow-plotting-function
#| message: false
#| echo: false
#| eval: true

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from yellowbrick.cluster import KElbowVisualizer

def kmeans_elbow(
    X,
    palette,
    k_distortion=(1, 11),
    k_ch=(2, 8),
    figsize=(12, 5),
    title_fontsize=16,
    label_fontsize=9,
    color_grid="lightgray"
):
    """
    Display two K-Means elbow plots: distortion and Calinski-Harabasz.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    palette : list of colors
        Seaborn palette for the elbow plots.
    k_distortion : tuple (min_k, max_k)
        Range of K to search for distortion elbow.
    k_ch : tuple (min_k, max_k)
        Range of K to search for Calinski-Harabasz elbow.
    """
    sns.set_palette(palette)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    model = KMeans(random_state=4)

    viz1 = KElbowVisualizer(
        model,
        k=k_distortion,
        ax=ax1,
        metric="distortion",
        timings=False
    )
    viz1.fit(X)
    viz1.finalize()
    ax1.set_title("Distortion Elbow", fontsize=label_fontsize, fontweight="bold")
    ax1.set_xlabel("K", fontsize=label_fontsize)
    ax1.set_ylabel("Distortion", fontsize=label_fontsize)
    ax1.grid(axis="y", linestyle=":", color=color_grid, alpha=0.7)

    viz2 = KElbowVisualizer(
        model,
        k=k_ch,
        metric="calinski_harabasz",
        ax=ax2,
        timings=False
    )
    viz2.fit(X)
    viz2.finalize()
    ax2.set_title("Calinski-Harabasz Elbow", fontsize=label_fontsize, fontweight="bold")
    ax2.set_xlabel("K", fontsize=label_fontsize)
    ax2.set_ylabel("Calinski-Harabasz Score", fontsize=label_fontsize)
    ax2.grid(axis="y", linestyle=":", color=color_grid, alpha=0.7)

    fig.suptitle("K-Means Elbow Analysis",
                 fontsize=title_fontsize, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------

#| label: fig-iris-elbow
#| fig-cap: "Figure 15.3: K-means elbow analysis on the Iris dataset showing distortion and Calinski-Harabasz scores across values of $k$, both indicating $k = 3$."
#| message: false
#| echo: false
#| eval: true

kmeans_elbow(
    X=X,
    palette=palette
)

# ----------------------------------------------------------------------

#| label: iris-cluster-visuals-function
#| message: false
#| echo: false
#| eval: true

from yellowbrick.cluster import SilhouetteVisualizer

def kmeans_cluster_visuals(X, kmeans, y_kmeans, cluster_colors, font_main, font_alt, color_grid, color_line, scatter_color_edge):
    """
    Generate K-Means clustering diagnostics: silhouette plot, cluster scatter, waffle chart.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={'height_ratios': [1, 1.1]})
    ax1, ax2 = axes[0]
    ax3 = plt.subplot2grid((2,2), (1,0), colspan=2)

    s_viz = SilhouetteVisualizer(kmeans, ax=ax1, colors=cluster_colors)
    s_viz.fit(X)
    s_viz.finalize()
    ax1.set_title('Silhouette Plot of Clusters', fontsize=13, fontweight='bold', fontname=font_main)
    ax1.set_xlabel('Coefficient Values', fontsize=9, fontweight='bold', fontname=font_main)
    ax1.set_ylabel('Cluster Labels', fontsize=9, fontweight='bold', fontname=font_main)
    ax1.tick_params(labelsize=8)
    ax1.grid(axis='x', alpha=0.5, color=color_grid, linestyle='dotted')
    for spine in ax1.spines.values(): spine.set_color('None')

    for i, color in enumerate(cluster_colors):
        ax2.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], 
                    s=36, c=[color], label=f"Cluster {i+1}", linewidth=0.65, edgecolor=scatter_color_edge, alpha=0.9)
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=70, c='#42A2FC', marker='X', label='Centroids', linewidth=0.8, edgecolor='black', alpha=1)
    ax2.set_title('Cluster Distribution (PCA projection)', fontsize=13, fontweight='bold', fontname=font_main)
    ax2.grid(alpha=0.5, color=color_grid, linestyle='dotted')
    ax2.set_xlabel('PC1', fontsize=9, fontweight='bold', fontname=font_main)
    ax2.set_ylabel('PC2', fontsize=9, fontweight='bold', fontname=font_main)
    ax2.legend(loc='upper right', fontsize=8, frameon=False)
    ax2.tick_params(axis="both", labelsize=8)
    for spine in ax2.spines.values(): spine.set_color(color_line)

    unique, counts = np.unique(y_kmeans, return_counts=True)
    df_waffle = dict(zip(unique, counts))
    labels = ["Virginica", "Versicolor", "Setosa"]
    label_map = dict(zip(sorted(df_waffle.keys()), labels))
    values = [df_waffle[i] for i in sorted(df_waffle.keys())]
    value_labels = [f"{label_map[i]} - {round(100*v/sum(values),1)}%" for i, v in zip(sorted(df_waffle.keys()), values)]
    Waffle.make_waffle(ax=ax3, rows=6, values=values, colors=cluster_colors, labels=value_labels,
                       legend={'loc':'upper center', 'bbox_to_anchor':(0.5, -0.15), 'ncol': 3, 'borderpad': 2, 'frameon': False, 'fontsize':10})
    ax3.set_title('Cluster Membership Percentage', fontsize=13, fontweight='bold', fontname=font_main)

    plt.suptitle('Iris Clustering using K-Means', fontsize=15, fontweight='bold', fontname=font_main)
    plt.gcf().text(0.9, 0.03, 'kaggle.com/caesarmario', style='italic', fontsize=7, fontname=font_alt)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# ----------------------------------------------------------------------

#| label: iris-clustering-tsne
#| message: false
#| echo: true
#| eval: true

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

kmeans = KMeans(
  n_clusters=3, 
  init='k-means++', 
  n_init='auto',
  max_iter=300,
  algorithm='elkan',
  random_state=5363)
y_kmeans = kmeans.fit_predict(X)

df['cluster'] = y_kmeans

tsne = TSNE(
        n_components=2, 
        perplexity=50, 
        n_iter=1200, 
        random_state=4126, 
        method='barnes_hut',
        init="pca",
        n_jobs=-1
    )

X_tsne = tsne.fit_transform(X)

# ----------------------------------------------------------------------

#| label: silhouette-tsne-function
#| message: false
#| echo: false
#| eval: true

from yellowbrick.cluster import SilhouetteVisualizer

def plot_silhouette_tsne(X, X_tsne, y_kmeans, kmeans, palette, labels=None, figsize=(12,5), s=36):
    """
    Plots silhouette and t-SNE scatter together.

    Parameters
    ----------
    X : ndarray, original data used for clustering (n_samples, n_features)
    X_tsne : ndarray, t-SNE embedding (n_samples, 2)
    y_kmeans : ndarray, cluster labels (n_samples,)
    kmeans : fitted KMeans object
    palette : list, colour palette for clusters
    labels : list or None, custom cluster names (optional)
    figsize : tuple, figure size
    s : int, marker size for scatter
    """
    sns.set_palette(palette)
    n_clusters = kmeans.n_clusters
    if labels is None:
        labels = [f"Cluster {i+1}" for i in range(n_clusters)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    s_viz = SilhouetteVisualizer(kmeans, ax=ax1, colors=palette)
    s_viz.fit(X)
    s_viz.finalize()
    ax1.set_title('Silhouette Plot', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Silhouette Coefficient', fontsize=14)
    ax1.set_ylabel('Cluster', fontsize=14)
    ax1.tick_params(labelsize=8)
    ax1.grid(axis='x', linestyle=':', linewidth=0.5, color='lightgray', alpha=0.7)
    for spine in ax1.spines.values():
        spine.set_color('None')

    for i in range(n_clusters):
        ax2.scatter(
            X_tsne[y_kmeans == i, 0], X_tsne[y_kmeans == i, 1],
            s=s, color=palette[i], label=labels[i], alpha=0.85, edgecolor="w", linewidth=0.5
        )

    ax2.set_title('t-SNE Cluster Visualisation', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE 1', fontsize=14)
    ax2.set_ylabel('t-SNE 2', fontsize=14)
    ax2.grid(linestyle=':', linewidth=0.5, color='lightgray', alpha=0.7)
    ax2.legend(fontsize=14, frameon=False, loc='best')
    ax2.tick_params(labelsize=14)

    plt.suptitle('K-Means: Silhouette and t-SNE Plots', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ----------------------------------------------------------------------

#| label: fig-iris-silhouette-tsne
#| fig-cap: "Figure 15.4: Silhouette plot and t-SNE projection of the $k = 3$ k-means clusters on the Iris dataset, with cluster labels mapped to species names."
#| message: false
#| echo: false
#| eval: true

plot_silhouette_tsne(X, X_tsne, y_kmeans, kmeans, palette=palette, labels=['Virginica', 'Versicolor', 'Setosa'])

# ----------------------------------------------------------------------

#| label: iris-cluster-assignment
#| message: false
#| echo: true
#| eval: true

print(df.groupby(['species', 'cluster']).size().unstack(fill_value=0))

from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

confusion_matrix = pd.crosstab(df['species'], df['cluster'])
print("\nConfusion Matrix:")
print(confusion_matrix)

cost_matrix = confusion_matrix.max().max() - confusion_matrix
row_indices, col_indices = linear_sum_assignment(cost_matrix)

total_correct = confusion_matrix.values[row_indices, col_indices].sum()
accuracy = total_correct / len(df)
print(f"\nClustering Accuracy: {accuracy:.3f}")