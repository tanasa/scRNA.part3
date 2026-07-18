#!/usr/bin/env python
# coding: utf-8

# In[71]:


import palantir
import scanpy as sc
import pandas as pd
import os

# Plotting
import matplotlib
import matplotlib.pyplot as plt

# warnings
import warnings
from numba.core.errors import NumbaDeprecationWarning

warnings.filterwarnings(action="ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings(
    action="ignore", module="scanpy", message="No data for colormapping"
)

# Inline plotting
get_ipython().run_line_magic('matplotlib', 'inline')


# In[72]:


# Load sample data

data_dir = os.path.expanduser("./")

download_url = "https://dp-lab-data-public.s3.amazonaws.com/palantir/marrow_sample_scseq_counts.h5ad"
file_path = os.path.join(data_dir, "marrow_sample_scseq_counts.h5ad")

ad = sc.read(file_path, backup_url=download_url)
ad


# In[73]:


sc.pp.normalize_per_cell(ad)

palantir.preprocess.log_transform(ad)

sc.pp.highly_variable_genes(ad, n_top_genes=1500, flavor="cell_ranger")


# In[74]:


# Note in the manuscript, we did not use highly variable genes but scanpy by default uses only highly variable genes

sc.pp.pca(ad)

ad


# In[75]:


str(ad)


# In[76]:


ad.obs.head()


# In[77]:


ad.var.head()


# In[78]:


ad.uns


# In[79]:


ad.uns['hvg']


# In[80]:


ad.uns['pca']['variance_ratio'][:10]    # fraction of variance per PC
ad.uns['pca']['params']   


# In[81]:


ad.obsm['X_pca']


# In[82]:


ad.varm['PCs']


# In[ ]:





# In[83]:


sc.pp.highly_variable_genes(ad, n_top_genes=1500, flavor="cell_ranger")

# PCA
# Note in the manuscript, we did not use highly variable genes but scanpy by default uses only highly variable genes

sc.pp.pca(ad)
ad


# In[84]:


# Diffusion maps

dm_res = palantir.utils.run_diffusion_maps(ad, n_components=5)

ms_data = palantir.utils.determine_multiscale_space(ad)


# In[85]:


# Visualization

sc.pp.neighbors(ad)
sc.tl.umap(ad)

# Use scanpy functions to visualize umaps or FDL

sc.pl.embedding(
    ad,
    basis="umap",
    frameon=False,
)


# In[ ]:





# In[86]:


# MAGIC imputation
# Palantir uses MAGIC to impute the data for visualization and determining gene expression trends.


# In[87]:


imputed_X = palantir.utils.run_magic_imputation(ad)


# In[88]:


sc.pl.embedding(
    ad,
    basis="umap",
    layer="MAGIC_imputed_data",
    color=["CD34", "MPO", "GATA1", "IRF8"],
    frameon=False,
)
plt.show()


# In[89]:


# Diffusion maps visualization

palantir.plot.plot_diffusion_components(ad)
plt.show()


# In[90]:


# Running Palantir
# Palantir can automatically determine the terminal states as well. 
# In this dataset, we know the terminal states and we will set them using the terminal_states parameter


# In[91]:


terminal_states = pd.Series(
    ["DC", "Mono", "Ery"],
    index=["Run5_131097901611291", "Run5_134936662236454", "Run4_200562869397916"],
)

palantir.plot.highlight_cells_on_umap(ad, terminal_states)
plt.show()


# In[92]:


start_cell = "Run5_164698952452459"

pr_res = palantir.core.run_palantir(
    ad, start_cell, num_waypoints=500, terminal_states=terminal_states
)

# Visualizing Palantir results
palantir.plot.plot_palantir_results(ad, s=3)
plt.show()


# In[93]:


# Pseudotime: Pseudo time ordering of each cell

# Terminal state probabilities: Matrix of cells X terminal states. 
# Each entry represents the probability of the corresponding cell reaching the respective terminal state

# Entropy: A quantiative measure of the differentiation potential of each cell computed as the entropy of 
# the multinomial terminal state probabilities


# In[94]:


cells = [
    "Run5_164698952452459",
    "Run5_170327461775790",
    "Run4_121896095574750",
]

palantir.plot.plot_terminal_state_probs(ad, cells)
plt.show()


# In[95]:


palantir.plot.highlight_cells_on_umap(ad, cells)
plt.show()


# In[96]:


print("Gene expression trends")


# In[97]:


print("""Gene expression trends over pseudotime provide insights into the dynamic behavior of genes during
cellular development or progression.""")

print("""Gene expression trends over pseudotime provide insights into the dynamic behavior of genes during
cellular development or progression. By examining these trends, we can uncover the timing of gene expression
changes and identify pivotal regulators of cellular states.
Palantir provides tools for computing these gene expression trends.""")


# In[98]:


print("Gene expression trends - select branch cells")


# In[99]:


print("""Before computing the gene expression trends, we first need to select cells associated 
with a specific branch of the pseudotime trajectory.

We accomplish this by using the select_branch_cells function. 

The parameters q and eps are used to control the selection's tolerance. 
Select small values >=0 to be more sringent and larger values <1 to select more cells.""")


# In[100]:


# Gene expression trends - select branch cells

masks = palantir.presults.select_branch_cells(ad, q=.01, eps=.01)

palantir.plot.plot_branch_selection(ad)
plt.show()


# In[101]:


# To visualize a trajectory on the UMAP, we interpolate the UMAP coordinates of cells specific to each branch across pseudotime, 
# enabling us to draw a continuous path.

palantir.plot.plot_trajectory(ad, "Ery")

palantir.plot.plot_trajectory(
    ad, # your anndata
    "DC", # the branch to plot
    cell_color="palantir_entropy", # the ad.obs colum to color the cells by
    n_arrows=10, # the number of arrow heads along the path
    color="red", # the color of the path and arrow heads
    scanpy_kwargs=dict(cmap="viridis"), # arguments passed to scanpy.pl.embedding
    arrowprops=dict(arrowstyle="->,head_length=.5,head_width=.5", lw=3), # appearance of the arrow heads
    lw=3, # thickness of the path
    pseudotime_interval=(0, .9), # interval of the pseudotime to cover with the path
)


# In[102]:


palantir.plot.plot_trajectories(ad, pseudotime_interval=(0, .9))

# When using cell_color="branch_selection" be aware of the overlap between branches:

palantir.plot.plot_trajectories(ad, cell_color="branch_selection", pseudotime_interval=(0, .9))
plt.show()


# In[103]:


# Gene expression trends

gene_trends = palantir.presults.compute_gene_trends(
    ad,
    expression_key="MAGIC_imputed_data",
)

genes = ["CD34", "MPO", "GATA1", "IRF8"]
palantir.plot.plot_gene_trends(ad, genes)
plt.show()


# In[104]:


# Alternatively, the trends can be visualized on a heatmap using

palantir.plot.plot_gene_trend_heatmaps(ad, genes)
plt.show()


# In[105]:


palantir.plot.plot_trend(ad, "Ery", "KLF1", color="n_counts", position_layer="MAGIC_imputed_data")
plt.show()


# In[ ]:





# In[106]:


print("Clustering")


# In[107]:


# Clustering

more_genes = ad.var_names[:1000]
communities = palantir.presults.cluster_gene_trends(ad, "Ery", more_genes)


# In[108]:


palantir.plot.plot_gene_trend_clusters(ad, "Ery")
plt.show()


# In[109]:


palantir.plot.plot_trajectory(
    ad, # your anndata
    "DC", # the branch to plot
    cell_color="palantir_entropy", # the ad.obs colum to color the cells by
    n_arrows=10, # the number of arrow heads along the path
    color="red", # the color of the path and arrow heads
    scanpy_kwargs=dict(cmap="viridis"), # arguments passed to scanpy.pl.embedding
    arrowprops=dict(arrowstyle="->,head_length=.5,head_width=.5", lw=3), # appearance of the arrow heads
    lw=3, # thickness of the path
    pseudotime_interval=(0, .9), # interval of the pseudotime to cover with the path
)


# In[110]:


palantir.plot.plot_trajectories(ad, pseudotime_interval=(0, .9))
# When using cell_color="branch_selection" be aware of the overlap between branches:
palantir.plot.plot_trajectories(ad, cell_color = "branch_selection", pseudotime_interval=(0, .9))
plt.show()


# In[ ]:


gene_trends = palantir.presults.compute_gene_trends(
    ad,
    expression_key="MAGIC_imputed_data",
)


# In[ ]:


genes = ["CD34", "MPO", "GATA1", "IRF8"]
palantir.plot.plot_gene_trends(ad, genes)
plt.show()


# In[ ]:


palantir.plot.plot_gene_trend_heatmaps(ad, genes)
plt.show()


# In[ ]:


palantir.plot.plot_trend(ad, "Ery", "KLF1", color="n_counts", position_layer="MAGIC_imputed_data")
plt.show()


# In[ ]:


more_genes = ad.var_names[:1000]
communities = palantir.presults.cluster_gene_trends(ad, "Ery", more_genes)


# In[ ]:


palantir.plot.plot_gene_trend_clusters(ad, "Ery")
plt.show()


# In[ ]:





# In[ ]:


# Save results

file_path = os.path.join(data_dir, "marrow_sample_scseq_processed.h5ad")
ad.write(file_path)


# In[ ]:




