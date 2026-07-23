#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


print("Method 2")


# In[2]:


# how cell-cell interactions change between a subset of immune cells and trophoblast cells 
# as the trophoblast differentiate and invade the maternal uterus


# In[3]:


import os
import zipfile

import pandas as pd
import anndata

pd.set_option('display.max_columns', 100)

BASE_DIR = '/home/tanasa/Desktop/CCI_CellPhoneDB'


# In[4]:


# --- Input files ---
# cpdb_file_path      : (mandatory) CellphoneDB database zip file. Already downloaded earlier into cellphonedb-data/.
# meta_file_path      : (mandatory) tsv linking cell barcodes to cluster labels.
# counts_file_path    : (mandatory) normalized (log, not z-scaled) counts, h5ad recommended.
# microenvs_file_path : (optional) restricts interactions to cell types within the same microenvironment.
# All four (meta/counts/microenv/DEGs) ship inside CellphoneDB/notebooks/data_tutorial.zip.


# In[5]:


# CellPhone will randomly permute the cluster labels of all cells whitin each microenvironement (microenvs_file_path), 
# 1,000 times (default), to test whether the mean average receptor expression level in a cluster and the average ligand expression level 
# between the interacting clusters is higher than those of the rest cell pairs. 


# In[ ]:





# In[6]:


import os
import pandas as pd
import anndata

pd.set_option('display.max_columns', 100)

BASE_DIR = '/home/tanasa/Desktop/CCI_CellPhoneDB'

cpdb_file_path = os.path.join(BASE_DIR, 'cellphonedb-data', 'cellphonedb.zip')

notebooks_dir = os.path.join(BASE_DIR, 'CellphoneDB', 'notebooks')
data_dir = os.path.join(notebooks_dir, 'data')

meta_file_path = os.path.join(data_dir, 'metadata.tsv')
counts_file_path = os.path.join(data_dir, 'normalised_log_counts.h5ad')
microenvs_file_path = os.path.join(data_dir, 'microenvironment.tsv')

out_path = os.path.join(BASE_DIR, 'results', 'method2_noTF')
os.makedirs(out_path, exist_ok=True)


# In[7]:


adata = anndata.read_h5ad(counts_file_path)
adata.shape


# In[ ]:





# In[8]:


# Extract the tutorial data once (data_tutorial.zip -> notebooks/data/*)
if not os.path.exists(meta_file_path):
    with zipfile.ZipFile(tutorial_zip) as zf:
        zf.extractall(notebooks_dir)
    print(f'Extracted {tutorial_zip} -> {data_dir}')
else:
    print('Tutorial data already extracted.')


# In[9]:


assert os.path.exists(cpdb_file_path), f'Missing database zip: {cpdb_file_path}'
assert os.path.exists(meta_file_path), f'Missing meta file: {meta_file_path}'
assert os.path.exists(counts_file_path), f'Missing counts file: {counts_file_path}'
assert os.path.exists(microenvs_file_path), f'Missing microenvironment file: {microenvs_file_path}'


# In[10]:


# The metadata file is compossed of two columns:

# barcode_sample: this column indicates the barcode of each cell in the experiment.
# cell_type: this column denotes the cell label assigned.

metadata = pd.read_csv(meta_file_path, sep='\t')
metadata.head(3)


# In[11]:


adata = anndata.read_h5ad(counts_file_path)
adata.shape


# In[12]:


# Barcodes in metadata and counts must match (order-insensitive check)
sorted(adata.obs.index) == sorted(metadata['barcode_sample'])


# In[13]:


# 3) Micronevironments defines the cell types that belong to a a given microenvironment. 
# CellphoneDB will only calculate interactions between cells that belong to a given microenvironment

microenv = pd.read_csv(microenvs_file_path, sep='\t')
microenv.head(3)

microenv.groupby('microenvironment', group_keys=False)['cell_type'] \
    .apply(lambda x: list(x.value_counts().index))


# In[14]:


microenv.groupby('microenvironment', group_keys = False)['cell_type'].apply(lambda x : list(x.value_counts().index))


# In[ ]:





# In[15]:


print("""Run method 2 (statistical analysis""")
# The output of this method will be saved in output_path and also returned to the predefined variables.


# In[16]:


# CellphoneDB employs a geometric sketching procedure (Hie et al. 2019) to preserve the structure of the data without 
# losing information from lowly represented cells. For this tutorial, we have opted to manually downsample the count 
# matrix and the metadata file accordingly.


# In[17]:


from cellphonedb.src.core.methods import cpdb_statistical_analysis_method

cpdb_results = cpdb_statistical_analysis_method.call(
    cpdb_file_path = cpdb_file_path,
    meta_file_path = meta_file_path,
    counts_file_path = counts_file_path,
    counts_data = 'hgnc_symbol',
    active_tfs_file_path = None,
    microenvs_file_path = microenvs_file_path,
    score_interactions = True,
    iterations = 1000,
    threshold = 0.1,
    threads = 5,
    debug_seed = 42,
    result_precision = 3,
    pvalue = 0.05,
    subsampling = False,
    subsampling_log = False,
    subsampling_num_pc = 100,
    subsampling_num_cells = 1000,
    separator = '|',
    debug = False,
    output_path = out_path,
    output_suffix = None,
)


# In[18]:


list(cpdb_results.keys())


# In[19]:


# Description of output files

# Most output files share common columns:

# id_cp_interaction: Unique CellphoneDB identifier for each interaction stored in the database.
# interacting_pair: Name of the interacting pairs separated by “|”.
# partner A or B: Identifier for the first interacting partner (A) or the second (B). It could be: UniProt (prefix simple:) or complex (prefix complex:)
# gene A or B: Gene identifier for the first interacting partner (A) or the second (B). The identifier will depend on the input user list.
# secreted: True if one of the partners is secreted.
# Receptor A or B: True if the first interacting partner (A) or the second (B) is annotated as a receptor in our database.
# annotation_strategy: Curated if the interaction was annotated by the CellphoneDB developers. Otherwise, the name of the database where the interaction has been downloaded from.
# is_integrin: True if one of the partners is integrin.
# directionality: Indiicates the directionality of the interaction and the charactersitics of the interactors.
# classification: Pathway classification for the interacting partners.


# In[ ]:





# In[20]:


# The means table and significant_means, pvalues, interaction_scores)  —  has the structure :

# Rows = one row per interaction, with columns like id_cp_interaction, interacting_pair, 
# partner_a, partner_b, gene_a, gene_b, receptor_a, receptor_b, etc. — 
# this is where partner A vs. partner B for that interaction is identified.

# Columns after that = one column per ordered cell-type pair, named CellA|CellB (using the separator='|' you set), 
# e.g. PV MYH11|EVT_1.


# In[21]:


# — inspect results


# In[22]:


cpdb_results['pvalues'].head(2)


# In[23]:


cpdb_results['means'].head(2)


# In[24]:


cpdb_results['significant_means'].head(2)


# In[25]:


cpdb_results['interaction_scores'].head(2)

# scores: scores ranging from 0 to 100. The higher the score is, the more specific the interaction is expected to be.


# In[26]:


cpdb_results['deconvoluted'].head(2)


# In[27]:


cpdb_results['deconvoluted_percents'].head(2)


# In[28]:


# zoom -in example


# In[29]:


means = cpdb_results['means']
means[['interacting_pair', 'partner_a', 'partner_b', 'gene_a', 'gene_b', 'PV MYH11|EVT_1']]


# In[ ]:





# In[30]:


pvalues = cpdb_results['pvalues']
pvalues[['interacting_pair', 'partner_a', 'partner_b', 'gene_a', 'gene_b', 'PV MYH11|EVT_1']]


# In[ ]:





# In[31]:


iscores = cpdb_results['interaction_scores']
iscores[['interacting_pair', 'partner_a', 'partner_b', 'gene_a', 'gene_b', 'PV MYH11|EVT_1']]


# In[ ]:





# In[32]:


for key, df in cpdb_results.items():
    print(key, df.shape)


# In[33]:


# 2440 and 94/95 are just the row/column counts of those tables — here's what each dimension means:

# Rows: 2440
# This is the number of candidate ligand–receptor interactions from the CellphoneDB database 
# that survived filtering for your dataset — i.e., interactions where the relevant gene(s) 
# are expressed in at least threshold=0.1 (10%) of cells in at least one cell type in your data. 
# It's the same 2,440 rows across means, pvalues, significant_means, and interaction_scores because 
# they're all indexed by the same interaction list — 
# just reporting a different statistic per interaction/cell-pair.

# Columns: 94 (95 for significant_means)

# 13 metadata columns describing the interaction itself: id_cp_interaction, interacting_pair, partner_a, partner_b, 
# gene_a, gene_b, secreted, receptor_a, receptor_b, annotation_strategy, is_integrin, directionality, classification.
# 81 data columns, one per ordered cell-type pair (CellA|CellB) — these are the 9 cell types in your microenvironment file, 
# paired against each other in both directions including self-pairs (9 × 9 = 81).
# significant_means has one extra column (rank), giving 95 instead of 94 — a ranking of how consistently significant 
# that interaction is across cell pairs.


# In[34]:


# The deconvoluted file breaks a multi-subunit interaction back down into its individual gene/protein components, 
# showing expression per single cell type (not per cell-type pair).

# Context: many receptors and ligands in CellphoneDB aren't single proteins — 
# they're complexes made of multiple subunits (e.g., an integrin like integrin_a2b1_complex = ITGA2 + ITGB1). 

# The means/pvalues/significant_means tables report one combined value per interaction per cell-pair 
# (using the geometric mean of subunits for complexes), which is useful for summary stats but hides 
# which individual gene is actually expressed and how strongly.

# deconvoluted un-collapses that: it has one row per gene per interaction it participates in, with columns:

# gene_name, uniprot, protein_name — identifies the individual gene/protein.
# is_complex — whether this gene is a subunit of a complex (True) or a standalone partner (False).
# complex_name — which complex it belongs to, if any.
# id_cp_interaction — links it back to the interaction row in means/pvalues.
# one column per cell type (not per cell pair) — the mean expression of that gene in that cell type.
# So if you want to know "is the interaction CDH1_integrin_a2b1_complex driven by ITGA2 or ITGB1 in a given cell type, and how highly expressed is each subunit there," deconvoluted is where you look — it's the gene-level, single-cell-type view underlying the interaction-level, cell-pair-level summary tables.

# deconvoluted_percents is the same structure but reports percent of cells expressing that gene in each cell type, 
# rather than mean expression — useful for checking whether a subunit clears your threshold=0.1 filter 
# in a specific cell type.


# In[ ]:





# In[35]:


# https://ktplotspy.readthedocs.io/en/latest/notebooks/tutorial.html


# In[36]:


# search/filter results


# In[37]:


from cellphonedb.utils import search_utils

search_results = search_utils.search_analysis_results(
    query_cell_types_1 = ['EVT_1', 'EVT_2', 'GC', 'eEVT', 'iEVT'],
    query_cell_types_2 = ['PV MMP11', 'PV MYH11', 'PV STEAP4'],
    query_genes = ['TGFBR1'],
    query_interactions = ['CSF1_CSF1R'],
    significant_means = cpdb_results['significant_means'],
    deconvoluted = cpdb_results['deconvoluted'],
    interaction_scores = cpdb_results['interaction_scores'],
    query_minimum_score = 50,
    separator = '|',
    long_format = True,
    query_classifications = ['Signaling by Transforming growth factor'],
)

search_results.head()


# In[ ]:





# In[38]:


# plotting setup

import ktplotspy as kpy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


kpy.plot_cpdb_heatmap(
    pvals=cpdb_results['pvalues'],
    degs_analysis=False,
    figsize=(5, 5),
    title='Sum of significant interactions',
)


# In[40]:


# Interactions can also be plotted grouped by pathway.

from plotnine import theme, element_text


# In[41]:


from plotnine import theme, element_text, element_rect

p = kpy.plot_cpdb(
    adata=adata,
    cell_type1="PV MYH11|PV STEAP4|PV MMP11",        # fixed: was "PV MMPP11"
    cell_type2="EVT_1|EVT_2|GC|iEVT|eEVT|VCT_CCC",
    means=cpdb_results['means'],
    pvals=cpdb_results['pvalues'],
    celltype_key="cell_labels",
    genes=["TGFB2", "CSF1R"],
    figsize=(18, 6),                                  # was (10, 3) -> much more breathing room
    title="Interactions between\nPV and trophoblast",
    max_size=6,                                        # was 3 -> bigger, easier-to-read dots
    highlight_size=1.2,                                 # was 0.75 -> thicker significance ring
    degs_analysis=False,
    standard_scale=True,
    interaction_scores=cpdb_results['interaction_scores'],
    scale_alpha_by_interaction_scores=True,
)

p = p + theme(
    figure_size=(18, 6),
    axis_text_x=element_text(size=8, rotation=90, ha='center'),
    axis_text_y=element_text(size=11),
    plot_title=element_text(size=16, weight='bold'),
    legend_title=element_text(size=10),
    legend_text=element_text(size=8),
    panel_background=element_rect(fill='white'),
)

p

# p.save(
#    'results/method2_noTF/pv_trophoblast_dotplot.png',
#    dpi=300, width=18, height=6, limitsize=False
# )


# In[ ]:





# In[42]:


# dot plot faceted by classification


from plotnine import facet_wrap

p = kpy.plot_cpdb(
    adata=adata,
    cell_type1='PV MYH11',
    cell_type2='EVT_1|EVT_2|GC|iEVT|eEVT|VCT_CCC',
    means=cpdb_results['means'],
    pvals=cpdb_results['pvalues'],
    celltype_key='cell_labels',
    genes=['TGFB2', 'CSF1R', 'COL1A1'],
    figsize=(12, 8),
    title='Interactions between PV and trophoblast\ngrouped by classification',
    max_size=6,
    highlight_size=0.75,
    degs_analysis=False,
    standard_scale=True,
)
p + facet_wrap('~ classification', ncol=1)


# In[43]:


# p + facet_wrap("~ classification", ncol=1) — splits the plot into one stacked panel per pathway classification, 
# so instead of one dense x-axis you get several shorter panels grouped by biological pathway


# In[ ]:





# In[ ]:





# In[ ]:




