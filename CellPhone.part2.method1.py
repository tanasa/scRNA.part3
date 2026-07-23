#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import os
import zipfile

import pandas as pd
import anndata

pd.set_option('display.max_columns', 100)

BASE_DIR = '/home/tanasa/Desktop/CCI_CellPhoneDB'


# In[2]:


# --- Input files ---
# cpdb_file_path      : (mandatory) CellphoneDB database zip file. Already downloaded earlier into cellphonedb-data/.
# meta_file_path      : (mandatory) tsv linking cell barcodes to cluster labels.
# counts_file_path    : (mandatory) normalized (log, not z-scaled) counts, h5ad recommended.
# microenvs_file_path : (optional) restricts interactions to cell types within the same microenvironment.
# All four (meta/counts/microenv/DEGs) ship inside CellphoneDB/notebooks/data_tutorial.zip.


# In[ ]:





# In[3]:


cpdb_file_path = os.path.join(BASE_DIR, 'cellphonedb-data', 'cellphonedb.zip')

notebooks_dir = os.path.join(BASE_DIR, 'CellphoneDB', 'notebooks')
data_dir = os.path.join(notebooks_dir, 'data')
tutorial_zip = os.path.join(notebooks_dir, 'data_tutorial.zip')

meta_file_path = os.path.join(data_dir, 'metadata.tsv')
counts_file_path = os.path.join(data_dir, 'normalised_log_counts.h5ad')
microenvs_file_path = os.path.join(data_dir, 'microenvironment.tsv')

out_path = os.path.join(BASE_DIR, 'results', 'method1')
os.makedirs(out_path, exist_ok=True)


# In[4]:


# Extract the tutorial data once (data_tutorial.zip -> notebooks/data/*)
if not os.path.exists(meta_file_path):
    with zipfile.ZipFile(tutorial_zip) as zf:
        zf.extractall(notebooks_dir)
    print(f'Extracted {tutorial_zip} -> {data_dir}')
else:
    print('Tutorial data already extracted.')


# In[5]:


assert os.path.exists(cpdb_file_path), f'Missing database zip: {cpdb_file_path}'
assert os.path.exists(meta_file_path), f'Missing meta file: {meta_file_path}'
assert os.path.exists(counts_file_path), f'Missing counts file: {counts_file_path}'
assert os.path.exists(microenvs_file_path), f'Missing microenvironment file: {microenvs_file_path}'


# In[6]:


# The metadata file is compossed of two columns:

# barcode_sample: this column indicates the barcode of each cell in the experiment.
# cell_type: this column denotes the cell label assigned.

metadata = pd.read_csv(meta_file_path, sep='\t')
metadata.head(3)


# In[7]:


adata = anndata.read_h5ad(counts_file_path)
adata.shape


# In[8]:


# Barcodes in metadata and counts must match (order-insensitive check)
sorted(adata.obs.index) == sorted(metadata['barcode_sample'])


# In[9]:


# 3) Micronevironments defines the cell types that belong to a a given microenvironment. 
# CellphoneDB will only calculate interactions between cells that belong to a given microenvironment

microenv = pd.read_csv(microenvs_file_path, sep='\t')
microenv.head(3)

microenv.groupby('microenvironment', group_keys=False)['cell_type'] \
    .apply(lambda x: list(x.value_counts().index))


# In[ ]:





# In[10]:


print("""Run method 1 (basic analysis""")
# The output of this method will be saved in output_path and also returned to the predefined variables.


# In[ ]:





# In[11]:


# Run method 1 (basic analysis)

from cellphonedb.src.core.methods import cpdb_analysis_method

cpdb_results = cpdb_analysis_method.call(
    cpdb_file_path = cpdb_file_path,             # mandatory: CellphoneDB database zip file.
    meta_file_path = meta_file_path,             # mandatory: tsv file defining barcodes to cell label.
    counts_file_path = counts_file_path,         # mandatory: normalized count matrix (path or in-memory AnnData).
    counts_data = 'hgnc_symbol',                 # gene annotation used in the counts matrix.
    microenvs_file_path = microenvs_file_path,   # optional: restricts interactions to cells in the same microenvironment.
    score_interactions = True,                   # score interactions in addition to computing means.
    output_path = out_path,                      # where results are written.
    separator = '|',                             # separator used for "cellA|cellB" column labels.
    threads = 5,                                 # number of threads to use.
    threshold = 0.1,                             # min fraction of cells expressing a gene to be included.
    result_precision = 3,                        # rounding for mean values in significant_means.
    debug = False,                               # save intermediate tables (pkl) if True.
    output_suffix = None,                        # custom suffix instead of a timestamp in output filenames.
)



# In[12]:


print(cpdb_results.keys())


# In[13]:


# # Inspect results


# In[14]:


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


# In[15]:


# Means fields:

# means: Mean values for all the interacting partners: mean value refers to the total mean of 
# the individual partner average expression values in the corresponding interacting pairs of cell types. 

# If one of the mean values is 0, then the total mean is set to 0.

cpdb_results['means_result'].head(2)


# In[16]:


# Interaction scores fields:

# scores: Interaction scores ranging between 0 and 100. The higher the score is, the more specific the interaction is expected

cpdb_results['interaction_scores'].head(2)


# In[17]:


# Deconvoluted fields:

cpdb_results['deconvoluted'].head(2)

# Deconvoluted fields:

# gene_name: Gene identifier for one of the subunits that are participating in the interaction defined in “means.csv” file. The identifier will depend on the input of the user list.
# uniprot: UniProt identifier for one of the subunits that are participating in the interaction defined in “means.csv” file.
# is_complex: True if the subunit is part of a complex. Single if it is not, complex if it is.
# protein_name: Protein name for one of the subunits that are participating in the interaction defined in “means.csv” file.
# complex_name: Complex name if the subunit is part of a complex. Empty if not.
# id_cp_interaction: Unique CellphoneDB identifier for each of the interactions stored in the database.
# mean: Mean expression of the corresponding gene in each cluster.


# In[18]:


# Deconvoluted percents fields:
# percents: Percent of cells (clusters) expressing the given gene.

cpdb_results['deconvoluted_percents'].head(2)


# In[ ]:





# In[ ]:





# In[ ]:




