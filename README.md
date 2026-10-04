# facebook_page-page_dataset

This webgraph is a page-page graph of verified Facebook sites. Nodes represent official Facebook pages while the links are mutual likes between sites. Node features are extracted from the site descriptions that the page owners created to summarize the purpose of the site. This graph was collected through the Facebook Graph API in November 2017 and restricted to pages from 4 categories which are defined by Facebook. These categories are: politicians, governmental organizations, television shows and companies. The task related to this dataset is multi-class node classification for the 4 site categories.

This repository provides a pipeline to convert the raw **Facebook Large Page-Page Network dataset** (MUSAE) from the **[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/527/facebook+large+page+page+network)** into a consolidated, GNN-friendly compressed NumPy matrix (`facebook.npz`) matching the format distributed by GraphMining.ai.


## Dataset Overview
The source data represents a social webgraph of verified Facebook pages:
* **Nodes (22,470):** Official Facebook pages categorized into 4 types (`tvshow`, `government`, `company`, `politician`).
* **Edges (171,002 raw / 342,004 processed):** Mutual likes between pages.
* **Features (4,714 raw tokens):** Word tags extracted from descriptions.

## NPZ Dataset Overview

The generated `facebook.npz` file contains a compressed archive of aligned NumPy arrays representing the complete graph topology, node properties, and categorical targets. 

### Internal Key Structure

When you unpack the `.npz` file, it exposes five primary components:

| Key Name | Array Shape | Data Type | Graph Element / Description |
| :--- | :--- | :--- | :--- |
| `X` | `(22470, 128)` | `float32` | **Node Feature Matrix:** SVD-compressed 128-dimensional dense representations derived from the original 4,714 description token IDs. |
| `edges_x` | `(342004,)` | `int64` | **Source Nodes:** The initiating node IDs (`id_1`) for the explicit graph connection pairs. |
| `edges_y` | `(342004,)` | `int64` | **Target Nodes:** The terminating node IDs (`id_2`) for the explicit graph connection pairs. |
| `y` | `(22470,)` | `int8` / `int64` | **Target Labels:** Integer-encoded category index representing the page function (`0`: tvshow, `1`: government, `2`: company, `3`: politician). |
| `page_name` | `(22470,)` | `object` (str) | **Metadata Strings:** Human-readable text strings recording the verified name of each Facebook page. |

### Graph Topology & Connectivity

* **Nodes Structure:** The dataset features exactly **22,470 uniquely indexed nodes** (ranging continuously from `0` to `22469`). The ordering is strictly aligned across the rows of `X`, `y`, and `page_name`.
* **Directed Framework Compliance:** The original graph contains **171,002 undirected relationships** (mutual page likes). In this `.npz` file, these relationships are explicitly doubled into **342,004 directed connections** (`edges_x` \(\rightarrow\) `edges_y` and `edges_y` \(\rightarrow\) `edges_x`). This allows for zero-overhead compatibility with deep learning matrix frameworks.
* **Loading Warning:** Because the `page_name` key contains raw Python strings rather than uniform scalars, you **must** pass the `allow_pickle=True` flag when parsing this file:
  ```python
  graph_data = np.load("facebook.npz", allow_pickle=True)
  ```

## 🔗 PyTorch Geometric Integration

Because this `.npz` file uses the identical array naming convention and dimensions as the GraphMining.ai backend, you can seamlessly integrate it with the PyTorch Geometric (`PyG`) ecosystem in two ways.

### Option 1: Drop-In Offline Replacement (Bypassing Downloads)
The `torch_geometric.datasets.FacebookPagePage` loader automatically downloads and looks for an internal `facebook.npz` asset in its raw directory. If you are working in an environment without internet access or wish to use your custom-built file (e.g., containing your specific SVD random state seed), you can drop it directly into the expected cache folder structure:

```python
import os
import shutil
from torch_geometric.datasets import FacebookPagePage

# Define your target PyG root cache directory
dataset_root = "./data/FacebookPagePage"
raw_dir = os.path.join(dataset_root, "raw")
os.makedirs(raw_dir, exist_ok=True)

# Copy your newly generated facebook.npz into the raw folder
shutil.copy("facebook.npz", os.path.join(raw_dir, "facebook.npz"))

# Initialize the dataset loader; PyG will detect the local file and skip downloading
dataset = FacebookPagePage(root=dataset_root)
data = dataset[0]

print(data) 
# Output: Data(x=[22470, 128], edge_index=[2, 342004], y=[22470])
```

### Option 2: Instantiating a Native PyG `Data` Object Manually
If you want to maintain access to your custom `page_name` string array directly alongside your graph tensors (which standard `FacebookPagePage` strips away), you can construct a native PyG `Data` object completely manually from the `.npz` file:

```python
import numpy as np
import torch
from torch_geometric.data import Data

# 1. Load your customized npz archive safely
npz_data = np.load("facebook.npz", allow_pickle=True)

# 2. Extract and format edge components into an aligned 2xN edge index
edge_index = torch.tensor(
    np.vstack([npz_data['edges_x'], npz_data['edges_y']]), 
    dtype=torch.long
)

# 3. Build the primary PyG Data tensor block
pyg_graph = Data(
    x=torch.tensor(npz_data['X'], dtype=torch.float32),
    edge_index=edge_index,
    y=torch.tensor(npz_data['y'], dtype=torch.long)
)

# 4. Attach your custom metadata strings seamlessly
pyg_graph.page_name = npz_data['page_name']

print(pyg_graph)
# Output: Data(x=[22470, 128], edge_index=[2, 342004], y=[22470])
print(f"Node 50 text metadata: {pyg_graph.page_name[50]}")
```


---

## Transformation Pipeline
The included notebook automates the following pipeline steps:
1. **Source:** Uses the downloaded and extracted files from facebook_large folder (Source :  **[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/527/facebook+large+page+page+network)** Dataset ID: `527`).
2. **Feature Compression:** Takes the variable-length dictionary of 4,714 unique token IDs, constructs a sparse bag-of-words binary matrix, and applies **TruncatedSVD/PCA** down to a uniform **128-dimensional dense feature matrix**.
3. **Graph Directedness:** Duplicates and reverses the 171,002 undirected edge coordinates to format a dual-directional `edge_index` (`342,004` entries) compliant with deep learning frameworks.
4. **Textual Preservation:** Maps and aligns raw string `page_name` arrays into the archive matrix using serialized object pickling.

---

## Getting Started - if you need to recreate the .npz

Run via Google Colab or locally.


---

## Output Structure Verification
The pipeline generates a single `facebook.npz` binary file. To load it back safely into your training script with string array tracking active, toggle `allow_pickle=True`:

```python
import numpy as np

# Load the output file
data = np.load("facebook.npz", allow_pickle=True)

print(data['X'].shape)          # Node Features matrix -> (22470, 128)
print(data['edges_x'].shape)    # Source Edge Pointers -> (342004,)
print(data['edges_y'].shape)    # Target Edge Pointers -> (342004,)
print(data['y'].shape)          # Encoded Target Class Labels (0-3) -> (22470,)
print(data['page_name'][1])     # Original string page names -> U.S. Consulate General Mumbai

```

##  Citations & License
This dataset is licensed under a **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. If you use this data in academic work, please cite the repository curators:

```text
@misc{rozemberczki2019multiscale,
      title={Multi-scale Attributed Node Embedding},
      author={Benedek Rozemberczki and Carl Allen and Rik Sarkar},
      year={2019},
      eprint={1909.13021},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
**[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/527/facebook+large+page+page+network)**

