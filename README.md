# TBM_advance_classification
Code and data repository for the paper **Challenges and Opportunities of Data Driven Advance Classification of Hard Rock TBMs** by Georg H. Erharter<sup>1</sup>, Paul Unterlaß<sup>2</sup>, Nedim Radončić<sup>3</sup>, Thomas Marcher<sup>2</sup>, Jamal Rostami<sup>4</sup>

1)	Norwegian Geotechnical Institute, Sandakerveien 140, Oslo, Norway
2)	Institute of Rock Mechanics and Tunnelling, Graz University of Technology, Rechbauerstraße 12, Graz, Austria
3)	iC Consulenten ZT GmbH, Schönbrunnerstraße 12, Vienna, Austria
4)	Colorado School of Mines, 1500 Illinois St, Golden, Colorado, United States of America
* correspondence: georg.erharter@ngi.no

The paper is currently in the review phase.

Code authors: Georg H. Erharter & Paul Unterlaß

## Synthetic TBM operational data
The synthetic Tunnel Boring Machine (TBM) operational data can be found in the folder "data". Datasets for 3 different TBMs were generated which are denoted TBM A, -B, -C. The data was synthezised using generative adverserial networks (GANs) based on real TBM operational data.

For each TBM 4 different files are given (replace X with A, B, C for the different TBMs):
- **TBM_X_0_synthetic_raw**: direct output of the GANs in vectors of length 4096. DO NOT USE FOR GEOTECHNICAL PURPOSES. Type: parquet file. See section 2.2.1 in paper.
- **TBM_X_1_synthetic_realistic.zip**: Synthetic TBM data after post processing. Type: zipped .csv file. See section 2.2.1 in paper.
- **TBM_X_2_synthetic_advance.xlsx**: Synthetic TBM operational data after basic data cleaning where standstills have been removed. Type: excel file. See section 3.1 in paper.
- **TBM_X_2_synthetic_strokes.xlsx**: Finally processed TBM operational data with stroke-wise aggregation. Type: excel file. See section 3.5 in paper.


## Requirements
The environment is set up using `conda`.

To do this create an environment called `TBM_data` using `environment.yaml` with the help of `conda`. If you get pip errors, install pip libraries manually, e.g. `pip install pandas`
```bash
conda env create --file environment.yaml
```

Activate the new environment with:

```bash
conda activate Jv
```

### contact
georg.erharter@ngi.no
