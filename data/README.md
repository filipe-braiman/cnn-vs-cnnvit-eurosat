# Dataset

This project uses the **EuroSAT dataset**, a satellite image dataset for land use and land cover classification derived from Sentinel-2 imagery.

Due to repository size constraints, the dataset is **not included in this repository**. It can be obtained from the following sources:

* Zenodo (official academic archive):
  [https://zenodo.org/record/7711810](https://zenodo.org/record/7711810)

* Kaggle (convenient for running experiments directly in Kaggle notebooks):
  [https://www.kaggle.com/datasets/apollo2506/eurosat-dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)

---

# Dataset Structure

The EuroSAT RGB dataset is organized into class-specific directories:

```
EuroSAT_RGB/
├── AnnualCrop
├── Forest
├── HerbaceousVegetation
├── Highway
├── Industrial
├── Pasture
├── PermanentCrop
├── Residential
├── River
└── SeaLake
```

Each directory contains **64×64 RGB satellite image tiles** representing a specific land-use class.

---

# Classes Used in This Project

This study focuses on **binary classification** between two agricultural land-use categories:

* **AnnualCrop**
* **PermanentCrop**

These classes were selected due to their **visual similarity**, making the task suitable for evaluating how different model architectures capture **fine-grained spatial patterns** in satellite imagery.