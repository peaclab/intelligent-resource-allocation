# Job Grouping Based Intelligent Resource Prediction Framework

## Overview
📖 This repository includes code for [Job Grouping Based Intelligent Resource Prediction Framework](https://www.bu.edu/peaclab/files/2025/06/JSSPP_2025_paper_19.pdf).

This project provides a framework for intelligent resource prediction in HPC environments. By leveraging historical job data and machine learning techniques, it aims to improve resource allocation efficiency and job scheduling performance.

- Predicts execution time, memory, and CPU requirements for batch jobs.
- Reduces both underpredictions and overpredictions compared to baseline methods.
- Designed for integration with existing HPC schedulers.
- Developed by the PEACLab at Boston University in collaboration with Sandia National Laboratories.

Maintainer & Developer: [Beste Oztop](https://github.com/beste-oztop)


## Abstract
In High-Performance Computing (HPC) systems, users estimate the resources needed for job submissions based on their best knowledge. However, underestimating the required execution time, number of processors, or memory size can lead to early job terminations. Conversely, overestimating resource requests results in inefficiencies in job backfilling, wasted compute power, unused memory, and poor job scheduling, ultimately reducing overall system efficiency.

As we enter the exascale era, efficient resource utilization is more critical than ever. Existing schedulers lack mechanisms to predict the resource requirements of batch jobs. To address this challenge, we design a data-driven recommendation framework that leverages historical job information to predict three key parameters for batch jobs: execution time, maximum memory size, and maximum number of CPU cores required.

In contrast to existing machine learning (ML) based resource prediction methods, we introduce an online resource suggestion framework that considers both underestimates and overestimates in batch job resource provisioning. Our framework outperforms the baseline method with no grouping mechanism by achieving over 98% success in eliminating underpredictions and reducing the amount of overpredictions.

## Repository Structure
The repository is organized as follows:

```bash
intelligent-resource-allocation/
├── data/        
│   ├── fugaku/    
│   ├── m100/    
│   ├── nrel_eagle/ 
├── notebooks/         
│   ├── fugaku.ipynb     
│   ├── m100_cineca.ipynb 
│   ├── nrel_eagle.ipynb 
├── scripts/           
│   ├── fugaku_data_preprocessing.py 
│   ├── m100_data_preprocessing.py  
│   ├── nrel_eagle_data_preprocessing.py 
│   ├── ml_model_training.py      
│   ├── baseline_xgboost.py      
│   ├── kmeans_clustering.py  
├── LICENSE  
└── README.md   
```

- 💿 **data/**: Directory for publicly available datasets included in this work
    - **fugaku/**: Fugaku dataset from April 2024, source: [F-DATA](https://zenodo.org/records/11467483).
    - **m100/**: M100 dataset from April 2024, source: [M100 ExaData](https://www.nature.com/articles/s41597-023-02174-3#ref-CR17).
    - **nrel_eagle/**: Eagle dataset from Nov 2018 to Feb 2023, source: [NREL Eagle supercomputer jobs](https://data.openei.org/submissions/5860).
- 📒 **notebooks/**: Jupyter notebooks for data preprocessing and model training.
    - **fugaku.ipynb**: Jupyter notebook for Fugaku dataset analysis.
    - **m100_cineca.ipynb**: Jupyter notebook for M100 dataset analysis.
    - **nrel_eagle.ipynb**: Jupyter notebook for NREL Eagle dataset analysis.
- 🎬 **scripts/**: Python scripts for data preprocessing, model training, and evaluation.
    - **fugaku_data_preprocessing.py**: Data preprocessing for Fugaku dataset.
    - **m100_data_preprocessing.py**: Data preprocessing for M100 dataset.
    - **nrel_eagle_data_preprocessing.py**: Data preprocessing for Eagle dataset.
    - **ml_model_training.py**: Functions for training and evaluating machine learning models.
    - **baseline_xgboost.py**: Implementation of the baseline XGBoost model.
    - **kmeans_clustering.py**: Functions for KMeans clustering to group jobs.
- 🪪 **LICENSE**: Project license.
- 👓 **README.md**: Project overview and usage instructions.



## Citation
If you use this code in your research, please cite the following paper:
```bibtex
@inproceedings{oztop2025intelligent,
  title     = {Job Grouping Based Intelligent Resource Prediction Framework},
  author    = {Oztop, Beste and Schwaller, Benjamin and Leung, Vitus J. and Brandt, Jim and Kulis, Brian and Egele, Manuel and Coskun, Ayse K.},
  booktitle = {Workshop on Job Scheduling Strategies for Parallel Processing},
  year      = {2025},
  organization = {Springer}
}
```
Copy and paste the above BibTeX entry to cite our work.


## Authors
This code is part of the Job Grouping Based Intelligent Resource Prediction Framework developed by the PEACLab at Boston University in collaboration with Sandia National Laboratories. The authors of the paper are:
Beste Oztop (1), Benjamin Schwaller (2), Vitus J. Leung (2), Jim Brandt (2), Brian Kulis (1), Manuel Egele (1), and Ayse K. Coskun (1)

Affiliations: (1) Department of Electrical and Computer Engineering, Boston University (2) Sandia National Laboratories

This work has been partially funded by Sandia National Laboratories. Sandia National Laboratories is a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC., a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy’s National Nuclear Security Administration under Contract DENA0003525.


## License
This project is licensed under the BSD 3-Clause License - see the LICENSE file for details