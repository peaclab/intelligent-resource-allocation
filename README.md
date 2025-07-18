# Job Grouping Based Intelligent Resource Prediction Framework

## Overview
===========
📖 This repository includes code for [Job Grouping Based Intelligent Resource Prediction Framework](https://www.bu.edu/peaclab/files/2025/06/JSSPP_2025_paper_19.pdf).

This project provides a framework for intelligent resource prediction in HPC environments. By leveraging historical job data and machine learning techniques, it aims to improve resource allocation efficiency and job scheduling performance.

- Predicts execution time, memory, and CPU requirements for batch jobs.
- Reduces both underpredictions and overpredictions compared to baseline methods.
- Designed for integration with existing HPC schedulers.
- Developed by the PEACLab at Boston University in collaboration with Sandia National Laboratories.

Maintainer & Developer: [Beste Oztop](https://github.com/beste-oztop)


## Abstract
===========
In High-Performance Computing (HPC) systems, users estimate the resources needed for job submissions based on their best knowledge. However, underestimating the required execution time, number of processors, or memory size can lead to early job terminations. Conversely, overestimating resource requests results in inefficiencies in job backfilling, wasted compute power, unused memory, and poor job scheduling, ultimately reducing overall system efficiency.

As we enter the exascale era, efficient resource utilization is more critical than ever. Existing schedulers lack mechanisms to predict the resource requirements of batch jobs. To address this challenge, we design a data-driven recommendation framework that leverages historical job information to predict three key parameters for batch jobs: execution time, maximum memory size, and maximum number of CPU cores required.

In contrast to existing machine learning (ML) based resource prediction methods, we introduce an online resource suggestion framework that considers both underestimates and overestimates in batch job resource provisioning. Our framework outperforms the baseline method with no grouping mechanism by achieving over 98% success in eliminating underpredictions and reducing the amount of overpredictions.

## Repository Structure
===========
The repository is organized as follows:

```
intelligent-resource-allocation/
├── src/                 # Source code for the resource prediction framework
│   ├── group.py         # Historical batch job data grouping logic
│   ├── load.py          # Loading and preprocessing job data
│   ├── rec.py           # Resource recommendation logic
│   ├── train.py         # Model training and evaluation
├── LICENSE              # License information
└── README.md            # Project documentation (this file)
```

- **src/**: Main implementation, including model training, prediction, and evaluation.
- **LICENSE**: Project license.
- **README.md**: Project overview and usage instructions.



## Citation
===========
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
===========
This code is part of the Job Grouping Based Intelligent Resource Prediction Framework developed by the PEACLab at Boston University in collaboration with Sandia National Laboratories. The authors of the paper are:
Beste Oztop (1), Benjamin Schwaller (2), Vitus J. Leung (2), Jim Brandt (2), Brian Kulis (1), Manuel Egele (1), and Ayse K. Coskun (1)

Affiliations: (1) Department of Electrical and Computer Engineering, Boston University (2) Sandia National Laboratories

This work has been partially funded by Sandia National Laboratories. Sandia National Laboratories is a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC., a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy’s National Nuclear Security Administration under Contract DENA0003525.


## License
This project is licensed under the BSD 3-Clause License - see the LICENSE file for details