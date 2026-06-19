# Robust Kernels for Robust Location Estimation

<p>
  <a href="https://par.nsf.gov/servlets/purl/10284588"><img src="http://img.shields.io/badge/Paper-PDF-brightgreen.svg"></a>
  <a href="https://github.com/Joaggi/Robust-kernels-for-robust-location-estimation/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  </a>
</p>


Robust kernels for robust location estimation
November 2020Neurocomputing 429(1) Follow journal
DOI: 10.1016/j.neucom.2020.10.090
Project: Robust Clustering Algorithms

This paper shows that least-square estimation (mean calculation) in a reproducing kernel Hilbert space (RKHS) F corresponds to different M-estimators in the original space depending on the kernel function associated with F. In particular, we present a proof of the correspondence of mean estimation in an RKHS for the Gaussian kernel with robust estimation in the original space performed with the Welsch M-estimator. This result is generalized to other types of M-estimators. This generalization facilitates the definition of new robust kernels associated to Huber, Tukey, Cauchy and Andrews M-estimators. The new kernels are empirically evaluated in different clustering tasks where state-of-the-art robust clustering methods are compared to kernel-based clustering using robust kernels. The results show that some robust kernels perform on a par with the best state-of-the-art robust clustering methods.


## 📂 Project Structure

The project is organized into specialized MATLAB packages (denoted by the `+` prefix) and supporting Python scripts for analysis.

```text
.
├── matlab/
│   ├── +salt_and_pepper/          # Experiments for salt-and-pepper noise scenarios
│   │   ├── jaffe/                 # Experiments specifically using the Jaffe dataset
│   │   ├── experiments/           # Directory containing algorithm-specific experiments:
│   │   │   ├── KMeans/            # K-Means algorithm benchmarks
│   │   │   ├── NMF/               # Non-negative Matrix Factorization benchmarks
│   │   │   ├── KernelKMeans/      # Kernelized K-Means benchmarks
│   │   │   ├── Kernelconvexnmf/   # Kernel Convex NMF benchmarks
│   │   │   ├── Kernelseminmfrule/ # Semi-NMF with rule-based updates
│   │   │   ├── Kernelseminmfnnls/ # Semi-NMF with NNLS updates
│   │   │   ├── RMNMF/             # Robust NMF benchmarks
│   │   │   └── NNMF/              # Non-negative matrix factorization benchmarks
│   │   └── Contamination.m        # Script to simulate salt-and-pepper noise
│   ├── +robust_andrews_kernel/    # Experiments implementing Andrews kernels
│   │   ├── att/                   # Experiments on the ATT dataset
│   │   ├── digits/                # Experiments on the Digits dataset
│   │   ├── iris/                  # Experiments on the Iris dataset
│   │   └── wineq/                 # Experiments on the Wine Quality dataset
│   ├── +theano_proofs/            # Scripts related to Theano-based proofs
│   ├── +mercel_kernel/            # Scripts related to Mercel kernel proofs
│   ├── +papers/                   # Reference paper PDFs for the project
│   └── ...                   
├── python/
│   ├── mercel_kernel/            # Scripts related to Mercel kernel proofs
│   ├── papers/                    # Additional research papers in PDF format
│   └── ...
├── dataset/

```

## 🚀 Core Algorithms

The repository implements several variations of clustering and matrix factorization techniques:

### Clustering Algorithms
- **K-Means**: Standard K-Means implementation.
- **Kernel K-Means**: K-Means extended with Radial Basis Function (RBF) and robust kernels.

### Matrix Factorization (NMF) Variations
- **NMF (Non-negative Matrix Factorization)**: Standard Nif implementation.
- **Kernel Convex NMF**: NMF where the non-negativity constraint is replaced by a convex constraint in a kernel space.
- **Semi-NMF (Semi-Nonnegative Matrix Factorization)**: 
    - **Rule-based Semi-NMF**: Utilizing specific update rules for robustness.
    - **NNLS-based Semi-NMF**: Using Non-Negative Least Squares for the semi-nonnegative constraint.
- **Robust NMF (RMNMF)**: Specifically designed to handle outliers in the data.
- **NNMF**: Non-negative matrix factorization applied to specific data structures.

## 📊 Datasets Used

The algorithms are evaluated on a variety of benchmark datasets in MATLAB format:

 1 ./abalone.mat                                                                                                                                   
 2 ./abalone_uniform_contamination.mat                                                                                                             
 3 ./ar.mat                                                                                                                                        
 4 ./att.mat                                                                                                                                       
 5 ./att_occlussion_contamination.mat                                                                                                              
 6 ./att_uniform_contamination.mat                                                                                                                 
 7 ./balance_scale.mat                                                                                                                             
 8 ./digits.mat                                                                                                                                    
 9 ./glass.mat                                                                                                                                     
10 ./__init__.py                                                                                                                                   
11 ./ionosphere.mat                                                                                                                                
12 ./iris.mat                                                                                                                                      
13 ./iris_uniform_contamination.mat                                                                                                                
14 ./jaffe.mat                                                                                                                                     
15 ./jaffe_occlusion_contamination.mat                                                                                                             
16 ./jaffe_uniform_contamination.mat                                                                                                               
17 ./movement_libras.mat                                                                                                                           
18 ./movement_libras_uniform_contamination.mat 

## 🛠️ Implementation Details

### MATLAB (Primary Engine)
The main experimental framework is implemented in MATLAB. The codebase is organized into packages (using the `+` convention) to separate different algorithmic families.

### Python (Analysis & Plotting)
Python scripts are used for post-processing, calculating metrics, and generating publication-quality plots:
- **Metrics calculation**: Computing Purity, NMI (Normalized Mutual Information), and Clustering Accuracy.
- **Visualization**: Generating bar plots and error bars for experimental results.

## 🧪 Running Experiments

Experiments are organized as MATLAB functions that load datasets, apply noise (if applicable), run the algorithm, and save the resulting metrics.

To run a specific experiment (e.g., Kernel K-Means on salt-and-pepper data):
1. Ensure you have the required MATLAB toolboxes (e.g., NMF toolbox) installed or present in the path.
2. Navigate to the relevant experiment folder.
3. Run the `.m` experiment script (e.g., `kernelkmeans_experiment.m`).

**Note**: Many scripts rely on specific path configurations (e.g., `G:/Dropbox/...`). You may need to update the `addpath` commands in the `.m` files to match your local environment.

## 📈 Results & Metrics

The performance of the algorithms is evaluated using:
- **Clustering Accuracy**
- **Purity**
- **Normalized Mutual Information (NMI)**

Plots generated include accuracy comparisons across different kernel parameters and sensitivity analysis under noise.

## 📖 Cite as 


```
@article{GALLEGO2021174,
title = {Robust kernels for robust location estimation},
journal = {Neurocomputing},
volume = {429},
pages = {174-186},
year = {2021},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2020.10.090},
url = {https://www.sciencedirect.com/science/article/pii/S0925231220317033},
author = {Joseph A. Gallego and Fabio A. González and Olfa Nasraoui},
keywords = {Robust statistics, M-estimators, Kernel methods, Kernel clustering, Kernel matrix factorization},
abstract = {This paper shows that least-square estimation (mean calculation) in a reproducing kernel Hilbert space (RKHS) F corresponds to different M-estimators in the original space depending on the kernel function associated with F. In particular, we present a proof of the correspondence of mean estimation in an RKHS for the Gaussian kernel with robust estimation in the original space performed with the Welsch M-estimator. This result is generalized to other types of M-estimators. This generalization facilitates the definition of new robust kernels associated to Huber, Tukey, Cauchy and Andrews M-estimators. The new kernels are empirically evaluated in different clustering tasks where state-of-the-art robust clustering methods are compared to kernel-based clustering using robust kernels. The results show that some robust kernels perform on a par with the best state-of-the-art robust clustering methods.}
}
```



