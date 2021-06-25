# Gaussian variational approximation with a factor covariance structure

This MATLAB software implements VAFC approach (Variational approximation
with factor covariance) for the three high-dimensional dataset 
(Colon, Leukemia, Cancer).

The an extended version of the code is taken from the supplementar material of


V. M.-H. Ong, D. J. Nott, and M. S. Smith (2018)
@article{Ong.2018,
  title={Gaussian variational approximation with a factor covariance structure},
  author={Ong, Victor M-H and Nott, David J and Smith, Michael S},
  journal={Journal of Computational and Graphical Statistics},
  volume={27},
  number={3},
  pages={465--478},
  year={2018},
  publisher={Taylor \& Francis}
} 

For further details please see the README.txt file in the supplementary material to this paper.

# How to run VAFC for three high-dim dataset (Colon, Leukemia, Cancer)

* Main program : The main files are Factor_highdim_NK.m. This will produce
the necessary *.mat file for running the following scripts below.
* Run Output/CV_error.m: Produce the missclassification rates of Ong et al (2018) for p=4,20 and save temporal outputs to compute AUROC's in R using the next file.
* Run Output/Comp_AUC.R: Produce and print results in columns 3,4 of Table 3

# Acknowledgement

We would like to thank the authors  Ong et al (2018) for providing their Matlab code.

