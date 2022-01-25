# ML_Engine_for_UE-VBS_Selection
A Two Stage Machine Learning Approach for 5G Mobile Network Augmentation through Dynamic Selection and Activation of UE_VBSs.

## Note: Use Jupyter NbViewer to load the code if GitHub doesn't render
Please copy the links of the .ipynb files and paste it in [NbViewer](https://nbviewer.org/) if GitHub fails to load it in the browser.

## Helpful Documents and Notebooks
- Check out the [Data](./master_simulation_1000_2500.csv) used to train these models.
- Check out the two-stage machine learning model [Code!](./06-Cluster_Head_and_Sum_Rate_Calculation.ipynb)
- Check out the project presentation [Deck!](./Docs/PPT-Review.pdf)
- Check out the details of the project in this [Paper!](./Docs/PAPER-A_Two_Stage_Machine_Learning_Approach_for_5G_Mobile_Network_Augmentation_through_Dynamic_Selection_and_Activation_of_UE_VBSs.pdf)

## Abstract
The 5G cellular network is the new generation
of mobile networks that focus and identifies the current and
future needs of the heterogeneous devices in the context of
wireless access in a dense network scenario. This research builds
on the user equipment-based virtual base station (UE-VBS)
concept, which utilizes smartphones to provide base station
services to other UEs in the perimeter by capitalizing the
capabilities of the new UE in terms of massive connectivity and
the enhanced resources such as computing, and battery-power it
offers. However, in a dynamic network architecture like the 5G
network, the selection of a qualified UE to become a UE-VBS is
a challenging task with the introduction of newer technologies,
UE hardware configurations, and network infrastructures. In
this context, to automate the successful identification of a
potential UE to act as a VBS is a need, a machine learning (ML)
model can be employed. Hence, in this paper, we propose and
explore a two-stage machine learning approach to dynamically
choose the eligible UEs that will be activated on the fly as
UE-VBS to support data rate expansion and improve quality
of service (QoS) in locations where infrastructure is lacking,
and a more agile network operation is required. Furthermore,
the UEs are clustered based on their Euclidean distance using
MeanShift, Affinity Propagation, OPTICS, K-Means, Spectral
Clustering, Agglomerative Clustering and BIRCH unsupervised
ML approaches, and the devices are categorized based on their
eligibility to become a UE-VBS for the corresponding cluster
using Random Forest, AdaBoost, and Gradient Boosting supervised
ML classification approaches. Then, using a heuristic
algorithm, we determine the optimal cluster head (CH) by
exploiting the findings of our classification model. The proposed
framework is simulated for a nonuniform distribution of the
UEs in time and space and quantified using statistical analysis.
Our simulation results demonstrate that the proposed model
achieves an accuracy of around 95% using Random Forest
classifier and the K-Means clustering.

## Code Walkthrough
1. [Simulation and Data Generator](./01-Simulation_and_Dataset_Generation.ipynb)
2. [Comparison of Clustering Algorithms](./02-Comparison_of_Clustering_Algorithms.ipynb)
3. [Classification](./03-Classification.ipynb)
4. [SMOTE](04-SMOTE.ipynb)
5. [Validation and Testing](./05-Validation_and_Testing.ipynb)
6. [Pipelined Model, Heuristic Algorithm for Identification of Cluster Heads and Sum Rate Calculation](./06-Cluster_Head_and_Sum_Rate_Calculation.ipynb)

## Result
In this paper, we delved into the implementation of a twostage
machine learning approach for 5G mobile network
augmentation through dynamic selection and activation of
UE-VBSs. For our investigations, a dataset has been generated
and pre-processed by simulating the UEs geographical coordinates as a non-uniform distribution in the time and
space domain. By using the dataset generated, we trained
a two-stage machine learning model to dynamically select
UE-VBSs for each of the clusters around the primary BS.
Initially, the UEs distributed around the primary BS were
clustered into groups using the K-means as it is found to be
the most appropriate for clustering with a silhouette score of
0.46. Then, a classifier to determine the eligibility of the UEs
to become a UE-VBS for each cluster has been pipelined in
the second stage. Furthermore, the class imbalance towards
the majority class has been conquered using SMOTE before
classification. Of the considered classification algorithms, our
investigations based on the K-fold testing revealed that the
Random forest classifier attains the highest mean accuracy
score of 0.97. Finally, our proposed model has been subjected
to statistical testing and analysis using SHAP and LIME.
From the performance analysis carried out, we advocate that
our proposed model-aided dynamic selection of a UE-VBS
for each of the clusters around the primary BS is efficient as
it has resulted in a better achievable data rate than random
selection.

# Installation Requirements
## Environment
It is recommended to have a Linux or macOS development environment for convenience, although the code runs on Windows 10. <br>
Use Anaconda to manage your packages and Python 3 (version >= 3.6.0 recommended). <br>
It is also recommended to run the code on <strong>Jupyter Notebook</strong>.

## Dependencies
### With Anaconda, no need to install
- matplotlib
- scikit-learn
- numpy
- pandas
- seaborn
- scipy
### Others
Remember to use conda, not pip for installing these
- missingno
- imblearn
- shap
- lime
### Latex for documentation - Ubuntu
- texlive-full
