<div align="center">


<h1>Dynamic and Rapid Deep Synthesis of Molecular MRI Signals
</h1>

<img src = "figures/architecture.png" width = "700">  

<div>
    <br>
    <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FDinorNagar%2FMolecular-MRI-Generator&label=visitors&countColor=%23263759&style=flat">
    <h4 align="center">
        <a href="https://arxiv.org/abs/2305.19413" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2305.19413-b31b1b.svg">
        </a> 
        <img src="https://api.infinitescript.com/badgen/badge?ltext=language&rtext=Python&color=fc9003")>
    </h4>
    
</div>
【<a href='https://github.com/DinorNagar' target='_blank'>Dinor Nagar</a> |
<a href='https://github.com/operlman' target='_blank'>Or Perlman</a> |
<a href='https://github.com/vnikale' target='_blank'>Nikita Vladimirov</a>】
<div>
<a href='https://mri-ai.github.io/' target='_blank'>Momentum Lab, Tel Aviv University</a>
</div>
</div>


## 📚 Overview

<strong>Molecular-MRI-generator </strong> is an open-source deep learning framework that was developed to accelerate
the molecular MRI pipeline compared to the state of the art mathematical models, such as Bloch-McConnell equations.
In this process we predict a set of signals corresponding to a specific acquisition protocol which was created using
a set of tissue and scanner parameters.

We developed two unique architectures to address the problem:
- <strong>Dynamic network</strong> that was trained data of 9 different acquisition protocol in a way that each prediction calculates
the next signal element in the trajectory given the previous element. The acquisition protocols doesn't necessary have the same signal length.
- <strong>Application optimized network</strong> that was designed for a case where a research group is interested in
investigation a specific acquisition protocol with predefined signal length, while further accelerating the prediction
time for each signal.

This repository hosts the code for our suggested models described in our published article. 


## ⚡ Getting Started

### Organizing the data
Before setting up the framework,we first need to create the data for training or evaluating the model. We used
Bloch-McConnell simulator that can be found <a href='https://mri-ai.github.io/' target='_blank'>here</a>. This simulator
was implemented in MATLAB and the data is stored in dictionaries saved as .mat files. To arrange the dataset
efficiently, for every simulated acquisition protocol we saved multiple dictionaries that describes multiple scenarios
of different values of input tissue and scanner parameters. Each dictionary was named by the values of the scanner parameter.
Furthermore, for every dictionary we created a text file with the same that contains the values of the parameter B1.

An example to such dataset for a specific protocol is described below:

```
L-arginine
   |- dict_tp_1_Trec_1_B0_3_angle_60.mat
   |- dict_tp_1_Trec_1_B0_3_angle_60.txt
   |- dict_tp_1_Trec_1_B0_11.7_angle_90.mat
   |- dict_tp_1_Trec_1_B0_11.7_angle_90.txt
   ...
```

For protocols that have also changing values for the parameter offset_ppm, we also included a text file in the same
manner with an addition of 'ppm' to the name of the file.
An example to such case is described below:

```
MT
   |- dict_tp_1_Trec_1_B0_3_angle_60.mat
   |- dict_tp_1_Trec_1_B0_3_angle_60.txt
   |- dict_tp_1_Trec_1_B0_3_angle_60ppm.txt
   |- dict_tp_8_Trec_8_B0_11.7_angle_90.mat
   |- dict_tp_8_Trec_8_B0_11.7_angle_90.txt
   |- dict_tp_8_Trec_8_B0_11.7_angle_90ppm.txt
   ...
```

### Setting up the environment
1. Clone Repo

```bash
git clone https://github.com/sczhou/ProPainter.git
```
For efficient arrangement, make sure that the folder tree which contains both the scripts and the datasets,
looks like the following tree:

```
dataset-dynamic
   |- example-protocol-1
   |- example-protocol-2
   ...
dataset-application-optimized
   |- example-protocol-1
   |- example-protocol-2
   ...   
  
train-script-1
train-script-2
...
```

2. Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```




## 🏋️ Training The Model
The training process is divided into two steps where each step relates to the specific network.

### Dynamic Network
For the dynamic network, run the following script:
```bash
python train_dynamic_network.py
```

### Application Optimized Network
For the application optimized network, run the following script:
```bash
python train_application_optimized_network.py
```

## 🏄🏻 Prediction Example
For setting an example, we added the pretrained weights for both of the networks, in addition to an example dictionary for every dictionary.
Run the following commands to try it out:

```bash
# The first example (Dynamic network)
python predict_dynamic.py

# The second example (Application optimized network)
python predict_application_optimized.py
```

For each example, the results will appear in `stats` folder which is located on the path `example_scenarios\*` where `*` indicates one of the two network directory.<br />
After running the corresponding example script, the following files will be created:
* __trajectories.png__ - One predicted scenario trajectory and the corresponding ground truth. 
* __statistical_graph.png__ - Graph of the predicted elements of the signal compared to the ground truth elements.
* __predicted_dict.mat__ - The new predicted dictionary created by the model for  the specific scenario.
* __stats.txt__ - Text file which saves the calculated statistical coefficient results.


## 🚀 Contributing
We believe in sharing information between other research group and contribute data. 
Whether you have a question or a bug fix, please let us know.


## 📑 References
If you found our work useful for research or software development, 
we would highly appreciate your support citing our linked paper.   
``` # TO CHANGE
@misc{dinornagar2023molecular-mri-generator,
title={Dynamic and Rapid Deep Synthesis of Molecular MRI Signals},
author={Dinor Nagar, Or Perlman, Nikita Vladimirov},
year={2023},
eprint={2305.19413},
archivePrefix={arXiv},
primaryClass={Physices.Medical Physics}
}
```

