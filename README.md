![python version](https://img.shields.io/badge/python-v.3.9-blue)
![license](https://img.shields.io/badge/license-MIT-orange)
[![author](https://img.shields.io/badge/teslim-homepage)](https://teslim404.com)
# SeparationML
Source code and trained models for the paper "*Leveraging Physics-Informed Neural Networks and Reinforcement Learning for Predictive Modeling and Control in Electrochemical Separation Processes*". 


<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Set-up environment"> ➤ Set-up environment</a></li>
    <li>
          <a href="#Models"> ➤ Models</a>
          <ul>
            <li><a href="#Predictive">Physics Informed Neural Network</a></li>
            <li><a href="#Controller">Reinforcement Learning</a></li>
          </ul>
    </li>
    <li><a href="#How-to-cite"> ➤ How to cite</a></li>
    <li><a href="#License"> ➤ License</a></li>
    <li><a href="#References"> ➤ References</a></li>
  </ol>
</details>


<!-- Set-up environment -->
<h2 id="Set-up environment">Set-up environment</h2>

Clone this repository and then use `setup.sh` to setup a virtual environment `separationML` with the required dependencies in `requirements.txt`. 

```bash
chmod +x setup.sh
git clone https://github.com/EnthusiasticTeslim/separationML.git
cd separationML
sh env.sh
source binfo/bin/activate
```


<!-- Training Models-->
<h2 id="Models"> Models </h2>

> [!IMPORTANT]  
> All scripts for training models are available in Docker mode in folder `docker`.

<h3 id="Predictive"> Physics Informed Neural Network </h3>

Use the training script as follows:
```bash
usage: 
python train_test_PINN.py --help 

arguments:
  --params PARAMS       path to the base parameters for the physics model
  --model_path          path to save the model
  --images_path         path to the data
  --device DEVICE       device to run the model
  --epochs EPOCHS       number of epochs
  --lambda_physics      weight for physics loss
  --lambda_data         weight for data loss
  --hidden_dim          number of neuron in hidden units
  --n_layers            number of hidden layers
  --output_dim          number of output features
  --train               train the model
  --test                test the model
  --include_physics     include physics in the training 
  --num_points          number to time points to generate in ODE
  --n_samples           number of LHS-based samples to generate
  --seed SEED           seed for reproducibility
```
The model and its plott will be saved in `model_path` and `images`, respectively. For example, 
- Train a PINN model, you can use:

  ```bash
  python train_test_PINN.py --train --test --include_physics
  ```
- Train a basic NN model, you can use:

  ```bash
  python train_test_PINN.py --train --test
  ```

More information on PINN is available [here](./src/nn/logic.md).

<h3 id="Controller"> Reinforcement Learning</h3>

```
xxxxxx

```
More information on PINN is available [here](./src/physics/logic.md).

<!-- How-to-cite-->
<h2 id="How-to-cite">How to cite</h2>

```
@article{doi,
  author = {Teslim Olayiwola, Kyle Terito, Jose Romagnoli},
  title = {Leveraging Physics-Informed Neural Networks and Reinforcement Learning for Predictive Modeling and Control in Electrochemical Separation Processes},
  journal = {n/a},
  year = {n/a},
  volume = {n/a},
  number = {n/a},
  doi = {https://doi.org/},
  preprint = {Manuscript in Preparation}
}
```

<!-- References -->
<h2 id="References">References</h2>

- Ortiz et al (2005). *Brackish Water Desalination by Electrodialysis: Batch Recirculation Operation Modeling*, [J. Membr. Sci. 2005, 252 (1), 65–75](10.1016/j.memsci.2004.11.021).

- Karniadakis et al (2021). *Physics-Informed Machine Learning*, [Nat Rev Phys 2021, 3 (6), 422–440](https://doi.org/10.1038/s42254-021-00314-5). 

- Raissi et al (2019). *Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations*, [Journal of Computational Physics 2019, 378, 686–707](https://doi.org/10.1016/j.jcp.2018.10.045).  

- Teslim et al (2024). *Synergizing Data-Driven and Knowledge-Based Hybrid Models for Ionic Separations*, [ACS EST Engg. 2024](https://pubs.acs.org/doi/10.1021/acsestengg.4c00405)