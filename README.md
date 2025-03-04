![python version](https://img.shields.io/badge/python-v.3.9-blue)
![license](https://img.shields.io/badge/license-MIT-orange)
[![author](https://img.shields.io/badge/teslim-homepage)](https://teslim404.com)
# separationML
Source code and trained models for the paper "*Physics-informed Data-driven control of Electrochemical Separation Processes*". 


## Set-up environment

Clone this repository and then use `setup.sh` to setup a virtual environment `separation` with the required dependencies in `requirements.txt`. 

```bash
chmod +x setup.sh
git clone https://github.com/EnthusiasticTeslim/separationML.git
cd separationML
sh env.sh
source binfo/bin/activate
```


## Training RL

The notebooks for training the RL-based controllers are contained [here](./notebooks). More information on RL environment is available [logic.md](./OptiDial/logic.md).


> [!IMPORTANT]  
> All modules in setting up the RL controller are available in [OptiDial](./OptiDial/).

## How-to-cite

Cite the paper using the following:

```
@article{doi,
  author = {Teslim Olayiwola, Kyle Terito, Jose Romagnoli},
  title = {Physics-informed Data-driven control of Electrochemical Separation Processes},
  journal = {n/a},
  year = {n/a},
  volume = {n/a},
  number = {n/a},
  doi = {https://doi.org/}
}
```

<!-- References -->
<h2 id="References">References</h2>

- Ortiz et al (2005). *Brackish Water Desalination by Electrodialysis: Batch Recirculation Operation Modeling*, [J. Membr. Sci. 2005, 252 (1), 65–75](10.1016/j.memsci.2004.11.021).

- Teslim et al (2024). *Synergizing Data-Driven and Knowledge-Based Hybrid Models for Ionic Separations*, [ACS EST Engg. 2024](https://pubs.acs.org/doi/10.1021/acsestengg.4c00405)