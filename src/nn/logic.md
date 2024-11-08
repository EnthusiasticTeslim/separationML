# Physics-Informed Neural Network (PINN)

This repository contains a Physics-Informed Neural Network (PINN) framework implemented in [`model.py`](model.py). The framework includes the following core components:

- **Neural Network**: A fully connected neural network (class `FCN`).
- **Preprocessing**: The `preprocess_input` method, which normalizes the input features.
- **Physics Integration**: The `physics` method to solve the ODE function for the studied system.
- **Data Augmentation**: The `augment` method, which generates new combinations of input features.

### Physics Simulation

The `physics` method integrates with a simulation model of the electrochemical system located in [`model.py`](../physics/model.py). This model predicts key outputs like salt removal efficiency (SRE) and energy consumption per kilogram (EC), given a set of input features. For each feature combination, the `physics` method iterates, solves the ODE, and returns the corresponding SRE and EC.

### Data Augmentation

A Latin Hypercube Sampling (LHS) approach is used to augment the dataset with additional, diverse combinations of model inputs, efficiently capturing a broader range of possible values. The steps include:

1. **Define Parameter Ranges**: Determine the minimum and maximum values for each parameter in the experimental data.
2. **Generate LHS Samples**: Use the `scipy.stats.qmc` module to generate LHS samples within these specified ranges.
3. **Return Augmented Data**: Combine original data with the new LHS samples to create a comprehensive dataset.

### Training

The training process for the model minimizes a composite loss function defined as:

    Total Loss = λ_physics * loss_physics + λ_data * loss_data

where:
- **loss_physics** and **loss_data** represent the loss contributions from physics-based and neural network-driven components, respectively.
- **λ_physics** and **λ_data** are regularization parameters for balancing the physics and neural network contributions.

To compute the loss:

1. **Data Loss**: Calculated as the mean squared error (MSE) between the experimental target values and the FCN output, based on features from the experimental dataset.
2. **Physics Loss**: Calculated as the MSE between the ODE-based physics model predictions and the FCN output, using the augmented dataset generated from the experimental feature space.

This framework combines data-driven learning with physics-based constraints, leveraging the power of PINNs to improve model accuracy and generalizability.
