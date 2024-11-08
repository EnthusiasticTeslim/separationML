# data handling
import numpy as np
from tqdm import tqdm
import yaml
import os
# deep learning
import torch
import torch.nn as nn
# plotting
import matplotlib.pyplot as plt
# special class
from nn import PINN
from physics import electrodialysis

import argparse

    
class Trainer(PINN):
    '''Physics-informed neural network'''
    def __init__(self,
                 estimator: callable, 
                 input_dim: int, 
                 hidden_dim: int, 
                 n_layers: int,
                 output_dim: int, 
                 params: dict,
                 model_dir: str,
                 device: str = 'cpu',
                 num_points: int= 10,
                 n_samples: int= 100):
        '''Initialize the PINN
        Args:
            int input_size: number of input features
            int hidden_size: number of hidden units
            int output_size: number of output features
        '''
        super(Trainer, self).__init__(
                                        estimator=estimator, params=params, 
                                        input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=output_dim, 
                                        model_dir=model_dir, 
                                        num_points=num_points, n_samples=n_samples)
        self.device = device

    '''Training the model'''
    def compile(self, 
                      inputs: tuple,
                      target: np.array,
                      lambda_pinn: float= 0.5,
                      lambda_data: float= 0.5,
                      epoch: int = 1000,
                      include_physics: bool = True,
                      lr: float = 0.001):
        
        # augment the data for physics loss to enhance the training
        phys_augmented = self.augment(inputs)
        # scale the features to (0, 1) and write the scaler to yaml file
        inputs = self.preprocess_input(inputs, first_time=True)
        phys_augmented_norm = self.preprocess_input(phys_augmented)
        # optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn_physics = nn.MSELoss()
        loss_fn_data = nn.MSELoss()
        # record for loss function
        recorder_epoch_physics = []
        recorder_epoch_data = []
        recorder_epoch_total = []
        # convert target to tensor
        y = torch.tensor(target, dtype=torch.float32).reshape(-1, 1)
        for epoch in tqdm(range(epoch)):
            optimizer.zero_grad()
            # data driven loss
            out_nn = self(inputs).to(self.device)
            loss_data = loss_fn_data(out_nn, y)
            # physics loss
            if include_physics:
                out_phy_augmented = self.physics(phys_augmented).reshape(-1, 1)
                out_nn_augmented = self(phys_augmented_norm) 
                loss_physics = loss_fn_physics(out_phy_augmented, out_nn_augmented)
            else:
                loss_physics = torch.tensor(0.0)
            # total loss
            loss = lambda_data*loss_data + lambda_pinn*loss_physics
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                tqdm.write(f'Epoch {epoch}: physics = {loss_physics.item():.3f}, data = {loss_data.item():.3f}, total = {loss.item():.3f}')
                recorder_epoch_physics.append(loss_physics.item())
                recorder_epoch_data.append(loss_data.item())
                recorder_epoch_total.append(loss.item())

        # save model
        if include_physics:
            model_type = 'actual_pinn'
        else:
            model_type = 'actual_fnn'

        torch.save(self.state_dict(), f'{self.model_dir}/{model_type}_model.pth')
        torch.save(optimizer.state_dict(), f'{self.model_dir}/{model_type}_opt.pth')

        return recorder_epoch_physics, recorder_epoch_data, recorder_epoch_total
    
    '''Testing the model'''
    def test(self, inputs: tuple, include_physics: bool = True):
        
        if include_physics:
            model_type = 'actual_pinn'
        else:
            model_type = 'actual_fnn'

        trained_model = f'{self.model_dir}/{model_type}_model.pth'

        if os.path.exists(trained_model):
            self.load_state_dict(torch.load(trained_model, weights_only=True))
        else:
            print('Model not found')

        with torch.no_grad():
            inputs = self.preprocess_input(inputs)
            predictions = self(inputs)

        return predictions


def main(args):
    # set the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # device
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'gpu':
        device = torch.device('cuda:0')

    # ******** Data ********
    # A. experiment
    N = np.array([30, 40, 50, 80, 100])
    V = np.array([0.1, 0.3, 0.5, 0.8, 1.5])
    T_tot = np.array([30, 60, 90, 120, 150])
    features = (N, V, T_tot)
    target = np.array([0.027, 0.276, 0.69, 0.97, 0.99])
    # B. physics model
    estimator = electrodialysis(print_info=False)
    # C. load the base parameters
    params = yaml.safe_load(open(f"{args.params}"))


    # ******** Set up the model ********
    number_inputs = len(features)
    model = Trainer(
                    estimator=estimator, params=params,
                    input_dim=number_inputs, hidden_dim=args.hidden_dim, output_dim=args.output_dim, n_layers=args.n_layers, 
                    model_dir=args.model_path, device=device, num_points=args.num_points, n_samples=args.n_samples
                    )
    

    # ******** Training the model ********
    if args.train:
        print('Training the model')
        epoch_physics, epoch_data, epoch_total = model.compile(
                    inputs=features, target=target, 
                    lambda_pinn=args.lambda_physics, 
                    lambda_data=args.lambda_data, 
                    include_physics=args.include_physics,
                    epoch=args.epochs)

        fig, axs = plt.subplots(figsize=(6, 4))
        time_steps = np.linspace(0, args.epochs, len(epoch_data))
        if args.include_physics:
            axs.plot(time_steps, epoch_physics, label='Physics')
            
        axs.plot(time_steps, epoch_data, label='Data')
        axs.plot(time_steps, epoch_total, label='Total')
        axs.set_xlabel('Epochs', fontsize=14)
        axs.set_ylabel('Loss', fontsize=14)
        axs.legend()
        plt.show()
        # save the figure
        if args.include_physics:
            figure_name = 'pinn'
        else:
            figure_name = 'fnn'
        fig.savefig(f'{args.images_path}/{figure_name}_loss.png')
    
    else:
        print('Model not trained. if testing works, it means the model is already trained')


    # ******** Testing the model ********
    if args.test:
        fig2, axs2 = plt.subplots(figsize=(6, 4))
        test_predictions = model.test(features, include_physics=args.include_physics)
        axs2.scatter(target, test_predictions.numpy())
        axs2.plot(target, target, 'r--')
        axs2.set_xlabel('Experimental', fontsize=14)
        axs2.set_ylabel('Predicted', fontsize=14)
        plt.show()
        # save the figure
        if args.include_physics:
            figure_name = 'pinn'
        else:
            figure_name = 'fnn'
        fig2.savefig(f'{args.images_path}/{figure_name}_predictions.png')
    else:
        print('Model not tested')

if __name__ == '__main__':
    args = argparse.ArgumentParser('Training the PINN model')
    args.add_argument('--params', type=str, default='./physics/params.yaml', help='path to the base parameters for the physics model')
    args.add_argument('--model_path', type=str, default='../models', help='path to save the model')
    args.add_argument('--images_path', type=str, default='../images', help='path to the data')
    args.add_argument('--device', type=str, default='cpu', help='device to run the model')
    args.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    args.add_argument('--lambda_physics', type=float, default=0.1, help='weight for physics loss')
    args.add_argument('--lambda_data', type=float, default=0.9, help='weight for data loss')
    args.add_argument('--hidden_dim', type=int, default=16, help='number of neuron in hidden units')
    args.add_argument('--n_layers', type=int, default=3, help='number of hidden layers')
    args.add_argument('--output_dim', type=int, default=1, help='number of output features')
    args.add_argument('--train', action='store_true', default=False, help='train the model')
    args.add_argument('--test', action='store_true', default=False, help='test the model')
    args.add_argument('--include_physics', action='store_true', default=False, help='include physics in the training')
    args.add_argument('--num_points', type=int, default=50, help='number to time points to generate in ODE')
    args.add_argument('--n_samples', type=int, default=100, help='number of LHS-based samples to generate')
    args.add_argument('--seed', type=int, default=104, help='seed for reproducibility')

    args = args.parse_args()
    main(args)



# to train the model
# !python src/train_test_PINN.py --train

# to test the model
# !python src/train_test_PINN.py --test