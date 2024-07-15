import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from torch import nn
from src.posterior_networks.PosteriorNetwork import PosteriorNetwork
from src.dataset_manager.CustomDataset import CustomDataset
from torch.utils.data import DataLoader
import torchvision.transforms as trn



torch.manual_seed(1)
np.random.seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
num_workers = 8 

if __name__=='__main__':
    name = 'alexnet_baseline'
    # Load model
    seed_model=123
    directory_model='./saved_models'
    architecture='alexnet'
    input_dims=[32, 32, 3]
    output_dim=10
    hidden_dims=[64, 64, 64]
    kernel_dim=5
    latent_dim=6
    no_density=True
    density_type='radial_flow'
    n_density=6
    k_lipschitz=None
    budget_function='id'
    N = torch.tensor([3543., 3644., 3570., 3593., 3633., 3539., 3647., 3577., 3693., 3561.])
    batch_size=256
    lr=5e-4
    loss='CE'
    training_mode='joint'
    regr=1e-5

    model = PosteriorNetwork(N=N,
                            input_dims=input_dims,
                            output_dim=output_dim,
                            hidden_dims=hidden_dims,
                            kernel_dim=kernel_dim,
                            latent_dim=latent_dim,
                            architecture=architecture,
                            k_lipschitz=k_lipschitz,
                            no_density=no_density,
                            density_type=density_type,
                            n_density=n_density,
                            budget_function=budget_function,
                            batch_size=batch_size,
                            lr=lr,
                            loss=loss,
                            regr=regr,
                            seed=seed_model)

    model_path = './saved_models/model-dpn-123-CIFAR10-[0.6, 0.8]-0.0-255.0-123-alexnet-[32, 32, 3]-10-[64, 64, 64]-5-6-True-radial_flow-6-None-id-200-20-2-256-0.0005-CE-joint-1e-05-8'
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)

    # Dataset
    data_root_dir='./data/CIFAR-10-C'
    corruptions = [
                    'gaussian_noise','motion_blur','glass_blur','saturate','snow',
                    'spatter','contrast','impulse_noise','speckle_noise','gaussian_blur',
                    'shot_noise','brightness','pixelate','elastic_transform','zoom_blur',
                    'jpeg_compression','fog','frost','defocus_blur'
    ]

    transform = trn.Compose([trn.ToTensor(), trn.Normalize(0, 255)])

    # Evaluation
    model.eval()
    errs = {}
    for corruption in corruptions:
        dataset = CustomDataset(images_path=f"{data_root_dir}/{corruption}", labels_path=f"{data_root_dir}/labels", transform=transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        correct = 0
        for data, target in data_loader:
            data = data.double().to(device)
            target = target.to(device)
            pred = model(data, None, return_output='hard', compute_loss=False)
            correct += pred.eq(target).sum()
        errs[corruption] = (1 - 1.*correct/len(dataset)).item()

        print(f"Corruption: {corruption}, Error: {errs[corruption]}")
    print(f'Average: {np.mean(list(errs.values()))}')

    # Save results
    results = pd.DataFrame(list(errs.items()), columns=['name', name]).set_index('name').T
    print(results)
    results.to_csv(f'./saved_results/CIFAR-10-C/{name}.csv')



