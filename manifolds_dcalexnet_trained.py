import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.utils.make_manifold_data import make_manifold_data
from mftma.utils.activation_extractor import extractor
from mftma.utils.analyze_pytorch import analyze
from collections import OrderedDict
import models_dc

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def main():
    modelpth = '/home/annatruzzi/deepcluster_models/alexnet/'
    checkpoint = torch.load(modelpth+'checkpoint_dc.pth.tar')['state_dict']
    checkpoint_new = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace(".module", '') # remove 'module.' of dataparallel
        checkpoint_new[name]=v

    model = models_dc.alexnet(sobel=True, bn=True, out=10000) 
    model.load_state_dict(checkpoint_new)
    model.cuda()
    model = model.eval()
    sampled_classes = 100
    examples_per_class = 50

    ###############  CHANGE DATASET LOADING TO YOUR STIMULI #########   
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    img_pth = '/data/ILSVRC2012/val_in_folders'
    train_dataset = datasets.ImageFolder(img_pth,transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = make_manifold_data(train_dataset, sampled_classes, examples_per_class, seed=0)
    data = [d.to(device) for d in data]
    activations = extractor(model, data, layer_types=['ReLu'])
    list(activations.keys())
    with open ('./activations_dict_keys.txt','w') as file:
	file.write(list(activations.keys()))

    for layer, data, in activations.items():
        X = [d.reshape(d.shape[0], -1).T for d in data]
        # Get the number of features in the flattened data
        N = X[0].shape[0]
        # If N is greater than 5000, do the random projection to 5000 features
        if N > 5000:
            print("Projecting {}".format(layer))
            M = np.random.randn(5000, N)
            M /= np.sqrt(np.sum(M*M, axis=1, keepdims=True))
            X = [np.matmul(M, d) for d in X]
        activations[layer] = X

    capacities = []
    radii = []
    dimensions = []
    correlations = []

    for k, X, in activations.items():
        # Analyze each layer's activations
        a, r, d, r0, K = manifold_analysis_corr(X, 0, 300, n_reps=1)
        
        # Compute the mean values
        a = 1/np.mean(1/a)
        r = np.mean(r)
        d = np.mean(d)
        print("{} capacity: {:4f}, radius {:4f}, dimension {:4f}, correlation {:4f}".format(k, a, r, d, r0))
        
        # Store for later
        capacities.append(a)
        radii.append(r)
        dimensions.append(d)
        correlations.append(r0)


    ##### Plot the results
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].plot(capacities, linewidth=5)
    axes[1].plot(radii, linewidth=5)
    axes[2].plot(dimensions, linewidth=5)
    axes[3].plot(correlations, linewidth=5)

    axes[0].set_ylabel(r'$\alpha_M$', fontsize=18)
    axes[1].set_ylabel(r'$R_M$', fontsize=18)
    axes[2].set_ylabel(r'$D_M$', fontsize=18)
    axes[3].set_ylabel(r'$\rho_{center}$', fontsize=18)

    names = list(activations.keys())
    names = [n.split('_')[1] + ' ' + n.split('_')[2] for n in names]
    for ax in axes:
        ax.set_xticks([i for i, _ in enumerate(names)])
        ax.set_xticklabels(names, rotation=90, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.show()
    


if __name__ == '__main__':
    main()
