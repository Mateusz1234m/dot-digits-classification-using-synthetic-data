import torch
import torch.nn as nn
import torchvision.models.resnet
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_generation_module import DigitsGenerator
import matplotlib.pyplot as plt

# Paths
FONTS_PATH = 'fonts/'
BACKGROUNDS_PATH = 'backgrounds/'
WRONG_PREDS_PATH = 'wrong_preds/'

# Hyperparameters
in_channels = 1
size = (75, 50)
num_classes = 10
learning_rate = 3e-4
batch_size = 64
n_epochs = 100
# model_path = 'models/best_83.pt'
best = 0


class NeuralNetwork(nn.Module):
    """
    Neural network class - Resnet18
    """
    def __init__(self, in_channels, n_classes):
        super(NeuralNetwork, self).__init__()
        self.device = device
        self.resnet18 = torchvision.models.resnet.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet18.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                              bias=False)
        self.resnet18.fc = nn.Linear(512, n_classes)
        self.resnet18.to(device=device)

    def forward(self, x):
        x = F.softmax(self.resnet18(x), dim=1)
        return x


class DigitsDataset(Dataset):
    """
    Dataset class - requires
    """
    def __init__(self, digits_generator, length, transform):
        """
        Initializes dataset
        :param digits_generator: synthetic data generator - generates images of digits
        :param length: length of dataset - defines length of epoch, but every epoch new images will be generated
        :param transform: transforms for inputs
        """
        self.digits_generator = digits_generator
        self.transform = transform
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # get image and label from image generator
        img, label = self.digits_generator.generate_image()

        # transform image
        img = self.transform(img)

        # convert label to tensor of proper size
        temp_label = [0] * num_classes
        temp_label[int(label)] = 1
        label = torch.Tensor(temp_label)

        return img, label


def check_accuracy(loader, model):
    """
    Checks accuracy on given dataloader
    :param loader: dataloader
    :param model: evauated model
    :return: accuracy of the model on given dataloader
    """

    # initialize counters
    n_correct = 0
    n_samples = 0

    # turn model to evaluation mode
    model.eval()

    with torch.no_grad():

        # get data and labels from loader
        for data, labels in loader:

            # send data and labels to loaders
            data = data.to(device=device)
            labels = labels.to(device=device)

            # get result from forward pass
            scores = model(data)

            # get predictions
            _, predictions = scores.max(1)
            if labels.dim() > 1:
                _, labels = labels.max(1)

            # count number of correct predictions
            n_correct += (predictions == labels).sum()
            n_samples += predictions.size(0)

    # turn model back to training mode
    model.train()

    return n_correct / n_samples


def save_wrong_predictions(model, loader):
    """
    Checks accuracy on given dataloader and saves wrong prediction
    :param loader: dataloader
    :param model: evauated model
    :return: accuracy of the model on given dataloader
    """

    # initialize counters
    n_correct = 0
    n_samples = 0

    # turn model to evaluation mode
    model.eval()

    with torch.no_grad():

        # get data and labels from loader
        for data, labels in loader:

            # send data and labels to loaders
            data = data.to(device=device)
            labels = labels.to(device=device)

            # get result from forward pass
            scores = model(data)

            # get predictions
            _, predictions = scores.max(1)
            if labels.dim() > 1:
                _, labels = labels.max(1)

            # get indices of wrong predictions
            indices = (predictions != labels.view_as(predictions)).nonzero()

            # count number of correct predictions
            n_correct += (predictions == labels).sum()
            n_samples += predictions.size(0)

            # get wrongly predicted data
            wrong_samples = data[indices]

            # get wrong predicitons
            wrong_preds = predictions[indices]

            # get actual classes
            actual_preds = labels.view_as(predictions)[indices]

            for i in range(indices.shape[0]):

                # remove batch dimension
                sample = wrong_samples[i].squeeze(dim=0)

                # get predictiton and actual class
                wrong_pred = wrong_preds[i]
                actual_pred = actual_preds[i]

                # convert tensor of data to image
                img = transforms.ToPILImage()(sample)

                # save image
                img.save('wrong_preds/wrong_idx{}_pred{}_actual{}.png'.format(
                    indices[i], wrong_pred.item(), actual_pred.item()))

    # turn model back to training mode
    model.train()

    return n_correct / n_samples




if __name__ == "__main__":

    # Initialize lists, where accuracies will be stored
    accs_synth = []
    accs_c1 = []
    accs_c2 = []
    accs_c3 = []
    accs_c4 = []
    accs_c5 = []

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize digits generator
    digits_generator = DigitsGenerator(fonts_path=FONTS_PATH, backgrounds_path=BACKGROUNDS_PATH,
                                       digit_height_ratio_min=0.3,
                                       digit_height_ratio_max=1, digit_rotation_angle_min=-45,
                                       digit_rotation_angle_max=45, background_rotation_angle_min=-30,
                                       background_rotation_angle_max=30, digit_position_max_shift=300,
                                       noise_factor_max=0, resize_factor_max=200,
                                       digit_shift_max_x=10, digit_shift_max_y=75, color_shift_max=10,
                                       final_resize_factor_max=500, shadow_blur_max=10, shadow_cover_ratio_min=50,
                                       shadow_cover_ratio_max=100,
                                       shadow_distance_max=15, glow_cover_ratio_min=0, glow_cover_ratio_max=1,
                                       glow_blur_max=1, glow_distance_max=4, digit_blur_max=4, background_size_min=600,
                                       background_size_max=1000)

    # Initialize synthetic training dataset
    train_dataset = DigitsDataset(digits_generator, length=1280, transform=transforms.Compose([transforms.ToTensor(),
                                                                                               transforms.Resize(size)
                                                                                               ]))

    # Initialize synthetic test dataset
    test_dataset_synthetic = DigitsDataset(digits_generator, length=256, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(size)]))

    # Initialize test datasets with real images
    # Every dataset contains the same images but differently cropped
    test_dataset_c1 = torchvision.datasets.ImageFolder(
        root='C:/Users/mateu/Desktop/projekty/dot_classification/test_dataset/crop_1',
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Resize(size)]))
    test_dataset_c2 = torchvision.datasets.ImageFolder(
        root='C:/Users/mateu/Desktop/projekty/dot_classification/test_dataset/crop_2',
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Resize(size)]))
    test_dataset_c3 = torchvision.datasets.ImageFolder(
        root='C:/Users/mateu/Desktop/projekty/dot_classification/test_dataset/crop_3',
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Resize(size)]))
    test_dataset_c4 = torchvision.datasets.ImageFolder(
        root='C:/Users/mateu/Desktop/projekty/dot_classification/test_dataset/crop_4',
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Resize(size)]))
    test_dataset_c5 = torchvision.datasets.ImageFolder(
        root='C:/Users/mateu/Desktop/projekty/dot_classification/test_dataset/crop_5',
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Resize(size)]))

    # Initialize train dataloader with synthetic images
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    # Initialize test dataloader with synthetic images
    test_loader_synthetic = DataLoader(dataset=test_dataset_synthetic, batch_size=batch_size, shuffle=False)

    # Initialize test dataloader with real images
    test_loader_c1 = DataLoader(dataset=test_dataset_c1, batch_size=batch_size, shuffle=False)
    test_loader_c2 = DataLoader(dataset=test_dataset_c2, batch_size=batch_size, shuffle=False)
    test_loader_c3 = DataLoader(dataset=test_dataset_c3, batch_size=batch_size, shuffle=False)
    test_loader_c4 = DataLoader(dataset=test_dataset_c4, batch_size=batch_size, shuffle=False)
    test_loader_c5 = DataLoader(dataset=test_dataset_c5, batch_size=batch_size, shuffle=False)

    # Initialize neural network
    model = NeuralNetwork(in_channels=in_channels, n_classes=num_classes).to(device)
    # model.load_state_dict(torch.load(model_path)) # remember to change best above

    # Specify loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for i in range(n_epochs):
        print(f"Epoch: {i}")
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        # Test on test sets
        acc_synth = check_accuracy(test_loader_synthetic, model)
        acc_c1 = check_accuracy(test_loader_c1, model)
        acc_c2 = check_accuracy(test_loader_c2, model)
        acc_c3 = check_accuracy(test_loader_c3, model)
        acc_c4 = check_accuracy(test_loader_c4, model)
        acc_c5 = check_accuracy(test_loader_c5, model)

        # Append accuracies
        accs_synth.append(acc_synth)
        accs_c1.append(acc_c1)
        accs_c2.append(acc_c2)
        accs_c3.append(acc_c3)
        accs_c4.append(acc_c4)
        accs_c5.append(acc_c5)

        # Print accuracies
        print(f"Accuracy on synthetic set: {acc_synth * 100:.2f}")
        print(f"Accuracy on test set c1: {acc_c1 * 100:.2f}")
        print(f"Accuracy on test set c2: {acc_c2 * 100:.2f}")
        print(f"Accuracy on test set c3: {acc_c3 * 100:.2f}")
        print(f"Accuracy on test set c4: {acc_c4 * 100:.2f}")
        print(f"Accuracy on test set c5: {acc_c5 * 100:.2f}")
        print(f"Best: {best * 100:.2f}")

        # Save model
        if max(acc_c1, acc_c2, acc_c3, acc_c4, acc_c5) > best:
            best = max(acc_c1, acc_c2, acc_c3, acc_c4, acc_c5)
            torch.save(model.state_dict(), f"models/best_{int(max(acc_c1, acc_c2, acc_c3, acc_c4, acc_c5) * 100)}.pt")
            print("Saving model...")

    # Specify limits on the plot
    plt.ylim([0, 1])
    plt.xlim([0, n_epochs])

    # Specify x values
    x = np.arange(n_epochs)

    # Plot accuracies on synthetic test set
    y = np.asarray(accs_synth)
    plt.plot(x, y, color='#ff0000', label='Synthetic')

    # Plot accuracies on real test set c1
    y = np.asarray(accs_c1)
    plt.plot(x, y, color='#0008ff', label='c1')

    # Plot accuracies on real test set c2
    y = np.asarray(accs_c2)
    plt.plot(x, y, color='#0055ff', label='c2')

    # Plot accuracies on real test set c3
    y = np.asarray(accs_c3)
    plt.plot(x, y, color='#0080ff', label='c3')

    # Plot accuracies on real test set c4
    y = np.asarray(accs_c4)
    plt.plot(x, y, color='#00b3ff', label='c4')

    # Plot accuracies on real test set c5
    y = np.asarray(accs_c5)
    plt.plot(x, y, color='#00eaff', label='c5')

    # Show legend
    plt.legend()

    # Save or show plot
    # plt.savefig('results.png')
    plt.show()
