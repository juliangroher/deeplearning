import pandas as pd
import numpy as np
from torch import optim
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from time import process_time, process_time_ns
from time import time, perf_counter


def reset_parameters(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)  # Why 1?
            torch.nn.init.constant_(m.bias, 0)  # Why 0?
        elif isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)


# Dataset subclass for our own image set
# TODO: change to support multiple labels
class CustomImageDataset(Dataset):
    def __init__(self, img_labels, img_dir, transform=None, target_transform=None, preprocessor=None):
        self.img_labels = img_labels  # pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + '.jpg')
        image = read_image(img_path)
        image = image[:3, :, :]  # remove alpha channel

        if self.preprocessor != None:
            image = image / 255  # required for preprocessor | normalize each picture
            image = self.preprocessor(image)
        else:
            image = 2 * (image / 255 - 0.5)  # normalize each picture

        label = self.img_labels.iloc[idx, 1]
        label = cloudy_labels.index(label)  # TODO multilabel

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def Net():
    # encoder = mobilenet_v3_small(weights=network_weights)
    encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in encoder.parameters():
        param.requires_grad = False

    decoder = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=120),
        nn.ReLU(),
        nn.Linear(in_features=120, out_features=17)#,
        #nn.Sigmoid()
    )
    for param in decoder.parameters():
        param.requires_grad = True

    encoder.fc = decoder
    # encoder.classifier = decoder
    return encoder


class MultilabelDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, target_transform=None, preprocessor=None):
        self.img_dataframe = dataframe  # pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.preprocessor = preprocessor

        # Encode labels
        tags = ['haze', 'primary', 'agriculture', 'clear', 'water', 'habitation', 'road', 'cultivation', 'slash_burn',
                'cloudy', 'partly_cloudy', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming',
                'selective_logging', 'blow_down']
        split_tags = [row.split(" ") for row in self.img_dataframe["tags"]]
        mlb = MultiLabelBinarizer(classes=tags)
        mlb.fit(split_tags)
        self.img_labels = mlb.transform(split_tags).astype(
            'float32')  # BCELoss does not accept integers *for some reason*
    def __len__(self):
            return len(self.img_labels)

    def __getitem__(self, idx):
        # print(idx)
        # print(self.img_dataframe[idx-10:idx+10])
        img_path = os.path.join(self.img_dir, self.img_dataframe.iloc[idx, 0] + '.jpg')
        # print(img_path)
        pre_time = process_time()
        image = read_image(img_path)
        image = image[:3, :, :]  # remove alpha channel
        loaded_time = process_time()

        if self.preprocessor != None:
            image = image / 255  # required for preprocessor | normalize each picture
            image = self.preprocessor(image)
        else:
            image = 2 * (image / 255 - 0.5)  # normalize each picture

        processed_time = process_time()
        labels = self.img_labels[idx]

        transform_time = process_time()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)

        post_time = process_time()

        # print("load", loaded_time - pre_time, "processing", processed_time - loaded_time, "labels", transform_time - processed_time, "transform", post_time - transform_time)
        # print("total", post_time - pre_time)
        return image, labels


if __name__ == "__main__":

    def filter_labels(dataframe, label_filter=None):
        # label_filter can be used to only consider images
        # with a certain label
        # Can be used to build a primitve classifier
        # considering only one class of labels
        for idx, row in dataframe.iterrows():
            tags = row['tags']
            labels = tags.strip().split(' ')
            intersection = set(label_filter).intersection(set(labels))
            intersectionList = list(intersection)

            # only considers one (first) tag atm
            dataframe.loc[idx].tags = intersectionList[0] if intersectionList else None

        dataframe = dataframe.dropna()
        return dataframe


    # !ls /datasets/deeplearningproject

    local = True

    label_path = 'datasets/deeplearningproject/train_classes.csv'
    if not local:
        label_path = '/' + label_path
    label_df = pd.read_csv(label_path)

    cloudy_labels = ['clear', 'partly_cloudy', 'cloudy', 'haze']

    # We need to remember to exclude the images which have been filtered here
    label_mapping = filter_labels(label_df, label_filter=cloudy_labels)







    # First import what we need
    import torch.nn as nn
    import torch.nn.functional as F



    # reset_parameters(model)

    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, ResNet50_Weights
    import torch.nn as nn
    import torch.nn.functional as F

    network_weights = MobileNet_V3_Small_Weights.DEFAULT
    network_weights = ResNet50_Weights.DEFAULT





    model = Net()


    print('Network parameters:\n')

    print(model)
    for name, param in model.named_parameters(): print('parameter', name, param.shape)

    # Print parameter shapes
    def accuracy(scores, yb):
        # print("scores", scores.dtype)
        # print("yb", yb.dtype)
        # Binarize predictions via thresholding
        scores[scores >= 0.5] = 1

        return (scores == yb).float().mean()


    # weights = ResNet50_Weights.DEFAULT
    # preprocess = weights.transforms()
    #
    # img = images[0]
    # print(img.shape)
    # img_transformed = preprocess(img)
    # print(img_transformed.shape)
    #
    # scores = model(images)
    # print(scores.shape)

    loss_func = torch.nn.BCEWithLogitsLoss()


    # print('Accuracy', accuracy(scores,labels))

    # loss_func = F.cross_entropy
    # loss = loss_func(scores, labels)
    # print('Loss', loss)
    images_path = 'datasets/temp-small-subset'

    # images_path = '/datasets/deeplearningproject/train-jpg'
    if not local:
        images_path = '/' + images_path
    # preprocess = network_weights.transforms()
    preprocess = ResNet50_Weights.IMAGENET1K_V1.transforms()

    trainset = CustomImageDataset(
        label_mapping.iloc[:400],
        images_path,
        preprocessor=preprocess
    )

    testset = CustomImageDataset(
        label_mapping.iloc[400:500],
        images_path,
        preprocessor=preprocess
    )

    from sklearn.preprocessing import MultiLabelBinarizer

    from time import process_time




    from sklearn.model_selection import train_test_split


    images_path = 'datasets/deeplearningproject/train-jpg'

    # images_path = 'datasets/temp-small-subset'
    if not local:
        images_path = '/' + images_path
    # preprocess = network_weights.transforms()
    preprocess = ResNet50_Weights.IMAGENET1K_V1.transforms()#network_weights.transforms()

    label_path = "datasets/deeplearningproject/train_classes.csv"

    labels = pd.read_csv(label_path)
    train_data, test_daa = train_test_split(labels[:], train_size=0.9)

    trainset = MultilabelDataset(
        train_data,
        images_path,
        preprocessor=preprocess
    )

    testset = MultilabelDataset(
        test_daa,
        images_path,
        preprocessor=preprocess
    )

    from torch import optim

    def adam_optimizer(model):
        return optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)




    import matplotlib.pyplot as plt
    def base_optimizer(model, lr=0.001, momentum=0.9):
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


    # Function handle that returns an optimizer
    def base_lr_scheduler(t, T, lr):
        return lr


    # Function handle that updates the learning rate
    # (note this is a dummy implementation that does nothing)
    def fit(model,
            opt_func=adam_optimizer,
            lr_scheduler=base_lr_scheduler,
            bs=256,
            epochs=1,
            batches_per_epoch=None,  # Default: Use entire training set
            show_summary=True,
            workers=0):
        # Set up data loaders
        if batches_per_epoch == None:
            # Use all images
            train_dl = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                   shuffle=True, num_workers=workers, persistent_workers=False)
            valid_dl = torch.utils.data.DataLoader(testset, batch_size=bs,
                                                   shuffle=False, num_workers=workers, persistent_workers=False)
            batches_per_epoch = len(train_dl)
        else:
            # Only use a subset of the data
            subset_indices = list(range(batches_per_epoch * bs))
            train_dl = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices),
                                                   num_workers=workers, persistent_workers=False)

            # Use one fourth for validation
            subset_indices = list(range(int(np.ceil(batches_per_epoch / 4)) * bs))
            valid_dl = torch.utils.data.DataLoader(testset, batch_size=bs,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices),
                                                   num_workers=workers, persistent_workers=False)

        # Initialize optimizer
        opt = opt_func(model)

        # For book keeping
        train_loss_history = []
        valid_loss_history = []
        plot_time_train = []
        plot_time_valid = []

        # Index of current batch
        t = 1

        # Total number of batches
        T = batches_per_epoch * epochs

        print('Epochs:', epochs, 'Batches per epoch:', batches_per_epoch, 'Total number of batches', T)

        # Get initial validation loss and accuracy
        model.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                valid_acc = sum(accuracy(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl) / len(valid_dl)
                valid_loss = sum(loss_func(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl) / len(valid_dl)
                valid_loss_history.append(valid_loss.detach().cpu().numpy())
                plot_time_valid.append(t)
            else:
                valid_acc = sum(accuracy(model(xb), yb) for xb, yb in valid_dl) / len(valid_dl)
                valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl) / len(valid_dl)
                valid_loss_history.append(valid_loss.detach().cpu().numpy())
                plot_time_valid.append(t)

        # Train
        pre_training_time = perf_counter()
        for epoch in range(epochs):
            model.train()  # Train mode
            for xb, yb in train_dl:
                # Update learning rate
                opt.param_groups[0]['lr'] = lr_scheduler(t, T, lr=opt.param_groups[0]['lr'])

                # Forward prop

                if torch.cuda.is_available():
                    pred = model(xb.cuda())
                    loss = loss_func(pred, yb.cuda())
                else:
                    pred = model(xb)
                    # print(pred.shape)
                    # print(yb.shape)
                    # print(pred)
                    # print(yb)
                    loss = loss_func(pred, yb)

                    # Book keeping
                train_loss_history.append(loss.detach().cpu().numpy())
                plot_time_train.append(t)
                t += 1

                # Backward prop (calculate gradient)

                loss.backward()

                # Update model parameters

                opt.step()
                opt.zero_grad()

                # Validation loss and accuracy
                if t % 10 == 0:  # print every 10 mini-batches
                    model.eval()  # Test mode
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            valid_acc = sum(accuracy(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl) / len(valid_dl)
                            valid_loss = sum(loss_func(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl) / len(valid_dl)
                            valid_loss_history.append(valid_loss.detach().cpu().numpy())
                            plot_time_valid.append(t - 1)
                            print('t', t, 'lr', opt.param_groups[0]['lr'], 'train loss', loss.detach().cpu().numpy(),
                                  'val loss', valid_loss.detach().cpu().numpy(), 'val accuracy',
                                  valid_acc.detach().cpu().numpy())
                        else:
                            valid_acc = sum(accuracy(model(xb), yb) for xb, yb in valid_dl) / len(valid_dl)
                            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl) / len(valid_dl)
                            valid_loss_history.append(valid_loss.detach().cpu().numpy())
                            plot_time_valid.append(t - 1)
                            print('t', t, 'lr', opt.param_groups[0]['lr'], 'train loss', loss.detach().cpu().numpy(),
                                  'val loss', valid_loss.detach().cpu().numpy(), 'val accuracy',
                                  valid_acc.detach().cpu().numpy())
                    model.train()  # Back to train mode

            PATH = "model2.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss.detach().cpu().numpy(),
                "valid_loss_history": valid_loss_history,
            }, PATH)

        post_train_time = perf_counter()
        print(f"Training took {post_train_time - pre_training_time} seconds with {workers} workers.\n")

        # Summary
        if show_summary:
            plt.figure()
            lines = []
            labels = []
            l, = plt.plot(plot_time_train, train_loss_history)
            lines.append(l)
            labels.append('Training')
            l, = plt.plot(plot_time_valid, valid_loss_history)
            lines.append(l)
            labels.append('Validation')
            plt.title('Loss')
            plt.legend(lines, labels, loc=(1, 0), prop=dict(size=14))
            plt.show()

        return train_loss_history

    # Function to fit a model
    model = Net()

    # Re-initialize weights
    # reset_parameters(model)
    if torch.cuda.is_available():
        model.cuda()

    # Train with default settings.
    # train_loss_history = fit(model, bs = 256, epochs = 1, workers=i)
    # train_loss_history = fit(model, bs = i, epochs = 1, workers=0)
    train_loss_history = fit(model, bs = 512, epochs = 10, workers=4)
