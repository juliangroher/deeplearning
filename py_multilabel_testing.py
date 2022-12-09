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
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, ResNet50_Weights
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
from time import process_time
from sklearn.model_selection import train_test_split
from torch import optim
from sklearn.metrics import fbeta_score


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


def explore_learning_rates(model, min_rate, max_rate, dataset):
    learning_rates = np.logspace(start=max_rate, stop=min_rate)
    train_dl = torch.utils.data.DataLoader(dataset, batch_size=128,
                                           shuffle=True, num_workers=0, persistent_workers=False)
    model.train()
    batch = 0
    for xb, yb in train_dl:
        pass


# From https://pytorch.org/tutorials/beginner/saving_loading_models.html
def save_model_and_optimizer(path, model, optimizer, epoch, loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, path+f"_{epoch}.pt")


def load_model_and_optimizer(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    return model, optimizer, epoch, loss


def activate_all_grad(model):
    for param in model.parameters():
        param.requires_grad = True


def activate_only_decoder_grad(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters(): # ! change
        param.requires_grad = True


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

        """ Labels and their count
        Counter({'agriculture': 12315,
         'partly_cloudy': 7261,
         'primary': 37513,
         'road': 8071,
         'clear': 28431,
         'cloudy': 2089,
         'water': 7411,
         'habitation': 3660,
         'cultivation': 4547,
         'slash_burn': 209,
         'haze': 2697,
         'bare_ground': 862,
         'selective_logging': 340,
         'artisinal_mine': 339,
         'blow_down': 101,
         'blooming': 332,
         'conventional_mine': 100})
         """

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

        if self.preprocessor is not None:
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


def visualize_histories(train_loss_history,
                        valid_loss_history,
                        valid_accuracy_history,
                        class_wise_f2_history,
                        row_wise_f2_mean_history,
                        plot_time_train,
                        plot_time_valid
                        ):
    # Display training and validation losses
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
    plt.legend(lines, labels, loc="upper right", prop=dict(size=14))
    plt.tight_layout()
    plt.show()

    # Display accuracy and average F2 score
    plt.figure()
    lines = []
    labels = []
    l, = plt.plot(plot_time_valid[1:], row_wise_f2_mean_history)
    lines.append(l)
    labels.append('F2')
    l, = plt.plot(plot_time_valid[1:], valid_accuracy_history)
    lines.append(l)
    labels.append('Accuracy')
    plt.title('F2 / Accuracy')
    plt.legend(lines, labels, loc="lower right", prop=dict(size=14))
    plt.show()

    # Display the class-wise F2 scores
    tags = ['haze', 'primary', 'agriculture', 'clear', 'water', 'habitation', 'road', 'cultivation', 'slash_burn',
            'cloudy', 'partly_cloudy', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming',
            'selective_logging', 'blow_down']

    for i in range(3):
        plt.figure()
        lines = []
        labels = []
        for j in range(6):
            if i == 2 and j == 5:
                continue

            class_idx = 6 * i + j
            class_f2 = class_wise_f2_history[:, class_idx]

            l, = plt.plot(plot_time_valid[1:], class_f2)
            lines.append(l)
            labels.append(tags[class_idx])

        plt.title("Some class-wise F2 scores")
        plt.legend(lines, labels, loc="upper left", prop=dict(size=14))
        plt.show()


def evaluate_model(model, dataLoader):
    with torch.no_grad():
        tp_fp_tn_fn = torch.zeros((4, 17)).cuda()  # For class-wise
        row_wise_fbeta_scores = []
        beta = 2
        n = 0

        for xb, yb in dataLoader:
            xb = xb.cuda()
            yb = yb.cuda()
            n += xb.shape[0]  # Keep a running count of the amount of data points
            yhat = model(xb)

            # Binarize yhat
            yhat[yhat >= 0] = 1
            yhat[yhat < 0] = 0

            """ Labels and their count
                    Counter({'agriculture': 12315,
                     'partly_cloudy': 7261,
                     'primary': 37513,
                     'road': 8071,
                     'clear': 28431,
                     'cloudy': 2089,
                     'water': 7411,
                     'habitation': 3660,
                     'cultivation': 4547,
                     'slash_burn': 209,
                     'haze': 2697,
                     'bare_ground': 862,
                     'selective_logging': 340,
                     'artisinal_mine': 339,
                     'blow_down': 101,
                     'blooming': 332,
                     'conventional_mine': 100})
                     """

            # What happens if we set every prediction to 1?
            # yhat = torch.ones(yhat.shape)
            """(0.16881248354911804,
             [0.27165505290031433,
              0.9848564267158508,
              0.6865255832672119,
              0.9197127819061279,
              0.5361173748970032,
              0.3306685984134674,
              0.5643631219863892,
              0.3572688698768616,
              0.018257059156894684,
              0.2173469364643097,
              0.5263158082962036,
              0.008586850017309189,
              0.1062387004494667,
              0.03598848357796669,
              0.03598848357796669,
              0.03598848357796669,
              0.011018609628081322],
             0.48350867708931444)"""

            # What happens if we set every prediction to 0?
            """
            (0.8314781188964844,
             [0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0],
             0.0)"""
            # yhat = torch.zeros(yhat.shape)

            # What happens if we set primary and clear to 1 and everything else to 0?
            """(0.7702714800834656,
             [0.0,
              0.0,
              0.672374427318573,
              0.0,
              0.5175158977508545,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0],
             0.1338047402185708)"""
            # yhat = torch.zeros(yhat.shape)
            # yhat[:, (2, 4)] = 1

            # What happens if we set agriculture, primary, and clear to 1 and everything else to 0?
            """(0.720413863658905,
             [0.2731660306453705,
              0.0,
              0.6895214915275574,
              0.0,
              0.5267295837402344,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0],
             0.1575685234063051)
            """
            # yhat = torch.zeros(yhat.shape)
            # yhat[:, (0, 2, 4)] = 1

            # Calculation of true/false positives/negatives
            trues = (yhat == yb)
            falses = (yhat != yb)

            # TP
            batch_tp = torch.logical_and(trues, yhat)
            tp_fp_tn_fn[0, :] += batch_tp.sum(dim=0)
            # FP
            batch_fp = torch.logical_and(falses, yhat)
            tp_fp_tn_fn[1, :] += batch_fp.sum(dim=0)
            # TN
            batch_tn = torch.logical_and(trues, torch.logical_not(yhat))
            tp_fp_tn_fn[2, :] += batch_tn.sum(dim=0)
            # FN
            batch_fn = torch.logical_and(falses, torch.logical_not(yhat))
            tp_fp_tn_fn[3, :] += batch_fn.sum(dim=0)

            batch_row_wise_fbeta_scores = ((1 + beta ** 2) * batch_tp.sum(dim=1)) / ((1 + beta ** 2) * batch_tp.sum(dim=1) + (beta ** 2) * batch_fn.sum(dim=1) + batch_fp.sum(dim=1))
            # batch_row_wise_fbeta_scores = ((1 + beta ** 2) * batch_tp.sum(dim=1)) / ((1 + beta ** 2) * batch_tp.sum(dim=1) + (beta ** 2) * batch_fn.sum(dim=1) + batch_fp.sum(dim=1))
            row_wise_fbeta_scores.extend(batch_row_wise_fbeta_scores.tolist())

        # Accuracy calculation
        accuracy = torch.sum(tp_fp_tn_fn[(0,2), :]) / n  # (TP + TN) / n
        accuracy = accuracy / 17  # Mean over all classes

        # Class-wise f_beta score
        tp, fp, tn, fn = torch.vsplit(tp_fp_tn_fn, 4)
        class_wise_fbeta_scores = ((1 + beta**2) * tp) / ((1 + beta**2) * tp + (beta**2) * fn + fp)

        row_wise_fbeta_score_mean = np.mean(row_wise_fbeta_scores)
        return accuracy.item(), class_wise_fbeta_scores.tolist()[0], row_wise_fbeta_score_mean


if __name__ == "__main__":
    local = True

    label_path = 'datasets/deeplearningproject/train_classes.csv'
    if not local:
        label_path = '/' + label_path
    label_df = pd.read_csv(label_path)

    network_weights = MobileNet_V3_Small_Weights.DEFAULT
    network_weights = ResNet50_Weights.DEFAULT

    model = Net()
    print('Network parameters:\n')
    print(model)
    # Print parameter shapes
    for name, param in model.named_parameters(): print('parameter', name, param.shape)

    def accuracy(scores, yb):
        # print("scores", scores.dtype)
        # print("yb", yb.dtype)
        # Binarize predictions via thresholding
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0

        return (scores == yb).float().mean()

    # Gotten by computing the fraction of occurrences of each label in the entire dataset
    # and then taking its reciprocal
    loss_weights = torch.Tensor(
        [
            15.00889877641824,
            1.0790659238130782,
            3.286967113276492,
            1.4237627941331645,
            5.462015922277695,
            11.059836065573771,
            5.015363647627308,
            8.90235319991203,
            193.67942583732057,
            19.377213977979896,
            5.574851948767387,
            404.79,
            46.95939675174014,
            119.4070796460177,
            121.92469879518072,
            119.05588235294118,
            400.7821782178218
        ]
    ).cuda()

    # idf-scores of all the weights
    loss_weights = torch.Tensor(
        [
             2.708643276959167,
             0.0760957815487335,
             1.1899652895442587,
             0.35330322196059244,
             1.697817938147538,
             2.4033201737034853,
             1.6125059307745246,
             2.1863156462738864,
             5.266204348134086,
             2.9640978385423313,
             1.7182657607186502,
             6.003368414110806,
             3.849283329435204,
             4.782538492718447,
             4.803403631182408,
             4.77959298248869,
             5.993418083257637
         ]
    ).cuda()
    loss_func = torch.nn.BCEWithLogitsLoss(weight=loss_weights)

    images_path = 'datasets/temp-small-subset'

    if not local:
        images_path = '/' + images_path
    # preprocess = network_weights.transforms()
    preprocess = ResNet50_Weights.IMAGENET1K_V1.transforms()


    images_path = 'datasets/deeplearningproject/train-jpg'

    # images_path = 'datasets/temp-small-subset'
    if not local:
        images_path = '/' + images_path
    # preprocess = network_weights.transforms()
    preprocess = ResNet50_Weights.IMAGENET1K_V1.transforms()#network_weights.transforms()

    label_path = "datasets/deeplearningproject/train_classes.csv"

    labels = pd.read_csv(label_path)
    labels = labels.sample(frac=1, random_state=0)  # Randomize dataset

    train_data, test_daa = train_test_split(labels[:], train_size=0.9, random_state=0)

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

    def adam_optimizer(model):
        return optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


    import matplotlib.pyplot as plt
    def base_optimizer(model, lr=0.001, momentum=0.9):
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


    # Function handle that updates the learning rate
    # (note this is a dummy implementation that does nothing)
    # https://www.geeksforgeeks.org/adjusting-learning-rate-of-a-neural-network-in-pytorch/
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    from torch.optim.lr_scheduler import StepLR
    def base_lr_scheduler(optimizer, step_size=4, gamma=0.4):
        return StepLR(optimizer, step_size, gamma)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    # LR for the smaller, pruned dataset
    '''
    def better_lr_scheduler(optimizer, factor=0.1, patience=4, threshold=0.005, cooldown=2):
        return ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=factor, patience=patience, threshold=threshold, cooldown=cooldown)
    '''
    def better_lr_scheduler(optimizer, factor=0.1, patience=2, threshold=0.1, cooldown=1):
        return ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=factor, patience=patience, threshold=threshold, cooldown=cooldown)

    # Function to fit a model
    def fit(model,
            opt_func=adam_optimizer,
            lr_scheduler=better_lr_scheduler,
            bs=256,
            epochs=1,
            batches_per_epoch=None,  # Default: Use entire training set
            show_summary=True,
            workers=0,
            save_path=""):

        # Set up data loaders
        if batches_per_epoch == None:
            # Use all images
            train_dl = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                   shuffle=True, num_workers=workers, persistent_workers=workers)
            valid_dl = torch.utils.data.DataLoader(testset, batch_size=bs,
                                                   shuffle=False, num_workers=workers, persistent_workers=workers)
            batches_per_epoch = len(train_dl)
        else:
            # Only use a subset of the data
            subset_indices = list(range(batches_per_epoch * bs))
            train_dl = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices),
                                                   num_workers=workers, persistent_workers=workers)

            # Use one fourth for validation
            subset_indices = list(range(int(np.ceil(batches_per_epoch / 4)) * bs))
            valid_dl = torch.utils.data.DataLoader(testset, batch_size=bs,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices),
                                                   num_workers=workers, persistent_workers=workers)

        # Initialize optimizer
        opt = opt_func(model)
        lrs = lr_scheduler(opt)

        # For book keeping
        train_loss_history = []
        valid_loss_history = []
        plot_time_train = []
        plot_time_valid = []
        valid_acc_history = []
        class_wise_f2_history = []
        row_wise_f2_mean_history = []

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
            if epoch == 2:
                model.cpu()
                activate_all_grad(model)
                model.cuda()

            model.train()  # Train mode
            for xb, yb in train_dl:
                # Update learning rate
                # opt.param_groups[0]['lr'] = lr_scheduler(t,T,lr=opt.param_groups[0]['lr'])

                # Forward prop
                if torch.cuda.is_available():
                    pred = model(xb.cuda())
                    loss = loss_func(pred, yb.cuda())
                else:
                    pred = model(xb)
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
                if t % 20 == 0:  # print every 10 mini-batches
                    model.eval()  # Test mode
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            valid_acc, class_wise_f2, row_wise_f2_mean = evaluate_model(model, valid_dl)
                            valid_acc_history.append(valid_acc)
                            class_wise_f2_history.append(class_wise_f2)
                            row_wise_f2_mean_history.append(row_wise_f2_mean)
                            # valid_acc = sum(accuracy(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl) / len(
                            #     valid_dl)
                            valid_loss = sum(loss_func(model(xb.cuda()), yb.cuda()) for xb, yb in valid_dl) / len(
                                valid_dl)
                            valid_loss_history.append(valid_loss.detach().cpu().numpy())
                            plot_time_valid.append(t - 1)
                            print('t', t, 'lr', opt.param_groups[0]['lr'], 'train loss', loss.detach().cpu().numpy(),
                                  'val loss', valid_loss.detach().cpu().numpy(), 'val accuracy',
                                  valid_acc, "row-wise f2 mean", row_wise_f2_mean)
                        else:
                            valid_acc, class_wise_f2, row_wise_f2_mean = evaluate_model(model, valid_dl)
                            valid_acc_history.append(valid_acc)
                            class_wise_f2_history.append(class_wise_f2)
                            row_wise_f2_mean_history.append(row_wise_f2_mean)
                            # valid_acc = sum(accuracy(model(xb), yb) for xb, yb in valid_dl) / len(valid_dl)
                            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl) / len(valid_dl)
                            valid_loss_history.append(valid_loss.detach().cpu().numpy())
                            plot_time_valid.append(t - 1)
                            print('t', t, 'lr', opt.param_groups[0]['lr'], 'train loss', loss.detach().cpu().numpy(),
                                  'val loss', valid_loss.detach().cpu().numpy(), 'val accuracy',
                                  valid_acc, "row-wise f2 mean", row_wise_f2_mean)
                    model.train()  # Back to train mode

            # Adjust learning rate
            lrs.step(valid_loss)

            PATH = f"{save_path}_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss.detach().cpu().numpy(),
                "valid_loss_history": valid_loss_history,
                "valid_accuracy_history": valid_acc_history,
                "class_wise_f2_history": class_wise_f2_history,
                "row_wise_f2_mean_history": row_wise_f2_mean_history,
                "train_loss_history": train_loss_history,
                "plot_time_train": plot_time_train,
                "plot_time_valid": plot_time_valid,
            }, PATH)

        post_train_time = perf_counter()
        print(f"Training took {post_train_time - pre_training_time} seconds with {workers} workers.\n")

        # Summary
        if show_summary:
            visualize_histories(train_loss_history, valid_loss_history, valid_acc_history, class_wise_f2_history, row_wise_f2_mean_history, plot_time_train, plot_time_valid)
            '''plt.figure()
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

            plt.figure()
            lines = []
            labels = []
            l, = plt.plot(plot_time_valid, row_wise_f2_mean_history)
            lines.append(l)
            labels.append('F2')
            l, = plt.plot(plot_time_valid, valid_acc_history)
            lines.append(l)
            labels.append('Accuracy')
            plt.title('F2 / Accuracy')
            plt.legend(lines, labels, loc=(1, 0), prop=dict(size=14))
            plt.show()'''

        return train_loss_history

    '''
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
                                                   shuffle=True, num_workers=workers, persistent_workers=workers)
            valid_dl = torch.utils.data.DataLoader(testset, batch_size=bs,
                                                   shuffle=False, num_workers=workers, persistent_workers=workers)
            batches_per_epoch = len(train_dl)
        else:
            # Only use a subset of the data
            subset_indices = list(range(batches_per_epoch * bs))
            train_dl = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices),
                                                   num_workers=workers, persistent_workers=workers)

            # Use one fourth for validation
            subset_indices = list(range(int(np.ceil(batches_per_epoch / 4)) * bs))
            valid_dl = torch.utils.data.DataLoader(testset, batch_size=bs,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices),
                                                   num_workers=workers, persistent_workers=workers)

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
                # opt.param_groups[0]['lr'] = lr_scheduler(t, T, lr=opt.param_groups[0]['lr'])
                opt.param_groups[0]['lr'] = lr_scheduler(opt)

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

        return train_loss_history'''

    # Function to fit a model
    model = Net()

    # Re-initialize weights
    # reset_parameters(model)
    if torch.cuda.is_available():
        model.cuda()

    # Train with default settings.
    # train_loss_history = fit(model, bs = 256, epochs = 1, workers=i)
    # train_loss_history = fit(model, bs = i, epochs = 1, workers=0)
    # train_loss_history = fit(model, bs = 64, epochs = 12, workers=4, save_path="model_with_extended_bookkeeping_and_consistent_splitting_and_idf_weights")

    # Clean this up tomorrow TODO:
    # Function to fit a model
    model = Net()

    # Re-initialize weights
    # reset_parameters(model)
    if torch.cuda.is_available():
        model.cuda()

    # Smaller, pruned, and augmented dataset. Clean this up later TODO:
    pruned_path = "datasets/deeplearningproject/delete_classes.csv"
    pruned_labels = pd.read_csv(pruned_path)
    pruned_labels = pruned_labels.sample(frac=1, random_state=0)

    # Overwrite the training/test sets
    label_path = "datasets/deeplearningproject/train_classes.csv"

    labels = pd.read_csv(label_path)
    '''
    # Take out all the elements of labels that are contained in pruned_labels
    a = pruned_labels.copy()
    b = labels.copy()
    a = a[a["image_name"].str.contains("train")]
    # a = a["image_name"].str.split("_", expand=True)[1]
    df_all = b.merge(a.drop_duplicates(), on=['image_name'],
                     how='left', indicator=True)
    df_unused_only = df_all[df_all['_merge'] == 'left_only']
    df_unused_only = df_unused_only[["image_name", "tags_x"]]
    df_unused_only.rename(columns={'tags_x': 'tags'}, inplace=True)
    labels = df_unused_only'''

    # Take out all the original training images from the pruned set
    # This is done so we don't accidentally train on images that we want to test on
    pruned_labels = pruned_labels[pruned_labels["image_name"].str.startswith("train")]

    labels = labels.sample(frac=1, random_state=0)  # Randomize dataset
    _, test_daa = train_test_split(labels[:], train_size=0.6, random_state=0)
    '''
    trainset = MultilabelDataset(
        pruned_labels,
        images_path,
        preprocessor=preprocess
    )

    testset = MultilabelDataset(
        test_daa,
        images_path,
        preprocessor=preprocess
    )
    '''
    # train_loss_history = fit(model, bs=64, epochs=200, workers=4, save_path="model_with_pruned_data_proper_set_and_different_lrs_v2")


    checkpoint = torch.load("model_with_extended_bookkeeping_and_consistent_splitting_and_idf_weights_7.pt")
    plot_time_valid = checkpoint["plot_time_valid"]
    row_wise_f2_mean_history = checkpoint["row_wise_f2_mean_history"]
    valid_acc_history = checkpoint["valid_accuracy_history"]
    class_wise_f2_history = np.array(checkpoint["class_wise_f2_history"])
    plot_time_train = checkpoint["plot_time_train"]
    train_loss_history = checkpoint["train_loss_history"]
    valid_loss_history = checkpoint["valid_loss_history"]
    visualize_histories(train_loss_history, valid_loss_history, valid_acc_history, class_wise_f2_history, row_wise_f2_mean_history, plot_time_train, plot_time_valid)


    model.load_state_dict(checkpoint["model_state_dict"])

    train_dl = torch.utils.data.DataLoader(testset, batch_size=128,  # batch_size=testset.__len__(),
                                           shuffle=True, num_workers=4, persistent_workers=4)
    model.eval()
    # model.cpu()
    evaluate_model(model, train_dl)

    '''
    for xb, yb in train_dl:
        model.cpu()
        yhat = model(xb)
        hazy_clear_predictions = yhat[:, :]
        hazy_clear_binarized = hazy_clear_predictions > 0
        hazy_clear_amounts = torch.sum(hazy_clear_binarized, axis=0)
        label_frequencies = [0.06662713999851776, 0.9267274389189456, 0.3042318239086934, 0.702364188838657, 0.18308258603226366, 0.090417253390647, 0.1993873366436918, 0.11232985004570271, 0.005163171026952247, 0.05160700610192939, 0.1793769608933027, 0.002470416759307295, 0.021294992465228885, 0.00837471281405173, 0.00820178364090022, 0.008399416981644803, 0.002495120926900368]
        hazy_clear_frequencies = hazy_clear_amounts / yhat.shape[0]
        for i, label in enumerate(['haze', 'primary', 'agriculture', 'clear', 'water', 'habitation', 'road', 'cultivation', 'slash_burn',
                                   'cloudy', 'partly_cloudy', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming',
                                   'selective_logging', 'blow_down']):
            print(label, "real:", np.round(label_frequencies[i], decimals=4), "predicted:", hazy_clear_frequencies[i])
        print(accuracy(yhat.cpu(), yb))
        print(f"f2score: {fbeta_score(yb, yhat.detach().cpu(), beta=2, average=None)}")

    '''