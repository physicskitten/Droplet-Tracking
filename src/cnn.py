import torch
from torch import nn
import torchvision
import cv2
import numpy as np
from PIL import Image

import pickle
import random

class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()

        self.feature_extraction = nn.Sequential(
            #in_channels: int, out_channels: int
            nn.Conv2d(3, 6, kernel_size=5, stride=1),
            # number of channels
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1920, 960),
            nn.ReLU(inplace=True),
            nn.Linear(960, 480),
            nn.ReLU(inplace=True),
            nn.Linear(480, 240),
            nn.ReLU(inplace=True),
            nn.Linear(240, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 1),
            nn.Sigmoid()
        )

        self.classifier3 = nn.Sequential(
            nn.Linear(1920, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 1),
            nn.Sigmoid()
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(65*65*3, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = torch.flatten(x, 1)
        prob = self.classifier3(x)
        return prob

def load_images(name):
    dir = f"../training_data/{name}"
    with open(f"{dir}/training_data_seed.pickle", "rb") as handle:
        td_seed = pickle.load(handle)
    x_true = []
    y_true = []
    x_false = []
    y_false = []

    for (frame_id, value) in td_seed.points_in_frame.items():
        if frame_id >= 4:
            for point_id in value["particle_ids"]:
                img = Image.open(f"{dir}/image/fid{frame_id}_pid{point_id}.jpg").convert("RGB")
                if img.size[0] == 65 and img.size[1] == 65:
                    img = np.array(img, dtype="float64")
                    img = np.array([img[:, :, 0], img[:, :, 1], img[:, :, 2]])
                    if point_id in td_seed.true_ids:
                        x_true.append(img)
                        y_true.append(1)
                    else:
                        x_false.append(img)
                        y_false.append(0)

    return x_true, y_true, x_false, y_false

class TrainingDataSeed:
    def __init__(self, name, src_video, output_video,true_ids):
        """
        :param name: (str) name of this seed
        :param src_video: (str) name of src video
        :param true_ids: [int] ids of true positives"""
        self.name = name
        self.src_video = src_video
        self.output_video = output_video
        self.true_ids = Set(true_ids)
        self.points_in_frame = {} # {"frame_id": {"particle_ids": [int],"nodes": [[int]]}

    def add_points(self, frame_id, particle_ids, nodes):
        f_ids = list(self.points_in_frame.keys)[:10]
        if frame_id not in f_ids:
            self.points_in_frame[frame_id] = {"particle_ids": [], "nodes": []}
        self.points_in_frame[frame_id]["particle_ids"] += particle_ids
        """for node in nodes:
            if node[0] > 490 or node[1] > 490:
                plot_circles([[200, 200, 1]], [999], color=(0, 200, 0))
                cv2.imshow("test", frame)
                cv2.waitKey(0)
                raise ValueError(f"node outside frame: {node}")"""
        self.points_in_frame[frame_id]["nodes"] += [node[0:2] for node in nodes]

def train(model, criterion, optimizer, epoch, train_loader):
    for (x_data, y_data) in train_loader:
        optimizer.zero_grad()
        outputs = cnn_image(x_data)
        y_data = y_data.unsqueeze(1)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()
        # print("Param", list(model.parameters())[0][0][0][0][0])
    print(outputs)
    # print(outputs)
    print("loss", loss)

if __name__ == "__main__":
    cnn_image = ImageCNN()

    x_true, y_true, x_false, y_false = load_images("C1a")
    x_data = x_true + random.choices(x_false, k=len(x_true))
    y_data = y_true + [0 for i in range(len(x_true))]
    x_data = torch.tensor(x_data)
    print(x_data.dtype)
    x_data = torch.tensor(x_data).double()
    # img_folder = torchvision.datasets.ImageFolder(root="C:/Users/jinno/Documents/portfolio/portfolio/droplet_tracker/training_data/C1a")
    # data_loader = torch.utils.data.DataLoader(img_folder)
    y_data = torch.tensor(y_data).double()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(cnn_image.parameters(), lr=0.00001)
    if torch.cuda.is_available():
        x_data = x_data.cuda()
        y_data = y_data.cuda()
        cnn_image = cnn_image.cuda()
        criterion = criterion.cuda()
        print("on cuda")
    # train
    x_data = x_data.float()
    print(len(x_data))
    normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # x_data = normalize(x_data)
    y_data = y_data.float()
    train_data = torch.utils.data.TensorDataset(x_data, y_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True)

    for i in range(1000):
        for (x_data, y_data) in train_loader:
            optimizer.zero_grad()
            outputs = cnn_image(x_data)
            y_data = y_data.unsqueeze(1)
            loss = criterion(outputs, y_data)
            loss.backward()
            optimizer.step()
            # print("Param", list(model.parameters())[0][0][0][0][0])
        print([float(i) for i in list(outputs)])
        print([float(i) for i in list(y_data)])
        # print(outputs)
        print("loss", loss)


    """for i in range(100):
        train(cnn_image, criterion, optimizer, i, train_loader)"""