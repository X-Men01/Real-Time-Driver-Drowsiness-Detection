from torch import nn
import torch

class Custom_CNN(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super(Custom_CNN, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,stride=1,padding=1,),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1,),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),)
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units * 2,kernel_size=3,stride=1,padding=1,),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units * 2,out_channels=hidden_units * 2,kernel_size=3,stride=1,padding=1,),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 2,out_channels=hidden_units * 4,kernel_size=3,stride=1,padding=1,),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units * 4,out_channels=hidden_units * 4,kernel_size=3,stride=1,padding=1,),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 4,out_channels=hidden_units * 4,kernel_size=3,stride=1,padding=1,),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units * 4,out_channels=hidden_units * 4,kernel_size=3,stride=1,padding=1,),
            nn.BatchNorm2d(hidden_units * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), )

        self.classifier = nn.Sequential(

            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_units *8*8 *4, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor):
        # print(x.shape)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        # print(x.shape)
        x = self.classifier(x)
        return x