import torch
import torch.nn as nn

def Flatten():
    def forward(self, x):
        return torch.flatten(x, star_dim=1)

def stn(nn.Module):
    
    def __init__(self, num_agfs):
        self.num_agfs = num_agfs
        
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout(p=0.1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout(p=0.1)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ELU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(num_features=256),
            nn.ELU()
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.BatchNorm1d(num_features=256)
        )
        self.densefc = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ELU(),
            nn.Dropout(p=0.5)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=256, out_features=self.num_afgs)
        )
        
        
    def forward(self, x):
        
        #Convolution
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        
        #Global Average Pooling
        out = self.global_avg_pool(out)
        
        #Dense Layer
        out = self.densefc(out)
        out = self.output_layer(out)
        
        return out
        
        
    '''
    def forward_intermediate(self, x):
        
        #Convolution
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        
        #Global Average Pooling
        out_pool = self.global_avg_pool(out7)
        
        #Dense Layer
        out = self.densefc(out_pool)
        out = self.output_layer(out)
        
        return out1, out2, out3, out4, out5, out6, out7, out_pool, out 
    '''
        