import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    
class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten()
        )
    
    def forward(self, x):
        return self.layer(x)

class STN(nn.Module):
    
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
            GlobalAvgPool(),
            nn.BatchNorm1d(num_features=256)
        )
        self.densefc = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ELU(),
            nn.Dropout(p=0.5)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=256, out_features=self.num_agfs)
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
        
        
    def forward_all(self, x):
        #GlobalAvgPool for intermediate layers
        gap = GlobalAvgPool()
        
        #Convolution
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        
#         #Global Average Pooling
#         out_pool = self.global_avg_pool(out7)
        
#         #Dense Layer
#         out = self.densefc(out_pool)
#         out = self.output_layer(out)
        
        return [gap(out1), gap(out2), gap(out3), gap(out4), gap(out5), gap(out6), gap(out7)]
    
    def forward_intermediate(self, x, layer_no):
        assert 1 <= layer_no <= 7 and isinstance(layer_no, int)
        
        #layers
        layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]
        
        #GlobalAvgPool for intermediate layers
        gap = GlobalAvgPool()
        
        for i in range(layer_no):
            x = layers[i](x)
        
        return gap(x)
    
class MLP(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        
        self.fc = nn.Sequential(
                    
                    nn.Linear(self.input_dims, out_features=1024),
                    nn.ELU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(in_features=1024, out_features=self.output_dims)
        
        )
        
    def forward(self, x):
        return self.fc(x)
        