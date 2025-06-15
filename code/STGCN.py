import torch
import torch.nn as nn


class STGCN(nn.Module):

    def __init__(self, in_channels, num_class, A, edge_importance_weighting=False, dropout=0.5, kernel_size=9):
        super(STGCN, self).__init__()
        self.A = A.float()  # ensure A is float32
        padding = (kernel_size - 1) // 2

        # Layer 1 (64 channels)
        self.gcn1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.tcn1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(kernel_size, 1), padding=(padding, 0)),
            nn.BatchNorm2d(64)
        )
        # Replace Identity with a projection from in_channels to 64 channels:
        self.res1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.drop1 = nn.Dropout(dropout)

        # Layer 2 (64 channels)
        self.gcn2 = nn.Conv2d(64, 64, kernel_size=1)
        self.tcn2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(kernel_size, 1), padding=(padding, 0)),
            nn.BatchNorm2d(64)
        )
        self.res2 = nn.Identity()
        self.drop2 = nn.Dropout(dropout)

        # Layer 3 (64 channels)
        self.gcn3 = nn.Conv2d(64, 64, kernel_size=1)
        self.tcn3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(kernel_size, 1), padding=(padding, 0)),
            nn.BatchNorm2d(64)
        )
        self.res3 = nn.Identity()
        self.drop3 = nn.Dropout(dropout)

        # Layer 4 (128 channels, stride=2)
        self.gcn4 = nn.Conv2d(64, 128, kernel_size=1)
        self.tcn4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(kernel_size, 1), stride=(2, 1), padding=(padding, 0)),
            nn.BatchNorm2d(128)
        )
        # Residual branch for layer 4 projects from 64 to 128 channels with stride 2 on time dimension:
        self.res4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=(2, 1)),
            nn.BatchNorm2d(128)
        )
        self.drop4 = nn.Dropout(dropout)

        # Layer 5 (128 channels)
        self.gcn5 = nn.Conv2d(128, 128, kernel_size=1)
        self.tcn5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(kernel_size, 1), padding=(padding, 0)),
            nn.BatchNorm2d(128)
        )
        self.res5 = nn.Identity()
        self.drop5 = nn.Dropout(dropout)

        # Layer 6 (128 channels)
        self.gcn6 = nn.Conv2d(128, 128, kernel_size=1)
        self.tcn6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(kernel_size, 1), padding=(padding, 0)),
            nn.BatchNorm2d(128)
        )
        self.res6 = nn.Identity()
        self.drop6 = nn.Dropout(dropout)

        # Layer 7 (256 channels, stride=2)
        self.gcn7 = nn.Conv2d(128, 256, kernel_size=1)
        self.tcn7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(kernel_size, 1), stride=(2, 1), padding=(padding, 0)),
            nn.BatchNorm2d(256)
        )
        self.res7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=(2, 1)),
            nn.BatchNorm2d(256)
        )
        self.drop7 = nn.Dropout(dropout)

        # Layer 8 (256 channels)
        self.gcn8 = nn.Conv2d(256, 256, kernel_size=1)
        self.tcn8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(kernel_size, 1), padding=(padding, 0)),
            nn.BatchNorm2d(256)
        )
        self.res8 = nn.Identity()
        self.drop8 = nn.Dropout(dropout)

        # Layer 9 (256 channels)
        self.gcn9 = nn.Conv2d(256, 256, kernel_size=1)
        self.tcn9 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(kernel_size, 1), padding=(padding, 0)),
            nn.BatchNorm2d(256)
        )
        self.res9 = nn.Identity()
        self.drop9 = nn.Dropout(dropout)

        # Edge importance weighting
        if edge_importance_weighting:
            self.w1 = nn.Parameter(torch.ones_like(self.A))
            self.w2 = nn.Parameter(torch.ones_like(self.A))
            self.w3 = nn.Parameter(torch.ones_like(self.A))
            self.w4 = nn.Parameter(torch.ones_like(self.A))
            self.w5 = nn.Parameter(torch.ones_like(self.A))
            self.w6 = nn.Parameter(torch.ones_like(self.A))
            self.w7 = nn.Parameter(torch.ones_like(self.A))
            self.w8 = nn.Parameter(torch.ones_like(self.A))
            self.w9 = nn.Parameter(torch.ones_like(self.A))
        else:
            self.w1 = self.w2 = self.w3 = self.w4 = self.w5 = self.w6 = self.w7 = self.w8 = self.w9 = 1

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print("X shape",x.shape)
        if x.dim() == 3:
            x = x.unsqueeze(2)
        # print("X shape",x.shape)
        # Layer 1
        A1 = self.A * self.w1
        x1 = torch.einsum('nctv,vw->nctw', x, A1)
        # After gcn1 and tcn1, x1 is shape [N, 64, T, V]
        x1 = self.relu(self.tcn1(self.gcn1(x1)) + self.res1(x))
        x1 = self.drop1(x1)

        # Layer 2
        A2 = self.A * self.w2
        x2 = torch.einsum('nctv,vw->nctw', x1, A2)
        x2 = self.relu(self.tcn2(self.gcn2(x2)) + self.res2(x1))
        x2 = self.drop2(x2)

        # Layer 3
        A3 = self.A * self.w3
        x3 = torch.einsum('nctv,vw->nctw', x2, A3)
        x3 = self.relu(self.tcn3(self.gcn3(x3)) + self.res3(x2))
        x3 = self.drop3(x3)

        # Layer 4
        A4 = self.A * self.w4
        x4 = torch.einsum('nctv,vw->nctw', x3, A4)
        x4 = self.relu(self.tcn4(self.gcn4(x4)) + self.res4(x3))
        x4 = self.drop4(x4)

        # Layer 5
        A5 = self.A * self.w5
        x5 = torch.einsum('nctv,vw->nctw', x4, A5)
        x5 = self.relu(self.tcn5(self.gcn5(x5)) + self.res5(x4))
        x5 = self.drop5(x5)

        # Layer 6
        A6 = self.A * self.w6
        x6 = torch.einsum('nctv,vw->nctw', x5, A6)
        x6 = self.relu(self.tcn6(self.gcn6(x6)) + self.res6(x5))
        x6 = self.drop6(x6)

        # Layer 7
        A7 = self.A * self.w7
        x7 = torch.einsum('nctv,vw->nctw', x6, A7)
        x7 = self.relu(self.tcn7(self.gcn7(x7)) + self.res7(x6))
        x7 = self.drop7(x7)

        # Layer 8
        A8 = self.A * self.w8
        x8 = torch.einsum('nctv,vw->nctw', x7, A8)
        x8 = self.relu(self.tcn8(self.gcn8(x8)) + self.res8(x7))
        x8 = self.drop8(x8)

        # Layer 9
        A9 = self.A * self.w9
        x9 = torch.einsum('nctv,vw->nctw', x8, A9)
        x9 = self.relu(self.tcn9(self.gcn9(x9)) + self.res9(x8))
        x9 = self.drop9(x9)

        # Global pooling and classifier
        out = self.pool(x9).view(x9.size(0), -1)
        out = self.fc(out)
        # out = self.softmax(out)
        return out
