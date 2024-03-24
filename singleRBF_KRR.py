import torch
import torch.nn as nn

class RBFRegressor(nn.Module):
    def __init__(self, x_train, y_train, sigma=1, lamda=0.1):
        super(RBFRegressor, self).__init__()

        self.num_samples = x_train.shape[1]
        self.feature_dim = x_train.shape[0]
        self.lamda = torch.tensor(lamda).float()
        self.alpha = torch.ones(1, self.num_samples)
        self.beta = torch.tensor(0.0)  # y_train.mean()
        self.x_train = x_train
        self.y_train = y_train.reshape(-1)
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight.data = torch.tensor(sigma).float() / self.x_train.shape[0]
        self.train_ker_norm = 0.0
        self.device = self.x_train.device

        # calculate the initial inner-product
        ele0 = torch.pow(torch.linalg.norm(self.x_train, dim=0),2)
        ele1 = torch.matmul(torch.ones(self.num_samples, 1, device=self.device, dtype=torch. float), ele0.view(1, self.num_samples))
        ele2 = -2*torch.matmul(torch.transpose(self.x_train, 0, 1), self.x_train)
        self.train_gram = ele1 + ele2 + torch.transpose(ele1,0,1)

    def forward(self, x_val):
        assert x_val.shape[0] == self.x_train.shape[0], 'but found {} and {}'.format(x_val.shape[0],
                                                                                     self.x_train.shape[0])
        device = x_val.device
        N_val = x_val.shape[1]
        N = self.num_samples
        M = self.x_train.shape[0]

         # N, N
        x_train_kernel = torch.exp(-1.0 * self.weight * self.train_gram)

        # calculate the initial inner-product of the val data
        ele0 =torch.pow(torch.linalg.norm(self.x_train,dim=0),2)
        ele1 = torch.matmul(torch.ones(N_val, 1, device=self.device, dtype=torch. float), ele0.view(1, self.num_samples))
        ele2 = -2*torch.matmul(torch.transpose(x_val, 0, 1), self.x_train)
        ele4 = torch.pow(torch.linalg.norm(x_val,dim=0),2)
        ele3 = torch.matmul(ele4.view(N_val,1), torch.ones(1, self.num_samples, device=self.device, dtype=torch. float))
        x_val_kernel = torch.transpose(torch.exp(-1.0 * self.weight * (ele1+ele2+ele3)), 0, 1)  # N, N_val

        ele1 = self.y_train.reshape(1, N) - self.beta.repeat(1, N)  # y_mean
        ele2 = 1 * x_train_kernel + self.lamda * torch.eye(N, N, device=device, dtype=torch. float)
        # self.alpha = torch.linalg.solve(ele2, ele1.T).T   # 1, N
        self.alpha = torch.matmul(ele1, torch.inverse(ele2)) * 1
        self.train_ker_norm = torch.matmul(torch.matmul(self.alpha, x_train_kernel), torch.transpose(self.alpha, 1, 0))
        self.alpha_norm = torch.norm(self.alpha)
        y_pred_val = torch.matmul(self.alpha, x_val_kernel) + self.beta  # (1, N)

        return y_pred_val.reshape(-1)

    @staticmethod
    def mad_loss(pred, target):
        # mean absolute deviation
        loss = (torch.abs(pred.reshape(-1) - target.reshape(-1))).sum() / target.shape[0]
        return loss

    @staticmethod
    def rsse_loss(pred, target):
        # RSSE
        tmp = ((target.reshape(-1) - target.mean()) ** 2).sum()
        loss = ((pred.reshape(-1) - target.reshape(-1)) ** 2).sum() / tmp
        return loss

    @staticmethod
    def mse_loss(pred, target):
        # MSE
        loss = ((pred.reshape(-1) - target.reshape(-1)) ** 2).sum() / target.shape[0]
        return loss

    @staticmethod
    def rmse_loss(pred, target):
        # RMSE
        loss = ((pred.reshape(-1) - target.reshape(-1)) ** 2).sum() / target.shape[0]
        loss = torch.sqrt(loss)
        return loss

    def norm_print(self):
        print("train kernel norm: ", format(self.train_ker_norm))
        print("alpha norm:  ", format(self.alpha_norm))
        print("weight norm:", format(self.weight.norm()))



