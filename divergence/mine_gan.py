# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sns.set()

# %%
# x = np.array([(i, j) for i in range(-4, 5, 2) for j in range(-4, 5, 2)] * 10000)
x = np.array([(i, j) for i in range(-8, 9, 2) for j in range(-8, 9, 2)] * 10000)
x = x + 0.1 * np.random.randn(*x.shape)
plt.scatter(x[:, 0], x[:, 1], s=2.0)
plt.xlim((-10, 10))
plt.ylim((-10, 10))
plt.show()
print(x.shape)  # (250000, 2)

BATCH_SIZE = 256
DEVICE = "cuda"


# %%
class Generator(nn.Module):
    def __init__(self, input_size=10, output_size=2, hidden_size=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Discriminator(nn.Module):
    def __init__(self, input_size=2, output_size=1, hidden_size=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


# %%
class Mine(nn.Module):
    def __init__(self, noise_size=3, sample_size=2, output_size=1, hidden_size=128):
        super().__init__()
        self.fc1_noise = nn.Linear(noise_size, hidden_size, bias=False)
        self.fc1_sample = nn.Linear(sample_size, hidden_size, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.ma_et = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, noise, sample):
        x_noise = self.fc1_noise(noise)
        x_sample = self.fc1_sample(sample)
        x = F.relu(x_noise + x_sample + self.fc1_bias)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# %%
def update_target(ma_net, net, update_rate=1e-1):
    # update moving average network parameters using network
    for ma_net_param, net_param in zip(ma_net.parameters(), net.parameters()):
        ma_net_param.data.copy_((1.0 - update_rate) * ma_net_param.data + update_rate * net_param.data)


# %%
def learn_discriminator(x, G, D, M, D_opt, zero_gp=True):
    """
    real_samples : torch.Tensor
    G : Generator network
    D : Discriminator network
    M : Mutual Information Neural Estimation(MINE) network
    D_opt : Optimizer of Discriminator
    """
    z = torch.randn((BATCH_SIZE, 10))
    z = z.to(DEVICE)
    x_tilde = G(z)
    Dx_tilde = D(x_tilde)

    if zero_gp:
        # zero centered gradient penalty  : https://arxiv.org/abs/1801.04406
        x.requires_grad = True
        Dx = D(x)
        grad = torch.autograd.grad(
            Dx, x, create_graph=True, grad_outputs=torch.ones_like(Dx), retain_graph=True, only_inputs=True
        )[0].view(BATCH_SIZE, -1)
        grad = grad.norm(dim=1)
        gp_loss = torch.mean(grad**2)
    else:
        Dx = D(x)

    loss = 0.0
    gan_loss = -torch.mean(torch.log(Dx) + torch.log(1 - Dx_tilde))
    loss += gan_loss
    if zero_gp:
        loss = gan_loss + 1.0 * gp_loss

    D_opt.zero_grad()
    loss.backward()
    D_opt.step()

    if zero_gp:
        return gan_loss.item(), gp_loss.item()

    return gan_loss.item(), 0


def learn_generator(x, G, D, M, G_opt, G_ma, mi_obj=False):
    """
    real_samples : torch.Tensor
    G : Generator network
    D : Discriminator network
    M : Mutual Information Neural Estimation(MINE) network
    G_opt : Optimizer of Generator
    mi_reg : add Mutual information objective
    """
    z = torch.randn((BATCH_SIZE, 10))
    z_bar = torch.narrow(torch.randn((BATCH_SIZE, 10)), dim=1, start=0, length=3)
    z = z.to(DEVICE)
    z_bar = z_bar.to(DEVICE)
    x = x.to(DEVICE)

    x_tilde = G(z)
    Dx_tilde = D(x_tilde)
    # Dx = D(x)

    loss = 0.0
    gan_loss = -torch.mean(torch.log(Dx_tilde))
    loss += gan_loss
    if mi_obj:
        z = torch.narrow(z, dim=1, start=0, length=3)  # slice for MI
        mi = torch.mean(M(z, x_tilde)) - torch.log(torch.mean(torch.exp(M(z_bar, x_tilde))) + 1e-8)
        loss -= 0.01 * mi

    G_opt.zero_grad()
    loss.backward()
    G_opt.step()

    update_target(G_ma, G)
    return gan_loss.item()


def learn_mine(G, M, M_opt, ma_rate=0.001):
    """
    Mine is learning for MI of (input, output) of Generator.
    """
    z = torch.randn((BATCH_SIZE, 10))
    z_bar = torch.narrow(torch.randn((BATCH_SIZE, 10)), dim=1, start=0, length=3)
    z = z.to(DEVICE)
    z_bar = z_bar.to(DEVICE)  # shape: (BATCH_SIZE, 3)
    x_tilde = G(z)

    # Mutual Information Neural Estimation
    et = torch.mean(torch.exp(M(z_bar, x_tilde)))
    if M.ma_et is None:
        M.ma_et = et.detach().item()
    M.ma_et += ma_rate * (et.detach().item() - M.ma_et)
    z = torch.narrow(z, dim=1, start=0, length=3)  # slice for MI
    mutual_information = torch.mean(M(z, x_tilde)) - torch.log(et) * et.detach() / M.ma_et

    loss = -mutual_information
    M_opt.zero_grad()
    loss.backward()
    M_opt.step()

    return mutual_information.item()


# %%

G = Generator().to(DEVICE)
G_ma = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)
M = Mine().to(DEVICE)

G_ma.load_state_dict(G.state_dict())

G_opt = torch.optim.Adam(G.parameters(), lr=3e-4)
D_opt = torch.optim.Adam(D.parameters(), lr=3e-4)
M_opt = torch.optim.Adam(M.parameters(), lr=3e-4)
z_test = torch.randn((20000, 10))

z_test = torch.randn((20000, 10)).to(DEVICE)


# %%
def train(epoch=100, is_zero_gp=False, is_mi_obj=False):
    for i in range(1, epoch):
        np.random.shuffle(x)
        iter_num = len(x) // BATCH_SIZE
        d_loss_arr, gp_loss_arr, g_loss_arr, mi_arr = [], [], [], []
        for j in tqdm(range(iter_num)):
            batch = x[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]
            batch = torch.from_numpy(batch).float().to(DEVICE)
            d_loss, gp_loss = learn_discriminator(batch, G, D, M, D_opt, zero_gp=is_zero_gp)
            g_loss = learn_generator(batch, G, D, M, G_opt, G_ma, mi_obj=is_mi_obj)
            mi = learn_mine(G, M, M_opt)

            d_loss_arr.append(d_loss)
            gp_loss_arr.append(gp_loss)
            g_loss_arr.append(g_loss)
            mi_arr.append(mi)

        print(
            "D loss : {0}, GP_loss : {1} G_loss : {2}, MI : {3}".format(
                round(np.mean(d_loss_arr), 4),
                round(np.mean(gp_loss_arr)),
                round(np.mean(g_loss_arr), 4),
                round(np.mean(mi_arr), 4),
            )
        )
        if i % 10 == 0:
            x_test = G_ma(z_test).data.cpu().numpy()
            plt.title("Epoch {0}".format(i))
            plt.scatter(x_test[:, 0], x_test[:, 1], s=2.0)
            plt.xlim((-10, 10))
            plt.ylim((-10, 10))
            plt.show()


# %%
train(epoch=100, is_zero_gp=True, is_mi_obj=False)
# %%
train(epoch=100, is_zero_gp=True, is_mi_obj=True)
# %%
