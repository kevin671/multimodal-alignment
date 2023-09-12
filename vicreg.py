import torch


class Expander(torch.nn.Module):
    #  The expanders have 3 fully-connected layers of size 8192
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 8192)
        self.fc2 = torch.nn.Linear(8192, 8192)
        self.fc3 = torch.nn.Linear(8192, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class VICReg(torch.nn.Module):
    """

     Examples:
    >>> loss = InfoNCE()
    >>> batch_size, num_negative, embedding_size = 32, 48, 128
    >>> query = torch.randn(batch_size, embedding_size)
    >>> positive_key = torch.randn(batch_size, embedding_size)
    >>> negative_keys = torch.randn(num_negative, embedding_size)
    >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, lambda_=1.0, mu=1.0, nu=1.0):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.lambda_ = lambda_
        self.mu = mu
        self.nu = nu

    def forward(self, z_a, z_b):
        """
        Args:
            Given a batch of images, representations Y_a and Y_b are computed by the encoder f_a and f_b.
            The representations are fed to an expander h_a and h_b, producing the embeddings Z_a and Z_b.
            z_a: N x D
            z_b: N x D
        """

        # invariance loss
        sim_loss = self.mse_loss(z_a, z_b)

        # variance loss
        std_z_a = torch.sqrt(torch.var(z_a, dim=0) + 1e-4)
        std_z_b = torch.sqrt(torch.var(z_b, dim=0) + 1e-4)
        std_loss = torch.mean(torch.relu(1 - std_z_a)) + torch.mean(torch.relu(1 - std_z_b))

        # covariance loss
        z_a = z_a - torch.mean(z_a, dim=0)
        z_b = z_b - torch.mean(z_b, dim=0)
        cov_z_a = torch.matmul(z_a.T, z_a) / (z_a.shape[0] - 1)
        cov_z_b = torch.matmul(z_b.T, z_b) / (z_b.shape[0] - 1)
        off_diagonal_za = cov_z_a - torch.diag(torch.diag(cov_z_a))  # D x D
        off_diagonal_zb = cov_z_b - torch.diag(torch.diag(cov_z_b))  # D x D
        conv_loss = torch.mean(torch.pow(off_diagonal_za, 2)) + torch.mean(torch.pow(off_diagonal_zb, 2))

        loss = self.lambda_ * sim_loss + self.mu * std_loss + self.nu * conv_loss
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        return loss
