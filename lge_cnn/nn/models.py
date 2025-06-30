from argparse import ArgumentParser

import torch
from lge_cnn.nn import *
import numpy as np
import pytorch_lightning as pl


"""
    (Non-equivariant) Activation functions
"""


def get_activation(a_type):
    if a_type == 'relu':
        return torch.nn.ReLU()
    elif a_type == 'leakyrelu':
        return torch.nn.LeakyReLU()
    elif a_type == 'tanh':
        return torch.nn.Tanh()
    elif a_type == 'sigmoid':
        return torch.nn.Sigmoid()
    else:
        print("Option {} for activation function unimplemented.".format(a_type))
        return None


"""
    L-CNN model definition using LConvBilin modules
"""


class LConvBilinNet(pl.core.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        # number of lattice sites
        self.sites = np.prod(hparams.dims)

        # validation loss for progress bar
        self.vloss = 0.0

        # torch module lists
        self.convs = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()

        # add convolutions and pooling layer
        if hasattr(hparams, 'symmetric'):
            use_symmetric = hparams.symmetric
        else:
            use_symmetric = False

        conv_ch = list(hparams.conv_channels)
        conv_ch.insert(0, 2 * len(hparams.dims) * (len(hparams.dims) - 1) // 2)
        for i in range(len(conv_ch) - 1):
            conv = LConvBilin(dims=hparams.dims,
                              kernel_size=hparams.conv_kernel_size[i],
                              dilation=hparams.conv_dilation[i],
                              n_in=conv_ch[i],
                              n_out=conv_ch[i + 1],
                              nc=hparams.nc,
                              init_w=hparams.init_weight_factor,
                              use_symmetric=use_symmetric)

            self.convs.append(conv)

        # tracing layer
        self.tr = LTrace(hparams.dims)

        # input size of first linear layer
        linear_sizes = list(hparams.linear_sizes)
        linear_sizes.insert(0, 2 * conv_ch[-1])

        # output size of last layer
        linear_sizes.append(1)

        # add linear layers
        for i in range(len(linear_sizes) - 1):
            linear = torch.nn.Linear(linear_sizes[i], linear_sizes[i + 1], bias=True)
            self.linears.append(linear)
            if i < len(linear_sizes) - 2:
                self.linears.append(get_activation(hparams.activation))

        self.out_mode = hparams.out_mode

        # output normalization
        if hasattr(hparams, 'output_norm'):
            output_norm = hparams.output_norm
        else:
            output_norm = 1.0

        # datasets
        self.train_dataset = YMDatasetHDF5(self.hparams.train_path,
                                           mode_in='uw', mode_out=self.out_mode, use_idx=False,
                                           output_normalization=output_norm)
        self.val_dataset = YMDatasetHDF5(self.hparams.val_path,
                                         mode_in='uw', mode_out=self.out_mode , use_idx=False,
                                         output_normalization=output_norm)
        self.test_dataset = YMDatasetHDF5(self.hparams.test_path,
                                          mode_in='uw', mode_out=self.out_mode, use_idx=False,
                                          output_normalization=output_norm)

        # check if dimensions match dataset
        for dataset in [self.train_dataset, self.val_dataset, self.test_dataset]:
            if isinstance(dataset, YMDatasetHDF5):
                if len(dataset.dims) != len(hparams.dims):
                    raise ValueError("Dimension mismatch between model ({}) and dataset ({})!".format(hparams.dims,
                                                                                                      dataset.dims))

                if (dataset.dims != hparams.dims).all():
                    raise ValueError("Dimension mismatch between model ({}) and dataset ({})!".format(hparams.dims,
                                                                                                      dataset.dims))
            else:
                print("Warning: cannot determine dimensions of dataset.")

    def forward(self, x):
        # store batch size
        batch_dim = x.shape[0]

        # apply GCMConv layers
        # x stays (batch_dim, lattice, channels, matrix structure)
        for i, conv in enumerate(self.convs):
            x = conv(x)

        # take trace
        # x becomes (batch_dim, lattice, channels)
        x = self.tr(x)

        x = x.view(batch_dim, self.sites, -1)
        # x = x[:, :, :, 0]

        # option to average over lattice sites
        if self.hparams.global_average:
            # x becomes (batch_dim, 2 * channels)
            x = torch.mean(x, dim=1)
        else:
            # x is (batch_dim, lattice sites, 2 * channels)
            # combine batch_dim and lattice sites into single dimension
            x = x.view(batch_dim * self.sites, -1)

        for i, layer in enumerate(self.linears):
            x = layer(x)

        # restore shape after linear layers
        if not self.hparams.global_average:
            x = x.view(batch_dim, self.sites, -1)

        return x

    """
        Prediction and testing
    """

    def evaluate(self, data='test', cuda=True):
        if data == 'train':
            dataset = self.train_dataset
            dataloader = self.train_dataloader()
        elif data == 'val':
            dataset = self.val_dataset
            dataloader = self.val_dataloader()
        elif data == 'test':
            dataset = self.test_dataset
            dataloader = self.test_dataloader()
        else:
            print("Unknown dataset.")
            return None

        X = []
        Y_pred = []
        Y_true = []

        dataset.use_idx = True

        with torch.no_grad():
            for x, y, idx in dataloader:
                beta = dataset.get_beta(idx)
                if cuda:
                    x = x.cuda()

                output = self(x)

                X.append(beta)
                Y_pred.append(output.detach().cpu().numpy())

                if self.hparams.global_average:
                    y = torch.mean(y, dim=1)
                y = y.view(output.shape)
                Y_true.append(y.detach().cpu().numpy())

        dataset.use_idx = False

        X = np.array(X).flatten()
        Y_pred = np.array(Y_pred)
        Y_true = np.array(Y_true)

        # combine batches
        num_batches = Y_pred.shape[0]
        batch_size = Y_pred.shape[1]
        rest_shape = Y_pred.shape[2:]

        Y_pred = Y_pred.reshape((num_batches * batch_size, *rest_shape))
        Y_true = Y_true.reshape((num_batches * batch_size, *rest_shape))

        return X, Y_pred, Y_true

    def mse(self, global_average=True, data='test', cuda=True):
        X, Y_pred, Y_true = self.evaluate(data, cuda)

        if global_average:
            Y_pred = np.mean(Y_pred, axis=1)
            Y_true = np.mean(Y_true, axis=1)

        mse_value = np.mean((Y_pred - Y_true).flatten() ** 2)

        return mse_value

    def update_dims(self, new_dims):
        if len(new_dims) != len(self.hparams.dims):
            raise ValueError("Cannot change lattice dimensions, only lattice size!")

        self.hparams.dims = new_dims
        self.sites = np.prod(new_dims)

        for conv in self.convs:
            conv.update_dims(new_dims)

        self.tr.update_dims(new_dims)

    """
        Additional methods
    """

    def close(self):
        self.train_dataset.close()
        self.val_dataset.close()
        self.test_dataset.close()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    """
        pytorch_lightning methods
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # architecture
        parser.add_argument('--dims', type=int, nargs='+')
        parser.add_argument('--nc', type=int, default=3)

        parser.add_argument('--global_average', action='store_true')
        parser.add_argument('--conv_channels', type=int, nargs='+')
        parser.add_argument('--conv_kernel_size', type=int, nargs='+')
        parser.add_argument('--conv_dilation', type=int, nargs='+')
        parser.add_argument('--init_weight_factor', type=float, default=1.0)
        parser.add_argument('--symmetric', action='store_true', default=False)

        parser.add_argument('--linear_sizes', type=int, nargs='*', default=[])
        parser.add_argument('--activation', type=str, default='leakyrelu')

        # datasets
        parser.add_argument('--train_path', type=str)
        parser.add_argument('--val_path', type=str)
        parser.add_argument('--test_path', type=str)
        parser.add_argument('--out_mode', type=str)
        parser.add_argument('--output_norm', type=float, default=1.0)
        parser.add_argument('--num_workers', type=int, default=0)

        # optimizer (AMSGradW)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=10)
        parser.add_argument('--amsgrad', action='store_true', default=True)

        # lr scheduler (CosineAnnealingWarmRestarts)
        parser.add_argument('--use_scheduling', action='store_true')
        parser.add_argument('--T_0', type=int, default=10)
        parser.add_argument('--T_mult', type=int, default=1)
        parser.add_argument('--eta_min', type=float, default=0)

        return parser

    def get_progress_bar_dict(self):

        # call .item() only once but store elements without graphs
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        lr = self._scheduler.get_lr()[0] if self._scheduler is not None else self.hparams.lr

        tqdm_dict = {
            'loss': '{:.2E}'.format(avg_training_loss),
            'val_loss': '{:.2E}'.format(self.vloss),
            'lr': '{:.2E}'.format(lr)
        }

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict['split_idx'] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            tqdm_dict['v_num'] = self.trainer.logger.version

        return tqdm_dict

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                            weight_decay=self.hparams.weight_decay, amsgrad=self.hparams.amsgrad)

        return_dict = {'optimizer': self._optimizer}

        self._scheduler = None

        if self.hparams.use_scheduling:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._optimizer,
                T_0=self.hparams.T_0,
                T_mult=self.hparams.T_mult,
                eta_min=self.hparams.eta_min,
                last_epoch=-1
            )

        if self._scheduler is not None:
            return_dict['lr_scheduler'] = self._scheduler

        return return_dict

    def loss(self, x, y):
        output = self(x).flatten()

        if self.hparams.global_average:
            y = torch.mean(y, dim=1)

        y = y.flatten()

        loss = torch.nn.functional.mse_loss(output, y)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        logs = {'loss': loss}

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        self.vloss = avg_loss

        return {'avg_val_loss': avg_loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        return {'val_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


"""
    Baseline CNN model
"""


class BaselineNet(pl.core.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # output normalization (optional)
        if hasattr(hparams, 'output_norm'):
            self.output_norm = hparams.output_norm
        else:
            self.output_norm = 1.0

        # input mode
        if hasattr(hparams, 'in_mode'):
            self.mode_in = hparams.in_mode
        else:
            self.mode_in = 'uw'

        # output mode
        self.out_mode = hparams.out_mode

        # number of lattice sites
        self.sites = np.prod(hparams.dims)
        self.dims = hparams.dims

        # validation loss for progress bar
        self.vloss = 0.0

        # torch module lists
        self.convs = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()

        # add convolutions
        # channel multiplication factor
        d = len(hparams.dims)
        if self.mode_in == 'uw':
            # links (U), plaquettes (W), daggered plaquettes (W^t)
            self.nch_factor = 2 * hparams.nc ** 2 * (d + 2 * d * (d-1) // 2)
        elif self.mode_in == 'uw_legacy':
            # links (U) and plaquettes (W)
            self.nch_factor = 2 * hparams.nc ** 2 * (d + d * (d-1) // 2)
        elif self.mode_in == 'u':
            # links (U)
            self.nch_factor = 2 * hparams.nc ** 2 * d
        else:
            raise NotImplementedError("Unknown in_mode {}".format(self.mode_in))

        conv_ch = list(hparams.conv_channels)
        conv_ch.insert(0, self.nch_factor)
        for i in range(len(conv_ch) - 1):
            conv = CConv2d(in_channels=conv_ch[i],
                           out_channels=conv_ch[i + 1],
                           kernel_size=hparams.conv_kernel_size[i],
                           bias=True)

            self.convs.append(conv)
            self.convs.append(get_activation(hparams.activation))

        # input size of first linear layer
        linear_sizes = list(hparams.linear_sizes)
        linear_sizes.insert(0, conv_ch[-1])

        # output size of last layer
        linear_sizes.append(1)

        # add linear layers
        for i in range(len(linear_sizes) - 1):
            linear = torch.nn.Linear(linear_sizes[i], linear_sizes[i + 1], bias=True)
            self.linears.append(linear)
            if i < len(linear_sizes) - 2:
                self.linears.append(get_activation(hparams.activation))


        # datasets
        self.train_dataset = YMDatasetHDF5(self.hparams.train_path,
                                           mode_in=self.mode_in, mode_out=self.out_mode, use_idx=False,
                                           output_normalization=self.output_norm)
        self.val_dataset = YMDatasetHDF5(self.hparams.val_path,
                                         mode_in=self.mode_in, mode_out=self.out_mode, use_idx=False,
                                         output_normalization=self.output_norm)
        self.test_dataset = YMDatasetHDF5(self.hparams.test_path,
                                          mode_in=self.mode_in, mode_out=self.out_mode, use_idx=False,
                                          output_normalization=self.output_norm)

    def forward(self, x):
        # store batch size
        batch_dim = x.shape[0]

        # change shape
        x = x.view(batch_dim, *self.dims,  self.nch_factor)
        x = x.permute(0, 3, 1, 2)

        # apply CNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x)

        # option to average over lattice sites
        if self.hparams.global_average:
            x = torch.mean(x, dim=[2, 3])
            x = x.view(batch_dim, -1)
        else:
            # combine batch_dim and lattice sites into single dimension
            x = x.view(batch_dim * self.sites, -1)

        for i, layer in enumerate(self.linears):
            x = layer(x)

        # restore shape after linear layers
        if not self.hparams.global_average:
            x = x.view(batch_dim, self.sites, -1)

        return x

    """
        Prediction and testing
    """

    def evaluate(self, data='test', cuda=True):
        if data == 'train':
            dataset = self.train_dataset
            dataloader = self.train_dataloader()
        elif data == 'val':
            dataset = self.val_dataset
            dataloader = self.val_dataloader()
        elif data == 'test':
            dataset = self.test_dataset
            dataloader = self.test_dataloader()
        else:
            print("Unknown dataset.")
            return None

        X = []
        Y_pred = []
        Y_true = []

        dataset.use_idx = True

        with torch.no_grad():
            for x, y, idx in dataloader:
                beta = dataset.get_beta(idx)
                if cuda:
                    x = x.cuda()

                output = self(x)

                X.append(beta)
                Y_pred.append(output.detach().cpu().numpy())

                if self.hparams.global_average:
                    y = torch.mean(y, dim=1)
                y = y.view(output.shape)
                Y_true.append(y.detach().cpu().numpy())

        dataset.use_idx = False

        X = np.array(X).flatten()
        Y_pred = np.array(Y_pred)
        Y_true = np.array(Y_true)

        # combine batches
        num_batches = Y_pred.shape[0]
        batch_size = Y_pred.shape[1]
        rest_shape = Y_pred.shape[2:]

        Y_pred = Y_pred.reshape((num_batches * batch_size, *rest_shape))
        Y_true = Y_true.reshape((num_batches * batch_size, *rest_shape))

        return X, Y_pred, Y_true

    def mse(self, global_average=True, data='test', cuda=True):
        X, Y_pred, Y_true = self.evaluate(data, cuda)

        if global_average:
            Y_pred = np.mean(Y_pred, axis=1)
            Y_true = np.mean(Y_true, axis=1)

        mse_value = np.mean((Y_pred - Y_true).flatten() ** 2)

        return mse_value

    def update_dims(self, new_dims):
        if len(new_dims) != len(self.hparams.dims):
            raise ValueError("Cannot change lattice dimensions, only lattice size!")

        self.hparams.dims = new_dims
        self.dims = new_dims
        self.sites = np.prod(new_dims)

    """
        Additional methods
    """

    def close(self):
        self.train_dataset.close()
        self.val_dataset.close()
        self.test_dataset.close()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    """
        pytorch_lightning methods
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # architecture
        parser.add_argument('--dims', type=int, nargs='+')
        parser.add_argument('--nc', type=int, default=3)

        parser.add_argument('--global_average', action='store_true', default=True)
        parser.add_argument('--conv_channels', type=int, nargs='+')
        parser.add_argument('--conv_kernel_size', type=int, nargs='+')

        parser.add_argument('--linear_sizes', type=int, nargs='*', default=[])
        parser.add_argument('--activation', type=str)

        # datasets
        parser.add_argument('--train_path', type=str)
        parser.add_argument('--val_path', type=str)
        parser.add_argument('--test_path', type=str)
        parser.add_argument('--out_mode', type=str)
        parser.add_argument('--in_mode', type=str, default='uw')

        # optimizer (AMSGradW)
        parser.add_argument('--lr', type=float, default=3e-2)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=50)
        parser.add_argument('--amsgrad', action='store_true', default=True)

        # lr scheduler (CosineAnnealingWarmRestarts)
        parser.add_argument('--use_scheduling', action='store_true', default=False)
        parser.add_argument('--T_0', type=int, default=10)
        parser.add_argument('--T_mult', type=int, default=1)
        parser.add_argument('--eta_min', type=float, default=0)

        return parser

    def get_progress_bar_dict(self):
        # call .item() only once but store elements without graphs
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        lr = self._scheduler.get_lr()[0] if self._scheduler is not None else self.hparams.lr

        tqdm_dict = {
            'loss': '{:.2E}'.format(avg_training_loss),
            'val_loss': '{:.2E}'.format(self.vloss),
            'lr': '{:.2E}'.format(lr)
        }

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict['split_idx'] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            tqdm_dict['v_num'] = self.trainer.logger.version

        return tqdm_dict

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=True, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=0)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=0)

    def configure_optimizers(self):
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                            weight_decay=self.hparams.weight_decay, amsgrad=self.hparams.amsgrad)

        return_dict = {'optimizer': self._optimizer}

        self._scheduler = None

        if self.hparams.use_scheduling:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._optimizer,
                T_0=self.hparams.T_0,
                T_mult=self.hparams.T_mult,
                eta_min=self.hparams.eta_min,
                last_epoch=-1
            )

        if self._scheduler is not None:
            return_dict['lr_scheduler'] = self._scheduler

        return return_dict

    def loss(self, x, y):
        output = self(x).flatten()

        if self.hparams.global_average:
            y = torch.mean(y, dim=1)

        y = y.flatten()

        loss = torch.nn.functional.mse_loss(output, y)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        logs = {'loss': loss}

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        self.vloss = avg_loss

        return {'avg_val_loss': avg_loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        return {'val_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


"""
    New LCNN model definiton using separate LConv and LBilin
"""


class LCNN(pl.core.LightningModule):
    def __init__(self, hparams):
        super(LCNN, self).__init__()

        self.hparams = hparams

        # number of lattice sites
        self.sites = np.prod(hparams.dims)

        # validation loss for progress bar
        self.vloss = 0.0

        # L1 regularization
        if hasattr(hparams, 'L1'):
            self.L1 = hparams.L1
        else:
            self.L1 = 0.0

        # torch module lists
        self.convs = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()

        conv_inter_ch = list(hparams.conv_inter_channels)
        conv_ch = list(hparams.conv_channels)
        conv_ch.insert(0, 2 * len(hparams.dims) * (len(hparams.dims) - 1) // 2)
        for i in range(len(conv_ch) - 1):
            conv = LConvBilin2(dims=hparams.dims,
                               kernel_size=hparams.conv_kernel_size[i],
                               dilation=hparams.conv_dilation[i],
                               n_in=conv_ch[i],
                               n_inter=conv_inter_ch[i],
                               n_out=conv_ch[i + 1],
                               nc=hparams.nc,
                               use_unit_elements=True,
                               extended=False)

            self.convs.append(conv)

        # tracing layer
        self.tr = LTrace(hparams.dims)

        # input size of first linear layer
        linear_sizes = list(hparams.linear_sizes)
        linear_sizes.insert(0, 2 * conv_ch[-1])

        # output size of last layer
        linear_sizes.append(1)

        # add linear layers
        for i in range(len(linear_sizes) - 1):
            linear = torch.nn.Linear(linear_sizes[i], linear_sizes[i + 1], bias=True)
            self.linears.append(linear)
            if i < len(linear_sizes) - 2:
                self.linears.append(get_activation(hparams.activation))

        self.out_mode = hparams.out_mode

        # datasets
        self.train_dataset = YMDatasetHDF5(self.hparams.train_path,
                                           mode_in='uw', mode_out=self.out_mode, use_idx=False)
        self.val_dataset = YMDatasetHDF5(self.hparams.val_path,
                                         mode_in='uw', mode_out=self.out_mode , use_idx=False)
        self.test_dataset = YMDatasetHDF5(self.hparams.test_path,
                                          mode_in='uw', mode_out=self.out_mode, use_idx=False)

    def forward(self, x):
        # store batch size
        batch_dim = x.shape[0]

        # apply GCMConv layers
        # x stays (batch_dim, lattice, channels, matrix structure)
        for i, conv in enumerate(self.convs):
            x = conv(x)

        # take trace
        # x becomes (batch_dim, lattice, channels)
        x = self.tr(x)

        x = x.view(batch_dim, self.sites, -1)
        # x = x[:, :, :, 0]

        # option to average over lattice sites
        if self.hparams.global_average:
            # x becomes (batch_dim, 2 * channels)
            x = torch.mean(x, dim=1)
        else:
            # x is (batch_dim, lattice sites, 2 * channels)
            # combine batch_dim and lattice sites into single dimension
            x = x.view(batch_dim * self.sites, -1)

        for i, layer in enumerate(self.linears):
            x = layer(x)

        # restore shape after linear layers
        if not self.hparams.global_average:
            x = x.view(batch_dim, self.sites, -1)

        return x

    """
        Prediction and testing
    """

    def evaluate(self, data='test', cuda=True):
        if data == 'train':
            dataset = self.train_dataset
            dataloader = self.train_dataloader()
        elif data == 'val':
            dataset = self.val_dataset
            dataloader = self.val_dataloader()
        elif data == 'test':
            dataset = self.test_dataset
            dataloader = self.test_dataloader()
        else:
            print("Unknown dataset.")
            return None

        X = []
        Y_pred = []
        Y_true = []

        dataset.use_idx = True

        with torch.no_grad():
            for x, y, idx in dataloader:
                beta = dataset.get_beta(idx)
                if cuda:
                    x = x.cuda()

                output = self(x)

                X.append(beta)
                Y_pred.append(output.detach().cpu().numpy())

                if self.hparams.global_average:
                    y = torch.mean(y, dim=1)
                y = y.view(output.shape)
                Y_true.append(y.detach().cpu().numpy())

        dataset.use_idx = False

        X = np.array(X).flatten()
        Y_pred = np.array(Y_pred)
        Y_true = np.array(Y_true)

        # combine batches
        num_batches = Y_pred.shape[0]
        batch_size = Y_pred.shape[1]
        rest_shape = Y_pred.shape[2:]

        Y_pred = Y_pred.reshape((num_batches * batch_size, *rest_shape))
        Y_true = Y_true.reshape((num_batches * batch_size, *rest_shape))

        return X, Y_pred, Y_true

    def mse(self, global_average=True, data='test', cuda=True):
        X, Y_pred, Y_true = self.evaluate(data, cuda)

        if global_average:
            Y_pred = np.mean(Y_pred, axis=1)
            Y_true = np.mean(Y_true, axis=1)

        mse_value = np.mean((Y_pred - Y_true).flatten() ** 2)

        return mse_value

    def update_dims(self, new_dims):
        if len(new_dims) != len(self.hparams.dims):
            raise ValueError("Cannot change lattice dimensions, only lattice size!")

        self.hparams.dims = new_dims
        self.sites = np.prod(new_dims)

        for conv in self.convs:
            conv.update_dims(new_dims)

        self.tr.update_dims(new_dims)

    """
        Additional methods
    """

    def close(self):
        self.train_dataset.close()
        self.val_dataset.close()
        self.test_dataset.close()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    """
        pytorch_lightning methods
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # architecture
        parser.add_argument('--dims', type=int, nargs='+')
        parser.add_argument('--nc', type=int, default=3)

        parser.add_argument('--global_average', action='store_true')
        parser.add_argument('--conv_channels', type=int, nargs='+')
        parser.add_argument('--conv_inter_channels', type=int, nargs='+')
        parser.add_argument('--conv_kernel_size', type=int, nargs='+')
        parser.add_argument('--conv_dilation', type=int, nargs='+')

        parser.add_argument('--linear_sizes', type=int, nargs='*', default=[])
        parser.add_argument('--activation', type=str, default='leakyrelu')

        # datasets
        parser.add_argument('--train_path', type=str)
        parser.add_argument('--val_path', type=str)
        parser.add_argument('--test_path', type=str)
        parser.add_argument('--out_mode', type=str)
        parser.add_argument('--num_workers', type=int, default=0)

        # optimizer (AMSGradW)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=10)
        parser.add_argument('--amsgrad', action='store_true', default=True)

        # lr scheduler (CosineAnnealingWarmRestarts)
        parser.add_argument('--use_scheduling', action='store_true')
        parser.add_argument('--T_0', type=int, default=10)
        parser.add_argument('--T_mult', type=int, default=1)
        parser.add_argument('--eta_min', type=float, default=0)

        return parser

    def get_progress_bar_dict(self):

        # call .item() only once but store elements without graphs
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        lr = self._scheduler.get_lr()[0] if self._scheduler is not None else self.hparams.lr

        tqdm_dict = {
            'loss': '{:.2E}'.format(avg_training_loss),
            'val_loss': '{:.2E}'.format(self.vloss),
            'lr': '{:.2E}'.format(lr)
        }

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict['split_idx'] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            tqdm_dict['v_num'] = self.trainer.logger.version

        return tqdm_dict

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_dataset,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)

    def configure_optimizers(self):
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                            weight_decay=self.hparams.weight_decay, amsgrad=self.hparams.amsgrad)

        return_dict = {'optimizer': self._optimizer}

        self._scheduler = None

        if self.hparams.use_scheduling:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._optimizer,
                T_0=self.hparams.T_0,
                T_mult=self.hparams.T_mult,
                eta_min=self.hparams.eta_min,
                last_epoch=-1
            )

        if self._scheduler is not None:
            return_dict['lr_scheduler'] = self._scheduler

        return return_dict

    def loss(self, x, y):
        output = self(x).flatten()

        if self.hparams.global_average:
            y = torch.mean(y, dim=1)

        y = y.flatten()

        loss = torch.nn.functional.mse_loss(output, y)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        if self.L1 != 0.0:
            # Add L1 regularization
            L1_reg = torch.tensor(0., requires_grad=True).cuda()
            for name, param in self.named_parameters():
                if 'weight' in name:
                    L1_reg = L1_reg + torch.norm(param, 1)
            loss = loss + self.L1 * L1_reg

        logs = {'loss': loss}

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        self.vloss = avg_loss

        return {'avg_val_loss': avg_loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)

        return {'val_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    
    
    
