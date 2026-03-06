from __future__ import print_function, division
import random
import torch
import numpy as np
from matplotlib import rc
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from .config import Config

# from .synthesizers.ctgan_dp import CTGANSynthesizer
from .custom_models.synthesizers.ctgan import CTGANSynthesizer
from .custom_models.synthesizers.tvae import TVAESynthesizer
from sdv.tabular import GaussianCopula

# from faker import Faker
# from tqdm.auto import tqdm


def normalization(gen, real, cont_columns):
    for cols in cont_columns:
        gen = gen[gen[cols] >= min(real[cols])]
        gen = gen[gen[cols] <= max(real[cols])]
    return gen


def set_seed(seed):
    # Seeds for reproduceable runs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class trainModel(Config):
    def __init__(
        self,
        jsonpath,
    ):
        # extend config class
        super().__init__(jsonpath=jsonpath)
        if torch.cuda.is_available():
            self.use_cuda = True
        # set-up the seed for training
        set_seed(self.seed)

    def build(
        self,
        discrete_columns,
        name_columns,
        address_columns,
        model_type="Gaussian Copula",
        **kwargs
    ):
        self.discrete_columns = discrete_columns
        self.synthetic_data = None
        self.name_columns = name_columns
        self.address_columns = address_columns
        self.model_type = model_type
        if model_type == "TVAE":
            # print(kwargs, model_type)
            if len(kwargs) > 0:
                kwargs = kwargs["kwargs"]
                self.batch_size_tvae = kwargs["batch_size"]
                self.epochs_tvae = kwargs["epochs"]
                # print(self.epochs_tvae, self.batch_size)
            self.model = TVAESynthesizer(
                embedding_dim=self.embedding_dim_tvae,
                compress_dims=self.compress_dims_tvae,
                decompress_dims=self.decompress_dims_tvae,
                l2scale=self.l2scale_tvae,
                batch_size=self.batch_size_tvae,
                epochs=self.epochs_tvae,
                loss_factor=self.loss_factor_tvae,
                cuda=self.use_cuda,
            )
        elif model_type == "Gaussian Copula":
            # print(kwargs, model_type)
            self.model = GaussianCopula()
        else:
            # print(kwargs, model_type)
            if len(kwargs) > 0:
                kwargs = kwargs["kwargs"]
                self.batch_size = kwargs["batch_size"]
                self.epochs = kwargs["epochs"]
                self.generator_lr = kwargs["gen_lr"]
                self.discriminator_lr = kwargs["dis_lr"]
                # print(self.batch_size, self.epochs, self.generator_lr, self.discriminator_lr)
            self.model = CTGANSynthesizer(
                embedding_dim=self.embedding_dim,
                generator_dim=self.generator_dim,
                discriminator_dim=self.discriminator_dim,
                generator_lr=self.generator_lr,
                generator_decay=self.generator_decay,
                discriminator_lr=self.discriminator_lr,
                discriminator_decay=self.discriminator_decay,
                batch_size=self.batch_size,
                discriminator_steps=self.discriminator_steps,
                log_frequency=self.log_frequency,
                verbose=self.verbose,
                epochs=self.epochs,
                pac=self.pac,
                cuda=self.use_cuda,
            )

    def fit(self, data):
        self.data = data.copy()
        if self.model_type == "TVAE" or self.model_type == "Gaussian Copula":
            history = self.model.fit(data)
        else:
            history = self.model.fit(data, self.discrete_columns)
        return history

    def generate(self, samples, seed=123):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        samples *= 6
        syntheticData = self.model.sample(samples)
        cont_columns = [
            i for i in syntheticData.columns.values if i not in self.discrete_columns
        ]
        self.syntheticData = normalization(
            syntheticData, self.data, self.data.columns.values
        )
        return self.syntheticData
