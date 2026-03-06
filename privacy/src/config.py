import logging
import os
import json
from unicodedata import decimal

class Config:
    def __init__(self, jsonpath, job_name='data_privacy'):
        self.job_name = job_name
        self.jsonpath = jsonpath
        
        with open(jsonpath, 'r') as j:
            input_argument = json.loads(j.read())

        # base features
        self.filename = input_argument["input"][0]['filename']
        self.ignore_columns =  input_argument["input"][0]['ignore_columns']
        self.discrete_columns =  input_argument["input"][0]['discrete_columns']
        self.name_coumns =  input_argument["input"][0]['name_coumns']
        self.address_coumns =  input_argument["input"][0]['address_coumns']
        self.model_type = input_argument["input"][0]["model_type"]
        self.timers = input_argument["timers"][0]
        # common model param
        self.use_cuda = input_argument["common_model_param"][0]['use_cuda']
        self.verbose = input_argument["common_model_param"][0]['verbose']

        
        # set-up the parameters for tvae
        self.seed_tvae =input_argument["tvae_param"][0]['seed']
        self.embedding_dim_tvae = input_argument["tvae_param"][0]['embedding_dim']
        self.batch_size_tvae = input_argument["tvae_param"][0]['batch_size']
        self.epochs_tvae = input_argument["tvae_param"][0]['epochs']
        self.compress_dims_tvae = input_argument["tvae_param"][0]['compress_dims']
        self.decompress_dims_tvae = input_argument["tvae_param"][0]['decompress_dims']
        self.l2scale_tvae = input_argument["tvae_param"][0]['l2scale']
        self.loss_factor_tvae = input_argument["tvae_param"][0]['loss_factor']
        
        # set-up the parameters for ctgan
        self.seed =input_argument["ctgan_param"][0]['seed']
        self.embedding_dim = input_argument["ctgan_param"][0]['embedding_dim']
        self.batch_size = input_argument["ctgan_param"][0]['batch_size']
        self.epochs = input_argument["ctgan_param"][0]['epochs']
        self.sample_rate = input_argument["ctgan_param"][0]['sample_rate']
        self.sigma = input_argument["ctgan_param"][0]['sigma']
        self.max_per_sample_grad_norm = input_argument["ctgan_param"][0]['max_per_sample_grad_norm']
        self.generator_dim = input_argument["ctgan_param"][0]['generator_dim']
        self.discriminator_dim = input_argument["ctgan_param"][0]['discriminator_dim']
        self.generator_lr = input_argument["ctgan_param"][0]['generator_lr']
        self.generator_decay = input_argument["ctgan_param"][0]['generator_decay']
        self.discriminator_lr = input_argument["ctgan_param"][0]['discriminator_lr']
        self.discriminator_decay = input_argument["ctgan_param"][0]['discriminator_decay']
        self.discriminator_steps = input_argument["ctgan_param"][0]['discriminator_steps']
        self.log_frequency = input_argument["ctgan_param"][0]['log_frequency']
        self.pac = input_argument["ctgan_param"][0]['pac']
        self.optimizer = input_argument["ctgan_param"][0]['optimizer']
        self.dp_mode = input_argument["ctgan_param"][0]['dp_mode']
        

    def load_config(self, params, config_type=None, immutable_params=None):
        try:
            for key in params:
                if immutable_params is not None and key in immutable_params:
                    continue
                value = params[key]
                if not isinstance(value, list):
                    value = int(value) if str(value).isdigit() else value

                if config_type is None:
                    setattr(self, key, value)
                else:
                    prop = getattr(self, config_type)
                    prop[key] = value
                logging.debug("Config {key} updated with value: {value}".format(key=key, value=value))
            logging.debug(self.__dict__)
        except Exception as e:
            logging.error(e)

if __name__ == "__main__":
    # test code
    jsonpath = r"C:\Users\pp9596\Documents\Bitbucket\00_ai_coe\aace-prod-p-bitbucket-5\privacy\params.json" 
    confi = Config(jsonpath=jsonpath)
    confi.load_config((confi.__dict__))
    print("Parameters - ", confi.__dict__)
