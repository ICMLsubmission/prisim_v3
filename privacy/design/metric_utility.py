import sys
import os
import scipy
import math
import pandas as pd
import numpy as np
from numpy import trace
from scipy import stats
from numpy import cov
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



def standardize_data(data):
    stand_data = data.copy(deep=True)
    for col in list(data.columns):
        col_vals = np.array(data[col].values)
        col_vals = col_vals.reshape(-1,1)
        scaler = MinMaxScaler(feature_range= (0, 1))
        scaler.fit(col_vals)
        col_vals = scaler.transform(col_vals)
        col_vals = col_vals.squeeze(1)
        stand_data[col] = col_vals
    return stand_data

def compute_gqi(real_data, gen_data):  
        corr_G=gen_data.corr(method='pearson')
        corr_R=real_data.corr(method='pearson')

        df_R = corr_R.where(np.triu(np.ones(corr_R.shape)).astype(np.bool)) 
        df_R = df_R.stack().reset_index()
        df_R.columns = ['Feature1_R','Feature2_R','Corr_R']

        df_G = corr_G.where(np.triu(np.ones(corr_G.shape)).astype(np.bool))  
        df_G = df_G.stack().reset_index()
        df_G.columns = ['Feature1_G','Feature2_G','Corr_G']

        df_RG=pd.concat([df_R,df_G],axis=1)
        df_RG.drop(['Feature1_R','Feature2_R','Feature1_G','Feature2_G'], axis=1,inplace=True)
        val = 1-abs(np.mean(df_RG.diff(axis=1).Corr_G.values))
        return val

class metricsCalculator():
    def __init__(self, real_data, gen_data):
        self.real_data = real_data
        self.gen_data = gen_data

    def compute_JS(self):
        """
        Computes the Jensen-Shannon distance between each respective features of the real and generated datasets
        Args:
            real_data (DataFrame):
                the real data file in a dataframe.
            gen_data (DataFrame):
                the generated data file in a dataframe.
        Returns:
            feat_wise_dist (list):
                feature-wise JS distance (len = #features)
            avg_js_dist (float):
                avg of the feature-wise JS distances
        """
        feat_wise_dist = list(distance.jensenshannon(self.real_data.values, self.gen_data.values, axis=0))
        f = [d for d in feat_wise_dist if not math.isnan(d) and not math.isinf(d)]
        avg_js = sum(f)/len(f)
        return feat_wise_dist, avg_js



    def compute_fid(self, image=False):
        """
        Computes the Frechet Inception Distance between real and generated datasets
        Args:
            image (bool):
                specify whether the provided dataset is tabular data or image embeddings
            real_data (DataFrame):
                a) the real data file in a dataframe in case of tabular data
                b) the real image embeddings as np array in case of image data
            gen_data (DataFrame):
                a) the generated data file in a dataframe in case of tabular data
                b) the generated image embeddings as np array in case of image data
        Returns:
            fid (float):
                the computed Frechet Inception Distance between the 2 datasets
        """
        if not image:
            real_data = standardize_data(self.real_data).values
            gen_data = standardize_data(self.gen_data).values

        # calculate mean and covariance statistics
        mu1, sigma1 = real_data.mean(axis=0), cov(real_data, rowvar=False)
        mu2, sigma2 = gen_data.mean(axis=0), cov(gen_data, rowvar=False)

        ssdiff = np.sum((mu1 - mu2)**2.0) # calculate sum squared difference between means
        covmean = sqrtm(sigma1.dot(sigma2)) # calculate sqrt of product between cov
        
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real	
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean) # calculate score
        return fid
    def compute_a_precision_b_recall(self, alpha_beta, discreteFeatures=[], metric_type=None):
        data = self.real_data.sample(frac=alpha_beta)
        # print(alpha_beta)
        # _, data = train_test_split(self.real_data, test_size=alpha_beta, stratify=self.real_data[discreteFeatures])
        centerpoint = np.mean(data , axis=0)
        projection_data = self.gen_data
        if metric_type=="beta_recall":
            data = self.gen_data.sample(frac=alpha_beta)
            # _, data = train_test_split(self.gen_data, test_size=alpha_beta, stratify=self.gen_data[discreteFeatures])
            centerpoint = np.mean(data , axis=0)
            projection_data = self.real_data
        covariance  = np.cov(data , rowvar=False)
        # print(metric_type)
        # print(covariance)
        try:
            covariance_pm1 = np.linalg.matrix_power(covariance, -1)
        except:
            covariance_pm1 = np.linalg.pinv(covariance)
        distances=[]
        for i,val in enumerate(projection_data.values):
            p1 = val
            p2 = centerpoint
            distance = (p1-p2).T.dot(covariance_pm1).dot(p1-p2)
        distances.append(distance)
        return(np.mean(1-stats.chi2.cdf(distances, data.shape[1]))) #chi2


# test cases
# if __name__=='__main__':
#     real_data = pd.read_csv('../data/HR_pre_processed_data.csv').iloc[:, 1:]
#     # gen_data = pd.read_csv('../data/tvae_hr_gen_5k.csv').iloc[:, 1:]
#     gen_data = pd.read_csv('../data/private.csv').iloc[:, 1:]
#     gen_data = gen_data.sample(n = len(real_data))

#     #################### Test JS-distance implementation ###################
#     metrics_calc = Metrics_calculator(real_data, real_data)
#     feat_wise_dist, avg_js_dist = metrics_calc.compute_JS()
#     print("avg JS dist (with self) =  ", avg_js_dist) # This should come out to be 0 for the test to pass

#     metrics_calc = Metrics_calculator(real_data, gen_data)
#     feat_wise_dist, avg_js_dist = metrics_calc.compute_JS()
#     print("avg JS dist (b/w real and generated) =  ", avg_js_dist)

#     #################### Test Frechet Inception-distance implementation (tabular) ###################
#     metrics_calc = Metrics_calculator(real_data, real_data)
#     fid = metrics_calc.compute_fid()
#     print("\nFrechet Inception-distance (with self) = ", fid) # This should come out to be ~~0 for the test to pass

#     metrics_calc = Metrics_calculator(real_data, gen_data)
#     fid = metrics_calc.compute_fid()
#     print("Frechet Inception-distance (b/w real and generated) = ", fid) 