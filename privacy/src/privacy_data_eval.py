import numpy as np
import plotly.graph_objects as go
from tqdm.auto import tqdm
from math import ceil
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots
from .helper import *


def normalization(gen, real, columns):
    for cols in columns:
        gen = gen[gen[cols] >= min(real[cols])]
        gen = gen[gen[cols] <= max(real[cols])]
    return gen


# def privacy_chart(X, vmu, mlu):
#     # pio.renderers.default = 'browser'
#     # pio.renderers.default = 'iframe_connected'
#     # fig = go.Figure()
#     fig = make_subplots(rows=1, cols=2, subplot_titles = ("Privacy Chart", "Utility Chart"))
#     fig.add_trace(go.Scatter(x=X, y=vmu,
#                         mode='lines+markers',
#                         name="Privacy", line=dict(color='orange')), row=1, col=1)

#     fig.add_trace(go.Scatter(x=X, y=mlu,
#                         mode='lines+markers',
#                         name="ML Utility", line=dict(color='green')), row=1, col=2)


#     fig['layout']['xaxis']['title'] = fig['layout']['xaxis2']['title'] ='Scenarios'
#     fig['layout']['yaxis']['title'] = 'Privacy'
#     fig['layout']['yaxis2']['title'] = 'Utility'

#     fig.update_layout(hovermode='x')
#     fig.update_layout(height=450, width=900)
#     fig.show()


class vulnarabilitymitigation:
    def __init__(
        self,
        data=None,
        discrete_columns=None,
        name_columns=None,
        address_columns=None,
        discrete_label_encode=None,
        n_samples=500,
        pretrained_model=None,
        filename_prefix=None,
    ):
        self.data = data
        self.syntheticData = pretrained_model.generate(n_samples * 6)
        self.discreteColumns = discrete_columns
        self.privateData = None
        self.vm_data = None
        self.vm_data_address_name = None
        self.addressColumns = address_columns
        self.nameColumns = name_columns
        self.private_data_at_diff_iter = {}
        self.private_metric_at_diff_iter = {}
        self.discreteLabelEncode = discrete_label_encode
        # self.targetColumn = None
        self.n_samples = n_samples
        self.filename_prefix = filename_prefix

    def privacyEvaluation(self):
        assert self.data is not None, "Dataset is not defined"
        assert self.syntheticData is not None, "Synthetic Data is not defined"
        assert self.discreteColumns is not None, "Discrete features are not defined"
        pbar = tqdm(total=100)
        pbar.update(10)
        NP_samples = self.syntheticData.copy(deep=True)
        vmu = []
        alu = []
        # mlu = []
        scene_cnt = 1
        distances = eval_privacy(self.data, NP_samples)["distances"]
        cutoff = np.percentile(distances, 50)
        vm_data = NP_samples.copy(deep=True)
        # y_o = self.data[self.targetColumn]
        Xfeatures = list(self.data.columns)
        if "states_countries" in Xfeatures:
            Xfeatures.remove("states_countries")
        # Xfeatures.remove(self.targetColumn)
        X_o = self.data[Xfeatures]
        # regressor = True
        # if self.targetColumn in self.discreteColumns:
        #     regressor = False
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X_o, y_o, test_size=1 / 3, random_state=0
        # )
        # _, ml_score_O = ml_training(
        #     X_train, X_test, y_train, y_test, regressor=regressor
        # )
        privacy_metric_max = None
        discard_percentages = [50, 65, 75, 85, 90, 95, 97, 99, 100]
        tolerance_power = [50]
        updation_pbar_value = 80 // len(tolerance_power)
        # feature_importance, feature_order = get_frufs_feature_importance(
        #     self.data, self.discreteColumns
        # )
        for i in tolerance_power:
            pbar.update(updation_pbar_value)
            if privacy_metric_max is not None and privacy_metric_max == 100:
                continue
            response = eval_privacy_latest(self.data, vm_data, discard_percent=i)
            privacy_metric, distances, cutoff = (
                response["Metric"],
                response["distances"],
                response["cutoff"],
            )
            sampled_privacy_data = vm_data.sample(self.n_samples, random_state=1)
            # au_score = analytical_utility_score(sampled_privacy_data, self.discreteColumns, feature_importance, feature_order)
            au_score = round(
                analytical_utility_score(
                    self.data[Xfeatures], sampled_privacy_data[Xfeatures]
                )
                * 100,
                2,
            )
            # print(au_score)
            # y_p = sampled_privacy_data[self.targetColumn]
            # X_p = sampled_privacy_data[Xfeatures]
            # ml_score = private_data__ml_training(
            #     X_o, X_p, y_o, y_p, regressor=regressor
            # )  # private_data__ml_training(X_o, X_p, y_o, y_p, regressor=regressor)
            # # mlu_val = ceil((ml_score / ml_score_O) * 100)
            vmu.append(max(ceil(privacy_metric), 0))
            # mlu.append(max(ceil(mlu_val), 0))
            alu.append(max(au_score, 0))
            vm_data = vm_data #privacy_mitigate(vm_data, distances, cutoff)
            self.private_data_at_diff_iter[
                "Scenario " + str(scene_cnt)
            ] = sampled_privacy_data.copy(deep=True)
            self.private_metric_at_diff_iter["Scenario " + str(scene_cnt)] = [
                max(ceil(privacy_metric), 0),
                # max(ceil(mlu_val), 0),
                max(au_score, 0),
            ]
            scene_cnt += 1
            privacy_metric_max = ceil(privacy_metric)
        self.vmu = vmu
        self.alu = alu
        # self.mlu = mlu
        pbar.update(100 - (updation_pbar_value * len(tolerance_power)) - 10)
        pbar.close()

    def privacyGeneration(self, threshold, n_samples, filepath):
        self.privateData = None
        self.vmData = None
        self.vm_data = self.syntheticData.copy(deep=True)
        # fetching scores for original data & private data
        # print(self.private_metric_at_diff_iter)
        values = self.private_metric_at_diff_iter['Scenario 1']
        # print(values)
        privacy_score, au_privacy_score = (values[0], values[1])
        # print("Private Data Saved in your local system")
        if len(self.vm_data) > n_samples:
            # print("lenght is initialiing in  privcy generation")
            # self.vm_data = self.vm_data[:n_samples]
            self.vm_data = self.vm_data.sample(
                n=n_samples, random_state=123
            )  # [:n_samples]
        self.vm_data = normalization(
            self.vm_data.copy(), self.data, self.data.columns.values
        )
        self.vm_data_encoded = self.vm_data.copy()
        if self.discreteLabelEncode is not None:
            for i in self.discreteLabelEncode.keys():
                self.vm_data[i] = self.discreteLabelEncode[i].inverse_transform(
                    self.vm_data[i].values
                )
        if self.addressColumns is not None or self.nameColumns is not None:
            # self.vm_data_address_name, self.privateStateCountries = add_name_address(self.vm_data.copy(deep=True), "saved_files/{}privateData_{}_{}.csv".format(self.filename_prefix,threshold, time.strftime("%Y%m%d-%H%M%S")))
            self.vm_data_address_name = add_name_address(
                self.vm_data.copy(deep=True), filepath
            )
            return (
                self.vm_data_address_name,
                self.vm_data_encoded,
                privacy_score,
                au_privacy_score,
            )
        else:
            # self.vm_data.to_csv("saved_files/privateData_{}_{}.csv".format(threshold, time.strftime("%Y%m%d-%H%M%S")),index=False)
            # print("saved_files/privateData.csv file saved in your local directory")
            self.vm_data.to_csv(filepath, index=False)
            return (
                self.vm_data,
                self.vm_data_encoded,
                privacy_score,
                au_privacy_score,
            )
