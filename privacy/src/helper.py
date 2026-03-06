from calendar import weekday
from cmath import isnan
import scipy
import hashlib
import pgeocode
import numpy as np
import pandas as pd
import lightgbm as ltb
from .FRUFS import FRUFS

# from FRUFS import FRUFS
from faker import Faker
from numpy import trace
from scipy import stats
from tqdm.auto import tqdm
from numpy import cov
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, f1_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN

def categoricalSuppression(original_data, feature_categroical_noise_bin):
    data = original_data.copy()
    # for feature in feature_categroical_noise_bin.keys():
    #     auto_bin = feature_categroical_noise_bin[feature]["auto"]
    #     if auto_bin != None and auto_bin > 0:
    #         series = pd.value_counts(data[feature])
    #         threshold_percent = series.values[-1 * auto_bin] / series.sum() * 100 + 1
    #         mask = (series / series.sum() * 100).lt(threshold_percent)
    #         data[feature] = np.where(
    #             data[feature].isin(series[mask].index), "Other", data[feature]
    #         )
    for i in feature_categroical_noise_bin:
        data[i] = data[i].astype(str)
        for j in range(len(feature_categroical_noise_bin[i]["group_name"])):
            bins_keys = list(feature_categroical_noise_bin[i]["bins"].keys())
            bins = map(str, feature_categroical_noise_bin[i]["bins"][bins_keys[j]])
            data[i] = np.where(data[i].isin(bins), feature_categroical_noise_bin[i]["group_name"][j], data[i])
    return data


def numericalBinning(original_data, feature_numerical_noise_bin):
    data = original_data.copy()
    for feature in feature_numerical_noise_bin.keys():
        noise = feature_numerical_noise_bin[feature]["noise"]
        auto_bin = list(set(np.ravel(feature_numerical_noise_bin[feature]["bins"])))
        auto_bin.sort()
        custom_bin = []#feature_numerical_noise_bin[feature]["custom"]
        if noise > 0:
            # Add Laplace Noise
            noise /= 10
            loc = np.median(data[feature].values)
            std = data[feature].values.std()
            scale = std / loc

            data[feature] = data[feature].values + np.random.laplace(
                loc / std, noise * (std * 0.1), data.shape[0]
            )
            data[feature] = data[feature].apply(lambda x: round(x, 2))

        if len(auto_bin) > 0:
            if auto_bin[0] < data[feature].min():
                auto_bin[0] = data[feature].min()
            if auto_bin[-1] > data[feature].max():
                auto_bin[-1] = data[feature].max()
            data.rename(columns={feature: "noisy_feature"}, inplace=True)
            data["noisy_feature_binned"] = pd.cut(data["noisy_feature"], bins=auto_bin, include_lowest=True)
            smart_bin = {}
            c = 0
            for i in data.groupby("noisy_feature_binned").mean().index:
                smart_bin[i] = (
                    data.groupby("noisy_feature_binned").mean().noisy_feature.values[c]
                )
                c += 1
            data[feature] = data["noisy_feature_binned"]
            data.drop(["noisy_feature", "noisy_feature_binned"], axis=1, inplace=True)
            # data = data.dropna()
            data[feature] = data[feature].apply(
                lambda x: f"{int(x.left)} - {int(x.right)}"
            )
        elif len(custom_bin) > 1:
            custom_bin.sort()
            minimum_value = data[feature].min()
            maximum_value = data[feature].max()
            if custom_bin[0] > minimum_value:
                custom_bin += [minimum_value]
            if custom_bin[-1] < maximum_value:
                custom_bin.append(minimum_value)
            data.rename(columns={feature: "noisy_feature"}, inplace=True)
            data["noisy_feature_binned"] = pd.cut(
                data["noisy_feature"], bins=custom_bin, include_lowest=True
            )
            smart_bin = {}
            c = 0
            for i in data.groupby("noisy_feature_binned").mean().index:
                smart_bin[i] = (
                    data.groupby("noisy_feature_binned").mean().noisy_feature.values[c]
                )
                c += 1
            data[feature] = data["noisy_feature_binned"]
            data.drop(["noisy_feature", "noisy_feature_binned"], axis=1, inplace=True)
            # data = data.dropna()
            data[feature] = data[feature].apply(
                lambda x: f"{int(x.left)} - {int(x.right)}"
            )
    return data[original_data.columns.values]

def date_zip_noise_binning(data, features_noise_bin):
    for i in features_noise_bin.keys():
        # data[i] = pd.to_datetime(data[i],  errors="coerce")
        if features_noise_bin[i]["noise"] != 0 :
            data[i] = pd.to_datetime(data[i])
            noise_value = features_noise_bin[i]["noise"]
            data[i] += pd.DateOffset(noise_value)# pd.to_timedelta(list(map(lambda x: int(x), noise_value)), unit="d")
            # data[i] += pd.to_timedelta(list(map(lambda x: int(x), np.random.uniform(-noise_value, noise_value, data.shape[0]))), unit="d")
        elif features_noise_bin[i]["aggregation"] is not None:
            if features_noise_bin[i]["aggregation"] == "Weekday-Weekend":
                for idx in range(len(data[i].values)):
                    if pd.isna(data[i][idx]):
                        data[i][idx] = np.nan
                    elif data[i][idx].weekday()>=5:
                        data[i][idx] = "Weekend"
                    else:
                        data[i][idx] = "Weekday"

                # data[i]=data[i].apply(lambda x: "Weekend" if x.weekday()>=5 else "Weekday")
            elif features_noise_bin[i]["aggregation"] == "Week of the year":
                for idx in range(len(data[i].values)):
                    try:
                        data[i][idx] =  "Week " + str(data[i][idx].isocalendar().week) + " of " + str(data[i][idx].year)
                    except:
                        data[i][idx] = np.nan
                # data[i] = data[i].map(lambda x : "Week " + str(x.isocalendar().week) + " of " + str(x.year))
            elif features_noise_bin[i]["aggregation"] == "Month":
                data[i] = data[i].map(lambda x : str(x.month) + "-" + str(x.year))
            elif features_noise_bin[i]["aggregation"] == "Quarter":
                data[i] = data[i].map(lambda x : str(x.quarter) + "-" + str(x.year))
            elif features_noise_bin[i]["aggregation"] == "Year":
                data[i] = data[i].dt.year
            elif features_noise_bin[i]["aggregation"] == "County":
                nomi = pgeocode.Nominatim('us')
                data[i] = nomi.query_postal_code(data[i].astype(str).values).county_name
            elif features_noise_bin[i]["aggregation"] == "State":
                nomi = pgeocode.Nominatim('us')
                data[i] = nomi.query_postal_code(data[i].astype(str).values).state_name
    return data
            
def string_hashing(data, features_hash):
    for i in features_hash.keys():
        if features_hash[i]["hashing"]:
            data[i] = data[i].apply(lambda x: hashlib.sha1(repr(x).encode('utf-8')).hexdigest())
        elif features_hash[i]["masking"]:
            data[i] = data[i].apply(lambda x: str(x)[:3]+'XX' )
    return data
def outlier_detection(data):
    try:
        X = StandardScaler().fit_transform(data.values)
        db = DBSCAN(eps=3.0, min_samples=100).fit(X)
        labels = db.labels_
        data['outlier_flag']=labels
        num_outliers=data[data['outlier_flag']==-1].count()[0]
        return {"error":False, "value":round(num_outliers/data.shape[0]*100,2), "data":data}
    except:
        return {"error": True, "value":"No numerical or categorial feature found in the data!"}

def eval_privacy(original_data, private_data, cutoff=0):
    distances = []
    if len(original_data)>5000:
        ary = scipy.spatial.distance.cdist(
            original_data, private_data, metric="minkowski"
        )
    else:
        ary = scipy.spatial.distance.cdist(
            original_data, private_data, metric="mahalanobis"
        )
    distances = ary.min(axis=0)
    risky_indexes = np.where(distances < cutoff)  # GREATER
    privacy = (1 - len(risky_indexes[0]) / private_data.shape[0]) * 100
    return {"Metric": round(privacy, 2), "distances": distances}


def eval_privacy_latest(original_data, private_data, discard_percent, cutoff=None):
    if len(original_data)>5000:
        ary = scipy.spatial.distance.cdist(
            original_data, private_data, metric="minkowski"
        )
    else:
        ary = scipy.spatial.distance.cdist(
            original_data, private_data, metric="mahalanobis"
        )
    distances = ary.min(axis=0)
    if cutoff is not None:
        cutoff = cutoff
    else:
        cutoff = np.percentile(distances, discard_percent)
    risky_indexes = np.where(distances < cutoff)
    privacy = (1 - len(risky_indexes[0]) / private_data.shape[0]) * 100
    return {"Metric": round(privacy, 2), "distances": distances, "cutoff": cutoff}


def hybrid_dp(
    original_data,
    synthetic_data,
    noise_injected_data,
    dataset_type,
    feature_baseline_injected_noise_bins,
):
    data, privacy_value, fid_value = (None, None, None)
    # get static bin intervals from originald data
    for i in feature_baseline_injected_noise_bins.keys():
        feature = i
        noise = feature_baseline_injected_noise_bins[i]["noise"]
        bin_num = feature_baseline_injected_noise_bins[i]["bins"]
        if noise <= 0 or bin_num == "--Select Bin--":
            continue
        _, bin_points = pd.cut(noise_injected_data[feature], bins=bin_num, retbins=True, include_lowest=True)

        # create copy of the data
        data = noise_injected_data.copy(deep=True)

        # Add Laplace Noise
        loc = np.median(data[feature].values)
        std = data[feature].values.std()
        scale = std / loc

        data[feature] = data[feature].values + np.random.laplace(
            loc / std, noise * (std * 0.1), data.shape[0]
        )  # Add Lalplace noise
        data.rename(columns={feature: "noisy_feature"}, inplace=True)

        # smart binning

        data["noisy_feature_binned"] = pd.cut(
            data["noisy_feature"], bins=bin_points, include_lowest=True
        )  # use static bin points

        smart_bin = {}
        c = 0
        for i in data.groupby("noisy_feature_binned").mean().index:
            smart_bin[i] = (
                data.groupby("noisy_feature_binned").mean().noisy_feature.values[c]
            )
            c += 1

        data = data.replace({"noisy_feature_binned": smart_bin})
        data["noisy_feature_binned"] = pd.to_numeric(data["noisy_feature_binned"])
        data[feature] = data["noisy_feature_binned"]
        data.drop(["noisy_feature", "noisy_feature_binned"], axis=1, inplace=True)
        data = data.dropna()
        data = data[list(noise_injected_data.columns)]
        noise_injected_data = data.copy()
    distances = eval_privacy(original_data, synthetic_data, cutoff=0)["distances"]
    cutoff_flp = np.percentile(distances, 50)
    if dataset_type != "Original Data":
        projected_data = synthetic_data
    else:
        projected_data = original_data
    privacy_value = int(
        eval_privacy_latest(projected_data, noise_injected_data, None, cutoff_flp)[
            "Metric"
        ]
    )
    fid_value = int(
        np.ceil(analytical_utility_score(projected_data, noise_injected_data) * 100)
    )
    return (data, privacy_value, fid_value)


def privacy_mitigate(syn_private_data, distances, cutoff):
    private_data = syn_private_data.copy(deep=True)
    private_data["JP"] = distances
    private_data["vul"] = private_data["JP"].apply(lambda x: 1 if x < cutoff else 0)
    private_data = private_data[private_data["vul"] == 0].iloc[
        :, :-2
    ]  # vulnerable samples remvoed
    return private_data


def add_name_address(data, path_to_save=None):
    data[["States", "Countries"]] = data["states_countries"].str.split(
        "_", 1, expand=True
    )
    address = []
    names = []
    cols = ["Name"]
    cols.extend(data.columns)
    cols.append("Address")
    faker = Faker(locale="en_US")
    for i in tqdm(range(len(data))):
        names.append(faker.name())
        # address.append(faker.address())
        location = faker.address()
        location = location.split(",") if "," in location else location.split(" ")
        location.insert(-1, data["States"].iloc[i])
        location.insert(-1, data["Countries"].iloc[i])
        address.append(",".join(location))
    data["Name"] = names
    data["Address"] = address
    data = data[cols]
    if "states_countries" in list(data.columns.values):
        data.drop(["states_countries", "States", "Countries"], inplace=True, axis=1)
    if path_to_save is not None:
        data.to_csv(path_to_save, index=False)
    return data


def private_data__ml_training(X_o, X_p, y_o, y_p, regressor=True):
    if regressor:
        model = RandomForestRegressor(random_state=0, n_estimators=100)
        model.fit(X_p, y_p)
    else:
        # model = RandomForestClassifier(n_estimators=100, random_state=0)
        # X_o = StandardScaler().fit_transform(X_o)
        # X_p = StandardScaler().fit_transform(X_p)
        # le = LabelEncoder()
        y_o = LabelEncoder().fit_transform(y_o)
        y_p = LabelEncoder().fit_transform(y_p)
        model = XGBClassifier(eval_metric='mlogloss')
        model.fit(X_p, y_p)
    CV_scores = []
    start = 0
    for fold in range(1, 10):
        end = start + int(len(X_o) / 10)
        X_test_real = X_o.iloc[start:end]
        # y_test_real = y_p.iloc[start:end]
        # X_test_real = X_o[start:end]
        y_test_real = y_o[start:end]
        y_pred = model.predict(X_test_real)
        if regressor:
            score = r2_score(y_test_real, y_pred)
        else:
            score = round(f1_score(y_test_real, y_pred, average="macro"), 3)
        CV_scores.append(score)
        start = end
    return np.mean(CV_scores)


def ml_training(X_train, X_test, y_train, y_test, random_state=1, regressor=True):
    # function trains MLP regression model on input argument data
    if not regressor:
        # classifier = MLPClassifier(random_state=1,hidden_layer_sizes=(100,10), max_iter=200)
        # classifier = RandomForestClassifier(n_estimators=100, random_state=0)
        # X_train = StandardScaler().fit_transform(X_train)
        # X_test = StandardScaler().fit_transform(X_test)
        y_train = LabelEncoder().fit_transform(y_train)
        y_test = LabelEncoder().fit_transform(y_test)
        classifier = XGBClassifier(eval_metric='mlogloss')
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        f1 = round(f1_score(y_test, y_pred, average="macro"), 3)
        return classifier, f1
    else:
        # regr = MLPRegressor(random_state=1,hidden_layer_sizes=(100,10), max_iter=200, early_stopping=True)
        regr = RandomForestRegressor(random_state=0, n_estimators=100)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        r_score = round(r2_score(y_test, y_pred), 3)
        return regr, r_score


def get_feature_importance(X, y, regressor=True):
    # function to get feature importance on dataset passed as a parameter in this function
    if not regressor:
        classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
        classifier.fit(X, y)
        important_features = classifier.feature_importances_
        feat_importances = pd.Series(important_features, index=X.columns).sort_values()
        return feat_importances
    else:
        regr = RandomForestRegressor(random_state=0, n_estimators=1000)
        regr.fit(X, y)
        important_features = regr.feature_importances_
        feat_importances = pd.Series(important_features, index=X.columns).sort_values()
        return feat_importances


def get_frufs_feature_importance(originalData, discreteColumns):
    model_frufs = FRUFS(
        model_r=ltb.LGBMRegressor(),
        model_c=ltb.LGBMClassifier(class_weight="balanced"),
        categorical_features=discreteColumns,
        k=0.5,
        n_jobs=-1,
        verbose=1,
    )
    _ = model_frufs.fit_transform(originalData)
    org_FI = model_frufs.feat_imps_
    feature_order = model_frufs.columns_
    return org_FI, feature_order


def analytical_utility_score(
    original_data,
    private_data,
    cat_cols=None,
    feature_importance=None,
    feature_order=None,
):
    # data = data.corr().values
    # m, n = data.shape
    # values = []
    # for i in range(m):
    #     for j in range(n):
    #         if (i<j):
    #             values.append(data[i][j])
    # val = round(np.mean(np.abs(values)),2)
    # val = round(data.corr().abs().mean().mean(),2)
    # --------FRUFS Code--------
    # model_frufs = FRUFS(model_r=ltb.LGBMRegressor(),
    #                     model_c=ltb.LGBMClassifier( class_weight="balanced"),
    #                     categorical_features=cat_cols,k=0.5, n_jobs=-1, verbose=1)
    # _ = model_frufs.fit_transform(data)
    # #feature importance dictiornary
    # feat_name_importance={}
    # c=0
    # for col in model_frufs.columns_:
    #     feat_name_importance[col]=model_frufs.feat_imps_[c]
    #     c+=1
    # #feature importance in current data according to importance on original data
    # data_FI=[]
    # for k in feature_order.values:
    #     data_FI.append(feat_name_importance[k])
    # #average of top 3
    # diff_FI=np.divide(abs(feature_importance[:]-data_FI[:]),feature_importance[:])
    # avg_FI=np.mean(diff_FI)
    # return round((1-avg_FI)*100,2)
    # ------Precision------
    # return round(precision(original_data, private_data), 2)
    # ------FID------
    return round(1 - fid(original_data, private_data), 4)


def precision(original_data, private_data):
    support = original_data
    # Covariance matrix
    covariance = np.cov(support, rowvar=False)
    # Covariance matrix power of -1
    covariance_pm1 = np.linalg.matrix_power(covariance, -1)
    # # Center point
    centerpoint = np.mean(support, axis=0)
    # Distances between non-outlier synthetic data-points and original (finding min distance)
    distances = []
    for i, val in enumerate(private_data.values):
        p1 = val
        p2 = centerpoint
        distances.append((p1 - p2).T.dot(covariance_pm1).dot(p1 - p2))
    return np.mean(1 - stats.chi2.cdf(distances, 10))


def standardize_data(data):
    stand_data = data.copy(deep=True)
    for col in list(data.columns):
        col_vals = np.array(data[col].values)
        col_vals = col_vals.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(col_vals)
        col_vals = scaler.transform(col_vals)
        col_vals = col_vals.squeeze(1)
        stand_data[col] = col_vals
    return stand_data


def fid(original_data, private_data):
    real_data = standardize_data(original_data).values
    gen_data = standardize_data(private_data).values

    # calculate mean and covariance statistics
    mu1, sigma1 = real_data.mean(axis=0), cov(real_data, rowvar=False)
    mu2, sigma2 = gen_data.mean(axis=0), cov(gen_data, rowvar=False)

    ssdiff = np.sum(
        (mu1 - mu2) ** 2.0
    )  # calculate sum squared difference between means
    covmean = sqrtm(sigma1.dot(sigma2))  # calculate sqrt of product between cov

    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)  # calculate score
    return fid


def general_quality_index(data_o, data_g):
    corr_G = data_g.corr(method="pearson")
    corr_R = data_o.corr(method="pearson")

    df_R = corr_R.where(np.triu(np.ones(corr_R.shape)).astype(np.bool))
    df_R = df_R.stack().reset_index()
    df_R.columns = ["Feature1_R", "Feature2_R", "Corr_R"]

    df_G = corr_G.where(np.triu(np.ones(corr_G.shape)).astype(np.bool))
    df_G = df_G.stack().reset_index()
    df_G.columns = ["Feature1_G", "Feature2_G", "Corr_G"]

    df_RG = pd.concat([df_R, df_G], axis=1)
    df_RG.drop(
        ["Feature1_R", "Feature2_R", "Feature1_G", "Feature2_G"], axis=1, inplace=True
    )
    val = 1 - abs(np.mean(df_RG.diff(axis=1).Corr_G.values))
    return val
