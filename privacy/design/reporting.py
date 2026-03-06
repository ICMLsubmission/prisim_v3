import geopandas as gpd
import lightgbm as ltb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, clear_output, display
from ipywidgets import ButtonStyle, widgets, Layout
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# from ..configurations.state_codes import country_codes, states_codes
from ..design.metric_utility import compute_gqi, metricsCalculator
from ..src.FRUFS import FRUFS
from ..src.helper import (
    private_data__ml_training,
    get_feature_importance,
    get_frufs_feature_importance,
)


def variable_selection(data, privateData, title, button_desc, error, function):
    # display(HTML(f'<h3 style="text-align: center;">{title}</h3><hr>'))
    style = {"description_width": "100px"}
    layout = {"width": "400px"}
    options = list(data.select_dtypes("number").columns)
    if "states_countries" in options:
        options.remove("states_countries")
    w = widgets.SelectMultiple(
        options=options,
        value=[options[0]],
        rows=len(options),
        description="<b>Select Variables</b>",
        disabled=False,
        style=style,
        layout=layout,
    )
    display(w)
    button = widgets.Button(description=f"{button_desc}", style=style, layout=layout)
    out = widgets.Output()
    with out:
        cols = options[:]
        if len(cols) <= 1 and error is not None:
            display(HTML(f"<h4>{error}</h4>"))
        elif privateData is None:
            function(data[cols], None)
        else:
            function(data[cols], privateData[cols])

    def on_button_clicked(_):
        with out:
            clear_output()
            cols = list(w.value)
            if len(cols) <= 1 and error is not None:
                display(HTML(f"<h4>{error}</h4>"))
            elif privateData is None:
                function(data[cols], None)
            else:
                function(data[cols], privateData[cols])

    button.on_click(on_button_clicked)
    b = widgets.VBox([button, out])
    display(b)


def descriptive(data, privateData):
    display(HTML("<h4>Original Data:</h4>"))
    display(HTML(data.describe().to_html()))
    if privateData is not None:
        display(HTML("<h4>Private Data:</h4>"))
        display(HTML(privateData.describe().to_html()))


def corr_plot(data, privateData):
    if privateData is not None:
        corr_p = privateData.corr()
        corr_ = data.corr()
        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        plt.title("Original Data Correlation Plot")
        mask = np.triu(np.ones_like(data.corr()))
        sns.heatmap(corr_, annot=True, fmt=".1f", mask=mask)
        plt.subplot(122)
        plt.title("Synthetic Data Correlation Plot")
        mask = np.triu(np.ones_like(privateData.corr()))
        sns.heatmap(corr_p, annot=True, fmt=".1f", mask=mask)
        plt.tight_layout()
        plt.show()
    else:
        corr_ = data.corr()
        plt.figure(figsize=(11, 7))
        sns.heatmap(
            corr_, annot=True, fmt=".2f", mask=np.triu(np.ones_like(data.corr()))
        )
        plt.show()


def plot(df, privateData, x, y, categorical):
    pbar = tqdm(total=100)
    pbar.update(50)
    sns.set(rc={"figure.figsize": (7, 7)})
    if x in categorical and y in categorical:
        if privateData is not None:
            plt.figure(figsize=(14, 5))
            # plt.subplot(121)
            sns.displot(x=df[x].values, y=df[y].values)
            plt.title(f"Original Data {x} vs. {y}")
            plt.xlabel(x)
            plt.ylabel(y)
            # plt.subplot(122)
            sns.displot(
                x=privateData[x].values, y=privateData[y].values, color="orange"
            )
            plt.title(f"Synthetic Data {x} vs. {y}")
            plt.xlabel(x)
            plt.ylabel(y)
            plt.tight_layout()
        else:
            sns.displot(x=df[x].values, y=df[y].values)
            plt.title(f"Original Data {x} vs. {y}")
            plt.xlabel(x)
            plt.ylabel(y)
    elif x in categorical or y in categorical:
        if privateData is not None:
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            plt.title(f"Original Data {x} vs. {y}")
            sns.scatterplot(x=df[x].values, y=df[y].values)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.subplot(122)
            plt.title(f"Synthetic Data {x} vs. {y}")
            sns.scatterplot(
                x=privateData[x].values, y=privateData[y].values, color="orange"
            )
            plt.xlabel(x)
            plt.ylabel(y)
            plt.tight_layout()
        else:
            sns.scatterplot(x=df[x].values, y=df[y].values)
            plt.title(f"Original Data {x} vs. {y}")
            plt.xlabel(x)
            plt.ylabel(y)
    else:
        if privateData is not None:
            # plt.figure(figsize=(14, 5))
            # plt.subplot(121)
            sns.displot(x=df[x].values, y=df[y].values, kind="kde")
            plt.title(f"Original Data {x} vs. {y}")
            plt.xlabel(x)
            plt.ylabel(y)
            sns.displot(
                x=privateData[x].values,
                y=privateData[y].values,
                color="orange",
                kind="kde",
            )
            plt.title(f"Synthetic Data {x} vs. {y}")
            plt.xlabel(x)
            plt.ylabel(y)
            # plt.tight_layout()
        else:
            sns.displot(x=df[x].values, y=df[y].values, kind="kde")
            plt.title(f"Original Data {x} vs. {y}")
            plt.xlabel(x)
            plt.ylabel(y)
    pbar.update(50)
    pbar.close()
    plt.show()


def kde_plot(data, privateData, col, hist=False):
    plt.figure(figsize=(10, 5))
    if hist:
        data = {"Real Data": data[col], "Synthetic Data": privateData[col]}
        fig = sns.histplot(data, palette=['blue', 'red'])
        # fig = sns.histplot(privateData[col], shade=True, color="r", label="Real Data", kde=True)
    else:
        fig = sns.kdeplot(data[col], shade=True, color="b", label="Synthetic Data")
        fig = sns.kdeplot(privateData[col], shade=True, color="r", label="Real Data")
        plt.legend()
    plt.title(f"KDEs for {col} between real vs private")
    plt.show()


def feature_importance(
    originalData_feat_imps_,
    originalData_columns,
    k,
    privateData_feat_imps_=None,
    privateData_columns=None,
):
    x_axis_orig = originalData_feat_imps_
    y_axis_orig = np.arange(len(originalData_columns))
    if privateData_feat_imps_ is not None:
        x_axis_priv = privateData_feat_imps_
        y_axis_priv = np.arange(len(privateData_columns))
        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        plt.title("Unsupervised Average Feature Importance on Original Data")
        sns.lineplot(
            x=x_axis_orig, y=[k for i in range(len(y_axis_orig))], linestyle="--"
        )
        sns.barplot(x=x_axis_orig, y=y_axis_orig, orient="h")
        if type(originalData_columns[0]) == str:
            plt.yticks(y_axis_orig, originalData_columns, size="small")
        else:
            plt.yticks(
                y_axis_orig,
                ["Feature " + str(i) for i in originalData_columns],
                size="small",
            )
        plt.xlabel("Importance Scores")
        plt.ylabel("Features")
        sns.despine()
        plt.subplot(122)
        plt.title("Unsupervised Average Feature Importance on Private Data")
        sns.lineplot(
            x=x_axis_priv, y=[k for i in range(len(y_axis_priv))], linestyle="--"
        )
        sns.barplot(x=x_axis_priv, y=y_axis_priv, orient="h")
        if type(privateData_columns[0]) == str:
            plt.yticks(y_axis_priv, privateData_columns, size="small")
        else:
            plt.yticks(
                y_axis_priv,
                ["Feature " + str(i) for i in privateData_columns],
                size="small",
            )
        plt.xlabel("Importance Scores")
        plt.ylabel("Features")
        sns.despine()
        plt.tight_layout()
        plt.show()
    else:
        plt.subplot(121)
        plt.title("Unsupervised Average Feature Importance on Original Data")
        sns.lineplot(
            x=x_axis_orig, y=[k for i in range(len(y_axis_orig))], linestyle="--"
        )
        sns.barplot(x=x_axis_orig, y=y_axis_orig, orient="h")
        if type(originalData_columns[0]) == str:
            plt.yticks(y_axis_orig, originalData_columns, size="small")
        else:
            plt.yticks(
                y_axis_orig,
                ["Feature " + str(i) for i in originalData_columns],
                size="small",
            )
        plt.xlabel("Importance Scores")
        plt.ylabel("Features")
        sns.despine()
        plt.show()


def data_profile(
    originalData, originalDataEncoded, privateData, privateDataEncoded, tab
):
    originalData = originalData.copy()
    originalDataEncoded = originalDataEncoded.copy()
    with tab:
        error = "Data is not initialized... Please run above blocks to initiate dataset"
        snapshot_output = widgets.Output()
        with snapshot_output:
            if originalData is None:
                display(HTML(f"<h4>{error}</h4>"))
                return None

            if "states_countries" in originalData.columns:
                originalData.drop(["states_countries"], axis=1, inplace=True)

            originalData.reset_index(inplace=True, drop=True)
            if privateData is not None:
                privateData = privateData.copy()
                privateDataEncoded = privateDataEncoded.copy()
                if "states_countries" in privateData.columns:
                    privateData.drop(["states_countries"], axis=1, inplace=True)
                privateData.reset_index(inplace=True, drop=True)
                display(HTML(privateData.head().to_html()))
            else:

                display(HTML(originalData.head().to_html()))
        descriptive_output = widgets.Output()
        with descriptive_output:
            if originalDataEncoded is None:
                display(HTML(f"<h4>{error}</h4>"))
                return None
            originalDataEncoded = originalDataEncoded.copy()
            if privateDataEncoded is not None:
                privateDataEncoded = privateDataEncoded.copy()
            title = "Descriptive Statistics"
            button_desc = "See Descriptive Statistics"
            variable_selection(
                originalDataEncoded,
                privateDataEncoded,
                title,
                button_desc,
                error,
                descriptive,
            )
        snapshot_box = widgets.VBox([snapshot_output])
        descriptive_box = widgets.VBox([descriptive_output])
        accordion_box = widgets.Accordion(
            children=[snapshot_box, descriptive_box], selected_index=0
        )
        accordion_box.set_title(0, "Click here to view Data Snapshot")
        accordion_box.set_title(1, "Click here to explore Descriptive Statistics")
        if privateData is not None:
            accordion_box.set_title(1, "Click here to compare Descriptive Statistics")
        display(accordion_box)


def visual_comparison(originalData, privateData, discrete_columns, tab):
    originalData = originalData.copy()
    if privateData is not None:
        privateData = privateData.copy()
    with tab:
        error = "Data is not initialized... Please run above blocks to initiate dataset"
        distribution_output = widgets.Output()
        with distribution_output:
            if originalData is None or privateData is None:
                display(HTML(f"<h4>{error}</h4>"))
                return None
            next_out = widgets.Output()
            with next_out:
                if privateData.columns[0] in discrete_columns:
                    kde_plot(privateData, originalData, privateData.columns[0], hist=True)
                else:
                    kde_plot(privateData, originalData, privateData.columns[0], hist=False)
            display(HTML("Select Feature for KDE Comparison"))
            style = {"description_width": "150px"}
            layout = {"width": "70%"}
            box_layout = widgets.Layout(
                display="flex", flex_flow="column", align_items="center", width="70%"
            )
            features = list(privateData.columns)
            if "states_countries" in features:
                features.remove("states_countries")
            options = features  # options[0]
            feature_wid = widgets.Dropdown(
                options=options,
                value=options[0],
                rows=len(options),
                description="<b>Feature:</b>",
                disabled=False,
                style=style,
                layout=layout,
            )
            next_button = widgets.Button(
                description=f"Proceed",
                style=ButtonStyle(button_color="orange"),
                layout=layout,
                tooltip="Proceed",
            )

            def on_next_button_clicked(_):
                with next_out:
                    clear_output()
                    feat = feature_wid.value
                    if feat in discrete_columns:
                        kde_plot(privateData, originalData, feat, hist=True)
                    else:
                        kde_plot(privateData, originalData, feat)

            next_button.on_click(on_next_button_clicked)
            v = widgets.VBox([feature_wid, next_button, next_out], layout=box_layout)
            display(v)
        visual_output = widgets.Output()
        with visual_output:
            if originalData is None or privateData is None:
                display(HTML(f"<h4>{error}</h4>"))
                return None
            style = {"description_width": "100px"}
            layout = {"width": "400px"}
            button = widgets.Button(
                description="Plot T-SNE",
                style=ButtonStyle(button_color="orange"),
                layout=layout,
                tooltip="Plot T-SNE",
            )
            out = widgets.Output()

            def on_button_clicked(_):
                with out:
                    clear_output()
                    pbar = tqdm(total=100)
                    pbar.update(25)
                    priv_tsne = privateData.copy(deep=True)
                    priv_tsne["Data Property"] = "Synthetic"
                    orig_tsne = originalData.copy(deep=True)
                    orig_tsne["Data Property"] = "Real"
                    pbar.update(25)
                    data_tsne = pd.concat([priv_tsne, orig_tsne])
                    tsne = TSNE(n_components=2, perplexity=5, n_iter=400)
                    pbar.update(25)
                    tsne_result = tsne.fit_transform(data_tsne.iloc[:, :-1])
                    tsne_final = data_tsne.copy(deep=True)
                    tsne_final["tsne-2d-one"] = tsne_result[:, 0]
                    tsne_final["tsne-2d-two"] = tsne_result[:, 1]
                    sns.set(rc={"figure.figsize": (10, 10)})
                    sns.scatterplot(
                        x="tsne-2d-one",
                        y="tsne-2d-two",
                        palette=[
                            "#ed7551",
                            "#2278b4",
                        ],  # sns.color_palette("tab10", 2),
                        data=tsne_final,
                        hue="Data Property",
                        style="Data Property",
                        legend="full",
                        alpha=0.8,
                    )
                    pbar.update(25)
                    pbar.close()
                    plt.show()

            button.on_click(on_button_clicked)
            b = widgets.VBox([button, out])
            display(b)
        frufs_output = widgets.Output()
        with frufs_output:
            style = {"description_width": "100px"}
            layout = {"width": "400px"}
            fruffs_button = widgets.Button(
                description="Plot FRUFS",
                style=ButtonStyle(button_color="orange"),
                layout=layout,
                tooltip="Plot FRUFS",
            )
            frufs_button_out = widgets.Output()

            def on_fruffs_button_clicked(_):
                with frufs_button_out:
                    clear_output()
                    pbar = tqdm(total=100)
                    pbar.update(25)
                    model_frufs_original = FRUFS(
                        model_r=ltb.LGBMRegressor(random_state=27),
                        model_c=ltb.LGBMClassifier(
                            random_state=27, class_weight="balanced"
                        ),
                        categorical_features=discrete_columns,
                        k=1.0,
                        n_jobs=-1,
                        random_state=27,
                    )
                    pbar.update(25)
                    model_frufs_private = FRUFS(
                        model_r=ltb.LGBMRegressor(random_state=27),
                        model_c=ltb.LGBMClassifier(
                            random_state=27, class_weight="balanced"
                        ),
                        categorical_features=discrete_columns,
                        k=1.0,
                        n_jobs=-1,
                        random_state=27,
                    )
                    _ = model_frufs_original.fit_transform(originalData)
                    pbar.update(25)
                    if privateData is not None:
                        _ = model_frufs_private.fit_transform(privateData)
                        feature_importance(
                            model_frufs_original.feat_imps_,
                            model_frufs_original.columns_,
                            model_frufs_original.k,
                            model_frufs_private.feat_imps_,
                            model_frufs_private.columns_,
                        )
                    else:
                        feature_importance(
                            model_frufs_original.feat_imps_,
                            model_frufs_original.columns_,
                            model_frufs_original.k,
                        )
                    pbar.update(25)
                    pbar.close()

            fruffs_button.on_click(on_fruffs_button_clicked)
            display(widgets.VBox([fruffs_button, frufs_button_out]))
        distribution_box = widgets.VBox([distribution_output])
        visual_box = widgets.VBox([visual_output])
        frufs_box = widgets.VBox([frufs_output])
        accordion_box = widgets.Accordion(
            children=[distribution_box, visual_box, frufs_box], selected_index=0
        )
        accordion_box.set_title(
            0, "Click here to explore the comparison of Univariate feature distribution"
        )
        accordion_box.set_title(1, "Click here to explore the T-SNE plot")
        accordion_box.set_title(2, "Click here to explore the comparison of FRUFS")
        display(accordion_box)


def qualitative_comparison(originalData, privateData, discrete_columns, tab):
    originalData = originalData.copy()
    if privateData is not None:
        privateData = privateData.copy()
    error = "Data is not initialized... Please run above blocks to initiate dataset"
    with tab:
        correlation_output = widgets.Output()
        with correlation_output:
            if originalData is None:
                display(HTML(f"<h4>{error}</h4>"))
                return None
            title = "Correlation Matrix"
            button_desc = "See Correlation Plot"
            validation_error = "Select atleast 2 or more variables to plot correlation"
            variable_selection(
                originalData,
                privateData,
                title,
                button_desc,
                validation_error,
                corr_plot,
            )
        visualization_output = widgets.Output()
        with visualization_output:
            style = {"description_width": "100px"}
            layout = {"width": "70%"}
            optns = list(originalData.columns)
            if "states_countries" in optns:
                optns.remove("states_countries")
            options = optns
            w1 = widgets.Dropdown(
                options=options,
                value=options[0],
                description="<b>X</b>",
                disabled=False,
                style=style,
                layout=layout,
            )
            display(w1)
            w2 = widgets.Dropdown(
                options=options,
                value=options[0],
                description="<b>Y</b>",
                disabled=False,
                style=style,
                layout=layout,
            )
            display(w2)
            button = widgets.Button(description="Show Plot", style=style, layout=layout)
            visual_plot_out = widgets.Output()
            # with visual_plot_out:
            #     _= plot(originalData, privateData, options[0], options[1], discrete_columns)
            def on_button_clicked(_):
                with visual_plot_out:
                    clear_output()
                    x = w1.value
                    y = w2.value
                    if x == y:
                        display(HTML("<h4>Please select different X & Y</h4>"))
                    else:
                        _ = plot(originalData, privateData, x, y, discrete_columns)

            button.on_click(on_button_clicked)
            b = widgets.VBox([button, visual_plot_out])
            display(b)
        alpha_beta_output = widgets.Output()
        with alpha_beta_output:
            layout = {"width": "300px"}
            button = widgets.Button(
                description="Plot α-Precision & β-Recall",
                style=ButtonStyle(button_color="orange"),
                layout=layout,
                tooltip=" α-Precision & β-Recall",
            )
            tsne_out = widgets.Output()

            def on_button_clicked(_):
                with tsne_out:
                    clear_output()
                    pbar = tqdm(total=100)
                    pbar.update(25)
                    metrics_calc = metricsCalculator(originalData, privateData)
                    alpha = np.linspace(0.5, 0.99, num=20)
                    beta = np.linspace(0.5, 0.99, num=20)
                    alpha_all = []
                    beta_all = []
                    for j in range(5):
                        alpha_pres = []
                        beta_rec = []
                        for a, b in zip(alpha, beta):
                            alpha_pres.append(
                                metrics_calc.compute_a_precision_b_recall(a, discrete_columns)
                            )
                            beta_rec.append(
                                metrics_calc.compute_a_precision_b_recall(
                                    b,  discrete_columns, metric_type="beta_recall"
                                )
                            )
                        alpha_all.append(alpha_pres)
                        beta_all.append(beta_rec)
                        pbar.update(10)
                    alpha_all = np.array(alpha_all)
                    beta_all = np.array(beta_all)
                    windows_P = pd.Series(np.mean(alpha_all, axis=0)).rolling(5)
                    moving_averages_P = windows_P.mean()
                    windows_R = pd.Series(np.mean(beta_all, axis=0)).rolling(5)
                    moving_averages_R = windows_R.mean()
                    # alpha curve
                    plt.figure(figsize=(10, 4))
                    plt.subplot(121)
                    plt.plot(alpha, moving_averages_P, "-", color="blue")
                    plt.title("α-Precision Curve")
                    plt.xlabel("α Values")
                    plt.ylabel("α-Precision")
                    plt.grid()
                    # beta curve
                    plt.subplot(122)
                    plt.plot(beta, moving_averages_R, "-", color="orange")
                    plt.title("β-Recall Curve")
                    plt.xlabel("β Values")
                    plt.ylabel("β-Recall")
                    plt.grid()
                    plt.tight_layout()
                    plt.show()
                    pbar.update(25)
                    pbar.close()

            button.on_click(on_button_clicked)
            b = widgets.VBox([button, tsne_out])
            display(b)
        correlation_box = widgets.VBox([correlation_output])
        visualization_box = widgets.VBox([visualization_output])
        alpha_beta_box = widgets.VBox([alpha_beta_output])
        accordion_box = widgets.Accordion(
            children=[correlation_box, visualization_box], selected_index=0
        )
        accordion_box.set_title(0, "Click here to explore feature correlations")
        accordion_box.set_title(1, "Click here to explore feature distributions")
        if privateData is not None:
            accordion_box = widgets.Accordion(
                children=[correlation_box, visualization_box, alpha_beta_box],
                selected_index=0,
            )
            accordion_box.set_title(
                0, "Click here to explore the comparison of feature correlations"
            )
            accordion_box.set_title(
                1, "Click here to explore the comparison of feature distributions"
            )
            accordion_box.set_title(
                2,
                "Click here to explore the comparison of α-Precision & β-Recall",
            )
        display(accordion_box)


def quantitative_comparison(originalData, privateData, auPrivacyScore, tab):
    originalData = originalData.copy()
    if privateData is not None:
        privateData = privateData.copy()
    with tab:
        if privateData is None:
            display(
                HTML(
                    "<h4>Data is not initialized... Please run above blocks to initiate dataset </h4>"
                )
            )
            return None
        gqi = compute_gqi(originalData, privateData)
        if len(originalData) > len(privateData):
            originalData = originalData.sample(n=len(privateData))
        else:
            privateData = privateData.sample(n=len(originalData))
        metrics_calc = metricsCalculator(originalData, privateData)
        jsd = metrics_calc.compute_JS()[1]
        # fid = metrics_calc.compute_fid()
        metricsHtmlString = f"""
            <table style="width: 65%;">
                <tbody>
                    <tr>
                        <td>
                            <div>
                                <strong style="text-align:left">Frechet Inception distance (FID):</strong>
                                <a title="FID computes the difference in distribution in a multivariate manner. Lower FID means similar to real distribution."><i class="fa fa-info-circle"></i></a>
                            </div>
                        </th>
                        <td style="text-align: left;">{round(1-(auPrivacyScore/100), 2)} &nbsp;&nbsp; <span style="color:blue; font-size:smaller">Lower is better (i.e. Lower bound is 0)</span>
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <div>
                                <strong style="text-align:left">Jensen-Shannon distance (JS):</strong>
                                <a title="JS computes the difference in distribution in an univariate manner. Lower JS means similar to real distribution."><i class="fa fa-info-circle"></i></a>
                            </div>
                        </th>
                        <td style="text-align: left;">{round(jsd, 2)} &nbsp;&nbsp; <span style="color:blue; font-size:smaller">Lower is better (i.e. Range is 0 to 1)</span>
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <div>
                                <strong style="text-align:left">Generation Quality Index (GQI):</strong>
                                <a title="GQI computes the difference in correlations in a bivariate manner. Higher GQI means similar to real Correlation."><i class="fa fa-info-circle"></i></a>
                            </div>
                        </th>
                        <td style="text-align: left;">{round(gqi, 2)} &nbsp;&nbsp; <span style="color:blue; font-size:smaller">Higher is better (i.e. Range is 0 to 1)</span>
                        </td>
                    </tr>
                </tbody>
            </table>
        """
        display(HTML(metricsHtmlString))


def ml_evaluation(originalData, privateData, discreteColumns, tab):
    originalData = originalData.copy()
    if privateData is not None:
        privateData = privateData.copy()
    with tab:
        options = list(originalData.select_dtypes("number").columns)
        try:
            # imp_feature = options[
            #     # 0
            #     # list(get_frufs_feature_importance(privateData, discreteColumns)[1])[0]
            # ]
            imp_feature = list(
                get_frufs_feature_importance(privateData, discreteColumns)[1]
            )[0]
        except:
            imp_feature = options[0]
        value = imp_feature if imp_feature in options else options[0]
        if "states_countries" in options:
            options.remove("states_countries")
        ml_evaluation_title = widgets.HTML(
            "<h2>Explore ML Utility with Private Data</h2><hr>"
        )
        style = {"description_width": "350px"}
        layout = {"width": "60%"}
        y_var = widgets.Dropdown(
            options=options,
            value=value,
            rows=len(options),
            description="<b>Select Target Variable:</b>",
            disabled=False,
            style=style,
            layout=layout,
            tootltip="Select Target Variable",
        )
        button = widgets.Button(
            description="Run ML Task",
            style=ButtonStyle(button_color="orange"),
            layout={"width": "30%"},
            tooltip="Run ML Task",
        )
        out = widgets.Output()

        def on_button_clicked(_):
            with out:
                clear_output()
                display(HTML("<b>Evaluating ML Utility...</b>"))
                pbar = tqdm(total=100)
                pbar.update(0)
                regressor = True
                if y_var.value in discreteColumns:
                    regressor = False
                features = options.copy()
                features.remove(y_var.value)
                X_o, X_p = originalData[features], privateData[features]
                y_o, y_p = originalData[y_var.value], privateData[y_var.value]
                X_train, X_test, y_train, y_test = train_test_split(
                    X_o, y_o, test_size=1 / 3, random_state=0
                )
                original_ml_metric = private_data__ml_training(
                    X_test, X_train, y_test, y_train, regressor=regressor
                )
                pbar.update(25)
                private_ml_metric = private_data__ml_training(
                    X_test, X_p, y_test, y_p, regressor=regressor
                )
                pbar.update(25)
                original_ml_utility = min(max(np.round(original_ml_metric, 2), 0), 1)
                private_ml_utility = min(
                    max(np.round(private_ml_metric, 2), 0), original_ml_utility
                )
                normalized_private_ml_utility = (
                    "Indefinite"
                    if original_ml_utility == 0
                    else int(min((private_ml_utility / original_ml_utility) * 100, 100))
                )
                normalized_orignial_ml_utility = (
                    "Indefinite" if original_ml_utility == 0 else 100
                )
                ml_metrics_table = f"""<center><table>
                                        <thead>
                                        <tr>
                                            <th ></th>
                                            <th  colspan="3" style="text-align:center">ML Evaluations</th>
                                        </tr>
                                        </thead>
                                        <tbody>
                                        <tr>
                                            <td ></td>
                                            <td ><b>Original ML-Utility</b></td>
                                            <td ><b>% of original ML-Utility</b></td>
                                            <td ></td>
                                        </tr>
                                        <tr>
                                            <td ><b>Original Dataset<br></b></td>
                                            <td >{original_ml_utility}</td>
                                            <td >{normalized_orignial_ml_utility}</td>
                                        </tr>
                                        <tr>
                                        <td ><b>Private Dataset</b></td>
                                        <td >{private_ml_utility}</td>
                                        <td >{normalized_private_ml_utility}</td>
                                    </tr>
                                        </tbody>
                                    </table></center><br><hr style="background-color=#00000021">"""

                originalFeatures = get_feature_importance(X_o, y_o, regressor=regressor)
                pbar.update(25)
                privateFeatures = get_feature_importance(X_p, y_p, regressor=regressor)
                pbar.update(25)
                display(HTML(ml_metrics_table))
                interactivePlot(originalFeatures, privateFeatures)
                pbar.close()

        button.on_click(on_button_clicked)
        horizontal_align = widgets.HBox([y_var, button])
        vertical_align = widgets.VBox([ml_evaluation_title, horizontal_align, out])
        display(vertical_align)


def interactivePlot(originalFeatures, privateFeatures):
    display(HTML("<h3>Feature Importantance for the ML task:</h3>"))
    cols = 2
    plt.figure(figsize=(12, 6))
    plt.subplot(1, cols, 1)
    plt.barh(
        originalFeatures.index.values[:30], originalFeatures.values[:30], color="blue"
    )
    plt.title("Feature Importance on Original Dataset")
    plt.xticks(np.arange(0, 1, 0.2))
    plt.xlim(0, 1)
    plt.grid(True)
    if cols == 2:
        plt.subplot(1, cols, cols)
        plt.barh(
            privateFeatures.index.values[:30],
            privateFeatures.values[:30],
            color="orange",
        )
        plt.title("Feature Importance on Private Dataset")
        plt.xticks(np.arange(0, 1, 0.2))
        plt.xlim(0, 1)
        plt.grid(True)
    plt.tight_layout()
    plt.show()


def world_process_df(df):
    world = pd.read_csv("privacy/configurations/maps_data/world.csv")
    world["geometry"] = gpd.GeoSeries.from_wkt(world["geometry"])
    world = gpd.GeoDataFrame(world, geometry="geometry")
    df = df.groupby(["countries"]).size().reset_index(name="counts")
    df = df.merge(world, left_on="countries", right_on="name")
    df = df[["countries", "counts", "geometry"]]
    df["counts"] = df["counts"] / df["counts"].sum() * 1500
    df = gpd.GeoDataFrame(df)
    df["geometry"] = df[
        "geometry"
    ].centroid  # np.array(list(map(lambda x: x.centroid, df["geometry"])), dtype="object")
    return world, df


def plot_country(world, original_df, private_df=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    world.plot(ax=ax, color="lightgray", edgecolor="grey", linewidth=0.5)
    original_df.plot(
        ax=ax,
        color="#000099",
        markersize="counts",
        alpha=0.5,
        categorical=False,
        label="Original",
    )
    if private_df is not None:
        private_df.plot(
            ax=ax,
            color="#ff5050",
            markersize="counts",
            alpha=0.4,
            categorical=False,
            label="Synthetic",
        )
    ax.axis("off")
    lgnd = plt.legend()
    lgnd.legendHandles[0]._sizes = [30]
    if private_df is not None:
        lgnd.legendHandles[1]._sizes = [30]
    plt.title("Location Distribution: World")
    plt.show()


def state_process_df(path, df):
    states_df = pd.read_csv(path)
    states_df["geometry"] = gpd.GeoSeries.from_wkt(states_df["geometry"])
    states_df = gpd.GeoDataFrame(states_df, geometry="geometry")
    df = df.groupby(["states"]).size().reset_index(name="counts")
    df = df.merge(states_df, left_on="states", right_on="states")
    df = df[["states", "counts", "geometry"]]
    df["counts"] = df["counts"] / df["counts"].sum() * 3000
    df = gpd.GeoDataFrame(df)
    df["geometry"] = df["geometry"].centroid
    return states_df, df


def state_plot(states_df, original_df, private_df=None, country_name=None):
    fig, ax = plt.subplots(figsize=(14, 14))
    states_df.plot(ax=ax, color="lightgray", edgecolor="grey", linewidth=0.5)
    original_df.plot(
        ax=ax,
        color="#000099",
        markersize="counts",
        alpha=0.5,
        categorical=False,
        label="Original",
    )
    if private_df is not None:
        private_df.plot(
            ax=ax,
            color="#ff5050",
            markersize="counts",
            alpha=0.4,
            categorical=False,
            label="Synthetic",
        )
    ax.axis("off")
    lgnd = plt.legend()
    lgnd.legendHandles[0]._sizes = [30]
    if private_df is not None:
        lgnd.legendHandles[1]._sizes = [30]
    plt.title(f"Location Distribution: {country_name}")
    plt.show()


def map_visualization(originalDf, privateDf, tab, discreteLabelEncoders=None):
    originalDf = originalDf.copy()
    if privateDf is not None:
        privateDf = privateDf.copy()
    with tab:
        originalDf = originalDf[["states_countries"]]
        originalDf["states_countries"] = discreteLabelEncoders[
            "states_countries"
        ].inverse_transform(originalDf["states_countries"].values)
        originalDf[["states", "countries"]] = originalDf["states_countries"].str.split(
            "_", 1, expand=True
        )
        for i in ["US", "USA", "United States"]:
            if i.lower() in originalDf["countries"].str.lower().values:
                originalDf["countries"] = originalDf["countries"].str.replace(
                    i, "United States of America"
                )

        world, processedOriginalDf = world_process_df(originalDf)
        if privateDf is not None:
            privateDf = privateDf[["states_countries"]]
            privateDf["states_countries"] = discreteLabelEncoders[
                "states_countries"
            ].inverse_transform(privateDf["states_countries"].values)
            privateDf[["states", "countries"]] = privateDf[
                "states_countries"
            ].str.split("_", 1, expand=True)
            for i in ["US", "USA", "United States"]:
                if i.lower() in privateDf["countries"].str.lower().values:
                    privateDf["countries"] = privateDf["countries"].str.replace(
                        i, "United States of America"
                    )
            world, processedPrivateDf = world_process_df(privateDf)
            plot_country(world, processedOriginalDf, processedPrivateDf)
        else:
            plot_country(world, processedOriginalDf)

        display(HTML("Select country to see region wise distribution"))
        style = {"description_width": "150px"}
        layout = {"width": "70%"}
        options = [
            "United States of America",
            "Australia",
            "France",
            "Italy",
            "Argentina",
        ]
        next_out = widgets.Output()
        feature_wid = widgets.Dropdown(
            options=options,
            value=options[0],
            rows=len(options),
            description="<b>Select Country:</b>",
            disabled=False,
            style=style,
            layout=layout,
        )
        next_button = widgets.Button(
            description=f"Select Country",
            style=ButtonStyle(button_color="orange"),
            layout=Layout(margin="0px  0px 0px 250px", width="20%"),
            tooltip="Select Country",
        )

        def on_next_button_clicked(_):
            with next_out:
                clear_output()
                next_button.disabled = True
                # display(
                #     HTML("<b>&nbsp;&nbsp;&nbsp;&nbsp;Please wait for a while...</b>")
                # )
                state_path = f"privacy/configurations/maps_data/{feature_wid.value}.csv"
                state_df, originalStatesProcessedDf = state_process_df(
                    state_path, originalDf[originalDf["countries"] == feature_wid.value]
                )
                if privateDf is not None:
                    state_df, privateStatesProcessedDf = state_process_df(
                        state_path,
                        privateDf[privateDf["countries"] == feature_wid.value],
                    )  # get_state_processed_dfs(privateDf, "US", None)
                    state_plot(
                        state_df,
                        originalStatesProcessedDf,
                        privateStatesProcessedDf,
                        feature_wid.value,
                    )
                else:
                    state_plot(
                        state_df,
                        originalStatesProcessedDf,
                        private_df=None,
                        country_name=feature_wid.value,
                    )
                next_button.disabled = False

        next_button.on_click(on_next_button_clicked)
        v = widgets.VBox([feature_wid, next_button, next_out])
        display(v)


class dataProfiling:
    def __init__(
        self,
        originalData,
        originalEncodedData=None,
        privateData=None,
        privateDataEncoded=None,
        numericalColumns=None,
        discreteColumns=None,
        comparison=False,
        nameColumns=None,
        addressColumns=None,
        discreteLabelEncoders=None,
        auPrivacyScore=None,
    ):
        nameColumns = None if len(nameColumns)==0 else nameColumns
        addressColumns = None if len(addressColumns)==0 else addressColumns
        pbar = tqdm(total=100)
        pbar.update(10)
        descriptive_tab = widgets.Output()
        qualitative_tab = widgets.Output()
        pbar.update(10)
        layout = widgets.Layout(
            display="flex", flex_flow="column", align_items="stretch", width="945px"
        )
        if comparison:
            visual_tab = widgets.Output()
            quantitative_tab = widgets.Output()
            mlevaluation_tab = widgets.Output()
            if addressColumns is not None:
                location_tab = widgets.Output()
                tab = widgets.Tab(
                    children=[
                        descriptive_tab,
                        visual_tab,
                        qualitative_tab,
                        quantitative_tab,
                        mlevaluation_tab,
                        location_tab,
                    ],
                    layout=layout,
                )
                # location
                tab.set_title(5, "Location Comparison")
                map_visualization(
                    originalEncodedData,
                    privateDataEncoded,
                    location_tab,
                    discreteLabelEncoders,
                )
            else:
                tab = widgets.Tab(
                    children=[
                        descriptive_tab,
                        visual_tab,
                        qualitative_tab,
                        quantitative_tab,
                        mlevaluation_tab,
                    ],
                    layout=layout,
                )
            # data snapshot, desctiptive statistic
            tab.set_title(0, "Data Profiling")
            data_profile(
                originalData,
                originalEncodedData,
                privateData,
                privateDataEncoded,
                descriptive_tab,
            )

            # TSNE, Data distribution
            tab.set_title(1, "Visual Comparison")
            visual_comparison(
                originalEncodedData,
                privateDataEncoded,
                discreteColumns,
                visual_tab,
            )

            # correlation matrix, data visualization
            tab.set_title(2, "Qualitative Check")
            qualitative_comparison(
                originalEncodedData,
                privateDataEncoded,
                discreteColumns,
                qualitative_tab,
            )

            # GQI, JS, FID
            tab.set_title(3, "Quantitative Check")
            quantitative_comparison(
                originalEncodedData,
                privateDataEncoded,
                auPrivacyScore,
                quantitative_tab,
            )

            # ml evaluations
            tab.set_title(4, "ML Comparison")
            ml_evaluation(
                originalEncodedData,
                privateDataEncoded,
                discreteColumns,
                mlevaluation_tab,
            )
            pbar.update(50)
        else:
            if addressColumns is not None:
                location_tab = widgets.Output()
                tab = widgets.Tab(
                    children=[descriptive_tab, qualitative_tab, location_tab],
                    layout=layout,
                )
                # Location
                tab.set_title(2, "Location Exploration")
                map_visualization(
                    originalEncodedData,
                    None,
                    location_tab,
                    discreteLabelEncoders,
                )
            else:
                tab = widgets.Tab(
                    children=[descriptive_tab, qualitative_tab], layout=layout
                )
            # data snapshot, desctiptive statistics
            tab.set_title(0, "Data Profiling")
            data_profile(
                originalData,
                originalEncodedData,
                None,
                None,
                descriptive_tab,
            )

            # correlation matrix, data visualization
            tab.set_title(1, "Qualitative Check")
            qualitative_comparison(
                originalEncodedData, None, discreteColumns, qualitative_tab
            )
            pbar.update(50)
        pbar.update(30)
        pbar.close()
        display(tab)
        # if comparison:
        #     display(
        #         HTML(
        #             f'<br><br><hr><h2 style="color:orange; text-align:center"> Congratulations! You have successfully privatized your data ! </h2>'
        #         )
        #     )
