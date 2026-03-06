import os
import traceback
import pgeocode
import time
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from werkzeug.utils import secure_filename
from IPython.display import HTML, clear_output, display
from ipywidgets import  widgets, interact, ButtonStyle
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from ..src.helper import (
    outlier_detection
)


def variable_selection(data, pseudoData, features, title, button_desc, error, function):
    # display(HTML(f'<h3 style="text-align: center;">{title}</h3><hr>'))
    style = {"description_width": "100px"}
    layout = {"width": "400px"}
    options = features
    w = widgets.Dropdown(
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
        cols = options[0]
        if len(cols) <= 1 and error is not None:
            display(HTML(f"<h4>{error}</h4>"))
        else:
            function(data[cols], pseudoData[cols])

    def on_button_clicked(_):
        with out:
            clear_output()
            cols = w.value
            if len(cols) <= 1 and error is not None:
                display(HTML(f"<h4>{error}</h4>"))
            else:
                function(data[cols], pseudoData[cols])

    button.on_click(on_button_clicked)
    b = widgets.VBox([button, out])
    display(b)


def kde_hist_plot(data, pseudoData, col, hist=False):
    plt.figure(figsize=(10, 4))
    if pseudoData is not None:
        plt.subplot(121)
    else:
        plt.subplot(111)
    if hist:
        sns.histplot(data, bins=25)
    else:
        sns.histplot(data, bins=25)
    plt.xticks(rotation=45)
    plt.title(f"Original histplot for {col}")
    if pseudoData is not None:
        plt.subplot(122)
        sns.histplot(pseudoData, bins=25)
        plt.title(f"Pseudonymized histplot for {col}")
        # plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def descriptive(data, pseudoData,new_feature_names, tab):
    with tab:
        display(HTML("<h4>Original Data:</h4>"))
        display(HTML(data.head().to_html()))
        if pseudoData is not None:
            pseudoData = pseudoData.copy()
            display(HTML("<h4>Pseudonymized Data:</h4>"))
            pseudoData.rename(columns = new_feature_names, inplace=True)
            display(HTML(pseudoData.head().to_html()))
        # pseudoData.to_csv("save.csv")

def outlier(data, pseudoData, cat_features, aggregated_cat_features, fileNamePrefix, tab):
    with tab:
        originalDf = data.copy()
        if originalDf.shape[0]<=1000:
            clear_output()
            info_text = "There should be atleast 1000 samples to perform a robust outlier analysis"
            display(HTML("<h4 style='color:blue;text-align:center'>**{}**</h4>".format(info_text)))
            return None
        clear_output()
        if pseudoData is not None:
            psuedoDf = pseudoData.copy()
            for i in aggregated_cat_features:
                psuedoDf[i] = LabelEncoder().fit_transform(psuedoDf[i])
            payload = outlier_detection(psuedoDf)
            if payload["error"]:
                display(HTML("<h4 style='color:red;text-align:center'>**{}**</h4>".format(payload["value"])))
                return None
            p_outlier_value, p_outlier_data = payload["value"], payload["data"]
            
            display(HTML(f"<h4 style='text-align:center; width:100%; color:blue'>Proportion of Outliers in the Psuedonymized data is {p_outlier_value}%</h4>"))
            display(HTML("<hr><h4>Descriptive Statistics for Outlier set (Psuedonymized Data):</h4>"))
            display(HTML(p_outlier_data[p_outlier_data["outlier_flag"]==-1][list(originalDf.columns.values)].iloc[:,:-1].describe().to_html()))
            display(HTML("<h4>Descriptive Statistics for Non-Outlier set (Psuedonymized Data):</h4>"))
            display(HTML(p_outlier_data[p_outlier_data["outlier_flag"]==0][list(originalDf.columns.values)].iloc[:,:-1].describe().to_html()))
        else:
            for i in cat_features:
                originalDf[i] = LabelEncoder().fit_transform(originalDf[i])
            payload = outlier_detection(originalDf)
            print()
            if payload["error"]:
                display(HTML("<h4 style='color:red;text-align:center'>**{}**</h4>".format(payload["value"])))
                return None
            o_outlier_value, outlier_data = payload["value"], payload["data"]
            if o_outlier_value is None:
                display(HTML("<h4 style='color:blue;text-align:center'>**{}**</h4>".format("Not enough features to perform the analysis")))
                return None
            display(HTML(f"<h4 style='text-align:center; width:100%; color:blue'>Proportion of Outliers in the Original data is {o_outlier_value}%</h4>"))
            display(HTML("<hr><h4>Descriptive Statistics for Outlier set (Original Data):</h4>"))
            display(HTML(outlier_data[outlier_data["outlier_flag"]==-1][list(originalDf.columns.values)].iloc[:,:-1].describe().to_html()))
            display(HTML("<h4>Descriptive Statistics for Non-Outlier set (Original Data):</h4>"))
            display(HTML(outlier_data[outlier_data["outlier_flag"]==0][list(originalDf.columns.values)].iloc[:,:-1].describe().to_html()))
        download_button = widgets.Button(
                                        description="Download Outlier Data",
                                        style=ButtonStyle(button_color="orange"),
                                        tooltip="Download Outlier Data",
                                    )
        def handle_click(_):
            filename = secure_filename(
                    "outlier_flagged_data_"
                    + str(time.strftime("%Y%m%d-%H_%M_%S"))
                    + ".csv"
                )
            outlier_data["outlier_flag"] = outlier_data["outlier_flag"].apply(lambda x: "Outlier" if x ==-1 else "Non-Outlier")
            outlier_data.to_csv(os.path.join(os.getcwd(), filename), index=False)
            display(HTML("<h4 style='color:blue;text-align:center'>Outlier data saved at **{}**</h4>".format(os.path.join(os.getcwd(), filename))))
        download_button.on_click(handle_click)
        # display(download_button)
            
def privacyReporting(pseudoData, sensitive_features, tab):
    with tab:
        sensitive_features = ["Age",	"Department",	"Education",	"EducationField",	"Gender",	"MonthlyIncome"]
        if sensitive_features is None:
            display(HTML("<h4 style='text-align:center; width:100%; color:blue'>You haven't performed any aggregations (Binning/Suppression)!</h4>"))
            return None
        if len(sensitive_features) < 1:
            # print(sensitive_features)
            display(HTML("<h4 style='color:blue;text-align:center'>**{}**</h4>".format("Not enough features to perform the analysis")))
            return None
        try:
            print("my sensitive feature", sensitive_features)
            quassi_identifiers = ""
            for idx, value in enumerate(sensitive_features):
                if idx+1 == len(sensitive_features):
                    quassi_identifiers += str(value)
                else:
                    quassi_identifiers += str(value) + ", "
            privacy = 0
            min_k = 0
            max_k = 0
            median_k = 0
            anonymity_size = pseudoData.groupby(by=sensitive_features).count().iloc[:,1].values
            anonymity_size = [i for i in anonymity_size if i!=0]
            for i in anonymity_size:
                privacy+=(1-1/i)*100*i
            privacy /= sum(anonymity_size)
            min_k = round(np.min(anonymity_size), 2)
            max_k = round(np.max(anonymity_size), 2)
            median_k = round(np.median(anonymity_size), 0)
            privacy = round((1 - (1/median_k))*100, 3)# round(privacy, 2)
            privacyreport = f"""<center style="margin-top:-18px"><table><thead>
                                    <tr>
                                        <th colspan="2" style="text-align:center"><h3>Based on your transformations, below is the K-anonymity based evaluation report:</h3></th>
                                    </tr>
                                </thead><tbody>
                                        <tr>
                                            <td><p style="text-align:left;vertical-align:middle"><b>Selected Quassi-Identifiers:<b></p></td>
                                            <td><p style="text-align:left;vertical-align:middle">{quassi_identifiers}</p></td>
                                        </tr>
                                        <tr>
                                            <td><p style="text-align:left;vertical-align:middle"><b>Average re-identification risk:<b></p></td>
                                            <td><p style="text-align:left;vertical-align:middle">{round(100 - privacy, 3)}%</p></td>
                                        </tr>
                                        <tr>
                                            <td><p style="text-align:left;vertical-align:middle"><b>Minimum K-value (Based on k anonymity):<b></p></td>
                                            <td><p style="text-align:left;vertical-align:middle">{min_k}</p></td>
                                        </tr>
                                        <tr>
                                            <td><p style="text-align:left;vertical-align:middle"><b>Median K-value (Based on k anonymity):<b></p></td>
                                            <td><p style="text-align:left;vertical-align:middle">{median_k}</p></td>
                                        </tr>
                                        <tr>
                                            <td><p style="text-align:left;vertical-align:middle"><b>Maximum K-value (Based on k anonymity):<b></p></td>
                                            <td><p style="text-align:left;vertical-align:middle">{max_k}</p></td>
                                        </tr>
                                </tbody></table></center>"""
            display(HTML(privacyreport))
        except:
            display(HTML("<h4 style='color:blue;text-align:center'>**{}**</h4>".format("Not able to asses risk with selected QID's.")))
            print(traceback.format_exc())
            return None
        

class visualization:
    def __init__(self, data, pseudoData, primary_features, features, errorText, tab):
        self.data = data
        self.pseudoData = pseudoData
        self.features = features
        self.errorText = errorText
        self.tab = tab
        self.primaryFeatures = primary_features
        self.processedData = None
        self.processedPseudoData = None
    def create_widgets(self):
        options = self.features
        style = {"description_width": "150px"}
        layout = {"width": "400px"}
        self.datatype_radio = widgets.RadioButtons(options=["Tabular Data","Transactional Data"], description="<b>Select dataset type:</b>", 
                                                style = style, layout = {"width":"300px"})
        self.primary_feature =  widgets.Dropdown(options=list(self.primaryFeatures),
                                                description="<b>Select ID:</b>",style={"description_width": "70px"},layout={"width": "275px"})
        self.filter_type  =  widgets.Dropdown(options=list(self.data[self.primary_feature.value].unique()),
                                                description="<b>Select ID Value:</b>",style={"description_width": "125px"},layout={"width": "275px"})
        self.feature_select_transcational  = widgets.Dropdown(options=[i for i in self.features if i not in self.primary_feature.value], 
                                                              description="<b>Select Feature to explore:</b>", 
                                                              style={"description_width": "170px"},layout={"width": "350px"})
        self.feature_select_tabular = widgets.Dropdown(options=options,value=options[0],description="<b>Select Feature:</b>",
                                                        style=style,layout=layout)
        self.radio_box = widgets.HBox([self.datatype_radio])
        self.radio_respond_box = widgets.HBox([self.feature_select_tabular])
        self.box = widgets.VBox([self.radio_box, self.radio_respond_box])
        self.out = widgets.Output()
    def app(self):
        with self.tab:
            if len(self.features)<=0:
                display(HTML("<h4 style='color:red;text-align:center'>**{}**</h4>".format(self.errorText)))
                return None
        self.create_widgets()
        self.processedData = self.data[self.features[0]]
        if self.pseudoData is not None:
            self.processedPseudoData = self.pseudoData[self.features[0]]
        def radio_event(value):
            if value["new"] == "Transactional Data":
                self.radio_respond_box.children = [self.primary_feature, self.filter_type, self.feature_select_transcational]
                filter_event(None)
            else:
                self.radio_respond_box.children = [self.feature_select_tabular]
                feature_tabular_event({"new": self.feature_select_tabular.value})
        def primary_feature_event(value):
            options = list(self.data[value["new"]].unique())
            self.filter_type.options = options
            self.feature_select_transcational.options = [i for i in self.features if i!=value["new"]]
        def filter_event(value):
            cols = self.feature_select_transcational.value
            self.processedData = self.data[self.data[self.primary_feature.value]==self.filter_type.value][cols]
            self.processedPseudoData = None
            if self.pseudoData is not None:
                self.processedPseudoData = self.pseudoData[self.pseudoData[self.primary_feature.value]==self.filter_type.value][cols]
            with self.out:
                clear_output()
                kde_hist_plot(self.processedData, self.processedPseudoData, cols)
        def feature_transactional_event(value):
            cols = value["new"]
            self.processedData = self.data[self.data[self.primary_feature.value]==self.filter_type.value][cols]
            self.processedPseudoData = None
            if self.pseudoData is not None:
                self.processedPseudoData = self.pseudoData[self.pseudoData[self.primary_feature.value]==self.filter_type.value][cols]
            with self.out:
                clear_output()
                kde_hist_plot(self.processedData, self.processedPseudoData, cols)
        def feature_tabular_event(value):
            cols = value["new"]
            self.processedData = self.data[self.data[cols]!="NA"][cols] # self.data[cols]
            self.processedPseudoData = None
            if self.pseudoData is not None:
                self.processedPseudoData = self.pseudoData[self.pseudoData[cols]!="NA"][cols] # self.pseudoData[cols]
            with self.out:
                clear_output()
                kde_hist_plot(self.data[self.data[cols]!="NA"][cols], self.processedPseudoData, cols)
        
        with self.tab:
            clear_output()
            self.datatype_radio.observe(radio_event, names="value")
            self.primary_feature.observe(primary_feature_event, names="value")
            self.filter_type.observe(filter_event, names="value")
            self.feature_select_transcational.observe(feature_transactional_event, names="value")
            self.feature_select_tabular.observe(feature_tabular_event, names="value")
            cols = self.feature_select_tabular.value
            self.processedData = self.data[self.data[cols]!="NA"][cols] # self.data[cols]
            if self.pseudoData is not None:
                self.processedPseudoData = self.pseudoData[self.pseudoData[cols]!="NA"][cols] # self.pseudoData[cols]
            with self.out:
                clear_output()
                kde_hist_plot(self.processedData, self.processedPseudoData, cols)
            display(self.box)
            display(self.out)

def categorical_visualization(data, pseudoData, discreteFeatures, tab):
    with tab:
        style = {"description_width": "100px"}
        layout = {"width": "400px"}
        options = discreteFeatures
        w = widgets.Dropdown(
            options=options,
            value=options[0],
            description="<b>Select Feature</b>",
            disabled=False,
            style=style,
            layout=layout,
        )
        display(w)
        out = widgets.Output()
        with out:
            print("innnn category")
            cols = options[0]
            if len(cols) <= 1:
                display(HTML(f"<h4>Discrete Data not found in this dataset...</h4>"))
            else:
                print(data[data[cols]!="NA"][cols].unique())
                if pseudoData is not None:
                    kde_hist_plot(data[data[cols]!="NA"][cols], pseudoData[pseudoData[cols]!="NA"][cols], cols)
                else:
                    kde_hist_plot(data[data[cols]!="NA"][cols], None, cols)

        def change_feature_graph(_):
            with out:
                clear_output()
                cols = w.value
                if len(cols) <= 1:
                    display(
                        HTML(f"<h4>Discrete Data not found in this dataset...</h4>")
                    )
                else:
                    print(data[data[cols]!="NA"][cols].unique())
                    if pseudoData is not None:
                        kde_hist_plot(data[data[cols]!="NA"][cols], pseudoData[pseudoData[cols]!="NA"][cols], cols)
                    else:
                        kde_hist_plot(data[data[cols]!="NA"][cols], None, cols)

        w.observe(change_feature_graph, names="value")
        b = widgets.VBox([out])
        display(b)

def location_visualization(data, zip_columns, mode="county", tab =None):
    with tab:
        out = widgets.Output()
        out1 = widgets.Output()
        try:
            temp_data = data.groupby(by=[zip_columns]).count().iloc[:,0]
            temp_data = pd.DataFrame(temp_data).reset_index()
            temp_data.rename(columns = {temp_data.columns[1]:"count"}, inplace=True)
            nomi = pgeocode.Nominatim('us')
            temp_data[zip_columns] = temp_data[zip_columns].astype(str)
            temp_data['zip_mode']=nomi.query_postal_code(temp_data[zip_columns].values).county_name
            temp_data['ZIP_Latitude'] = (nomi.query_postal_code(temp_data[zip_columns].tolist()).latitude)
            temp_data['ZIP_Longitude'] = (nomi.query_postal_code(temp_data[zip_columns].tolist()).longitude)
            # temp_data = temp_data.groupby(by=[zip_columns]).count()
            # temp_data['count'] = temp_data.groupby(by=[zip_columns]).count().iloc[:,0].values
            if mode is not None:
                if mode == "County":
                    temp_data['County']=nomi.query_postal_code(temp_data[zip_columns].values).county_name
                    title = "County Level Distribution"
                else:
                    temp_data['State']=nomi.query_postal_code(temp_data[zip_columns].values).state_name
                    title = "State Level Distribution"
                new_temp_df = temp_data.copy()
                new_temp_df=pd.DataFrame(new_temp_df.groupby(by=[mode]).count().iloc[:,0])
                new_temp_df.reset_index(drop=False,inplace=True)

                new_temp_df['Latitude']=temp_data.groupby(by=[mode]).ZIP_Latitude.mean().values
                new_temp_df['Longitude']=temp_data.groupby(by=[mode]).ZIP_Longitude.mean().values
                if mode == "state":
                    new_temp_df['Latitude']=new_temp_df.groupby(by=[mode]).Latitude.mean().values
                    new_temp_df['Longitude']=new_temp_df.groupby(by=[mode]).Longitude.mean().values
                new_temp_df["count"]= new_temp_df.iloc[:, 1]
            
            with out:
                fig = px.scatter_geo(
                    lon = temp_data['ZIP_Longitude'],
                    lat = temp_data['ZIP_Latitude'],
                    scope = "usa",
                    color =  temp_data['count']
                    )
                fig.update_layout(
                                title = 'Zip Code Level Distribution'
                            )
                fig.show(renderer="svg")
            
            with out1:
                if mode is not None:
                    fig = px.scatter_geo(
                        lon = new_temp_df['Longitude'],
                        lat = new_temp_df['Latitude'],
                        scope = "usa",
                        color =  new_temp_df['count'], 
                        size = new_temp_df['count']
                        )
                    fig.update_layout(title = title )
                    fig.show(renderer="svg")
        except:
            # with out:
            text = "Current framework only supports ZIP code Geo-visualizations for United States(US)."
            out  = widgets.HTML("<h4 style='color:blue;text-align:center'>**{}**</h4>".format(text))
        display(widgets.VBox([out, out1]))
        

class pseudoDataProfiling:
    def __init__(
        self,
        originalData,
        processedData,
        numericalColumns,
        discreteColumns,
        nameColumns,
        addressColumns,
        zip_columns = None,
        mode = None,
        fileNamePrefix = None,
        new_feature_names = None,
        sensitive_features = None
    ):
        pbar = tqdm(total=100)
        pbar.update(10)
        descriptive_tab = widgets.Output()
        n_visualization_tab = widgets.Output()
        c_visualization_tab = widgets.Output()
        outlier_tab = widgets.Output()
        layout = widgets.Layout(
            display="flex", flex_flow="column", align_items="stretch", width="99%"
        )
        tab = widgets.Tab(
            children=[descriptive_tab, n_visualization_tab, c_visualization_tab, outlier_tab],
            layout=layout,
        )
        pbar.update(20)
        if zip_columns is not None and len(zip_columns)>0:
            l_visualization_tab = widgets.Output()
            tab = widgets.Tab(
                children=[descriptive_tab, n_visualization_tab, c_visualization_tab, l_visualization_tab, outlier_tab],
                layout=layout,
            )
            tab.set_title(3, "Zip Codes Viz.")
            location_visualization(originalData, zip_columns[0], mode,  tab = l_visualization_tab)
        # data snapshot, desctiptive statistics
        pbar.update(20)
        tab.set_title(0, "Data Profiling")
        descriptive(originalData, processedData, new_feature_names, descriptive_tab)

        # correlation matrix, data visualization
        pbar.update(20)
        tab.set_title(1, "Numerical Viz.")
        
        visualization(
            originalData, processedData, numericalColumns, numericalColumns, "Numerical Data not found in this dataset...",n_visualization_tab
        ).app()

        pbar.update(30)
        tab.set_title(2, "Categorical Viz.")
        visualization(
            originalData, processedData, discreteColumns, discreteColumns, "Discrete Data not found in this dataset...",c_visualization_tab
        ).app()

        tab.set_title(len(tab.children)-1, "Outlier Analysis")
        outlier(data=originalData.loc[:, originalData.columns.isin(numericalColumns + discreteColumns)], pseudoData=None, cat_features=discreteColumns, aggregated_cat_features=None, fileNamePrefix=fileNamePrefix, tab=outlier_tab)
        if processedData is not None:
            p_reporting_tab = widgets.Output()
            tab.children += (p_reporting_tab, )
            tab.set_title(1, "Numerical Comp.")
            tab.set_title(2, "Categorical Comp.")
            tab.set_title(len(tab.children)-1, "Risk Assessment")
            privacyReporting(processedData, sensitive_features, p_reporting_tab)
            if sensitive_features is None:
                sensitive_features = []
            outlier(data=originalData.loc[:, originalData.columns.isin(numericalColumns + discreteColumns)], 
                    pseudoData=processedData.loc[:, originalData.columns.isin(numericalColumns + discreteColumns)], cat_features=discreteColumns, 
                    aggregated_cat_features=discreteColumns + list(set(numericalColumns) & set(sensitive_features)), fileNamePrefix=fileNamePrefix, tab=outlier_tab)
        display(tab)
        pbar.close()