import os
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from IPython.display import HTML, clear_output, display
from ipywidgets import ButtonStyle, widgets

from ..src.data_handling import process_data
from .html_codes import *
from .psedonym_reporting import pseudoDataProfiling
from .pseudonymization_fe import pseudonymization
from .reporting import dataProfiling
from .training_generation_fe import trainingGeneration

def get_discarded_values(data):
    discarded_features = [col for col in data.columns if data[col].isnull().all()]
    if len(discarded_features)==0:
        discarded_features = [None]
    return discarded_features
def get_identifiers(features):
    dId = ['name','address','phone','cellphone','fax','email','insurance','social','security', 'city','state','num','no',
                    'ip','url','medical record','beneficiary num','no.','lincense','account','id','number','certificate','license','national','serial','identifier','code']
    return set([i for i in features for j in dId if i.lower().__contains__(j)])

def save_feature_selection(selection_data, filename):
    if not os.path.exists("privacy/configurations/dataset_config"):
        os.makedirs("privacy/configurations/dataset_config")
    with open(os.path.join("privacy/configurations/dataset_config", filename), "w+") as fhandle:
        json.dump(selection_data, fhandle)

def load_features(filename):
    if os.path.exists(os.path.join("privacy/configurations/dataset_config", filename)):
        with open(os.path.join("privacy/configurations/dataset_config", filename), "r") as fhandle:
            load_config_feat_types = json.load(fhandle)
        return load_config_feat_types
    return None

def seggregate_numeric_discrete(data, discarded_features, config_feattype_filename, max_count=15):
    try:
        load_config_feat_types = load_features(config_feattype_filename)
        if load_config_feat_types  is not None:
            discrete_columns = load_config_feat_types["discrete"]
            continous_columns = load_config_feat_types["continous"]
            name_columns = load_config_feat_types["name"]
            zip_columns = load_config_feat_types["zip"]
            date_columns = load_config_feat_types["date"]
            address_columns = load_config_feat_types["address"]
            discard_columns = load_config_feat_types["discard"]
            return discrete_columns, continous_columns, name_columns, address_columns, date_columns, zip_columns, discard_columns
        else:
            discarded_features = get_discarded_values(data)
            discrete_columns = data.nunique() < max_count
            continous_columns = np.invert(discrete_columns)
            discrete_columns = data.loc[:, discrete_columns].columns
            continous_columns = data.loc[:, continous_columns].columns
            continous_columns = list(set(continous_columns) - set(discarded_features))
            discrete_columns = list(set(discrete_columns) - set(discarded_features))
    except:
        discrete_columns = [None]
        continous_columns = [None]
    discrete_columns = list(discrete_columns)
    continous_columns = list(continous_columns)
    features = list(data.columns)
    dId = ['name','address','phone','cellphone','fax','email','insurance','social','security', 'city','state','num','no',
                    'ip','url','medical record','beneficiary num','no.','lincense','account','id','number','certificate','license','national','serial','identifier','code']
    name_columns = list(set([i for i in features for j in dId if i.lower().__contains__(j)]))
    address_columns = [
        i for i in list(continous_columns) if i!= None and i.lower().__contains__("addr")
    ]
    date_columns = [
        i for i in list(continous_columns) if i!= None and i.lower().__contains__("date")
    ]
    zip_columns = [
        i for i in list(continous_columns) if i!= None and i.lower().__contains__("zip")
    ]
    
    continous_columns = list(set(continous_columns) - set(name_columns))
    discrete_columns = list(set(discrete_columns) - set(name_columns))
    address_columns = list(set(address_columns) - set(name_columns))
    zip_columns = list(set(zip_columns) - set(name_columns))

    name_columns = name_columns if len(name_columns) > 0 else [None]
    address_columns = address_columns if len(address_columns) > 0 else [None]
    date_columns = date_columns if len(date_columns) > 0 else [None]
    zip_columns = zip_columns if len(zip_columns) > 0 else [None]
    continous_columns = [i for i in continous_columns if i not in name_columns]
    continous_columns = [i for i in continous_columns if i not in address_columns and i not in zip_columns and i not in date_columns]
    
    return discrete_columns, continous_columns, name_columns, address_columns, date_columns, zip_columns, discarded_features


def set_headings(heading):
    out = widgets.Output(layout={"width": "100%"})
    out.append_stdout(heading)
    return out


def value_remover_from_list(
    value,
    continous_list,
    discrete_list,
    name_list,
    address_list,
    feature_list,
    discard_list,
    date_list,
    zip_list
):
    if value in continous_list:
        continous_list.remove(value)
    elif value in discrete_list:
        discrete_list.remove(value)
    elif value in name_list:
        name_list.remove(value)
    elif value in address_list:
        address_list.remove(value)
    elif value in feature_list:
        feature_list.remove(value)
    elif value in discard_list:
        discard_list.remove(value)
    elif value in date_list:
        date_list.remove(value)
    elif value in zip_list:
        zip_list.remove(value)
    return (
        continous_list,
        discrete_list,
        name_list,
        address_list,
        feature_list,
        discard_list,
        date_list,
        zip_list
    )


def processing_values(
    selected_list,
    continous_list,
    discrete_list,
    name_list,
    address_list,
    feature_list,
    discard_list,
    date_list,
    zip_list,
    change,
):
    for i in selected_list:
        if i == None:
            continue
        if change == "Continous":
            (
                continous_list,
                discrete_list,
                name_list,
                address_list,
                feature_list,
                discard_list,
                date_list,
                zip_list
            ) = value_remover_from_list(
                i,
                continous_list,
                discrete_list,
                name_list,
                address_list,
                feature_list,
                discard_list,
                date_list,
                zip_list
            )
            continous_list += [i]
            if None in continous_list:
                continous_list.remove(None)
        elif change == "Discrete":
            (
                continous_list,
                discrete_list,
                name_list,
                address_list,
                feature_list,
                discard_list,
                date_list,
                zip_list
            ) = value_remover_from_list(
                i,
                continous_list,
                discrete_list,
                name_list,
                address_list,
                feature_list,
                discard_list,
                date_list,
                zip_list
            )
            discrete_list += [i]
            if None in discrete_list:
                discrete_list.remove(None)
        elif change == "DID/String":
            (
                continous_list,
                discrete_list,
                name_list,
                address_list,
                feature_list,
                discard_list,
                date_list,
                zip_list
            ) = value_remover_from_list(
                i,
                continous_list,
                discrete_list,
                name_list,
                address_list,
                feature_list,
                discard_list,
                date_list,
                zip_list
            )
            name_list += [i]
            if None in name_list:
                name_list.remove(None)
        elif change == "Address":
            (
                continous_list,
                discrete_list,
                name_list,
                address_list,
                feature_list,
                discard_list,
                date_list,
                zip_list
            ) = value_remover_from_list(
                i,
                continous_list,
                discrete_list,
                name_list,
                address_list,
                feature_list,
                discard_list,
                date_list,
                zip_list
            )
            address_list += [i]
            if None in address_list:
                address_list.remove(None)
        elif change == "Date":
            (
                continous_list,
                discrete_list,
                name_list,
                address_list,
                feature_list,
                discard_list,
                date_list,
                zip_list
            ) = value_remover_from_list(
                i,
                continous_list,
                discrete_list,
                name_list,
                address_list,
                feature_list,
                discard_list,
                date_list,
                zip_list
            )
            date_list += [i]
            if None in date_list:
                date_list.remove(None)
        elif change == "Zip Codes":
            (
                continous_list,
                discrete_list,
                name_list,
                address_list,
                feature_list,
                discard_list,
                date_list,
                zip_list
            ) = value_remover_from_list(
                i,
                continous_list,
                discrete_list,
                name_list,
                address_list,
                feature_list,
                discard_list,
                date_list,
                zip_list
            )
            zip_list += [i]
            if None in zip_list:
                zip_list.remove(None)
        elif change == "Discard":
            (
                continous_list,
                discrete_list,
                name_list,
                address_list,
                feature_list,
                discard_list,
                date_list,
                zip_list
            ) = value_remover_from_list(
                i,
                continous_list,
                discrete_list,
                name_list,
                address_list,
                feature_list,
                discard_list,
                date_list,
                zip_list
            )
            discard_list += [i]
            if None in discard_list:
                discard_list.remove(None)
    return (
        continous_list,
        discrete_list,
        name_list,
        address_list,
        feature_list,
        discard_list,
        date_list,
        zip_list
    )


class dataProcessing:
    def __init__(self, data, filename_prefix="test", psuedoMode=False, **kwargs):
        self.df = data
        self.discrete, self.continous, self.name, self.address, self.date, self.zipcodes, self.discard = (
            [None],
            [None],
            [None],
            [None],
            [None],
            [None], 
            [None]
        )
        self.discard_bckup = get_discarded_values(self.df)
        self.features_bckup = list(set(list(self.df.columns.values)) - set(self.discard_bckup))
        self.features = deepcopy(self.features_bckup)
        self.filename_prefix = filename_prefix
        self.psuedoMode = psuedoMode
        self.s3_details = kwargs

    def create_selection_widgets(self):
        self.feature_headings = widgets.HTML("<b>Features:</b>")
        self.features_selection = widgets.SelectMultiple(
            options=self.features,
            value=[],
            rows=7,
            disabled=False,
            layout={"width": "80%"},
        )
        self.discarded_heading = widgets.HTML("<b>Discarded Features:</b>")
        self.discarded_selection = widgets.SelectMultiple(
            options=self.discard,
            value=[],
            rows=6,
            disabled=False,
            layout={"width": "80%"},
        )
        self.continous_headings = widgets.HTML("<b>Continous:</b>")
        self.continous_selection = widgets.SelectMultiple(
            options=self.continous,
            value=[],
            rows=6,
            disabled=False,
            layout={"width": "90%"},
        )
        self.discrete_headings = widgets.HTML("<b>Discrete:</b>")
        self.discrete_selection = widgets.SelectMultiple(
            options=self.discrete,
            value=[],
            rows=6,
            disabled=False,
            layout={"width": "90%"},
        )
        self.name_heading = widgets.HTML("<b>Direct Identifiers (DID)/String:</b>")
        self.name_selection = widgets.SelectMultiple(
            options=self.name, value=[], rows=2, disabled=False, layout={"width": "90%"}
        )
        self.address_headings = widgets.HTML("<b>Address:</b>")
        self.address_selection = widgets.SelectMultiple(
            options=self.address,
            value=[],
            rows=2,
            disabled=False,
            layout={"width": "90%"},
        )
        self.date_headings = widgets.HTML("<b>Date:</b>")
        self.date_selection = widgets.SelectMultiple(
            options=self.date, value=[], rows=2, disabled=False, layout={"width": "90%"}
        )
        self.zip_headings = widgets.HTML("<b>Zip Codes:</b>")
        self.zip_selection = widgets.SelectMultiple(
            options=self.zipcodes,
            value=[],
            rows=2,
            disabled=False,
            layout={"width": "90%"},
        )

    def create_button_widgets(self):
        self.reset_button = widgets.Button(
            description=f"Reset",
            style=ButtonStyle(button_color="orange"),
            layout={"width": "50%"},
            tooltip="Reset",
        )
        self.feature_selection_dropdown = widgets.Dropdown(
            description="<b>Select Feature Type:</b>",
            options=["--None--", "Continous", "Discrete", "DID/String", "Address", "Date", "Zip Codes","Discard"],
            values=["--None--"],
            layout={"width": "80%"},
            style={"description_width": "150px"},
            title="Select Feature Type:",
        )
        self.preselect_button = widgets.Button(
            description=f"Auto-Select",
            style=ButtonStyle(button_color="orange"),
            layout={"width": "50%"},
            tooltip="Auto-Select",
        )
        self.finish_button = widgets.Button(
            description=f"Finalize Feature Selection & Explore Data",
            style=ButtonStyle(button_color="orange"),
            layout={"width": "40%"},
            tooltip="Finalize Feature Selection & Explore Data",
        )

        self.reset_button.on_click(self.reset)
        self.feature_selection_dropdown.observe(self.on_change)
        self.finish_button.on_click(self.finish_data_selection)
        self.preselect_button.on_click(self.preselect)

    def display_data_processing(self):
        self.create_selection_widgets()
        self.create_button_widgets()
        clear_output()
        
        self.data_processing_output = widgets.Output()
        left_box_layout = widgets.Layout(
            display="flex", flex_flow="column", align_items="flex-start", width="60%"
        )
        right_box_layout = widgets.Layout(
            display="flex",
            flex_flow="column",
            align_items="flex-start",
            justify_content="center",
            width="60%",
        )
        output_finish_box_layout = widgets.Layout(
            display="flex",
            flex_flow="column",
            align_items="center",
            width="100%",
            justify_content="center",
        )

        feature_box = widgets.VBox(
            [
                self.feature_headings,
                self.features_selection,
                self.discarded_heading,
                self.discarded_selection,
            ],
            layout=left_box_layout,
        )

        button_box = widgets.VBox(
            [self.preselect_button, self.feature_selection_dropdown, self.reset_button],
            layout=output_finish_box_layout,
        )

        disc_cont_box = widgets.VBox(
            [
                self.continous_headings,
                self.continous_selection,
                self.name_heading,
                self.name_selection,
                self.date_headings,
                self.date_selection,
            ],
            layout=right_box_layout,
        )
        name_addr_box = widgets.VBox(
            [
                self.discrete_headings,
                self.discrete_selection,
                self.address_headings,
                self.address_selection,
                self.zip_headings,
                self.zip_selection,
            ],
            layout=right_box_layout,
        )
        self.response_box = widgets.VBox()
        h1 = widgets.HBox([feature_box, button_box, disc_cont_box, name_addr_box])
        h2 = widgets.HBox([self.finish_button, self.response_box], layout=output_finish_box_layout)
        h3 = widgets.HBox(
            [self.data_processing_output],
            layout=widgets.Layout(
                display="flex", flex_flow="row", align_items="stretch", width="100%"
            ),
        )
        title = """<hr style="background-color=#00000021"><h1 id="-1.2-Verify-the-data-types-of-features-"><span style="color:orange"> 1.2 Verify the data types of features </span>
                    <a class="anchor-link" href="#-1.2-Verify-the-data-types-of-features-">¶</a>
                </h1>"""
        display(HTML(title))
        display(HTML(preprocessing_html_codes))
        display(h1)
        display(h2)
        display(h3)

    def set_options(self, explode_feature_bckup=False):
        self.continous_selection.options = self.continous
        self.discrete_selection.options = self.discrete
        self.name_selection.options = self.name
        self.address_selection.options = self.address
        self.features_selection.options = self.features
        self.date_selection.options = self.date
        self.zip_selection.options = self.zipcodes
        if explode_feature_bckup:
            self.features = deepcopy(self.features_bckup)
            self.features_selection.options = self.features
        self.discarded_selection.options = self.discard

    def on_change(self, change):
        if change["type"] == "change" and change["name"] == "value":
            if change["new"] == "--None--":
                return None
            if len(self.features_selection.value) > 0:
                (
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes
                ) = processing_values(
                    self.features_selection.value,
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                    change["new"],
                )
                self.features_selection.value = []
                if len(self.features) == 0:
                    self.features += [None]
                self.set_options()
            if len(self.continous_selection.value) > 0:
                (
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                ) = processing_values(
                    self.continous_selection.value,
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                    change["new"],
                )
                self.continous_selection.value = []
                if len(self.continous) == 0:
                    self.continous += [None]
                self.set_options()
            if len(self.discrete_selection.value) > 0:
                (
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                ) = processing_values(
                    self.discrete_selection.value,
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                    change["new"],
                )
                self.discrete_selection.value = []
                if len(self.discrete) == 0:
                    self.discrete += [None]
                self.set_options()
            if len(self.name_selection.value) > 0:
                (
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                ) = processing_values(
                    self.name_selection.value,
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                    change["new"],
                )
                self.name_selection.value = []
                if len(self.name) == 0:
                    self.name += [None]
                self.set_options()
            if len(self.address_selection.value) > 0:
                (
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                ) = processing_values(
                    self.address_selection.value,
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                    change["new"],
                )
                self.address_selection.value = []
                if len(self.address) == 0:
                    self.address += [None]
                self.set_options()
            if len(self.date_selection.value) > 0:
                (
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                ) = processing_values(
                    self.date_selection.value,
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                    change["new"],
                )
                self.date_selection.value = []
                if len(self.date) == 0:
                    self.date += [None]
                self.set_options()
            if len(self.zip_selection.value) > 0:
                (
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                ) = processing_values(
                    self.zip_selection.value,
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                    change["new"],
                )
                self.zip_selection.value = []
                if len(self.zipcodes) == 0:
                    self.zipcodes += [None]
                self.set_options()
            if len(self.discarded_selection.value) > 0:
                (
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                ) = processing_values(
                    self.discarded_selection.value,
                    self.continous,
                    self.discrete,
                    self.name,
                    self.address,
                    self.features,
                    self.discard,
                    self.date,
                    self.zipcodes,
                    change["new"],
                )
                self.discarded_selection.value = []
                if len(self.discard) == 0:
                    self.discard += [None]
                self.set_options()
        self.feature_selection_dropdown.value = "--None--"

    def reset(self, _):
        self.discrete = [None]
        self.continous = [None]
        self.name = [None]
        self.address = [None]
        self.discard = [None]
        self.date = [None]
        self.zipcodes = [None]
        self.set_options(explode_feature_bckup=True)

    def continous_feat_drag(self, _):
        for i in self.features_selection.value:
            self.continous, self.discrete, self.name, self.address = processing_values(
                i, self.continous, self.discrete, self.name, self.address
            )
        self.set_options()

    def discrete_feat_drag(self, _):
        for i in self.features_selection.value:
            self.discrete, self.continous, self.name, self.address = processing_values(
                i, self.discrete, self.continous, self.name, self.address
            )
        self.set_options()

    def name_feat_drag(self, _):
        for i in self.features_selection.value:
            self.name, self.continous, self.discrete, self.address = processing_values(
                i, self.name, self.continous, self.discrete, self.address
            )
        self.set_options()

    def address_feat_drag(self, _):
        for i in self.features_selection.value:
            self.address, self.continous, self.discrete, self.name = processing_values(
                i, self.address, self.continous, self.discrete, self.name
            )
        self.set_options()

    def preselect(self, _):
        (
            self.discrete,
            self.continous,
            self.name,
            self.address,
            self.date,
            self.zipcodes,
            self.discard
        ) = seggregate_numeric_discrete(self.df, self.discard_bckup, self.filename_prefix + ".json", max_count=15)
        # self.discard = deepcopy(self.discard_bckup)
        self.features_selection.options = []
        self.features = [None]
        self.set_options()

    def finish_data_selection(self, _):
        with self.data_processing_output:
            try:
                clear_output()
                tempList = list(set(self.discard_bckup) - set(self.discard))
                if len(tempList) > 0:
                    error_string = '''<h4 style="color:red;">{} these features do not contain any values!</h4>'''.format(tuple(tempList))
                    self.response_box.children = [widgets.HTML(error_string)]
                    return None
                self.response_box.children = [widgets.HTML(f'<h4 style="color:blue; text-align:center">Data pre-processing in progress...<h4>')]
                if len(self.continous)>1: # self.continous contains [None] as default thus it conditioned with 1
                    features = self.df[self.continous].columns.values
                    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                    string_features = set(features) - set(self.df.select_dtypes(include=numerics).columns.values)
                    if len(string_features)>0:
                        error_string = '''<h4 style="color:red;">{} these continous features contain text values!</h4>'''.format(tuple(string_features))
                        self.response_box.children = [widgets.HTML(error_string)]
                        # print(error_string)
                        return None
                        
                processedDF, encodedDf, discreteLabelEncoders = process_data(
                    data=self.df,
                    numericalColumns=self.continous,
                    discreteColumns=self.discrete,
                    nameColumns=self.name,
                    addressColumns=self.address,
                    dateColumns=self.date,
                    zipColumns=self.zipcodes,
                    psuedoMode=self.psuedoMode
                )
                self.finish_button.disabled = True
                self.preselect_button.disabled = True
                self.reset_button.disabled = True
                clear_output()
                self.response_box.children = [widgets.HTML(f'<h4 style="color:green; text-align:center">Feature Selection Sucess!<h4>')]
                save_feature_selection({"discard":self.discard if len(self.discard) >0 else [None], 
                                        "name":self.name if len(self.name) >0 else [None], 
                                        "continous" : self.continous if len(self.continous) >0 else [None], 
                                        "discrete":self.discrete if len(self.discrete) >0 else [None], 
                                        "zip":self.zipcodes if len(self.zipcodes) >0 else [None],
                                        "date":self.date if len(self.date) >0 else [None],
                                        "address":self.address if len(self.address) >0 else [None]}, self.filename_prefix + ".json")
                self.processedData = processedDF
                self.encodedDf = encodedDf

                self.discreteLabelEncoders = discreteLabelEncoders
                self.discreteColumns = (
                    [] if len(self.discrete) == 0 else self.discrete
                )
                self.nameColumns = [] if len(self.name) == 0 else self.name
                self.addressColumns = [] if len(self.address) == 0 else self.address
                self.numericalColumns = (
                    [] if len(self.continous) == 0 else self.continous
                )
                self.dateColumns = (
                    [] if len(self.date) == 0 else self.date
                )
                self.zipColumns = (
                    [] if len(self.zipcodes) == 0 else self.zipcodes
                )
                if self.psuedoMode:
                    if self.dateColumns is not None:
                        for i in self.dateColumns:
                            self.processedData[i] = pd.to_datetime(self.processedData[i], infer_datetime_format=True,  errors="coerce")
                    title = """<hr style="background-color:#00000021"><h2 id="-1.3-EDA-on-the-Original-Dataset-"><span style="color:orange"> 1.3 EDA on the Original Dataset </span>
                            <a class="anchor-link" href="#-1.3-EDA-on-the-Original-Dataset-">¶</a>
                            </h2>"""
                    display(HTML(title))
                    pseudoDataProfiling(
                        self.processedData,
                        None,
                        self.numericalColumns,
                        self.discreteColumns,
                        self.nameColumns,
                        self.addressColumns,
                        self.zipColumns,
                        mode = None,
                        new_feature_names = None
                    )
                    title = """<hr style="background-color:#00000021"><h2 id="-1.4-pseudonymization-"><span style="color:orange"> 1.4 Data Pseudonymization </span>
                            <a class="anchor-link" href="#-1.4-pseudonymization-">¶</a>
                            </h2>"""
                    display(HTML(title))
                    display(HTML(pseudomization_html_code))
                    pseudonymization(
                        self.processedData,
                        self.numericalColumns,
                        self.discreteColumns,
                        self.nameColumns,
                        self.addressColumns,
                        self.dateColumns,
                        self.zipColumns,
                        self.s3_details
                    ).app()
                    return None
                title = """<hr style="background-color:#00000021"><h2 id="-1.3-EDA-on-the-Original-Dataset-"><span style="color:orange"> 1.3 EDA on the Original Dataset </span>
                            <a class="anchor-link" href="#-1.3-EDA-on-the-Original-Dataset-">¶</a>
                            </h2>"""
                display(HTML(title))
                display(HTML(dataprofiling_html_codes))
                _ = dataProfiling(
                    originalData=self.processedData,
                    originalEncodedData=self.encodedDf,
                    numericalColumns=self.numericalColumns,
                    discreteColumns=self.discreteColumns,
                    nameColumns=self.nameColumns,
                    addressColumns=self.addressColumns,
                    discreteLabelEncoders=self.discreteLabelEncoders,
                )
                title = """<br><hr style="background-color:#00000021"><h2 id="-1.4-Training-the-Generative-Model-"><span style="color:orange"> 1.4 Training the Generative Model  </span>
                            <a class="anchor-link" href="#-1.4-Training-the-Generative-Model-">¶</a>
                            </h2>"""
                display(HTML(title))
                display(HTML(training_html_codes))
                self.model = trainingGeneration(
                    self.encodedDf,
                    self.discreteColumns,
                    self.nameColumns,
                    self.addressColumns,
                    originalData=self.processedData,
                    discreteLabelEncoders=self.discreteLabelEncoders,
                    filename_prefix=self.filename_prefix,
                )
                self.model.train()
            except AssertionError as msg:
                clear_output()
                display(
                    HTML(f'<h4 style="color:red;text-align:center">**{msg}**</h4><hr>')
                )

