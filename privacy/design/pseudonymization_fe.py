import os
import numpy as np
import pandas as pd
import seaborn as sns
import traceback
from ..src.helper import numericalBinning, categoricalSuppression, string_hashing, date_zip_noise_binning
from .psedonym_reporting import pseudoDataProfiling
from IPython.display import HTML, clear_output, display
from ipywidgets import ButtonStyle, Layout, widgets


def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def popup(text):
    display(HTML("<script>alert('{}');</script>".format(text)))
class numericalDiscretizationNoise:
    def __init__(self, df, numerical):
        self.df = df
        self.numerical = numerical
        self.numerical_feature_base = {i:{"bins":[], "freq" : [], "noise":0} for i in self.numerical}
        self.numerical_feature_name_hashing = {i:i for i in self.numerical}

    def update_binners(self,feature, bin_number, max_value, min_value):
        def update(change):
            if change["name"] == "value" and change["type"] == "change":
                if change["new"] >= max_value:
                    text = "Please don't exceed the next bin end value."
                    self.error_box.children = [widgets.HTML("<h4 style='color:red;text-align:center'>**{}**</h4>".format(text))]
                    return None
                if min_value is not None:
                    if change["new"] <= min_value:
                        text = "Please don't go lower than the previous bin end value."
                        self.error_box.children = [widgets.HTML("<h4 style='color:red;text-align:center'>**{}**</h4>".format(text))]
                        return None
                self.error_box.children = []
                self.numerical_feature_base[feature]["bins"][bin_number][1] = change["new"]
                self.numerical_feature_base[feature]["bins"][bin_number+1][0] = change["new"]
                print("before::::", self.numerical_feature_base[feature]["bins"])
                print("Raveling::::",set(np.ravel(self.numerical_feature_base[feature]["bins"])))
                ab = list(set(np.ravel(self.numerical_feature_base[feature]["bins"])))
                ab.sort()
                print("after::::", ab)
                slice_df_feature = pd.cut(self.df[feature], bins=ab)
                slice_df_feature.dropna(inplace=True)
                list_intervals = list(slice_df_feature.unique())
                list_intervals.sort()
                self.numerical_feature_base[feature]["freq"] = []
                for l in list_intervals:
                    self.numerical_feature_base[feature]["freq"].append(len(slice_df_feature[slice_df_feature == l]))
                self.frequency_based_values.children = [widgets.HTML(f'<b>Frequency</b>')] + [widgets.Label(value='{}'.format(self.numerical_feature_base[feature]["freq"][i])) for i in range(self.binning_selection.value)]
                self.fs_childs[bin_number+2].value = change["new"]
        return update
    
    def create_binners(self, feature, set_binners = False, fill_only_values = False):
        if not set_binners:
            slice_df_feature = pd.cut(self.df[feature], bins=self.binning_selection.value)
            slice_df_feature.dropna(inplace=True)
            list_intervals = list(slice_df_feature.unique())
            list_intervals.sort()
            self.numerical_feature_base[feature] = {"bins":[], "freq" : [], "noise":0}
            self.info_box.children = []
            for l in list_intervals:
                self.numerical_feature_base[feature]["bins"].append([round(l.left, 2), round(l.right, 2)])
                self.numerical_feature_base[feature]["freq"].append(len(slice_df_feature[slice_df_feature == l]))
        if fill_only_values:
            return None
        minimum_value = self.df[feature].min() #self.numerical_feature_base[feature]["bins"][0][0]
        maximum_value = self.df[feature].max() # self.numerical_feature_base[feature]["bins"][-1][1]
        self.fs_childs = [widgets.HTML(f'<b>Start</b>')]
        self.fe_childs = [widgets.HTML(f'<b>End</b>')]
        self.fv_childs = [widgets.HTML(f'<b>Frequency</b>')]
        bin_labels_list = []
        for i in range(len(self.numerical_feature_base[feature]["freq"])):
        # for i in range(len()):
            start_value = self.numerical_feature_base[feature]["bins"][i][0]
            end_value = self.numerical_feature_base[feature]["bins"][i][1]
            layout = {"width":"90%"}
            start_temp_textbox = widgets.BoundedFloatText(value=start_value,  min=minimum_value, max=maximum_value, disabled=True, layout=layout)
            end_temp_textbox = widgets.BoundedFloatText(value=end_value, min=minimum_value, max=maximum_value, layout=layout)
            if i==0:
                start_temp_textbox.value = minimum_value #self.df[feature].min()
            if i+1 == len(self.numerical_feature_base[feature]["freq"]):
                # end_temp_textbox.max = maximum_value
                end_temp_textbox.disabled = True
                end_temp_textbox.value = maximum_value #self.df[feature].max()
            else:
                previous_bin_min_value = self.numerical_feature_base[feature]["bins"][i-1][1] if i>0 else None 
                next_bin_max_value = self.numerical_feature_base[feature]["bins"][i+1][1]
                end_temp_textbox.observe(self.update_binners(feature, i, next_bin_max_value, previous_bin_min_value))
                # end_temp_textbox.max=self.numerical_feature_base[feature]["bins"][i+1][1]
            freq_value_label = widgets.Label(value='{}'.format(self.numerical_feature_base[feature]["freq"][i]))
            bin_labels_list.append(widgets.HTML(f'<b>Bin {i+1}:</b>'))
            self.fs_childs.append(start_temp_textbox)
            self.fe_childs.append(end_temp_textbox)
            self.fv_childs.append(freq_value_label)
        if self.binning_selection.value > len(self.numerical_feature_base[feature]["freq"]):
            text = "This feature has a skewed distribution, only {} bins possible for the current selection.".format(len(self.numerical_feature_base[feature]["freq"]))
            self.info_box.children = [widgets.HTML("<h4 style='color:blue;text-align:center'>**{}**</h4>".format(text))]
        self.bin_labels.children = bin_labels_list
        self.frequency_based_values.children = self.fv_childs
        self.frequency_start_values.children = self.fs_childs
        self.frequency_end_values.children = self.fe_childs
            
    def numerical_discretization_noise_app(self):
        if len(self.numerical)==0:
            h = widgets.HTML("<h4 style='text-align:center; width:100%; color:red'> There is no continous (numerical) data available in your dataset... </h4>")
            return h
        def on_change_binning(change):
            if self.numerical_feature_base == None:
                self.numerical_feature_base = {i:{"bins":[], "freq" : [], "noise":0} for i in self.numerical}
            if change["type"] == "change" and change["name"] == "value":
                if self.binning_selection.value != "--None--":
                    if len(self.feature_selection.value) > 1:
                        for i in self.feature_selection.value:
                            self.create_binners(i, fill_only_values=True)
                    else:
                        self.create_binners(self.feature_selection.value[0])
                else:
                    reset(index=None)
                    for i, val in enumerate(self.feature_selection.value):
                        self.numerical_feature_base[self.feature_selection.value[i]]["bins"] = []
                        self.numerical_feature_base[self.feature_selection.value[i]]["freq"] = []
        def on_change_feature(change):
            if self.numerical_feature_base == None:
                self.numerical_feature_base = {i:{"bins":[], "freq" : [], "noise":0} for i in self.numerical}
            if change["type"] == "change" and change["name"] == "value":
                feature_name_text_box.value = ""
                # if len(self.feature_selection.value)>1:
                #     feature_name_text_box.disabled = True
                #     # for i in self.feature_selection.value:
                #     if len(self.numerical_feature_base[change["new"][0]]["bins"]) > 0:
                #         self.binning_selection.value = len(self.numerical_feature_base[change["new"][0]]["bins"]) 
                #         self.radio_buttons.value = "Binning" 
                #     else:
                #         self.radio_buttons.value = "Noise"
                #         self.noise_selection.value = self.numerical_feature_base[change["new"][0]]["noise"]
                            
                # else:
                if len(self.feature_selection.value)>1:
                    feature_name_text_box.disabled = True
                else:
                    feature_name_text_box.disabled = False
                if len(self.numerical_feature_base[change["new"][0]]["bins"]) > 0:
                    self.radio_buttons.value = "Binning"
                    self.binning_selection.value = len(self.numerical_feature_base[change["new"][0]]["bins"])
                    if len(self.feature_selection.value)==1:
                        self.create_binners(change["new"][0], set_binners=True)
                else:
                    self.radio_buttons.value = "Noise"
                    self.noise_selection.value = self.numerical_feature_base[change["new"][0]]["noise"]
        def on_change_radio(change):
            if change["type"] == "change" and change["name"] == "value":
                if change["new"]=="Noise":
                    for i, val in enumerate(self.feature_selection.value):
                        reset(i)
                        self.numerical_feature_base[self.feature_selection.value[i]]["bins"] = []
                        self.numerical_feature_base[self.feature_selection.value[i]]["freq"] = []
                        self.right_side_box.children = [self.noise_box]
                else:
                    for i, val in enumerate(self.feature_selection.value):
                        reset(i)
                        self.numerical_feature_base[self.feature_selection.value[i]]["noise"] = 0
                        self.right_side_box.children = [self.binning_box]
        def on_change_noise(change):
            if change["type"] == "change" and change["name"] == "value":
                for i in self.feature_selection.value:
                    self.numerical_feature_base[i]["noise"] = change["new"]
        
        def reset(index = None, hard_reset=False):
            self.binning_selection.value = "--None--"
            if index is not None:
                self.noise_selection.value = self.numerical_feature_base[self.feature_selection.value[index]]["noise"]
            self.frequency_based_values.children = []
            self.frequency_start_values.children = []
            self.frequency_end_values.children = []
            self.bin_labels.children = []
            def reset_event(_):
                if hard_reset:
                    self.numerical_feature_base = {i:{"bins":[], "freq" : [], "noise":0} for i in self.numerical}
            return reset_event
        def feature_name_change_event(_):
            if feature_name_text_box.value != "":
                self.numerical_feature_name_hashing[self.feature_selection.value[0]] = feature_name_text_box.value
            
        self.feature_selection_tag = widgets.HTML("<b>Select Feature:</b>")
        self.feature_selection = widgets.SelectMultiple(options=list(self.numerical_feature_base.keys()),
                                          description="", rows=10)
        self.feature_selection.observe(on_change_feature)

        feature_tag = widgets.HTML("<b>Change Feature Name:</b>")
        feature_name_text_box = widgets.Text(placeholder = "Please mention new feature name") 
        feature_name_text_box.observe(feature_name_change_event)
        
        feature_box = widgets.VBox([self.feature_selection_tag, self.feature_selection, feature_tag, feature_name_text_box])
        verical_line_html = '''<style>
                .vertical {
                    border-left: 2px solid black;
                    height: 230px;
                    position:absolute;
                    left: 50%;
                }
            </style><div class = "vertical"></div>'''
        vertical_line = widgets.HTML(verical_line_html)
        radio_button_tag = widgets.HTML("<b>Select Method:</b>")
        self.radio_buttons = widgets.RadioButtons(options = ["Noise", "Binning"], layout={"margin":"60px 0px 0px 10px", "width":"150px"})
        self.radio_buttons.observe(on_change_radio)
        radio_button_box = widgets.VBox([radio_button_tag, self.radio_buttons])

        noise_tag = widgets.HTML("<b>Add Noise (in %):</b>")
        self.noise_selection = widgets.Dropdown(options = list(range(0,101,10)))
        self.noise_selection.observe(on_change_noise)
        self.noise_box = widgets.VBox([noise_tag, self.noise_selection])
        
        binning_tag = widgets.HTML("<b>Select Bin:</b>")
        binning_options = ["--None--"] + list(range(1,11))
        self.binning_selection = widgets.Dropdown(options = binning_options)
        self.binning_selection.observe(on_change_binning)
        self.frequency_based_values = widgets.VBox([], layout={"width":"75%"})
        self.frequency_start_values = widgets.VBox([])
        self.frequency_end_values = widgets.VBox([])
        self.bin_labels = widgets.VBox([])
        self.binning_custom_box = widgets.HBox([self.frequency_start_values, self.frequency_end_values, self.frequency_based_values])
        self.binning_box = widgets.VBox([binning_tag, self.binning_selection, self.binning_custom_box])
        self.error_box = widgets.VBox()
        self.info_box = widgets.VBox()
        self.right_side_box = widgets.VBox([self.noise_box])
        h = widgets.HBox([feature_box, vertical_line, radio_button_box, self.right_side_box], layout=widgets.Layout(width="910px"))
        h = widgets.VBox([h, self.info_box, self.error_box])
        return h

class categoricalSupression:
    def __init__(self, df, discrete):
        self.df = df
        self.discrete = discrete       
        self.discrete_feature_base = {i:{"group_name":[], "bins":{}} for i in self.discrete}
        self.categorical_feature_name_hashing = {i:i for i in discrete}
        
    def on_textbox_update(self, i):
        def on_update(change):
            if change["type"] == "change" and change["name"] == "value":
                self.discrete_feature_base[self.categorical_feature_dropdown.value]["group_name"][i] = change["new"]
                dropdown_old_value = self.group_drop_down.value
                self.group_drop_down.options = ["--None--"] + self.discrete_feature_base[self.categorical_feature_dropdown.value]["group_name"]
                if dropdown_old_value == change["old"]:
                    self.group_drop_down.value = change["new"]
        return on_update
    def update_event(self,change):
            if change["new"] == "--None--":
                return None
            index = self.discrete_feature_base[self.categorical_feature_dropdown.value]["group_name"].index(self.group_drop_down.value)
            keys = list(self.discrete_feature_base[self.categorical_feature_dropdown.value]["bins"])
            selected_key = keys[index]
            for i in range(self.categorical_binning.value):
                if len(self.select_box_childs[i].value)>0:            
                    selected_values = self.select_box_childs[i].value
                    new_selected_key = keys[i]
                    x = self.discrete_feature_base[self.categorical_feature_dropdown.value]["bins"][new_selected_key]
                    self.discrete_feature_base[self.categorical_feature_dropdown.value]["bins"][new_selected_key] = [item for item in x if item not in selected_values]
                    self.discrete_feature_base[self.categorical_feature_dropdown.value]["bins"][selected_key] += selected_values
                    self.select_box_childs[i].value = []
            for i in range(self.categorical_binning.value):
                self.select_box_childs[i].options = self.discrete_feature_base[self.categorical_feature_dropdown.value]["bins"][f'Group {i}']
            self.group_drop_down.value = "--None--"
    def create_binners(self, set_binners = False):
        if not set_binners:
            list_values = list(self.df[self.categorical_feature_dropdown.value].value_counts().index)[::-1]
            temp_list = list(split(list_values, self.categorical_binning.value))
            self.discrete_feature_base[self.categorical_feature_dropdown.value]["bins"] = {f'Group {i}': l for i,l in enumerate(temp_list)}
            self.discrete_feature_base[self.categorical_feature_dropdown.value]["group_name"] = [f'Group {i+1}'for i in range(len(temp_list))]
        self.select_box_childs, self.text_box_childs = [], []
        for i in range(self.categorical_binning.value):
            options = self.discrete_feature_base[self.categorical_feature_dropdown.value]["bins"][f'Group {i}']
            text_value = self.discrete_feature_base[self.categorical_feature_dropdown.value]["group_name"][i]
            temp_select_widget = widgets.SelectMultiple(options = options, layout={"width":"100px"})
            temp_text_widget = widgets.Text(value= text_value, layout={"width":"100px"}, placeholder="Group Name")
            temp_text_widget.observe(self.on_textbox_update(i))
            self.select_box_childs.append(temp_select_widget)
            self.text_box_childs.append(temp_text_widget)
        group_dropdown_options = ["--None--"] + self.discrete_feature_base[self.categorical_feature_dropdown.value]["group_name"]
        width_value = 300 if self.categorical_binning.value > 2 else 209
        self.group_drop_down = widgets.Dropdown(options = group_dropdown_options, description = "Change Grouping:", style = {"description_width":"110px"}, layout={"width":"{}px".format(width_value)})
        self.group_drop_down.observe(self.update_event, names="value")
        self.select_box.children = self.select_box_childs
        self.text_box.children= self.text_box_childs
        if self.categorical_binning.value > 1:
            self.group_drop_box.children = [self.group_drop_down]
        else:
            self.group_drop_box.children = []
        
            
    def categorical_suppresser_app(self):
        if len(self.discrete)==0:
            h = widgets.HTML("<h4 style='text-align:center; width:100%; color:red'> There is no categorical data available in your dataset... </h4>")
            return h
        def on_change_feature(change):
            if self.discrete_feature_base == None:
                self.discrete_feature_base = {i:{"group_name":[], "bins":{}} for i in self.discrete}
            if change["type"] == "change" and change["name"] == "value":
                self.binning_options = ["--None--"] + list(range(1, len(self.df[self.categorical_feature_dropdown.value].unique())+1))
                self.categorical_binning.options = self.binning_options
                self.categorical_binning.value = self.binning_options[0]
                bin_value_current_feature = len(self.discrete_feature_base[self.categorical_feature_dropdown.value]["bins"])
                reset()
                feature_name_text_box.value = ""
                if bin_value_current_feature > 0:
                    self.categorical_binning.value = bin_value_current_feature
                    self.create_binners(set_binners=True)
        def on_change_binning(change):
            if change["type"] == "change" and change["name"] == "value":
                if change["new"] != "--None--":
                    self.create_binners()
                else:   
                    reset()
        def reset(hard_reset=False):
            self.select_box.children = []
            self.text_box.children = []
            self.group_drop_box.children = []
            self.discrete_feature_base[self.categorical_feature_dropdown.value]["bins"] = {}
            self.discrete_feature_base[self.categorical_feature_dropdown.value]["group_name"] = []
            # print("calling")
            # self.discrete_feature_base[self.categorical_feature_dropdown.value] = {"group_name":[], "bins":{}}
            def reset_event(_):
                print("in reset_event")
                if hard_reset:
                    self.discrete_feature_base = {i:{"group_name":[], "bins":{}} for i in self.discrete}
            return reset_event

        def feature_name_change_event(_):
            if feature_name_text_box.value != "":
                self.categorical_feature_name_hashing[self.categorical_feature_dropdown.value] = feature_name_text_box.value
        feature_selection_tag = widgets.HTML("<b>Select Feature:</b>")
        self.categorical_feature_dropdown = widgets.Select(
                                                options=list(self.discrete_feature_base.keys()),
                                                rows = 10, layout={"width":"98%"})
        feature_tag = widgets.HTML("<b>Change Feature Name:</b>")
        feature_name_text_box = widgets.Text(placeholder = "Please mention new feature name", layout={"width":"98%"}) 
        feature_name_text_box.observe(feature_name_change_event)

        feature_box = widgets.VBox([feature_selection_tag, self.categorical_feature_dropdown, feature_tag, feature_name_text_box],
                                   layout={"width":"35%"})
        self.categorical_feature_dropdown.observe(on_change_feature)
        
        self.binning_options =  ["--None--"] + list(range(1, len(self.df[self.categorical_feature_dropdown.value].unique())+1))
        self.categorical_binning = widgets.Dropdown(options = self.binning_options,description='Select No. of Groups:',
                                                   style={"description_width": "150px"},
                                                layout={"width": "50%", "margin":"30px 0px 0px 0px"},)
        self.categorical_binning.observe(on_change_binning)
       
        box_layout = widgets.Layout(display="flex", width="100%")
        bin_box = widgets.VBox([self.categorical_binning,  widgets.HTML("<BR>")], layout= box_layout)
        
        self.select_box = widgets.HBox()
        self.text_box = widgets.HBox()
        self.group_drop_box = widgets.VBox(layout=widgets.Layout(display="flex", width="100%", align_items="center"))
        vbox_multiple_boxes = widgets.VBox([self.select_box, self.text_box], layout=widgets.Layout(display="flex", width="100%", align_items="center"))
        binn_supressor_box = widgets.VBox([bin_box,  vbox_multiple_boxes, self.group_drop_box], layout=box_layout)
        c_h = widgets.HBox([feature_box, binn_supressor_box],layout=widgets.Layout(width="910px"))
        return c_h
class dateZipHashing:
    def __init__(self, data, date, zip):
        self.df = data
        self.date = date
        self.zip = zip
        self.dz_feature_base = {i:{"noise": 0, "aggregation":None} for i in self.date + self.zip}
    def create_widgets(self):
        box_layout = widgets.Layout(display="flex", align_items="center", width="100%")
        self.feature_selection = widgets.Dropdown(options=self.date + self.zip,
                                          description="Select Features:", style = {"description_width":"185px"},layout={"width": "45%"})
        self.feature_selection_box = widgets.VBox([self.feature_selection, widgets.HTML("<BR>")],  layout=box_layout)
        
        if len(self.date) > 0:
            self.radio_buttons = widgets.RadioButtons(id = "Method", options = ["Offset", "Aggregation"], description="Select Method:", style = {"description_width":"185px"},layout={"width": "45%"})
            self.radio_button_box = widgets.VBox([self.radio_buttons], layout = box_layout)
            self.noise_selection = widgets.IntText(id = "noise", min=-30, max=30, value=0, description = "Add Offset (in days):", style = {"description_width":"185px"},layout={"width": "45%"})    
            self.noise_box = widgets.VBox([self.noise_selection], layout = box_layout)
            discrete_options = ["--None--", "Weekday-Weekend", "Week of the year", "Month", "Quarter", "Year"]
            self.date_aggregate_selection = widgets.Dropdown(id = "date_agg", options = discrete_options, value=discrete_options[0], description = "Select Date Aggregation Value:", style = {"description_width":"185px"},layout={"width": "45%"})
            self.date_aggregation_box = widgets.VBox([self.date_aggregate_selection], layout =box_layout)
        else:
            self.radio_button_box = widgets.VBox([], layout = box_layout)
            self.noise_box = widgets.VBox([], layout = box_layout)
            self.date_aggregation_box = widgets.VBox([], layout =box_layout)
        
        
        if len(self.zip)>0:
            discrete_options = ["--None--", "County", "State"]
            self.zip_aggregate_selection = widgets.Dropdown(options = discrete_options, value=discrete_options[0], description = "Select ZIP-Code Aggregation Value:", style = {"description_width":"250px"},layout={"width": "55%", "margin":"0px 0px 0px -97px"})
            self.zip_aggregation_box = widgets.VBox([self.zip_aggregate_selection], layout=box_layout)
        else:
            self.zip_aggregation_box = widgets.VBox([], layout=box_layout)
        

    def app(self):
        if len(self.date)==0 and len(self.zip)==0:
            h = widgets.HTML("<h4 style='text-align:center; width:100%; color:red'> There is no Date & Zip Codes data available in your dataset... </h4>")
            return h
        self.create_widgets()
        def on_change_feature(change):
            if self.dz_feature_base == None:
                self.dz_feature_base = {i:{"noise": 0, "aggregation":None, "hashing":None} for i in zip(self.discrete, self.date, self.zip)}
            if change["type"] == "change" and change["name"] == "value":
                if self.feature_selection.value in self.date:
                    if self.dz_feature_base[self.feature_selection.value]["aggregation"] == None:
                        self.radio_buttons.value = "Offset"
                        self.noise_aggregation_box.children = [self.radio_button_box, self.noise_box]
                    else: 
                        self.radio_buttons.value = "Aggregation"
                        self.noise_aggregation_box.children = [self.radio_button_box, self.date_aggregate_selection]
                else:
                    if self.dz_feature_base[self.feature_selection.value]["aggregation"] != None:
                        self.zip_aggregate_selection.value = self.dz_feature_base[self.feature_selection.value]["aggregation"]
                    else: 
                        self.zip_aggregate_selection.value = "--None--"
                    self.noise_aggregation_box.children = [self.zip_aggregation_box]
        def on_change_radio(change):
            if change["type"] == "change" and change["name"] == "value":
                pass
                if change["new"]=="Offset":
                    self.dz_feature_base[self.feature_selection.value]["aggregation"]  = None
                    self.noise_aggregation_box.children = [self.radio_button_box, self.noise_box]
                else:
                    self.dz_feature_base[self.feature_selection.value]["noise"]  = 0
                    self.noise_aggregation_box.children = [self.radio_button_box, self.date_aggregation_box]
        def on_change_values(change):
            if change["type"] == "change" and change["name"] == "value":
                if self.feature_selection.value in self.date and self.radio_buttons.value == "Offset":
                    self.dz_feature_base[self.feature_selection.value]["noise"] = change["new"]
                elif self.feature_selection.value in self.date and self.radio_buttons.value == "Aggregation":
                    self.dz_feature_base[self.feature_selection.value]["aggregation"] = change["new"]
                    if change["new"] == "--None--":
                        self.dz_feature_base[self.feature_selection.value]["aggregation"] = None
                elif self.feature_selection.value in self.zip:
                    self.dz_feature_base[self.feature_selection.value]["aggregation"] = change["new"]
                    if change["new"] == "--None--":
                        self.dz_feature_base[self.feature_selection.value]["aggregation"] = None
            
        self.feature_selection.observe(on_change_feature)
        if len(self.date) > 0:
            self.radio_buttons.observe(on_change_radio)
            self.noise_selection.observe(on_change_values)
            self.date_aggregate_selection.observe(on_change_values)
        if len(self.zip)>0:
            self.zip_aggregate_selection.observe(on_change_values)
        if len(self.date)>0:
            self.noise_aggregation_box = widgets.VBox([self.radio_button_box, self.noise_box], layout=widgets.Layout(display="flex", align_items="center", width="100%"))
        else:
            self.noise_aggregation_box = widgets.VBox([self.zip_aggregation_box], layout=widgets.Layout(display="flex", align_items="center", width="100%"))
        h = widgets.VBox([self.feature_selection_box, self.noise_aggregation_box], layout=widgets.Layout(width="910px"))
        return h

class hashing:
    def __init__(self, data, numerical_columns, categorical_columns, date_columns):
        self.df = data
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.date_columns = date_columns
        self.hashing_feature_base = {i:{"hashing": False, "masking": False} for i in list(self.df.columns) if i not in self.numerical_columns and i not in self.categorical_columns and i not in self.date_columns}
    def create_widgets(self):
        box_layout = widgets.Layout(display="flex", align_items="center", width="100%")
        feature_selection_tag = widgets.HTML("<b>Select Feature:</b>")
        self.feature_selection = widgets.SelectMultiple(options=list(self.hashing_feature_base.keys()),
                                          description="", rows=10)
        feature_box = widgets.VBox([feature_selection_tag, self.feature_selection])
        verical_line_html = '''<style>
                .vertical {
                    border-left: 2px solid black;
                    height: 165px;
                    position:absolute;
                }
            </style><div class = "vertical"></div>'''
        vertical_line = widgets.HTML(verical_line_html)
        self.feature_selection_box = widgets.HBox([self.feature_selection, vertical_line], layout={"width":"45%"})
        self.drop_down_hash_mask = widgets.Dropdown(options = ["--None--", "Hash", "Mask"], value="--None--", description="Do you want to mask/hash the selected features?", style = {"description_width":"315px"},layout={"width": "70%"})
        self.drop_down_hash_mask_box = widgets.VBox([self.drop_down_hash_mask], layout = box_layout)
        self.mask_value = widgets.IntText(min=0, max=20, description="Please define length of mask:")
        self.mask_value_box = widgets.VBox([], layout = box_layout)
    def app(self):
        self.create_widgets()
        def on_change_feature(change):
            if self.hashing_feature_base == None:
                self.hashing_feature_base =  {i:{"hashing": False, "masking": False} for i in list(self.df.columns) if i not in self.numerical_columns and i not in self.categorical_columns and i not in self.date_columns}
            if change["type"] == "change" and change["name"] == "value":
                self.drop_down_hash_mask.value = "--None--"
                
        def on_change_drop_down_hash_mask(change):
            if change["type"] == "change" and change["name"] == "value":
                if change["new"]=="Hash":
                    for i in self.feature_selection.value:
                        self.hashing_feature_base[i]["hashing"] = True
                elif change["new"]=="Mask":
                    # self.mask_value_box.children = [self.mask_value]
                    for i in self.feature_selection.value:
                        self.hashing_feature_base[i]["masking"] = True
                else:
                    for i in self.feature_selection.value:
                        self.hashing_feature_base[i]["hashing"] = False
                        self.hashing_feature_base[i]["masking"] = False
                        self.mask_value_box = []
                
            
        self.feature_selection.observe(on_change_feature)
        self.drop_down_hash_mask.observe(on_change_drop_down_hash_mask)
        self.noise_aggregation_box = widgets.VBox([self.drop_down_hash_mask_box, self.mask_value_box], layout=widgets.Layout(display="flex", align_items="center", width="100%"))
        h = widgets.HBox([self.feature_selection_box, self.drop_down_hash_mask_box],layout=widgets.Layout(width="910px") )
        return h
class pseudonymization:
    def __init__(self, df, numerical, discrete, name, address, date, zipcodes, s3_details):
        self.df = df
        self.numerical = [] if numerical is None else numerical
        self.discrete = [] if discrete is None else discrete
        self.name = [] if name is None else name
        self.address = [] if address is None else address
        self.date = [] if date is None else date
        self.zipcodes = [] if zipcodes is None else zipcodes
        self.numerical_feature_base = {
            i: {"auto": None, "custom": [], "noise": 0} for i in self.numerical
        }
        self.discrete_feature_base = {
            i: {"auto": None, "custom": {}} for i in self.discrete
        }
        self.newData = None
        self.s3_details = s3_details


    def app(self):
        copied_data = self.df.copy()
        numerical_psuedo_mode = numericalDiscretizationNoise(copied_data, self.numerical)
        categorical_psuedo_mode = categoricalSupression(copied_data, self.discrete)
        date_zip = dateZipHashing(copied_data, self.date, self.zipcodes)
        string_hashing_obj = hashing(copied_data, self.numerical, self.discrete, self.date)
        numerical_accordion_child = numerical_psuedo_mode.numerical_discretization_noise_app()
        categorical_accordion_child = categorical_psuedo_mode.categorical_suppresser_app()
        date_zip_child = date_zip.app()
        string_hashing_child = string_hashing_obj.app()
        accordion_settings = widgets.Accordion(
            children=[numerical_accordion_child, categorical_accordion_child, date_zip_child, string_hashing_child],
            selected_index=0,
            layout=widgets.Layout(
                display="table", flex_flow="column", align_items="center", width="945px"
            ),
        )
        accordion_settings.set_title(0, "Numerical (Noise and Binning)")
        accordion_settings.set_title(1, "Categorical (Level Suppression)")
        accordion_settings.set_title(2, "Date and ZIP-Code (Noise/Aggregation)")
        accordion_settings.set_title(3, "Hashing/Masking of DIDs (Safe Harbor)")
        display(accordion_settings)
        # display(widgets.VBox([accordion_settings], layout =  widgets.Layout(display="flex", align_items="center", width="100%")))
        EDA_out = widgets.Output()

        def perform_click_event(_):
            with EDA_out:
                clear_output()
                text = "Pseudonymization in progress..."
                error.children = [widgets.HTML("<h4 style='color:blue;text-align:center'>**{}**</h4>".format(text))]
                path = filePath.value
                if path.strip() == "":
                    text = "Please enter valid path (such as 'C:/pseudonomized_data.csv')!"
                    error.children = [widgets.HTML("<h4 style='color:red;text-align:center'>**{}**</h4>".format(text))]
                    return None
                elif path[-3:].lower() != "csv":
                    text = "Please enter valid path (such as 'C:/pseudonomized_data.csv' and extension should be 'csv')!"
                    error.children = [widgets.HTML("<h4 style='color:red;text-align:center'>**{}**</h4>".format(text))]
                    return None
                # elif len(numerical_psuedo_mode.error_box.children) > 0 :
                #     text = '''Please check errors in "Numerical (Noise and Binning) section!"'''
                #     error.children = [widgets.HTML("<h4 style='color:red;text-align:center'>**{}**</h4>".format(text))]
                #     return None
                self.numerical_feature_base = numerical_psuedo_mode.numerical_feature_base
                self.discrete_feature_base = categorical_psuedo_mode.discrete_feature_base
                self.newData = numericalBinning(self.df, self.numerical_feature_base)
                self.newData = categoricalSuppression(self.newData, self.discrete_feature_base)
                self.newData = date_zip_noise_binning(self.newData, date_zip.dz_feature_base)
                self.newData = string_hashing(self.newData, string_hashing_obj.hashing_feature_base)
                new_feature_names = {}
                new_feature_names.update(numerical_psuedo_mode.numerical_feature_name_hashing)
                new_feature_names.update(categorical_psuedo_mode.categorical_feature_name_hashing)
                try:
                    datasaver = self.newData.copy()
                    if self.s3_details["s3_access_key"] != "" or self.s3_details["s3_security_key"] != "" or self.s3_details["s3_security_token"] != "":
                        # path = os.path.join(self.s3_details["s3_bucket"], os.path.basename(path))
                        datasaver.rename(columns = new_feature_names, inplace=True)
                        path = self.s3_details["s3_bucket"] + "/" + os.path.basename(path)
                        datasaver.to_csv(path, storage_options={
                                        "key": self.s3_details["s3_access_key"],
                                        "secret": self.s3_details["s3_security_key"],
                                        "token": self.s3_details["s3_security_token"]}, index=False)
                    else:
                        # path = os.path.basename(path)
                        datasaver.rename(columns = new_feature_names, inplace=True)
                        datasaver.to_csv(path, index=False)
                        display(HTML("<h4 style='color:blue;text-align:center'>**File saved at {}**</h4>".format(os.path.join(os.getcwd(), path))))
                except PermissionError:
                    text = "File is already opened in different program!"
                    error.children = [widgets.HTML("<h4 style='color:red;text-align:center'>**{}**</h4>".format(text))]
                    return None
                except:
                    print(traceback.format_exc())
                    text = "Please check file path that you provided!"
                    error.children = [widgets.HTML("<h4 style='color:red;text-align:center'>**{}**</h4>".format(text))]
                    return None
                mode = None
                if len(date_zip.dz_feature_base) > 0:
                    # if len(self.date)>0:
                    #     for i in self.date:
                    #         self.df[i] = pd.to_datetime(self.df[i])
                    if len(self.zipcodes)>0 and not string_hashing_obj.hashing_feature_base[self.zipcodes[0]]["masking"]:
                        mode = None if date_zip.dz_feature_base[self.zipcodes[0]]["aggregation"]=="--None--" else date_zip.dz_feature_base[self.zipcodes[0]]["aggregation"]
                error.children = []
                sensitive_features = [i for i in self.numerical_feature_base if len(self.numerical_feature_base[i]["bins"])>0]
                sensitive_features += [i for i in self.discrete_feature_base if len(self.discrete_feature_base[i]["bins"])>0]
                sensitive_features += [i for i in date_zip.dz_feature_base if date_zip.dz_feature_base[i]["aggregation"] is not None]
                sensitive_features = None if len(sensitive_features)==0 else sensitive_features
                title = """<hr style="background-color:#00000021"><h2 id="-1.4-EDA"><span style="color:#ffc729"> 1.5 EDA on the Datasets </span>
                                <a class="anchor-link" href="#-1.4-EDA">¶</a>
                                </h2>"""
                display(HTML(title))
                pseudoDataProfiling(
                    self.df,
                    self.newData,
                    self.numerical,
                    self.discrete,
                    self.name,
                    self.address,
                    self.zipcodes,
                    mode,
                    fileNamePrefix = path,
                    new_feature_names = new_feature_names,
                    sensitive_features=sensitive_features
                )
        style = {"description_width": "300px"}
        layout = {"width": "83%"}
        filePath = widgets.Text(
            placeholder='Enter file path for your pseudonymized data (like "C:/pseudonomized_data.csv")',
            description="<b>Pseudonymized Data File Path:</b>",
            disabled=False,
            style=style,
            layout=layout,
        )
        perfom_button = widgets.Button(
            description="Perform Pseudonymization",
            style=ButtonStyle(button_color="orange"),
            layout={"width": "30%"},
        )
        perfom_button.on_click(perform_click_event)
        filepath_box = widgets.HBox([filePath],layout=widgets.Layout(
                    display="flex",
                    flex_flow="column",
                    align_items="center",
                    width="100%",
                    justify_content="center"))
        error = widgets.VBox()
        error_display = widgets.HBox([error],layout=widgets.Layout(
                    display="flex",
                    flex_flow="column",
                    align_items="center",
                    width="100%",
                    justify_content="center"))
        display(widgets.VBox([filePath,
            widgets.HBox(
                [perfom_button],
                layout=widgets.Layout(
                    display="flex",
                    flex_flow="column",
                    align_items="center",
                    width="100%",
                    justify_content="center",
                ),
            ), error_display])
        )
        display(EDA_out)
