from __future__ import print_function, division
import warnings

from torch import unsafe_chunk

warnings.filterwarnings("ignore")
import pickle
from ipywidgets import widgets, ButtonStyle
from IPython.display import HTML, display, clear_output
from .training_generation_fe import trainingGeneration
from .privacy_generation_fe import Privacy
from .html_codes import modelloading_html_codes


class pretrainedModelGeneration:
    def __init__(self):
        # self.data = data
        self.discreteColumns = None
        self.nameColumns = None
        self.addressColumns = None
        self.syntheticData = None
        self.encodedDf = None
        self.samples_value = None
        self.model = None
        self.originalData = None
        self.stateCountryData = None
        self.discreteLabelEncoders = None
        self.filename_prefix = None
        self.dataEvaluate = None
        self.pretrainedModelObj = None

    def loadingModel(self):
        title = """<h2 id="-2.1-Model-Uploading-"><span style="color:orange"> 2.1 Model Uploading </span>
                    <a class="anchor-link" href="#-2.1-Model-Uploading-">¶</a>
                    </h2>"""
        display(HTML(title))
        display(HTML(modelloading_html_codes))
        style = {"description_width": "20%"}
        layout = {"width": "70%"}
        modelPath = widgets.Text(
            placeholder="Enter model path (*.pkl)",
            description="<b>Model Path:</b>",
            disabled=False,
            style=style,
            layout={"width": "70%"},
        )
        button = widgets.Button(
            description="Import Model",
            style=ButtonStyle(button_color="orange"),
            layout={"width": "20%"},
            tooltip="Upload Model",
        )
        out = widgets.Output()
        box_layout = widgets.Layout(
            display="flex", flex_flow="column", align_items="stretch", width="100%"
        )

        def on_button_clicked(_):
            with out:
                clear_output()
                path = modelPath.value
                if path is None or "pkl" != path.split(".")[-1]:
                    display(
                        HTML(
                            '<h4 style="color:red; text-align:center">Please enter valid model file path</h4>'
                        )
                    )
                else:
                    try:
                        with open(path, "rb") as file:
                            unserialized_model_data = pickle.load(file)
                    except:
                        display(
                            HTML(
                                '<h4 style="color:red; text-align:center">Please check the file path once...</h4>'
                            )
                        )
                        return None
                    if "modelObj" not in unserialized_model_data:
                        display(
                            HTML(
                                '<h4 style="color:red; text-align:center">Invalid model uploaded</h4>'
                            )
                        )
                    else:
                        model_description = unserialized_model_data["model_description"]
                        self.filename_prefix = unserialized_model_data[
                            "filename_prefix"
                        ]
                        self.encodedDf = unserialized_model_data["modelObj"].data
                        self.discreteColumns = unserialized_model_data[
                            "modelObj"
                        ].discrete_columns
                        self.nameColumns = unserialized_model_data[
                            "modelObj"
                        ].name_columns
                        self.addressColumns = unserialized_model_data[
                            "modelObj"
                        ].address_columns
                        self.discreteLabelEncoders = unserialized_model_data[
                            "discreteLabelEncoders"
                        ]
                        self.originalData = unserialized_model_data["originalData"]
                        self.dataEvaluate = unserialized_model_data["dataEvaluateObj"]
                        # self.pretrainedModelObj = trainingGeneration(encodedDf = self.encodedDf, discrete_columns = self.discreteColumns, name_columns = self.nameColumns, address_columns = self.addressColumns, model = unserialized_model_data["modelObj"])
                        privacy_output = widgets.Output()
                        with privacy_output:
                            display(
                                HTML(
                                    '<center><h4 style="color:green" style="margin:auto 20vw">Model Imported successfully!</h4></center>'
                                )
                            )
                            display(HTML(f"<center>{model_description}</center>"))
                            privacyModule = Privacy(
                                self.originalData,
                                self.encodedDf,
                                self.discreteColumns,
                                self.nameColumns,
                                self.addressColumns,
                                self.discreteLabelEncoders,
                                self.filename_prefix,
                                unserialized_model_data["modelObj"],
                            )
                            privacyModule.validationGenerationExploration()
                        display(privacy_output)

        button.on_click(on_button_clicked)
        h = widgets.HBox([modelPath, button])
        v = widgets.HBox([out], layout=box_layout)
        display(h)
        display(v)

    def samplesSelection(self):
        self.pretrainedModelCtganObj.samplesSelection()
