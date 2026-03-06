import os
import warnings
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")
import pickle
import time
import numpy as np
import plotly.express as px

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets, ButtonStyle, Layout
from IPython.display import HTML, display, clear_output
from ..src.synthetic_data_gen import trainModel
from ..src.config import Config
from .html_codes import model_description_header


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def timeConversion(minutes):
    seconds = minutes * 60
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    if seconds > 0:
        minutes += 1
        if minutes > 59:
            hour += 1
            minutes = 0
    return hour, minutes


class trainingGeneration(Config):
    def __init__(
        self,
        encodedDf,
        discrete_columns,
        name_columns,
        address_columns,
        originalData=None,
        discreteLabelEncoders=None,
        dataEvaluateObj=None,
        model=None,
        filename_prefix=None,
    ):
        # self.data = data
        super().__init__(jsonpath="privacy/configurations/params.json")
        self.discreteColumns = discrete_columns
        self.nameColumns = name_columns
        self.addressColumns = address_columns
        self.model = (
            trainModel(jsonpath="privacy/configurations/params.json")
            if model is None
            else model
        )
        self.syntheticData = None
        self.encodedDf = encodedDf
        self.originalData = originalData
        self.discreteLabelEncoders = discreteLabelEncoders
        self.dataEvaluate = dataEvaluateObj
        self.filename_prefix = filename_prefix
        self.targetVariable = None
        self.samplesValue = None

    def tvae_model_settings(self):
        outer_box_layout = Layout(
            display="flex",
            flex_flow="column",
            align_items="stretch",
            border="solid",
            margin="10px 0px 10px 0px",
            width="100%",
        )
        style = {"description_width": "100px"}
        box_title = widgets.HTML(
            "<p style='width: 300px'><center style='color:orange'><b>TVAE Model Settings</b></center></p>"
        )

        epoch_range = widgets.HTML(
            "<p style='width: 50px'><center style='color:#5d5dfffa'>Ideal Range: [1000, 2000, 3000, 4000, 5000]</center></p>"
        )
        self.tvae_epoch_selection = widgets.IntText(
            value=self.epochs_tvae,
            description="<b>Epoch:</b>",
            disabled=False,
            layout={"width": "40%"},
            style=style,
        )

        batch_range = widgets.HTML(
            "<p style='width: 50px'><center style='color:#5d5dfffa'>Ideal Range: [32, 64, 128, 256, 512, 1024]</center></p>"
        )
        self.tvae_batch_size_selection = widgets.IntText(
            value=self.batch_size_tvae,
            description="<b>Batch Size:</b>",
            layout={"width": "40%"},
            disabled=False,
            style=style,
        )

        lr_range = widgets.HTML(
            "<p style='width: 50px'><center style='color:#5d5dfffa'>Ideal Range: [0.0001, 0.001, 0.01, 0.1]</center></p>"
        )
        self.tvae_lr_selection = widgets.BoundedFloatText(
            value=0.01,
            min=0.0,
            max=1.0,
            step=0.001,
            layout={"width": "40%"},
            description="<b>Learning Rate:</b>",
            disabled=False,
            style=style,
        )

        epoch_h_box = widgets.HBox([self.tvae_epoch_selection, epoch_range])
        batch_h_box = widgets.HBox([self.tvae_batch_size_selection, batch_range])
        lr_h_box = widgets.HBox([self.tvae_lr_selection, lr_range])
        box = widgets.VBox(
            [epoch_h_box, batch_h_box, lr_h_box], layout=outer_box_layout
        )
        display(
            widgets.VBox(
                [box_title, box],
                layout=Layout(
                    display="flex-end", flex_flow="column", align_items="center"
                ),
            )
        )

    def ctgan_model_settings(self):
        outer_box_layout = Layout(
            display="flex",
            flex_flow="column",
            align_items="stretch",
            border="solid",
            margin="10px 0px 10px 0px",
            width="100%",
        )
        style = {"description_width": "120px"}
        box_title = widgets.HTML(
            "<p style='width: 300px'><center style='color:orange'><b>CT-GAN Model Settings</b></center></p>"
        )

        epoch_range = widgets.HTML(
            "<p style='width: 50px'><center style='color:#5d5dfffa'>Ideal Range:  [1000, 2000, 3000, 4000, 5000]</center></p>"
        )
        self.ctgan_epoch_selection = widgets.IntText(
            value=self.epochs,
            description="<b>Epoch:</b>",
            layout={"width": "40%"},
            disabled=False,
            style=style,
        )

        batch_range = widgets.HTML(
            "<p style='width: 50px'><center style='color:#5d5dfffa'>Ideal Range: [32, 64, 128, 256, 512, 1024]</center></p>"
        )
        self.ctgan_batch_size_selection = widgets.IntText(
            value=self.batch_size,
            description="<b>Batch Size:</b>",
            layout={"width": "40%"},
            disabled=False,
            style=style,
        )

        gen_lr_range = widgets.HTML(
            "<p style='width: 50px'><center style='color:#5d5dfffa'>Ideal Range: [0.0001, 0.001, 0.01, 0.1]</center></p>"
        )
        self.ctgan_generator_lr_selection = widgets.BoundedFloatText(
            value=self.generator_lr,
            min=0.0,
            max=1.0,
            step=0.001,
            layout={"width": "40%"},
            description="<b>Generator LR:</b>",
            disabled=False,
            style=style,
        )

        dis_lr_range = widgets.HTML(
            "<p style='width: 50px'><center style='color:#5d5dfffa'>Ideal Range: [0.0001, 0.001, 0.01, 0.1]</center></p>"
        )
        self.ctgan_discriminator_lr_selection = widgets.BoundedFloatText(
            value=self.discriminator_lr,
            min=0.0,
            max=1.0,
            step=0.001,
            layout={"width": "40%"},
            description="<b>Discriminator LR:</b>",
            disabled=False,
            style=style,
        )

        epoch_h_box = widgets.HBox([self.ctgan_epoch_selection, epoch_range])
        batch_h_box = widgets.HBox([self.ctgan_batch_size_selection, batch_range])
        gen_lr_h_box = widgets.HBox([self.ctgan_generator_lr_selection, gen_lr_range])
        dis_lr_h_box = widgets.HBox(
            [self.ctgan_discriminator_lr_selection, dis_lr_range]
        )
        box = widgets.VBox(
            [epoch_h_box, batch_h_box, gen_lr_h_box, dis_lr_h_box],
            layout=outer_box_layout,
        )
        display(
            widgets.VBox(
                [box_title, box],
                layout=Layout(display="flex", flex_flow="column", align_items="center"),
            )
        )

    def option_handler(self, model_option):
        if (
            model_option == "High Speed, Decent Quality (Gaussian Copula)"
        ):  #'Gaussian Copula':
            print(
                "\n\n\n",
                color.BOLD
                + color.GREEN
                + "This model doesn't require any hyper-parameter tuning."
                + color.END,
            )
            etaValues = timeConversion(
                self.timers["gc"] * self.encodedDf.shape[0] * self.encodedDf.shape[1]
            )
        elif model_option == "Medium Speed, Medium Quality (T-VAE)":  #'TVAE':
            self.tvae_model_settings()
            etaValues = timeConversion(
                self.timers["tvae"] * self.encodedDf.shape[0] * self.encodedDf.shape[1]
            )
        elif model_option == "Low Speed, High Quality (CT-GAN)":  #'CT-GAN':
            self.ctgan_model_settings()
            etaValues = timeConversion(
                self.timers["ctgan"] * self.encodedDf.shape[0] * self.encodedDf.shape[1]
            )
        self.eta = f"{int(etaValues[0])} hrs: {int(etaValues[1])} mins"
        if int(etaValues[0]) == 0:
            self.eta = f"{int(etaValues[1])} mins"
        info = "<a title='ETA for model training is based on the default selection!'><i class='fa fa-info-circle'></i></a>"
        display(
            HTML(
                f"<center style='color:green'> <b>ETA for model training: </b>{self.eta} {info}</center>"
            )
        )

    def advance_setting(self):
        # initiated everything with Tvae settings
        etaValues = timeConversion(
            self.timers["tvae"] * self.encodedDf.shape[0] * self.encodedDf.shape[1]
        )
        # self.eta = f"{int(etaValues[0])}hrs: {int(etaValues[1])}mins: {int(np.ceil(etaValues[2]))}secs"
        box_layout = widgets.Layout(display="flex", flex_flow="row")

        self.model_options = widgets.RadioButtons(
            options=[
                "High Speed, Decent Quality (Gaussian Copula)",
                "Medium Speed, Medium Quality (T-VAE)",
                "Low Speed, High Quality (CT-GAN)",
            ],
            value="Medium Speed, Medium Quality (T-VAE)",
            layout=Layout(margin="5px  10px 10px 30px", width="305px"),
        )
        operation_selection_button = widgets.Button(
            description="Select Model Type:",
            layout=Layout(margin="0 10px 0 0", height="30px", width="150px"),
        )
        operation_selection_button.style.button_color = "#FFFFFF"
        operation_selection_button.style.font_weight = "bold"

        out = widgets.interactive_output(
            self.option_handler, {"model_option": self.model_options}
        )

        box = widgets.HBox(
            [
                widgets.VBox([operation_selection_button, self.model_options]),
                out,
            ],
            layout=box_layout,
        )
        accordion_advance_settings = widgets.Accordion(
            children=[box], selected_index=None
        )
        accordion_advance_settings.set_title(0, "Advance Settings")
        return accordion_advance_settings

    def train(self):
        box_layout = widgets.Layout(display="flex", flex_flow="row")
        style = {"description_width": "100px"}
        layout = {"width": "50%"}

        # epoch_value = widgets.IntText(value=2000, style=style, layout=layout, description='<b>Epoch</b>')
        model_path = widgets.Text(
            placeholder=' Enter your model save path like "C:/example.pkl"',
            description="<b>Model Save Path:</b>",
            style={"description_width": "110px"},
            disabled=False,
            layout=layout,
        )
        note = HTML(
            f"<b>Note:</b> If you don't specify a path, your model will be saved at <b>saved_models/</b>"
        )
        accordion_advance_settings = self.advance_setting()

        button = widgets.Button(
            description="Train & Export Model",
            style=ButtonStyle(button_color="#64a1db"),
            layout=layout,
            tooltip="Train & Save Model",
        )
        out = widgets.Output()

        def on_button_clicked(_):
            with out:
                self.model_file_name = secure_filename(
                    self.model_options.value.split("(")[1][:-1]
                    + "_"
                    + self.filename_prefix
                    + "_"
                    + str(time.strftime("%Y%m%d-%H_%M_%S"))
                    + ".pkl"
                )
                clear_output()
                if not os.path.exists("saved_models"):
                    os.mkdir("saved_models/")
                if ".pkl" in model_path.value:
                    filename = model_path.value
                elif len(model_path.value) > 1:
                    display(
                        HTML(
                            f'<center><h4 style="color:red"> Please provide filename ending with ".pkl"!<h4></center>'
                        )
                    )
                    return None
                else:
                    filename = f"saved_models/{self.model_file_name}"
                display(
                    HTML(
                        f"<center><b style='align:center'>Model training started...</b></center>"
                    )
                )
                # display(
                #     HTML(
                #         f"<center style='color:green'> <b>ETA for model training: </b>{self.eta}</center>"
                #     )
                # )
                extra_args = {}
                model_type = "Gaussian Copula"
                model_description = """<table><thead>
                                            <tr>
                                                <th colspan="2" style="text-align:center"><h3>Imported Model details are as follows:</h3></th>
                                            </tr>
                                        </thead>
                                    """
                feature_len = (
                    self.originalData.shape[1]
                    if "states_countries" not in list(self.originalData.columns)
                    else self.originalData.shape[1] - 1
                )
                model_description += f"""<tbody>
                                                <tr>
                                                    <td><p style="text-align:left;vertical-align:middle"><b>Generative Model Type:<b></p></td>
                                                    <td><p style="text-align:left;vertical-align:middle">{self.model_options.value}</p></td>
                                                </tr>
                                                <tr>
                                                    <td><p style="text-align:left;vertical-align:middle"><b>Model trained on:<b></p></td>
                                                    <td><p style="text-align:left;vertical-align:middle">{self.filename_prefix}.csv</p></td>
                                                </tr>
                                                <tr>
                                                    <td><p style="text-align:left;vertical-align:middle"><b>No. of Features in Dataset:<b></p></td>
                                                    <td><p style="text-align:left;vertical-align:middle">{feature_len}</p></td>
                                                </tr>
                                                <tr>
                                                    <td><p style="text-align:left;vertical-align:middle"><b>No. of Samples in Dataset:<b></p></td>
                                                    <td><p style="text-align:left;vertical-align:middle">{self.originalData.shape[0]}</p></td>
                                                </tr>
                                                """

                if self.model_options.value == "Medium Speed, Medium Quality (T-VAE)":
                    model_type = "TVAE"
                    extra_args["epochs"] = self.tvae_epoch_selection.value
                    extra_args["batch_size"] = self.tvae_batch_size_selection.value
                    extra_args["lr"] = self.tvae_lr_selection.value
                    model_description += f"""<tr>
                                                    <td><p style="text-align:left;vertical-align:middle"><b>Trained for epochs:<b></p></td>
                                                    <td><p style="text-align:left;vertical-align:middle">{self.tvae_epoch_selection.value}</p></td>
                                                </tr>
                                                <tr>
                                                    <td><p style="text-align:left;vertical-align:middle"><b>Batch size:<b></p></td>
                                                    <td><p style="text-align:left;vertical-align:middle">{self.tvae_batch_size_selection.value }</p></td>
                                                </tr>
                                                <tr>
                                                    <td><p style="text-align:left;vertical-align:middle"><b>Learning Rate:<b></p></td>
                                                    <td><p style="text-align:left;vertical-align:middle">{self.tvae_lr_selection.value}</p></td>
                                                </tr>"""
                elif self.model_options.value == "Low Speed, High Quality (CT-GAN)":
                    model_type = "CT-GAN"
                    extra_args["epochs"] = self.ctgan_epoch_selection.value
                    extra_args["batch_size"] = self.ctgan_batch_size_selection.value
                    extra_args["gen_lr"] = self.ctgan_generator_lr_selection.value
                    extra_args["dis_lr"] = self.ctgan_discriminator_lr_selection.value
                    model_description += f"""<tr>
                                                    <td><p style="text-align:left;vertical-align:middle"><b>Trained for epochs:<b></p></td>
                                                    <td><p style="text-align:left;vertical-align:middle">{self.ctgan_epoch_selection.value}</p></td>
                                                </tr>
                                                <tr>
                                                    <td><p style="text-align:left;vertical-align:middle"><b>Batch size:<b></p></td>
                                                    <td><p style="text-align:left;vertical-align:middle">{self.ctgan_batch_size_selection.value}</p></td>
                                                </tr>
                                                <tr>
                                                    <td><p style="text-align:left;vertical-align:middle"><b>Generator Learning Rate:<b></p></td>
                                                    <td><p style="text-align:left;vertical-align:middle">{self.ctgan_generator_lr_selection.value}</p></td>
                                                </tr>
                                                <tr>
                                                    <td><p style="text-align:left;vertical-align:middle"><b>Discriminator Learning Rate:<b></p></td>
                                                    <td><p style="text-align:left;vertical-align:middle">{self.ctgan_discriminator_lr_selection.value}</p></td>
                                                </tr>"""
                self.model.build(
                    discrete_columns=self.discreteColumns,
                    name_columns=self.nameColumns,
                    address_columns=self.addressColumns,
                    model_type=model_type,
                    kwargs=extra_args,
                )
                history = self.model.fit(self.encodedDf)
                model_description += """</tbody></table>"""
                serializing_model = {
                    "model_description": model_description,
                    "modelObj": self.model,
                    "filename_prefix": self.filename_prefix,
                    "originalData": self.originalData,
                    "discreteLabelEncoders": self.discreteLabelEncoders,
                    "dataEvaluateObj": self.dataEvaluate,
                }
                with open(filename, "wb") as file:
                    # A new file will be created
                    pickle.dump(serializing_model, file)
                display(
                    HTML(
                        f'<center><h4 style="color:green"> Model trained successfully!<h4></center>'
                    )
                )
                display(
                    HTML(
                        f"<center>Model saved in your system at <b>{filename}</b></center>"
                    )
                )

        button.on_click(on_button_clicked)
        box_layout = widgets.Layout(
            display="flex", flex_flow="column", align_items="center", width="100%"
        )
        v = widgets.VBox([model_path, accordion_advance_settings])
        v2 = widgets.VBox([button, out], layout=box_layout)
        # v3 = widgets.VBox([out], layout=box_layout, style = {"text_align": "center", "align_items":"center", "width":"100%"})
        display(v)
        display(note)
        display(v2)
        # display(v3)

    def samplesSelection(self):
        style = {"description_width": "230px"}
        layout = {"width": "60%"}
        samples_value = widgets.IntText(
            value=5000,
            style=style,
            layout=layout,
            description="<b>No. of Samples to be generated: </b>",
        )
        button = widgets.Button(
            description="Select No. of Samples",
            style=ButtonStyle(button_color="orange"),
            layout={"width": "40%"},
            tooltip="Select Samples",
        )
        out = widgets.Output()
        box_layout = widgets.Layout(
            display="flex", flex_flow="column", align_items="center", width="100%"
        )

        def on_button_clicked(_):
            with out:
                clear_output()
                if samples_value.value < 500:
                    display(
                        HTML(
                            f'<h4 style="color:red" style="align:center"> Please select atleast {500} samples<h4>'
                        )
                    )
                    return None
                display(
                    HTML(
                        f'<h4 style="color:#0078ff" style="align:center">Please wait for while.....</h4>'
                    )
                )
            self.syntheticData = self.model.generate(samples_value.value)
            self.samplesValue = samples_value.value
            with out:
                clear_output()
                display(
                    HTML(
                        f'<h4 style="color:green" style="align:center"> {self.samplesValue} samples selected!<h4>'
                    )
                )

        button.on_click(on_button_clicked)
        v = widgets.VBox([samples_value, button, out], layout=box_layout)
        display(v)
