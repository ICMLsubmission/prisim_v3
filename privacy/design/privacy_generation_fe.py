import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from click import option
from IPython.display import HTML, clear_output, display
from ipywidgets import ButtonStyle, interact, widgets
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots
from torch import layer_norm
from tqdm.auto import tqdm

from ..design.evaulate import evaluate
from ..design.html_codes import *
from ..design.reporting import dataProfiling
from ..src.helper import hybrid_dp
from ..src.privacy_data_eval import vulnarabilitymitigation

# init_notebook_mode()


class Privacy:
    def __init__(
        self,
        processed_data,
        data,
        discrete_columns,
        name_columns,
        address_columns,
        discrete_label_encode,
        filename_prefix,
        model=None,
    ):
        self.processedData = processed_data
        self.encodedDf = data
        self.syntheticData = None
        self.discreteColumns = None if len(discrete_columns)==0 else discrete_columns
        self.nameColumns = None if len(name_columns)==0 else name_columns
        self.addressColumns = None if len(address_columns)==0 else address_columns
        self.discreteLabelEncoders = discrete_label_encode
        self.fileNamePrefix = filename_prefix
        self.trainedModel = model
        self.privateData = None
        self.privateDataEncoded = None
        self.targetColumn = None
        self.privacyScore = None
        self.mlPrivacyScore = None
        self.auPrivacyScore = None
        self.thresholdIdx = None
        self.private_injected_data = None
        self.selected_data = "Original Data"
        self.mlu = []
        self.vmu = []
        self.alu = []

    def privacyBuild(
        self,
    ):
        if self.encodedDf is None:
            display(
                HTML("<h4>Data is not uploaded yet... Please upload data first</h4>")
            )
            return False
        if self.syntheticData is None:
            display(
                HTML(
                    "<h4>Samples were not selected... Please select samples from above code block</h4>"
                )
            )
            return False
        self.privacyMitigate = vulnarabilitymitigation(
            self.encodedDf,
            self.syntheticData,
            self.discreteColumns,
            self.nameColumns,
            self.addressColumns,
            self.discreteLabelEncoders,
            self.fileNamePrefix,
        )
        return True

    def validationGenerationExploration(self):
        # display(HTML("<b>Privatizing your data...</b>"))
        # pbar = tqdm(total=100)
        # pbar.update(50)
        title = """<br><hr style="background-color:#00000021"><h2 id="-2.2-Synthetic-Data-Generation-"><span style="color:orange"> 2.2 Synthetic Data Generation </span>
                        <a class="anchor-link" href="#-2.2-Synthetic-Data-Generation-">¶</a>
                    </h2>"""
        display(HTML(title))
        display(HTML(privatedatageneration_html_codes))
        style = {"description_width": "350px"}
        layout = {"width": "60%"}
        optns = list(self.encodedDf.columns)
        if "states_countries" in optns:
            optns.remove("states_countries")
        options = optns
        try:
            imp_feature = options[
                0
            ]  # list(get_frufs_feature_importance(self.encodedDf, self.discreteColumns)[1])[0]
        except:
            imp_feature = options[0]
        value = imp_feature if imp_feature in options else options[0]
        # pbar.update(50)
        # pbar.close()
        y_var = widgets.Dropdown(
            options=options,
            value=value,
            rows=len(options),
            description="<b>Select Target variable to explore ML Utility:</b>",
            disabled=False,
            style=style,
            layout=layout,
        )
        # style = {'description_width': '350px'}
        # layout = {'width': '80%'}
        box_layout = widgets.Layout(
            display="flex", flex_flow="column", align_items="stretch", width="100%"
        )
        samples_value = widgets.IntText(
            value=1500,
            style=style,
            layout=layout,
            description="<b>No. of Samples to be generated: </b>",
            tooltip="No. of Samples to be generated:",
        )

        privacy_field = widgets.IntText(
            value=50,
            style=style,
            layout=layout,
            description="<b>Required Privacy (%): </b>",
            title="Required Utility (%)",
        )
        button = widgets.Button(
            description="Generate Synthetic Data",
            style=ButtonStyle(button_color="orange"),
            layout={"width": "30%"},
            tooltip="Generate Synthetic Data",
        )
        out = widgets.Output()
        out2 = widgets.Output()
        box_layout = widgets.Layout(
            display="flex", flex_flow="column", align_items="center", width="100%"
        )

        def on_button_clicked(_):
            with out:
                clear_output()
                if samples_value.value < 500:
                    display(
                        HTML(
                            f'<h4 style="color:red" style="text-align:center"> Please select atleast {500} samples<h4>'
                        )
                    )
                    return None
            with out2:
                clear_output()
                # display(
                #     HTML(
                #         "<b>Generating Private Synthetic data & evaluating Utility...</b>"
                #     )
                # )

                n_samples = samples_value.value
                self.privacyMitigate = vulnarabilitymitigation(
                    self.encodedDf,
                    self.discreteColumns,
                    self.nameColumns,
                    self.addressColumns,
                    self.discreteLabelEncoders,
                    n_samples=n_samples,
                    pretrained_model=self.trainedModel,
                    filename_prefix=self.fileNamePrefix,
                )
                self.syntheticData = self.privacyMitigate.syntheticData
                self.generateAndSave(n_samples, privacy_field.value)

        button.on_click(on_button_clicked)
        v = widgets.VBox(
            [samples_value, button, out],
            layout=box_layout,
        )
        v2 = widgets.VBox([out2])
        display(v)
        display(v2)

    def privacyChartOuput(self, out):
        with out:
            clear_output()
            display(HTML("<b>Privacy Risk Tolerance [0-50]:</b>"))
            interact(
                lambda sliderVal: self.privacy_chart(sliderVal),
                sliderVal=widgets.IntSlider(
                    value=self.selected_risk_tolerance,
                    min=0,
                    max=50,
                    step=1,
                    description=" ",
                    style={"description_width": "70px"},
                    layout={"width": "52%"},
                ),
            )

    def generateAndSave(self, n_samples, required_privacy):
        self.privacyMitigate.privacyEvaluation()
        scenarios = self.privacyMitigate.private_data_at_diff_iter
        self.vmu = self.privacyMitigate.vmu
        self.alu = self.privacyMitigate.alu
        self.selected_risk_tolerance = 100 - required_privacy
        self.privacy_output = widgets.Output()
        display(widgets.VBox([self.privacy_output]))
        # self.privacyChartOuput(self.privacy_output)

        style = {"description_width": "250px"}
        layout = {"width": "80%"}
        box_layout = widgets.Layout(
            display="flex", flex_flow="column", align_items="center", width="100%"
        )
        privacy_field = required_privacy
        filePath = widgets.Text(
            placeholder='Enter your path like "C:/example.csv"',
            description="<b>Enter Synthetic Data Save File Path:</b>",
            disabled=False,
            style=style,
            layout=layout,
        )
        next_button = widgets.Button(
            description=f"Download & Explore Synthetic Dataset",
            style=ButtonStyle(button_color="orange"),
            layout={"width": "40%"},
            tooltip="Download & Explore Synthetic Dataset",
        )
        next_out = widgets.Output()
        html_out = widgets.Output()

        def on_next_button_clicked(_):
            with next_out:
                clear_output()
                threshold = 50 #list(scenarios)[self.thresholdIdx - 1]
                if filePath.value == "" or ".csv" not in filePath.value:
                    display(
                        HTML(
                            '<hr><h4 style="width: 100%;text-align:center;color:red">Please enter a valid filepath to save your synthetic data</h4><hr>'
                        )
                    )
                    return None
                (
                    self.privateData,
                    self.privateDataEncoded,
                    self.privacyScore,
                    self.auPrivacyScore,
                ) = self.privacyMitigate.privacyGeneration(
                    threshold, n_samples, filePath.value
                )
                display(
                    HTML(
                        f'<h4 style="width: 100%;text-align:center;color:green" > Your Synthetic data stored sucessfully at {filePath.value}!<h4>'
                    )  # Your private data for selected privacy of {95%} is stored sucessfully at {}!
                )
            with html_out:
                clear_output()
                self.explore()

        next_button.on_click(on_next_button_clicked)

        vAlign2 = widgets.VBox([filePath, next_button, next_out], layout=box_layout)
        display(vAlign2)
        vAlign3 = widgets.VBox([html_out])
        display(vAlign3)

    def explore(self):
        eda_out = widgets.Output()
        downstream_out = widgets.Output()
        v = widgets.VBox([eda_out])
        display(v)
        with eda_out:
            clear_output()
            self.nameColumns = [] if self.nameColumns is None else self.nameColumns
            self.addressColumns = [] if self.addressColumns is None else self.addressColumns
            title = f'<hr style="background-color=#00000021"><h2 id="-2.3-EDA-on-the-Synthetic-Data-"><span style="color:orange"> 2.3 EDA on the Synthetic Data </span></h2>'
            display(HTML(title))
            display(HTML(edageneration_html_code))
            _ = dataProfiling(
                originalData=self.processedData,
                originalEncodedData=self.encodedDf,
                privateData=self.privateData,
                privateDataEncoded=self.privateDataEncoded,
                comparison=True,
                discreteColumns=self.discreteColumns,
                nameColumns=self.nameColumns,
                addressColumns=self.addressColumns,
                discreteLabelEncoders=self.discreteLabelEncoders,
                auPrivacyScore=self.auPrivacyScore,
            )

    def privacy_chart(self, sliderVal):
        # adding orginal privacy value by default
        vmu = [0] + self.vmu
        # mlu = [100] + self.mlu
        alu = [100] + self.alu
        x = [0] + list(np.arange(1, len(vmu) + 1))
        privacyLabels = (
            ["Original", "Synthetic"]
            + [" " for _ in range(len(x) - 4)]
            + ["Fully Private"]
        )
        utlityLabels = (
            ["Original", "Synthetic"]
            + [" " for _ in range(len(x) - 4)]
            + ["Fully Private"]
        )
        utilityCriteria = "Analytical Utility"
        custom_privacy = min(vmu, key=lambda vmu: abs(vmu - (100 - sliderVal)))
        privacy_idx = vmu.index(custom_privacy)
        self.thresholdIdx = privacy_idx
        self.riskTolerance = sliderVal
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Privacy Chart", f"{utilityCriteria} Chart")
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=vmu, mode="lines", name="Privacy", line=dict(color="#a868e3")
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=alu,
                mode="lines",
                name="Analytial Utility",
                line=dict(color="orange"),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[privacy_idx],
                y=[vmu[privacy_idx]],
                text=f"Privacy: {vmu[privacy_idx]}%",
                textposition="top center",
                name="Privacy for Selected Tolerance (PST)",
                mode="markers+text",
                marker_symbol="square-dot",
                marker_size=13,
                marker_line_color="black",
                marker_color="lightskyblue",
                marker_line_width=2,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[privacy_idx],
                y=[alu[privacy_idx]],
                text=f"AU: {int(alu[privacy_idx])}%",
                textposition="top center",
                name="Analytial Utilty for Selected Tolerance (AUST)",
                mode="markers+text",
                marker_symbol="hexagon-dot",
                marker_size=13,
                marker_line_color="#ad0b05",
                marker_color="#f7b7b5",
                marker_line_width=2,
            ),
            row=1,
            col=2,
        )
        privacyLabels[privacy_idx] = "PST"
        utlityLabels[privacy_idx] = "AUST"
        fig.update_layout(
            xaxis=dict(tickmode="array", tickvals=x, ticktext=privacyLabels),
            yaxis=dict(tickmode="linear", tick0=0, dtick=10),
            xaxis2=dict(tickmode="array", tickvals=x, ticktext=utlityLabels),
            legend=dict(
                orientation="h",
                x=0.5,
                y=1.25,
                xanchor="center",
                yanchor="top",
                borderwidth=1,
            ),
            hovermode=False,
            height=600,
            width=800,
        )
        fig.update_xaxes(tickangle=300)
        fig.show()
