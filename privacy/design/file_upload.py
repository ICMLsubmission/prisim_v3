import os
from ..src.data_handling import dataReading
from .html_codes import datasetuploading_html_codes, psuedo_datasetuploading_html_codes
from .data_processing import dataProcessing
from ipywidgets import widgets, Layout, ButtonStyle
from IPython.display import HTML, display, clear_output

class fileUpload:
    def __init__(self, psuedoMode):
        self.dataFrame = None
        self.filename_prefix = None
        self.preProcesser = None
        # display(HTML('<h3>Upload Dataset</h3><hr>'))
        filePath = widgets.Text(
            placeholder="Enter File path",
            description="<b>File Path:</b>",
            disabled=False,
            layout={"width": "60%"}
        )
        s3 = widgets.Text(
            placeholder="Enter S3 File path",
            description="<b>S3 File Path:</b>",
            disabled=False,
            layout={"width": "60%"},
        )
        s3_access_key = widgets.Text(
            placeholder="Enter S3 Access Key",
            description="<b>S3 Access Key:</b>",
            disabled=False,
            layout={"width": "95%"},
            style = {"description_width":"120px"}
        )
        s3_security_key = widgets.Text(
            placeholder="Enter S3 Security Key",
            description="<b>S3 Security Key:</b>",
            disabled=False,
            layout={"width": "95%"},
            style = {"description_width":"120px"}
        )
        s3_security_token = widgets.Text(
            placeholder="Enter S3 Security Token",
            description="<b>S3 Security Token:</b>",
            disabled=False,
            layout={"width": "95%"},
            style = {"description_width":"120px"}
        )
        s3_advance_settings = widgets.VBox([s3_access_key, s3_security_key, s3_security_token])
        accordion_settings = widgets.Accordion(
            children=[s3_advance_settings],
            selected_index = None,
            layout=widgets.Layout(
                display="table", flex_flow="column", align_items="center", width="550px"
            ),
        )
        accordion_settings.set_title(0, "S3 Credentials")
        button = widgets.Button(
            description="Import",
            style=ButtonStyle(button_color="orange"),
            tooltip="Import",
        )
        box_layout = widgets.Layout(
            display="flex", flex_flow="column", align_items="stretch", width="100%"
        )
        out = widgets.Output()

        def on_button_clicked(_):
            with out:
                clear_output()
                display(
                    HTML(
                        f"<h4 style='text-align:center; width:60%; color:blue'>File importing...</h4>"
                    )
                )
                path = filePath.value.strip()
                if path == "":
                    path = s3.value.strip()
                if path is None or "csv" != path.split(".")[-1]:
                    clear_output()
                    display(
                        HTML(
                            '<h4 style="color:red;text-align:center; width:60%">Please enter valid file path</h4>'
                        )
                    )
                else:
                    resp = dataReading(path=path, psuedoMode=psuedoMode, 
                                        s3_access_key=s3_access_key.value, 
                                        s3_security_key=s3_security_key.value, 
                                        s3_security_token=s3_security_token.value)
                    if resp["error"] == "FileNotFoundError":
                        clear_output()
                        display(
                            HTML(
                                '<h4 style="color:red;text-align:center; width:60%">File not found on provided location1</h4>'
                            )
                        )
                        return None
                    elif resp["error"] == "PermissionError":
                        clear_output()
                        display(
                            HTML(
                                '<h4 style="color:red;text-align:center; width:60%">Provided credentials are invalid!</h4>'
                            )
                        )
                        return None
                    elif resp["error"] == "OSError" and (s3_access_key.value=="" or s3_security_key.value=="" or s3_security_token.value==""):
                        clear_output()
                        display(
                            HTML(
                                '<h4 style="color:red;text-align:center; width:60%">Please provide credentials to fetch the data!</h4>'
                            )
                        )
                        return None
                    elif resp["error"] == "InternalError":
                        clear_output()
                        display(
                            HTML(
                                '<h4 style="color:red;text-align:center; width:60%">Something broken... Please check with different dataset or Contact to ZS Admin</h4>'
                            )
                        )
                        return None
                    elif resp["error"] == "CustomError":
                        clear_output()
                        display(
                            HTML(
                                '<h4 style="color:red;text-align:center; width:60%">Please check the file path once... or Provide full file path or check the file contains data</h4>'
                            )
                        )
                        return None
                    
                    clear_output()
                    dataset_description = f"""<center style="margin-right:380px; margin-top:-18px"><table><thead>
                                            <tr>
                                                <th colspan="2" style="text-align:center"><h3>Imported Dataset details are as follows:</h3></th>
                                            </tr>
                                        </thead><tbody>
                                                <tr>
                                                    <td><p style="text-align:left;vertical-align:middle"><b>Dataset Name:<b></p></td>
                                                    <td><p style="text-align:left;vertical-align:middle">{resp["fileName"]}</p></td>
                                                </tr>
                                                <tr>
                                                    <td><p style="text-align:left;vertical-align:middle"><b>No. of Features in Dataset:<b></p></td>
                                                    <td><p style="text-align:left;vertical-align:middle">{resp["dataFrame"].shape[1]}</p></td>
                                                </tr>
                                                <tr>
                                                    <td><p style="text-align:left;vertical-align:middle"><b>No. of Samples in Dataset:<b></p></td>
                                                    <td><p style="text-align:left;vertical-align:middle">{resp["dataFrame"].shape[0]}</p></td>
                                                </tr></tbody></table></center>"""
                    self.filename_prefix = resp["fileName"].split(".")[0]
                    display(
                        HTML(
                            f'<h4 style="color:green;text-align:center; width:60%">Imported Successfully!<h4>'
                        )
                    )
                    if os.path.exists(os.path.join("privacy/configurations/dataset_config", self.filename_prefix+".json")):
                        display(
                            HTML(
                                f'<h4 style="color:blue;text-align:center; width:60%">Feature type configuration found for the given file!<br>Please click Auto-Select button in next section to load the saved configurations.<h4>'
                            )
                        )
                    display(HTML(dataset_description))
                    self.dataFrame = resp["dataFrame"]
                    preprocess_output = widgets.Output()
                    with preprocess_output:
                        clear_output()
                        self.preProcesser = dataProcessing(
                            self.dataFrame.copy(),
                            self.filename_prefix,
                            psuedoMode,
                            s3_bucket = os.path.dirname(path),
                            s3_access_key = s3_access_key.value, 
                            s3_security_key = s3_security_key.value, 
                            s3_security_token = s3_security_token.value
                        )
                        self.preProcesser.display_data_processing()
                    display(preprocess_output)

        button.on_click(on_button_clicked)
        title = """<h1 id="-1.1-Dataset-Importing-"><span style="color:orange"> 1.1 Dataset Importing </span>
                    <a class="anchor-link" href="#-1.1-Dataset-Importing-">¶</a>
                    </h1>"""
        display(HTML(title))
        if psuedoMode:
            display(HTML(psuedo_datasetuploading_html_codes))
        else:
            display(HTML(datasetuploading_html_codes))
        H = widgets.HBox([filePath])
        H1 = widgets.HBox([s3])
        V = widgets.VBox(
            [accordion_settings, button],
            layout={"width": "60%", "align_items": "center", "flex_flow": "columns"},
        )
        v2 = widgets.Box([out], layout=box_layout)

        display(H)
        display(HTML('<h2 style="width:60%;text-align:center">OR</h2>'))
        display(H1)
        display(V)
        display(v2)
