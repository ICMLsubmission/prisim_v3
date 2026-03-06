import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets, ButtonStyle
from IPython.display import HTML, display, clear_output
from math import ceil
from ..src.data_evaluation import get_metrics

class evaluate:
    def __init__(self, data, private_data=None, discrete_columns=None, target_column = None, privacy_score=0, ml_privacy_score=0):
        self.data = data
        self.privateData = private_data
        self.discreteColumns = discrete_columns
        self.targetVariable = target_column
        self.privacy_score = privacy_score
        self.ml_privacy_score = ml_privacy_score
    def eval(self):
        # if self.targetVariable is None:
        #     self.variable_selection()
        # else:
        response = get_metrics(self.data, target_variable = self.targetVariable, discrete_variables = self.discreteColumns, private_data = self.privateData)
        analytical_utility, ml_utility, data_title = response["analytical_utility"], response["ml_utility"],  response["data_title"]
        self.interactivePrint(analytical_utility, ml_utility, self.privacy_score, self.ml_privacy_score, data_title)
        self.interactivePlot(response["importantFeatures"])
    def variable_selection(self):
        style = {'description_width': '40%'}
        layout = {'width': '60%'}
        optns = list(self.data.columns)
        if "states_countries" in optns:
            optns.remove("states_countries")
        options = optns # options[0]
        value = options[0] if self.targetVariable is None else self.targetVariable
        y_var = widgets.Dropdown(options=options, value=value, rows=len(options), 
                                description='<b>Select Target variable for ML model:</b>', disabled=False, style = style, layout = layout)
        next_button = widgets.Button(description=f'Train ML Model', style=ButtonStyle(button_color='orange'), layout=layout, tooltip='Train ML Model')
        box_layout = widgets.Layout(display='flex',
                flex_flow='column',
                align_items='center',
                width='60%')
        next_out = widgets.Output()
        def on_next_button_clicked(_):
            with next_out:
                clear_output()
                y = y_var.value
                self.targetVariable = y
                X = list(self.data.columns)
                X.remove(y)
                response = get_metrics(self.data, target_variable = self.targetVariable, discrete_variables = self.discreteColumns, private_data = self.privateData)
                analytical_utility, ml_utility, data_title = response["analytical_utility"], response["ml_utility"],response["data_title"]
                self.interactivePrint(analytical_utility, ml_utility, self.privacy_score, self.ml_privacy_score, data_title)
                # self.interactivePlot(response["importantFeatures"])
        next_button.on_click(on_next_button_clicked)
        n_b =  widgets.VBox([next_button, next_out], layout=box_layout)
        display(y_var)
        display(n_b)
    def interactivePrint(self, analytical_utility, ml_utility, privacy_score, ml_privacy_score, data_title):
        if data_title["VM"]:
            html_string = f'''<h3>Evaluation Metrics:</h3><hr>
            <table>
                <thead>
                <tr>
                    <th ></th>
                    <th  colspan="2">ML Utility(ML-U)</th>
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
                    <td >{round(ml_utility["original"],2)}</td>
                    <td >100</td>
                </tr>
                <tr>
                <td ><b>Private Dataset</b></td>
                <td >{round(ml_privacy_score/100,2)}</td>
                <td >{ceil((ml_privacy_score/ml_utility["original"])*1)}</td>
            </tr>
                </tbody>
            </table>
            <br><br>
            <b>Privacy measurement</b> for the private data: {privacy_score}.
            <br>
            <hr>'''
            # <b>Note:</b>
            # <ol>
            #     <li>GQI measures the similarity of the distributions of bi-variate correlations between original and private data. The higher, the better.</li>
            #     <li>ML utility is calculated with the adjusted R-squared value (for regression task) & F-score value (for classification task)</li>
            #     <li>Privacy is represented with proportion of non-vulnerable population in the data</li>
            # </ol>   
            # '''
        else:
            html_string = f'''	
            <br>
            <h3>&rArr;   ML Utility on original dataset = {ml_utility["original"]} </h3><hr>
        '''
        display(HTML(html_string))
    def interactivePlot(self, importantFeatures):
        display(HTML("<h3>Feature Importantance for the ML task:</h3>"))
        cols = len(importantFeatures)
        plt.figure(figsize = (12,6))
        plt.subplot(1,cols, 1)
        plt.barh(importantFeatures["original"].index.values[:30], importantFeatures["original"].values[:30], color='blue')
        plt.title("Feature Importance on real dataset")
        plt.xticks(np.arange(0, 1, 0.2))
        plt.xlim(0,1)
        plt.grid(True)
        if cols==2:
            plt.subplot(1,cols, cols)
            plt.barh(importantFeatures["VM"].index.values[:30], importantFeatures["VM"].values[:30], color='orange')
            plt.title("Feature Importance on private dataset")
            plt.xticks(np.arange(0, 1, 0.2))
            plt.xlim(0,1)
            plt.grid(True) 
        plt.tight_layout()
        plt.show()
        display(HTML(f'<br><br><hr><h2 style="color:orange"> Congratulations! You have successfully privatized your data ! </h2>'))
