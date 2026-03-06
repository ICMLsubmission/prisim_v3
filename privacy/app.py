import os
import yaml
import traceback
from .src.config import Config
from .design.file_upload import fileUpload
from .design.data_processing import dataProcessing
from .design.reporting import dataProfiling
from .design.evaulate import evaluate
from .design.training_generation_fe import trainingGeneration
from .design.privacy_generation_fe import Privacy
from .design.loading_generating import pretrainedModelGeneration
from IPython.display import HTML, display


def read_configurations_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


class dataPrivacy(Config):
    def __init__(self):
        os.chdir("../")
        # self.configurations = read_configurations_yaml(
        #     "privacy/configurations/privacy_configurations.yaml"
        # )["PRIVACY"]
        # if self.configurations["ENVIRONMENT"] == "test":
        #     super().__init__(jsonpath="privacy/configurations/params.json")
        self.loadingPretrainedModel = False
        self.dataEvaluate = None
        self.EDA = False

    def dataReading(self, psuedoMode=False):
        self.datareader = fileUpload(psuedoMode)

    def dataPreProcessing(self):
        try:
            self.filename_prefix = self.datareader.filename_prefix
            self.data = self.datareader.dataFrame
            self.preProcesser = dataProcessing(self.data.copy())
            self.preProcesser.display_data_processing()
        except:
            if self.configurations["DEBUG"]:
                print(traceback.format_exc())
            display(
                HTML(
                    '<hr><h4 style="color:red">Oho something went wrong, Don\'t Worry!! <br>Please run above blockcells</h4><hr>'
                )
            )

    def dataReporting(self):
        try:
            self.processedData = self.preProcesser.processedData
            self.encodedDf = self.preProcesser.encodedDf
            self.discreteLabelEncoders = self.preProcesser.discreteLabelEncoders
            self.discreteColumns = self.preProcesser.discreteColumns
            self.nameColumns = self.preProcesser.nameColumns
            self.numericalColumns = self.preProcesser.numericalColumns
            self.addressColumns = self.preProcesser.addressColumns
            _ = dataProfiling(
                originalData=self.processedData,
                originalEncodedData=self.encodedDf,
                numericalColumns=self.numericalColumns,
                discreteColumns=self.discreteColumns,
                nameColumns=self.nameColumns,
                addressColumns=self.addressColumns,
                discreteLabelEncoders=self.discreteLabelEncoders,
            )
        except:
            if self.configurations["DEBUG"]:
                print(traceback.format_exc())
            display(
                HTML(
                    '<hr><h4 style="color:red">Oho something went wrong, Don\'t Worry!! <br>Please run above blockcells</h4><hr>'
                )
            )

    def mapObjectValuesToVaribales(self, object):
        if self.loadingPretrainedModel:
            self.filename_prefix = object.filename_prefix
            self.processedData = object.originalData
            self.encodedDf = object.encodedDf
            self.discreteLabelEncoders = object.discreteLabelEncoders
            self.discreteColumns = object.discreteColumns
            self.nameColumns = object.nameColumns
            self.addressColumns = object.addressColumns
            self.dataEvaluate = object.dataEvaluate
        else:
            self.processedData = object.processedData
            self.encodedDf = object.encodedDf
            self.discreteLabelEncoders = object.discreteLabelEncoders
            self.discreteColumns = object.discreteColumns
            self.nameColumns = object.nameColumns
            self.addressColumns = object.addressColumns

    def training(self):
        try:
            if not self.EDA:
                self.mapObjectValuesToVaribales(self.preProcesser)
            self.model = trainingGeneration(
                self.encodedDf,
                self.discreteColumns,
                self.nameColumns,
                self.addressColumns,
                originalData=self.processedData,
                discreteLabelEncoders=self.discreteLabelEncoders,
                dataEvaluateObj=self.dataEvaluate,
                filename_prefix=self.filename_prefix,
            )
            self.model.train()
        except:
            if self.configurations["DEBUG"]:
                print(traceback.format_exc())
            display(
                HTML(
                    '<hr><h4 style="color:red">Oho something went wrong, Don\'t Worry!! <br>Please run above blockcells</h4><hr>'
                )
            )

    def loadModelGenerateData(self):
        self.modelObj = pretrainedModelGeneration()
        self.modelObj.loadingModel()
        self.loadingPretrainedModel = True

    def generateSyntheticData(self):
        try:
            self.filename_prefix = self.modelObj.filename_prefix
            self.processedData = self.modelObj.originalData
            self.encodedDf = self.modelObj.encodedDf
            self.discreteLabelEncoders = self.modelObj.discreteLabelEncoders
            self.discreteColumns = self.modelObj.discreteColumns
            self.nameColumns = self.modelObj.nameColumns
            self.addressColumns = self.modelObj.addressColumns
            self.modelObj.pretrainedModelObj.samplesSelection()
        except:
            if self.configurations["DEBUG"]:
                print(traceback.format_exc())
            display(
                HTML(
                    '<hr><h4 style="color:red">Oho something went wrong, Don\'t Worry!! <br>Please run above blockcells</h4><hr>'
                )
            )

    def privacyGenerationReporting(self):
        try:
            self.n_samples = self.modelObj.pretrainedModelObj.samplesValue
            self.syntheticData = self.modelObj.pretrainedModelObj.syntheticData
            self.privacyModule = Privacy(
                self.processedData,
                self.encodedDf,
                self.syntheticData,
                self.discreteColumns,
                self.nameColumns,
                self.addressColumns,
                self.discreteLabelEncoders,
                self.filename_prefix,
            )
            self.privacyModule.privacyBuild()
            self.privacyModule.validationGenerationExploration(self.n_samples)
        except:
            if self.configurations["DEBUG"]:
                print(traceback.format_exc())
            display(
                HTML(
                    '<hr><h4 style="color:red">Oho something went wrong, Don\'t Worry!! <br>Please run above blockcells</h4><hr>'
                )
            )
