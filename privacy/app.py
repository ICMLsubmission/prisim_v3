import os
import traceback
from IPython.display import HTML, display

# Lightweight imports only - always needed
from .design.file_upload import fileUpload


def read_configurations_yaml(file_path):
    import yaml
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


class dataPrivacy:
    def __init__(self):
        os.chdir("../")
        self.loadingPretrainedModel = False
        self.dataEvaluate = None
        self.EDA = False

    def dataReading(self, psuedoMode=False):
        self.datareader = fileUpload(psuedoMode)

    def dataPreProcessing(self):
        from .design.data_processing import dataProcessing
        try:
            self.filename_prefix = self.datareader.filename_prefix
            self.data = self.datareader.dataFrame
            self.preProcesser = dataProcessing(self.data.copy())
            self.preProcesser.display_data_processing()
        except Exception:
            display(HTML('<hr><h4 style="color:red">Something went wrong. Please re-run above cells.</h4><hr>'))

    def dataReporting(self):
        from .design.reporting import dataProfiling
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
        except Exception:
            display(HTML('<hr><h4 style="color:red">Something went wrong. Please re-run above cells.</h4><hr>'))

    def training(self):
        from .design.training_generation_fe import trainingGeneration
        from .src.config import Config
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
        except Exception:
            display(HTML('<hr><h4 style="color:red">Something went wrong. Please re-run above cells.</h4><hr>'))

    def loadModelGenerateData(self):
        from .design.loading_generating import pretrainedModelGeneration
        self.modelObj = pretrainedModelGeneration()
        self.modelObj.loadingModel()
        self.loadingPretrainedModel = True

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
        except Exception:
            display(HTML('<hr><h4 style="color:red">Something went wrong. Please re-run above cells.</h4><hr>'))

    def privacyGenerationReporting(self):
        from .design.privacy_generation_fe import Privacy
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
        except Exception:
            display(HTML('<hr><h4 style="color:red">Something went wrong. Please re-run above cells.</h4><hr>'))
