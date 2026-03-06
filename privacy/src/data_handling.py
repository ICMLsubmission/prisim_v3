import pandas as pd
from sklearn.preprocessing import LabelEncoder


def dataReading(path, psuedoMode, s3_access_key, s3_security_key, s3_security_token):
    if "s3" in path.lower():
        try:
            dataFrame = pd.read_csv(path,
                                storage_options={
                                    "key": s3_access_key,
                                    "secret": s3_security_key,
                                    "token": s3_security_token,}
                                )
        except FileNotFoundError:
            return {"error":"FileNotFoundError"}
        except PermissionError:
            return {"error":"PermissionError"}
        except OSError:
            return {"error":"OSError"}
        except:
            return {"error":"CustomError"}
    else:
        try:
            dataFrame = pd.read_csv(path, low_memory=False)
        except FileNotFoundError:
            return {"error":"FileNotFoundError"}
        except:
            return {"error":"CustomError"}
    if not psuedoMode:
        try:
            dataFrame.fillna(dataFrame.mean(), inplace=True)
        except:
            return {"error":"InternalError"}
    fileName = path.split("\\")[-1] if "\\" in path else path.split("/")[-1]
    return {"dataFrame": dataFrame, "fileName": fileName, "error":None}


def process_data(data, numericalColumns, discreteColumns, nameColumns, addressColumns, dateColumns, zipColumns, psuedoMode):
    if None in numericalColumns:
        numericalColumns.remove(None)
    if None in discreteColumns:
        discreteColumns.remove(None)
    if None in nameColumns:
        nameColumns.remove(None)
    if None in addressColumns:
        addressColumns.remove(None)
    if None in dateColumns:
        dateColumns.remove(None)
    if None in zipColumns:
        zipColumns.remove(None)
    assert data is not None, "Data cannot be processed... Please upload data file"
    # assert (
    #     len(numericalColumns) != 0
    # ), "Please select atleast 1 value as numerical feature"
    if len(discreteColumns) > 0:
        assert (
            len(set(numericalColumns) & set(discreteColumns)) == 0
        ), "Please select discrete values which are not selected in 'numerical' features"
    if len(nameColumns) > 0:
        assert (
            len(set(numericalColumns) & set(nameColumns)) == 0
        ), "Please select name values which are not selected in 'numerical' features"
        assert (
            len(set(discreteColumns) & set(nameColumns)) == 0
        ), "Please select name values which are not selected in 'discrete' features"
    if len(addressColumns) > 0:
        assert (
            len(set(numericalColumns) & set(addressColumns)) == 0
        ), "Please select address values which are not selected in 'numerical' features"
        assert (
            len(set(discreteColumns) & set(addressColumns)) == 0
        ), "Please select address values which are not selected in 'discrete' features"
        assert (
            len(set(nameColumns) & set(addressColumns)) == 0
        ), "Please select address values which are not selected in 'name' features"
        try:
            if not psuedoMode:
                print("in psudeo")
                states = list(map(lambda x: x.split(",")[-3].strip(), data[addressColumns[0]]))
                countries = list(
                    map(lambda x: x.split(",")[-2].strip(), data[addressColumns[0]])
                )
                states_countries = [states[i] + "_" + countries[i] for i in range(len(states))]
                data["states_countries"] = states_countries
                discreteColumns.append("states_countries") 
        except:
            assert (1/1)==0, "Address should be in proper format to process data!"
        
    if len(zipColumns) > 0:
        pass
        # for i in zipColumns:
            # assert (data["ZIP"].astype(str).str.len()==5).sum() == len(data), "Invalid ZIP codes in {} columns (More/Less than 5 digits detected)".format(i)

    ignoreColumns = set(data.columns.values) - (
        set(numericalColumns)
        | set(discreteColumns)
        | set(nameColumns)
        | set(addressColumns)
        | set(dateColumns)
        | set(zipColumns)
    )
    if len(ignoreColumns) > 0:
        data.drop(ignoreColumns, inplace=True, axis=1)
    processedData = data.copy()
    data.drop(nameColumns, inplace=True, axis=1)
    data.drop(addressColumns, inplace=True, axis=1)
    discreteLabelEncode = {}
    encodedData = None
    # for i in dateColumns:
    #     processedData[i] = pd.to_datetime(processedData[i], infer_datetime_format=True)
    processedData[numericalColumns] = processedData[numericalColumns].fillna(method="ffill")
    processedData[numericalColumns] = processedData[numericalColumns].fillna(method="bfill")
    processedData[discreteColumns] = processedData[discreteColumns].fillna("NA")
    if not psuedoMode:
        encodedData = data.copy()
        for i in encodedData.columns:
            encodedData[i] = processedData[i].values
            if encodedData[i].dtype == "O":
                discreteLabelEncode[i] = LabelEncoder()
                encodedData[i] = discreteLabelEncode[i].fit_transform(encodedData[i])
    return processedData, encodedData, discreteLabelEncode
