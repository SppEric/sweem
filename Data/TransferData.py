import pandas as pd
import os

def TransferData(input_path, omics_data_file, output_path):
    if os.path.exists(output_path):
        print(f"the output path {output_path} exists")
    else:
        os.mkdir(output_path)
        print(f"created the output path {output_path}")
        
    clinical_data = pd.read_csv(input_path + "clinical.csv.gz")
    clinical_data = clinical_data.dropna(axis=0)
    patientID = clinical_data.sample_id.tolist()

    for i in range(len(omics_data_file)):
        temp = pd.read_csv(input_path + omics_data_file[i] + ".csv.gz", index_col=0)
        patientID_temp = temp.index.tolist()
        patientID = list(set(patientID) & set(patientID_temp))

    omics_output = pd.DataFrame(index=patientID)
    
    for i in range(len(omics_data_file)):
        temp = pd.read_csv(input_path + omics_data_file[i] + ".csv.gz", index_col=0)
        temp = temp.dropna(axis=1)
        genes_intersect = list(set(temp.columns.tolist()))
        x = "_" + omics_data_file[i] + "."
        genes_intersect = x.join(genes_intersect) + "_" + omics_data_file[i]
        genes_intersect = genes_intersect.split(".")
        temp.columns = genes_intersect
        print(f"there are {len(genes_intersect)} genes in {omics_data_file[i]} omics data")
        omics_output = omics_output.join(temp)
    print(f"there are {omics_output.shape[1]} features in {omics_output.shape[0]} samples")

    clinical_data.index = clinical_data.sample_id.tolist()
    clinical_data = clinical_data.loc[patientID, ["sample_id", "ostime", "status"]]
    clinical_data.columns = ["SAMPLE_ID", "OS_MONTHS", "OS_EVENT"]
    output = omics_output.join(clinical_data)
    output.to_csv(output_path + "output.csv")

    train_output = output.sample(frac=0.8, replace=False, random_state=2495)
    output_temp = output.drop(labels=train_output.index.tolist())
    test_output = output_temp.sample(frac=0.5, replace=False, random_state=2495)
    val_output = output_temp.drop(labels=test_output.index.tolist())
    train_output.to_csv(output_path + "train.csv", index=0)
    test_output.to_csv(output_path + "test.csv", index=0)
    val_output.to_csv(output_path + "validation.csv", index=0)

input_path = "./"
omics_data_file = ["rna", "scna", "mutation", "methy"]
output_path = input_path + "Multiple/"
TransferData(input_path, omics_data_file, output_path)
