import pandas as pd
import os

def TransferData(input_path, omics_data_file, module_file, output_path):
    if os.path.exists(output_path):
        print("the output path ", output_path, "exists!!")
    else:
        os.mkdir(output_path)
        print("create the output path ", output_path, "!!!")
    pathway_module = pd.read_csv(
        input_path + module_file, index_col=0
    )  ########### pathway data input########################
    pathway_select = pathway_module.sample(
        100, replace=False, random_state=2495
    )  # We random select 100 pathways for test the experiment
    # pathway_select=pathway_module # select the pathway you want
    pathway_names = pathway_select.index.tolist()
    pathway_gene = pathway_select.columns[pathway_select.sum(0) > 0].tolist()
    clinical_data = pd.read_csv(input_path + "clinical.csv.gz")
    clinical_data = clinical_data.dropna(axis=0)
    patientID = clinical_data.sample_id.tolist()

    for i in range(len(omics_data_file)):
        temp = pd.read_csv(input_path + omics_data_file[i] + ".csv.gz", index_col=0)
        patientID_temp = temp.index.tolist()
        patientID = list(set(patientID) & set(patientID_temp))
    pathway_output = pd.DataFrame(index=pathway_names)
    print(
        "there are ",
        pathway_output.shape[0],
        " pathways and ",
        len(pathway_gene),
        " genes",
    )
    omics_output = pd.DataFrame(index=patientID)
    for i in range(len(omics_data_file)):
        temp = pd.read_csv(input_path + omics_data_file[i] + ".csv.gz", index_col=0)
        temp = temp.dropna(axis=1)
        genes_intersect = list(set(pathway_gene) & set(temp.columns.tolist()))
        temp = temp.loc[patientID, genes_intersect]
        pathway_temp = pathway_select.loc[:, genes_intersect]
        x = "_" + omics_data_file[i] + "."
        genes_intersect = x.join(genes_intersect) + "_" + omics_data_file[i]
        genes_intersect = genes_intersect.split(".")
        pathway_temp.columns = genes_intersect
        temp.columns = genes_intersect
        print(
            "there are ",
            len(genes_intersect),
            " genes in ",
            omics_data_file[i],
            " omics data",
        )
        pathway_output = pathway_output.join(pathway_temp)
        omics_output = omics_output.join(temp)
    print(
        "there are ",
        omics_output.shape[1],
        " features in ",
        omics_output.shape[0],
        " samples",
    )
    print("#################################################")
    print("#################################################")
    clinical_data.index = clinical_data.sample_id.tolist()
    clinical_data = clinical_data.loc[patientID, ["sample_id", "ostime", "status"]]
    clinical_data.columns = ["SAMPLE_ID", "OS_MONTHS", "OS_EVENT"]
    output = omics_output.join(clinical_data)

    train_output = output.sample(frac=0.8, replace=False, random_state=2495)
    output_temp = output.drop(labels=train_output.index.tolist())
    test_output = output_temp.sample(frac=0.5, replace=False, random_state=2495)
    val_output = output_temp.drop(labels=test_output.index.tolist())
    train_output.to_csv(output_path + "train.csv", index=0)
    test_output.to_csv(output_path + "test.csv", index=0)
    val_output.to_csv(output_path + "validation.csv", index=0)
    pathway_output.to_csv(output_path + "pathway_module_input.csv")
    return train_output, test_output, val_output


def TransferDataWithClinical(input_path, omics_data_file, module_file, output_path):
    if os.path.exists(output_path):
        print("the output path ", output_path, "exists!!")
    else:
        os.mkdir(output_path)
        print("create the output path ", output_path, "!!!")
    pathway_module = pd.read_csv(
        input_path + module_file, index_col=0
    )  ########### pathway data input########################
    pathway_select = pathway_module.sample(
        100, replace=False, random_state=2495
    )  # We random select 100 pathways for test the experiment
    # pathway_select=pathway_module # select the pathway you want
    pathway_names = pathway_select.index.tolist()
    pathway_gene = pathway_select.columns[pathway_select.sum(0) > 0].tolist()
    clinical_data = pd.read_csv(input_path + "clinical.csv.gz")
    patientID = clinical_data.sample_id.tolist()
    for i in range(len(omics_data_file)):
        temp = pd.read_csv(input_path + omics_data_file[i] + ".csv.gz", index_col=0)
        patientID_temp = temp.index.tolist()
        patientID = list(set(patientID) & set(patientID_temp))
    pathway_output = pd.DataFrame(index=pathway_names)
    print(
        "there are ",
        pathway_output.shape[0],
        " pathways and ",
        len(pathway_gene),
        " genes",
    )
    omics_output = pd.DataFrame(index=patientID)
    for i in range(len(omics_data_file)):
        temp = pd.read_csv(input_path + omics_data_file[i] + ".csv.gz", index_col=0)
        genes_intersect = list(set(pathway_gene) & set(temp.columns.tolist()))
        temp = temp.loc[patientID, genes_intersect]
        pathway_temp = pathway_select.loc[:, genes_intersect]
        x = "_" + omics_data_file[i] + "."
        genes_intersect = x.join(genes_intersect) + "_" + omics_data_file[i]
        genes_intersect = genes_intersect.split(".")
        pathway_temp.columns = genes_intersect
        temp.columns = genes_intersect
        print(
            "there are ",
            len(genes_intersect),
            " genes in ",
            omics_data_file[i],
            " omics data",
        )
        pathway_output = pathway_output.join(pathway_temp)
        omics_output = omics_output.join(temp)
    print(
        "there are ",
        omics_output.shape[1],
        " features in ",
        omics_output.shape[0],
        " samples",
    )
    clinical_data.index = clinical_data.sample_id.tolist()
    clinical_data = clinical_data.loc[
        patientID, ["sample_id", "age", "ostime", "status"]
    ]
    clinical_data.columns = ["SAMPLE_ID", "Clinical", "OS_MONTHS", "OS_EVENT"]
    output = omics_output.join(clinical_data)
    print("Add one clinical information in the dataframe !!!")
    print("#################################################")
    print("#################################################")

    train_output = output.sample(frac=0.8, replace=False, random_state=2495)
    output_temp = output.drop(labels=train_output.index.tolist())
    test_output = output_temp.sample(frac=0.5, replace=False, random_state=2495)
    val_output = output_temp.drop(labels=test_output.index.tolist())
    train_output.to_csv(output_path + "train.csv", index=0)
    test_output.to_csv(output_path + "test.csv", index=0)
    val_output.to_csv(output_path + "validation.csv", index=0)
    pathway_output.to_csv(output_path + "pathway_module_input.csv")
    return train_output, test_output, val_output


input_path = "./"
module_file = "pathway_module.csv"

# single omic data ###############################

# single omic data input
# omics_data_file=['rna']

# single omic data output path
# output_path=input_path+'Single/'

# TransferData(input_path,omics_data_file,module_file,output_path)

# single omic data output path w/ clinical
# output_path=input_path+'SingleWithClinical/'

# TransferDataWithClinical(input_path,omics_data_file,module_file,output_path)

# multiple omic data ###############################

# multiple omic data input
omics_data_file = ["rna", "scna", "methy"]

# multiple omic data output path
output_path = input_path + "Multiple/"

TransferData(input_path, omics_data_file, module_file, output_path)

# multiple omic data output path w/ clinical
# output_path=input_path+'MultiWithClinical/'

# TransferDataWithClinical(input_path,omics_data_file,module_file,output_path)
