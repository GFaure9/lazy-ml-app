import streamlit as st
import yaml
import os
from pathlib import Path
import subprocess
import pandas as pd
from yaml_ml import modules


def generate_yaml(config: dict, yaml_path: str):
    with open(yaml_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

def add_hyperparameter():
    if 'num_hyperparameters' not in st.session_state:
        st.session_state['num_hyperparameters'] = 0
    st.session_state['num_hyperparameters'] += 1

def reset_hyperparameters():
    if 'num_hyperparameters' in st.session_state:
        del st.session_state['num_hyperparameters']
        for key in list(st.session_state.keys()):
            if key.startswith('hyperparam_'):
                del st.session_state[key]

def convert_yes_no_to_bool(s: str) -> bool:
    assert s in ["yes", "no"]
    return True if s=="yes" else False


def main():
    st.markdown("<h1 style='color: #61CBF4;'>⌛ LazyML</h1>", unsafe_allow_html=True)
    # st.image("logo.png", width=300)
    st.markdown("---")

    file_path = st.text_input("Paste your dataset file path here")
    if os.path.isfile(file_path):
        st.success(f"File detected: {file_path}")

    file_name, file_extension, file_folder = "", "", ""
    columns = []

    if file_path:
        file_name = Path(file_path).stem
        file_extension = Path(file_path).suffix.split(".")[-1]
        file_folder = os.path.dirname(file_path)

        try:
            if file_extension in ["csv", "txt"]:
                df = pd.read_csv(file_path, nrows=0)
                columns = df.columns.tolist()
        except Exception as e:
            st.error(f"Error reading columns: {e}")

    st.markdown("<h4 style='color: #61CBF4;'>⚙️ Loading</h4>", unsafe_allow_html=True)

    st.write(f"Detected folder : {file_folder}")
    st.write(f"Detected file name: {file_name}")
    st.write(f"Detected extension: {file_extension}")

    separator = st.selectbox("Column separator", [",", ";", ":", "<TAB>", "<SPACE>"])
    if separator == "<TAB>":
        separator = "\t"
    if separator == "<SPACE>":
        separator = " "

    loading = {
        "folder": file_folder,
        "name": file_name,
        "format": file_extension,
        "separator": separator,
    }

    st.markdown("<h4 style='color: #61CBF4;'>🎯 Target Variable</h4>", unsafe_allow_html=True)
    target_var = st.text_input("Name", key="target_var")

    preprocessing = {}
    if columns:
        st.subheader("Preprocessing")
        for col_name in columns:
            with st.container(border=True):
                st.subheader(col_name)
                col_type = st.selectbox("Type", ["cont", "cat"], key=f"type_{col_name}")
                cleaning = st.multiselect("Cleaning", ["remove_col", "remove_nans", "remove_outliers"],
                                          key=f"cleaning_{col_name}")
                replace_nans = st.selectbox("NaN Replacements",
                                            modules["preprocessing"][col_type]["replace_nans"], key=f"replace_{col_name}") if "remove_nans" not in cleaning else None
                if replace_nans == "value":
                    replace_nans_value = st.text_input("Value", "0" if col_type == "cont" else "", key="replace_nans_value")
                    if col_type == "cont":
                        replace_nans_value = float(replace_nans_value)
                    replace_nans = {replace_nans: replace_nans_value}
                scaling = st.selectbox("Scaling",
                                       ["min_max", "abs_max", "standard", "robust"],
                                       key=f"scaling_{col_name}", index=None) if col_type == "cont" else None
                encoding = st.selectbox("Encoding", ["None", "binary", "one_hot", "ordinary"],
                                        key=f"encoding_{col_name}", index=None) if col_type == "cat" else None

            preprocessing[col_name] = {
                "type": col_type,
                "cleaning": cleaning,
            }
            if replace_nans:
                preprocessing[col_name]["replace_nans"] = replace_nans
            else:
                if "replace_nans" in preprocessing[col_name].keys():
                    del preprocessing[col_name]["replace_nans"]
            if scaling:
                preprocessing[col_name]["scaling"] = scaling
            if encoding:
                preprocessing[col_name]["encoding"] = encoding

    st.markdown("<h4 style='color: #61CBF4;'>✂️ Dataset</h4>", unsafe_allow_html=True)
    stratified_split = st.selectbox("Stratified Split?", ["yes", "no"], index=1)
    dataset = {
        "split": {
            "stratified": convert_yes_no_to_bool(stratified_split),
            "train": st.slider("Train Size (%)", 0, 100, 80),
            # "val": st.slider("Validation Size (%)", 0, 100, 0),
        }
    }
    # if dataset["split"]["val"] == 0:
    #     del dataset["split"]["val"]

    st.markdown("<h4 style='color: #61CBF4;'>🤖 Model</h4>", unsafe_allow_html=True)
    index = None
    if target_var in preprocessing.keys():
        if preprocessing[target_var]["type"] == "cont":
            index = 1
        if preprocessing[target_var]["type"] == "cat":
            index = 0
    model_type = st.selectbox("Prediction Type", ["classification", "regression"], index=index)
    model_names = []
    if model_type:
        model_names = modules["models"][model_type]
    model_name = st.selectbox("Model", model_names, key="model")

    model = {model_type: {model_name: {}}}
    col_add, col_remove = st.columns(2)
    with col_add:
        st.button("Add Hyperparameter", on_click=add_hyperparameter)
    with col_remove:
        st.button("Reset", on_click=reset_hyperparameters)
    if 'num_hyperparameters' in st.session_state:
        for i in range(st.session_state['num_hyperparameters']):
            col_param_name, col_param_type, col_param_value = st.columns([1, 1, 2])
            with col_param_name:
                param_name = st.text_input("Hyperparameter", key=f"hyperparam_name_{i}")
            with col_param_type:
                param_type = st.selectbox("Type", ['String', 'Float/Int', 'Boolean'], key=f"hyperparam_type_{i}")
            with col_param_value:
                if param_type == 'String':
                    param_value = st.text_input("Value", key=f"hyperparam_value_string_{i}")
                elif param_type == 'Float/Int':
                    param_value = st.number_input("Value", key=f"hyperparam_value_num_{i}")
                elif param_type == 'Boolean':
                    param_value = st.radio("Value", [True, False], key=f"hyperparam_value_bool_{i}")
            model[model_type][model_name][param_name] = param_value
    else:
        model = {model_type: {model_name: {}}}

    st.markdown("<h4 style='color: #61CBF4;'>📏 Evaluation</h4>", unsafe_allow_html=True)
    scores = []
    if model_type:
        scores = modules["scores"][model_type]
    score = st.multiselect("Metrics", scores, key=f"score_{model_type}")

    st.markdown("<h4 style='color: #61CBF4;'>📤 Outputs</h4>", unsafe_allow_html=True)
    output_folder = st.text_input("Output Folder", "outputs/")
    model_name_save = st.text_input("Pipeline Name", "My_Pipeline")
    logs = st.selectbox("Save logs?", ["yes", "no"], index=1)
    logs = convert_yes_no_to_bool(logs)

    config = {
        "logs": logs,
        "loading": loading,
        "target_var": target_var,
        "preprocessing": preprocessing,
        "dataset": dataset,
        "model": model,
        "score": score,
        "output_folder": output_folder,
        "name": model_name_save,
    }

    if st.button("Generate Configuration File and Run Pipeline"):
        yaml_path = f"{model_name_save}.yaml"
        generate_yaml(config, yaml_path)
        st.success(f"YAML file saved at: {yaml_path}")

        command = f"python -m yaml_ml --cfg={yaml_path}"
        subprocess.run(command, shell=True, text=True)

        st.success(f"Model trained and evaluated! See results at: {output_folder}/res/{model_name_save}__info.txt")


if __name__ == "__main__":
    main()
    # todo: add minimal comments to structure a bit the code