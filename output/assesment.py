import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime, timedelta
import numpy as np
import re
import matplotlib.pyplot as plt

# Load data
model_output_file = 'onc_gbm_Highdata_ADV_output_2024-10-15.xlsx'
ground_truth_file = 'GBM Clinical Data RedCap Export.xlsx'


def load_data():
    model_df = pd.read_excel(model_output_file)
    ground_truth_df = pd.read_excel(ground_truth_file)
    return model_df, ground_truth_df


# Create patient-level dictionaries for model output and ground truth
def process_model_output(model_df):
    model_data = {}
    for _, row in model_df.iterrows():
        patient_id = row['doc_idx']
        if pd.isna(patient_id):
            continue

        # Ensure patient_id is consistent and numeric for easier handling
        patient_id = str(patient_id).strip()
        if patient_id.startswith('PM'):
            patient_id = patient_id[2:]  # Remove 'PM' prefix
        patient_id = int(patient_id)

        if patient_id not in model_data:
            model_data[patient_id] = []

        model_output = row['model_output']
        if isinstance(model_output, str):
            # Extracting diagnosis date and reason from the model output
            diagnosis_matches = re.findall(r"DiagnosisDate\(Datetime=\{\{(.*?)\}\}, Reason=\{\{(.*?)\}\}\)",
                                           model_output)

            # Adding extracted information to model_data
            for diagnosis_date, diagnosis_reason in diagnosis_matches:
                diagnosis_date = diagnosis_date.strip("' ")  # Remove extra quotes and whitespace
                diagnosis_reason = diagnosis_reason.strip("' ")  # Remove extra quotes and whitespace
                model_data[patient_id].append([diagnosis_reason, diagnosis_date])
    return model_data


def process_ground_truth(ground_truth_df):
    ground_truth_data = {}
    for _, row in ground_truth_df.iterrows():
        patient_id = row['PM ID #']
        if pd.isna(patient_id):
            continue

        # Ensure patient_id is numeric for easier handling
        patient_id = int(patient_id)

        if patient_id not in ground_truth_data:
            ground_truth_data[patient_id] = []
        ground_truth_data[patient_id].append([row['Method of diagnosis'],
                                              row['Date of initial diagnosis'].strftime('%m/%d/%Y') if pd.notnull(
                                                  row['Date of initial diagnosis']) else 'N/A'])
    return ground_truth_data


# Define evaluation functions

def strict_match(model_tuple, gold_tuple, consider_method=True):
    # A strict match if the time matches and optionally any method in gold_tuple is present in model_tuple
    if model_tuple['Time'] != gold_tuple['Time']:
        return False
    if consider_method:
        model_methods = set(model_tuple['Method'])
        gold_methods = set(gold_tuple['Method'])
        return any(gold_method in model_methods for gold_method in gold_methods)
    return True

def relaxed_match(model_tuple, gold_tuple, grace_period_days=10, relax_to='day', consider_method=True):
    # Relaxed time match
    if relax_to == 'day':
        time_match = abs((model_tuple['Time'] - gold_tuple['Time']).days) <= grace_period_days
    elif relax_to == 'month':
        time_match = model_tuple['Time'].year == gold_tuple['Time'].year and model_tuple['Time'].month == gold_tuple[
            'Time'].month
    elif relax_to == 'year':
        time_match = model_tuple['Time'].year == gold_tuple['Time'].year
    else:
        raise ValueError("Invalid relax_to parameter. Must be 'day', 'month', or 'year'.")

    # Relaxed method match (inclusion criteria)
    if consider_method:
        model_methods = set(model_tuple['Method'])
        gold_methods = set(gold_tuple['Method'])
        method_match = any(
            gold_method in model_method for gold_method in gold_methods for model_method in model_methods)
    else:
        method_match = True

    return time_match and method_match


# Evaluate model performance
def evaluate_model(model_data, ground_truth_data, evaluation_mode='strict', consider_method=True):
    y_true = []
    y_pred = []

    for patient_id in ground_truth_data:
        if patient_id not in model_data:
            y_true.append(0)
            y_pred.append(0)
            continue

        model_tuples = [{'Time': pd.to_datetime(entry[1], errors='coerce'), 'Method': set(entry[0].split(', '))} for entry in model_data[patient_id]]
        gold_tuple = {'Time': pd.to_datetime(ground_truth_data[patient_id][0][1], errors='coerce'), 'Method': set(ground_truth_data[patient_id][0][0].split(', '))}

        match = False
        for model_tuple in model_tuples:
            if evaluation_mode == 'strict':
                match = strict_match(model_tuple, gold_tuple, consider_method=consider_method)
            else:
                match = relaxed_match(model_tuple, gold_tuple, relax_to=evaluation_mode, consider_method=consider_method)

            if match:
                break

        y_true.append(1)
        y_pred.append(1 if match else 0)

    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    return f1, precision, recall

# Output dataframe to report F1, Precision, Recall
def df_output(model_data, ground_truth_data):
    metrics = []
    settings = ['strict', 'day', 'month', 'year']
    for setting in settings:
        for consider_method in [False,True]:
            method_label = "with Method" if consider_method else "without Method"
            f1, precision, recall = evaluate_model(model_data, ground_truth_data, evaluation_mode=setting,
                                                   consider_method=consider_method)
            metrics.append([f"{setting.capitalize()} ({method_label})", f1, precision, recall])

    # Create a DataFrame
    df_metrics = pd.DataFrame(metrics, columns=['Evaluation Mode', 'F1 Score', 'Precision', 'Recall'])
    df_metrics = df_metrics.style.format({'F1 Score': '{:.2f}', 'Precision': '{:.2f}', 'Recall': '{:.2f}'}).set_caption(
        "Evaluation Metrics Report").set_table_styles([
        {
            'selector': 'caption',
            'props': [('color', 'blue'),
                      ('font-size', '16px')]
        },
        {
            'selector': 'th',
            'props': [('background-color', '#f0f0f0'),
                      ('font-weight', 'bold')]
        },
        {
            'selector': 'td',
            'props': [('padding', '10px')]
        }
    ])
    return df_metrics


# Plot function to visualize metrics
def plot_metrics(df_metrics):
    df_metrics_numeric = df_metrics.data.set_index('Evaluation Mode')
    df_metrics_numeric.plot(kind='bar', figsize=(12, 8), rot=45)
    plt.title('Evaluation Metrics for Different Modes')
    plt.ylabel('Score')
    plt.xlabel('Evaluation Mode')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


# Main function to execute the code
def main():
    model_df, ground_truth_df = load_data()
    model_data = process_model_output(model_df)
    ground_truth_data = process_ground_truth(ground_truth_df)

    # Output metrics DataFrame
    df_metrics = df_output(model_data, ground_truth_data)
    print(df_metrics)

    # Plot metrics
    # plot_metrics(df_metrics)


if __name__ == "__main__":
    main()