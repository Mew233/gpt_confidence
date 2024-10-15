import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from datetime import datetime, timedelta
import numpy as np
import re
import matplotlib.pyplot as plt
from torchdrug.metrics import accuracy

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

        # Ensure patient_id is consistent and can be handled flexibly
        patient_id = str(patient_id).strip()
        if patient_id.startswith('PM'):
            patient_id = patient_id[2:]  # Remove 'PM' prefix if it exists
        try:
            patient_id = int(patient_id)
        except ValueError:
            print(f"Skipping patient ID {patient_id} due to format issues.")
            continue

        if patient_id not in model_data:
            model_data[patient_id] = []

        model_output = row['model_output']
        if isinstance(model_output, str):
            # Extracting all dates from the model output
            date_matches = re.findall(r"Datetime=\{\{(.*?)\}\}|Datetime=\{(.*?)\}", model_output)
            # Flattening the list of matches to get all date values
            date_matches = [date.strip() for match in date_matches for date in match if date]

            # Adding extracted information to model_data
            for diagnosis_date in date_matches:
                model_data[patient_id].append(diagnosis_date)
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

def strict_match(model_tuples, gold_tuple):
    # A strict match if any of the dates in model_tuples matches the gold tuple date
    gold_time = gold_tuple['Time']
    for model_tuple in model_tuples:
        if model_tuple['Time'] == gold_time:
            return True
    return False


def relaxed_match(model_tuples, gold_tuple, grace_period_days=10, relax_to='day'):
    gold_time = gold_tuple['Time']

    for model_tuple in model_tuples:
        # Relaxed time match
        if relax_to == 'day':
            time_match = abs((model_tuple['Time'] - gold_time).days) <= grace_period_days
        elif relax_to == 'month':
            time_match = model_tuple['Time'].year == gold_time.year and model_tuple['Time'].month == gold_time.month
        elif relax_to == 'year':
            time_match = model_tuple['Time'].year == gold_time.year
        else:
            raise ValueError("Invalid relax_to parameter. Must be 'day', 'month', or 'year'.")

        if time_match:
            return True
    return False


# Evaluate model performance

def evaluate_model(model_data, ground_truth_data, evaluation_mode='strict'):
    y_true = []
    y_pred = []
    unmatched_patients = []

    for patient_id in ground_truth_data:
        if patient_id not in model_data:
            y_true.append(0)
            y_pred.append(0)
            unmatched_patients.append(patient_id)
            continue

        model_tuples = [{'Time': pd.to_datetime(entry, errors='coerce')} for entry in model_data[patient_id]]

        gold_tuple = {'Time': pd.to_datetime(ground_truth_data[patient_id][0][1], errors='coerce'),
                      'Method': set(ground_truth_data[patient_id][0][0].split(', '))}

        if evaluation_mode == 'strict':
            match = strict_match(model_tuples, gold_tuple)
        else:
            match = relaxed_match(model_tuples, gold_tuple, relax_to=evaluation_mode)

        if not match:
            unmatched_patients.append(patient_id)

        y_true.append(1)
        y_pred.append(1 if match else 0)

    # Print unmatched patients
    print(f"Unmatched patients in {evaluation_mode} evaluation: {unmatched_patients}")

    # Calculate metrics
    # f1 = f1_score(y_true, y_pred, average='micro')
    # precision = precision_score(y_true, y_pred, average='micro')
    # recall = recall_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


# Output dataframe to report F1, Precision, Recall
def df_output(model_data, ground_truth_data):
    metrics = []
    settings = ['strict', 'day', 'month', 'year']
    for setting in settings:
        accuracy = evaluate_model(model_data, ground_truth_data, evaluation_mode=setting)
        metrics.append([f"{setting.capitalize()}", accuracy])

    # Create a DataFrame
    df_metrics = pd.DataFrame(metrics, columns=['Evaluation Mode', 'Accuracy'])
    df_metrics = df_metrics.style.format(
        {'Accuracy': '{:.2f}'}).set_caption(
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
    plot_metrics(df_metrics)


if __name__ == "__main__":
    main()