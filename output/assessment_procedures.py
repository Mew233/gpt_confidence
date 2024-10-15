import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from datetime import datetime, timedelta
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

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
            date_matches = [pd.to_datetime(date.strip(), errors='coerce') for match in date_matches for date in match if
                            date]

            # Adding extracted information to model_data in the correct format
            for diagnosis_date in date_matches:
                if pd.notna(diagnosis_date):
                    model_data[patient_id].append({'Time': diagnosis_date})
    return model_data


def process_ground_truth(ground_truth_df):
    ground_truth_data = {}
    for _, row in ground_truth_df.iterrows():
        patient_id = row['PM ID #']
        if pd.isna(patient_id):
            continue

        # Ensure patient_id is numeric for easier handling
        try:
            patient_id = int(patient_id)
        except ValueError:
            print(f"Skipping patient ID {patient_id} due to format issues.")
            continue

        if patient_id not in ground_truth_data:
            ground_truth_data[patient_id] = []

        # Iterate over all columns to find and extract all 'Date of procedure'
        for column in ground_truth_df.columns:
            if 'Date of procedure' in column:
                procedure_dates = row[column]
                if pd.notna(procedure_dates):
                    procedure_dates_list = [pd.to_datetime(date.strip(), errors='coerce') for date in
                                            str(procedure_dates).split(',') if date.strip()]
                    for date in procedure_dates_list:
                        if pd.notna(date):
                            ground_truth_data[patient_id].append({'Time': date})
    return ground_truth_data


# Define evaluation functions


# Define evaluation functions

# Define evaluation functions

def strict_match(model_tuples, gold_tuples):
    # A strict match if each date in gold_tuples is present in model_tuples
    match_results = []
    for gold_time in gold_tuples:
        if gold_time in model_tuples:
            match_results.append(1)  # Match found
        else:
            match_results.append(0)  # No match found
    return match_results

def relaxed_match(model_tuples, gold_tuples, grace_period_days=10, relax_to='day'):
    match_results = []
    for gold_time in gold_tuples:
        matched = False
        for model_time in model_tuples:
            if relax_to == 'day':
                if abs((model_time - gold_time).days) <= grace_period_days:
                    matched = True
                    break
            elif relax_to == 'month':
                if model_time.year == gold_time.year and model_time.month == gold_time.month:
                    matched = True
                    break
            elif relax_to == 'year':
                if model_time.year == gold_time.year:
                    matched = True
                    break
            else:
                raise ValueError("Invalid relax_to parameter. Must be 'day', 'month', or 'year'.")
        match_results.append(1 if matched else 0)
    return match_results

# Evaluate model performance and categorize unmatched patients

def evaluate_model(model_data, ground_truth_data, evaluation_mode='strict'):
    f1_scores = []
    precision_scores = []
    recall_scores = []
    y_true_all = []
    y_pred_all = []

    unmatched_patients = []
    lack_of_data_patients = []
    inaccurate_patients = []

    for patient_id in ground_truth_data:
        if patient_id not in model_data:
            unmatched_patients.append(patient_id)
            lack_of_data_patients.append(patient_id)
            continue

        model_tuples = [pd.to_datetime(entry['Time'], errors='coerce') for entry in model_data[patient_id] if pd.notna(entry['Time'])]
        gold_tuples = [pd.to_datetime(entry['Time'], errors='coerce') for entry in ground_truth_data[patient_id] if pd.notna(entry['Time'])]

        if len(gold_tuples) == 0:
            continue

        if evaluation_mode == 'strict':
            match_results = strict_match(model_tuples, gold_tuples)
        else:
            match_results = relaxed_match(model_tuples, gold_tuples, relax_to=evaluation_mode)

        y_true = [1] * len(gold_tuples)
        y_pred = match_results

        # Append individual match results to calculate metrics at patient level
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

        # Calculate patient-level metrics
        f1 = f1_score(y_true, y_pred, average='binary')
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

        # Determine if the mismatch is due to lack of data or inaccuracy
        if sum(match_results) < len(gold_tuples):
            unmatched_patients.append(patient_id)
            if len(model_tuples) == 0:
                lack_of_data_patients.append(patient_id)
            else:
                inaccurate_patients.append(patient_id)

    # Print unmatched patients and categorize reasons
    print(f"Unmatched patients in {evaluation_mode} evaluation: {unmatched_patients}")
    print(f"Patients lacking data in {evaluation_mode} evaluation: {lack_of_data_patients}")
    print(f"Patients with inaccuracies in {evaluation_mode} evaluation: {inaccurate_patients}")

    # Calculate macro metrics across all patients
    macro_f1 = np.mean(f1_scores) if f1_scores else 0
    macro_precision = np.mean(precision_scores) if precision_scores else 0
    macro_recall = np.mean(recall_scores) if recall_scores else 0
    accuracy = accuracy_score(y_true_all, y_pred_all) if y_true_all else 0

    return macro_f1, macro_precision, macro_recall, accuracy, inaccurate_patients

# Output dataframe to report F1, Precision, Recall, Accuracy

def df_output(model_data, ground_truth_data):
    metrics = []
    settings = ['strict', 'day', 'month', 'year']
    final_inaccurate_patients = set()
    for setting in settings:
        f1, precision, recall, accuracy, inaccurate_patients = evaluate_model(model_data, ground_truth_data,
                                                                              evaluation_mode=setting)
        metrics.append([f"{setting.capitalize()}", f1, precision, recall, accuracy])
        final_inaccurate_patients.update(inaccurate_patients)

    # Recalculate accuracy excluding patients with lack of data
    print(f"Recalculating accuracy excluding patients lacking data...")
    y_true = []
    y_pred = []
    for patient_id in ground_truth_data:
        if patient_id in final_inaccurate_patients:
            y_true.append(1)
            y_pred.append(0)  # Mark as incorrect
        else:
            y_true.append(1)
            y_pred.append(1)  # Mark as correct

    recalculated_accuracy = accuracy_score(y_true, y_pred)
    print(f"Recalculated Accuracy (excluding patients lacking data): {recalculated_accuracy:.2f}")

    # Create a DataFrame
    df_metrics = pd.DataFrame(metrics, columns=['Evaluation Mode', 'F1 Score', 'Precision', 'Recall', 'Accuracy'])
    df_metrics = df_metrics.style.format(
        {'F1 Score': '{:.2f}', 'Precision': '{:.2f}', 'Recall': '{:.2f}', 'Accuracy': '{:.2f}'}).set_caption(
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
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("Set2", 4)
    metrics = ['F1 Score', 'Precision', 'Recall', 'Accuracy']
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.barplot(x=df_metrics_numeric.index, y=df_metrics_numeric[metric], ax=axes[i], palette=[colors[i]], width=0.4, edgecolor='black')
        axes[i].set_title(f'{metric} for Procedures Date', fontsize=16, fontweight='bold', color='darkblue')
        axes[i].set_ylabel('Score', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Evaluation Mode', fontsize=14, fontweight='bold')
        axes[i].set_ylim(0, 1.1)
        axes[i].tick_params(axis='x', labelrotation=45, labelsize=12)
        axes[i].tick_params(axis='y', labelsize=12)
        axes[i].grid(visible=True, linestyle='--', linewidth=0.5, axis='y')

        # Add labels on top of each bar
        for p in axes[i].patches:
            axes[i].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                            ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

    plt.tight_layout(h_pad=0.5, w_pad=0.5)  # Add more space between subplots
    plt.savefig("../output/procedures_assessment.png", dpi=300)
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