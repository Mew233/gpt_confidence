import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Define the categories and their corresponding keyword patterns (case-insensitive)
categories = {
    'Encounter Note': [r'encounter',r"visitnotes"],
    'Radiology Test': [r'mri', r'scan', r'mrbrain', r'pet',r"rt",r"tmz",r"mrz",r"rmz",r"cthead",r"ct head"],
    'Genomic Test': [r'50 genes', r'50 gene', r'foundation', r'exact-1',r"OncRNA","cfDNATest", r"ex1"],
    'Surgical Pathology': [r'surgpath', r'craniotomy',r'surg',r"ResectionBrain",r"Biopsy"],
    'Molecular Oncology': [r'fish', r'mgmt', r'oncomine', r'tert']
}

# Function to categorize files based on the provided patterns
def categorize_file(file_name):
    for category, patterns in categories.items():
        for pattern in patterns:
            if re.search(pattern, file_name, re.IGNORECASE):  # Case-insensitive search
                return category
    return None  # File does not match any category

# Main function to traverse directories and classify files
def classify_patient_files(base_directory):
    patient_file_count = defaultdict(int)  # To store file counts per patient
    category_file_count = defaultdict(lambda: defaultdict(int))  # To store counts per category per patient
    unclassified_files = defaultdict(list)  # To store unclassified files

    # Traverse the directory
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            patient_id = os.path.basename(root)  # Assuming patient ID is the folder name
            patient_file_count[patient_id] += 1  # Count file for the patient
            file_category = categorize_file(file)

            if file_category:
                category_file_count[patient_id][file_category] += 1
            else:
                unclassified_files[patient_id].append(file)

    return patient_file_count, category_file_count, unclassified_files

# Print results in a user-friendly way
def print_results(patient_file_count, category_file_count, unclassified_files):
    print("File Count per Patient:")
    for patient, count in patient_file_count.items():
        print(f"Patient {patient}: {count} files")

    print("\nFile Count per Category for Each Patient:")
    for patient, categories in category_file_count.items():
        print(f"\nPatient {patient}:")
        for category, count in categories.items():
            print(f"  {category}: {count} files")

    print("\nUnclassified Files:")
    for patient, files in unclassified_files.items():
        if files:
            print(f"\nPatient {patient}:")
            for file in files:
                print(f"  {file}")

# Replace 'your_directory_path' with the path of the folder containing patient subfolders
base_directory = '../HIPPA_data/OneDrive_1_7-19-2024'
patient_file_count, category_file_count, unclassified_files = classify_patient_files(base_directory)

# Output the results
# print_results(patient_file_count, category_file_count, unclassified_files)

# Convert the values (file counts) to a list
file_counts = list(patient_file_count.values())

# Calculate the mean of files per patient
mean_file_count = np.mean(file_counts)

# Print the result
print(f"The mean number of files per patient is: {mean_file_count}")

# Initialize dictionary to hold the total count per category
category_totals = defaultdict(int)
category_counts = defaultdict(int)

# Iterate over patients and their file counts
for patient, categories in category_file_count.items():
    for category, count in categories.items():
        category_totals[category] += count
        category_counts[category] += 1

# Calculate the average number of files per category
category_averages = {category: total / round(category_counts[category],3) for category, total in category_totals.items()}

# Print the average file count per category
print(category_averages)

# Create a horizontal bar plot using Seaborn
desired_order = ['Encounter Note', 'Radiology Test', 'Genomic Test', 'Surgical Pathology', 'Molecular Oncology']

# Re-arrange the dictionary according to the desired order
rearranged_category_averages_dict = {category: category_averages[category] for category in desired_order}

df_category_averages = pd.DataFrame(list(rearranged_category_averages_dict.items()), columns=['Category', 'Average Number of Files'])
plt.figure(figsize=(8, 8))
sns.barplot(x='Average Number of Files', y='Category', data=df_category_averages, color='crimson',width=0.2)

for index, value in enumerate(df_category_averages['Average Number of Files']):
    plt.text(value - 1, index, f'{round(value, 1)}', va='center', size=18,color='yellow')

# Add titles and labels
plt.title('Average Files Per Patient',fontsize=20)
plt.xlabel('')
plt.ylabel('')
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
# Display the plot
plt.tight_layout()
plt.savefig("../output/average-number-of-files-per-patient.png", dpi=300)

plt.show()


