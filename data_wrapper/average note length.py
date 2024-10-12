# import os
# import re
# from collections import defaultdict
# from PyPDF2 import PdfReader
#
# # Define the categories and their corresponding keyword patterns (case-insensitive)
# categories = {
#     'Encounter Note': [r'encounter',r"visitnotes"],
#     'Radiology Test': [r'mri', r'scan', r'mrbrain', r'pet',r"rt",r"tmz",r"mrz",r"rmz",r"cthead",r"ct head"],
#     'Genomic Test': [r'50 genes', r'50 gene', r'foundation', r'exact-1',r"OncRNA","cfDNATest", r"ex1"],
#     'Surgical Pathology': [r'surgpath', r'craniotomy',r'surg',r"ResectionBrain",r"Biopsy"],
#     'Molecular Oncology': [r'fish', r'mgmt', r'oncomine', r'tert']
# }
#
# # Function to categorize files based on the provided patterns
# def categorize_file(file_name):
#     for category, patterns in categories.items():
#         for pattern in patterns:
#             if re.search(pattern, file_name, re.IGNORECASE):  # Case-insensitive search
#                 return category
#     return None  # File does not match any category
#
# # Function to count the number of pages in a PDF file
# def count_pdf_pages(file_path):
#     try:
#         with open(file_path, 'rb') as f:
#             reader = PdfReader(f)
#             return len(reader.pages)
#     except Exception as e:
#         print(f"Error reading PDF file {file_path}: {e}")
#         return 0
#
# # Function to count the number of words in a PDF file
# def count_pdf_words(file_path):
#     try:
#         with open(file_path, 'rb') as f:
#             reader = PdfReader(f)
#             total_words = 0
#             for page_num in range(len(reader.pages)):
#                 page = reader.pages[page_num]
#                 text = page.extract_text()  # Extract text from the page
#                 if text:  # If text extraction is successful
#                     total_words += len(text.split())  # Count words by splitting the text
#             return total_words
#     except Exception as e:
#         print(f"Error reading PDF file {file_path}: {e}")
#         return 0
#
# # Main function to traverse directories and classify files, counting pages and words
# def classify_patient_files(base_directory):
#     patient_category_pages = defaultdict(lambda: defaultdict(int))  # Pages per category per patient
#     patient_category_words = defaultdict(lambda: defaultdict(int))  # Word count per category per patient
#
#     # Traverse the directory
#     for root, dirs, files in os.walk(base_directory):
#         for file in files:
#             patient_id = os.path.basename(root)  # Assuming patient ID is the folder name
#             file_path = os.path.join(root, file)
#             file_category = categorize_file(file)
#
#             if file_category and file.lower().endswith('.pdf'):
#                 # Count pages
#                 patient_category_pages[patient_id][file_category] += count_pdf_pages(file_path)
#                 # Count words
#                 patient_category_words[patient_id][file_category] += count_pdf_words(file_path)
#
#     return patient_category_pages, patient_category_words
#
# # Replace 'your_directory_path' with the path of the folder containing patient subfolders
# base_directory = '../HIPPA_data/OneDrive_1_7-19-2024'
# patient_category_pages, patient_category_words = classify_patient_files(base_directory)
#
# # Print results
# print("Page Counts per Category for Each Patient:")
# for patient, categories in patient_category_pages.items():
#     print(f"\nPatient {patient}:")
#     for category, pages in categories.items():
#         print(f"  {category}: {pages} pages")
#
# print("\nWord Counts per Category for Each Patient:")
# for patient, categories in patient_category_words.items():
#     print(f"\nPatient {patient}:")
#     for category, words in categories.items():
#         print(f"  {category}: {words} words")

import re
from collections import defaultdict


def parse_data(file_content):
    # To hold the sum of pages and words for each category
    page_data = defaultdict(lambda: {"total_pages": 0, "patient_count": 0})
    word_data = defaultdict(lambda: {"total_words": 0, "patient_count": 0})

    current_patient = None
    inside_word_section = False

    for line in file_content.splitlines():
        # Check if we are switching to the Word Counts section
        if "Word Counts per Category" in line:
            inside_word_section = True
            continue

        # Detect patient block
        patient_match = re.match(r"Patient (\w+):", line)
        if patient_match:
            current_patient = patient_match.group(1)
            continue

        # Parse page counts per category (while not in word section)
        if not inside_word_section:
            page_match = re.match(r"\s+(\w+(?: \w+)*): (\d+) pages", line)
            if page_match:
                category, pages = page_match.groups()
                pages = int(pages)

                # Update page data for this category
                if pages > 0:
                    page_data[category]["total_pages"] += pages
                    page_data[category]["patient_count"] += 1

        # Parse word counts per category (inside word section)
        if inside_word_section:
            word_match = re.match(r"\s+(\w+(?: \w+)*): (\d+) words", line)
            if word_match:
                category, words = word_match.groups()
                words = int(words)

                # Update word data for this category
                if words > 0:
                    word_data[category]["total_words"] += words
                    word_data[category]["patient_count"] += 1

    return page_data, word_data


def calculate_averages(data):
    averages = {}
    for category, stats in data.items():
        total = stats["total_pages"] if "total_pages" in stats else stats["total_words"]
        count = stats["patient_count"]
        averages[category] = total / count if count > 0 else 0
    return averages


def main():
    # Read the file content
    file_path = '../output/retrived text.txt.txt'
    with open(file_path, 'r') as file:
        content = file.read()

    # Parse the file data for both pages and word counts
    page_data, word_data = parse_data(content)

    # Calculate averages for both
    avg_pages_per_category = calculate_averages(page_data)
    avg_words_per_category = calculate_averages(word_data)

    # Display results
    print("Average Pages Per Category:")
    for category, avg in avg_pages_per_category.items():
        print(f"{category}: {avg:.2f} pages per patient")

    print("\nAverage Words Per Category (ignoring zero word counts):")
    for category, avg in avg_words_per_category.items():
        print(f"{category}: {avg:.2f} words per patient")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data (replace these values with your actual data)
categories = ['Encounter Note', 'Radiology Test', 'Genomic Test', 'Surgical Pathology', 'Molecular Oncology']

# Average number of pages per patient per category
average_pages = [9.58, 8.53, 21.08, 6.43, 7.99]

# Average word counts per patient per category
average_words = [2260.96, 817.82, 2335.47, 912.87, 235.80]

# Create a DataFrame
df = pd.DataFrame({
    'Category': categories,
    'Average Pages': average_pages,
    'Average Words': average_words
})

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 8))

# Set position for each bar
bar_height = 0.2
index = np.arange(len(categories))

# Plotting the average pages (horizontal bar plot)
bar1 = ax1.barh(index - bar_height/2, df['Average Pages'], bar_height, label='Average Pages', color='orange')

# Create a secondary axis for word counts
ax2 = ax1.twiny()
bar2 = ax2.barh(index + bar_height/2, df['Average Words'], bar_height, label='Average Words', color='indigo', alpha=0.7)

# Adding values on top of bars for both pages and words
for i, v in enumerate(df['Average Pages']):
    ax1.text(v + 0.5, i - bar_height/2, f'{round(v, 2)}', va='center', fontsize=18, color='black')

for i, v in enumerate(df['Average Words']):
    ax2.text(v  +5 , i + bar_height/2, f'{round(v, 2)}', va='center', fontsize=18, color='black')

# Labels and title
ax1.set_ylabel('', fontsize=18)
ax1.set_xlabel('Average Pages', fontsize=18, color='orange',loc='center')
ax2.set_xlabel('Average Words', fontsize=18, color='indigo',loc='center')
plt.title('Average Pages and Word Counts per Patient', fontsize=20)

# Set y-axis category labels
ax1.set_yticks(index)
ax1.set_yticklabels(df['Category'], fontsize=18)

# Set x-axis limits (adjust according to your data)
ax1.set_xlim(0, 25)  # For pages (you can change 25 to any limit you want)
ax2.set_xlim(0, 2500)  # For words (change 2500 to any limit you want)

# Add a legend
# ax1.legend(loc='lower right')
# ax2.legend(loc='lower right')

# Tight layout for better spacing
plt.tight_layout()

# Save plot
plt.savefig("../output/average-pages-and-words-per-patient-horizontal.png", dpi=300)

# Show plot
plt.show()


