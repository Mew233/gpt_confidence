# import os
# import re
# from collections import defaultdict
# # from PyPDF2 import PdfReader
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
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
# model = ocr_predictor(pretrained=True)
# # PDF
# doc = DocumentFile.from_pdf("path/to/your/doc.pdf")
# # Analyze
# result = model(doc)
#
# def count_pdfreader = PdfReader(f)
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