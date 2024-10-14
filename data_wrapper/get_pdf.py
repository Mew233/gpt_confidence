import os
import re
from collections import defaultdict
# from PyPDF2 import PdfReader
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
from torch.cuda import device
import re
import os
from tqdm import tqdm

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

forward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True, resolve_blocks=False).to(forward_device)


def text_extraction(text, category):
    """
    Extract relevant text based on category.

    Parameters:
    - text (str): The full text extracted from the PDF.
    - category (str): The category of the document (e.g., 'Encounter Note').

    Returns:
    - str: The extracted text as a single string.
    """
    if category == 'Encounter Note':
        # Extract "History of Present Illness" section
        history_pattern = re.compile(
            r'(history\s*of\s*present\s*illness[:\s]*)(.*?)(past\s*medical\s*history[.:\s]*|\Z)',
            re.IGNORECASE | re.DOTALL)
        history_match = history_pattern.search(text)
        history_text = history_match.group(2).strip() if history_match else ''

        # Extract "Assessment and Plan" section (with variations like "Assessment", "Assessment&Plan")
        assessment_pattern = re.compile(r'(assessment(?:\s*and\s*plan)?(?:&plan)?.*?)(additional\s*documentation|\Z)', re.IGNORECASE | re.DOTALL)
        assessment_match = assessment_pattern.search(text)
        assessment_text = assessment_match.group(1).strip() if assessment_match else ''

        # Combine the extracted sections
        extracted_text = (history_text + '\n' + assessment_text).strip()
        return extracted_text if history_text or assessment_text else 'Relevant sections not found.'

    elif category == 'Radiology Test':
        # Extract text after "Reading physician" and before "Comparison:" (with variations)
        reading_pattern = re.compile(r'(reading\s*physician.*?)(comparison:|\Z)', re.IGNORECASE | re.DOTALL)
        reading_match = reading_pattern.search(text)
        reading_text = reading_match.group(1).strip() if reading_match else ''

        # Extract text after "Impression:" and before "last resulted:" (with variations)
        impression_pattern = re.compile(r'(impression:.*?)(last\s*resulted:|\Z)', re.IGNORECASE | re.DOTALL)
        impression_match = impression_pattern.search(text)
        impression_text = impression_match.group(1).strip() if impression_match else ''

        # Combine the extracted sections
        extracted_text = (reading_text + '\n' + impression_text).strip()
        return extracted_text if reading_text or impression_text else 'Relevant sections not found.'

    elif category == 'Surgical Pathology':
        # Extract information after "Surgical PathologyReport", "Specimen Date", or "Collection Date"
        # and before "MICROSCOPIC DESCRIPTION:" or "Comment" (with variations)
        pathology_pattern = re.compile(
            r'(surgical\s*pathologyreport|specimen\s*date|collection\s*date).*?(microscopic\s*description:|gross\s*description:|comment:|note:|\Z)',
            re.IGNORECASE | re.DOTALL)
        pathology_match = pathology_pattern.search(text)
        pathology_text = pathology_match.group(0).strip() if pathology_match else ''

        return pathology_text if pathology_text else 'Relevant sections not found.'

    elif category == 'Molecular Oncology':
        # Extract information after "Oncomine Comprehensive", "Molecular Oncology", "Collection Date", or "Final Report"
        # and before "Test information", "Comment", or "Findings" (with variations)
        molecular_pattern = re.compile(r'(oncomine\s*comprehensive|molecular\s*oncology|collection\s*date|final\s*report).*?(test\s*information|comment|findings|\Z)', re.IGNORECASE | re.DOTALL)
        molecular_match = molecular_pattern.search(text)
        molecular_text = molecular_match.group(0).strip() if molecular_match else ''

        return molecular_text if molecular_text else 'Relevant sections not found.'

    # Skip extraction for 'Genomic Test'
    elif category == 'Genomic Test':
        return 'Extraction skipped for Genomic Test category.'
    else:
        return 'Category not recognized.'


def read_pdf_pages(file_path, category):
    """
    Read pages from a PDF file.

    Parameters:
    - file_path (str): Path to the PDF file.
    - category (str): The category of the document to decide reading scope.

    Returns:
    - str: Text extracted from the PDF.
    """
    try:
        with open(file_path, 'rb') as f:
            doc = DocumentFile.from_pdf(f.read())
            result = model(doc)
            text = ''
            # If it is not an Encounter Note, read only the first 3 pages to save time
            max_pages = len(result.pages) if category == 'Encounter Note' else min(3, len(result.pages))
            for i in range(max_pages):
                page = result.pages[i]
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            text += word.value + ' '
                        text += '\n' + ' '
            return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return ''


def _check_note(output_file_path):
    """
    Check if the patient output file exists and has contents.

    Parameters:
    - output_file_path (str): The path to the patient's output file.

    Returns:
    - bool: True if the file exists and is not empty, False otherwise.
    """
    return os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0

def main(big_folder_path):
    """
    Process all patient folders within a big folder.

    Parameters:
    - big_folder_path (str): The path to the big folder containing individual patient folders.

    Returns:
    - total_length (int): Total length of all patient notes.
    - avg_length (float): Average length of all patient notes.
    """
    total_length = 0
    total_files = 0
    output_folder = os.path.join(os.path.dirname(big_folder_path), 'postprocessed_gbm')
    os.makedirs(output_folder, exist_ok=True)

    patient_folders = [f for f in os.listdir(big_folder_path) if os.path.isdir(os.path.join(big_folder_path, f))]

    try:
        # Iterate through each patient folder in the big folder with a progress bar
        with tqdm(total=len(patient_folders), desc="Processing Patients") as pbar:
            for patient_folder_name in patient_folders:
                patient_folder_path = os.path.join(big_folder_path, patient_folder_name)

                # Create a text file to store the extracted information
                output_file_path = os.path.join(output_folder, f"{patient_folder_name}.txt")

                # Check if the patient file already exists and has content
                if _check_note(output_file_path):
                    # Skip processing this patient if the file already exists and has content
                    pbar.update(1)
                    continue

                with open(output_file_path, 'w') as output_file:
                    patient_total_length = 0
                    files_processed = 0
                    # Define order of categories
                    category_order = ['Encounter Note', 'Radiology Test', 'Genomic Test', 'Surgical Pathology',
                                      'Molecular Oncology']
                    # Iterate through each category in defined order
                    for category in category_order:
                        for file_name in os.listdir(patient_folder_path):
                            if file_name.endswith('.pdf') and categorize_file(file_name) == category:
                                file_path = os.path.join(patient_folder_path, file_name)

                                if category == 'Molecular Oncology':
                                    # Skip reading PDF pages for Molecular Oncology category
                                    continue

                                # Read the text from the PDF
                                extracted_text = read_pdf_pages(file_path, category)

                                # Extract relevant text based on the category
                                relevant_text = text_extraction(extracted_text, category)

                                # Write the output to the text file
                                output_file.write(f"{category} [\st]\n{relevant_text}\n[\en]\n\n")

                                # Update total length for this patient
                                patient_total_length += len(relevant_text)
                                files_processed += 1

                    # Print the length of the text and the number of files processed for this patient
                    print(
                        f"Patient: {patient_folder_name}, Length of text: {patient_total_length}, Files processed: {files_processed}")

                    # Update overall totals
                    total_length += patient_total_length
                    total_files += files_processed

                # Update progress bar
                pbar.update(1)
    except Exception as e:
        print(f"Error processing folder {big_folder_path}: {e}")

    # Calculate average length
    avg_length = total_length / total_files if total_files > 0 else 0
    print(f"Total Length: {total_length}, Average Length: {avg_length}")
    return total_length, avg_length

if __name__ == "__main__":
    # Example usage
    # Replace 'example_big_folder' with the path to the actual big folder containing patient folders
    big_folder_path = '../HIPPA_data'
    main(big_folder_path)
