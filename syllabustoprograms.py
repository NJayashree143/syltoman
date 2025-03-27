#Code to extract lab programs from syllabus pdf.
#Refer my git for code
#it should extrace only lab programs from pdf
#it should work for all 4 pdf

import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path):
    """Extract text from the given PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def clean_text(text):
    """Remove headers, footers, and unwanted lines."""
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Remove empty lines
        if not line:
            continue

        # Remove placeholders like "1. :"
        if re.match(r"^\d+\.\s*:$", line):
            continue

        # Remove standalone numbers (page numbers)
        if re.match(r"^\d+$", line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def extract_lab_programs(text):
    """Extract and split all lab programs correctly."""
    start_keywords = ["Experiments ", "Laboratory Component", "Lab Section", "Programming Exercises:", "List of Experiments"]
    end_keywords = ["Teaching-Learning Process", "Course outcomes", "Assessment Details", "SEE for IC"]

    lab_sections = []
    start_indices = []

    # Find all occurrences of the start keyword
    for keyword in start_keywords:
        start_indices.extend([m.start() + len(keyword) for m in re.finditer(keyword, text, re.IGNORECASE)])

    if not start_indices:
        return ["No lab programs found."]

    # Sort the start indices
    start_indices.sort()

    # Extract sections
    for start_index in start_indices:
        end_index = len(text)  # Default to end of text if no end keyword is found

        for end_keyword in end_keywords:
            found_index = text.lower().find(end_keyword.lower(), start_index)
            if found_index != -1 and found_index < end_index:
                end_index = found_index
                break  # Stop at the first found end keyword

        extracted_text = text[start_index:end_index].strip()
        extracted_text = clean_text(extracted_text)

        # Split lab programs correctly
        lab_programs = re.split(r"\n(?=\d+\.\s|\bDevelop|\bWrite|\bDesign|\bImplement|\bSimulate)", extracted_text)
        lab_programs = [prog.strip() for prog in lab_programs if prog.strip()]

        # Remove "1. :" if it exists
        lab_programs = [prog for prog in lab_programs if prog not in ["1. :"]]

        # Fix numbering
        cleaned_programs = []
        for i, prog in enumerate(lab_programs, 1):
            prog = re.sub(r"^\d+\.\s+", "", prog)  # Remove old numbering
            cleaned_programs.append(f"{i}. {prog}")  # Add correct numbering

        lab_sections.append(cleaned_programs)

    return lab_sections

# Run the script
pdf_path = "syl3.pdf"  # Change this to your file path

text = extract_text_from_pdf(pdf_path)
lab_programs = extract_lab_programs(text)

# Display extracted lab programs
print("Extracted Lab Programs List:\n")
for section_num, section in enumerate(lab_programs, 1):
    print(f"Lab Section {section_num}:\n")
    for program in section:
        print(program)
    print()

# Save to a text file
with open("lab_programs.txt", "w", encoding="utf-8") as f:
    for section_num, section in enumerate(lab_programs, 1):
        f.write(f"Lab Section {section_num}:\n\n")
        for program in section:
            f.write(program + "\n")
        f.write("\n")