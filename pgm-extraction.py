import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path):
    """Extract text from the given PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return ""

def clean_text(text):
    """Remove headers, footers, and unwanted lines."""
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Remove empty lines
        if not line:
            continue

        # Remove date formats like 11.12.2022
        if re.match(r"^\d{1,2}\.\d{1,2}\.\d{4}$", line):
            continue

        # Remove standalone numbers (page numbers)
        if re.match(r"^\d+$", line):
            continue

        # Remove unwanted headers
        if line.lower() in ["experiments", "list of experiments", "programming exercises"]:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def extract_lab_programs(text):
    """Extract and split lab programs correctly."""
    start_keywords = ["Programming Exercises:", "Experiments", "List of Experiments"]
    end_keywords = ["Course outcomes", "Assessment Details", "SEE for IC"]

    # Find start position
    start_index = -1
    for keyword in start_keywords:
        index = text.lower().find(keyword.lower())
        if index != -1:
            start_index = index + len(keyword)  # Skip the keyword itself
            break

    if start_index == -1:
        return ["No lab programs found."]

    # Extract everything after the detected start keyword
    extracted_text = text[start_index:]

    # Find end position (first occurrence of any end keyword)
    for end_keyword in end_keywords:
        end_index = extracted_text.lower().find(end_keyword.lower())
        if end_index != -1:
            extracted_text = extracted_text[:end_index]
            break  # Stop at the first matching end keyword

    # Remove headers/footers
    extracted_text = clean_text(extracted_text)

    # Split programs using strong detection methods
    lab_programs = re.split(r"\n(?=\bDevelop|\bWrite|\bDesign)", extracted_text)

    # Remove unwanted whitespace and numbering issues
    lab_programs = [prog.strip() for prog in lab_programs if prog.strip()]

    return lab_programs

def function1(pdf_path, save_to_file=False):
    """Agent function to extract lab programs from a PDF and optionally save to a text file."""
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return "No text found in the PDF."

    lab_programs = extract_lab_programs(text)

    # Display extracted lab programs
    output = "Extracted Lab Programs List:\n"
    if lab_programs and lab_programs[0] != "No lab programs found.":
        for i, program in enumerate(lab_programs, 1):
            output += f"{i}. {program}\n\n"  # Add an extra newline for spacing after each program
    else:
        output += "No lab programs found in the document."

    # Optional: Save to a text file
    if save_to_file:
        with open("lab_programs.txt", "w", encoding="utf-8") as f:
            for program in lab_programs:
                f.write(program + "\n\n")  # Add an extra newline for spacing in the file as well

    return output

# Example usage
if __name__ == "__main__":
    pdf_path = r"C:\Users\janan\venv\Manual-B\syl2.pdf"  # Change this to your file path
    result = function1(pdf_path, save_to_file=True)
    print(result)