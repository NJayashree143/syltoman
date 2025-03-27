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

        # Remove placeholders like "1. :"
        if re.match(r"^\d+\.\s*:$", line):
            continue

        # Remove standalone numbers (page numbers)
        if re.match(r"^\d+$", line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def extract_lab_programs_function1(text):
    """Extract and split lab programs for function1."""
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

def extract_lab_programs_function2(text):
    """Extract and split all lab programs for function2."""
    start_keywords = ["Laboratory Component", "Lab Section", "Programming Exercises:", "List of Experiments"]
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

def function1(pdf_path, save_to_file=False):
    """Agent function to extract lab programs from a PDF and optionally save to a text file."""
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return "No text found in the PDF."

    lab_programs = extract_lab_programs_function1(text)

    # Display extracted lab programs
    output = "Extracted Lab Programs List:\n"
    if lab_programs and lab_programs[0] != "No lab programs found.":
        for i, program in enumerate(lab_programs, 1):
            output += f"{i}. {program}\n\n"  # Add an extra newline for spacing after each program
    else:
        output += "No lab programs found in the document."

    # Optional: Save to a text file
    if save_to_file:
        with open("lab_programs_function1.txt", "w", encoding="utf-8") as f:
            for program in lab_programs:
                f.write(program + "\n\n")  # Add an extra newline for spacing in the file as well

    return output

def function2(pdf_path, save_to_file=False):
    """AI agent function to extract lab programs from a PDF and optionally save to a text file."""
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return "No text found in the PDF."

    lab_programs = extract_lab_programs_function2(text)

    # Prepare output
    output = "Extracted Lab Programs List:\n"
    if lab_programs and lab_programs[0] != "No lab programs found.":
        for section_num, section in enumerate(lab_programs, 1):
            output += f"Lab Section {section_num}:\n"
            for program in section:
                output += f"{program}\n"
            output += "\n"
    else:
        output += "No lab programs found in the document."

    # Optional: Save to a text file
    if save_to_file:
        with open("lab_programs_function2.txt", "w", encoding="utf-8") as f:
            for section_num, section in enumerate(lab_programs, 1):
                f.write(f"Lab Section {section_num}:\n\n")
                for program in section:
                    f.write(program + "\n")
                f.write("\n")

    return output

def decision_making_agent(pdf_path):
    """Decide which function to call based on the content of the PDF."""
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return "No text found in the PDF."

    # Simple heuristic to decide which function to use
    if "Programming Exercises" in text or "Experiments" in text:
        return function1(pdf_path, save_to_file=True)
    else:
        return function2(pdf_path, save_to_file=True)

# Example usage
if __name__ == "__main__":
    pdf_path = r"C:\Users\janan\venv\Manual-B\syl1.pdf"  # Change this to your file path
    result = decision_making_agent(pdf_path)
    print(result)