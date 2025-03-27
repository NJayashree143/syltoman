import fitz  # PyMuPDF
import re
import chromadb
import os
from langchain_ollama import OllamaLLM
#HI

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

# Define a collection for storing outputs
collection_name = "program_outputs"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "Collection for storing program outputs"},
)


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
        if not line or re.match(r"^\d+\.\s*:$", line) or re.match(r"^\d+$", line):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_lab_programs_function1(text):
    """Extract and split lab programs for function1."""
    start_keywords = ["Programming Exercises:", "Experiments", "List of Experiments"]
    end_keywords = ["Course outcomes", "Assessment Details", "SEE for IC"]

    start_index = -1
    for keyword in start_keywords:
        index = text.lower().find(keyword.lower())
        if index != -1:
            start_index = index + len(keyword)
            break

    if start_index == -1:
        return ["No lab programs found."]

    extracted_text = text[start_index:]

    for end_keyword in end_keywords:
        end_index = extracted_text.lower().find(end_keyword.lower())
        if end_index != -1:
            extracted_text = extracted_text[:end_index]
            break

    extracted_text = clean_text(extracted_text)
    lab_programs = re.split(r"\n(?=\bDevelop|\bWrite|\bDesign)", extracted_text)
    return [prog.strip() for prog in lab_programs if prog.strip()]


def extract_lab_programs_function2(text):
    """Extract and split all lab programs for function2."""
    start_keywords = ["Laboratory Component", "Lab Section", "Programming Exercises:", "List of Experiments"]
    end_keywords = ["Teaching-Learning Process", "Course outcomes", "Assessment Details", "SEE for IC"]

    lab_sections = []
    start_indices = []

    for keyword in start_keywords:
        start_indices.extend([m.start() + len(keyword) for m in re.finditer(keyword, text, re.IGNORECASE)])

    if not start_indices:
        return ["No lab programs found."]

    start_indices.sort()

    for start_index in start_indices:
        end_index = len(text)

        for end_keyword in end_keywords:
            found_index = text.lower().find(end_keyword.lower(), start_index)
            if found_index != -1 and found_index < end_index:
                end_index = found_index
                break

        extracted_text = text[start_index:end_index].strip()
        extracted_text = clean_text(extracted_text)

        lab_programs = re.split(r"\n(?=\d+\.\s|\bDevelop|\bWrite|\bDesign|\bImplement|\bSimulate)", extracted_text)
        lab_sections.append([prog.strip() for prog in lab_programs if prog.strip()])

    return lab_sections


def function1(pdf_path, save_to_file=False):
    """Agent function to extract lab programs from a PDF and optionally save to a text file."""
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return "No text found in the PDF."

    lab_programs = extract_lab_programs_function1(text)

    output = "Extracted Lab Programs List:\n"
    if lab_programs and lab_programs[0] != "No lab programs found.":
        for i, program in enumerate(lab_programs, 1):
            output += f"{i}. {program}\n\n"
    else:
        output += "No lab programs found in the document."

    if save_to_file:
        with open("lab_programs_function1.txt", "w", encoding="utf-8") as f:
            f.write(output)

    return output


def function2(pdf_path, save_to_file=False):
    """AI agent function to extract lab programs from a PDF and optionally save to a text file."""
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return "No text found in the PDF."

    lab_programs = extract_lab_programs_function2(text)

    output = "Extracted Lab Programs List:\n"
    if lab_programs and lab_programs[0] != "No lab programs found.":
        for section_num, section in enumerate(lab_programs, 1):
            output += f"Lab Section {section_num}:\n"
            for program in section:
                output += f"{program}\n"
            output += "\n"
    else:
        output += "No lab programs found in the document."

    if save_to_file:
        with open("lab_programs_function2.txt", "w", encoding="utf-8") as f:
            f.write(output)

    return output


def store_output_in_chromadb(output, doc_id):
    """Store the output from the agent in ChromaDB."""
    try:
        if not output.strip():
            print("No output to store in ChromaDB.")
            return

        # Check if the document ID already exists
        existing_docs = collection.get(ids=[doc_id])
        if existing_docs['documents']:
            print(f"Document ID '{doc_id}' already exists. Updating the document.")
            collection.update(
                documents=[output],
                ids=[doc_id]
            )
        else:
            collection.add(
                documents=[output],
                ids=[doc_id]
            )
        print(f"Successfully stored output in ChromaDB with ID: {doc_id}")
    except Exception as e:
        print(f"Error storing output in ChromaDB: {e}")


def retrieve_programs_from_chromadb():
    """Retrieve all stored programs from ChromaDB."""
    try:
        results = collection.get()  # Retrieve all documents
        if results['documents']:
            return results['documents']
        else:
            print("No documents found in the collection.")
            return []
    except Exception as e:
        print(f"Error retrieving programs from ChromaDB: {e}")
        return []


def generate_explanation(program):
    """Generate an explanation for the given program using a language model."""
    try:
        llm_model = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")
        explanation = llm_model.generate([f"Explain the following program:\n{program}"])  # Pass a list of prompts
        return explanation[0]  # Return the first explanation
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return "Explanation could not be generated. Please ensure the Ollama server is running."


def load_textbook_for_explanation(textbook_path, program_name):
    """Load a textbook (PDF) and extract explanations for a given program."""
    textbook_text = extract_text_from_pdf(textbook_path)
    if not textbook_text.strip():
        return "No text found in the textbook."

    # Search for the program name in the textbook
    program_index = textbook_text.lower().find(program_name.lower())
    if program_index == -1:
        return f"No explanation found for '{program_name}' in the textbook."

    # Extract a section around the program name
    start_index = max(0, program_index - 500)  # 500 characters before
    end_index = min(len(textbook_text), program_index + 500)  # 500 characters after
    explanation = textbook_text[start_index:end_index]

    return explanation


def explain_stored_programs():
    """Retrieve programs from ChromaDB and generate explanations for each."""
    programs = retrieve_programs_from_chromadb()
    explanations = {}

    for program in programs:
        # Try generating explanation using Ollama
        explanation = generate_explanation(program)
        if explanation == "Explanation could not be generated. Please ensure the Ollama server is running.":
            # Fallback to textbook explanation
            textbook_path = r"C:\Users\janan\venv\Manual-B\python.pdf"  # Path to your textbook
            program_name = program.split("\n")[0]  # Use the first line of the program as the name
            explanation = load_textbook_for_explanation(textbook_path, program_name)

        explanations[program] = explanation

    return explanations


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


def check_stored_outputs():
    """Retrieve and print all stored outputs from ChromaDB."""
    try:
        results = collection.get()  # Retrieve all documents
        if results['documents']:
            for doc_id, doc_content in zip(results['ids'], results['documents']):
                print(f"Document ID: {doc_id}")
                print(f"Document Content: {doc_content}\n")
        else:
            print("No documents found in the collection.")
    except Exception as e:
        print(f"Error retrieving programs from ChromaDB: {e}")


# Example usage
if __name__ == "__main__":
    pdf_path = r"C:\Users\janan\venv\Manual-B\syl1.pdf"  # Change this to your file path
    output_from_agent = decision_making_agent(pdf_path)

    # Store the output in ChromaDB
    store_output_in_chromadb(output_from_agent, doc_id="syl1_output")

    # Check the stored outputs in ChromaDB
    check_stored_outputs()

    # Generate explanations for stored programs
    explanations = explain_stored_programs()

    for program, explanation in explanations.items():
        print(f"Program:\n{program}\nExplanation:\n{explanation}\n")