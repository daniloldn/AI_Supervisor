from  pathlib import Path
from pypdf import PdfReader


def load_pdf(folder_path:str) ->list[dict]:

    """
    Load all PDFs from a folder and extract their text.

    Returns:
        A list of dictionaries like:
        [
            {"source": "file_name.pdf", "text": "..."},
            ...
        ]
    """

    folder = Path(folder_path)

    #checking if folder exists
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    #list of pdfs
    pdf_list = list(folder.glob("*.pdf"))

    documents = []

    for pdf in pdf_list:
        reader = PdfReader(pdf)

        full_text = []

        #reading the pages of the given pdf
        for pages in reader.pages:
            page_text = pages.extract_text()

            #if page not empty
            if page_text:
                full_text.append(page_text)

            #document appending 
        documents.append(
                {
                    "source": pdf.name, 
                    "text": "\n".join(full_text)
                }
            )

    return documents