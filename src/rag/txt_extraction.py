#!/usr/bin/env python3
"""
PDF to Text Extraction Script

This script extracts text from all PDF files in data/docs/{dataset} folder
and saves them as .txt files with the same name.

Uses PyMuPDF (fitz) for better text extraction quality, with fallback to pypdf.
"""

import argparse
import re
from pathlib import Path

# Try to import PyMuPDF (fitz) first - better text extraction
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    try:
        from pypdf import PdfReader
        HAS_PYPDF = True
    except ImportError:
        HAS_PYPDF = False


def clean_text(text: str) -> str:
    """
    Clean extracted text by normalizing whitespace and fixing common issues.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Normalize multiple spaces to single space
    text = re.sub(r' +', ' ', text)
    
    # Fix spacing issues where letters are separated (e.g., "T h e" -> "The")
    # This pattern matches single letters separated by spaces, but preserves
    # intentional single-letter words and abbreviations
    text = re.sub(r'\b([a-zA-Z]) ([a-zA-Z]) ([a-zA-Z]) ([a-zA-Z]) ([a-zA-Z])\b', r'\1\2\3\4\5', text)
    text = re.sub(r'\b([a-zA-Z]) ([a-zA-Z]) ([a-zA-Z]) ([a-zA-Z])\b', r'\1\2\3\4', text)
    text = re.sub(r'\b([a-zA-Z]) ([a-zA-Z]) ([a-zA-Z])\b', r'\1\2\3', text)
    text = re.sub(r'\b([a-zA-Z]) ([a-zA-Z])\b', r'\1\2', text)
    
    # Normalize line breaks - multiple newlines to double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove trailing whitespace from lines
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove excessive spaces around punctuation
    text = re.sub(r' +([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?]) +', r'\1 ', text)
    
    return text.strip()


def extract_text_with_pymupdf(pdf_path: Path) -> str:
    """
    Extract text using PyMuPDF (fitz) - provides cleaner extraction.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text


def extract_text_with_pypdf(pdf_path: Path) -> str:
    """
    Extract text using pypdf (fallback method).
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_text_from_pdf(pdf_path: Path, clean: bool = True) -> str:
    """
    Extract text from a PDF file using the best available method.
    
    Args:
        pdf_path: Path to the PDF file
        clean: Whether to clean the extracted text
        
    Returns:
        Extracted text as a string
    """
    try:
        if HAS_PYMUPDF:
            text = extract_text_with_pymupdf(pdf_path)
        elif HAS_PYPDF:
            text = extract_text_with_pypdf(pdf_path)
        else:
            raise ImportError("Neither PyMuPDF nor pypdf is installed. Please install one: pip install pymupdf")
        
        if clean:
            text = clean_text(text)
        
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from {pdf_path}: {str(e)}")


def extract_pdfs_from_dataset(dataset: str, clean: bool = True, overwrite: bool = False):
    """
    Extract text from all PDF files in data/docs/{dataset} folder.
    
    Args:
        dataset: Name of the dataset folder
        clean: Whether to clean the extracted text
        overwrite: Whether to overwrite existing .txt files
    """
    # Get paths
    main_dir = Path(__file__).parent.parent.parent
    docs_dir = main_dir / "data" / "docs" / dataset
    
    # Check if docs directory exists
    if not docs_dir.exists():
        raise FileNotFoundError(f"Document directory not found: {docs_dir}")
    
    # Get all PDF files
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {docs_dir}")
        return
    
    # Print extraction method being used
    if HAS_PYMUPDF:
        print("Using PyMuPDF (fitz) for text extraction (recommended)")
    elif HAS_PYPDF:
        print("Using pypdf for text extraction (fallback)")
    else:
        raise ImportError("Neither PyMuPDF nor pypdf is installed. Please install one: pip install pymupdf")
    
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    
    # Process each PDF file
    for pdf_path in pdf_files:
        try:
            txt_path = pdf_path.with_suffix('.txt')
            
            # Skip if file exists and overwrite is False
            if txt_path.exists() and not overwrite:
                print(f"Skipping {pdf_path.name} (txt file already exists, use --overwrite to replace)")
                continue
            
            print(f"Processing {pdf_path.name}...")
            
            # Extract text
            text = extract_text_from_pdf(pdf_path, clean=clean)
            
            # Save as .txt file with the same name
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"  Saved text to {txt_path.name} ({len(text)} characters)")
            
        except Exception as e:
            print(f"  Error processing {pdf_path.name}: {str(e)}")
            continue
    
    print(f"\nExtraction complete! Processed {len(pdf_files)} PDF file(s).")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract text from PDF files in data/docs/{dataset} folder"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset folder (e.g., 'diabetes', 'lendingclub')"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Disable text cleaning (keep raw extracted text)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt files"
    )
    
    args = parser.parse_args()
    
    try:
        extract_pdfs_from_dataset(args.dataset, clean=not args.no_clean, overwrite=args.overwrite)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
