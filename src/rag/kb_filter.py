#!/usr/bin/env python3
"""
Knowledge Base Filter Script for RAG Pipeline

This script processes PDFs from data/docs/{dataset-name}, extracts text,
creates token-based chunks (300-500 tokens with overlap), and uses vLLM
to filter chunks that contain useful information for natural language explanations.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from pypdf import PdfReader
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except Exception as e:
        raise Exception(f"Error extracting text from {pdf_path}: {str(e)}")


def create_token_chunks(
    text: str,
    tokenizer,
    min_tokens: int = 300,
    max_tokens: int = 500,
    overlap_tokens: int = 50
) -> List[Dict[str, Any]]:
    """
    Create chunks of text based on token count with overlap.
    
    Args:
        text: Input text to chunk
        tokenizer: Tokenizer instance for counting tokens
        min_tokens: Minimum tokens per chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        
    Returns:
        List of chunk dictionaries with text and token_count
    """
    # Tokenize the entire text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    start_idx = 0
    chunk_index = 0
    
    while start_idx < len(tokens):
        # Determine chunk end (try to get max_tokens, but at least min_tokens)
        end_idx = min(start_idx + max_tokens, len(tokens))
        
        # If we're near the end and have less than min_tokens, extend to end
        if end_idx - start_idx < min_tokens and end_idx < len(tokens):
            end_idx = min(start_idx + min_tokens, len(tokens))
        
        # Extract chunk tokens and decode to text
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        token_count = len(chunk_tokens)
        
        chunks.append({
            "chunk_index": chunk_index,
            "text": chunk_text,
            "token_count": token_count
        })
        
        chunk_index += 1
        
        # Move start position with overlap
        if end_idx >= len(tokens):
            break
        start_idx = end_idx - overlap_tokens
    
    return chunks


def filter_chunk_with_llm(
    llm: LLM,
    chunk_text: str,
    prompt_template: str
) -> bool:
    """
    Use vLLM to determine if a chunk contains useful information.
    
    Args:
        llm: Initialized vLLM LLM instance
        chunk_text: Text chunk to evaluate
        prompt_template: Prompt template with {chunk_text} placeholder
        
    Returns:
        True if chunk is useful, False otherwise
    """
    # Format the prompt
    prompt = prompt_template.format(chunk_text=chunk_text)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,  # Low temperature for deterministic yes/no
        max_tokens=10,     # Only need yes/no response
        stop=None
    )
    
    # Generate response
    outputs = llm.generate([prompt], sampling_params)
    
    # Extract response
    if outputs and len(outputs) > 0:
        response = outputs[0].outputs[0].text.strip().lower()
        # Check if response indicates usefulness
        return "yes" in response or "true" in response or "1" in response
    
    return False


def process_pdfs(
    dataset_name: str,
    model_name: str = "QuantTrio/Qwen3-30B-A3B-Thinking-2507-AWQ",
    num_chunks: Optional[int] = None,
    min_tokens: int = 300,
    max_tokens: int = 500,
    overlap_tokens: int = 50
):
    """
    Main processing function.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'diabetes')
        model_name: vLLM model name
        num_chunks: Optional limit on number of chunks per document (for debugging)
        min_tokens: Minimum tokens per chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
    """
    # Get paths
    main_dir = Path(__file__).parent.parent.parent
    docs_dir = main_dir / "data" / "docs" / dataset_name
    output_dir = main_dir / "data" / "kb" / dataset_name
    
    # Check if docs directory exists
    if not docs_dir.exists():
        raise FileNotFoundError(f"Document directory not found: {docs_dir}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {docs_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    
    # Initialize tokenizer (use same model for tokenizer)
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Initialize vLLM LLM
    print(f"Initializing vLLM with model {model_name}...")
    llm = LLM(model=model_name, trust_remote_code=True)
    
    # Placeholder prompt template (user will refine later)
    prompt_template = """Does the following text chunk contain useful information for creating natural language explanations? 
Respond with 'yes' or 'no' only.

Chunk: {chunk_text}"""
    
    # Process each PDF
    total_chunks_processed = 0
    total_chunks_useful = 0
    
    for pdf_path in pdf_files:
        pdf_name = pdf_path.name
        print(f"\nProcessing: {pdf_name}")
        
        try:
            # Extract text
            print("  Extracting text...")
            text = extract_text_from_pdf(str(pdf_path))
            
            if not text.strip():
                print(f"  Warning: No text extracted from {pdf_name}, skipping...")
                continue
            
            # Create chunks
            print("  Creating chunks...")
            chunks = create_token_chunks(
                text, tokenizer, min_tokens, max_tokens, overlap_tokens
            )
            
            print(f"  Created {len(chunks)} chunks")
            
            # Limit chunks if num_chunks is specified
            if num_chunks is not None:
                chunks = chunks[:num_chunks]
                print(f"  Limited to first {len(chunks)} chunks (debug mode)")
            
            # Filter chunks with LLM
            print("  Filtering chunks with LLM...")
            filtered_chunks = []
            for chunk in chunks:
                is_useful = filter_chunk_with_llm(llm, chunk["text"], prompt_template)
                chunk["is_useful"] = is_useful
                filtered_chunks.append(chunk)
                
                if is_useful:
                    total_chunks_useful += 1
                total_chunks_processed += 1
                
                # Progress indicator
                if total_chunks_processed % 10 == 0:
                    print(f"    Processed {total_chunks_processed} chunks...")
            
            # Save results
            output_file = output_dir / f"{pdf_path.stem}.json"
            output_data = {
                "source_pdf": pdf_name,
                "chunks": filtered_chunks
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            useful_count = sum(1 for c in filtered_chunks if c["is_useful"])
            print(f"  Saved {len(filtered_chunks)} chunks ({useful_count} useful) to {output_file}")
            
        except Exception as e:
            print(f"  Error processing {pdf_name}: {str(e)}")
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total chunks processed: {total_chunks_processed}")
    print(f"Total chunks marked as useful: {total_chunks_useful}")
    if total_chunks_processed > 0:
        print(f"Usefulness rate: {100 * total_chunks_useful / total_chunks_processed:.2f}%")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter PDF documents for RAG knowledge base'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Name of the dataset (e.g., diabetes)'
    )
    parser.add_argument(
        '--num-chunks',
        type=int,
        default=None,
        help='Limit number of chunks per document for debugging (default: process all)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default="QuantTrio/Qwen3-30B-A3B-Thinking-2507-AWQ",
        help='vLLM model name (default: QuantTrio/Qwen3-30B-A3B-Thinking-2507-AWQ)'
    )
    parser.add_argument(
        '--min-tokens',
        type=int,
        default=300,
        help='Minimum tokens per chunk (default: 300)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=500,
        help='Maximum tokens per chunk (default: 500)'
    )
    parser.add_argument(
        '--overlap-tokens',
        type=int,
        default=50,
        help='Number of tokens to overlap between chunks (default: 50)'
    )
    
    args = parser.parse_args()
    
    process_pdfs(
        dataset_name=args.dataset,
        model_name=args.model,
        num_chunks=args.num_chunks,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens
    )


if __name__ == "__main__":
    main()
