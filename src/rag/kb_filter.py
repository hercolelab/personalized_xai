#!/usr/bin/env python3
"""
Knowledge Base Filter Script for RAG Pipeline

This script processes text files from data/docs/{dataset-name}, creates token-based 
chunks (300-500 tokens with overlap), and uses Ollama to filter chunks that contain 
useful information for natural language explanations.
"""

import argparse
import json
import os
import re
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env if present (project root or current dir)
load_dotenv()


def load_text_from_file(txt_path: str) -> str:
    """
    Load text from a text file.
    
    Args:
        txt_path: Path to the text file
        
    Returns:
        Text content as a string
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise Exception(f"Error loading text from {txt_path}: {str(e)}")


def find_sentence_boundaries(text: str) -> List[int]:
    """
    Find sentence boundary positions in text.
    Sentence boundaries are defined as: . ! ? followed by whitespace or end of text.
    
    Args:
        text: Input text
        
    Returns:
        List of character positions where sentences end (exclusive, i.e., position after sentence end)
    """
    boundaries = [0]  # Start of text is always a boundary
    # Pattern: sentence-ending punctuation followed by whitespace or end of string
    pattern = r'[.!?]+(?:\s+|$)'
    
    for match in re.finditer(pattern, text):
        # Position after the punctuation and whitespace
        boundaries.append(match.end())
    
    # Always include the end of text as a boundary
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    
    return boundaries


def find_next_sentence_start(text: str, pos: int, sentence_boundaries: List[int]) -> int:
    """
    Find the start of the next sentence from position pos.
    The start is the position after a sentence boundary.
    
    Args:
        text: Full text
        pos: Current position
        sentence_boundaries: List of sentence boundary positions
        
    Returns:
        Character position of next sentence start
    """
    for boundary in sentence_boundaries:
        if boundary > pos:
            return boundary
    return len(text)


def find_previous_sentence_end(text: str, pos: int, sentence_boundaries: List[int]) -> int:
    """
    Find the end of the previous sentence before position pos.
    
    Args:
        text: Full text
        pos: Current position
        sentence_boundaries: List of sentence boundary positions
        
    Returns:
        Character position of previous sentence end
    """
    for boundary in reversed(sentence_boundaries):
        if boundary <= pos:
            return boundary
    return 0


def create_token_chunks(
    text: str,
    encoding,
    min_tokens: int = 300,
    max_tokens: int = 500,
    overlap_tokens: int = 50
) -> List[Dict[str, Any]]:
    """
    Create chunks of text based on token count with overlap.
    Chunks are aligned to sentence boundaries to avoid truncating sentences.
    
    Args:
        text: Input text to chunk
        encoding: Tiktoken encoding instance for counting tokens
        min_tokens: Minimum tokens per chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        
    Returns:
        List of chunk dictionaries with text and token_count
    """
    if not text.strip():
        return []
    
    # Find sentence boundaries in the original text
    sentence_boundaries = find_sentence_boundaries(text)
    
    chunks = []
    start_char = 0
    chunk_index = 0
    
    while start_char < len(text):
        # Start from a sentence boundary
        start_char = find_next_sentence_start(text, start_char, sentence_boundaries)
        if start_char >= len(text):
            break
        
        # Find the target end position that gives us approximately max_tokens
        # We'll search for a character position that yields close to max_tokens
        low = start_char
        high = len(text)
        target_end_char = min(start_char + max_tokens * 4, len(text))  # Rough estimate: ~4 chars per token
        
        # Refine the target position by checking token counts
        for _ in range(10):  # Limit iterations
            mid = (low + high) // 2
            candidate_text = text[start_char:mid]
            candidate_tokens = encoding.encode(candidate_text)
            token_count = len(candidate_tokens)
            
            if token_count < max_tokens:
                low = mid
                target_end_char = mid
            elif token_count > max_tokens:
                high = mid
            else:
                target_end_char = mid
                break
        
        # Now find the appropriate sentence boundary
        # Priority: complete sentences, even if it means exceeding max_tokens
        end_char = None
        
        # First, check if target_end_char is already at a sentence boundary
        if target_end_char in sentence_boundaries:
            end_char = target_end_char
        else:
            # Find the next sentence boundary after target_end_char to complete the sentence
            for boundary in sentence_boundaries:
                if boundary <= start_char:
                    continue
                if boundary < target_end_char:
                    continue
                # This is the next sentence boundary - use it to complete the sentence
                end_char = boundary
                break
        
        # If we still don't have an end_char, use the end of text
        if end_char is None:
            end_char = len(text)
        
        # Verify we have at least min_tokens, if not, extend forward
        candidate_text = text[start_char:end_char]
        candidate_tokens = encoding.encode(candidate_text)
        if len(candidate_tokens) < min_tokens:
            # Need to extend to get min_tokens - find next boundary that gives us min_tokens
            for boundary in sentence_boundaries:
                if boundary <= end_char:
                    continue
                candidate_text = text[start_char:boundary]
                candidate_tokens = encoding.encode(candidate_text)
                if len(candidate_tokens) >= min_tokens:
                    end_char = boundary
                    break
        
        # Extract the final chunk text
        chunk_text = text[start_char:end_char].strip()
        
        if not chunk_text:
            # If we got empty text, advance and try again
            start_char = find_next_sentence_start(text, start_char + 1, sentence_boundaries)
            continue
        
        # Get accurate token count
        chunk_tokens = encoding.encode(chunk_text)
        token_count = len(chunk_tokens)
        
        chunks.append({
            "chunk_index": chunk_index,
            "text": chunk_text,
            "token_count": token_count
        })
        
        chunk_index += 1
        
        # Move start position with overlap
        # Calculate how many characters to go back for overlap
        if end_char >= len(text):
            break
        
        # Estimate character position for overlap
        # We want to go back by approximately overlap_tokens worth of characters
        if token_count > 0:
            chars_per_token = len(chunk_text) / token_count
            overlap_chars = int(overlap_tokens * chars_per_token)
            new_start = max(start_char + 1, end_char - overlap_chars)
        else:
            new_start = start_char + 1
        
        # Ensure we don't go backwards
        start_char = max(start_char + 1, new_start)
    
    return chunks


def filter_chunk_with_llm(
    model_name: str,
    chunk_text: str,
    prompt_template: str,
    api_url: str,
    api_key: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Use Ollama to determine if a chunk contains useful information.
    
    Args:
        model_name: Ollama model name (e.g., 'llama3', 'gpt-oss:20b-cloud')
        chunk_text: Text chunk to evaluate
        prompt_template: Prompt template with {chunk_text} placeholder
        api_url: Ollama API URL (local or cloud)
        api_key: Optional API key for cloud models
        
    Returns:
        Tuple of (is_useful: bool, response_text: str)
    """
    # Format the prompt
    prompt = prompt_template.format(chunk_text=chunk_text)
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": { "temperature": 0 }
    }
    
    try:
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        response_text = response.json()['response']
        
        if response_text:
            response = str(response_text).strip()
            
            # Try to extract JSON from the response
            try:
                # Look for JSON object in the response (handle nested braces)
                # Find the first { and try to match balanced braces
                start_idx = response.find('{')
                if start_idx != -1:
                    brace_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(response)):
                        char = response[i]
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    
                    if brace_count == 0:
                        json_str = response[start_idx:end_idx].strip()
                        result = json.loads(json_str)
                        final_answer = result.get('final_answer', '').strip().lower()
                        # Handle both "Yes"/"No" and "yes"/"no"
                        is_useful = final_answer in ["yes", "y"]
                        return (is_useful, response)
            except (json.JSONDecodeError, AttributeError, KeyError, ValueError) as e:
                # If JSON parsing fails, continue to fallback methods
                pass
            
            # Fallback: Look for the old format # FINAL ANSWER: Yes/No
            final_answer_match = re.search(
                r'#\s*FINAL\s+ANSWER:\s*(Yes|No|yes|no|YES|NO)',
                response,
                re.IGNORECASE | re.MULTILINE
            )
            
            if final_answer_match:
                answer = final_answer_match.group(1).lower()
                is_useful = answer == "yes"
                return (is_useful, response)
            
            # Final fallback: check for yes/no in response
            response_lower = response.lower()
            is_useful = "yes" in response_lower or "true" in response_lower or "1" in response_lower
            return (is_useful, response)
        
        return (False, "")
    except Exception as e:
        error_msg = f"Error calling Ollama: {str(e)}"
        print(f"    Warning: {error_msg}")
        return (False, error_msg)


def get_prompt_template(dataset_name: str) -> str:
    """
    Generate a dataset-specific prompt template for filtering chunks.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'diabetes', 'lendingclub')
        
    Returns:
        Prompt template string with {chunk_text} placeholder
    """
    if dataset_name.lower() == "diabetes":
        return """You are an expert evaluator for an Explainable AI (XAI) system focused on Diabetes prediction. 
    Your task is to determine if the following text chunk contains specific "grounding information" necessary to translate technical data (like SHAP values or Counterfactuals) into human-centered narratives.

    # RUBRIC FOR POSITIVE ANSWER
    To answer "Yes", the text must contain at least one of the following regarding diabetes risk factors (e.g., BMI, Glucose, Insulin, Blood Pressure, Age, Pregnancies, Pedigree Function):

    1. **Feature Meaning:** Definitions or simple explanations of what a medical feature is.
    2. **Risk Impact:** Explanations of *how* or *why* a specific feature increases or decreases diabetes risk (mechanism).
    3. **Actionable Advice:** Suggestions on how a user might modify this feature (e.g., lifestyle changes, diet, exercise) to lower their risk.
    4. **Contextual Values:** Information about normal ranges, thresholds, or what constitutes a "high" or "low" value (crucial for explaining counterfactual changes).

    # RUBRIC FOR NEGATIVE ANSWER
    If the text is generic, unrelated to diabetes features, or purely structural (like references or formatting), the answer is "No".

    Chunk: {chunk_text}

    # THINKING PROCESS
    Think through your reasoning step by step:
    1. Identify any specific diabetes features mentioned in the text.
    2. Check if the text provides definitions, causal impacts, actionable advice, or value ranges for those features.
    3. Determine if this information would help a non-expert understand *why* their risk is high or *how* to change it.

    # OUTPUT FORMAT
    Provide your final answer in the exact format specified: 
    {{
        "reasoning": "<your reasoning process here>",
        "final_answer": "<Yes/No>"
    }}
    """
    
    elif dataset_name.lower() == "lendingclub":
        return """You are an expert evaluator for an Explainable AI (XAI) system focused on Loan Default prediction. 
    Your task is to determine if the following text chunk contains specific "grounding information" necessary to translate technical data (like SHAP values or Counterfactuals) into human-centered narratives.

    # RUBRIC FOR POSITIVE ANSWER
    To answer "Yes", the text must contain at least one of the following regarding loan risk factors:

    The relevant features to search for are:
    - Loan amount (loan_amnt): The amount of money requested/borrowed
    - Duration of repayment period (term): The loan term length (e.g., 36 months, 60 months)
    - Debt-To-Income ratio (dti): Ratio of monthly debt payments to monthly income
    - Revolving line utilization rate (revol_util): Percentage of available credit being used
    - Annual Income (annual_inc): Yearly income of the borrower
    - FICO credit score (fico): Credit score (fico_high, fico_low, or average)
    - Public record bankruptcies (pub_rec_bankruptcies): Number of bankruptcies on public record
    - Employment length years (emp_length): Years of employment
    - Number of credit inquiries in the past 6 months (inq_last_6mths): Recent credit inquiries
    - Home ownership (home_ownership): Ownership status (own, rent, mortgage, etc.)
    - Purpose (purpose): Loan purpose (debt consolidation, credit card, etc.)
    - Open accounts (open_acc): Number of open credit accounts

    For each relevant feature found, the text should provide at least one of:
    1. **Feature Meaning:** Definitions or simple explanations of what the feature means.
    2. **Risk Impact:** Explanations of *how* or *why* a specific feature increases or decreases loan default risk (mechanism).
    3. **Actionable Advice:** Suggestions on how a borrower might modify this feature (e.g., reduce debt, improve credit score) to improve their loan approval chances.
    4. **Contextual Values:** Information about normal ranges, thresholds, or what constitutes a "high" or "low" value (crucial for explaining counterfactual changes).

    # RUBRIC FOR NEGATIVE ANSWER
    If the text is generic, unrelated to loan/credit features, or purely structural (like references or formatting), the answer is "No".

    Chunk: {chunk_text}

    # THINKING PROCESS
    Think through your reasoning step by step:
    1. Identify any specific loan/credit features mentioned in the text (from the list above).
    2. Check if the text provides definitions, causal impacts, actionable advice, or value ranges for those features.
    3. Determine if this information would help a non-expert understand *why* their loan application was rejected/accepted or *how* to improve their chances.

    # OUTPUT FORMAT
    Provide your final answer in the exact format specified: 
    {{
        "reasoning": "<your reasoning process here>",
        "final_answer": "<Yes/No>"
    }}
    """
    
    else:
        # Generic template for unknown datasets
        return """You are an expert evaluator for an Explainable AI (XAI) system. 
    Your task is to determine if the following text chunk contains specific "grounding information" necessary to translate technical data (like SHAP values or Counterfactuals) into human-centered narratives.

    # RUBRIC FOR POSITIVE ANSWER
    To answer "Yes", the text must contain at least one of the following:

    1. **Feature Meaning:** Definitions or simple explanations of what a feature is.
    2. **Risk Impact:** Explanations of *how* or *why* a specific feature increases or decreases risk (mechanism).
    3. **Actionable Advice:** Suggestions on how a user might modify this feature to improve outcomes.
    4. **Contextual Values:** Information about normal ranges, thresholds, or what constitutes a "high" or "low" value (crucial for explaining counterfactual changes).

    # RUBRIC FOR NEGATIVE ANSWER
    If the text is generic, unrelated to relevant features, or purely structural (like references or formatting), the answer is "No".

    Chunk: {chunk_text}

    # THINKING PROCESS
    Think through your reasoning step by step:
    1. Identify any specific features mentioned in the text.
    2. Check if the text provides definitions, causal impacts, actionable advice, or value ranges for those features.
    3. Determine if this information would help a non-expert understand *why* their outcome occurred or *how* to change it.

    # OUTPUT FORMAT
    Provide your final answer in the exact format specified: 
    {{
        "reasoning": "<your reasoning process here>",
        "final_answer": "<Yes/No>"
    }}
    """


def process_txt_files(
    dataset_name: str,
    model_name: str = "gpt-oss:20b-cloud",
    num_chunks: Optional[int] = None,
    min_tokens: int = 300,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    tokenizer_model: Optional[str] = None
):
    """
    Main processing function.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'diabetes')
        model_name: Ollama model name (e.g., 'llama3', 'mistral')
        num_chunks: Optional limit on number of chunks per document (for debugging)
        min_tokens: Minimum tokens per chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        tokenizer_model: Optional encoding name for tiktoken (default: cl100k_base)
    """
    # Get paths
    main_dir = Path(__file__).parent.parent.parent
    docs_dir = main_dir / "data" / "docs" / dataset_name
    output_dir = main_dir / "data" / "kb" / dataset_name
    per_doc_dir = output_dir / "per_doc"
    
    # Check if docs directory exists
    if not docs_dir.exists():
        raise FileNotFoundError(f"Document directory not found: {docs_dir}")
    
    # Create output directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    per_doc_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all text files
    txt_files = list(docs_dir.glob("*.txt"))
    if not txt_files:
        print(f"No text files found in {docs_dir}")
        return
    
    print(f"Found {len(txt_files)} text file(s) to process")
    
    # Initialize encoding for text chunking
    # Note: This encoding is only used for chunking, not for Ollama inference
    # Using tiktoken (cl100k_base) which has no max length limit and works well for token counting
    encoding_name = tokenizer_model or "cl100k_base"
    print(f"Loading tiktoken encoding for chunking: {encoding_name}...")
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        print(f"  Successfully loaded encoding: {encoding_name}")
    except Exception as e:
        print(f"  Warning: Could not load encoding {encoding_name}: {str(e)}")
        print("  Falling back to cl100k_base encoding...")
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            print("  Successfully loaded cl100k_base encoding")
        except Exception as e2:
            raise Exception(f"Failed to load encoding (both {encoding_name} and cl100k_base failed): {str(e2)}")
    
    # Determine if using cloud model and set API URL accordingly
    is_cloud_model = '-cloud' in model_name or ':cloud' in model_name
    if is_cloud_model:
        api_url = "https://ollama.com/api/generate"
        # Get API key from environment variable
        api_key = os.getenv('OLLAMA_API_KEY')
        if not api_key:
            print("[Warning] OLLAMA_API_KEY not set. Cloud models require authentication.")
            print("[Info] Set OLLAMA_API_KEY environment variable or run 'ollama signin'")
    else:
        api_url = "http://localhost:11434/api/generate"
        api_key = None
    
    print(f"Using Ollama API: {api_url}")
    if is_cloud_model:
        print(f"Model: {model_name} (cloud model)")
    else:
        print(f"Model: {model_name} (local model)")
    
    # Generate dataset-specific prompt template
    prompt_template = get_prompt_template(dataset_name)
    
    # Process each text file
    total_chunks_processed = 0
    total_chunks_useful = 0
    all_useful_chunks = []  # Collect all useful chunks for final KB
    
    for txt_path in txt_files:
        txt_name = txt_path.name
        print(f"\nProcessing: {txt_name}")
        
        try:
            # Load text
            print("  Loading text...")
            text = load_text_from_file(str(txt_path))
            
            if not text.strip():
                print(f"  Warning: No text found in {txt_name}, skipping...")
                continue
            
            # Create chunks
            print("  Creating chunks...")
            chunks = create_token_chunks(
                text, encoding, min_tokens, max_tokens, overlap_tokens
            )
            
            print(f"  Created {len(chunks)} chunks")
            
            # Limit chunks if num_chunks is specified
            if num_chunks is not None:
                chunks = chunks[:num_chunks]
                print(f"  Limited to first {len(chunks)} chunks (debug mode)")
            
            # Filter chunks with LLM
            print("  Filtering chunks with Ollama...")
            filtered_chunks = []
            for chunk in chunks:
                is_useful, llm_response = filter_chunk_with_llm(
                    model_name, chunk["text"], prompt_template, api_url, api_key
                )
                chunk["is_useful"] = is_useful
                chunk["llm_response"] = llm_response
                filtered_chunks.append(chunk)
                
                if is_useful:
                    # Create clean chunk with only required fields for final KB
                    useful_chunk = {
                        "source_txt": txt_name,
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"]
                    }
                    all_useful_chunks.append(useful_chunk)
                    total_chunks_useful += 1
                total_chunks_processed += 1
                
                # Progress indicator
                if total_chunks_processed % 10 == 0:
                    print(f"    Processed {total_chunks_processed} chunks...")
            
            # Save results to per_doc subfolder
            output_file = per_doc_dir / f"{txt_path.stem}.json"
            output_data = {
                "source_txt": txt_name,
                "chunks": filtered_chunks
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            useful_count = sum(1 for c in filtered_chunks if c["is_useful"])
            print(f"  Saved {len(filtered_chunks)} chunks ({useful_count} useful) to {output_file}")
            
        except Exception as e:
            print(f"  Error processing {txt_name}: {str(e)}")
            continue
    
    # Save all useful chunks to final KB file
    kb_output_file = output_dir / f"kb-{dataset_name}.json"
    kb_output_data = {
        "dataset": dataset_name,
        "total_useful_chunks": len(all_useful_chunks),
        "chunks": all_useful_chunks
    }
    
    with open(kb_output_file, 'w', encoding='utf-8') as f:
        json.dump(kb_output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(all_useful_chunks)} useful chunks to {kb_output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total chunks processed: {total_chunks_processed}")
    print(f"Total chunks marked as useful: {total_chunks_useful}")
    if total_chunks_processed > 0:
        print(f"Usefulness rate: {100 * total_chunks_useful / total_chunks_processed:.2f}%")
    print(f"Output directory: {output_dir}")
    print(f"Final KB file: {kb_output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter text documents for RAG knowledge base'
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
        default="gpt-oss:20b-cloud",
        help='Ollama model name (default: gpt-oss:20b-cloud)'
    )
    parser.add_argument(
        '--tokenizer-model',
        type=str,
        default=None,
        help='Tiktoken encoding name for chunking (default: cl100k_base). Options: cl100k_base, p50k_base, r50k_base'
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
    
    process_txt_files(
        dataset_name=args.dataset,
        model_name=args.model,
        num_chunks=args.num_chunks,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        tokenizer_model=args.tokenizer_model
    )


if __name__ == "__main__":
    main()
