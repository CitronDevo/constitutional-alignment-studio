"""
Utility functions for data I/O operations.

Handles loading conversation data and saving constitutional examples.
"""

import json
import os
from pathlib import Path
from typing import List
import warnings

from src.models import ConversationTurn, ConstitutionalExample


def load_dataset(filepath: str) -> List[ConversationTurn]:
    """
    Load conversation dataset from JSONL file.

    Each line should contain a JSON object with 'user' and 'assistant' keys.

    Args:
        filepath: Path to the JSONL file (e.g., data/input/dev.jsonl)

    Returns:
        List of validated ConversationTurn objects

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    conversations = []
    skipped_lines = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                data = json.loads(line)
                conversation = ConversationTurn(**data)
                conversations.append(conversation)

            except json.JSONDecodeError as e:
                skipped_lines.append((line_num, f"Invalid JSON: {e}"))
            except Exception as e:
                skipped_lines.append((line_num, f"Validation error: {e}"))

    # Log warnings for skipped lines
    if skipped_lines:
        warning_msg = f"Skipped {len(skipped_lines)} invalid lines:\n"
        for line_num, reason in skipped_lines[:5]:  # Show first 5
            warning_msg += f"  Line {line_num}: {reason}\n"
        if len(skipped_lines) > 5:
            warning_msg += f"  ... and {len(skipped_lines) - 5} more"
        warnings.warn(warning_msg, UserWarning)

    if not conversations:
        raise ValueError(f"No valid conversations found in {filepath}")

    return conversations


def load_saved_examples(filepath: str) -> List[ConstitutionalExample]:
    """
    Load previously saved constitutional examples.

    Args:
        filepath: Path to the saved examples JSON file

    Returns:
        List of ConstitutionalExample objects (empty list if file doesn't exist)
    """
    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both list format and single-object format
        items = data if isinstance(data, list) else [data]

        # Load examples with backward compatibility for optional metadata fields
        examples = []
        for item in items:
            if 'source_id' not in item:
                warnings.warn(
                    "Skipping saved example because source_id is missing.",
                    UserWarning
                )
                continue
            if 'model_used' not in item:
                item['model_used'] = None
            if 'timestamp' not in item:
                item['timestamp'] = None

            examples.append(ConstitutionalExample(**item))

        return examples

    except Exception as e:
        warnings.warn(f"Failed to load saved examples: {e}", UserWarning)
        return []


def save_example(filepath: str, example: ConstitutionalExample) -> dict:
    """
    Save a constitutional example to the output file using upsert logic.

    UPSERT LOGIC:
    - If an example with the same source_id exists, it will be REPLACED
    - If no example with this source_id exists, it will be APPENDED
    - This ensures one-to-one mapping: one task → one golden example

    Args:
        filepath: Path to save the example (e.g., data/output/constitutional_examples.json)
        example: The validated constitutional example to save

    Returns:
        Dictionary with keys:
        - 'action': 'inserted' or 'updated'
        - 'previous': The previous example if updated, None if inserted

    Raises:
        IOError: If file operations fail
    """
    # Ensure output directory exists
    output_dir = Path(filepath).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing examples
    existing_examples = load_saved_examples(filepath)

    # Check if an example with this source_id already exists
    existing_index = None
    previous_example = None

    if example.source_id is not None:
        for i, ex in enumerate(existing_examples):
            if ex.source_id == example.source_id:
                existing_index = i
                previous_example = ex
                break

    # Upsert logic
    if existing_index is not None:
        # UPDATE: Replace existing example
        existing_examples[existing_index] = example
        action = 'updated'
    else:
        # INSERT: Append new example
        existing_examples.append(example)
        action = 'inserted'

    # Save all examples
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(
                [ex.model_dump() for ex in existing_examples],
                f,
                indent=2,
                ensure_ascii=False
            )
    except Exception as e:
        raise IOError(f"Failed to save example: {e}") from e

    return {
        'action': action,
        'previous': previous_example
    }


def count_saved_examples(filepath: str) -> int:
    """
    Count how many constitutional examples have been saved.

    Args:
        filepath: Path to the saved examples file

    Returns:
        Number of saved examples
    """
    examples = load_saved_examples(filepath)
    return len(examples)


def export_examples_for_training(filepath: str, output_filepath: str) -> None:
    """
    Export saved examples in a format ready for model training.

    This creates a clean JSONL file suitable for fine-tuning or evaluation.

    Args:
        filepath: Path to the saved examples JSON file
        output_filepath: Path to write the training-ready JSONL
    """
    examples = load_saved_examples(filepath)

    output_dir = Path(output_filepath).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_filepath, 'w', encoding='utf-8') as f:
        for example in examples:
            # Format as JSONL for training
            training_item = {
                "principle": example.principle,
                "original": example.original_response,
                "critique": example.critique,
                "rewrite": example.rewrite
            }
            f.write(json.dumps(training_item, ensure_ascii=False) + '\n')


# Testing
if __name__ == "__main__":
    print("=" * 70)
    print("UTILS MODULE DEMONSTRATION")
    print("=" * 70)

    # Test 1: Create sample dataset
    print("\n[TEST 1] Creating sample dataset...")
    sample_data_dir = Path("data/input")
    sample_data_dir.mkdir(parents=True, exist_ok=True)

    sample_conversations = [
        {"user": "Hello!", "assistant": "Hi there! How can I help?"},
        {"user": "What's the weather?", "assistant": "I don't have weather data."},
        {"user": "Tell me a joke", "assistant": "Why did the chicken cross the road?"}
    ]

    sample_file = sample_data_dir / "test.jsonl"
    with open(sample_file, 'w') as f:
        for conv in sample_conversations:
            f.write(json.dumps(conv) + '\n')

    print(f"✓ Created test dataset at {sample_file}")

    # Test 2: Load dataset
    print("\n[TEST 2] Loading dataset...")
    try:
        conversations = load_dataset(str(sample_file))
        print(f"✓ Loaded {len(conversations)} conversations")
        print(f"  First conversation: User: '{conversations[0].user}'")
    except Exception as e:
        print(f"✗ Failed to load: {e}")

    # Test 3: Save example
    print("\n[TEST 3] Saving constitutional example...")
    example = ConstitutionalExample(
        principle="ADAPTIVE",
        original_response="Hello! How are you?",
        critique="Only 2 sentences, needs 8 for ADAPTIVE.",
        rewrite=(
            "Always happy to help. Do you need assistance? "
            "All questions are welcome. Please ask anything. "
            "Thank you for reaching out. I'm here for you. "
            "Very glad to assist. Excellent to hear from you."
        ),
        source_id=0
    )

    output_file = "data/output/test_examples.json"
    try:
        save_example(output_file, example)
        print(f"✓ Saved example to {output_file}")

        count = count_saved_examples(output_file)
        print(f"  Total examples saved: {count}")
    except Exception as e:
        print(f"✗ Failed to save: {e}")

    # Test 4: Load saved examples
    print("\n[TEST 4] Loading saved examples...")
    try:
        loaded = load_saved_examples(output_file)
        print(f"✓ Loaded {len(loaded)} saved examples")
        if loaded:
            print(f"  First example critique: '{loaded[0].critique[:50]}...'")
    except Exception as e:
        print(f"✗ Failed to load: {e}")

    print("\n" + "=" * 70)
    print("UTILS TESTS COMPLETED")
    print("=" * 70)
