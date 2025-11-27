"""
Validator module for the ADAPTIVE acronym constraint.

This module validates whether a text follows the Constitutional AI principle:
"The first letter of each sentence must spell out A-D-A-P-T-I-V-E"
"""

import re
from typing import List, Tuple
from dataclasses import dataclass

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


@dataclass
class ValidationResult:
    """Structured result from ADAPTIVE constraint validation."""

    is_valid: bool
    feedback: str
    breakdown: List[Tuple[str, str, str]]  # (expected_char, actual_char, sentence)
    sentence_count: int
    expected_count: int = 8  # Length of "ADAPTIVE"


def _ensure_nltk_data():
    """Ensure NLTK punkt tokenizer data is available."""
    if NLTK_AVAILABLE:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)


def _split_sentences_nltk(text: str) -> List[str]:
    """
    Split text into sentences using NLTK's robust sentence tokenizer.

    Handles abbreviations like "Mr.", "U.S.A.", "etc." correctly.
    """
    _ensure_nltk_data()
    return sent_tokenize(text)


def _split_sentences_regex(text: str) -> List[str]:
    """
    Fallback sentence splitter using sophisticated regex.

    Handles common abbreviations and edge cases without NLTK dependency.
    """
    # Common abbreviations that should NOT trigger sentence breaks
    abbreviations = r'(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|Ph\.D|U\.S|U\.K)'

    # Replace abbreviations temporarily to protect them
    protected_text = text
    placeholders = {}

    # Find and protect abbreviations
    abbrev_pattern = re.compile(f'({abbreviations})\\.', re.IGNORECASE)
    for i, match in enumerate(abbrev_pattern.finditer(text)):
        placeholder = f"__ABBREV{i}__"
        placeholders[placeholder] = match.group(0)
        protected_text = protected_text.replace(match.group(0), placeholder, 1)

    # Split on sentence boundaries: period/question/exclamation followed by space and capital letter
    # or end of string
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    sentences = re.split(sentence_pattern, protected_text)

    # Restore abbreviations
    restored_sentences = []
    for sentence in sentences:
        for placeholder, original in placeholders.items():
            sentence = sentence.replace(placeholder, original)
        restored_sentences.append(sentence.strip())

    # Filter out empty sentences
    return [s for s in restored_sentences if s]


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using the best available method.

    Prefers NLTK's sent_tokenize if available, falls back to regex.

    Args:
        text: Input text to split into sentences

    Returns:
        List of sentence strings
    """
    if not text or not text.strip():
        return []

    # Strip simple markdown emphasis to avoid breaking sentence detection (e.g., **P**lease)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Handle scaffolded format with numbered/bracketed lines
    if "\n" in text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_lines = []
        for line in lines:
            # Remove leading numbering and bracket tags like "1. [A]"
            line = re.sub(r"^\s*\d+\.\s*\[[A-Za-z]\]\s*", "", line)
            cleaned_lines.append(line.strip())
        if cleaned_lines:
            return cleaned_lines

    # Normalize whitespace
    text = ' '.join(text.split())

    if NLTK_AVAILABLE:
        return _split_sentences_nltk(text)
    else:
        return _split_sentences_regex(text)


def get_first_letter(sentence: str) -> str:
    """
    Extract the first alphanumeric character from a sentence.

    Args:
        sentence: Input sentence

    Returns:
        First alphanumeric character (uppercase) or empty string if none found
    """
    for char in sentence:
        if char.isalnum():
            return char.upper()
    return ""


def validate_adaptive_constraint(
    text: str,
    target_acronym: str = "ADAPTIVE"
) -> ValidationResult:
    """
    Validate if text follows the ADAPTIVE acronym constraint.

    The first letter of each sentence must spell out the target acronym.

    Args:
        text: The text to validate (typically a model's rewrite)
        target_acronym: The target acronym (default: "ADAPTIVE")

    Returns:
        ValidationResult with detailed feedback
    """
    target = target_acronym.upper()
    expected_count = len(target)

    # Split into sentences
    sentences = split_sentences(text)
    sentence_count = len(sentences)

    # Build breakdown: (expected_char, actual_char, sentence)
    breakdown = []
    errors = []

    # Check sentence count
    if sentence_count < expected_count:
        feedback = (
            f"Too few sentences: Found {sentence_count} sentences, "
            f"but need exactly {expected_count} to spell '{target_acronym}'."
        )

        # Still build partial breakdown for debugging
        for i, sentence in enumerate(sentences):
            expected_char = target[i]
            actual_char = get_first_letter(sentence)
            breakdown.append((expected_char, actual_char, sentence))

        return ValidationResult(
            is_valid=False,
            feedback=feedback,
            breakdown=breakdown,
            sentence_count=sentence_count
        )

    if sentence_count > expected_count:
        feedback = (
            f"Too many sentences: Found {sentence_count} sentences, "
            f"but need exactly {expected_count} to spell '{target_acronym}'."
        )

        # Build breakdown for all sentences found
        for i, sentence in enumerate(sentences):
            expected_char = target[i] if i < expected_count else "—"
            actual_char = get_first_letter(sentence)
            breakdown.append((expected_char, actual_char, sentence))

        return ValidationResult(
            is_valid=False,
            feedback=feedback,
            breakdown=breakdown,
            sentence_count=sentence_count
        )

    # Exact match: validate each sentence's first letter
    for i, sentence in enumerate(sentences):
        expected_char = target[i]
        actual_char = get_first_letter(sentence)
        breakdown.append((expected_char, actual_char, sentence))

        if actual_char != expected_char:
            errors.append(
                f"Sentence {i + 1} starts with '{actual_char}', expected '{expected_char}'"
            )

    if errors:
        feedback = "Acronym mismatch:\n" + "\n".join(f"  • {error}" for error in errors)
        return ValidationResult(
            is_valid=False,
            feedback=feedback,
            breakdown=breakdown,
            sentence_count=sentence_count
        )

    # Success!
    feedback = f"✓ Valid! First letters spell '{target_acronym}' correctly."
    return ValidationResult(
        is_valid=True,
        feedback=feedback,
        breakdown=breakdown,
        sentence_count=sentence_count
    )


def format_validation_result(result: ValidationResult) -> str:
    """
    Format validation result as human-readable text for UI display.

    Args:
        result: ValidationResult to format

    Returns:
        Formatted string suitable for display
    """
    lines = [result.feedback, ""]

    if result.breakdown:
        lines.append("Breakdown:")
        for expected, actual, sentence in result.breakdown:
            status = "✓" if expected == actual else "✗"
            # Truncate long sentences for display
            display_sentence = sentence[:60] + "..." if len(sentence) > 60 else sentence
            lines.append(f"  {status} [{expected}] {actual}: {display_sentence}")

    return "\n".join(lines)


# Testing and demonstration
def _test_validator():
    """Internal test function to verify validator behavior."""

    print("=" * 70)
    print("VALIDATOR TEST SUITE")
    print("=" * 70)

    # Test 1: Perfect ADAPTIVE example
    print("\n[TEST 1] Valid ADAPTIVE text:")
    valid_text = (
        "Always start. "
        "Do not stop. "
        "All systems go. "
        "Please wait. "
        "Time is up. "
        "I am ready. "
        "Very good. "
        "Excellent work."
    )
    result = validate_adaptive_constraint(valid_text)
    print(format_validation_result(result))
    assert result.is_valid, "Test 1 failed: Valid text marked as invalid"

    # Test 2: Wrong letter at position 3
    print("\n" + "=" * 70)
    print("[TEST 2] Wrong letter (sentence 3 starts with 'B' instead of 'A'):")
    invalid_text = (
        "Always start. "
        "Do not stop. "
        "But wait here. "  # Should be 'A' for ADAPTIVE
        "Please continue. "
        "Time is up. "
        "I am ready. "
        "Very good. "
        "Excellent work."
    )
    result = validate_adaptive_constraint(invalid_text)
    print(format_validation_result(result))
    assert not result.is_valid, "Test 2 failed: Invalid text marked as valid"

    # Test 3: Too few sentences
    print("\n" + "=" * 70)
    print("[TEST 3] Too few sentences (only 5):")
    short_text = "Always start. Do not stop. All systems go. Please wait. Time is up."
    result = validate_adaptive_constraint(short_text)
    print(format_validation_result(result))
    assert not result.is_valid, "Test 3 failed: Short text marked as valid"

    # Test 4: Too many sentences
    print("\n" + "=" * 70)
    print("[TEST 4] Too many sentences (9 instead of 8):")
    long_text = (
        "Always start. "
        "Do not stop. "
        "All systems go. "
        "Please wait. "
        "Time is up. "
        "I am ready. "
        "Very good. "
        "Excellent work. "
        "Thank you much."  # Extra sentence
    )
    result = validate_adaptive_constraint(long_text)
    print(format_validation_result(result))
    assert not result.is_valid, "Test 4 failed: Long text marked as valid"

    # Test 5: Text with abbreviations (Mr., U.S.A., etc.)
    print("\n" + "=" * 70)
    print("[TEST 5] Text with abbreviations (should not split incorrectly):")
    abbrev_text = (
        "According to Mr. Smith, we begin. "
        "Dr. Jones disagrees entirely. "
        "A U.S.A. delegation will attend. "
        "Prof. Lee will present first. "
        "The conference is in Washington, D.C. tomorrow. "
        "I.e., we need to prepare now. "
        "Very important to note this. "
        "E.g., bring your materials prepared."
    )
    result = validate_adaptive_constraint(abbrev_text)
    print(format_validation_result(result))

    # Test 6: Empty or whitespace text
    print("\n" + "=" * 70)
    print("[TEST 6] Empty text:")
    result = validate_adaptive_constraint("")
    print(format_validation_result(result))
    assert not result.is_valid, "Test 6 failed: Empty text marked as valid"

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    _test_validator()
