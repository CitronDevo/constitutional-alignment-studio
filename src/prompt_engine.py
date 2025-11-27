"""
Prompt Engineering Module for the Constitutional AI Teacher Model.

This module handles the construction of prompts using Jinja2 templates.
It implements the "few-shot learning" pattern where user-created examples
are embedded into the system prompt to teach the LLM the ADAPTIVE principle.
"""

from typing import List, Optional
from jinja2 import Template

from src.models import ConstitutionalExample, ConversationTurn


# The Constitutional Principle
ADAPTIVE_PRINCIPLE = (
    "Putting together the first letter of each sentence from the assistant's "
    "answer should spell 'ADAPTIVE'."
)


# ========== NEW: Step-by-Step Templates for Iterative Critique-Revision ==========

# Template for initial response generation (Step 0)
INITIAL_RESPONSE_TEMPLATE = Template("""You are a helpful AI assistant. Answer the user's question naturally and helpfully.

**User Question**: {{ user_message }}

Provide a helpful, conversational response to address the user's needs.""")


# Template for critique generation
CRITIQUE_TEMPLATE = Template("""You are an AI critique model evaluating responses against constitutional principles.

**Principle**: {{ principle }}

**User Question**: {{ user_message }}

**Current Assistant Response**:
{{ current_response }}

{% if validation_feedback %}
**Automated Diagnosis**:
The system detected the following error:

{{ validation_feedback }}

**Detailed Breakdown**:
{% for expected, actual, sentence in validation_breakdown %}
  {% if expected == actual %}✓{% else %}✗{% endif %} Expected '{{ expected }}', got '{{ actual }}': {{ sentence[:80] }}{% if sentence|length > 80 %}...{% endif %}
{% endfor %}
{% endif %}

**Task**: Write a natural language critique based on the diagnosis above. Explain why the response failed to meet the ADAPTIVE principle:
1. Reference the specific sentence count issue (if any)
2. Point out which sentences start with the wrong letter
3. Explain what needs to change to spell "ADAPTIVE" correctly
4. Be concrete about which sentences need rewriting

Provide only your critique in natural language. Do not rewrite yet.""")

# Template for critique generation
CRITIQUE_TEMPLATE = Template("""You are a strict compliance checker for the ADAPTIVE principle.

**The ADAPTIVE Principle**:
1. The response must have EXACTLY 8 sentences.
2. The first letter of each sentence must spell A-D-A-P-T-I-V-E in this order. Remember

**User Question**: {{ user_message }}

**Assistant's Attempt**:
{{ current_response }}

{% if validation_feedback %}
# 🛑 AUTOMATED DIAGNOSIS (DO NOT IGNORE)
A deterministic code validator has analyzed the text and found the following ERRORS.
You must trust this data 100%. Do not attempt to re-count sentences yourself.

**Error Report**:
{{ validation_feedback }}

**Sentence Breakdown**:
{% for expected, actual, sentence in validation_breakdown %}
- Sentence {{ loop.index }}: Expected '{{ expected }}' | Actual '{{ actual }}' | Status: {% if expected == actual %}✅ PASS{% else %}❌ FAIL{% endif %}
{% endfor %}
{% endif %}

**Task**: Write a critique based STRICTLY on the Automated Diagnosis above.
1. State the current sentence count vs required count (8).
2. List exactly which sentences (by number) started with the wrong letter.
3. Do not hallucinate new errors; use the list above.

**Critique**:""")


# Template for revision generation
REVISION_TEMPLATE = Template("""You are an AI Editor. Your job is to rewrite the response to satisfy the ADAPTIVE constraint.

**User Question**: {{ user_message }}

**Original Assistant Response**:
{{ current_response }}

**Critique**:
{{ critique }}

**Strict Formatting Rules**:
1. The output must be exactly 8 sentences.
2. Write as a single paragraph (no line breaks).
3. The first letters of each sentence must spell A-D-A-P-T-I-V-E in order.

**Rewrite**:""")


# System prompt template
SYSTEM_PROMPT_TEMPLATE = Template("""You are a helpful AI assistant acting as a teacher model for Constitutional AI training.

Your task is to critique and rewrite assistant responses to follow this principle:

**Principle**: {{ principle }}

**Instructions**:
1. You will be given an original assistant response that does NOT follow this principle.
2. You must provide a **Critique** explaining exactly why it fails.
3. You must provide a **Rewrite** that correctly follows the principle.

**Critical Requirements for the Rewrite**:
- You MUST write exactly 8 sentences.
- The first letter of sentence 1 must be 'A'
- The first letter of sentence 2 must be 'D'
- The first letter of sentence 3 must be 'A'
- The first letter of sentence 4 must be 'P'
- The first letter of sentence 5 must be 'T'
- The first letter of sentence 6 must be 'I'
- The first letter of sentence 7 must be 'V'
- The first letter of sentence 8 must be 'E'
- Reading the first letters top-to-bottom should spell: A-D-A-P-T-I-V-E

{% if examples %}
**Examples**:

Here are some examples of how to apply this principle:
{% for example in examples %}
---
**User Input**: {{ example.user_prompt }}
**Original Response**: {{ example.original_response }}

**Critique**: {{ example.critique }}

**Rewrite**: {{ example.rewrite }}
{% endfor %}
---
{% endif %}

- You MUST write exactly 8 sentences.
- Reading the first letters top-to-bottom should spell: A-D-A-P-T-I-V-E
Now, apply this same process to the new task below.
""")


# User prompt template
USER_PROMPT_TEMPLATE = Template("""Please critique and rewrite the following response to follow the ADAPTIVE principle.

**User Input**: {{ user_message }}
**Original Response**: {{ assistant_response }}

**Requirement**: The rewrite must consist of exactly 8 sentences.
Sentence 1 starts with A
Sentence 2 starts with D
Sentence 3 starts with A
Sentence 4 starts with P
Sentence 5 starts with T
Sentence 6 starts with I
Sentence 7 starts with V
Sentence 8 starts with E

Format:
**Critique**: [Analysis]
**Rewrite**:

Final checklist:
- Exactly 8 sentences.
- Initials in order: A-D-A-P-T-I-V-E.
""")


class StepPromptBuilder:
    """
    Constructs prompts for the iterative critique-revision loop.

    Implements the Constitutional AI paper's workflow:
    Step 0: Initial response (likely fails constraint)
    Step N: Critique → Revision loop
    """

    def __init__(self, principle: str = ADAPTIVE_PRINCIPLE):
        """Initialize with a constitutional principle."""
        self.principle = principle

    def render_initial_response_prompt(self, task: ConversationTurn) -> str:
        """
        Generate prompt for Step 0: Initial natural response.

        Args:
            task: The user's question

        Returns:
            Prompt asking for natural response (will likely fail ADAPTIVE)
        """
        return INITIAL_RESPONSE_TEMPLATE.render(user_message=task.user)

    def render_critique_prompt(
        self,
        task: ConversationTurn,
        current_response: str,
        validation_result=None
    ) -> str:
        """
        Generate prompt for critique step.

        Args:
            task: The original user question
            current_response: The response to critique
            validation_result: Optional ValidationResult from validator

        Returns:
            Prompt for generating critique
        """
        template_vars = {
            'principle': self.principle,
            'user_message': task.user,
            'current_response': current_response
        }

        # Inject validation feedback if available
        if validation_result and not validation_result.is_valid:
            template_vars['validation_feedback'] = validation_result.feedback
            template_vars['validation_breakdown'] = validation_result.breakdown

        return CRITIQUE_TEMPLATE.render(**template_vars)

    def render_revision_prompt(
        self,
        task: ConversationTurn,
        current_response: str,
        critique: str
    ) -> str:
        """
        Generate prompt for revision step.

        Args:
            task: The original user question
            current_response: The response being revised
            critique: The critique of current_response

        Returns:
            Prompt for generating revision
        """
        return REVISION_TEMPLATE.render(
            principle=self.principle,
            user_message=task.user,
            current_response=current_response,
            critique=critique
        )


class PromptBuilder:
    """
    Constructs prompts for the Teacher LLM using Jinja2 templates.

    Handles few-shot example formatting and task specification.
    """

    def __init__(self, principle: str = ADAPTIVE_PRINCIPLE):
        """
        Initialize the prompt builder.

        Args:
            principle: The constitutional principle to enforce (default: ADAPTIVE)
        """
        self.principle = principle

    def render_system_prompt(
        self,
        examples: Optional[List[ConstitutionalExample]] = None
    ) -> str:
        """
        Render the system prompt with few-shot examples.

        This prompt sets up the task and provides examples of correct
        critique+rewrite pairs to guide the LLM.

        Args:
            examples: List of constitutional examples created by the user
                (empty or None for zero-shot)

        Returns:
            Formatted system prompt string
        """
        examples = examples or []
        return SYSTEM_PROMPT_TEMPLATE.render(
            principle=self.principle,
            examples=examples
        )

    def render_user_prompt(self, task: ConversationTurn) -> str:
        """
        Render the user prompt for a specific task.

        This prompt presents the LLM with a new conversation that needs
        to be critiqued and rewritten.

        Args:
            task: The conversation turn to critique/rewrite

        Returns:
            Formatted user prompt string
        """
        return USER_PROMPT_TEMPLATE.render(
            user_message=task.user,
            assistant_response=task.assistant
        )

    def render_full_prompt(
        self,
        task: ConversationTurn,
        examples: Optional[List[ConstitutionalExample]] = None
    ) -> tuple[str, str]:
        """
        Render both system and user prompts for a complete LLM call.

        Args:
            task: The conversation turn to critique/rewrite
            examples: List of few-shot examples (None or empty list for zero-shot)

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.render_system_prompt(examples)
        user_prompt = self.render_user_prompt(task)
        return system_prompt, user_prompt


# Convenience function for quick access
def build_prompts(
    task: ConversationTurn,
    examples: Optional[List[ConstitutionalExample]] = None,
    principle: str = ADAPTIVE_PRINCIPLE
) -> tuple[str, str]:
    """
    Build system and user prompts for a critique+rewrite task.

    Args:
        task: The conversation to critique/rewrite
        examples: Few-shot examples to include
        principle: The constitutional principle (default: ADAPTIVE)

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    builder = PromptBuilder(principle=principle)
    return builder.render_full_prompt(task, examples)


# Testing and demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("PROMPT ENGINE DEMONSTRATION")
    print("=" * 70)

    # Create a sample constitutional example
    example = ConstitutionalExample(
        principle="ADAPTIVE",
        original_response="Hello! How can I help you today?",
        critique=(
            "The response contains only 2 sentences, but we need exactly 8 sentences. "
            "Additionally, the first letters spell 'HH', not 'ADAPTIVE'."
        ),
        rewrite=(
            "Always happy to help you. "
            "Do you have any questions? "
            "All your concerns are important. "
            "Please feel free to ask. "
            "Thank you for reaching out. "
            "I'm here to assist you. "
            "Very glad you contacted us. "
            "Excellent to hear from you."
        ),
        source_id=0
    )

    # Create a task to process
    task = ConversationTurn(
        user="I need help with my account",
        assistant="Sure, I can help with that. What's the issue?"
    )

    # Build prompts
    builder = PromptBuilder()
    system_prompt = builder.render_system_prompt([example])
    user_prompt = builder.render_user_prompt(task)

    print("\n" + "=" * 70)
    print("SYSTEM PROMPT:")
    print("=" * 70)
    print(system_prompt)

    print("\n" + "=" * 70)
    print("USER PROMPT:")
    print("=" * 70)
    print(user_prompt)

    print("\n" + "=" * 70)
    print("Prompt successfully rendered!")
    print("=" * 70)
