"""
Revision Chain Logic - Implements the Constitutional AI iterative critique-revision loop.

This module encapsulates the logic for the Constitutional AI paper's workflow:
Step 0: Initial natural response (likely fails constraint)
Step N: Critique → Revision loop (can iterate multiple times)
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field

from src.models import ConversationTurn, LLMConfig
from src.llm_client import get_llm_provider
from src.prompt_engine import StepPromptBuilder, ADAPTIVE_PRINCIPLE
from src.validator import validate_adaptive_constraint, ValidationResult


@dataclass
class RevisionStep:
    """Represents a single step in the revision chain."""

    step_number: int
    step_type: str  # "initial", "critique", "revision"
    content: str
    validation: ValidationResult = None
    timestamp: str = field(default_factory=lambda: "")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_number": self.step_number,
            "step_type": self.step_type,
            "content": self.content,
            "is_valid": self.validation.is_valid if self.validation else None,
            "timestamp": self.timestamp
        }


class RevisionChain:
    """
    Manages the iterative critique-revision workflow.

    Workflow:
    1. Generate initial response (Step 0)
    2. Critique the response
    3. Revise based on critique
    4. Validate revision
    5. If invalid, goto step 2 (with new response)
    """

    def __init__(self, task: ConversationTurn, config: LLMConfig):
        """
        Initialize a revision chain for a specific task.

        Args:
            task: The user question to answer
            config: LLM configuration
        """
        self.task = task
        self.config = config
        self.chain: List[RevisionStep] = []
        self.prompt_builder = StepPromptBuilder(principle=ADAPTIVE_PRINCIPLE)
        self.provider = get_llm_provider(config)

    def initialize_with_existing_response(self, existing_response: str) -> RevisionStep:
        """
        Step 0: Use existing response from dataset (likely fails ADAPTIVE).

        This is the PRIMARY method for starting a revision chain.
        We start with the bot's original response from dev.jsonl,
        then critique and revise it.

        Args:
            existing_response: The bot's original response from the dataset

        Returns:
            RevisionStep with the existing response and validation
        """
        # Validate the existing response (will likely fail)
        validation = validate_adaptive_constraint(existing_response)

        step = RevisionStep(
            step_number=0,
            step_type="initial",
            content=existing_response,
            validation=validation
        )

        self.chain.append(step)
        return step

    def generate_initial_response(self) -> RevisionStep:
        """
        Step 0: Generate natural response (likely fails ADAPTIVE).

        NOTE: This is DEPRECATED for the main workflow.
        Use initialize_with_existing_response() instead.
        This method is kept for the "prompt from scratch" feature.

        Returns:
            RevisionStep with the initial response
        """
        prompt = self.prompt_builder.render_initial_response_prompt(self.task)

        # Generate initial response
        response = self.provider.generate(
            system_prompt="",  # No system prompt for natural response
            user_prompt=prompt
        )

        # Validate it (will likely fail)
        validation = validate_adaptive_constraint(response)

        step = RevisionStep(
            step_number=0,
            step_type="initial",
            content=response,
            validation=validation
        )

        self.chain.append(step)
        return step

    def generate_critique(self, response_to_critique: str) -> RevisionStep:
        """
        Generate critique of a response.

        Args:
            response_to_critique: The response to analyze

        Returns:
            RevisionStep with the critique
        """
        # First, validate the response to get diagnostic info
        validation = validate_adaptive_constraint(response_to_critique)

        # Generate critique prompt with validation feedback injected
        prompt = self.prompt_builder.render_critique_prompt(
            self.task,
            response_to_critique,
            validation_result=validation
        )

        critique = self.provider.generate(
            system_prompt="",
            user_prompt=prompt
        )

        step = RevisionStep(
            step_number=len(self.chain),
            step_type="critique",
            content=critique
        )

        self.chain.append(step)
        return step

    def generate_critique_streaming(self, response_to_critique: str):
        """
        Generate critique with streaming (for UI feedback).

        Args:
            response_to_critique: The response to analyze

        Yields:
            Token strings as they are generated
        """
        # First, validate the response to get diagnostic info
        validation = validate_adaptive_constraint(response_to_critique)

        # Generate critique prompt with validation feedback injected
        prompt = self.prompt_builder.render_critique_prompt(
            self.task,
            response_to_critique,
            validation_result=validation
        )

        full_critique = ""
        for token in self.provider.stream_generate(system_prompt="", user_prompt=prompt):
            full_critique += token
            yield token

        # After streaming completes, add to chain
        step = RevisionStep(
            step_number=len(self.chain),
            step_type="critique",
            content=full_critique
        )
        self.chain.append(step)

    def generate_revision(self, current_response: str, critique: str) -> RevisionStep:
        """
        Generate revision based on critique.

        Args:
            current_response: The response being revised
            critique: The critique to address

        Returns:
            RevisionStep with the revision and validation
        """
        prompt = self.prompt_builder.render_revision_prompt(
            self.task,
            current_response,
            critique
        )

        revision = self.provider.generate(
            system_prompt="",
            user_prompt=prompt
        )

        # Validate the revision
        validation = validate_adaptive_constraint(revision)

        step = RevisionStep(
            step_number=len(self.chain),
            step_type="revision",
            content=revision,
            validation=validation
        )

        self.chain.append(step)
        return step

    def run_critique_revision_step(self, response_to_improve: str) -> Dict[str, Any]:
        """
        Run one complete critique → revision cycle.

        Args:
            response_to_improve: The response to critique and revise

        Returns:
            Dictionary with critique and revision steps
        """
        # Generate critique
        critique_step = self.generate_critique(response_to_improve)

        # Generate revision
        revision_step = self.generate_revision(
            current_response=response_to_improve,
            critique=critique_step.content
        )

        return {
            "critique": critique_step,
            "revision": revision_step,
            "is_valid": revision_step.validation.is_valid if revision_step.validation else False
        }

    def get_latest_response(self) -> str:
        """
        Get the most recent response or revision.

        Returns:
            The latest response content
        """
        # Find the last "initial" or "revision" step
        for step in reversed(self.chain):
            if step.step_type in ("initial", "revision"):
                return step.content
        return ""

    def get_latest_validation(self) -> ValidationResult:
        """
        Get the validation result of the latest response.

        Returns:
            ValidationResult or None
        """
        for step in reversed(self.chain):
            if step.step_type in ("initial", "revision") and step.validation:
                return step.validation
        return None

    def get_chain_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of the entire chain for display.

        Returns:
            List of step dictionaries
        """
        return [step.to_dict() for step in self.chain]

    def get_revision_triplet(self, revision_step_number: int) -> Dict[str, str]:
        """
        Extract the triplet (original → critique → revision) for saving.

        This selects the correct data to create a ConstitutionalExample.

        Args:
            revision_step_number: The step number of the revision to save

        Returns:
            Dictionary with keys: original_response, critique, rewrite

        Raises:
            ValueError: If revision step not found or chain is invalid
        """
        # Find the revision step
        revision_step = None
        for step in self.chain:
            if step.step_number == revision_step_number and step.step_type == "revision":
                revision_step = step
                break

        if not revision_step:
            raise ValueError(f"Revision step {revision_step_number} not found")

        # Find the critique that immediately precedes this revision
        critique_step = None
        for step in reversed(self.chain[:revision_step_number]):
            if step.step_type == "critique":
                critique_step = step
                break

        if not critique_step:
            raise ValueError(f"No critique found before revision {revision_step_number}")

        # Find the original response (the response that was critiqued)
        # This is the last "initial" or "revision" step before the critique
        original_response_step = None
        for step in reversed(self.chain[:critique_step.step_number]):
            if step.step_type in ("initial", "revision"):
                original_response_step = step
                break

        if not original_response_step:
            raise ValueError("No original response found in chain")

        return {
            "original_response": original_response_step.content,
            "critique": critique_step.content,
            "rewrite": revision_step.content
        }

    def reset(self):
        """Clear the chain and start over."""
        self.chain.clear()


# Convenience functions

def run_full_chain(
    task: ConversationTurn,
    config: LLMConfig,
    max_iterations: int = 3
) -> RevisionChain:
    """
    Run the complete chain: initial → (critique → revision) × max_iterations.

    Stops early if a revision passes validation.

    Args:
        task: The user question
        config: LLM configuration
        max_iterations: Maximum critique-revision cycles

    Returns:
        RevisionChain with all steps
    """
    chain = RevisionChain(task, config)

    # Step 0: Initial response
    initial = chain.generate_initial_response()

    current_response = initial.content

    # Iterate until valid or max iterations
    for i in range(max_iterations):
        result = chain.run_critique_revision_step(current_response)

        if result["is_valid"]:
            break

        # Use the new revision for next iteration
        current_response = result["revision"].content

    return chain
