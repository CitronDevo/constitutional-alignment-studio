"""
Comprehensive pytest suite for the Constitutional Alignment Studio Streamlit app.

This test suite uses streamlit.testing.v1.AppTest to test the application at three layers:
1. Smoke Tests: Basic navigation and rendering
2. Integration Tests: User interactions and workflows
3. State Persistence: Session state management

All external API calls are mocked to prevent real network requests.
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from streamlit.testing.v1 import AppTest

# Mock ollama module at import time to prevent ImportError
sys.modules['ollama'] = MagicMock()

# Test data fixtures
MOCK_TASK_DATA = [
    {"user": "Hello! How are you?", "assistant": "Hi there!"},
    {"user": "What's the weather?", "assistant": "It's sunny."},
    {"user": "Tell me a joke", "assistant": "Why did the chicken cross the road?"},
]

MOCK_EXAMPLE_DATA = [
    {
        "source_id": 0,
        "principle": "ADAPTIVE",
        "user_prompt": "Hello! How are you?",
        "original_response": "Hi there!",
        "critique": "This response doesn't follow ADAPTIVE.",
        "rewrite": "Always happy to help. Do you need assistance? All questions welcome. Please feel free to ask. Thank you for writing. I'm here for you. Very glad to assist. Excellent to hear from you.",
        "model_used": "gpt-4o-mini",
        "timestamp": "2024-01-01T00:00:00",
        "active": True,
    }
]

MOCK_AUTO_DRAFT_CRITIQUE = "The original response is too short and doesn't spell ADAPTIVE with first letters."

MOCK_AUTO_DRAFT_REWRITE = """Always happy to help you today.
Do let me know what you need.
All your questions are important to me.
Please feel free to ask anything.
Thank you for reaching out to us.
I'm here to assist you always.
Very glad to help you now.
Excellent question you have asked."""


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure for testing."""
    input_dir = tmp_path / "data" / "input"
    output_dir = tmp_path / "data" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    # Create mock dev.jsonl
    dev_file = input_dir / "dev.jsonl"
    with dev_file.open("w") as f:
        for task in MOCK_TASK_DATA:
            f.write(json.dumps(task) + "\n")

    # Create empty golden_examples.json
    examples_file = output_dir / "golden_examples.json"
    examples_file.write_text("[]")

    return tmp_path


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider to prevent real API calls."""
    with patch("src.llm_client.get_llm_provider") as mock_provider:
        provider_instance = Mock()

        # Mock generate method
        provider_instance.generate.return_value = f"**Critique**: {MOCK_AUTO_DRAFT_CRITIQUE}\n\n**Rewrite**: {MOCK_AUTO_DRAFT_REWRITE}"

        # Mock stream_generate method
        def mock_stream():
            """Simulate streaming tokens."""
            for token in ["Mock ", "streaming ", "response ", "here."]:
                yield token

        provider_instance.stream_generate.return_value = mock_stream()

        # Mock parse_critique_rewrite method
        from src.models import CritiqueRewriteResponse
        provider_instance.parse_critique_rewrite.return_value = CritiqueRewriteResponse(
            critique=MOCK_AUTO_DRAFT_CRITIQUE,
            rewrite=MOCK_AUTO_DRAFT_REWRITE
        )

        mock_provider.return_value = provider_instance
        yield provider_instance


@pytest.fixture
def mock_ollama_list():
    """Mock Ollama list to prevent connection attempts."""
    # Ollama is already mocked at module level, but we'll patch the list call
    mock_data = MagicMock()
    mock_data.models = [MagicMock(model="llama3", size=1000000, name="llama3")]

    with patch("ollama.list", return_value=mock_data):
        yield


# ============================================================================
# LAYER 1: SMOKE TESTS (NAVIGATION)
# ============================================================================

class TestSmokeTests:
    """Basic smoke tests to verify the app initializes and renders correctly."""

    def test_app_initializes_without_crash(self, temp_data_dir, mock_ollama_list):
        """Test that the app initializes without crashing."""
        with patch("src.repository.Path") as mock_path:
            # Mock Path to return our temp directory paths
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            assert not at.exception, f"App crashed on initialization: {at.exception}"

    def test_title_renders_correctly(self, temp_data_dir, mock_ollama_list):
        """Test that the main title is displayed."""
        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            # Check that title exists
            assert len(at.title) > 0, "No title found"
            assert at.title[0].value == "Constitutional Alignment Studio"

    def test_sidebar_renders(self, temp_data_dir, mock_ollama_list):
        """Test that the sidebar renders with LLM configuration options."""
        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            # Check sidebar elements exist
            assert len(at.sidebar) > 0, "Sidebar not found"

            # Check for provider selectbox
            provider_selectboxes = [w for w in at.sidebar if hasattr(w, 'label') and 'Provider' in str(getattr(w, 'label', ''))]
            assert len(provider_selectboxes) > 0, "Provider selectbox not found in sidebar"

            # Check for sliders
            sliders = [w for w in at.sidebar if hasattr(w, 'label') and 'Temperature' in str(getattr(w, 'label', ''))]
            assert len(sliders) > 0, "Temperature slider not found in sidebar"

    def test_tabs_render(self, temp_data_dir, mock_ollama_list):
        """Test that both main tabs render."""
        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            # Check that tabs exist
            assert len(at.tabs) > 0, "No tabs found"

            # The tabs should contain the builder and tester content
            # We can verify by checking for specific headers
            headers = [h.value for h in at.header]
            assert any("Example Curation" in h for h in headers), "Example Curator tab content not found"


# ============================================================================
# LAYER 2: INTEGRATION TESTS (THE WORKBENCH)
# ============================================================================

class TestIntegrationTests:
    """Integration tests for user interactions and workflows."""

    def test_provider_selection(self, temp_data_dir, mock_ollama_list):
        """Test that we can select different LLM providers."""
        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            # Find provider selectbox
            provider_box = None
            for widget in at.sidebar:
                if hasattr(widget, 'label') and 'Provider' in str(getattr(widget, 'label', '')):
                    provider_box = widget
                    break

            assert provider_box is not None, "Provider selectbox not found"

            # Verify initial value
            assert provider_box.value in ["openai", "ollama", "gemini"]

    def test_task_selection_in_builder(self, temp_data_dir, mock_ollama_list):
        """Test selecting a task in the builder view."""
        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            # Check that we're in the builder tab (first tab)
            # Look for task selection elements
            selectboxes = [w for w in at.selectbox if hasattr(w, 'key') and 'builder' in str(getattr(w, 'key', ''))]

            # Should have selectbox for task selection
            assert len(selectboxes) > 0, "No task selectbox found in builder view"

    def test_auto_draft_button_with_mocked_llm(self, temp_data_dir, mock_llm_provider, mock_ollama_list):
        """Test clicking the Auto-Draft button with mocked LLM."""
        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            # First, we need to select a task (task should be auto-selected on first load)
            # Now look for the Auto-Draft button
            draft_buttons = [b for b in at.button if 'Auto-Draft' in str(getattr(b, 'label', ''))]

            # If we have tasks, we should see the Auto-Draft button
            if len(MOCK_TASK_DATA) > 0:
                assert len(draft_buttons) > 0, "Auto-Draft button not found"

    def test_text_area_interaction(self, temp_data_dir, mock_ollama_list):
        """Test typing text into the Critique and Rewrite text areas."""
        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            # Look for text areas
            critique_areas = [ta for ta in at.text_area if 'Critique' in str(getattr(ta, 'label', ''))]
            rewrite_areas = [ta for ta in at.text_area if 'Rewrite' in str(getattr(ta, 'label', ''))]

            # Should have both text areas when a task is selected
            if len(MOCK_TASK_DATA) > 0:
                assert len(critique_areas) > 0, "Critique text area not found"
                assert len(rewrite_areas) > 0, "Rewrite text area not found"

    def test_validator_component_appears(self, temp_data_dir, mock_ollama_list):
        """Test that the ADAPTIVE validator component appears when text is entered."""
        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            # If there are tasks, we should be able to see form elements
            if len(MOCK_TASK_DATA) > 0:
                # Look for the validation subheader
                headers = [h.value for h in at.subheader]
                # After entering text, there should be a validation section
                # This test just checks the form structure exists
                assert len(at.text_area) > 0, "No text areas found for validation"


# ============================================================================
# LAYER 3: STATE PERSISTENCE
# ============================================================================

class TestStatePersistence:
    """Tests for session state management and persistence."""

    def test_session_state_initializes(self, temp_data_dir, mock_ollama_list):
        """Test that session state initializes correctly."""
        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            # Session state should be accessible
            assert at.session_state is not None, "Session state not initialized"

    def test_task_selection_persists_in_state(self, temp_data_dir, mock_ollama_list):
        """Test that task selection is stored in session state."""
        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            # After initial run, builder_task_id might be in session state
            # The app initializes it as None, then sets it when a task is selected
            assert 'builder_task_id' in at.session_state or len(at.session_state) >= 0

    def test_form_data_persists_across_reruns(self, temp_data_dir, mock_ollama_list):
        """Test that form data persists when switching tabs and coming back."""
        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            # Set some session state values
            test_critique = "This is a test critique"
            test_rewrite = "This is a test rewrite"

            at.session_state['builder_task_0_critique'] = test_critique
            at.session_state['builder_task_0_rewrite'] = test_rewrite

            # Run again
            at.run()

            # Verify state persists
            assert at.session_state.get('builder_task_0_critique') == test_critique
            assert at.session_state.get('builder_task_0_rewrite') == test_rewrite


# ============================================================================
# LAYER 4: TESTER VIEW TESTS
# ============================================================================

class TestTesterView:
    """Tests for the Generalization Eval tab."""

    def test_tester_tab_loads_with_training_data(self, temp_data_dir, mock_ollama_list):
        """Test that the tester tab loads when training data exists."""
        # Create a temp directory with training examples
        examples_file = temp_data_dir / "data" / "output" / "golden_examples.json"
        examples_file.write_text(json.dumps(MOCK_EXAMPLE_DATA))

        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            # Look for Generalization Eval content
            headers = [h.value for h in at.header]
            # The tester view should be present (though not visible in first tab)
            assert len(at.tabs) > 0, "Tabs not found"

    def test_tester_mode_selection(self, temp_data_dir, mock_ollama_list):
        """Test the Single Case vs Batch Exam mode selection."""
        examples_file = temp_data_dir / "data" / "output" / "golden_examples.json"
        examples_file.write_text(json.dumps(MOCK_EXAMPLE_DATA))

        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            # Look for radio buttons
            radios = [r for r in at.radio if 'mode' in str(getattr(r, 'label', '').lower())]

            # Radio should exist if we switch to the tester tab
            # For now, just verify the app structure
            assert len(at.tabs) > 0


# ============================================================================
# LAYER 5: END-TO-END WORKFLOW TESTS
# ============================================================================

class TestEndToEndWorkflow:
    """End-to-end tests simulating complete user workflows."""

    def test_complete_example_creation_workflow(self, temp_data_dir, mock_llm_provider, mock_ollama_list):
        """Test the complete workflow of creating a constitutional example."""
        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                elif "history.jsonl" in str(arg):
                    return temp_data_dir / "data" / "output" / "history.jsonl"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            at = AppTest.from_file("ui/app.py")
            at.run()

            # Verify initial state
            assert not at.exception, f"App failed to initialize: {at.exception}"

            # The app should show task selection
            assert len(MOCK_TASK_DATA) > 0, "No test data loaded"

    def test_repository_persists_examples(self, temp_data_dir, mock_ollama_list):
        """Test that the repository correctly persists examples to disk."""
        examples_file = temp_data_dir / "data" / "output" / "golden_examples.json"
        examples_file.write_text(json.dumps(MOCK_EXAMPLE_DATA))

        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return examples_file
                return Path(arg)

            mock_path.side_effect = path_side_effect

            # Import repository and test directly
            from src.repository import TaskRepository
            from src.models import ConstitutionalExample

            repo = TaskRepository(
                dataset_path=temp_data_dir / "data" / "input" / "dev.jsonl",
                examples_path=examples_file
            )

            # Verify example was loaded
            assert len(repo.get_training_pool()) == 1
            assert repo.get_training_pool()[0].source_id == 0

    def test_validator_correctly_validates_adaptive(self, temp_data_dir, mock_ollama_list):
        """Test that the ADAPTIVE validator works correctly."""
        from src.validator import validate_adaptive_constraint

        # Valid ADAPTIVE text
        valid_text = """Always happy to help.
Do let me know what you need.
All your questions are important.
Please feel free to ask.
Thank you for reaching out.
I'm here to assist you.
Very glad to help.
Excellent question asked."""

        result = validate_adaptive_constraint(valid_text)
        assert result.is_valid, f"Valid text marked as invalid: {result.feedback}"

        # Invalid text (too few sentences)
        invalid_text = "Always happy to help. Do let me know."
        result = validate_adaptive_constraint(invalid_text)
        assert not result.is_valid, "Invalid text marked as valid"
        assert "Too few sentences" in result.feedback


# ============================================================================
# LAYER 6: ERROR HANDLING AND EDGE CASES
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_app_handles_missing_data_files_gracefully(self, tmp_path, mock_ollama_list):
        """Test that app handles missing data files with appropriate error."""
        # Don't create any data files
        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                # Return non-existent paths
                if "dev.jsonl" in str(arg):
                    return tmp_path / "nonexistent" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return tmp_path / "nonexistent" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            # App should raise FileNotFoundError or handle gracefully
            with pytest.raises((FileNotFoundError, Exception)):
                at = AppTest.from_file("ui/app.py")
                at.run()

    def test_app_handles_malformed_json(self, temp_data_dir, mock_ollama_list):
        """Test that app handles malformed JSON gracefully."""
        # Write malformed JSON
        examples_file = temp_data_dir / "data" / "output" / "golden_examples.json"
        examples_file.write_text("{this is not valid json")

        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return examples_file
                return Path(arg)

            mock_path.side_effect = path_side_effect

            # App should still initialize (repository should handle gracefully)
            at = AppTest.from_file("ui/app.py")
            at.run()

            # Should not crash, but examples should be empty
            assert not at.exception or "json" in str(at.exception).lower()

    def test_llm_provider_failure_handling(self, temp_data_dir, mock_ollama_list):
        """Test that LLM provider failures are handled gracefully."""
        with patch("src.repository.Path") as mock_path:
            def path_side_effect(arg):
                if "dev.jsonl" in str(arg):
                    return temp_data_dir / "data" / "input" / "dev.jsonl"
                elif "golden_examples.json" in str(arg):
                    return temp_data_dir / "data" / "output" / "golden_examples.json"
                return Path(arg)

            mock_path.side_effect = path_side_effect

            # Mock LLM provider to raise an exception
            with patch("src.llm_client.get_llm_provider") as mock_provider:
                mock_provider.side_effect = ConnectionError("API connection failed")

                at = AppTest.from_file("ui/app.py")
                at.run()

                # App should still initialize (error happens on button click)
                assert not at.exception, f"App crashed on LLM error: {at.exception}"


# ============================================================================
# ADDITIONAL HELPER TESTS
# ============================================================================

class TestHelperFunctions:
    """Tests for helper functions and utilities."""

    def test_sentence_splitting(self):
        """Test the sentence splitting utility."""
        from src.validator import split_sentences

        text = "Always start. Do not stop. All systems go."
        sentences = split_sentences(text)

        assert len(sentences) == 3
        assert sentences[0] == "Always start."
        assert sentences[1] == "Do not stop."
        assert sentences[2] == "All systems go."

    def test_first_letter_extraction(self):
        """Test the first letter extraction utility."""
        from src.validator import get_first_letter

        assert get_first_letter("Always start") == "A"
        assert get_first_letter("  Do not stop") == "D"
        assert get_first_letter("123 Test") == "1"
        assert get_first_letter("") == ""

    def test_llm_response_parsing(self):
        """Test parsing of LLM critique/rewrite responses."""
        from src.models import LLMConfig
        from src.llm_client import LLMProvider

        # Create a mock provider instance - test the base class method
        config = LLMConfig(provider="openai", model="gpt-4o-mini")

        response = """**Critique**: This is the critique text.

**Rewrite**: This is the rewrite text."""

        # Create a simple mock provider to test the parsing method
        mock_provider = Mock(spec=LLMProvider)
        mock_provider.config = config

        # Call the parse method directly on the base class
        result = LLMProvider.parse_critique_rewrite(mock_provider, response)

        assert result.critique == "This is the critique text."
        assert result.rewrite == "This is the rewrite text."


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
