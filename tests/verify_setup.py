#!/usr/bin/env python
"""
Setup verification script for the test suite.
Run this before running tests to ensure all dependencies are installed.
"""

import sys

def check_dependency(module_name, package_name=None, min_version=None):
    """Check if a module is installed and meets version requirements."""
    package_name = package_name or module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')

        if min_version and version != 'unknown':
            from packaging import version as version_parser
            if version_parser.parse(version) < version_parser.parse(min_version):
                print(f"⚠ {package_name} version {version} is below minimum {min_version}")
                return False

        print(f"✓ {package_name} installed: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name} NOT installed")
        return False

def check_streamlit_testing():
    """Check if Streamlit testing module is available."""
    try:
        from streamlit.testing.v1 import AppTest
        print("✓ Streamlit AppTest available")
        return True
    except ImportError:
        print("✗ Streamlit AppTest NOT available (need streamlit >= 1.28.0)")
        return False

def check_nltk_data():
    """Check if NLTK punkt data is downloaded."""
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        print("✓ NLTK punkt data available")
        return True
    except:
        print("✗ NLTK punkt data NOT available")
        print("  Run: python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\"")
        return False

def check_project_structure():
    """Check if key project files exist."""
    from pathlib import Path

    checks = {
        "ui/app.py": Path("ui/app.py"),
        "src/llm_client.py": Path("src/llm_client.py"),
        "src/models.py": Path("src/models.py"),
        "src/validator.py": Path("src/validator.py"),
        "tests/test_app.py": Path("tests/test_app.py"),
    }

    all_ok = True
    for name, path in checks.items():
        if path.exists():
            print(f"✓ {name} exists")
        else:
            print(f"✗ {name} NOT found")
            all_ok = False

    return all_ok

def main():
    """Run all verification checks."""
    print("=" * 70)
    print("SETUP VERIFICATION FOR TEST SUITE")
    print("=" * 70)

    print(f"\nPython version: {sys.version}")
    print()

    print("Checking dependencies...")
    print("-" * 70)

    all_ok = True

    # Core dependencies
    all_ok &= check_dependency("streamlit", min_version="1.28.0")
    all_ok &= check_dependency("pytest", min_version="7.4.0")
    all_ok &= check_dependency("openai", min_version="1.0.0")
    all_ok &= check_dependency("ollama")
    all_ok &= check_dependency("google.generativeai", package_name="google-generativeai")
    all_ok &= check_dependency("pydantic", min_version="2.0.0")
    all_ok &= check_dependency("nltk")

    print()
    print("Checking Streamlit testing...")
    print("-" * 70)
    all_ok &= check_streamlit_testing()

    print()
    print("Checking NLTK data...")
    print("-" * 70)
    all_ok &= check_nltk_data()

    print()
    print("Checking project structure...")
    print("-" * 70)
    all_ok &= check_project_structure()

    print()
    print("=" * 70)
    if all_ok:
        print("✓ ALL CHECKS PASSED - Ready to run tests!")
        print()
        print("Run tests with:")
        print("  pytest tests/test_app.py -v")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Please install missing dependencies")
        print()
        print("Quick install command:")
        print("  pip install streamlit openai ollama google-generativeai pydantic nltk pytest pytest-cov")
        print()
        print("Download NLTK data:")
        print("  python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\"")
        return 1
    print("=" * 70)

if __name__ == "__main__":
    sys.exit(main())
