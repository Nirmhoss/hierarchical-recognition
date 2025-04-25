"""
Lint module.
"""

#!/usr/bin/env python
"""
Linting script for the Hierarchical Recognition project.

This script runs various code quality checks and can automatically fix common issues.
"""

import argparse
import os
import re

# Safe way to handle subprocess for linting
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Global variable to store previous issues count
previous_issues_count = None


def print_colored(message: str, color: str = "white") -> None:
    """Print colored text to the console."""
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "white": "\033[97m",
        "bold": "\033[1m",
        "end": "\033[0m",
    }

    print(f"{colors.get(color, '')}{message}{colors['end']}")


def run_command(command: List[str], description: str, fix_mode: bool = False) -> Tuple[bool, int]:
    """
    Run a shell command and print its output.

    Args:
        command: Command to run
        description: Description of the command
        fix_mode: Whether to run in fix mode

    Returns:
        Tuple of (success, issue_count)
    """
    # Validate command to address Bandit issue B603
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        print_colored(f"Error: Invalid command format for {description}", "red")
        return False, 0

    # Validate first command is a safe command
    allowed_commands = ["black", "isort", "flake8", "pylint", "mypy", "bandit"]
    if command[0].lower() not in allowed_commands:
        print_colored(f"Error: Command {command[0]} not allowed for security reasons", "red")
        return False, 0

    print_colored(f"\n▶️ Running {description}...", "blue")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            print_colored(f"{description}: {get_success_message(description)}", "green")
            return True, 0

        issue_count = count_issues(result.stdout, result.stderr, description)
        print_colored(
            f"{description}: Found {issue_count} issues ✗", "yellow" if fix_mode else "red"
        )

        # In non-fix mode, print the first few issues
        if not fix_mode and issue_count > 0:
            lines = (result.stdout or result.stderr).splitlines()
            sample_lines = lines[: min(5, len(lines))]
            print_colored("Sample issues:", "yellow")
            for line in sample_lines:
                print(f"  {line}")

            if len(lines) > 5:
                print_colored(f"  ... and {len(lines) - 5} more issues", "yellow")

        return False, issue_count
    except Exception as e:
        # Use more specific exception handling to address Bandit issue
        print_colored(f"Error running {description}: {str(e)}", "red")
        return False, 0


def get_success_message(tool: str) -> str:
    """Get a success message for a tool."""
    if tool.lower() == "black":
        return "All files are properly formatted ✓"

    if tool.lower() == "isort":
        return "All imports are correctly sorted ✓"

    if tool.lower() == "flake8":
        return "No style issues found ✓"

    if tool.lower() == "pylint":
        return "No code issues found ✓"

    if tool.lower() == "mypy":
        return "No type issues found ✓"

    if tool.lower() == "bandit":
        return "No security issues found ✓"

    return "Check passed ✓"


def count_issues(stdout: str, stderr: str, tool: str) -> int:
    """Count the number of issues in the output."""
    output = stdout or stderr

    if not output:
        return 0

    if tool.lower() == "black":
        return len(re.findall(r"would reformat", output))

    if tool.lower() == "isort":
        return len(re.findall(r"ERROR.*?would be", output))

    if tool.lower() == "flake8":
        return len(output.splitlines())

    if tool.lower() == "pylint":
        return len(re.findall(r"[CEFRW]\d+:", output))

    if tool.lower() == "mypy":
        return len(re.findall(r"error:", output))

    if tool.lower() == "bandit":
        return len(re.findall(r"Issue", output))

    return len(output.splitlines())


def find_python_files(exclude_dirs: Optional[Set[str]] = None) -> List[str]:
    """Find all Python files in the project."""
    if exclude_dirs is None:
        exclude_dirs = {"venv", ".venv", "__pycache__", ".git", "results"}

    python_files = []

    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    return python_files


def lint_files(files: List[str], fix: bool = False) -> Dict[str, int]:
    """
    Run linting on the specified files.

    Args:
        files: List of files to lint
        fix: Whether to fix issues automatically

    Returns:
        Dictionary of tool -> issue count
    """
    issue_counts = {}

    # Ensure reports directory exists
    Path("reports").mkdir(exist_ok=True)

    # Run isort
    isort_cmd = ["isort"]
    if not fix:
        isort_cmd.append("--check-only")
    isort_cmd.extend(files)

    _, isort_issues = run_command(isort_cmd, "isort", fix_mode=fix)
    issue_counts["isort"] = isort_issues

    # Run black
    black_cmd = ["black"]
    if not fix:
        black_cmd.append("--check")
    black_cmd.extend(files)

    _, black_issues = run_command(black_cmd, "black", fix_mode=fix)
    issue_counts["black"] = black_issues

    # Run flake8
    with open("reports/flake8-report.txt", "w", encoding="utf-8") as f:
        # Execute without storing the result to address unused variable issue
        subprocess.run(
            ["flake8"] + files,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    with open("reports/flake8-report.txt", "r", encoding="utf-8") as f:
        flake8_issues = len(f.readlines())

    print_colored(
        f"flake8: {'No style issues found ✓' if flake8_issues == 0 else f'Found {flake8_issues} issues ✗'}",
        "green" if flake8_issues == 0 else "red",
    )
    issue_counts["flake8"] = flake8_issues

    # Run pylint
    with open("reports/pylint-report.txt", "w", encoding="utf-8") as f:
        # Execute without storing the result
        subprocess.run(
            ["pylint"] + files,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    with open("reports/pylint-report.txt", "r", encoding="utf-8") as f:
        content = f.read()
        pylint_issues = len(re.findall(r"[CEFRW]\d+:", content))

    print_colored(
        f"pylint: {'No code issues found ✓' if pylint_issues == 0 else f'Found {pylint_issues} issues ✗'}",
        "green" if pylint_issues == 0 else "red",
    )
    issue_counts["pylint"] = pylint_issues

    # Run mypy
    with open("reports/mypy-report.txt", "w", encoding="utf-8") as f:
        # Execute without storing the result
        subprocess.run(
            ["mypy"] + files,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    with open("reports/mypy-report.txt", "r", encoding="utf-8") as f:
        content = f.read()
        mypy_issues = len(re.findall(r"error:", content))

    print_colored(
        f"mypy: {'No type issues found ✓' if mypy_issues == 0 else f'Found {mypy_issues} issues ✗'}",
        "green" if mypy_issues == 0 else "red",
    )
    issue_counts["mypy"] = mypy_issues

    # Run bandit
    with open("reports/bandit-report.txt", "w", encoding="utf-8") as f:
        # Execute without storing the result
        subprocess.run(
            ["bandit", "-r"] + files,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    with open("reports/bandit-report.txt", "r", encoding="utf-8") as f:
        content = f.read()
        bandit_issues = len(re.findall(r"Issue", content))

    print_colored(
        f"bandit: {'No security issues found ✓' if bandit_issues == 0 else f'Found {bandit_issues} issues ✗'}",
        "green" if bandit_issues == 0 else "red",
    )
    issue_counts["bandit"] = bandit_issues

    return issue_counts


def add_missing_docstrings(files: List[str]) -> int:
    """
    Add missing module docstrings to Python files.

    Args:
        files: List of files to check

    Returns:
        Number of files fixed
    """
    print_colored("\n▶️ Adding missing module docstrings...", "blue")

    fixed_count = 0

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if file lacks a module docstring
        if not re.search(r'^""".*?"""', content, re.DOTALL) and not re.search(
            r"^'''.*?'''", content, re.DOTALL
        ):
            # Generate an appropriate docstring based on the module name
            module_name = os.path.basename(file_path).replace(".py", "")

            if module_name == "main":
                docstring = (
                    '"""\nmain.py - Main entry point for hierarchical object recognition system\n\n'
                    "This script integrates all components of the hierarchical object recognition system\n"
                    'and provides a command-line interface for training, evaluating, and making predictions.\n"""\n\n'
                )
            elif module_name == "data_utils":
                docstring = (
                    '"""\ndata_utils.py - Dataset handling for hierarchical object recognition\n\n'
                    "This module handles loading, preprocessing, and organizing the CIFAR-10 dataset\n"
                    'into a hierarchical structure for multi-level classification.\n"""\n\n'
                )
            elif module_name == "model":
                docstring = (
                    '"""\nmodel.py - Neural network architecture for hierarchical object recognition\n\n'
                    "This module defines the CNN-based model architecture with two output heads:\n"
                    'one for superclass (coarse) classification and one for class (fine-grained) classification.\n"""\n\n'
                )
            elif module_name == "train":
                docstring = (
                    '"""\ntrain.py - Training procedures for hierarchical object recognition\n\n'
                    "This module contains functions for training the hierarchical classification model,\n"
                    'including callbacks and training loop implementation.\n"""\n\n'
                )
            elif module_name == "evaluate":
                docstring = (
                    '"""\nevaluate.py - Evaluation metrics for hierarchical object recognition\n\n'
                    "This module contains functions for evaluating the hierarchical classification model,\n"
                    'including metrics specific to hierarchical classification performance.\n"""\n\n'
                )
            elif module_name == "visualize":
                docstring = (
                    '"""\nvisualize.py - Visualization utilities for hierarchical object recognition\n\n'
                    "This module provides functions for visualizing training progress, model predictions,\n"
                    'confusion matrices, and hierarchical classification results.\n"""\n\n'
                )
            else:
                docstring = f'"""\n{module_name.replace("_", " ").title()} module.\n"""\n\n'

            # Add docstring to the beginning of the file
            new_content = docstring + content

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            fixed_count += 1

    print_colored(f"✓ Added module docstrings to {fixed_count} files", "green")

    return fixed_count


def fix_trailing_whitespace_and_newlines(files: List[str]) -> int:
    """
    Fix trailing whitespace and ensure files end with a newline.

    Args:
        files: List of files to check

    Returns:
        Number of files fixed
    """
    print_colored("\n▶️ Fixing trailing whitespace and missing newlines...", "blue")

    fixed_count = 0

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Fix trailing whitespace
        new_content = re.sub(r"[ \t]+$", "", content, flags=re.MULTILINE)

        # Ensure file ends with exactly one newline
        if not new_content.endswith("\n"):
            new_content += "\n"
        else:
            while new_content.endswith("\n\n"):
                new_content = new_content[:-1]

        # Fix blank lines with whitespace
        new_content = re.sub(r"\n[ \t]+\n", "\n\n", new_content)

        # Ensure proper spacing between functions/classes (2 blank lines)
        new_content = re.sub(r"(\n)class ", r"\n\n\nclass ", new_content)
        new_content = re.sub(r"(\n)def ", r"\n\n\ndef ", new_content)
        new_content = re.sub(r"\n\n\n\n+class", r"\n\n\nclass", new_content)
        new_content = re.sub(r"\n\n\n\n+def", r"\n\n\ndef", new_content)

        # Write changes if content was modified
        if content != new_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            fixed_count += 1

    print_colored(f"✓ Fixed whitespace and newline issues in {fixed_count} files", "green")

    return fixed_count


def main() -> None:
    """Main function to run code quality checks."""
    global previous_issues_count

    parser = argparse.ArgumentParser(description="Run code quality checks.")
    parser.add_argument("--fix", action="store_true", help="Automatically fix issues")
    parser.add_argument("--files", nargs="+", help="Specific files to check")
    args = parser.parse_args()

    print_colored("Running code quality checks...", "bold")
    print_colored("-------------------------------", "bold")

    # Find Python files
    if args.files:
        python_files = [f for f in args.files if f.endswith(".py")]
    else:
        python_files = find_python_files()

    print(f"Found {len(python_files)} Python files to check")

    if args.fix:
        # First, apply automatic fixes
        add_missing_docstrings(python_files)
        fix_trailing_whitespace_and_newlines(python_files)

    # Run linting tools
    issue_counts = lint_files(python_files, fix=args.fix)

    # Print summary
    total_issues = sum(issue_counts.values())

    print_colored("\nSUMMARY:", "bold")
    print_colored("--------", "bold")

    if total_issues == 0:
        print_colored("No issues found! Your code meets all quality standards. 🎉", "green")
    else:
        print_colored(f"Total issues found: {total_issues}", "yellow" if args.fix else "red")

        for tool, count in issue_counts.items():
            if count > 0:
                print(f"- {tool.capitalize()}: {count} issues")

        print("\nReports are saved in the 'reports' directory.")

        if args.fix:
            if previous_issues_count is not None:
                fixed_issues = previous_issues_count - total_issues

                if fixed_issues > 0:
                    improvement = (fixed_issues / previous_issues_count) * 100
                    print_colored(f"\n🎉 Fixed {fixed_issues} issues!", "green")
                    print_colored(f"🔍 Improvement: {improvement:.1f}%", "green")

                    if improvement >= 90:
                        print_colored("✅ You've reached the 90% improvement target!", "green")
                    elif improvement >= 50:
                        print_colored("✅ You've reached the 50% improvement target!", "green")
                        print_colored("🚀 Keep going to reach the 90% target!", "blue")
                    else:
                        print_colored(
                            f"🚀 You're getting close to the 50% improvement target!", "blue"
                        )
                else:
                    print_colored("\n⚠️ No issues were fixed.", "yellow")

            if total_issues > 0:
                print_colored("\n📝 Next steps:", "bold")
                print("1. Review reports in the 'reports' directory")
                print("2. Manually fix remaining issues, focusing on:")
                print("   - Adding function and class docstrings")
                print("   - Adding type annotations")
                print("   - Fixing variable naming")
                print("   - Addressing any security issues")
                print("3. Run this script again to check your progress")
        else:
            # Store current issues count for comparison in fix mode
            previous_issues_count = total_issues

            if total_issues > 0:
                print_colored("\n💡 Tip: Run with --fix to automatically fix some issues:", "blue")
                print("  python scripts/lint.py --fix")


if __name__ == "__main__":
    main()
