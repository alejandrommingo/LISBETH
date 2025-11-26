# Contributing to Lisbeth News Harvester

Thank you for your interest in contributing! This document outlines the roles and workflows to ensure the project remains consistent and robust.

## Roles and Responsibilities

### 1. Maintainer
**Focus**: Core Logic & Architecture (`src/news_harvester/*.py`, `pyproject.toml`)
- Ensures the CLI and configuration management are stable.
- Reviews PRs for code quality and test coverage.
- Manages releases and dependencies.

### 2. Data Engineer
**Focus**: Collectors & Processing (`src/news_harvester/collectors/`, `src/news_harvester/processing/`)
- Maintains the GDELT integration and other collectors.
- Optimizes the HTML cleaning and text extraction algorithms.
- Monitors the quality of the downloaded data (e.g., handling new paywalls or layout changes).

### 3. Data Analyst
**Focus**: Data Usage & Validation (`data/`, `notebooks/`)
- Uses the generated datasets for analysis.
- Reports issues with data quality (e.g., "missing titles", "noisy text").
- Suggests new features or metrics.

## Development Workflow

1.  **Install Dependencies**:
    ```bash
    python -m pip install -e .[dev]
    ```

2.  **Run Tests**:
    ```bash
    pytest
    ```

3.  **Code Style**:
    - We use `ruff` for linting and formatting.
    - Run `ruff check .` and `ruff format .` before committing.

## Reporting Issues
When reporting an issue, please specify:
- The command run (including arguments).
- The error log (if any).
- The expected vs. actual result.
