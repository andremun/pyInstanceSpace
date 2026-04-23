# exploreIS Test Suite

Validation and unit tests for the `explore()` pipeline implementation.

## Directory Structure

```
tests/exploreIS/
├── README.md
│
├── prelim/
│   ├── test_prelim_validation.py
│   ├── test_prelim_unit.py
│   └── __init__.py
│
├── sifted/
│   ├── test_sifted_validation.py
│   ├── test_sifted_unit.py
│   └── __init__.py
│
├── pilot/     (TODO)
├── pythia/    (TODO)
└── trace/     (TODO)
```

## Test Types

### 1. Validation Tests (`test_<stage>_validation.py`)
- Compare Python implementation against MATLAB reference outputs
- Check for 5% tolerance threshold
- Display detailed comparison statistics
- Include visual banners for pass/fail status
- **Purpose:** Ensure correctness vs MATLAB reference

### 2. Unit Tests (`test_<stage>_unit.py`)
- Test edge cases and error handling
- No external dependencies (no MATLAB data required)
- Fast execution with pytest
- **Purpose:** Ensure robustness and code quality

**Unit test coverage:**
- Basic functionality
- Single/multiple instances
- NaN value handling
- Boundary conditions
- Input preservation
- Deterministic behavior

## Running Tests

### Run Validation Tests (visual output)
```bash
# PRELIM validation
python tests/exploreIS/prelim/test_prelim_validation.py

# SIFTED validation
python tests/exploreIS/sifted/test_sifted_validation.py
```

### Run Unit Tests (pytest)
```bash
# All unit tests
pytest tests/exploreIS/

# PRELIM unit tests
pytest tests/exploreIS/prelim/test_prelim_unit.py -v

# SIFTED unit tests
pytest tests/exploreIS/sifted/test_sifted_unit.py -v
```

## Output Format

Each validation test displays:

1. **Input Data Summary**
   - Data source
   - Dimensions
   - Parameters used

2. **Python Implementation Output**
   - Transformed data shape
   - Sample values

3. **MATLAB Reference**
   - Reference data source
   - Expected output shape

4. **Comparison Results**
   - Difference statistics (max/mean errors)
   - Detailed comparison table
   - Validation status (PASS/FAIL)

5. **Visual Status Banner**
   - Large "PASSED" banner on success
   - Error details on failure

## Validation Criteria

All tests must meet:
- **Tolerance:** Python outputs match MATLAB within 5% relative error
- **Shape:** Output dimensions must match exactly
- **Columns:** Feature/output names must match
