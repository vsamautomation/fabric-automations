# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Overview

This is a **Microsoft Fabric Power BI automation and analysis toolkit** that provides comprehensive workspace scanning, relationship analysis, and usage optimization for Power BI datasets. The project has evolved through multiple versions with significant architectural improvements.

## Key Architecture Components

### Core Analysis Pipeline
- **Data Collection**: Centralized functions to gather Power BI metadata from Fabric APIs
- **Relationship Analysis**: Detects table relationships, isolated tables, and circular dependencies
- **Usage Analysis**: Identifies unused columns, measures, and tables through DAX parsing
- **Lakehouse Integration**: Persists all analysis results to dedicated Fabric lakehouse tables

### Main File Structure
- `v02_enhanced_refactored.ipynb` - **Primary notebook** with optimized, function-based architecture
- `v02_enhanced.ipynb` - Previous version (keep for reference, but prefer refactored version)
- `v02.ipynb` - Original version
- `tests/main.py` - Power BI Admin API scanner for large-scale workspace analysis
- `tests/powerbi_analyzer.py` - Core analysis engine combining relationship and usage analysis
- `tests/relationship_analyzer.py` - Specialized relationship detection and circular dependency analysis
- `tests/dax_dependency_parser.py` - DAX expression parser for usage tracking

### Data Flow Architecture
1. **Discovery Phase**: Scan workspaces, datasets, reports, and dataflows
2. **Collection Phase**: Gather comprehensive metadata using `DatasetInfo` data structure
3. **Analysis Phase**: Process relationships, dependencies, and usage patterns
4. **Storage Phase**: Save structured results to lakehouse tables
5. **Insights Phase**: Generate actionable recommendations

## Development Commands

### Running Analysis Notebooks

**Primary notebook (recommended):**
```python
# Open v02_enhanced_refactored.ipynb in Microsoft Fabric
# Execute cells sequentially - the notebook is self-contained
```

### Python Module Testing

**Run comprehensive analysis:**
```bash
cd tests
python main.py  # Scans all workspaces and generates detailed metadata
```

**Test individual analyzers:**
```python
from powerbi_analyzer import PowerBIAnalyzer
from relationship_analyzer import RelationshipAnalyzer
from dax_dependency_parser import UsageAnalyzer

# Create analyzer instance
analyzer = PowerBIAnalyzer(include_hidden=False)

# Analyze from scan results
results = analyzer.analyze_scan_results(scan_data)
```

### Required Dependencies

**For Fabric notebooks:**
```python
!pip install semantic-link-labs
```

**For Python modules:**
- `requests` - Power BI REST API calls
- `msal` - Azure authentication
- `pandas` - Data manipulation
- `dataclasses` - Structured data containers

## Technical Architecture Details

### DatasetInfo Pattern
The refactored architecture uses a centralized `DatasetInfo` dataclass to eliminate redundant API calls:
```python
@dataclass
class DatasetInfo:
    ds_id: str
    ds_name: str
    ws_id: str
    ws_name: str
    dependencies_df: Optional[pd.DataFrame] = None
    tables_df: Optional[pd.DataFrame] = None
    relationships_df: Optional[pd.DataFrame] = None
    measures_df: Optional[pd.DataFrame] = None
    columns_df: Optional[pd.DataFrame] = None
```

### Analysis Functions
- `collect_dataset_info()` - Single API call per dataset to gather all metadata
- `analyze_table_usage()` - Processes table usage patterns from pre-collected data
- `analyze_column_usage()` - Analyzes column dependencies and usage
- `save_to_lakehouse()` - Persists results with timestamps

### Lakehouse Tables Created
- `workspace_analysis` - Basic workspace information
- `dataset_analysis` - Enhanced dataset context with usage metrics
- `table_analysis` - Table usage status with measures/relationships/dependencies count
- `column_usage_analysis` - Detailed column analysis with referenced-by information
- `usage_summary` - Dataset-report relationship patterns

## Authentication & Configuration

### Azure Configuration (for main.py)
```python
AZURE_CLIENT_ID = "your-client-id"
AZURE_TENANT_ID = "your-tenant-id"
SCOPE = ["https://analysis.windows.net/powerbi/api/.default"]
```

### Fabric Integration
Notebooks rely on automatic Fabric authentication when run within the Fabric environment.

## Performance Considerations

### Optimizations Implemented
- **Single-loop processing**: Eliminated duplicate API calls through centralized data collection
- **Cached data structures**: Reuse collected metadata across multiple analysis functions
- **Batch lakehouse writes**: Group related data for efficient storage operations
- **Qualified naming**: Use `'Table'[Column]` format for precise object referencing

### API Rate Limiting
- Main.py includes polling mechanisms with configurable intervals for scan operations
- Error handling and retry logic for workspace scanning failures

## Common Development Patterns

### Adding New Analysis Types
1. Extend `DatasetInfo` if new metadata is needed
2. Modify `collect_dataset_info()` to gather additional data
3. Create new analysis function following the pattern: `analyze_new_type(dataset_info: DatasetInfo) -> List[Dict]`
4. Add lakehouse table creation in the main processing loop

### Error Handling
- All functions include try/except blocks with descriptive error messages
- Non-critical errors allow processing to continue with warnings
- Critical errors are logged with context about the specific workspace/dataset

### Data Quality Patterns
- `sanitize_df_columns()` standardizes column names for Spark compatibility
- Null/empty DataFrame checks before processing
- Qualified naming for precise object identification across datasets

## Testing & Validation

### Notebook Testing
Run all cells in `v02_enhanced_refactored.ipynb` sequentially. Check lakehouse tables for:
- Non-empty result sets
- Proper timestamp columns
- Consistent workspace/dataset/object relationships

### Module Testing
```python
# Test with sample data
analyzer = PowerBIAnalyzer()
results = analyzer.analyze_enriched_reports(sample_reports)
summary = analyzer.get_overall_summary()
```

## Troubleshooting

### Common Issues
- **Authentication failures**: Ensure proper Azure app registration and permissions
- **Empty DataFrames**: Check workspace permissions and dataset accessibility  
- **Lakehouse errors**: Verify lakehouse attachment in Fabric notebook environment
- **Memory issues**: Process workspaces in batches for large tenants

### Debug Tips
- Enable verbose logging in analysis functions
- Use `display()` to inspect intermediate DataFrames
- Check `analysis_date` timestamps to verify successful table writes
- Review relationship and dependency counts for data validation