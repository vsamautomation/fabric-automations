# M Code to SQL Transformation - Usage Guide

## Overview

This module extends the `AI_Notebook_Refactored.py` workflow to extract Power Query M code from the `dataset_expressions` lakehouse table and transform it to SQL using AI agents (Claude or Gemini).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Fabric Lakehouse                             │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ dataset_         │  │ ai_dataset_      │                    │
│  │ expressions      │  │ context          │                    │
│  └────────┬─────────┘  └──────────────────┘                    │
└───────────┼──────────────────────────────────────────────────────┘
            │
            ▼
   ┌────────────────────┐
   │ MCodeExtractor     │  ← Extract M code from lakehouse
   └────────┬───────────┘
            │
            ▼
   ┌────────────────────┐
   │ MCodeToSQL         │  ← Build prompts for AI
   │ Integration        │
   └────────┬───────────┘
            │
            ▼
   ┌────────────────────┐
   │ TSQLMigrationPrep  │  ← Generate SQL via Claude/Gemini
   │ (AI Client)        │
   └────────┬───────────┘
            │
            ▼
   ┌────────────────────┐
   │ Generated SQL      │  ← T-SQL SELECT statements
   │ Results            │
   └────────────────────┘
```

## Installation

### 1. Required Files
- `AI_Notebook_Refactored.py` (existing)
- `m_code_extractor.py` (new module)

### 2. Prerequisites
```python
# Already installed via AI_Notebook_Refactored.py
%pip install -q -U semantic-link-labs google-genai anthropic typing_extensions pydantic
```

### 3. Lakehouse Table Required
The module expects a lakehouse table named `dataset_expressions` with schema:
```
dataset_id: string
dataset_name: string
workspace_id: string
workspace_name: string
table_name: string
expression: string (M code)
expression_type: string ('table', 'partition', 'calculated_column')
object_name: string (optional)
column_name: string (optional)
```

## Usage Examples

### Example 1: Basic M Code Extraction

```python
from m_code_extractor import MCodeExtractor
from pyspark.sql import SparkSession

# Initialize
spark = SparkSession.builder.getOrCreate()
extractor = MCodeExtractor(spark)

# Load M code expressions from lakehouse
extractor.load_expressions_from_lakehouse("dataset_expressions")

# Get summary of all datasets with M code
summary = extractor.get_datasets_with_expressions()
display(summary)

# Extract for a specific dataset
dataset_id = "your-dataset-id"
result = extractor.extract_by_dataset(dataset_id)

print(f"Dataset: {result.dataset_name}")
print(f"Total expressions: {result.total_expressions}")
print(f"Tables: {result.tables_with_expressions}")
```

### Example 2: Integration with AI Notebook (Full Workflow)

```python
# Step 1: Import both modules
from AI_Notebook_Refactored import TSQLMigrationPrep
from m_code_extractor import MCodeExtractor, MCodeToSQLIntegration
from pyspark.sql import SparkSession

# Step 2: Initialize components
spark = SparkSession.builder.getOrCreate()

# Initialize AI client
gemini_key = "your-api-key"
tsql_prep = TSQLMigrationPrep(api_key=gemini_key, agent_mode='gemini')

# Initialize M code extractor
m_extractor = MCodeExtractor(spark)
m_extractor.load_expressions_from_lakehouse("dataset_expressions")

# Step 3: Create integration
integration = MCodeToSQLIntegration(m_extractor, tsql_prep)

# Step 4: Generate SQL for a dataset
dataset_id = "5fef939e-8bd0-40e1-a0c5-a7a9a49094d1"
sql_results = integration.generate_sql_from_m_code(
    dataset_id=dataset_id,
    target_table_prefix="stg_"
)

# Step 5: View results
for transformation in sql_results['transformations']:
    print(f"\n{'='*80}")
    print(f"Table: {transformation['table_name']}")
    print(f"Type: {transformation['expression_type']}")
    print(f"\n--- ORIGINAL M CODE ---")
    print(transformation['original_m_code'])
    print(f"\n--- GENERATED SQL ---")
    print(transformation['generated_sql'])
```

### Example 3: Process Specific Table Only

```python
# Extract M code for one table
table_expressions = m_extractor.extract_by_table(
    dataset_id="your-dataset-id",
    table_name="Sales"
)

# Generate SQL for specific table
sql_results = integration.generate_sql_from_m_code(
    dataset_id="your-dataset-id",
    table_name="Sales",
    target_table_prefix="raw_"
)
```

### Example 4: Filter M Code by Pattern

```python
# Find all M code with specific operations
table_operations = m_extractor.filter_expressions_by_pattern(
    pattern="Table\\.SelectColumns",
    dataset_id="your-dataset-id"
)

join_operations = m_extractor.filter_expressions_by_pattern(
    pattern="Table\\.NestedJoin|Table\\.Join"
)

print(f"Found {len(table_operations)} SELECT operations")
print(f"Found {len(join_operations)} JOIN operations")
```

### Example 5: Export for External Processing

```python
# Extract M code
result = m_extractor.extract_by_dataset("your-dataset-id")

# Export to JSON format
export_data = m_extractor.export_for_ai_processing(
    extraction_result=result,
    include_metadata=True
)

# Save to file
import json
with open("m_code_export.json", "w") as f:
    json.dump(export_data, f, indent=2)
```

### Example 6: Custom AI Prompt Building

```python
from m_code_extractor import MCodeExpression

# Get expressions
expressions = m_extractor.extract_by_table("dataset-id", "Sales")

for expr in expressions:
    # Build custom prompt
    prompt = integration.build_m_to_sql_prompt(
        m_code_expression=expr,
        target_table_name="raw_sales",
        additional_context="""
Additional Requirements:
- Use ANSI SQL standards
- Add row_number() for versioning
- Include audit columns (created_date, modified_date)
"""
    )
    
    # Use prompt with AI client directly
    # ... your custom AI call here
```

## M to SQL Transformation Mapping

| M Operation | SQL Equivalent | Example |
|------------|----------------|---------|
| `Table.SelectColumns(table, {"Col1", "Col2"})` | `SELECT Col1, Col2 FROM table` | Column selection |
| `Table.SelectRows(table, each [Amount] > 1000)` | `WHERE Amount > 1000` | Row filtering |
| `Table.ReplaceValue(table, "old", "new", Replacer.ReplaceText, {"Col"})` | `REPLACE(Col, 'old', 'new')` | String replacement |
| `Table.AddColumn(table, "NewCol", each [Col1] + [Col2])` | `Col1 + Col2 AS NewCol` | Calculated column |
| `Table.RenameColumns(table, {{"Old", "New"}})` | `Old AS New` | Column alias |
| `Table.TransformColumnTypes(table, {{"Col", Int64.Type}})` | `CAST(Col AS BIGINT)` | Type conversion |
| `Table.Group(table, {"Col"}, {{"Sum", each List.Sum([Amount])}})` | `GROUP BY Col; SUM(Amount)` | Aggregation |
| `Table.Join(t1, "Key", t2, "Key", JoinKind.Inner)` | `INNER JOIN t2 ON t1.Key = t2.Key` | Table joins |

## Integration with Existing AI Notebook Workflow

### Complete Batch Processing Example

```python
# Load data from existing AI notebook workflow
data_context_pd = spark.table("ai_dataset_context").toPandas()
objects_pd = spark.table("ai_object_features").toPandas()

# Initialize M code processing
m_extractor = MCodeExtractor(spark)
m_extractor.load_expressions_from_lakehouse("dataset_expressions")

# Initialize AI client
tsql_prep = TSQLMigrationPrep(
    api_key="your-key",
    agent_mode='claude'
)

# Create integration
integration = MCodeToSQLIntegration(m_extractor, tsql_prep)

# Process all datasets with M code
datasets_with_mcode = m_extractor.get_datasets_with_expressions()

results_list = []
for _, row in datasets_with_mcode.iterrows():
    dataset_id = row['dataset_id']
    
    # Generate SQL transformations
    sql_result = integration.generate_sql_from_m_code(
        dataset_id=dataset_id,
        target_table_prefix="bronze_"
    )
    
    results_list.append(sql_result)

# Save results to lakehouse
import pandas as pd
results_df = pd.DataFrame([
    {
        'dataset_id': r['dataset_id'],
        'dataset_name': r['dataset_name'],
        'workspace_name': r['workspace_name'],
        'table_name': t['table_name'],
        'expression_type': t['expression_type'],
        'original_m_code': t['original_m_code'],
        'generated_sql': t['generated_sql'],
        'status': t['status']
    }
    for r in results_list
    for t in r['transformations']
])

spark_df = spark.createDataFrame(results_df)
spark_df.write.mode("overwrite").saveAsTable("m_to_sql_transformations")
```

## Troubleshooting

### Issue: `dataset_expressions` table not found

**Solution**: You need to create and populate this table first. Use the workspace scanner to extract M code:

```python
# Example: Extract M code from dataset
import sempy.fabric as fabric

dataset_id = "your-dataset-id"
workspace_id = "your-workspace-id"

# Get table expressions
tables = fabric.list_tables(dataset=dataset_id, workspace=workspace_id)

# Extract partition expressions (M code)
expressions_data = []
for _, table in tables.iterrows():
    table_name = table['Name']
    
    # Get table definition which includes M code
    # ... extraction logic here ...
    
# Save to lakehouse
expressions_df = pd.DataFrame(expressions_data)
spark.createDataFrame(expressions_df).write.mode("overwrite").saveAsTable("dataset_expressions")
```

### Issue: AI rate limits

**Solution**: Add delays between API calls:

```python
import time

for dataset_id in dataset_ids:
    sql_result = integration.generate_sql_from_m_code(dataset_id)
    time.sleep(2)  # 2 second delay between datasets
```

### Issue: Empty M code expressions

**Solution**: Check if tables have partition expressions defined. Not all Power BI tables use M code (e.g., DirectQuery tables don't have M expressions).

## Best Practices

1. **Batch Processing**: Process datasets in batches to manage API costs
2. **Error Handling**: Wrap AI calls in try/except blocks
3. **Caching**: Save intermediate results to lakehouse tables
4. **Validation**: Always review generated SQL before deploying
5. **Version Control**: Track M code and SQL transformations over time

## Cost Considerations

- **Gemini**: More cost-effective for bulk processing
- **Claude**: Better accuracy for complex M code transformations
- Use `claude-haiku` for simple transformations
- Use `claude-sonnet` or `gemini-pro` for complex logic

## Next Steps

1. Extract M code from your Power BI datasets using workspace scanner
2. Save to `dataset_expressions` lakehouse table
3. Run M code extractor to validate data
4. Generate SQL transformations using AI
5. Review and deploy SQL to Fabric Data Warehouse
