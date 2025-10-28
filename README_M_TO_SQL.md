# M Code to SQL Transformation Module

Transform Power Query M code from Power BI datasets to T-SQL for Fabric Lakehouse/Data Warehouse deployment.

## 📦 Files Created

1. **`m_code_extractor.py`** - Main module for extracting and processing M code
2. **`M_TO_SQL_USAGE_GUIDE.md`** - Comprehensive usage documentation
3. **`m_to_sql_transformer.py`** - (Optional) Standalone parser for M-to-SQL conversion
4. **`m_to_sql_cli.py`** - (Optional) Command-line tool

## 🚀 Quick Start

### Prerequisites

You need the `dataset_expressions` table in your Fabric Lakehouse with this structure:

```sql
CREATE TABLE dataset_expressions (
    dataset_id STRING,
    dataset_name STRING,
    workspace_id STRING,
    workspace_name STRING,
    table_name STRING,
    expression STRING,          -- M code
    expression_type STRING,     -- 'table', 'partition', etc.
    object_name STRING,
    column_name STRING
);
```

### Basic Usage (Fabric Notebook)

```python
# Import modules
from m_code_extractor import MCodeExtractor, MCodeToSQLIntegration
from AI_Notebook_Refactored import TSQLMigrationPrep
from pyspark.sql import SparkSession

# Initialize
spark = SparkSession.builder.getOrCreate()

# Step 1: Extract M code
extractor = MCodeExtractor(spark)
extractor.load_expressions_from_lakehouse("dataset_expressions")

# Step 2: Initialize AI client
tsql_prep = TSQLMigrationPrep(
    api_key="your-api-key",
    agent_mode='gemini'  # or 'claude'
)

# Step 3: Create integration
integration = MCodeToSQLIntegration(extractor, tsql_prep)

# Step 4: Generate SQL
results = integration.generate_sql_from_m_code(
    dataset_id="your-dataset-id",
    target_table_prefix="stg_"
)

# Step 5: View results
for t in results['transformations']:
    print(f"\n{'='*80}")
    print(f"Table: {t['table_name']}")
    print(f"\nOriginal M Code:\n{t['original_m_code']}")
    print(f"\nGenerated SQL:\n{t['generated_sql']}")
```

## 🔧 Integration with AI Notebook

This module extends `AI_Notebook_Refactored.py` workflow:

```python
# Use existing AI notebook infrastructure
from AI_Notebook_Refactored import main, TSQLMigrationPrep

# Add M code processing
from m_code_extractor import MCodeExtractor, MCodeToSQLIntegration

# Initialize both
tsql_prep = TSQLMigrationPrep(api_key=api_key, agent_mode='gemini')
m_extractor = MCodeExtractor(spark)
m_extractor.load_expressions_from_lakehouse("dataset_expressions")

# Create integration
integration = MCodeToSQLIntegration(m_extractor, tsql_prep)

# Process datasets
for dataset_id in dataset_ids:
    # Generate T-SQL CREATE TABLE scripts (existing functionality)
    migration_spec = tsql_prep.prepare_dataset_migration(dataset_id)
    create_table_sql = tsql_prep.generate_tsql_with_ai(migration_spec)
    
    # Generate M-to-SQL transformations (NEW functionality)
    transformation_sql = integration.generate_sql_from_m_code(
        dataset_id=dataset_id,
        target_table_prefix="stg_"
    )
    
    # Save both results
    # ... your lakehouse save logic
```

## 📊 Example Output

### Input (M Code):
```m
let
    Source = Sql.Database("server", "AdventureWorks"),
    Sales = Source{[Schema="Sales",Item="Orders"]}[Data],
    FilteredRows = Table.SelectRows(Sales, each [Amount] > 1000),
    SelectedColumns = Table.SelectColumns(FilteredRows, {"OrderID", "CustomerID", "Amount"}),
    AddedYear = Table.AddColumn(SelectedColumns, "Year", each Date.Year([OrderDate]))
in
    AddedYear
```

### Output (Generated SQL):
```sql
-- Generated T-SQL transformation for table: Orders
-- Source: AdventureWorks database

SELECT 
    OrderID,
    CustomerID,
    Amount,
    YEAR(OrderDate) AS Year  -- Calculated column
FROM stg_Orders
WHERE Amount > 1000  -- Filter condition from M code
```

## 🎯 Key Features

1. **Lakehouse Integration** - Reads M code from lakehouse tables
2. **AI-Powered** - Uses Claude or Gemini for accurate transformations
3. **Context-Aware** - Preserves dataset/workspace metadata
4. **Batch Processing** - Process multiple datasets efficiently
5. **Pattern Matching** - Find specific M operations across datasets
6. **Export Options** - JSON export for external tools

## 📚 M-to-SQL Operation Mapping

| M Operation | SQL Equivalent |
|-------------|----------------|
| `Table.SelectColumns()` | `SELECT col1, col2` |
| `Table.SelectRows()` | `WHERE condition` |
| `Table.ReplaceValue()` | `REPLACE(col, 'old', 'new')` |
| `Table.AddColumn()` | Calculated column in `SELECT` |
| `Table.RenameColumns()` | Column `AS` alias |
| `Table.Join()` | `INNER/LEFT/RIGHT JOIN` |
| `Table.Group()` | `GROUP BY` with aggregations |

## 🔍 Finding M Code in Your Datasets

If you don't have the `dataset_expressions` table yet, extract M code from Power BI:

```python
import sempy.fabric as fabric

# Get table metadata
tables = fabric.list_tables(dataset="dataset-id", workspace="workspace-id")

# M code is typically in table expressions/partitions
# You'll need to use TMSL or similar APIs to extract full expressions
```

## ⚙️ Configuration

### AI Provider Selection

**Gemini** (Recommended for bulk processing):
- Cost-effective
- Fast response times
- Good for straightforward transformations

**Claude** (Recommended for complex logic):
- Better understanding of context
- More accurate for complex M code
- Higher cost

```python
# Gemini
tsql_prep = TSQLMigrationPrep(api_key=gemini_key, agent_mode='gemini')

# Claude
tsql_prep = TSQLMigrationPrep(api_key=claude_key, agent_mode='claude')
```

## 💾 Save Results to Lakehouse

```python
# Convert results to DataFrame
import pandas as pd

results_df = pd.DataFrame([
    {
        'dataset_id': r['dataset_id'],
        'table_name': t['table_name'],
        'original_m_code': t['original_m_code'],
        'generated_sql': t['generated_sql'],
        'status': t['status']
    }
    for r in results_list
    for t in r['transformations']
])

# Save to lakehouse
spark_df = spark.createDataFrame(results_df)
spark_df.write.mode("overwrite").saveAsTable("m_to_sql_results")
```

## 🐛 Troubleshooting

**Issue**: `dataset_expressions` table not found
- **Fix**: Create and populate the table first using workspace scanner

**Issue**: No M code found for dataset
- **Fix**: Check if dataset uses Import mode (DirectQuery doesn't have M code)

**Issue**: AI rate limit errors
- **Fix**: Add delays between API calls (`time.sleep(2)`)

**Issue**: Invalid SQL generated
- **Fix**: Review and manually adjust complex transformations

## 📖 Documentation

- **Full Guide**: See `M_TO_SQL_USAGE_GUIDE.md`
- **AI Notebook**: See `AI_Notebook_Refactored.py`
- **Examples**: See usage guide for 6 detailed examples

## 🤝 Integration Architecture

```
Fabric Workspace Scanner (v02_enhanced_refactored.ipynb)
    ↓
Lakehouse Tables (dataset_expressions, ai_dataset_context)
    ↓
M Code Extractor (m_code_extractor.py)
    ↓
AI Integration (MCodeToSQLIntegration)
    ↓
T-SQL Generation (via Claude/Gemini)
    ↓
Fabric Data Warehouse Deployment
```

## ✅ Next Steps

1. **Extract M Code**: Run workspace scanner to populate `dataset_expressions`
2. **Test Extraction**: Use `MCodeExtractor` to validate M code
3. **Generate SQL**: Use integration module with AI
4. **Review Output**: Validate generated SQL
5. **Deploy**: Create views/stored procedures in Fabric DW

## 📝 Example Notebook Cell

```python
# Complete example for Fabric notebook
from m_code_extractor import MCodeExtractor, MCodeToSQLIntegration
from AI_Notebook_Refactored import TSQLMigrationPrep

# Config
gemini_key = "your-api-key"
dataset_id = "your-dataset-id"

# Initialize
extractor = MCodeExtractor()
extractor.load_expressions_from_lakehouse("dataset_expressions")

tsql_prep = TSQLMigrationPrep(api_key=gemini_key, agent_mode='gemini')
integration = MCodeToSQLIntegration(extractor, tsql_prep)

# Generate SQL
results = integration.generate_sql_from_m_code(
    dataset_id=dataset_id,
    target_table_prefix="stg_"
)

# Display
for t in results['transformations']:
    print(f"✅ {t['table_name']}: {t['status']}")
```

---

**For detailed documentation, see `M_TO_SQL_USAGE_GUIDE.md`**
