# AI-Optimized Lakehouse Schema Documentation

## Overview

This document describes the **AI-optimized lakehouse schema** designed for training machine learning models to recommend Power BI schema optimizations.

## Architecture

The solution creates **2 primary tables** optimized for fast AI queries and accurate model training:

1. **`ai_dataset_context`** - Dataset-level features with health scores
2. **`ai_object_features`** - Object-level features (columns & measures) with full context

---

## Table 1: `ai_dataset_context`

**Purpose**: Dataset-level aggregated features for training optimization priority models.

### Schema (16 columns)

| Column | Type | Description | Usage |
|--------|------|-------------|-------|
| `workspace_id` | string | Workspace identifier | Grouping |
| `workspace_name` | string | Workspace name | Context |
| `dataset_id` | string | Dataset identifier | Primary key |
| `dataset_name` | string | Dataset name | Context |
| **Size Metrics** ||||
| `total_tables` | int | Number of tables in dataset | Feature |
| `total_columns` | int | Number of columns | Feature |
| `total_measures` | int | Number of measures | Feature |
| `total_relationships` | int | Number of relationships | Feature |
| **Usage Metrics** ||||
| `report_count` | int | Reports using this dataset | Feature |
| `dataflow_count` | int | Dataflows using dataset | Feature |
| `connected_tables` | int | Tables with relationships | Feature |
| `isolated_tables` | int | Tables without relationships | Feature |
| **Quality Metrics** ||||
| `unused_tables` | int | Tables not used anywhere | Feature |
| `unused_columns` | int | Columns not referenced | Feature |
| `unused_measures` | int | Measures not used | Feature |
| `circular_chains` | int | Circular relationship count | Feature |
| **Health Scores (0-1)** ||||
| `relationship_health` | float | `connected_tables / total_tables` | Label/Feature |
| `usage_efficiency` | float | `1 - (unused_columns / total_columns)` | Label/Feature |
| `model_complexity` | float | Normalized complexity score | Feature |
| `optimization_score` | float | **Overall health (0-100)** | **Primary Label** |

### Optimization Score Formula

```python
optimization_score = (
    relationship_health * 30 +      # 30% weight
    usage_efficiency * 40 +          # 40% weight
    (connected_tables/total) * 20 +  # 20% weight
    (1 - unused_tables/total) * 10   # 10% weight
)
```

### Example Queries

**Find datasets needing optimization:**
```sql
SELECT 
    dataset_name,
    optimization_score,
    unused_columns,
    isolated_tables,
    relationship_health
FROM ai_dataset_context
WHERE optimization_score < 60
ORDER BY optimization_score ASC
```

**Training data for priority model:**
```python
features = [
    'total_tables', 'total_columns', 'total_measures',
    'unused_columns', 'isolated_tables', 'relationship_health'
]
target = 'optimization_score'
```

---

## Table 2: `ai_object_features`

**Purpose**: Object-level (column/measure) features for predicting removal candidates and usage patterns.

### Schema (23 columns)

| Column | Type | Description | Usage |
|--------|------|-------------|-------|
| **Full Lineage** ||||
| `workspace_id` | string | Workspace identifier | Context |
| `workspace_name` | string | Workspace name | Context |
| `dataset_id` | string | Dataset identifier | Context |
| `dataset_name` | string | Dataset name | Context |
| `table_name` | string | Table name | Context |
| `object_name` | string | Column/measure name | Identifier |
| `object_type` | string | `column`, `calculated_column`, `measure` | Feature |
| **Object Properties** ||||
| `data_type` | string | Text, Number, DateTime, etc. | Feature |
| `is_hidden` | boolean | Hidden from report view | Feature |
| `is_calculated` | boolean | Calculated column/measure | Feature |
| `has_dax` | boolean | Contains DAX expression | Feature |
| **Table Context** ||||
| `table_measure_count` | int | Measures in this table | Feature |
| `table_column_count` | int | Columns in this table | Feature |
| `table_relationship_count` | int | Relationships for this table | Feature |
| `table_is_isolated` | boolean | Table has no relationships | Feature |
| **Dataset Context (Denormalized)** ||||
| `dataset_total_tables` | int | Total tables in dataset | Feature |
| `dataset_relationship_health` | float | Dataset health score | Feature |
| `dataset_usage_efficiency` | float | Dataset efficiency score | Feature |
| **Usage Features** ||||
| `used_by_measures` | int | Count of measures referencing | Feature |
| `used_by_relationships` | int | Count of relationships using | Feature |
| `used_by_dependencies` | int | Total dependency count | Feature |
| `is_used` | boolean | **Is object used anywhere** | **Primary Label** |
| `usage_score` | float | Normalized usage score (0-1) | Label/Feature |
| **Explainability** ||||
| `referenced_by_list` | string (JSON) | Array of objects referencing this | Explainability |

### Usage Score Formula

**For Columns:**
```python
usage_score = min(
    (used_by_measures * 0.4 + 
     used_by_relationships * 0.4 + 
     used_by_dependencies * 0.2) / 5,
    1.0
)
```

**For Measures:**
```python
usage_score = min(
    (used_by_measures * 0.5 + 
     used_by_dependencies * 0.5) / 3,
    1.0
)
```

### Example Queries

**Find unused columns in isolated tables:**
```sql
SELECT 
    dataset_name,
    table_name,
    object_name,
    data_type,
    table_is_isolated,
    is_used,
    referenced_by_list
FROM ai_object_features
WHERE 
    is_used = FALSE 
    AND object_type = 'column'
    AND table_is_isolated = TRUE
ORDER BY dataset_name, table_name
```

**Training data for "should remove" classifier:**
```python
features = [
    'is_calculated', 'table_is_isolated', 
    'table_relationship_count', 'dataset_relationship_health',
    'used_by_measures', 'used_by_relationships'
]
target = 'is_used'  # Binary classification
```

**Get high-value unused columns:**
```sql
SELECT 
    object_name,
    table_name,
    dataset_name,
    usage_score,
    used_by_measures,
    used_by_relationships
FROM ai_object_features
WHERE 
    is_used = FALSE
    AND table_is_isolated = FALSE  -- In connected tables
    AND data_type IN ('Number', 'DateTime')  -- High-value types
ORDER BY usage_score DESC
```

---

## ML Model Examples

### Model 1: Column Removal Predictor

**Type**: Binary Classification

**Features**:
```python
X_features = [
    'is_calculated',
    'table_is_isolated',
    'table_relationship_count',
    'dataset_relationship_health',
    'used_by_measures',
    'used_by_relationships',
    'used_by_dependencies'
]

y_target = 'is_used'  # 0 = can remove, 1 = keep
```

**Use Case**: Predict which columns can be safely removed

---

### Model 2: Dataset Optimization Priority

**Type**: Regression

**Features**:
```python
X_features = [
    'total_tables', 
    'total_columns',
    'unused_columns',
    'isolated_tables',
    'relationship_health',
    'usage_efficiency',
    'circular_chains'
]

y_target = 'optimization_score'  # 0-100 score
```

**Use Case**: Rank datasets by optimization priority

---

### Model 3: Measure Impact Predictor

**Type**: Multi-class Classification

**Features**:
```python
X_features = [
    'has_dax',
    'used_by_measures',
    'used_by_dependencies',
    'table_measure_count',
    'dataset_usage_efficiency'
]

y_target = 'impact_level'  # Low/Medium/High (derived from usage_score)
```

**Use Case**: Predict impact of removing a measure

---

## Data Quality Checks

### Validation Queries

**Check for datasets with all columns unused:**
```sql
SELECT 
    dataset_name,
    COUNT(*) as total_objects,
    SUM(CASE WHEN is_used = FALSE THEN 1 ELSE 0 END) as unused_count
FROM ai_object_features
GROUP BY dataset_name
HAVING unused_count = total_objects
```

**Check health score distribution:**
```sql
SELECT 
    CASE 
        WHEN optimization_score >= 80 THEN 'Excellent'
        WHEN optimization_score >= 60 THEN 'Good'
        WHEN optimization_score >= 40 THEN 'Needs Attention'
        ELSE 'Critical'
    END as health_category,
    COUNT(*) as dataset_count
FROM ai_dataset_context
GROUP BY health_category
```

---

## Integration with AI Models

### Example: OpenAI/Azure OpenAI Integration

```python
import pandas as pd
from openai import AzureOpenAI

# Load data
datasets_df = spark.table("ai_dataset_context").toPandas()
objects_df = spark.table("ai_object_features").toPandas()

# Get dataset context
dataset_context = datasets_df[datasets_df['optimization_score'] < 60].to_dict('records')

# Generate recommendations
client = AzureOpenAI(...)
prompt = f"""
Given this Power BI dataset analysis:
- Total tables: {dataset_context[0]['total_tables']}
- Unused columns: {dataset_context[0]['unused_columns']}
- Isolated tables: {dataset_context[0]['isolated_tables']}
- Optimization score: {dataset_context[0]['optimization_score']}/100

Suggest 3 specific improvements to optimize this schema.
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

---

## Performance Considerations

### Query Optimization

- **Use dataset_id for filtering** when analyzing specific datasets
- **Denormalized dataset context** in object features eliminates joins
- **Index recommendations**:
  - `ai_dataset_context.optimization_score`
  - `ai_object_features.is_used`
  - `ai_object_features.dataset_id`

### Incremental Updates

```python
# Only recompute for changed datasets
changed_dataset_ids = ['abc123', 'def456']

# Filter and regenerate
filtered_results = filter_results(all_columns_df, all_tables_df, ...)
ai_dataset_context_df = generate_ai_dataset_context(...)
ai_object_features_df = generate_ai_object_features(...)

# Merge with existing data
existing_df = spark.table("ai_object_features").toPandas()
updated_df = existing_df[~existing_df['dataset_id'].isin(changed_dataset_ids)]
final_df = pd.concat([updated_df, ai_object_features_df], ignore_index=True)
```

---

## Future Enhancements

### V2.0 Considerations

1. **Add cardinality metrics** for columns
2. **Include query performance data** from Query Store
3. **Add temporal features** (last_used_date, created_date)
4. **Relationship cardinality** (1:1, 1:Many, Many:Many)
5. **DAX complexity scores** for measures
6. **Report-level features** (which reports use which objects)

---

## Usage in Fabric Notebooks

```python
# Import the analyzer
from fabric_workspace_analyzer import FabricWorkspaceAnalyzer

# Run complete analysis
analyzer = FabricWorkspaceAnalyzer()
analyzer.run_complete_analysis()

# Query AI-optimized tables
ai_datasets = spark.table("ai_dataset_context").toPandas()
ai_objects = spark.table("ai_object_features").toPandas()

# Train a model
from sklearn.ensemble import RandomForestClassifier

X = ai_objects[['is_calculated', 'table_is_isolated', 'used_by_measures']]
y = ai_objects['is_used']

model = RandomForestClassifier()
model.fit(X, y)

# Predict removal candidates
predictions = model.predict(X)
removal_candidates = ai_objects[predictions == 0]
```

---

## Support

For questions or enhancements, refer to:
- `fabric_workspace_analyzer.py` - Main analyzer class
- `dataClasses.py` - Data structures
- `PowerBIExtractor.py` - Report metadata extraction

---

**Version**: 1.0  
**Last Updated**: 2025-10-22  
**Optimized For**: AI/ML training and schema optimization recommendations
