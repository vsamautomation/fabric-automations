# AI Schema Quick Reference

## 🚀 Quick Start

```python
from fabric_workspace_analyzer import FabricWorkspaceAnalyzer

# Run analysis
analyzer = FabricWorkspaceAnalyzer()
analyzer.run_complete_analysis()

# Load tables
ai_datasets = spark.table("ai_dataset_context").toPandas()
ai_objects = spark.table("ai_object_features").toPandas()
```

---

## 📊 Common Queries

### 1. Find Datasets Needing Optimization

```sql
SELECT 
    dataset_name,
    optimization_score,
    unused_columns,
    unused_measures,
    isolated_tables
FROM ai_dataset_context
WHERE optimization_score < 60
ORDER BY optimization_score ASC
LIMIT 10
```

### 2. Find All Unused Columns

```sql
SELECT 
    workspace_name,
    dataset_name,
    table_name,
    object_name,
    data_type
FROM ai_object_features
WHERE 
    is_used = FALSE 
    AND object_type = 'column'
ORDER BY dataset_name, table_name
```

### 3. Find Unused Columns in Isolated Tables (High Priority)

```sql
SELECT 
    dataset_name,
    table_name,
    object_name,
    data_type,
    referenced_by_list
FROM ai_object_features
WHERE 
    is_used = FALSE
    AND table_is_isolated = TRUE
    AND object_type = 'column'
```

### 4. Dataset Health Score Distribution

```sql
SELECT 
    CASE 
        WHEN optimization_score >= 80 THEN 'Excellent (80-100)'
        WHEN optimization_score >= 60 THEN 'Good (60-79)'
        WHEN optimization_score >= 40 THEN 'Needs Attention (40-59)'
        ELSE 'Critical (0-39)'
    END as health_category,
    COUNT(*) as dataset_count,
    ROUND(AVG(unused_columns), 2) as avg_unused_columns,
    ROUND(AVG(isolated_tables), 2) as avg_isolated_tables
FROM ai_dataset_context
GROUP BY health_category
ORDER BY MIN(optimization_score) DESC
```

### 5. Find Unused Measures

```sql
SELECT 
    dataset_name,
    table_name,
    object_name,
    used_by_measures,
    used_by_dependencies
FROM ai_object_features
WHERE 
    is_used = FALSE
    AND object_type = 'measure'
ORDER BY dataset_name
```

### 6. Tables with Most Unused Columns

```sql
SELECT 
    dataset_name,
    table_name,
    COUNT(*) as total_columns,
    SUM(CASE WHEN is_used = FALSE THEN 1 ELSE 0 END) as unused_columns,
    ROUND(100.0 * SUM(CASE WHEN is_used = FALSE THEN 1 ELSE 0 END) / COUNT(*), 2) as unused_pct
FROM ai_object_features
WHERE object_type IN ('column', 'calculated_column')
GROUP BY dataset_name, table_name
HAVING unused_pct > 50
ORDER BY unused_pct DESC
```

### 7. Most Problematic Datasets (Composite View)

```sql
SELECT 
    dataset_name,
    optimization_score,
    total_tables,
    isolated_tables,
    unused_columns,
    unused_measures,
    relationship_health,
    usage_efficiency,
    CASE 
        WHEN isolated_tables > total_tables * 0.3 THEN 'Many isolated tables'
        WHEN unused_columns > total_columns * 0.5 THEN 'Many unused columns'
        WHEN unused_measures > total_measures * 0.5 THEN 'Many unused measures'
        WHEN relationship_health < 0.5 THEN 'Poor relationships'
        ELSE 'Other issues'
    END as primary_issue
FROM ai_dataset_context
WHERE optimization_score < 60
ORDER BY optimization_score ASC
```

### 8. High-Value Columns Not Being Used

```sql
SELECT 
    dataset_name,
    table_name,
    object_name,
    data_type,
    table_relationship_count,
    dataset_relationship_health
FROM ai_object_features
WHERE 
    is_used = FALSE
    AND table_is_isolated = FALSE  -- In connected tables
    AND data_type IN ('Int64', 'Decimal', 'DateTime', 'Date')  -- High-value types
    AND object_type = 'column'
ORDER BY dataset_relationship_health DESC, table_relationship_count DESC
```

---

## 🤖 ML Training Snippets

### Binary Classifier: Should Remove Column?

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = spark.table("ai_object_features").toPandas()
df = df[df['object_type'].isin(['column', 'calculated_column'])]

# Features
feature_cols = [
    'is_calculated', 
    'table_is_isolated',
    'table_relationship_count',
    'dataset_relationship_health',
    'used_by_measures',
    'used_by_relationships'
]

X = df[feature_cols].fillna(0)
y = df['is_used']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

### Regression: Predict Optimization Score

```python
from sklearn.ensemble import GradientBoostingRegressor

# Load dataset-level data
df = spark.table("ai_dataset_context").toPandas()

# Features
feature_cols = [
    'total_tables',
    'total_columns', 
    'unused_columns',
    'isolated_tables',
    'relationship_health',
    'circular_chains'
]

X = df[feature_cols].fillna(0)
y = df['optimization_score']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"R² Score: {model.score(X_test, y_test):.2f}")
```

---

## 🔍 Data Exploration

### Summary Statistics

```python
import pandas as pd

# Dataset-level summary
datasets = spark.table("ai_dataset_context").toPandas()
print("Dataset Statistics:")
print(datasets[['optimization_score', 'unused_columns', 'isolated_tables']].describe())

# Object-level summary
objects = spark.table("ai_object_features").toPandas()
print("\nObject Usage Summary:")
print(objects.groupby('object_type')['is_used'].value_counts())
```

### Usage Patterns

```python
# Group by dataset and object type
usage_summary = objects.groupby(['dataset_name', 'object_type']).agg({
    'is_used': ['count', 'sum', 'mean']
}).round(2)

print(usage_summary)
```

---

## 💡 Optimization Recommendations

### Rule-Based Recommendations

```python
def get_recommendations(dataset_row):
    """Generate recommendations based on dataset metrics"""
    recommendations = []
    
    if dataset_row['unused_columns'] > dataset_row['total_columns'] * 0.3:
        recommendations.append(f"Remove {dataset_row['unused_columns']} unused columns")
    
    if dataset_row['isolated_tables'] > 0:
        recommendations.append(f"Connect {dataset_row['isolated_tables']} isolated tables")
    
    if dataset_row['unused_measures'] > 5:
        recommendations.append(f"Remove {dataset_row['unused_measures']} unused measures")
    
    if dataset_row['relationship_health'] < 0.5:
        recommendations.append("Improve relationship design")
    
    return recommendations

# Apply to all datasets
datasets['recommendations'] = datasets.apply(get_recommendations, axis=1)
```

---

## 📈 Visualization Examples

### Health Score Distribution

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(datasets['optimization_score'], bins=20, edgecolor='black')
plt.xlabel('Optimization Score')
plt.ylabel('Number of Datasets')
plt.title('Dataset Health Score Distribution')
plt.axvline(x=60, color='r', linestyle='--', label='Needs Attention Threshold')
plt.legend()
plt.show()
```

### Unused Objects by Type

```python
unused = objects[objects['is_used'] == False]
unused_counts = unused.groupby('object_type').size()

plt.figure(figsize=(8, 6))
unused_counts.plot(kind='bar')
plt.xlabel('Object Type')
plt.ylabel('Count of Unused Objects')
plt.title('Unused Objects by Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## 🔗 Integration with Azure OpenAI

```python
from openai import AzureOpenAI
import json

client = AzureOpenAI(
    api_key="your-api-key",
    api_version="2024-02-01",
    azure_endpoint="your-endpoint"
)

# Get worst dataset
worst_dataset = datasets.nsmallest(1, 'optimization_score').iloc[0]

prompt = f"""
Analyze this Power BI dataset and suggest optimizations:

Dataset: {worst_dataset['dataset_name']}
Optimization Score: {worst_dataset['optimization_score']}/100
Total Tables: {worst_dataset['total_tables']}
Unused Columns: {worst_dataset['unused_columns']}
Isolated Tables: {worst_dataset['isolated_tables']}
Relationship Health: {worst_dataset['relationship_health']:.2f}

Provide 3 specific, actionable recommendations.
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

## ⚡ Performance Tips

1. **Filter by dataset_id** when analyzing specific datasets
2. **Use is_used column** for fast filtering (indexed)
3. **Leverage denormalized fields** to avoid joins
4. **Cache frequently used queries** in temp tables

```python
# Cache filtered results
spark.sql("""
    CREATE OR REPLACE TEMP VIEW unused_high_priority AS
    SELECT * FROM ai_object_features
    WHERE is_used = FALSE AND table_is_isolated = TRUE
""")

# Query the cached view
high_priority = spark.sql("SELECT * FROM unused_high_priority").toPandas()
```

---

## 📝 Export for Reporting

```python
# Export to Excel
with pd.ExcelWriter('workspace_analysis.xlsx') as writer:
    datasets.to_excel(writer, sheet_name='Datasets', index=False)
    objects[objects['is_used'] == False].to_excel(
        writer, sheet_name='Unused Objects', index=False
    )

# Export to CSV
datasets.to_csv('datasets_summary.csv', index=False)
objects.to_csv('objects_detail.csv', index=False)
```

---

For complete documentation, see **AI_SCHEMA_README.md**
