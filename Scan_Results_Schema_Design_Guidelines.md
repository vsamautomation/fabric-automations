
# Scan Results Schema Design Guidelines

### Description  
This document describes the most optimal way of saving scan results for RAG  Analysis using AI.  
The goal is to store metadata about Microsoft Fabric workspaces, datasets, tables, and columns in a structured format that allows AI models to analyze dependencies and suggest optimized schema designs.  

AI can use this data to:  
- Identify unused datasets, tables, and columns.  
- Recommend a **Star Schema** model with clear separation of **Fact** and **Dimension** tables.  
- Suggest optimized **stored procedures (SPROCs)** for data loading and transformations.  
- Detect redundant relationships and unused measures.  
- Suggest improved relationships and indexing strategies.  

---

## Major Tables and Their Columns

### 1. **Workspaces**
| Column Name | Description |
|--------------|-------------|
| workspace_id | Unique ID of the workspace in Fabric |
| workspace_name | Human-readable name of the workspace |
| created_date | Date the workspace was created |
| last_scan_date | Last time the workspace was analyzed |

---

### 2. **Datasets**
| Column Name | Description |
|--------------|-------------|
| dataset_id | Unique identifier for the dataset |
| workspace_id | Foreign key to the Workspaces table |
| dataset_name | Name of the dataset |
| description | Description or purpose of the dataset |
| created_date | Date the dataset was created |
| last_refreshed | Date the dataset was last refreshed |
| usage_count | Number of reports or queries using the dataset |
| last_used_date | Last time the dataset was used |

---

### 3. **Tables**
| Column Name | Description |
|--------------|-------------|
| table_id | Unique identifier for the table |
| dataset_id | Foreign key to the Datasets table |
| table_name | Table name within the dataset |
| table_type | Fact, Dimension, or Reference table |
| record_count | Approximate number of rows |
| storage_size_mb | Estimated size of the table |
| usage_count | Number of queries or measures referencing this table |
| last_used_date | Last time the table was used |

---

### 4. **Columns**
| Column Name | Description |
|--------------|-------------|
| column_id | Unique identifier for the column |
| table_id | Foreign key to the Tables table |
| column_name | Name of the column |
| data_type | Data type (e.g., INT, VARCHAR, DATE) |
| is_primary_key | Boolean indicating if this is a primary key |
| is_foreign_key | Boolean indicating if this is a foreign key |
| usage_count | Number of measures, visuals, or relationships using this column |
| last_used_date | Last time the column was used |
| referenced_by | List of objects referencing this column (measures, visuals, etc.) |

---

### 5. **Model Dependencies**
| Column Name | Description |
|--------------|-------------|
| dependency_id | Unique identifier for the dependency |
| source_object | The object where the dependency originates (e.g., measure or column) |
| target_object | The object being referenced |
| dependency_type | Type of dependency (e.g., relationship, calculation, measure, hierarchy) |
| workspace_id | Foreign key to Workspaces |
| dataset_id | Foreign key to Datasets |
| last_used_date | Last time this dependency was observed |

---

### 6. **Reports and Visuals**
| Column Name | Description |
|--------------|-------------|
| report_id | Unique identifier for the report |
| workspace_id | Foreign key to the Workspaces table |
| dataset_id | Dataset used in the report |
| report_name | Name of the report |
| visual_count | Number of visuals in the report |
| used_tables | List of tables used in visuals |
| used_columns | List of columns used in visuals |
| last_used_date | Last time the report was accessed |

---

### 7. **Relationships**
| Column Name | Description |
|--------------|-------------|
| relationship_id | Unique identifier for the relationship |
| dataset_id | Foreign key to the Datasets table |
| from_table | Source table name |
| from_column | Source column name |
| to_table | Target table name |
| to_column | Target column name |
| relationship_type | Type of relationship (One-to-Many, Many-to-One) |
| active | Boolean flag for whether the relationship is active |
| last_used_date | Last time the relationship was used |

---

### Notes
- This schema ensures full traceability from **Workspaces → Datasets → Tables → Columns → Dependencies**.  
- It allows AI to identify which objects are **critical**, **redundant**, or **unused**.  
- It supports schema optimization for **Star Schema** modeling, where **Fact Tables** are automatically identified by:  
  - High `record_count`  
  - Many `foreign_key` references  
  - Frequent usage in **measures or visuals**  

---

### Example Use Case
Once scan data is saved using this schema, AI can automatically:  
1. Analyze dependencies to detect unused objects.  
2. Generate a proposed **Star Schema** layout.  
3. Create SQL scripts or SPROCs for data loading and transformations.  
4. Provide visual recommendations for new data models.
