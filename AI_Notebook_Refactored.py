#!/usr/bin/env python
# coding: utf-8

# ## AI_Notebook_Refactored
# 
# New notebook

# In[1]:


# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install -q -U semantic-link-labs google-genai anthropic typing_extensions pydantic


# In[2]:


import sempy
import sempy_labs
import sempy.fabric as fabric
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col
from google import genai
from google.genai import types
import pandas as pd
import anthropic
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime
import time


# In[3]:


# Initialize Spark session
spark = SparkSession.builder.getOrCreate()


# In[1]:


# Power BI to T-SQL data type mapping
DATATYPE_MAPPING = {
    # Integer types
    'Int64': 'BIGINT',
    'Int32': 'INT',
    'Integer': 'INT',
    'Whole Number': 'INT',

    # Decimal / Numeric types
    'Decimal': 'DECIMAL(18, 2)',
    'Fixed Decimal Number': 'DECIMAL(18, 2)',
    'Currency': 'DECIMAL(19, 4)',
    'Double': 'FLOAT',
    'Percentage': 'DECIMAL(9, 4)',  # higher precision

    # String / Text types
    'String': 'VARCHAR(255)',
    'Text': 'VARCHAR(8000)',        # Fabric DW limit before using MAX
    'Large Text': 'VARCHAR(MAX)',

    # Date / Time types
    'DateTime': 'DATETIME2(6)',     # precision 6 = 1ms accuracy
    'DateTimeZone': 'DATETIME2(6)',
    'Date': 'DATE',
    'Time': 'TIME(6)',

    # Boolean
    'Boolean': 'BIT',
    'True/False': 'BIT',

    # Binary / Other
    'Binary': 'VARBINARY(MAX)',
    'Guid': 'UNIQUEIDENTIFIER',
    'Variant': 'VARCHAR(255)',      # SQL_VARIANT not supported
    'Unknown': 'VARCHAR(255)'
}


# In[5]:


@dataclass
class ColumnSpec:
    """Column specification for T-SQL generation"""
    column_name: str
    data_type: str
    tsql_data_type: str
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    referenced_table: Optional[str] = None
    referenced_column: Optional[str] = None


@dataclass
class TableSpec:
    """Table specification for T-SQL generation"""
    table_name: str
    columns: List[ColumnSpec]
    relationships_from: List[Dict]
    relationships_to: List[Dict]
    usage_metrics: Dict


@dataclass
class DatasetMigrationSpec:
    """Complete dataset migration specification"""
    dataset_id: str
    dataset_name: str
    workspace_id: str
    workspace_name: str
    tables: List[TableSpec]
    excluded_tables: List[str]
    excluded_columns: int
    excluded_measures: int
    total_relationships: int


# In[6]:


class TSQLMigrationPrep:
    """Prepares Power BI datasets for T-SQL migration"""
    def __init__(self, lakehouse: Optional[str] = None, api_key: Optional[str] = None, agent_mode: Optional[str] = None):
        """
        Initialize the prep tool

        Args:
            lakehouse: Lakehouse id/path
            api_key: AI agent API key
            agent_mode: AI provider ('claude' or 'gemini')
        """

        self.lakehouse = lakehouse
        self.api_key = api_key
        self.agent_mode = agent_mode
        self.client = None

        if api_key and agent_mode:
            if agent_mode == "claude":
                self.client = anthropic.Anthropic(api_key=api_key)
            elif agent_mode == "gemini":
                self.client = genai.Client(api_key=api_key)
            else:
                raise ValueError("Please specify the AI provider you are using. The script supports 'claude' or 'gemini' ")


        #Data Containers
        self.column_usage_df = pd.DataFrame()
        self.table_analysis_df = pd.DataFrame()
        self.dataset_analysis_df = pd.DataFrame()
        self.relationships_df = pd.DataFrame()
        
        print("✅ T-SQL Migration Prep initialized")

            
    def load_lakehouse_data(self, 
                           column_usage_df: pd.DataFrame,
                           table_analysis_df: pd.DataFrame,
                           dataset_analysis_df: Optional[pd.DataFrame] = None,
                           relationships_df: Optional[pd.DataFrame] = None):
        """
        Load analysis data from lakehouse tables
        
        Args:
            column_usage_df: Column usage analysis with data types
            table_analysis_df: Table usage analysis
            dataset_analysis_df: Dataset-level context
            relationships_df: Table relationships
        """
        print("\n📥 Loading lakehouse analysis data...")
        
        self.column_usage_df = column_usage_df
        self.table_analysis_df = table_analysis_df
        self.dataset_analysis_df = dataset_analysis_df if dataset_analysis_df is not None else pd.DataFrame()
        self.relationships_df = relationships_df if relationships_df is not None else pd.DataFrame()
        
        print(f"  ✅ Loaded {len(self.column_usage_df)} column records")
        print(f"  ✅ Loaded {len(self.table_analysis_df)} table records")
        print(f"  ✅ Loaded {len(self.dataset_analysis_df)} dataset records")
        print(f"  ✅ Loaded {len(self.relationships_df)} relationship records")

    
    def map_datatype_to_tsql(self, pbi_datatype: str) -> str:
        """
        Map Power BI data type to T-SQL data type
        
        Args:
            pbi_datatype: Power BI data type
            
        Returns:
            T-SQL data type
        """
        # Clean the datatype string
        pbi_datatype = str(pbi_datatype).strip()
        
        # Direct mapping
        if pbi_datatype in DATATYPE_MAPPING:
            return DATATYPE_MAPPING[pbi_datatype]
        
        # Partial matching for complex types
        pbi_lower = pbi_datatype.lower()
        
        if 'int' in pbi_lower or 'whole' in pbi_lower:
            return 'INT'
        elif 'decimal' in pbi_lower or 'number' in pbi_lower or 'currency' in pbi_lower:
            return 'DECIMAL(18, 2)'
        elif 'double' in pbi_lower or 'float' in pbi_lower:
            return 'FLOAT'
        elif 'text' in pbi_lower or 'string' in pbi_lower:
            return 'VARCHAR(255)'
        elif 'date' in pbi_lower:
            if 'time' in pbi_lower:
                return 'DATETIME2'
            return 'DATE'
        elif 'bool' in pbi_lower:
            return 'BIT'
        else:
            return 'VARCHAR(255)'  # Default fallback

    
    def prepare_dataset_migration(self, dataset_id: str) -> DatasetMigrationSpec:
        """
        Prepare specification for a single dataset
        
        Args:
            dataset_id: Dataset ID to prepare
            
        Returns:
            DatasetSpec with all details
        """
        print(f"\n🔄 Preparing specs for dataset: {dataset_id}")

                
        # Get dataset info
        if not self.dataset_analysis_df.empty:
            dataset_row = self.dataset_analysis_df[self.dataset_analysis_df['dataset_id'] == dataset_id]
            if dataset_row.empty:
                raise ValueError(f"Dataset {dataset_id} not found in dataset_analysis")
            dataset_info = dataset_row.iloc[0]
        else:
            # Try to get from table_analysis_df
            dataset_tables = self.table_analysis_df[self.table_analysis_df['dataset_id'] == dataset_id]
            if dataset_tables.empty:
                raise ValueError(f"Dataset {dataset_id} not found")
            dataset_info = {
                'dataset_name': dataset_tables.iloc[0]['dataset_name'],
                'workspace_id': dataset_tables.iloc[0]['workspace_id'],
                'workspace_name': dataset_tables.iloc[0]['workspace_name']
            }

                
        dataset_name = dataset_info.get('dataset_name', dataset_info.get('dataset', 'Unknown'))
        workspace_id = dataset_info.get('workspace_id', '')
        workspace_name = dataset_info.get('workspace_name', dataset_info.get('workspace', ''))

                
        # Filter to used tables only for this dataset
        used_tables_df = self.table_analysis_df[
            (self.table_analysis_df['dataset_id'] == dataset_id) &
            (self.table_analysis_df['is_used'] == True)
        ]
        
        # Filter to used columns for this dataset
        used_columns_df = self.column_usage_df[
            (self.column_usage_df['dataset_id'] == dataset_id) &
            (self.column_usage_df['is_used'] == True)
        ]
        
        # Get excluded counts
        unused_tables = self.table_analysis_df[
            (self.table_analysis_df['dataset_id'] == dataset_id) &
            (self.table_analysis_df['is_used'] == False)
        ]
        
        unused_columns = self.column_usage_df[
            (self.column_usage_df['dataset_id'] == dataset_id) &
            (self.column_usage_df['is_used'] == False)
        ]
        
        excluded_tables = unused_tables['table_name'].unique().tolist()
        excluded_columns_count = len(unused_columns)
        
        print(f"  📊 Found {len(used_tables_df)} used tables")
        print(f"  📊 Found {len(used_columns_df)} used columns")
        print(f"  ⚠️  Excluding {len(excluded_tables)} unused tables")
        print(f"  ⚠️  Excluding {excluded_columns_count} unused columns")

                
        # Build table specifications
        table_specs = []

        for _, table_row in used_tables_df.iterrows():
            table_name = table_row['table_name']

            # Get columns for this table
            table_columns = used_columns_df[used_columns_df['table_name'] == table_name]

                        
            # Build column specs
            column_specs = []
            for _, col_row in table_columns.iterrows():
                column_name = col_row['object_name']
                pbi_datatype = col_row.get('data_type', 'Unknown')
                tsql_datatype = self.map_datatype_to_tsql(pbi_datatype)

                                
                # Determine if column is in a relationship
                is_fk = False
                referenced_table = None
                referenced_column = None

                                
                if not self.relationships_df.empty:
                    # Check if this column is a foreign key
                    fk_rels = self.relationships_df[
                        (self.relationships_df['from_table'] == table_name) &
                        (self.relationships_df['from_column'] == column_name)
                    ]
                    
                    if not fk_rels.empty:
                        is_fk = True
                        rel = fk_rels.iloc[0]
                        referenced_table = rel.get('to_table', '')
                        referenced_column = rel.get('to_column', '')

                                    
                column_spec = ColumnSpec(
                    column_name=column_name,
                    data_type=pbi_datatype,
                    tsql_data_type=tsql_datatype,
                    is_nullable=True,  # Default to nullable
                    is_primary_key=False,  # Would need additional logic
                    is_foreign_key=is_fk,
                    referenced_table=referenced_table,
                    referenced_column=referenced_column
                )

                column_specs.append(column_spec)

                            
            # Get relationships for this table
            relationships_from = []
            relationships_to = []
            
            if not self.relationships_df.empty:
                # Relationships where this table is the FROM side
                rels_from = self.relationships_df[
                    self.relationships_df['from_table'] == table_name
                ]
                relationships_from = rels_from.to_dict('records')
                
                # Relationships where this table is the TO side
                rels_to = self.relationships_df[
                    self.relationships_df['to_table'] == table_name
                ]
                relationships_to = rels_to.to_dict('records')
            
            # Usage metrics
            usage_metrics = {
                'measures_count': int(table_row.get('table_measure_count', 0)),
                'relationships_count': int(table_row.get('table_relationship_count', 0)),
                'dependencies_count': int(table_row.get('dependencies', 0))
            }
            
            table_spec = TableSpec(
                table_name=table_name,
                columns=column_specs,
                relationships_from=relationships_from,
                relationships_to=relationships_to,
                usage_metrics=usage_metrics
            )
            
            table_specs.append(table_spec)

        # Total relationship count for dataset
        total_relationships = len(self.relationships_df[
            self.relationships_df['dataset_id'] == dataset_id
        ]) if not self.relationships_df.empty else 0

                
        print(f"  ✅ Migration spec prepared with {len(table_specs)} tables")

                
        migration_spec = DatasetMigrationSpec(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            workspace_id=workspace_id,
            workspace_name=workspace_name,
            tables=table_specs,
            excluded_tables=excluded_tables,
            excluded_columns=excluded_columns_count,
            excluded_measures=0,  # Would need measure data
            total_relationships=total_relationships
        )

        return migration_spec

            
    def export_migration_spec_to_json(self, migration_spec: DatasetMigrationSpec, output_path: str = ''):
        """
        Export migration spec to JSON file for AI consumption
        
        Args:
            migration_spec: Migration specification
            output_path: Path to save JSON file
        """
        
        # Convert to dict
        spec_dict = {
            'dataset_metadata': {
                'dataset_id': migration_spec.dataset_id,
                'dataset_name': migration_spec.dataset_name,
                'workspace_id': migration_spec.workspace_id,
                'workspace_name': migration_spec.workspace_name
            },
            'tables': [
                {
                    'table_name': table.table_name,
                    'columns': [
                        {
                            'column_name': col.column_name,
                            'original_data_type': col.data_type,
                            'tsql_data_type': col.tsql_data_type,
                            'is_nullable': col.is_nullable,
                            'is_primary_key': col.is_primary_key,
                            'is_foreign_key': col.is_foreign_key,
                            'referenced_table': col.referenced_table,
                            'referenced_column': col.referenced_column
                        }
                        for col in table.columns
                    ],
                    'relationships_from': table.relationships_from,
                    'relationships_to': table.relationships_to,
                    'usage_metrics': table.usage_metrics
                }
                for table in migration_spec.tables
            ],
            'exclusions': {
                'excluded_tables': migration_spec.excluded_tables,
                'excluded_columns_count': migration_spec.excluded_columns,
                'excluded_measures_count': migration_spec.excluded_measures
            },
            'metadata': {
                'total_relationships': migration_spec.total_relationships,
                'exported_at': datetime.now().isoformat(),
                'purpose': 'T-SQL CREATE TABLE generation for dataset migration'
            }
        }

        return spec_dict
    
    def generate_tsql_with_ai(self, migration_spec: DatasetMigrationSpec) -> str:
        """
        Generate T-SQL CREATE TABLE scripts using AI
        
        Args:
            migration_spec: Dataset migration specification
            
        Returns:
            Generated T-SQL scripts
        """
        if not self.client:
            raise ValueError("AI client not initialized. Provide api_key during initialization.")

        
        # Build structured prompt
        prompt = self._build_tsql_generation_prompt(migration_spec)
        
        if self.agent_mode == "claude":
            # Call Claude API
            response = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=8000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            tsql_scripts = response.content[0].text

        elif self.agent_mode == "gemini":
            #Call Gemini API
            response = self.client.models.generate_content(
                        model="gemini-2.0-flash-exp", 
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.1
                        )
                    )
            tsql_scripts = response.text

        return tsql_scripts
            
    def _build_tsql_generation_prompt(self, migration_spec: DatasetMigrationSpec) -> str:
        """Build the prompt for AI"""
        
        # Format tables and columns
        tables_section = []
        for table in migration_spec.tables:
            columns_info = []
            for col in table.columns:
                fk_info = f" (FK -> {col.referenced_table}.{col.referenced_column})" if col.is_foreign_key else ""
                columns_info.append(f"  - {col.column_name}: {col.tsql_data_type}{fk_info}")
            
            tables_section.append(f"""
Table: {table.table_name}
Columns:
{chr(10).join(columns_info)}
Usage: {table.usage_metrics['measures_count']} measures, {table.usage_metrics['relationships_count']} relationships
""")
        
        # Format relationships
        relationships_section = []
        for table in migration_spec.tables:
            for rel in table.relationships_from:
                relationships_section.append(
                    f"  - {rel.get('from_table', '')}.{rel.get('from_column', '')} -> "
                    f"{rel.get('to_table', '')}.{rel.get('to_column', '')} "
                    f"[{'Active' if rel.get('active', True) else 'Inactive'}]"
                )
        
        prompt = f"""You are an expert SQL developer specializing in dimensional modeling. Your task is to generate T-SQL CREATE TABLE scripts for Power BI dataset migration.

Dataset: {migration_spec.dataset_name}
Workspace: {migration_spec.workspace_name}

IMPORTANT CONSTRAINTS:
1. Use EXACT column names as provided (case-sensitive)
2. Use the EXACT T-SQL data types specified for each column
3. Only include tables and columns listed below (unused objects already filtered out)
4. Include helpful comments documenting the original Power BI context
5. Add a comment for EVERY column explaining its purpose or original context

EXCLUSIONS (already filtered - DO NOT include):
- {len(migration_spec.excluded_tables)} unused tables excluded
- {migration_spec.excluded_columns} unused columns excluded

TABLES TO CREATE:
{chr(10).join(tables_section)}

RELATIONSHIPS (for FOREIGN KEY constraints):
{chr(10).join(relationships_section) if relationships_section else '  - No relationships defined'}

OUTPUT REQUIREMENTS:
1. Generate complete CREATE TABLE statements for each table
2. Use proper T-SQL syntax compatible with SQL Server 2019+
3. Add comments explaining the table purpose
4. Add a comment for EVERY column explaining its purpose
5. Use a consistent naming convention
6. Do NOT include PRIMARY KEY constraints
7. Do NOT include FOREIGN KEY constraints
8. Do NOT include INDEX or KEY definitions

Generate the complete T-SQL migration script now:"""
        
        return prompt


# In[9]:


def main(agent_mode: str = 'claude', 
         api_key: Optional[str] = None,
         dataset_ids: Optional[List[str]] = None,
         process_all_datasets: bool = True,
         export_json: bool = True,
         generate_tsql: bool = True,
         save_to_lakehouse: bool = True,
         lakehouse_table_name: str = "tsql_migration_results"):
    """
    Main function to execute T-SQL migration workflow for multiple datasets
    
    Args:
        agent_mode: AI provider - 'claude' or 'gemini' (default: 'claude')
        api_key: API key for the selected AI provider
        dataset_ids: List of specific dataset IDs to process (optional)
        process_all_datasets: Process all datasets in data_context (default: True)
        export_json: Whether to export migration spec to JSON (default: True)
        generate_tsql: Whether to generate T-SQL scripts with AI (default: True)
        save_to_lakehouse: Save results to lakehouse table (default: True)
        lakehouse_table_name: Name of lakehouse table to save results
    
    Returns:
        Dictionary with all_results and summary statistics
    """
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting Batch T-SQL Migration Workflow")
    print(f"   AI Provider: {agent_mode.upper()}")
    print(f"   Generate T-SQL: {'Yes' if generate_tsql and api_key else 'No'}")
    print(f"{'='*80}\n")
    
    # Step 1: Load lakehouse tables (single read - efficient)
    print("📊 Step 1: Loading lakehouse tables...")
    data_context_spark = spark.table("ai_dataset_context")
    relationships_spark = spark.table("dataset_relationships")
    objects_spark = spark.read.table("ai_object_features")
    
    # Convert to pandas once for all datasets
    data_context_pd = data_context_spark.toPandas()
    relationships_pd = relationships_spark.toPandas()
    
    print(f"  ✅ Loaded {len(data_context_pd)} datasets from context")
    
    # Step 2: Prepare table analysis DataFrame (single operation)
    print("\n🔧 Step 2: Preparing table analysis...")
    tables = objects_spark.groupBy([
        'workspace_id', 
        'workspace_name',
        'dataset_id', 
        'dataset_name', 
        'table_name'
    ]).agg(
        F.mean('usage_score').alias('usage_score'),
        F.first('table_measure_count').alias('table_measure_count'),
        F.first('table_column_count').alias('table_column_count'),
        F.first('table_relationship_count').alias('table_relationship_count'),
        F.first('table_is_isolated').alias('table_is_isolated'),
        F.first('dataset_total_tables').alias('dataset_total_tables'),
        F.first('dataset_relationship_health').alias('dataset_relationship_health'),
        F.first('dataset_usage_efficiency').alias('dataset_usage_efficiency'),
        F.sum('used_by_dependencies').alias('dependencies')
    ).withColumn(
        'is_used', 
        F.when(F.col('usage_score') > 0, True).otherwise(False)
    ).withColumn(
        'usage_score',
        F.round(F.col('usage_score'), 3)
    )
    
    tables_pd = tables.toPandas()
    
    # Step 3: Filter columns
    print("\n🔧 Step 3: Filtering column data...")
    columns = objects_spark[objects_spark['object_type'] == 'column']
    columns_pd = columns.toPandas()
    
    print(f"  ✅ Prepared {len(tables_pd)} table records")
    print(f"  ✅ Prepared {len(columns_pd)} column records")
    
    # Step 4: Determine which datasets to process
    if process_all_datasets:
        datasets_to_process = data_context_pd['dataset_id'].unique().tolist()
    elif dataset_ids:
        datasets_to_process = dataset_ids
    else:
        raise ValueError("Either set process_all_datasets=True or provide dataset_ids list")
    
    print(f"\n📋 Total datasets to process: {len(datasets_to_process)}")
    
    # Step 5: Initialize TSQLMigrationPrep (single instance for all datasets)
    print("\n⚙️  Step 4: Initializing T-SQL Migration Prep...")
    prep = TSQLMigrationPrep(api_key=api_key, agent_mode=agent_mode)
    
    # Load data once for all datasets
    prep.load_lakehouse_data(
        column_usage_df=columns_pd,
        table_analysis_df=tables_pd,
        dataset_analysis_df=data_context_pd,
        relationships_df=relationships_pd
    )
    
    # Step 6: Process each dataset
    print(f"\n{'='*80}")
    print(f"🔄 Processing {len(datasets_to_process)} datasets...")
    print(f"{'='*80}\n")
    
    all_results = []
    successful_count = 0
    failed_count = 0
    
    for idx, dataset_id in enumerate(datasets_to_process, 1):
        try:
            print(f"\n{'─'*80}")
            print(f"📦 [{idx}/{len(datasets_to_process)}] Processing Dataset: {dataset_id}")
            print(f"{'─'*80}")
            
            # Prepare migration spec
            dataset_meta = prep.prepare_dataset_migration(dataset_id)
            
            result = {
                'dataset_id': dataset_id,
                'dataset_name': dataset_meta.dataset_name,
                'workspace_name': dataset_meta.workspace_name,
                'migration_spec': dataset_meta,
                'json_spec': None,
                'tsql_scripts': None,
                'status': 'success',
                'error': None,
                'timestamp': datetime.now().isoformat(),
                'tables_count': len(dataset_meta.tables),
                'columns_count': sum(len(t.columns) for t in dataset_meta.tables)
            }
            
            # Export to JSON
            if export_json:
                json_meta = prep.export_migration_spec_to_json(dataset_meta)
                result['json_spec'] = json_meta
            
            # Generate T-SQL scripts (only if API key provided)
            if generate_tsql and api_key:
                print(f"\n🤖 Generating T-SQL scripts using {agent_mode.upper()} AI...")
                tsql_scripts = prep.generate_tsql_with_ai(dataset_meta)
                result['tsql_scripts'] = tsql_scripts
                print(f"  ✅ T-SQL scripts generated successfully")
                
                # Add small delay to respect API rate limits
                if idx < len(datasets_to_process):  # Don't delay after last dataset
                    time.sleep(1)
            
            all_results.append(result)
            successful_count += 1
            
            print(f"\n✅ Dataset {dataset_meta.dataset_name} processed successfully")
            
        except Exception as e:
            failed_count += 1
            error_result = {
                'dataset_id': dataset_id,
                'dataset_name': 'Unknown',
                'workspace_name': 'Unknown',
                'migration_spec': None,
                'json_spec': None,
                'tsql_scripts': None,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'tables_count': 0,
                'columns_count': 0
            }
            all_results.append(error_result)
            print(f"\n❌ Error processing dataset {dataset_id}: {e}")
    
    # Step 7: Save results to lakehouse (optional)
    if save_to_lakehouse and all_results:
        print(f"\n{'='*80}")
        print(f"💾 Saving results to lakehouse table: {lakehouse_table_name}")
        print(f"{'='*80}\n")
        
        # Prepare data for lakehouse
        lakehouse_data = []
        for result in all_results:
            lakehouse_data.append({
                'dataset_id': result['dataset_id'],
                'dataset_name': result['dataset_name'],
                'workspace_name': result['workspace_name'],
                'status': result['status'],
                'tables_count': result['tables_count'],
                'columns_count': result['columns_count'],
                'has_tsql': result['tsql_scripts'] is not None,
                'tsql_scripts': result['tsql_scripts'] if result['tsql_scripts'] else '',
                'error_message': result['error'] if result['error'] else '',
                'timestamp': result['timestamp']
            })
        
        # Save to lakehouse
        lakehouse_df = pd.DataFrame(lakehouse_data)
        spark_df = spark.createDataFrame(lakehouse_df)
        spark_df.write.mode("overwrite").saveAsTable(lakehouse_table_name)
        
        print(f"  ✅ Results saved to {lakehouse_table_name}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"  ✅ Total Datasets Processed: {len(datasets_to_process)}")
    print(f"  ✅ Successful: {successful_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  📊 Total Tables: {sum(r['tables_count'] for r in all_results)}")
    print(f"  📊 Total Columns: {sum(r['columns_count'] for r in all_results)}")
    if generate_tsql and api_key:
        print(f"  🤖 T-SQL Scripts Generated: {sum(1 for r in all_results if r['tsql_scripts'] is not None)}")
    print(f"{'='*80}\n")
    
    return {
        'all_results': all_results,
        'summary': {
            'total_datasets': len(datasets_to_process),
            'successful': successful_count,
            'failed': failed_count,
            'total_tables': sum(r['tables_count'] for r in all_results),
            'total_columns': sum(r['columns_count'] for r in all_results)
        }
    }


# ## Usage Examples
# 
# ### Example 1: Process ALL datasets with Gemini
# ```python
# gemini_key = "your-gemini-api-key"
# 
# results = main(
#     agent_mode='gemini',
#     api_key=gemini_key,
#     process_all_datasets=True,
#     generate_tsql=True,
#     save_to_lakehouse=True
# )
# ```
# 
# ### Example 2: Process ALL datasets with Claude
# ```python
# claude_key = "your-claude-api-key"
# 
# results = main(
#     agent_mode='claude',
#     api_key=claude_key,
#     process_all_datasets=True,
#     generate_tsql=True
# )
# ```
# 
# ### Example 3: Process SPECIFIC datasets only
# ```python
# specific_datasets = [
#     "5fef939e-8bd0-40e1-a0c5-a7a9a49094d1",
#     "another-dataset-id"
# ]
# 
# results = main(
#     agent_mode='gemini',
#     api_key=gemini_key,
#     dataset_ids=specific_datasets,
#     process_all_datasets=False
# )
# ```
# 
# ### Example 4: Only prepare specs WITHOUT AI generation
# ```python
# results = main(
#     process_all_datasets=True,
#     generate_tsql=False,  # No AI calls = No cost
#     save_to_lakehouse=True
# )
# ```
# 
# ### Accessing Results
# ```python
# # Get all results
# all_results = results['all_results']
# 
# # Get specific dataset result
# dataset_result = all_results[0]
# print(dataset_result['tsql_scripts'])
# 
# # Get summary statistics
# summary = results['summary']
# print(f"Processed {summary['successful']} datasets successfully")
# 
# # Read results from lakehouse
# saved_results = spark.table("tsql_migration_results")
# display(saved_results)
# ```
# 

# In[14]:


# Configuration - CHANGE THESE VALUES
gemini_key = "your-gemini-api-key-here"
claude_key = "your-claude-api-key-here"

# Run for ALL datasets with Gemini (recommended for cost efficiency)
results = main(
    agent_mode='claude',      # Change to 'claude' to use Claude
    api_key=claude_key,       # Change to claude_key for Claude
    process_all_datasets=True,
    export_json=True,
    generate_tsql=True,
    save_to_lakehouse=True,
    lakehouse_table_name="tsql_migration_results"
)


# In[15]:


# View summary
print("\n📊 Summary:")
print(results['summary'])


# In[16]:


# View specific dataset result
if results['all_results']:
    first_result = results['all_results'][0]
    print(f"\nDataset: {first_result['dataset_name']}")
    print(f"Status: {first_result['status']}")
    print(f"Tables: {first_result['tables_count']}")
    print(f"Columns: {first_result['columns_count']}")
    
    if first_result['tsql_scripts']:
        print("\n📜 T-SQL Scripts:")
        print(first_result['tsql_scripts'][:500] + "...")  # First 500 chars


# In[17]:


# Read saved results from lakehouse
saved_results = spark.table("tsql_migration_results")
display(saved_results)

