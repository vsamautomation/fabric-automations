#!/usr/bin/env python
# coding: utf-8

"""
T-SQL Migration Preparation Script
Reads Power BI analysis data from lakehouse tables and prepares
structured migration specifications for AI-driven T-SQL generation.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import anthropic

# Power BI to SQL data type mapping
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


class TSQLMigrationPrep:
    """Prepares Power BI datasets for T-SQL migration"""
    
    def __init__(self, lakehouse_path: Optional[str] = None, claude_api_key: Optional[str] = None):
        """
        Initialize the migration prep tool
        
        Args:
            lakehouse_path: Path to lakehouse tables (if reading from files)
            claude_api_key: Claude API key for T-SQL generation
        """
        self.lakehouse_path = lakehouse_path
        self.claude_client = None
        
        if claude_api_key:
            self.claude_client = anthropic.Anthropic(api_key=claude_api_key)
        
        # Data containers
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
            return 'NVARCHAR(255)'
        elif 'date' in pbi_lower:
            if 'time' in pbi_lower:
                return 'DATETIME2'
            return 'DATE'
        elif 'bool' in pbi_lower:
            return 'BIT'
        else:
            return 'NVARCHAR(255)'  # Default fallback
    
    def prepare_dataset_migration(self, dataset_id: str) -> DatasetMigrationSpec:
        """
        Prepare migration specification for a single dataset
        
        Args:
            dataset_id: Dataset ID to prepare
            
        Returns:
            DatasetMigrationSpec with all migration details
        """
        print(f"\n🔄 Preparing migration spec for dataset: {dataset_id}")
        
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
                'dataset_name': dataset_tables.iloc[0]['dataset'],
                'workspace_id': dataset_tables.iloc[0]['workspace_id'],
                'workspace_name': dataset_tables.iloc[0]['workspace']
            }
        
        dataset_name = dataset_info.get('dataset_name', dataset_info.get('dataset', 'Unknown'))
        workspace_id = dataset_info.get('workspace_id', '')
        workspace_name = dataset_info.get('workspace_name', dataset_info.get('workspace', ''))
        
        # Filter to used tables only for this dataset
        used_tables_df = self.table_analysis_df[
            (self.table_analysis_df['dataset_id'] == dataset_id) &
            (self.table_analysis_df['usage'] == 'Used')
        ]
        
        # Filter to used columns for this dataset
        used_columns_df = self.column_usage_df[
            (self.column_usage_df['dataset_id'] == dataset_id) &
            (self.column_usage_df['usage'] == 'Used')
        ]
        
        # Get excluded counts
        unused_tables = self.table_analysis_df[
            (self.table_analysis_df['dataset_id'] == dataset_id) &
            (self.table_analysis_df['usage'] == 'Unused')
        ]
        
        unused_columns = self.column_usage_df[
            (self.column_usage_df['dataset_id'] == dataset_id) &
            (self.column_usage_df['usage'] == 'Unused')
        ]
        
        excluded_tables = unused_tables['table'].unique().tolist()
        excluded_columns_count = len(unused_columns)
        
        print(f"  📊 Found {len(used_tables_df)} used tables")
        print(f"  📊 Found {len(used_columns_df)} used columns")
        print(f"  ⚠️  Excluding {len(excluded_tables)} unused tables")
        print(f"  ⚠️  Excluding {excluded_columns_count} unused columns")
        
        # Build table specifications
        table_specs = []
        
        for _, table_row in used_tables_df.iterrows():
            table_name = table_row['table']
            
            # Get columns for this table
            table_columns = used_columns_df[used_columns_df['table'] == table_name]
            
            # Build column specs
            column_specs = []
            for _, col_row in table_columns.iterrows():
                column_name = col_row['column']
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
                'measures_count': int(table_row.get('measures', 0)),
                'relationships_count': int(table_row.get('relationships', 0)),
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
        
        print(f"  ✅ Migration spec prepared with {len(table_specs)} tables")
        return migration_spec
    
    def export_migration_spec_to_json(self, migration_spec: DatasetMigrationSpec, output_path: str):
        """
        Export migration spec to JSON file for AI consumption
        
        Args:
            migration_spec: Migration specification
            output_path: Path to save JSON file
        """
        print(f"\n💾 Exporting migration spec to: {output_path}")
        
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
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(spec_dict, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Migration spec exported successfully")
        print(f"  📊 {len(migration_spec.tables)} tables included")
        print(f"  📊 {sum(len(t.columns) for t in migration_spec.tables)} total columns")
    
    def generate_tsql_with_ai(self, migration_spec: DatasetMigrationSpec) -> str:
        """
        Generate T-SQL CREATE TABLE scripts using Claude AI
        
        Args:
            migration_spec: Dataset migration specification
            
        Returns:
            Generated T-SQL scripts
        """
        if not self.claude_client:
            raise ValueError("Claude API client not initialized. Provide api_key during initialization.")
        
        print(f"\n🤖 Generating T-SQL scripts using Claude AI...")
        
        # Build structured prompt
        prompt = self._build_tsql_generation_prompt(migration_spec)
        
        # Call Claude API
        response = self.claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        tsql_scripts = response.content[0].text
        
        print(f"  ✅ T-SQL scripts generated successfully")
        return tsql_scripts
    
    def _build_tsql_generation_prompt(self, migration_spec: DatasetMigrationSpec) -> str:
        """Build the prompt for Claude AI"""
        
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
                    f"[{'Active' if rel.get('is_active', True) else 'Inactive'}]"
                )
        
        prompt = f"""Generate T-SQL CREATE TABLE scripts for Power BI dataset migration.

Dataset: {migration_spec.dataset_name}
Workspace: {migration_spec.workspace_name}

IMPORTANT CONSTRAINTS:
1. Use EXACT column names as provided (case-sensitive)
2. Use the EXACT T-SQL data types specified for each column
3. Only include tables and columns listed below (unused objects already filtered out)
4. Implement FOREIGN KEY constraints based on relationships provided
5. Add indexes on foreign key columns for performance
6. Include helpful comments documenting the original Power BI context

EXCLUSIONS (already filtered - DO NOT include):
- {len(migration_spec.excluded_tables)} unused tables excluded
- {migration_spec.excluded_columns} unused columns excluded

TABLES TO CREATE:
{chr(10).join(tables_section)}

RELATIONSHIPS (for FOREIGN KEY constraints):
{chr(10).join(relationships_section) if relationships_section else '  - No relationships defined'}

OUTPUT REQUIREMENTS:
1. Generate complete CREATE TABLE statements for each table
2. Include PRIMARY KEY constraints 
3. Include FOREIGN KEY constraints based on relationships
4. Add CREATE INDEX statements for foreign key columns
5. Use proper T-SQL syntax compatible with SQL Server 2019+
6. Add comments explaining the table purpose
7. Use a consistent naming convention

Generate the complete T-SQL migration script now:"""
        
        return prompt


# Example usage function
def example_usage():
    """Example of how to use the TSQLMigrationPrep class"""
    
    # Load data from lakehouse (simulated here)
    column_usage_df = pd.DataFrame({
        'workspace_id': ['ws1', 'ws1', 'ws1'],
        'workspace': ['Workspace 1', 'Workspace 1', 'Workspace 1'],
        'dataset_id': ['ds1', 'ds1', 'ds1'],
        'dataset': ['Sales Dataset', 'Sales Dataset', 'Sales Dataset'],
        'table': ['Customers', 'Customers', 'Orders'],
        'column': ['CustomerID', 'CustomerName', 'OrderID'],
        'data_type': ['Int64', 'String', 'Int64'],
        'usage': ['Used', 'Used', 'Used']
    })
    
    table_analysis_df = pd.DataFrame({
        'workspace_id': ['ws1', 'ws1'],
        'workspace': ['Workspace 1', 'Workspace 1'],
        'dataset_id': ['ds1', 'ds1'],
        'dataset': ['Sales Dataset', 'Sales Dataset'],
        'table': ['Customers', 'Orders'],
        'measures': [5, 3],
        'relationships': [1, 1],
        'dependencies': [10, 8],
        'usage': ['Used', 'Used']
    })
    
    # Initialize
    prep = TSQLMigrationPrep(claude_api_key="your-api-key-here")
    
    # Load data
    prep.load_lakehouse_data(
        column_usage_df=column_usage_df,
        table_analysis_df=table_analysis_df
    )
    
    # Prepare migration spec
    migration_spec = prep.prepare_dataset_migration('ds1')
    
    # Export to JSON
    prep.export_migration_spec_to_json(
        migration_spec,
        output_path='migration_spec_ds1.json'
    )
    
    # Generate T-SQL (if Claude API key provided)
    # tsql_scripts = prep.generate_tsql_with_ai(migration_spec)
    # print(tsql_scripts)


if __name__ == "__main__":
    print("T-SQL Migration Preparation Tool")
    print("=" * 50)
    print("\nThis script prepares Power BI dataset analysis for T-SQL migration.")
    print("\nUsage:")
    print("  from tsql_migration_prep import TSQLMigrationPrep")
    print("  prep = TSQLMigrationPrep(claude_api_key='your-key')")
    print("  prep.load_lakehouse_data(...)")
    print("  migration_spec = prep.prepare_dataset_migration('dataset_id')")
    print("  prep.export_migration_spec_to_json(migration_spec, 'output.json')")
    print("  tsql = prep.generate_tsql_with_ai(migration_spec)")
