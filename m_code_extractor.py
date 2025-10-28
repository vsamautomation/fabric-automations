"""
M Code Extractor Module for AI Notebook
Extracts Power Query M code from dataset_expressions lakehouse table and prepares it for AI-based SQL transformation
"""

import pandas as pd
from pyspark.sql import SparkSession
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re


@dataclass
class MCodeExpression:
    """Represents a single M code expression from a dataset"""
    dataset_id: str
    dataset_name: str
    workspace_id: str
    workspace_name: str
    table_name: str
    expression: str
    expression_type: str  # e.g., 'table', 'partition', 'calculated_column'
    object_name: Optional[str] = None
    column_name: Optional[str] = None
    
    def get_context(self) -> str:
        """Get a formatted context string for AI prompts"""
        return f"""
Dataset: {self.dataset_name}
Workspace: {self.workspace_name}
Table: {self.table_name}
Expression Type: {self.expression_type}
Object: {self.object_name or 'N/A'}
"""


@dataclass
class MCodeExtractionResult:
    """Results from M code extraction"""
    dataset_id: str
    dataset_name: str
    workspace_name: str
    expressions: List[MCodeExpression]
    total_expressions: int
    tables_with_expressions: List[str]
    extraction_timestamp: str


class MCodeExtractor:
    """
    Extracts M code expressions from lakehouse tables for SQL transformation
    Integrates with AI_Notebook_Refactored.py workflow
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize M Code Extractor
        
        Args:
            spark: SparkSession instance (auto-created if not provided)
        """
        self.spark = spark if spark else SparkSession.builder.getOrCreate()
        self.expressions_df = pd.DataFrame()
        print("✅ M Code Extractor initialized")
    
    def load_expressions_from_lakehouse(self, table_name: str = "dataset_expressions") -> pd.DataFrame:
        """
        Load M code expressions from lakehouse table
        
        Args:
            table_name: Name of the lakehouse table containing expressions
            
        Returns:
            DataFrame with M code expressions
        """
        print(f"\n📥 Loading M code expressions from lakehouse table: {table_name}")
        
        try:
            # Read from lakehouse
            expressions_spark = self.spark.table(table_name)
            self.expressions_df = expressions_spark.toPandas()
            
            print(f"  ✅ Loaded {len(self.expressions_df)} expression records")
            
            # Show summary statistics
            if not self.expressions_df.empty:
                unique_datasets = self.expressions_df['dataset_id'].nunique() if 'dataset_id' in self.expressions_df.columns else 0
                unique_tables = self.expressions_df['table_name'].nunique() if 'table_name' in self.expressions_df.columns else 0
                
                print(f"  📊 Summary:")
                print(f"     - Unique datasets: {unique_datasets}")
                print(f"     - Unique tables: {unique_tables}")
            
            return self.expressions_df
            
        except Exception as e:
            print(f"  ❌ Error loading expressions from lakehouse: {e}")
            print(f"  ℹ️  Table '{table_name}' may not exist yet")
            return pd.DataFrame()
    
    def extract_by_dataset(self, dataset_id: str) -> Optional[MCodeExtractionResult]:
        """
        Extract all M code expressions for a specific dataset
        
        Args:
            dataset_id: Dataset ID to extract expressions for
            
        Returns:
            MCodeExtractionResult with all expressions for the dataset
        """
        print(f"\n🔍 Extracting M code for dataset: {dataset_id}")
        
        if self.expressions_df.empty:
            print("  ⚠️  No expressions data loaded. Call load_expressions_from_lakehouse() first")
            return None
        
        # Filter to specific dataset
        dataset_expressions = self.expressions_df[
            self.expressions_df['dataset_id'] == dataset_id
        ].copy()
        
        if dataset_expressions.empty:
            print(f"  ⚠️  No expressions found for dataset: {dataset_id}")
            return None
        
        # Get dataset metadata
        first_row = dataset_expressions.iloc[0]
        dataset_name = first_row.get('dataset_name', 'Unknown')
        workspace_name = first_row.get('workspace_name', 'Unknown')
        workspace_id = first_row.get('workspace_id', '')
        
        print(f"  📊 Dataset: {dataset_name}")
        print(f"  📊 Workspace: {workspace_name}")
        print(f"  📊 Found {len(dataset_expressions)} expressions")
        
        # Build MCodeExpression objects
        expressions = []
        for _, row in dataset_expressions.iterrows():
            expression = MCodeExpression(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                workspace_id=workspace_id,
                workspace_name=workspace_name,
                table_name=row.get('table_name', ''),
                expression=row.get('expression', ''),
                expression_type=row.get('expression_type', 'table'),
                object_name=row.get('object_name'),
                column_name=row.get('column_name')
            )
            expressions.append(expression)
        
        # Get unique tables
        tables_with_expressions = dataset_expressions['table_name'].unique().tolist()
        
        result = MCodeExtractionResult(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            workspace_name=workspace_name,
            expressions=expressions,
            total_expressions=len(expressions),
            tables_with_expressions=tables_with_expressions,
            extraction_timestamp=datetime.now().isoformat()
        )
        
        print(f"  ✅ Extracted expressions for {len(tables_with_expressions)} tables")
        
        return result
    
    def extract_by_table(self, dataset_id: str, table_name: str) -> List[MCodeExpression]:
        """
        Extract M code expressions for a specific table
        
        Args:
            dataset_id: Dataset ID
            table_name: Table name
            
        Returns:
            List of MCodeExpression objects for the table
        """
        if self.expressions_df.empty:
            print("  ⚠️  No expressions data loaded")
            return []
        
        # Filter to specific table
        table_expressions = self.expressions_df[
            (self.expressions_df['dataset_id'] == dataset_id) &
            (self.expressions_df['table_name'] == table_name)
        ].copy()
        
        if table_expressions.empty:
            return []
        
        # Build MCodeExpression objects
        expressions = []
        for _, row in table_expressions.iterrows():
            expression = MCodeExpression(
                dataset_id=dataset_id,
                dataset_name=row.get('dataset_name', ''),
                workspace_id=row.get('workspace_id', ''),
                workspace_name=row.get('workspace_name', ''),
                table_name=table_name,
                expression=row.get('expression', ''),
                expression_type=row.get('expression_type', 'table'),
                object_name=row.get('object_name'),
                column_name=row.get('column_name')
            )
            expressions.append(expression)
        
        return expressions
    
    def get_datasets_with_expressions(self) -> pd.DataFrame:
        """
        Get summary of all datasets that have M code expressions
        
        Returns:
            DataFrame with dataset summaries
        """
        if self.expressions_df.empty:
            return pd.DataFrame()
        
        # Group by dataset
        summary = self.expressions_df.groupby([
            'dataset_id', 
            'dataset_name', 
            'workspace_name'
        ]).agg({
            'table_name': 'nunique',
            'expression': 'count'
        }).reset_index()
        
        summary.columns = [
            'dataset_id', 
            'dataset_name', 
            'workspace_name', 
            'unique_tables', 
            'total_expressions'
        ]
        
        return summary
    
    def filter_expressions_by_pattern(self, 
                                     pattern: str, 
                                     dataset_id: Optional[str] = None) -> List[MCodeExpression]:
        """
        Filter expressions that match a specific pattern (useful for finding specific M operations)
        
        Args:
            pattern: Regex pattern to match in expression text
            dataset_id: Optional dataset ID to filter to
            
        Returns:
            List of matching MCodeExpression objects
        """
        if self.expressions_df.empty:
            return []
        
        # Filter dataframe
        df = self.expressions_df.copy()
        
        if dataset_id:
            df = df[df['dataset_id'] == dataset_id]
        
        # Apply regex filter
        mask = df['expression'].str.contains(pattern, case=False, na=False, regex=True)
        filtered_df = df[mask]
        
        # Build MCodeExpression objects
        expressions = []
        for _, row in filtered_df.iterrows():
            expression = MCodeExpression(
                dataset_id=row['dataset_id'],
                dataset_name=row.get('dataset_name', ''),
                workspace_id=row.get('workspace_id', ''),
                workspace_name=row.get('workspace_name', ''),
                table_name=row['table_name'],
                expression=row['expression'],
                expression_type=row.get('expression_type', 'table'),
                object_name=row.get('object_name'),
                column_name=row.get('column_name')
            )
            expressions.append(expression)
        
        return expressions
    
    def export_for_ai_processing(self, 
                                 extraction_result: MCodeExtractionResult,
                                 include_metadata: bool = True) -> Dict:
        """
        Export M code expressions in a format optimized for AI processing
        
        Args:
            extraction_result: MCodeExtractionResult to export
            include_metadata: Whether to include dataset metadata
            
        Returns:
            Dictionary formatted for AI consumption
        """
        export_data = {
            'processing_timestamp': datetime.now().isoformat(),
            'dataset_metadata': {
                'dataset_id': extraction_result.dataset_id,
                'dataset_name': extraction_result.dataset_name,
                'workspace_name': extraction_result.workspace_name,
                'total_expressions': extraction_result.total_expressions,
                'tables_count': len(extraction_result.tables_with_expressions)
            },
            'expressions_by_table': {}
        }
        
        # Group expressions by table
        for table_name in extraction_result.tables_with_expressions:
            table_expressions = [
                exp for exp in extraction_result.expressions 
                if exp.table_name == table_name
            ]
            
            export_data['expressions_by_table'][table_name] = [
                {
                    'expression_type': exp.expression_type,
                    'object_name': exp.object_name,
                    'column_name': exp.column_name,
                    'expression': exp.expression,
                    'context': exp.get_context() if include_metadata else None
                }
                for exp in table_expressions
            ]
        
        return export_data


class MCodeToSQLIntegration:
    """
    Integration module to connect M Code Extractor with AI-based SQL generation
    Works with TSQLMigrationPrep from AI_Notebook_Refactored.py
    """
    
    def __init__(self, 
                 m_extractor: MCodeExtractor,
                 tsql_prep = None):  # TSQLMigrationPrep instance
        """
        Initialize integration module
        
        Args:
            m_extractor: MCodeExtractor instance
            tsql_prep: TSQLMigrationPrep instance from AI_Notebook_Refactored
        """
        self.m_extractor = m_extractor
        self.tsql_prep = tsql_prep
        print("✅ M Code to SQL Integration initialized")
    
    def build_m_to_sql_prompt(self, 
                              m_code_expression: MCodeExpression,
                              target_table_name: str,
                              additional_context: Optional[str] = None) -> str:
        """
        Build AI prompt for M code to SQL transformation
        
        Args:
            m_code_expression: M code expression to transform
            target_table_name: Target SQL table name
            additional_context: Additional context for AI
            
        Returns:
            Formatted prompt string for AI
        """
        prompt = f"""You are an expert in Power Query M language and T-SQL. Your task is to transform the following Power Query M code into an equivalent T-SQL SELECT statement.

## CONTEXT
{m_code_expression.get_context()}

## SOURCE M CODE
```m
{m_code_expression.expression}
```

## TARGET SQL TABLE
The data is assumed to be loaded into a table named: `{target_table_name}`

## TRANSFORMATION REQUIREMENTS
1. Convert M language operations to equivalent SQL operations:
   - `Table.SelectColumns()` → `SELECT` clause
   - `Table.SelectRows()` → `WHERE` clause
   - `Table.ReplaceValue()` → `REPLACE()` function
   - `Table.AddColumn()` → Calculated column in `SELECT`
   - `Table.RenameColumns()` → Column aliases with `AS`
   - `Table.TransformColumnTypes()` → `CAST()` or schema definition
   
2. Handle M-specific patterns:
   - `[ColumnName]` references → Standard SQL column names
   - `each` keyword → SQL expressions
   - M operators (`&`, `<>`) → SQL equivalents (`+` or `CONCAT`, `<>` or `!=`)

3. Output requirements:
   - Generate valid T-SQL compatible with SQL Server 2019+ / Fabric Data Warehouse
   - Use proper quoting for column names with spaces or special characters
   - Include helpful comments explaining transformations
   - Preserve column names exactly as they appear in M code

4. Important constraints:
   - Assume source data is ALREADY loaded to the target table (don't handle web scraping, API calls, etc.)
   - Focus ONLY on transformation logic (filters, selections, calculations, etc.)
   - If M code includes data source operations (Web.BrowserContents, Excel.Workbook, etc.), note these as prerequisites

{additional_context if additional_context else ''}

## OUTPUT
Generate the complete T-SQL SELECT statement that replicates the M code transformation logic:
"""
        
        return prompt
    
    def generate_sql_from_m_code(self,
                                 dataset_id: str,
                                 table_name: Optional[str] = None,
                                 target_table_prefix: str = "stg_") -> Dict:
        """
        Generate SQL transformations for M code expressions in a dataset
        
        Args:
            dataset_id: Dataset ID to process
            table_name: Optional specific table name (if None, processes all tables)
            target_table_prefix: Prefix for target SQL table names
            
        Returns:
            Dictionary with M code and generated SQL for each table
        """
        print(f"\n🔄 Generating SQL transformations for dataset: {dataset_id}")
        
        if not self.tsql_prep or not self.tsql_prep.client:
            print("  ⚠️  AI client not initialized. Cannot generate SQL.")
            print("  ℹ️  Initialize TSQLMigrationPrep with api_key and agent_mode")
            return {}
        
        # Extract M code expressions
        extraction_result = self.m_extractor.extract_by_dataset(dataset_id)
        
        if not extraction_result:
            return {}
        
        results = {
            'dataset_id': dataset_id,
            'dataset_name': extraction_result.dataset_name,
            'workspace_name': extraction_result.workspace_name,
            'transformations': []
        }
        
        # Process each table (or specific table)
        tables_to_process = [table_name] if table_name else extraction_result.tables_with_expressions
        
        for tbl in tables_to_process:
            print(f"\n  🔹 Processing table: {tbl}")
            
            # Get expressions for this table
            table_expressions = [
                exp for exp in extraction_result.expressions 
                if exp.table_name == tbl
            ]
            
            for expr in table_expressions:
                # Build prompt
                target_table = f"{target_table_prefix}{tbl}"
                prompt = self.build_m_to_sql_prompt(expr, target_table)
                
                # Generate SQL using AI
                try:
                    if self.tsql_prep.agent_mode == "claude":
                        response = self.tsql_prep.client.messages.create(
                            model="claude-sonnet-4-5",
                            max_tokens=4000,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        generated_sql = response.content[0].text
                    
                    elif self.tsql_prep.agent_mode == "gemini":
                        from google.genai import types
                        response = self.tsql_prep.client.models.generate_content(
                            model="gemini-2.0-flash-exp",
                            contents=prompt,
                            config=types.GenerateContentConfig(temperature=0.1)
                        )
                        generated_sql = response.text
                    
                    results['transformations'].append({
                        'table_name': tbl,
                        'expression_type': expr.expression_type,
                        'object_name': expr.object_name,
                        'original_m_code': expr.expression,
                        'generated_sql': generated_sql,
                        'target_table': target_table,
                        'status': 'success'
                    })
                    
                    print(f"    ✅ Generated SQL for {expr.expression_type}")
                    
                except Exception as e:
                    print(f"    ❌ Error generating SQL: {e}")
                    results['transformations'].append({
                        'table_name': tbl,
                        'expression_type': expr.expression_type,
                        'object_name': expr.object_name,
                        'original_m_code': expr.expression,
                        'generated_sql': None,
                        'target_table': target_table,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        print(f"\n  ✅ Completed SQL generation for {len(results['transformations'])} expressions")
        
        return results


# Example usage and testing functions
def demo_m_code_extraction():
    """Demo function showing how to use the M Code Extractor"""
    
    print("="*80)
    print("🚀 M Code Extractor Demo")
    print("="*80)
    
    # Initialize extractor
    extractor = MCodeExtractor()
    
    # Load expressions from lakehouse
    extractor.load_expressions_from_lakehouse("dataset_expressions")
    
    # Get summary of datasets with expressions
    summary = extractor.get_datasets_with_expressions()
    print("\n📊 Datasets with M Code Expressions:")
    print(summary)
    
    # Extract for a specific dataset
    if not summary.empty:
        dataset_id = summary.iloc[0]['dataset_id']
        result = extractor.extract_by_dataset(dataset_id)
        
        if result:
            print(f"\n📦 Extracted {result.total_expressions} expressions")
            print(f"   Tables: {', '.join(result.tables_with_expressions)}")
    
    return extractor


if __name__ == "__main__":
    # Run demo if executed directly
    demo_m_code_extraction()
