"""
DAX Dependency Parser
Parses DAX expressions to identify table, column, and measure dependencies
"""

import re
from typing import Set, Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class DaxReference:
    """Represents a reference to a table, column, or measure in DAX"""
    table_name: str
    object_name: str
    object_type: str  # 'column', 'measure', 'table'
    original_reference: str
    

class DaxDependencyParser:
    """Parses DAX expressions to find dependencies"""
    
    def __init__(self):
        # Common DAX functions that reference tables/columns
        self.table_functions = {
            'FILTER', 'ALL', 'ALLEXCEPT', 'ALLSELECTED', 'VALUES', 'DISTINCT',
            'SUMMARIZE', 'ADDCOLUMNS', 'SELECTCOLUMNS', 'GROUPBY', 'CROSSJOIN',
            'UNION', 'INTERSECT', 'EXCEPT', 'NATURALINNERJOIN', 'NATURALLEFTOUTERJOIN',
            'RELATEDTABLE', 'TREATAS', 'CALCULATETABLE', 'TOPN', 'SAMPLE'
        }
        
        # Functions that typically reference columns
        self.column_functions = {
            'SUM', 'AVERAGE', 'MIN', 'MAX', 'COUNT', 'COUNTA', 'COUNTBLANK',
            'DISTINCTCOUNT', 'SELECTEDVALUE', 'HASONEVALUE', 'ISFILTERED',
            'ISCROSSFILTERED', 'RELATED', 'LOOKUPVALUE', 'EARLIER', 'EARLIEST'
        }
        
    def parse_expression(self, expression: str) -> Set[DaxReference]:
        """Parse a DAX expression and return all dependencies"""
        if not expression:
            return set()
            
        dependencies = set()
        
        # Find all table[column] references
        table_column_pattern = r"'?([A-Za-z_][A-Za-z0-9_\s]*)'?\[([A-Za-z_][A-Za-z0-9_\s]*)\]"
        matches = re.finditer(table_column_pattern, expression, re.IGNORECASE)
        
        for match in matches:
            table_name = match.group(1).strip().strip("'\"")
            object_name = match.group(2).strip()
            original_ref = match.group(0)
            
            # Determine if it's likely a column or measure based on context
            object_type = self._determine_object_type(expression, match, table_name, object_name)
            
            dependencies.add(DaxReference(
                table_name=table_name,
                object_name=object_name,
                object_type=object_type,
                original_reference=original_ref
            ))
        
        # Find standalone measure references (without table prefix)
        measure_pattern = r"\[([A-Za-z_][A-Za-z0-9_\s]*)\]"
        measure_matches = re.finditer(measure_pattern, expression, re.IGNORECASE)
        
        for match in measure_matches:
            # Skip if this is already captured as a table[column] reference
            if not any(ref.original_reference == match.group(0) for ref in dependencies):
                object_name = match.group(1).strip()
                original_ref = match.group(0)
                
                dependencies.add(DaxReference(
                    table_name="",  # No table specified, likely a measure
                    object_name=object_name,
                    object_type="measure",
                    original_reference=original_ref
                ))
        
        # Find table-only references (for table functions)
        table_pattern = r"(?:^|[^A-Za-z0-9_])'?([A-Za-z_][A-Za-z0-9_\s]*)'?(?=\s*[,\)])(?!\[)"
        table_matches = re.finditer(table_pattern, expression, re.IGNORECASE)
        
        for match in matches:
            table_name = match.group(1).strip().strip("'\"")
            
            # Check if this appears after a table function
            start_pos = max(0, match.start() - 50)  # Look back 50 characters
            context = expression[start_pos:match.start()]
            
            if any(func in context.upper() for func in self.table_functions):
                dependencies.add(DaxReference(
                    table_name=table_name,
                    object_name="",
                    object_type="table",
                    original_reference=match.group(0).strip()
                ))
        
        return dependencies
    
    def _determine_object_type(self, expression: str, match, table_name: str, object_name: str) -> str:
        """Determine if a reference is likely a column or measure based on context"""
        # Look at the context around the match
        start_pos = max(0, match.start() - 30)
        end_pos = min(len(expression), match.end() + 30)
        context = expression[start_pos:end_pos].upper()
        
        # Check for aggregation functions that typically work with columns
        if any(func in context for func in self.column_functions):
            return "column"
        
        # Check for table functions
        if any(func in context for func in self.table_functions):
            return "column"
        
        # Default to column for table[object] syntax
        return "column"
    
    def parse_calculated_column(self, expression: str, table_name: str) -> Set[DaxReference]:
        """Parse a calculated column expression"""
        dependencies = self.parse_expression(expression)
        
        # Filter out self-references to the same table (common in calculated columns)
        filtered_dependencies = set()
        for dep in dependencies:
            if dep.table_name != table_name or dep.object_type != "column":
                filtered_dependencies.add(dep)
        
        return filtered_dependencies
    
    def parse_calculated_table(self, expression: str) -> Set[DaxReference]:
        """Parse a calculated table expression"""
        return self.parse_expression(expression)
    
    def parse_measure(self, expression: str) -> Set[DaxReference]:
        """Parse a measure expression"""
        return self.parse_expression(expression)


class UsageAnalyzer:
    """Analyzes usage of tables, columns, and measures across datasets"""
    
    def __init__(self):
        self.parser = DaxDependencyParser()
        self.used_objects = {
            'tables': set(),
            'columns': set(),
            'measures': set()
        }
        
    def analyze_dataset_usage(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze usage within a single dataset"""
        dataset_id = dataset.get("id", "")
        dataset_name = dataset.get("name", "")
        
        all_dependencies = set()
        usage_by_type = {
            'measures': [],
            'calculated_columns': [],
            'calculated_tables': []
        }
        
        tables = dataset.get("tables", [])
        
        for table in tables:
            table_name = table.get("name", "")
            
            # Analyze measures
            measures = table.get("measures", [])
            for measure in measures:
                measure_name = measure.get("name", "")
                expression = measure.get("expression", "")
                
                if expression:
                    dependencies = self.parser.parse_measure(expression)
                    all_dependencies.update(dependencies)
                    
                    usage_by_type['measures'].append({
                        'table': table_name,
                        'name': measure_name,
                        'expression': expression,
                        'dependencies': [asdict(dep) for dep in dependencies]
                    })
            
            # Analyze calculated columns
            columns = table.get("columns", [])
            for column in columns:
                if column.get("columnType") == "Calculated":
                    column_name = column.get("name", "")
                    # Note: Column expressions might not be available in the API response
                    # This would need to be enhanced based on available data
                    
            # Analyze calculated table expressions
            sources = table.get("source", [])
            for source in sources:
                expression = source.get("expression", "")
                if expression:
                    dependencies = self.parser.parse_calculated_table(expression)
                    all_dependencies.update(dependencies)
                    
                    usage_by_type['calculated_tables'].append({
                        'table': table_name,
                        'expression': expression,
                        'dependencies': [asdict(dep) for dep in dependencies]
                    })
        
        # Track used objects
        for dep in all_dependencies:
            if dep.table_name:
                self.used_objects['tables'].add(f"{dataset_id}::{dep.table_name}")
                if dep.object_type == "column":
                    self.used_objects['columns'].add(f"{dataset_id}::{dep.table_name}::{dep.object_name}")
                elif dep.object_type == "measure":
                    self.used_objects['measures'].add(f"{dataset_id}::{dep.table_name}::{dep.object_name}")
            elif dep.object_type == "measure":
                # Measure without table prefix - could be in any table
                self.used_objects['measures'].add(f"{dataset_id}::*::{dep.object_name}")
        
        return {
            'dataset_id': dataset_id,
            'dataset_name': dataset_name,
            'total_dependencies': len(all_dependencies),
            'dependencies': [asdict(dep) for dep in all_dependencies],
            'usage_by_type': usage_by_type
        }
    
    def find_unused_objects(self, dataset: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Find unused tables, columns, and measures in a dataset"""
        dataset_id = dataset.get("id", "")
        unused = {
            'tables': [],
            'columns': [],
            'measures': []
        }
        
        tables = dataset.get("tables", [])
        
        for table in tables:
            table_name = table.get("name", "")
            is_hidden = table.get("isHidden", False)
            
            table_key = f"{dataset_id}::{table_name}"
            
            # Check if table is used
            if table_key not in self.used_objects['tables'] and not is_hidden:
                unused['tables'].append({
                    'table': table_name,
                    'is_hidden': is_hidden
                })
            
            # Check columns
            columns = table.get("columns", [])
            for column in columns:
                column_name = column.get("name", "")
                is_column_hidden = column.get("isHidden", False)
                column_type = column.get("columnType", "")
                
                column_key = f"{dataset_id}::{table_name}::{column_name}"
                
                if (column_key not in self.used_objects['columns'] and 
                    not is_column_hidden and 
                    column_type != "RowNumber"):  # Skip system columns
                    unused['columns'].append({
                        'table': table_name,
                        'column': column_name,
                        'data_type': column.get("dataType", ""),
                        'column_type': column_type,
                        'is_hidden': is_column_hidden
                    })
            
            # Check measures
            measures = table.get("measures", [])
            for measure in measures:
                measure_name = measure.get("name", "")
                is_measure_hidden = measure.get("isHidden", False)
                
                measure_key = f"{dataset_id}::{table_name}::{measure_name}"
                measure_key_any_table = f"{dataset_id}::*::{measure_name}"
                
                if (measure_key not in self.used_objects['measures'] and 
                    measure_key_any_table not in self.used_objects['measures'] and
                    not is_measure_hidden):
                    unused['measures'].append({
                        'table': table_name,
                        'measure': measure_name,
                        'is_hidden': is_measure_hidden
                    })
        
        return unused