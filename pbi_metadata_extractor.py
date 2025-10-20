"""
Power BI Report Metadata Extractor Module
=========================================
This module extracts columns, tables, and measures from Power BI report metadata JSON.
"""

import json
import pandas as pd
from typing import Dict, List, Set, Any
from collections import defaultdict

class PowerBIMetadataExtractor:
    """Extracts columns, tables, and measures from Power BI report metadata"""
    
    def __init__(self):
        self.tables = set()
        self.columns = set()
        self.measures = set()
        self.visual_details = []
        self.filter_details = []
        
    def extract_from_json_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from JSON data"""
        self._reset()
        
        # Extract from sections
        sections = data.get('sections', [])
        
        for section_idx, section in enumerate(sections):
            section_name = section.get('displayName', f'Section_{section_idx}')
            
            # Extract from section-level filters
            self._extract_from_filters(section.get('filters', []), 'section', section_name)
            
            # Extract from visual containers
            visual_containers = section.get('visualContainers', [])
            self._extract_from_visual_containers(visual_containers, section_name)
        
        # Compile results
        results = {
            'tables': sorted(list(self.tables)),
            'columns': sorted(list(self.columns)),
            'measures': sorted(list(self.measures)),
            'summary': {
                'total_tables': len(self.tables),
                'total_columns': len(self.columns),
                'total_measures': len(self.measures)
            },
            'visual_details': self.visual_details,
            'filter_details': self.filter_details
        }
        
        return results
    
    def _reset(self):
        """Reset all collections for new extraction"""
        self.tables.clear()
        self.columns.clear()
        self.measures.clear()
        self.visual_details.clear()
        self.filter_details.clear()
    
    def _extract_from_visual_containers(self, visual_containers: List[Dict], section_name: str):
        """Extract from visualContainers array"""
        for visual_idx, visual_container in enumerate(visual_containers):
            visual_config = visual_container.get('config', {})
            visual_name = visual_config.get('name', f'Visual_{visual_idx}')
            
            # Extract from visual-level filters
            self._extract_from_filters(
                visual_container.get('filters', []), 
                'visual', 
                f"{section_name}->{visual_name}"
            )
            
            # Extract from singleVisual
            single_visual = visual_config.get('singleVisual', {})
            if single_visual:
                self._extract_from_single_visual(single_visual, section_name, visual_name)
    
    def _extract_from_single_visual(self, single_visual: Dict, section_name: str, visual_name: str):
        """Extract from singleVisual object"""
        visual_type = single_visual.get('visualType', 'unknown')
        
        # Extract from projections
        projections = single_visual.get('projections', {})
        projection_refs = []
        
        for projection_type, projection_list in projections.items():
            for proj in projection_list:
                query_ref = proj.get('queryRef', '')
                if query_ref:
                    projection_refs.append(query_ref)
                    self._parse_query_ref(query_ref)
        
        # Extract from prototypeQuery
        prototype_query = single_visual.get('prototypeQuery', {})
        self._extract_from_prototype_query(prototype_query)
        
        # Store visual details
        self.visual_details.append({
            'section': section_name,
            'visual_name': visual_name,
            'visual_type': visual_type,
            'projection_refs': projection_refs,
            'has_prototype_query': bool(prototype_query)
        })
    
    def _extract_from_prototype_query(self, prototype_query: Dict):
        """Extract from prototypeQuery object"""
        # Extract tables from 'From' clause
        from_clause = prototype_query.get('From', [])
        for from_item in from_clause:
            entity = from_item.get('Entity', '')
            if entity:
                self.tables.add(entity)
        
        # Extract columns and measures from 'Select' clause
        select_clause = prototype_query.get('Select', [])
        for select_item in select_clause:
            name = select_item.get('Name', '')
            
            # Check if it's a Column
            if 'Column' in select_item:
                column_property = select_item['Column'].get('Property', '')
                if column_property and name:
                    self.columns.add(name)  # Store full reference (table.column)
                    # Also extract table name
                    if '.' in name:
                        table_name = name.split('.')[0]
                        self.tables.add(table_name)
            
            # Check if it's a Measure
            elif 'Measure' in select_item:
                measure_property = select_item['Measure'].get('Property', '')
                if measure_property and name:
                    self.measures.add(name)  # Store full reference (table.measure)
                    # Also extract table name
                    if '.' in name:
                        table_name = name.split('.')[0]
                        self.tables.add(table_name)
    
    def _extract_from_filters(self, filters: List[Dict], filter_type: str, context: str):
        """Extract from filters array"""
        for filter_idx, filter_obj in enumerate(filters):
            filter_name = filter_obj.get('name', f'Filter_{filter_idx}')
            
            # Extract from expression
            expression = filter_obj.get('expression', {})
            self._extract_from_expression(expression)
            
            # Extract from filter object (nested structure)
            filter_def = filter_obj.get('filter', {})
            if filter_def:
                # Extract tables from 'From' clause in filter
                from_clause = filter_def.get('From', [])
                for from_item in from_clause:
                    entity = from_item.get('Entity', '')
                    if entity:
                        self.tables.add(entity)
                
                # Extract from 'Where' clause - might contain column references
                where_clause = filter_def.get('Where', [])
                for where_item in where_clause:
                    self._extract_from_where_condition(where_item)
            
            # Store filter details
            self.filter_details.append({
                'filter_type': filter_type,
                'context': context,
                'filter_name': filter_name,
                'has_expression': bool(expression),
                'has_filter_def': bool(filter_def)
            })
    
    def _extract_from_expression(self, expression: Dict):
        """Extract from expression object"""
        if 'Column' in expression:
            # Extract table from SourceRef
            column_expr = expression['Column']
            source_ref = column_expr.get('Expression', {}).get('SourceRef', {})
            entity = source_ref.get('Entity', '')
            if entity:
                self.tables.add(entity)
            
            # Extract column property
            property_name = column_expr.get('Property', '')
            if property_name and entity:
                self.columns.add(f"{entity}.{property_name}")
    
    def _extract_from_where_condition(self, where_item: Dict):
        """Extract from WHERE condition"""
        condition = where_item.get('Condition', {})
        if 'In' in condition:
            expressions = condition['In'].get('Expressions', [])
            for expr in expressions:
                self._extract_from_expression(expr)
    
    def _parse_query_ref(self, query_ref: str):
        """Parse queryRef format (e.g., 'table.column' or 'table.measure')"""
        if '.' in query_ref:
            table_name, field_name = query_ref.split('.', 1)
            self.tables.add(table_name)
            # We'll determine if it's a column or measure from prototype query
            # For now, just store the full reference

    def create_dataframe_summaries(self, results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Create pandas DataFrame summaries from extraction results"""
        dataframes = {}
        
        # Tables DataFrame
        if results['tables']:
            tables_df = pd.DataFrame({
                'Table': results['tables'],
                'Type': 'Table'
            })
            dataframes['tables'] = tables_df
        
        # Columns DataFrame
        if results['columns']:
            columns_data = []
            for column in results['columns']:
                if '.' in column:
                    table, col_name = column.split('.', 1)
                    columns_data.append({'Table': table, 'Column': col_name, 'Full_Reference': column})
                else:
                    columns_data.append({'Table': '', 'Column': column, 'Full_Reference': column})
            
            columns_df = pd.DataFrame(columns_data)
            dataframes['columns'] = columns_df
        
        # Measures DataFrame
        if results['measures']:
            measures_data = []
            for measure in results['measures']:
                if '.' in measure:
                    table, measure_name = measure.split('.', 1)
                    measures_data.append({'Table': table, 'Measure': measure_name, 'Full_Reference': measure})
                else:
                    measures_data.append({'Table': '', 'Measure': measure, 'Full_Reference': measure})
            
            measures_df = pd.DataFrame(measures_data)
            dataframes['measures'] = measures_df
        
        return dataframes

    def get_metadata_counts(self, results: Dict[str, Any]) -> Dict[str, int]:
        """Get counts of extracted metadata"""
        return {
            'tables_count': len(results.get('tables', [])),
            'columns_count': len(results.get('columns', [])),
            'measures_count': len(results.get('measures', [])),
            'visuals_count': len(results.get('visual_details', [])),
            'filters_count': len(results.get('filter_details', []))
        }