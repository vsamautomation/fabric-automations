"""
M Code to SQL Transformer
Parses Power Query M language and generates equivalent SQL SELECT statements
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TransformationStep:
    """Represents a single M code transformation step"""
    step_name: str
    operation: str
    parameters: Dict
    sql_fragment: Optional[str] = None


@dataclass
class SQLQuery:
    """Represents the generated SQL query components"""
    select_columns: List[str] = field(default_factory=list)
    from_table: str = ""
    where_conditions: List[str] = field(default_factory=list)
    joins: List[str] = field(default_factory=list)
    column_transformations: Dict[str, str] = field(default_factory=dict)
    
    def to_sql(self) -> str:
        """Generate final SQL query"""
        lines = []
        
        # SELECT clause
        if self.column_transformations:
            select_items = []
            for col in self.select_columns:
                if col in self.column_transformations:
                    select_items.append(f"    {self.column_transformations[col]} AS {col}")
                else:
                    select_items.append(f"    {col}")
            lines.append("SELECT")
            lines.append(",\n".join(select_items))
        else:
            lines.append("SELECT " + ", ".join(self.select_columns))
        
        # FROM clause
        lines.append(f"FROM {self.from_table}")
        
        # JOIN clauses
        for join in self.joins:
            lines.append(join)
        
        # WHERE clause
        if self.where_conditions:
            lines.append("WHERE " + self.where_conditions[0])
            for condition in self.where_conditions[1:]:
                lines.append("    AND " + condition)
        
        return "\n".join(lines)


class MCodeParser:
    """Parses M code expressions into transformation steps"""
    
    def __init__(self):
        self.steps: List[TransformationStep] = []
        self.variable_map: Dict[str, str] = {}
    
    def parse_m_code(self, m_code: str) -> List[TransformationStep]:
        """Parse M code into transformation steps"""
        # Remove 'let' and 'in' keywords
        m_code = re.sub(r'^\s*let\s+', '', m_code, flags=re.IGNORECASE)
        m_code = re.sub(r'\s+in\s+\w+\s*$', '', m_code, flags=re.IGNORECASE)
        
        # Split into individual steps
        steps = self._split_steps(m_code)
        
        for step_text in steps:
            step = self._parse_step(step_text)
            if step:
                self.steps.append(step)
        
        return self.steps
    
    def _split_steps(self, m_code: str) -> List[str]:
        """Split M code into individual transformation steps"""
        # Match pattern: VariableName = Expression,
        pattern = r'(#?"[^"]+"|[\w]+)\s*=\s*([^,]+(?:,\s*\{[^}]+\})?)'
        matches = re.findall(pattern, m_code, re.DOTALL)
        
        steps = []
        for var_name, expression in matches:
            var_name = var_name.strip().strip('"').strip('#"')
            expression = expression.strip().rstrip(',')
            steps.append(f"{var_name} = {expression}")
        
        return steps
    
    def _parse_step(self, step_text: str) -> Optional[TransformationStep]:
        """Parse a single transformation step"""
        match = re.match(r'([^=]+)=\s*(.+)', step_text, re.DOTALL)
        if not match:
            return None
        
        step_name = match.group(1).strip().strip('"').strip('#"')
        expression = match.group(2).strip()
        
        # Identify operation type
        if 'Table.SelectColumns' in expression:
            return self._parse_select_columns(step_name, expression)
        elif 'Table.ReplaceValue' in expression:
            return self._parse_replace_value(step_name, expression)
        elif 'Table.SelectRows' in expression:
            return self._parse_select_rows(step_name, expression)
        elif 'Table.TransformColumnTypes' in expression:
            return self._parse_transform_types(step_name, expression)
        elif 'Table.RemoveColumns' in expression:
            return self._parse_remove_columns(step_name, expression)
        elif 'Table.RenameColumns' in expression:
            return self._parse_rename_columns(step_name, expression)
        elif 'Table.AddColumn' in expression:
            return self._parse_add_column(step_name, expression)
        elif any(src in expression for src in ['Sql.Database', 'Excel.Workbook', 'Csv.Document']):
            return self._parse_source(step_name, expression)
        else:
            # Generic operation
            return TransformationStep(step_name, 'unknown', {'expression': expression})
    
    def _parse_source(self, step_name: str, expression: str) -> TransformationStep:
        """Parse source operations"""
        if 'Sql.Database' in expression:
            match = re.search(r'Sql\.Database\("([^"]+)",\s*"([^"]+)"', expression)
            if match:
                server, database = match.groups()
                return TransformationStep(
                    step_name, 'source',
                    {'type': 'sql', 'server': server, 'database': database}
                )
        return TransformationStep(step_name, 'source', {'type': 'generic', 'expression': expression})
    
    def _parse_select_columns(self, step_name: str, expression: str) -> TransformationStep:
        """Parse Table.SelectColumns operation"""
        # Extract column list
        match = re.search(r'\{([^}]+)\}', expression)
        if match:
            columns_str = match.group(1)
            columns = [col.strip().strip('"') for col in columns_str.split(',')]
            return TransformationStep(
                step_name, 'select_columns',
                {'columns': columns}
            )
        return TransformationStep(step_name, 'select_columns', {})
    
    def _parse_replace_value(self, step_name: str, expression: str) -> TransformationStep:
        """Parse Table.ReplaceValue operation"""
        # Extract: old_value, new_value, replacer_type, columns
        pattern = r'Table\.ReplaceValue\([^,]+,\s*"([^"]*)",\s*"([^"]*)",\s*([^,]+),\s*\{([^}]+)\}'
        match = re.search(pattern, expression)
        
        if match:
            old_val, new_val, replacer_type, columns_str = match.groups()
            columns = [col.strip().strip('"') for col in columns_str.split(',')]
            
            return TransformationStep(
                step_name, 'replace_value',
                {
                    'old_value': old_val,
                    'new_value': new_val,
                    'replacer_type': replacer_type,
                    'columns': columns
                }
            )
        return TransformationStep(step_name, 'replace_value', {})
    
    def _parse_select_rows(self, step_name: str, expression: str) -> TransformationStep:
        """Parse Table.SelectRows (filtering) operation"""
        # Extract the filter condition
        match = re.search(r'each\s+\((.+?)\)(?:\s*\))?$', expression, re.DOTALL)
        
        if match:
            condition = match.group(1).strip()
            return TransformationStep(
                step_name, 'select_rows',
                {'condition': condition}
            )
        return TransformationStep(step_name, 'select_rows', {})
    
    def _parse_transform_types(self, step_name: str, expression: str) -> TransformationStep:
        """Parse Table.TransformColumnTypes operation"""
        # Extract type transformations
        match = re.search(r'\{\{([^}]+)\}\}', expression)
        if match:
            types_str = match.group(1)
            # Parse pairs of {"column", type}
            type_map = {}
            pairs = re.findall(r'\{"([^"]+)",\s*([^}]+)\}', types_str)
            for col, dtype in pairs:
                type_map[col] = dtype.strip()
            
            return TransformationStep(
                step_name, 'transform_types',
                {'type_map': type_map}
            )
        return TransformationStep(step_name, 'transform_types', {})
    
    def _parse_remove_columns(self, step_name: str, expression: str) -> TransformationStep:
        """Parse Table.RemoveColumns operation"""
        match = re.search(r'\{([^}]+)\}', expression)
        if match:
            columns = [col.strip().strip('"') for col in match.group(1).split(',')]
            return TransformationStep(step_name, 'remove_columns', {'columns': columns})
        return TransformationStep(step_name, 'remove_columns', {})
    
    def _parse_rename_columns(self, step_name: str, expression: str) -> TransformationStep:
        """Parse Table.RenameColumns operation"""
        pairs = re.findall(r'\{"([^"]+)",\s*"([^"]+)"\}', expression)
        rename_map = {old: new for old, new in pairs}
        return TransformationStep(step_name, 'rename_columns', {'rename_map': rename_map})
    
    def _parse_add_column(self, step_name: str, expression: str) -> TransformationStep:
        """Parse Table.AddColumn operation"""
        match = re.search(r'Table\.AddColumn\([^,]+,\s*"([^"]+)",\s*each\s+(.+?)\)', expression, re.DOTALL)
        if match:
            col_name, formula = match.groups()
            return TransformationStep(
                step_name, 'add_column',
                {'column': col_name, 'formula': formula.strip()}
            )
        return TransformationStep(step_name, 'add_column', {})


class SQLGenerator:
    """Generates SQL from parsed M code transformation steps"""
    
    def __init__(self, source_table: str = "source_table"):
        self.source_table = source_table
        self.query = SQLQuery()
        self.query.from_table = source_table
        self.all_columns: List[str] = []
    
    def generate_sql(self, steps: List[TransformationStep]) -> str:
        """Generate SQL from transformation steps"""
        for step in steps:
            self._process_step(step)
        
        # If no columns specified, use all
        if not self.query.select_columns:
            self.query.select_columns = self.all_columns if self.all_columns else ['*']
        
        return self.query.to_sql()
    
    def _process_step(self, step: TransformationStep):
        """Process a single transformation step"""
        if step.operation == 'source':
            self._process_source(step)
        elif step.operation == 'select_columns':
            self._process_select_columns(step)
        elif step.operation == 'replace_value':
            self._process_replace_value(step)
        elif step.operation == 'select_rows':
            self._process_select_rows(step)
        elif step.operation == 'transform_types':
            self._process_transform_types(step)
        elif step.operation == 'remove_columns':
            self._process_remove_columns(step)
        elif step.operation == 'rename_columns':
            self._process_rename_columns(step)
        elif step.operation == 'add_column':
            self._process_add_column(step)
    
    def _process_source(self, step: TransformationStep):
        """Process source definition"""
        if step.parameters.get('type') == 'sql':
            table = step.parameters.get('database', self.source_table)
            self.query.from_table = table
    
    def _process_select_columns(self, step: TransformationStep):
        """Process column selection"""
        columns = step.parameters.get('columns', [])
        self.query.select_columns = columns
    
    def _process_replace_value(self, step: TransformationStep):
        """Process value replacement"""
        old_val = step.parameters.get('old_value', '')
        new_val = step.parameters.get('new_value', '')
        columns = step.parameters.get('columns', [])
        
        for col in columns:
            # Store transformation
            self.query.column_transformations[col] = f"REPLACE({col}, '{old_val}', '{new_val}')"
    
    def _process_select_rows(self, step: TransformationStep):
        """Process row filtering"""
        condition = step.parameters.get('condition', '')
        sql_condition = self._convert_m_condition_to_sql(condition)
        if sql_condition:
            self.query.where_conditions.append(sql_condition)
    
    def _process_transform_types(self, step: TransformationStep):
        """Process type transformations - handled by table schema in SQL"""
        # Type casting can be added if needed
        type_map = step.parameters.get('type_map', {})
        # Store for potential CAST operations
        pass
    
    def _process_remove_columns(self, step: TransformationStep):
        """Process column removal"""
        # Handled implicitly by SELECT clause
        pass
    
    def _process_rename_columns(self, step: TransformationStep):
        """Process column renaming"""
        rename_map = step.parameters.get('rename_map', {})
        for old_name, new_name in rename_map.items():
            if old_name in self.query.select_columns:
                idx = self.query.select_columns.index(old_name)
                self.query.select_columns[idx] = new_name
                self.query.column_transformations[new_name] = old_name
    
    def _process_add_column(self, step: TransformationStep):
        """Process calculated column addition"""
        col_name = step.parameters.get('column', '')
        formula = step.parameters.get('formula', '')
        
        sql_formula = self._convert_m_formula_to_sql(formula)
        self.query.select_columns.append(col_name)
        self.query.column_transformations[col_name] = sql_formula
    
    def _convert_m_condition_to_sql(self, m_condition: str) -> str:
        """Convert M language condition to SQL WHERE clause"""
        # Handle common patterns
        
        # Pattern: [Column] <> "value" and [Column] <> "value2"
        if '<>' in m_condition and 'and' in m_condition.lower():
            # Extract all <> conditions
            parts = re.findall(r'\[([^\]]+)\]\s*<>\s*"([^"]+)"', m_condition)
            if parts:
                col = parts[0][0]
                values = [f"'{val}'" for _, val in parts]
                return f"{col} NOT IN ({', '.join(values)})"
        
        # Pattern: [Column] <> "value"
        match = re.search(r'\[([^\]]+)\]\s*<>\s*"([^"]+)"', m_condition)
        if match:
            col, val = match.groups()
            return f"{col} <> '{val}'"
        
        # Pattern: [Column] = "value"
        match = re.search(r'\[([^\]]+)\]\s*=\s*"([^"]+)"', m_condition)
        if match:
            col, val = match.groups()
            return f"{col} = '{val}'"
        
        # Pattern: [Column] > value
        match = re.search(r'\[([^\]]+)\]\s*([><=]+)\s*(\d+)', m_condition)
        if match:
            col, op, val = match.groups()
            return f"{col} {op} {val}"
        
        return m_condition.replace('[', '').replace(']', '')
    
    def _convert_m_formula_to_sql(self, m_formula: str) -> str:
        """Convert M language formula to SQL expression"""
        # Basic conversions
        sql_formula = m_formula
        sql_formula = sql_formula.replace('[', '').replace(']', '')
        sql_formula = sql_formula.replace('&', '+')  # String concatenation
        
        # Convert M functions to SQL
        sql_formula = re.sub(r'Text\.Upper\(([^)]+)\)', r'UPPER(\1)', sql_formula)
        sql_formula = re.sub(r'Text\.Lower\(([^)]+)\)', r'LOWER(\1)', sql_formula)
        sql_formula = re.sub(r'Text\.Length\(([^)]+)\)', r'LEN(\1)', sql_formula)
        sql_formula = re.sub(r'Date\.Year\(([^)]+)\)', r'YEAR(\1)', sql_formula)
        sql_formula = re.sub(r'Date\.Month\(([^)]+)\)', r'MONTH(\1)', sql_formula)
        
        return sql_formula


def transform_m_to_sql(m_code: str, source_table: str = "source_table") -> Tuple[str, List[TransformationStep]]:
    """
    Main function to transform M code to SQL
    
    Args:
        m_code: Power Query M language code
        source_table: Name of the source table in SQL
    
    Returns:
        Tuple of (generated SQL, list of transformation steps)
    """
    parser = MCodeParser()
    steps = parser.parse_m_code(m_code)
    
    generator = SQLGenerator(source_table)
    sql = generator.generate_sql(steps)
    
    return sql, steps


if __name__ == "__main__":
    # Example usage
    sample_m_code = """let
    Source = Sql.Database("server", "database"),
    #"Removed Other Columns" = Table.SelectColumns(Source,{"State", "Population estimate, July 1, 2019[2]"}),
    #"Replaced Value1" = Table.ReplaceValue(#"Removed Other Columns","U.S. ","",Replacer.ReplaceText,{"State"}),
    #"Filtered Rows" = Table.SelectRows(#"Replaced Value1", each ([State] <> "Contiguous United States" and [State] <> "Fifty states + D.C."))
in
    #"Filtered Rows"
    """
    
    sql, steps = transform_m_to_sql(sample_m_code, "raw_states_population")
    
    print("Generated SQL:")
    print("=" * 60)
    print(sql)
    print("\n" + "=" * 60)
    print(f"\nParsed {len(steps)} transformation steps")
