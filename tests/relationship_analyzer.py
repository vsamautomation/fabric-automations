"""
Power BI Relationship Analyzer
Extracts and analyzes relationships between tables in Power BI datasets
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class Relationship:
    """Represents a relationship between two tables"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    cardinality: str  # OneToMany, ManyToMany, OneToOne, ManyToOne
    is_active: bool
    cross_filter_direction: str  # Single, Both, None
    relationship_id: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None


class RelationshipAnalyzer:
    """Analyzes relationships in Power BI datasets"""
    
    def __init__(self):
        self.relationships: List[Relationship] = []
        
    def extract_relationships_from_dataset(self, dataset: Dict[str, Any]) -> List[Relationship]:
        """Extract relationships from a single dataset"""
        dataset_relationships = []
        dataset_id = dataset.get("id", "")
        dataset_name = dataset.get("name", "")
        
        # Look for relationships in the dataset structure
        relationships = dataset.get("relationships", [])
        
        for rel in relationships:
            relationship = Relationship(
                from_table=rel.get("fromTable", ""),
                from_column=rel.get("fromColumn", ""),
                to_table=rel.get("toTable", ""),
                to_column=rel.get("toColumn", ""),
                cardinality=rel.get("cardinality", ""),
                is_active=rel.get("isActive", True),
                cross_filter_direction=rel.get("crossFilterDirection", "Single"),
                relationship_id=rel.get("id", ""),
                dataset_id=dataset_id,
                dataset_name=dataset_name
            )
            dataset_relationships.append(relationship)
            
        return dataset_relationships
    
    def analyze_all_datasets(self, datasets: List[Dict[str, Any]]) -> List[Relationship]:
        """Analyze relationships across all datasets"""
        all_relationships = []
        
        for dataset in datasets:
            dataset_relationships = self.extract_relationships_from_dataset(dataset)
            all_relationships.extend(dataset_relationships)
            
        self.relationships = all_relationships
        return all_relationships
    
    def get_relationship_summary(self) -> Dict[str, Any]:
        """Get a summary of relationships analysis"""
        total_relationships = len(self.relationships)
        active_relationships = sum(1 for r in self.relationships if r.is_active)
        inactive_relationships = total_relationships - active_relationships
        
        cardinality_counts = {}
        for rel in self.relationships:
            cardinality_counts[rel.cardinality] = cardinality_counts.get(rel.cardinality, 0) + 1
        
        cross_filter_counts = {}
        for rel in self.relationships:
            cross_filter_counts[rel.cross_filter_direction] = cross_filter_counts.get(rel.cross_filter_direction, 0) + 1
        
        return {
            "total_relationships": total_relationships,
            "active_relationships": active_relationships,
            "inactive_relationships": inactive_relationships,
            "cardinality_distribution": cardinality_counts,
            "cross_filter_distribution": cross_filter_counts,
            "datasets_with_relationships": len(set(r.dataset_id for r in self.relationships if r.dataset_id))
        }
    
    def get_table_connections(self) -> Dict[str, List[str]]:
        """Get a mapping of tables and their connected tables"""
        connections = {}
        
        for rel in self.relationships:
            if rel.is_active:  # Only consider active relationships
                # Add connection from source to target
                if rel.from_table not in connections:
                    connections[rel.from_table] = []
                if rel.to_table not in connections[rel.from_table]:
                    connections[rel.from_table].append(rel.to_table)
                
                # Add reverse connection (bidirectional)
                if rel.to_table not in connections:
                    connections[rel.to_table] = []
                if rel.from_table not in connections[rel.to_table]:
                    connections[rel.to_table].append(rel.from_table)
        
        return connections
    
    def find_isolated_tables(self, all_tables: List[str]) -> List[str]:
        """Find tables that have no relationships"""
        connected_tables = set()
        for rel in self.relationships:
            if rel.is_active:
                connected_tables.add(rel.from_table)
                connected_tables.add(rel.to_table)
        
        return [table for table in all_tables if table not in connected_tables]
    
    def export_relationships(self) -> List[Dict[str, Any]]:
        """Export relationships as a list of dictionaries"""
        return [asdict(rel) for rel in self.relationships]
    
    def detect_circular_relationships(self) -> List[List[str]]:
        """Detect circular relationship chains"""
        connections = self.get_table_connections()
        visited = set()
        cycles = []
        
        def dfs(table: str, path: List[str], current_path: set):
            if table in current_path:
                # Found a cycle
                cycle_start = path.index(table)
                cycle = path[cycle_start:] + [table]
                cycles.append(cycle)
                return
            
            if table in visited:
                return
            
            visited.add(table)
            current_path.add(table)
            path.append(table)
            
            for connected_table in connections.get(table, []):
                dfs(connected_table, path, current_path)
            
            current_path.remove(table)
            path.pop()
        
        for table in connections:
            if table not in visited:
                dfs(table, [], set())
        
        return cycles