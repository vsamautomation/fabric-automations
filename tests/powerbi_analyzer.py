"""
Comprehensive Power BI Analysis Module
Integrates relationship analysis and usage analysis for Power BI datasets
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from relationship_analyzer import RelationshipAnalyzer, Relationship
from dax_dependency_parser import UsageAnalyzer, DaxReference


@dataclass
class DatasetAnalysisResult:
    """Complete analysis result for a dataset"""
    dataset_id: str
    dataset_name: str
    workspace_id: str
    workspace_name: str
    
    # Relationship analysis
    relationships: List[Dict[str, Any]]
    relationship_summary: Dict[str, Any]
    isolated_tables: List[str]
    circular_relationships: List[List[str]]
    
    # Usage analysis
    usage_analysis: Dict[str, Any]
    unused_objects: Dict[str, List[Dict[str, Any]]]
    
    # Combined insights
    insights: List[str]


class PowerBIAnalyzer:
    """Main analyzer that combines relationship and usage analysis"""
    
    def __init__(self, include_hidden: bool = False):
        self.relationship_analyzer = RelationshipAnalyzer()
        self.usage_analyzer = UsageAnalyzer()
        self.include_hidden = include_hidden
        self.analysis_results: List[DatasetAnalysisResult] = []
    
    def analyze_enriched_reports(self, enriched_reports: List[Dict[str, Any]]) -> List[DatasetAnalysisResult]:
        """Analyze a list of enriched reports (from main.py output)"""
        # Group reports by dataset to avoid duplicate analysis
        datasets_by_id = {}
        
        for report in enriched_reports:
            dataset_id = report.get("datasetId", "")
            if dataset_id and "tables" in report:
                # Reconstruct dataset structure from the enriched report
                dataset = {
                    "id": dataset_id,
                    "name": report.get("dataset_name", ""),
                    "tables": report.get("tables", []),
                    "relationships": []  # This might need to be populated from scan data
                }
                
                if dataset_id not in datasets_by_id:
                    datasets_by_id[dataset_id] = {
                        "dataset": dataset,
                        "workspace_id": report.get("dataset_ws_id", ""),
                        "workspace_name": report.get("dataset_ws_name", ""),
                        "reports": []
                    }
                
                datasets_by_id[dataset_id]["reports"].append(report)
        
        # Analyze each dataset
        results = []
        for dataset_info in datasets_by_id.values():
            result = self.analyze_single_dataset(
                dataset_info["dataset"],
                dataset_info["workspace_id"],
                dataset_info["workspace_name"]
            )
            results.append(result)
        
        self.analysis_results = results
        return results
    
    def analyze_scan_results(self, scan_results: List[Dict[str, Any]]) -> List[DatasetAnalysisResult]:
        """Analyze scan results directly from Power BI Admin API"""
        results = []
        
        for workspace_result in scan_results:
            workspace_id = workspace_result.get("workspaceId", "")
            workspace_name = workspace_result.get("workspaceName", "")
            
            scan_data = workspace_result.get("scanResult", {})
            workspaces = scan_data.get("workspaces", [])
            
            for workspace in workspaces:
                datasets = workspace.get("datasets", [])
                
                for dataset in datasets:
                    result = self.analyze_single_dataset(dataset, workspace_id, workspace_name)
                    results.append(result)
        
        self.analysis_results = results
        return results
    
    def analyze_single_dataset(self, dataset: Dict[str, Any], workspace_id: str, workspace_name: str) -> DatasetAnalysisResult:
        """Analyze a single dataset comprehensively"""
        dataset_id = dataset.get("id", "")
        dataset_name = dataset.get("name", "")
        
        # Relationship analysis
        relationships = self.relationship_analyzer.extract_relationships_from_dataset(dataset)
        
        # Get all table names for isolation analysis
        all_tables = [table.get("name", "") for table in dataset.get("tables", [])
                     if not table.get("isHidden", False) or self.include_hidden]
        
        isolated_tables = self.relationship_analyzer.find_isolated_tables(all_tables)
        circular_relationships = self.relationship_analyzer.detect_circular_relationships()
        
        relationship_summary = {
            "total_relationships": len(relationships),
            "active_relationships": sum(1 for r in relationships if r.is_active),
            "inactive_relationships": sum(1 for r in relationships if not r.is_active),
            "total_tables": len(all_tables),
            "connected_tables": len(all_tables) - len(isolated_tables),
            "isolated_tables_count": len(isolated_tables),
            "circular_chains": len(circular_relationships)
        }
        
        # Usage analysis
        usage_analysis = self.usage_analyzer.analyze_dataset_usage(dataset)
        unused_objects = self.usage_analyzer.find_unused_objects(dataset)
        
        # Generate insights
        insights = self._generate_insights(
            dataset, relationships, isolated_tables, 
            circular_relationships, usage_analysis, unused_objects
        )
        
        return DatasetAnalysisResult(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            workspace_id=workspace_id,
            workspace_name=workspace_name,
            relationships=[asdict(r) for r in relationships],
            relationship_summary=relationship_summary,
            isolated_tables=isolated_tables,
            circular_relationships=circular_relationships,
            usage_analysis=usage_analysis,
            unused_objects=unused_objects,
            insights=insights
        )
    
    def _generate_insights(self, dataset: Dict[str, Any], relationships: List[Relationship], 
                          isolated_tables: List[str], circular_relationships: List[List[str]],
                          usage_analysis: Dict[str, Any], unused_objects: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Generate actionable insights from the analysis"""
        insights = []
        
        # Relationship insights
        if len(isolated_tables) > 0:
            insights.append(f"Found {len(isolated_tables)} isolated tables with no relationships: {', '.join(isolated_tables[:3])}{'...' if len(isolated_tables) > 3 else ''}")
        
        if len(circular_relationships) > 0:
            insights.append(f"Detected {len(circular_relationships)} circular relationship chains - these may cause performance issues")
        
        inactive_count = sum(1 for r in relationships if not r.is_active)
        if inactive_count > 0:
            insights.append(f"{inactive_count} inactive relationships found - consider removing or activating them")
        
        # Usage insights
        total_unused_columns = len(unused_objects.get('columns', []))
        total_unused_measures = len(unused_objects.get('measures', []))
        total_unused_tables = len(unused_objects.get('tables', []))
        
        if total_unused_columns > 0:
            insights.append(f"{total_unused_columns} unused columns detected - consider removing to improve performance")
        
        if total_unused_measures > 0:
            insights.append(f"{total_unused_measures} unused measures found - cleanup recommended")
        
        if total_unused_tables > 0:
            insights.append(f"{total_unused_tables} unused tables identified - these may be consuming unnecessary resources")
        
        # Performance insights
        total_tables = len(dataset.get("tables", []))
        if len(isolated_tables) / total_tables > 0.3:
            insights.append("High percentage of isolated tables - consider reviewing data model design")
        
        # Complexity insights
        total_dependencies = usage_analysis.get('total_dependencies', 0)
        if total_dependencies > 100:
            insights.append("High number of dependencies detected - model may be complex and hard to maintain")
        
        return insights
    
    def get_overall_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all analyzed datasets"""
        if not self.analysis_results:
            return {}
        
        total_datasets = len(self.analysis_results)
        total_relationships = sum(r.relationship_summary["total_relationships"] for r in self.analysis_results)
        total_isolated_tables = sum(len(r.isolated_tables) for r in self.analysis_results)
        total_circular_chains = sum(len(r.circular_relationships) for r in self.analysis_results)
        
        total_unused_columns = sum(len(r.unused_objects.get('columns', [])) for r in self.analysis_results)
        total_unused_measures = sum(len(r.unused_objects.get('measures', [])) for r in self.analysis_results)
        total_unused_tables = sum(len(r.unused_objects.get('tables', [])) for r in self.analysis_results)
        
        datasets_with_issues = 0
        for result in self.analysis_results:
            has_issues = (
                len(result.isolated_tables) > 0 or
                len(result.circular_relationships) > 0 or
                len(result.unused_objects.get('columns', [])) > 0 or
                len(result.unused_objects.get('measures', [])) > 0 or
                len(result.unused_objects.get('tables', [])) > 0
            )
            if has_issues:
                datasets_with_issues += 1
        
        return {
            "total_datasets_analyzed": total_datasets,
            "datasets_with_issues": datasets_with_issues,
            "overall_health_score": round((total_datasets - datasets_with_issues) / total_datasets * 100, 1) if total_datasets > 0 else 0,
            "relationship_stats": {
                "total_relationships": total_relationships,
                "total_isolated_tables": total_isolated_tables,
                "total_circular_chains": total_circular_chains
            },
            "usage_stats": {
                "total_unused_columns": total_unused_columns,
                "total_unused_measures": total_unused_measures,
                "total_unused_tables": total_unused_tables
            },
            "top_issues": self._get_top_issues()
        }
    
    def _get_top_issues(self) -> List[Dict[str, Any]]:
        """Get top issues across all datasets"""
        issues = []
        
        for result in self.analysis_results:
            if len(result.isolated_tables) > 0:
                issues.append({
                    "type": "isolated_tables",
                    "dataset": result.dataset_name,
                    "workspace": result.workspace_name,
                    "count": len(result.isolated_tables),
                    "severity": "medium"
                })
            
            if len(result.circular_relationships) > 0:
                issues.append({
                    "type": "circular_relationships",
                    "dataset": result.dataset_name,
                    "workspace": result.workspace_name,
                    "count": len(result.circular_relationships),
                    "severity": "high"
                })
            
            unused_count = (len(result.unused_objects.get('columns', [])) + 
                          len(result.unused_objects.get('measures', [])) + 
                          len(result.unused_objects.get('tables', [])))
            
            if unused_count > 10:  # Threshold for significant unused objects
                issues.append({
                    "type": "unused_objects",
                    "dataset": result.dataset_name,
                    "workspace": result.workspace_name,
                    "count": unused_count,
                    "severity": "low" if unused_count < 20 else "medium"
                })
        
        # Sort by severity and count
        severity_order = {"high": 3, "medium": 2, "low": 1}
        issues.sort(key=lambda x: (severity_order[x["severity"]], x["count"]), reverse=True)
        
        return issues[:10]  # Return top 10 issues
    
    def export_results(self) -> Dict[str, Any]:
        """Export all analysis results"""
        return {
            "summary": self.get_overall_summary(),
            "dataset_analyses": [asdict(result) for result in self.analysis_results],
            "metadata": {
                "include_hidden_objects": self.include_hidden,
                "total_datasets": len(self.analysis_results),
                "analysis_version": "1.0.0"
            }
        }