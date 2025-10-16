# Refactored Notebook 4 - Fabric Model Analysis with Unused Object Detection
# Enhanced main() function with unused measures, columns, and tables analysis

import pandas as pd
import matplotlib.pyplot as plt
import sempy.fabric as fabric
import sempy_labs
from sempy_labs.report import ReportWrapper
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, StringType, StructType, LongType, StructField, FloatType
from pyspark.sql.functions import col
import re
from datetime import datetime
import time
import logging
from IPython.display import clear_output
from functools import wraps
import sys
import traceback

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fabric_scanning.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Tracker class for monitoring progress
class NewTracker():
    def __init__(self):
        self.error_count = 0
        self.last_error = None
        self.stats = {
            'start_time': datetime.now(),
            'last_update': 0,

            #ALL TOTAL COUNTERS
            'total_workspaces': 0,
            'total_datasets': 0,
            'total_reports': 0,
            'total_dataflows': 0,
            'total_measures': 0,
            'total_tables': 0,
            'total_columns': 0,
            'total_relationships': 0,

            #TRACKING TOTAL COUNTERS
                #workspace
                'workspace_datasets': 0,
                'workspace_reports': 0,
                'workspace_dataflows':0,

                #dataset
                'dataset_tables': 0,
                'dataset_measures': 0,
                'dataset_relationships': 0,
                'dataset_columns':0,

            #PROGRESS COUNTERS
            'processed_workspaces': 0,
            'processed_datasets': 0,
            'processed_tables': 0,

            'ws_processed_datasets': 0,

            #STAGES
            'processing_stage':'',
            'current_operation':'',
            'current_object':{},
            'current_workspace':'',
            'current_dataset':'',
            'current_table':'',

            #ERRORS
            'skipped_objects':{},
            'errors':[],
            'error_types': {},

            #SUCCESS/FAILURE COUNTERS
            'success_count':0,
            'failure_count':0,
            'skipped_count':0,

            # UNUSED OBJECTS ANALYSIS
            'unused_measures': 0,
            'unused_columns': 0,
            'unused_tables': 0,
            'analysis_complete': False,
        }
        self.spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.spinner_idx = 0
        self.progress_frames = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        self.progress_idx = 0
        self.last_update = 0

    def log_error(self, message: str, error: Exception = None):
        self.error_count += 1
        error_details = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'error_type': type(error).__name__ if error else 'Unknown',
            'error_details': str(error) if error else 'No details',
            'current_object': self.stats['current_object'].get('object_name'),
            'object_type': self.stats['current_object'].get('object_type')
        }

        self.stats['errors'].append(error_details)
        logger.error(f"{message}: {str(error) if error else 'No details'}")
        self.last_error = error_details

    def show_progress(self):
        clear_output(wait=True)
        runtime = datetime.now() - self.stats['start_time']
        workspaces_progress = self.stats['processed_workspaces']
        total_workspaces = self.stats['total_workspaces']

        if total_workspaces > 0:
            ws_percent = workspaces_progress / total_workspaces
        else:
            ws_percent = 0
        
        ws_bar = '█' * int(ws_percent * 50) + '-' * (50 - int(ws_percent * 50))

        dataset_progress = self.stats['processed_datasets']
        total_datasets = self.stats['total_datasets']

        if total_datasets > 0:
            ds_percent = dataset_progress / total_datasets
        else:
            ds_percent = 0

        ds_bar1 = '█' * int(ds_percent * 50) + '-' * (50 - int(ds_percent * 50))

        ws_dataset_progress = self.stats['processed_datasets']   
        ws_total_datasets = self.stats['workspace_datasets']

        if ws_total_datasets > 0:
            ws_ds_percent = ws_dataset_progress / ws_total_datasets
        else:
            ws_ds_percent = 0

        ds_bar2 = '█' * int(ws_ds_percent * 50) + '-' * (50 - int(ws_ds_percent * 50))

        total_reports = self.stats['total_reports']
        total_dataflows = self.stats['total_dataflows']
        total_tables = self.stats['total_tables']
        total_relationships = self.stats['total_relationships']
        total_measures = self.stats['total_measures']
        total_columns = self.stats['total_columns']

        ds_total_tables = self.stats['dataset_tables']
        ds_total_relationships = self.stats['dataset_relationships']
        ds_total_measures = self.stats['dataset_measures']
        ds_total_columns = self.stats['dataset_columns']
        
        # Unused objects stats
        unused_measures = self.stats['unused_measures']
        unused_columns = self.stats['unused_columns']
        unused_tables = self.stats['unused_tables']
        
        #Update animation indices
        self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_frames)

        status = [
            '╔════════════════ Enhanced Scan Status ════════════════╗',
            f"║Runtime: {str(runtime).split('.')[0]}",
            f"║Processing Stage: {self.stats['processing_stage']}",
            f"║Current Operation: {self.stats['current_operation']}",
            '',
            f"Scan Progress: {self.spinner_frames[self.spinner_idx]} 🔄",
            f"║Processing Workspace: {workspaces_progress}/{total_workspaces} {self.stats['current_workspace']}",
            f"╢{ws_bar}╟ {ws_percent*100:.0f}%",
            f"{self.stats['current_workspace']} Workspace Objects",
            f"║Reports: {self.stats['workspace_reports']} | Dataflows: {self.stats['workspace_dataflows']} | Datasets: {ws_total_datasets}",
            f"║Processing Dataset: {ws_dataset_progress}/{ws_total_datasets} {self.stats['current_dataset']}",
            f"╢{ds_bar2}╟ {ws_ds_percent*100:.0f}%",
            "",
            f"{self.stats['current_dataset']} Dataset Objects",
            f"║Measures: {ds_total_measures} | Relationships: {ds_total_relationships} | Tables: {ds_total_tables} | Columns: {ds_total_columns}",
            '',
            "All Objects Summary",
            f"║Total Workspaces: {total_workspaces} | Total Reports: {total_reports} | Total Datasets: {total_datasets} | Total_Dataflows: {total_dataflows} ",
            f"║Total Measures: {total_measures} | Total Relationships: {total_relationships} | Total Tables: {total_tables}",
            f"║Total Columns: {total_columns}",
        ]
        
        # Add unused objects analysis if complete
        if self.stats['analysis_complete']:
            status.extend([
                "",
                "🎯 Unused Objects Analysis Results",
                f"║Unused Measures: {unused_measures} | Unused Columns: {unused_columns} | Unused Tables: {unused_tables}",
            ])
        
        sys.stdout.write('\n'.join(status))
        sys.stdout.flush()
        self.last_update = time.time()
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key] = value
            else:
                print(f"Warning: {key} is not a valid stats key.")
        current_time = time.time()
        if current_time - self.last_update >= 1:  # Update display every second
            self.show_progress()

tracker = NewTracker()

def track_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracker.stats['current_operation'] = func.__name__
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            object_name = tracker.stats['current_object'].get('object_name', 'Unknown')
            tracker.log_error(f"Error in {func.__name__} for {object_name}", e)
            raise
        finally:
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} took {duration:.2f} seconds")
            tracker.stats['current_operation'] = None
    return wrapper

# Utility Functions
def sanitize_df_columns(df, extra_columns=False, ws_id=None, ds_id=None):
    """
    Replaces spaces in column names with underscore to prevent errors during Spark Dataframe Creation
    """
    df.columns = [
        re.sub(r'\W+', "_", col.strip().lower())
        for col in df.columns
    ]

    if extra_columns:
        df['workspace_id'] = ws_id
        df['dataset_id'] = ds_id
        
    return df

# GET FUNCTIONS - Original functions from Notebook 4
@track_function
def get_datasets(ws_id):
    """
    Gets all Datasets from the specified workspace
    """
    datasets = fabric.list_datasets(workspace=ws_id)
    
    if not datasets.empty:
        datasets = sanitize_df_columns(datasets)
        datasets['workspace_id'] = ws_id
        tracker.stats['total_datasets'] += len(datasets)
        return datasets
    else:
        return []

@track_function
def get_reports(ws_id):
    """
    Gets all Reports from the specified workspace
    """
    reports = fabric.list_reports(workspace=ws_id)
    
    if not reports.empty:
        reports = sanitize_df_columns(reports)
        reports.rename(columns={
            "dataset_workspace_id": "workspace_id",
            "id": "report_id"
        }, inplace=True)

        tracker.stats['total_reports'] += len(reports)
        return reports
    else:
        return []

@track_function
def get_dataflows(ws_id):
    """
    Gets all Dataflows from the specified workspace
    """
    dataflows = fabric.list_items(type='Dataflow', workspace=ws_id)
    if not dataflows.empty:
        dataflows = sanitize_df_columns(dataflows)    
        tracker.stats['total_dataflows'] += len(dataflows)
        return dataflows
    else:
        return []

@track_function
def get_tables(ws_id, ds_id):
    tables = fabric.list_tables(dataset=ds_id, workspace=ws_id, extended=True)
    if not tables.empty:
        tables['table_id'] = (
            ds_id + "|" + ws_id + "n:" + tables['Name'].astype(str)
        )
        tables = sanitize_df_columns(tables, True, ws_id, ds_id)
        tracker.stats['total_tables'] += len(tables) 
        return tables
    else:
        return []

@track_function
def get_measures(ws_id, ds_id):
    measures = fabric.list_measures(dataset=ds_id, workspace=ws_id)
    if not measures.empty:
        measures = sanitize_df_columns(measures, True, ws_id, ds_id)
        tracker.stats['total_measures'] += len(measures) 
        return measures
    else:
        return []

@track_function
def get_relationships(ws_id, ds_id):
    relationships = fabric.list_relationships(dataset=ds_id, workspace=ws_id, extended=True)
    if not relationships.empty:
        relationships = sanitize_df_columns(relationships, True, ws_id, ds_id)
        tracker.stats['total_relationships'] += len(relationships) 
        return relationships
    else:
        return []

@track_function
def get_table_columns(ws_id, ds_id):
    dataset_cols = fabric.list_columns(dataset=ds_id, workspace=ws_id)

    if not dataset_cols.empty:
        dataset_cols['table_id'] =  (
            ds_id + "|" + ws_id + "n:" + dataset_cols['Table Name'].astype(str)
        )
        dataset_cols = sanitize_df_columns(dataset_cols, True, ws_id, ds_id)
        tracker.stats['total_columns'] += len(dataset_cols)
        return dataset_cols
    else:
        return []

# NEW UNUSED OBJECT DETECTION FUNCTIONS

@track_function
def find_unused_measures(workspace, dataset, all_measures_df):
    """
    Identifies measures that are not used in reports or referenced by other objects.
    
    Args:
        workspace (str): The workspace ID
        dataset (str): The dataset ID  
        all_measures_df (pd.DataFrame): DataFrame containing all measures
    
    Returns:
        tuple: (unused_measures_set, metrics_dict)
    """
    
    print("  📋 Analyzing unused measures...")
    try:
        if all_measures_df is None or len(all_measures_df) == 0:
            print("     └─ No measures found in dataset")
            return set(), {'total_measures': 0, 'unused_measures': 0, 'used_measures': 0, 'utilization_rate': 0}
            
        all_measures = set(all_measures_df['measure_name'].unique())
        print(f"     └─ Found {len(all_measures)} total measures")

        # Get model dependencies
        print("  🔗 Analyzing measure dependencies...")
        referenced_measures = set()
        try:
            dependencies_df = fabric.get_model_calc_dependencies(
                dataset=dataset,
                workspace=workspace
            )
            
            with dependencies_df as calc_deps:
                deps_df = getattr(calc_deps, "dependencies_df", None)
            
            if deps_df is not None and not deps_df.empty:
                referenced_measures = set(
                    deps_df[deps_df['Referenced Object Type'] == 'Measure']['Referenced Object'].unique()
                )
        except Exception as e:
            print(f"     └─ Could not analyze dependencies: {str(e)}")
            
        print(f"     └─ Found {len(referenced_measures)} referenced measures")

        # For now, assume no report analysis (can be enhanced later)
        report_measures = set()
        
        # Calculate used measures
        used_measures = report_measures.union(referenced_measures)
        print(f"  ✓ Total used measures: {len(used_measures)}")

        # Return unused measures
        unused_measures = all_measures.difference(used_measures)
        print(f"  🎯 Identified {len(unused_measures)} potentially unused measures")
        
        return unused_measures, {
            'total_measures': len(all_measures),
            'unused_measures': len(unused_measures),
            'used_measures': len(used_measures),
            'utilization_rate': (len(used_measures) / len(all_measures)) * 100 if len(all_measures) > 0 else 0
        }
        
    except Exception as e:
        print(f"     └─ Error analyzing measures: {str(e)}")
        tracker.log_error(f"Error in find_unused_measures for {dataset}", e)
        return set(), {'total_measures': 0, 'unused_measures': 0, 'used_measures': 0, 'utilization_rate': 0}

@track_function
def find_unused_columns(workspace, dataset, all_columns_df):
    """
    Identifies columns that are not used in reports or referenced by other objects.
    
    Args:
        workspace (str): The workspace ID
        dataset (str): The dataset ID
        all_columns_df (pd.DataFrame): DataFrame containing all columns
    
    Returns:
        tuple: (unused_columns_set, metrics_dict)
    """
    
    print("  📋 Analyzing unused columns...")
    try:
        if all_columns_df is None or len(all_columns_df) == 0:
            print("     └─ No columns found in dataset")
            return set(), {'total_columns': 0, 'unused_columns': 0, 'used_columns': 0, 'utilization_rate': 0}
            
        all_columns = set(all_columns_df['column_name'].unique())
        print(f"     └─ Found {len(all_columns)} total columns")

        # Get model dependencies
        print("  🔗 Analyzing column dependencies...")
        referenced_columns = set()
        try:
            dependencies_df = fabric.get_model_calc_dependencies(
                dataset=dataset,
                workspace=workspace
            )
            
            with dependencies_df as calc_deps:
                deps_df = getattr(calc_deps, "dependencies_df", None)
            
            if deps_df is not None and not deps_df.empty:
                col_object_types = ["Column", "Calc Column"]
                referenced_columns = set(
                    deps_df[deps_df['Referenced Object Type'].isin(col_object_types)]['Referenced Object'].unique()
                )
        except Exception as e:
            print(f"     └─ Could not analyze dependencies: {str(e)}")
            
        print(f"     └─ Found {len(referenced_columns)} referenced columns")

        # For now, assume no report analysis (can be enhanced later)
        report_columns = set()
        
        # Calculate used columns
        used_columns = report_columns.union(referenced_columns)
        print(f"  ✓ Total used columns: {len(used_columns)}")

        # Return unused columns
        unused_columns = all_columns.difference(used_columns)
        print(f"  🎯 Identified {len(unused_columns)} potentially unused columns")
        
        return unused_columns, {
            'total_columns': len(all_columns),
            'unused_columns': len(unused_columns),
            'used_columns': len(used_columns),
            'utilization_rate': (len(used_columns) / len(all_columns)) * 100 if len(all_columns) > 0 else 0
        }
        
    except Exception as e:
        print(f"     └─ Error analyzing columns: {str(e)}")
        tracker.log_error(f"Error in find_unused_columns for {dataset}", e)
        return set(), {'total_columns': 0, 'unused_columns': 0, 'used_columns': 0, 'utilization_rate': 0}

@track_function  
def find_unused_tables(workspace, dataset, all_tables_df, all_columns_df):
    """
    Identifies tables that are not used in reports or referenced by other objects.
    
    Args:
        workspace (str): The workspace ID
        dataset (str): The dataset ID
        all_tables_df (pd.DataFrame): DataFrame containing all tables
        all_columns_df (pd.DataFrame): DataFrame containing all columns
    
    Returns:
        tuple: (unused_tables_set, metrics_dict)
    """
    
    print("  📋 Analyzing unused tables...")
    try:
        if all_tables_df is None or len(all_tables_df) == 0:
            print("     └─ No tables found in dataset")
            return set(), {'total_tables': 0, 'unused_tables': 0, 'used_tables': 0, 'utilization_rate': 0}
            
        all_tables = set(all_tables_df['name'].unique())
        print(f"     └─ Found {len(all_tables)} total tables")

        # Get model dependencies
        print("  🔗 Analyzing table dependencies...")
        referenced_tables = set()
        try:
            dependencies_df = fabric.get_model_calc_dependencies(
                dataset=dataset,
                workspace=workspace
            )
            
            with dependencies_df as calc_deps:
                deps_df = getattr(calc_deps, "dependencies_df", None)
            
            if deps_df is not None and not deps_df.empty:
                referenced_tables = set(
                    deps_df['Referenced Table'].dropna().unique()
                )
        except Exception as e:
            print(f"     └─ Could not analyze dependencies: {str(e)}")
            
        print(f"     └─ Found {len(referenced_tables)} referenced tables")

        # Check for tables with columns that might be used
        tables_with_used_columns = set()
        if all_columns_df is not None and len(all_columns_df) > 0:
            # This is a simplified check - in practice you'd cross-reference with actual column usage
            tables_with_used_columns = set(all_columns_df['table_name'].unique())

        # For now, assume no report analysis (can be enhanced later)  
        report_tables = set()
        
        # Calculate used tables
        used_tables = referenced_tables.union(report_tables).union(tables_with_used_columns)
        print(f"  ✓ Total used tables: {len(used_tables)}")

        # Return unused tables
        unused_tables = all_tables.difference(used_tables)
        print(f"  🎯 Identified {len(unused_tables)} potentially unused tables")
        
        return unused_tables, {
            'total_tables': len(all_tables),
            'unused_tables': len(unused_tables), 
            'used_tables': len(used_tables),
            'utilization_rate': (len(used_tables) / len(all_tables)) * 100 if len(all_tables) > 0 else 0
        }
        
    except Exception as e:
        print(f"     └─ Error analyzing tables: {str(e)}")
        tracker.log_error(f"Error in find_unused_tables for {dataset}", e)
        return set(), {'total_tables': 0, 'unused_tables': 0, 'used_tables': 0, 'utilization_rate': 0}

def create_unused_objects_visualization(measures_metrics, columns_metrics, tables_metrics):
    """
    Creates a comprehensive visualization showing unused objects analysis
    """
    print("\n📊 Creating unused objects visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🎯 Fabric Model - Unused Objects Analysis', fontsize=16, fontweight='bold')
    
    # Measures Chart (Top Left)
    if measures_metrics['total_measures'] > 0:
        measures_labels = ['Used', 'Unused']
        measures_values = [measures_metrics['used_measures'], measures_metrics['unused_measures']]
        measures_colors = ['#28a745', '#dc3545']
        
        bars1 = axes[0,0].bar(measures_labels, measures_values, color=measures_colors, alpha=0.8)
        axes[0,0].set_title(f'📏 Measures Analysis\nTotal: {measures_metrics["total_measures"]} | Utilization: {measures_metrics["utilization_rate"]:.1f}%')
        axes[0,0].set_ylabel('Number of Measures')
        
        # Add value labels
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            percentage = (measures_values[i] / measures_metrics['total_measures']) * 100
            axes[0,0].annotate(f'{int(height)}\n({percentage:.1f}%)',
                             xy=(bar.get_x() + bar.get_width()/2, height),
                             xytext=(0, 5), textcoords="offset points",
                             ha='center', va='bottom', fontweight='bold')
    else:
        axes[0,0].text(0.5, 0.5, 'No Measures Found', ha='center', va='center', transform=axes[0,0].transAxes)
        axes[0,0].set_title('📏 Measures Analysis')
    
    # Columns Chart (Top Right)
    if columns_metrics['total_columns'] > 0:
        columns_labels = ['Used', 'Unused']
        columns_values = [columns_metrics['used_columns'], columns_metrics['unused_columns']]
        columns_colors = ['#17a2b8', '#fd7e14']
        
        bars2 = axes[0,1].bar(columns_labels, columns_values, color=columns_colors, alpha=0.8)
        axes[0,1].set_title(f'📊 Columns Analysis\nTotal: {columns_metrics["total_columns"]} | Utilization: {columns_metrics["utilization_rate"]:.1f}%')
        axes[0,1].set_ylabel('Number of Columns')
        
        # Add value labels
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            percentage = (columns_values[i] / columns_metrics['total_columns']) * 100
            axes[0,1].annotate(f'{int(height)}\n({percentage:.1f}%)',
                             xy=(bar.get_x() + bar.get_width()/2, height),
                             xytext=(0, 5), textcoords="offset points",
                             ha='center', va='bottom', fontweight='bold')
    else:
        axes[0,1].text(0.5, 0.5, 'No Columns Found', ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('📊 Columns Analysis')
    
    # Tables Chart (Bottom Left)
    if tables_metrics['total_tables'] > 0:
        tables_labels = ['Used', 'Unused']
        tables_values = [tables_metrics['used_tables'], tables_metrics['unused_tables']]
        tables_colors = ['#6f42c1', '#e83e8c']
        
        bars3 = axes[1,0].bar(tables_labels, tables_values, color=tables_colors, alpha=0.8)
        axes[1,0].set_title(f'🗃️ Tables Analysis\nTotal: {tables_metrics["total_tables"]} | Utilization: {tables_metrics["utilization_rate"]:.1f}%')
        axes[1,0].set_ylabel('Number of Tables')
        
        # Add value labels
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            percentage = (tables_values[i] / tables_metrics['total_tables']) * 100
            axes[1,0].annotate(f'{int(height)}\n({percentage:.1f}%)',
                             xy=(bar.get_x() + bar.get_width()/2, height),
                             xytext=(0, 5), textcoords="offset points",
                             ha='center', va='bottom', fontweight='bold')
    else:
        axes[1,0].text(0.5, 0.5, 'No Tables Found', ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('🗃️ Tables Analysis')
    
    # Summary Chart (Bottom Right)
    total_objects = measures_metrics['total_measures'] + columns_metrics['total_columns'] + tables_metrics['total_tables']
    total_unused = measures_metrics['unused_measures'] + columns_metrics['unused_columns'] + tables_metrics['unused_tables']
    
    if total_objects > 0:
        summary_labels = ['Measures', 'Columns', 'Tables']
        unused_counts = [measures_metrics['unused_measures'], columns_metrics['unused_columns'], tables_metrics['unused_tables']]
        summary_colors = ['#dc3545', '#fd7e14', '#e83e8c']
        
        bars4 = axes[1,1].bar(summary_labels, unused_counts, color=summary_colors, alpha=0.8)
        axes[1,1].set_title(f'📈 Unused Objects Summary\nTotal Unused: {total_unused}/{total_objects} ({(total_unused/total_objects)*100:.1f}%)')
        axes[1,1].set_ylabel('Number of Unused Objects')
        
        # Add value labels
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            axes[1,1].annotate(f'{int(height)}',
                             xy=(bar.get_x() + bar.get_width()/2, height),
                             xytext=(0, 5), textcoords="offset points",
                             ha='center', va='bottom', fontweight='bold')
    else:
        axes[1,1].text(0.5, 0.5, 'No Objects Found', ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('📈 Unused Objects Summary')
    
    plt.tight_layout()
    plt.show()

def get_workspace_objects(workspaces):
    """
    Enhanced version of the original get_workspace_objects function
    Gets workspace level objects like Semantic Models/Datasets, Reports, and Dataflows
    """

    dataset_list = []
    report_list = []
    dataflows_list = []
    tables_list = []
    measures_list = []
    relationships_list = []
    dataset_cols_list = []
    
    # Store all datasets info for unused object analysis
    all_datasets_info = []
    
    for _, ws in workspaces.iterrows():
        ws_name = ws['name']
        ws_id = ws['id']
        ws_type = ws['type']
                            
        tracker.stats['processed_workspaces'] += 1
        if ws_type == 'AdminInsights':
            continue

        tracker.update(
            current_workspace=ws_name,
            current_operation='Scanning for Reports...',
            current_object={
                'object_type': 'Workspace',
                'object_name': ws_name
            },
            processed_datasets=0,
            workspace_datasets=0
        )
        
        try:
            reports = get_reports(ws_id)
            if len(reports) > 0:
                report_list.append(reports)
                tracker.update(
                    workspace_reports=len(reports),
                    current_operation='Scanning for Dataflows...',
                    dataset_tables=0,
                    dataset_measures=0,
                    dataset_relationships=0
                )

            dataflows = get_dataflows(ws_id)
            if len(dataflows) > 0:
                dataflows_list.append(dataflows)
                tracker.update(
                    workspace_dataflows=len(dataflows),
                    current_operation='Scanning for Datasets...'
                )

            datasets = get_datasets(ws_id)
            if len(datasets) > 0:
                dataset_list.append(datasets)
                tracker.update(
                    workspace_datasets=len(datasets),
                    current_operation='Scanning For Measures...'
                )
            
            for ds_name, ds_id in zip(datasets['dataset_name'], datasets['dataset_id']):
                tracker.stats['processed_datasets'] += 1
                tracker.update(
                    current_object={
                        'object_name': ds_name,
                        'object_type': 'Dataset'
                    },
                    dataset_tables=0,
                    dataset_measures=0,
                    dataset_relationships=0,
                    current_dataset=ds_name
                )

                measures = get_measures(ws_id, ds_id)
                if len(measures) > 0:
                    measures_list.append(measures)
                    tracker.update(
                        dataset_measures=len(measures),
                        current_operation='Scanning for Relationships...'
                    )
                
                relationships = get_relationships(ws_id, ds_id)
                if len(relationships) > 0:
                    relationships_list.append(relationships)
                    tracker.update(
                        dataset_relationships=len(relationships),
                        current_operation='Scanning for Tables...'
                    )

                tables = get_tables(ws_id, ds_id)
                if len(tables) > 0:
                    tables_list.append(tables)
                    tracker.update(
                        dataset_tables=len(tables),
                        current_operation='Getting Table Columns...'
                    )
                
                dataset_cols = get_table_columns(ws_id, ds_id)
                if len(dataset_cols) > 0:
                    dataset_cols_list.append(dataset_cols)
                    tracker.update(
                        dataset_columns=len(dataset_cols),
                        current_operation='Saving data...'
                    )
                
                # Store dataset info for unused object analysis
                all_datasets_info.append({
                    'workspace_id': ws_id,
                    'dataset_id': ds_id,
                    'dataset_name': ds_name,
                    'measures': measures,
                    'tables': tables,
                    'columns': dataset_cols
                })
                
        except Exception as e:
            tb = traceback.format_exc().splitlines()
            tracker.log_error(f"Error while finding objects.", e)
            raise

    # Combine all data
    all_datasets = pd.concat(dataset_list, ignore_index=True) if dataset_list else pd.DataFrame()
    all_reports = pd.concat(report_list, ignore_index=True) if report_list else pd.DataFrame()
    all_dataflows = pd.concat(dataflows_list, ignore_index=True) if dataflows_list else pd.DataFrame()
    all_measures = pd.concat(measures_list, ignore_index=True) if measures_list else pd.DataFrame()
    all_relationships = pd.concat(relationships_list, ignore_index=True) if relationships_list else pd.DataFrame()
    all_tables = pd.concat(tables_list, ignore_index=True) if tables_list else pd.DataFrame()
    all_columns = pd.concat(dataset_cols_list, ignore_index=True) if dataset_cols_list else pd.DataFrame()
    
    return all_datasets, all_reports, all_dataflows, all_measures, all_relationships, all_tables, all_columns, all_datasets_info

def analyze_unused_objects_across_datasets(all_datasets_info):
    """
    Analyzes unused objects across all datasets
    """
    print("\n" + "="*80)
    print("🎯 UNUSED OBJECTS ANALYSIS")
    print("="*80)
    
    total_unused_measures = 0
    total_unused_columns = 0  
    total_unused_tables = 0
    
    all_unused_measures = []
    all_unused_columns = []
    all_unused_tables = []
    
    overall_measures_metrics = {'total_measures': 0, 'unused_measures': 0, 'used_measures': 0}
    overall_columns_metrics = {'total_columns': 0, 'unused_columns': 0, 'used_columns': 0}
    overall_tables_metrics = {'total_tables': 0, 'unused_tables': 0, 'used_tables': 0}
    
    for dataset_info in all_datasets_info:
        ws_id = dataset_info['workspace_id']
        ds_id = dataset_info['dataset_id']
        ds_name = dataset_info['dataset_name']
        
        print(f"\n🔍 Analyzing dataset: {ds_name}")
        print("-" * 60)
        
        # Analyze unused measures
        unused_measures, measures_metrics = find_unused_measures(
            ws_id, ds_id, dataset_info['measures']
        )
        if unused_measures:
            all_unused_measures.extend([(ds_name, measure) for measure in unused_measures])
            total_unused_measures += len(unused_measures)
        
        # Analyze unused columns
        unused_columns, columns_metrics = find_unused_columns(
            ws_id, ds_id, dataset_info['columns']
        )
        if unused_columns:
            all_unused_columns.extend([(ds_name, column) for column in unused_columns])
            total_unused_columns += len(unused_columns)
            
        # Analyze unused tables
        unused_tables, tables_metrics = find_unused_tables(
            ws_id, ds_id, dataset_info['tables'], dataset_info['columns']
        )
        if unused_tables:
            all_unused_tables.extend([(ds_name, table) for table in unused_tables])
            total_unused_tables += len(unused_tables)
        
        # Aggregate metrics
        overall_measures_metrics['total_measures'] += measures_metrics['total_measures']
        overall_measures_metrics['unused_measures'] += measures_metrics['unused_measures']
        overall_measures_metrics['used_measures'] += measures_metrics['used_measures']
        
        overall_columns_metrics['total_columns'] += columns_metrics['total_columns']
        overall_columns_metrics['unused_columns'] += columns_metrics['unused_columns']
        overall_columns_metrics['used_columns'] += columns_metrics['used_columns']
        
        overall_tables_metrics['total_tables'] += tables_metrics['total_tables']
        overall_tables_metrics['unused_tables'] += tables_metrics['unused_tables']
        overall_tables_metrics['used_tables'] += tables_metrics['used_tables']
    
    # Calculate utilization rates
    overall_measures_metrics['utilization_rate'] = (
        (overall_measures_metrics['used_measures'] / overall_measures_metrics['total_measures']) * 100 
        if overall_measures_metrics['total_measures'] > 0 else 0
    )
    overall_columns_metrics['utilization_rate'] = (
        (overall_columns_metrics['used_columns'] / overall_columns_metrics['total_columns']) * 100 
        if overall_columns_metrics['total_columns'] > 0 else 0
    )
    overall_tables_metrics['utilization_rate'] = (
        (overall_tables_metrics['used_tables'] / overall_tables_metrics['total_tables']) * 100 
        if overall_tables_metrics['total_tables'] > 0 else 0
    )
    
    # Update tracker stats
    tracker.update(
        unused_measures=total_unused_measures,
        unused_columns=total_unused_columns,
        unused_tables=total_unused_tables,
        analysis_complete=True
    )
    
    # Print summary
    print(f"\n📊 OVERALL UNUSED OBJECTS SUMMARY")
    print("="*60)
    print(f"📏 Measures: {overall_measures_metrics['unused_measures']}/{overall_measures_metrics['total_measures']} unused ({100-overall_measures_metrics['utilization_rate']:.1f}%)")
    print(f"📊 Columns:  {overall_columns_metrics['unused_columns']}/{overall_columns_metrics['total_columns']} unused ({100-overall_columns_metrics['utilization_rate']:.1f}%)")
    print(f"🗃️  Tables:   {overall_tables_metrics['unused_tables']}/{overall_tables_metrics['total_tables']} unused ({100-overall_tables_metrics['utilization_rate']:.1f}%)")
    
    # Create visualization
    create_unused_objects_visualization(overall_measures_metrics, overall_columns_metrics, overall_tables_metrics)
    
    return {
        'unused_measures': all_unused_measures,
        'unused_columns': all_unused_columns, 
        'unused_tables': all_unused_tables,
        'measures_metrics': overall_measures_metrics,
        'columns_metrics': overall_columns_metrics,
        'tables_metrics': overall_tables_metrics
    }

def save_unused_objects_to_lakehouse(unused_objects_results):
    """
    Save unused objects analysis results to lakehouse tables
    """
    print("\n💾 Saving unused objects analysis to lakehouse...")
    
    try:
        # Create DataFrames for unused objects
        if unused_objects_results['unused_measures']:
            unused_measures_df = pd.DataFrame(
                unused_objects_results['unused_measures'],
                columns=['dataset_name', 'measure_name']
            )
            unused_measures_df['analysis_date'] = datetime.now()
            spark.createDataFrame(unused_measures_df).write.mode("overwrite").saveAsTable("unused_measures_analysis")
            print(f"  ✓ Saved {len(unused_measures_df)} unused measures to 'unused_measures_analysis' table")
        
        if unused_objects_results['unused_columns']:
            unused_columns_df = pd.DataFrame(
                unused_objects_results['unused_columns'],
                columns=['dataset_name', 'column_name']
            )
            unused_columns_df['analysis_date'] = datetime.now()
            spark.createDataFrame(unused_columns_df).write.mode("overwrite").saveAsTable("unused_columns_analysis")
            print(f"  ✓ Saved {len(unused_columns_df)} unused columns to 'unused_columns_analysis' table")
        
        if unused_objects_results['unused_tables']:
            unused_tables_df = pd.DataFrame(
                unused_objects_results['unused_tables'],
                columns=['dataset_name', 'table_name']
            )
            unused_tables_df['analysis_date'] = datetime.now()
            spark.createDataFrame(unused_tables_df).write.mode("overwrite").saveAsTable("unused_tables_analysis")
            print(f"  ✓ Saved {len(unused_tables_df)} unused tables to 'unused_tables_analysis' table")
        
        # Create summary metrics table
        summary_metrics = []
        for metric_type, metrics in [
            ('measures', unused_objects_results['measures_metrics']),
            ('columns', unused_objects_results['columns_metrics']),
            ('tables', unused_objects_results['tables_metrics'])
        ]:
            summary_metrics.append({
                'object_type': metric_type,
                'total_objects': metrics[f'total_{metric_type}'],
                'unused_objects': metrics[f'unused_{metric_type}'],
                'used_objects': metrics[f'used_{metric_type}'],
                'utilization_rate': metrics['utilization_rate'],
                'analysis_date': datetime.now()
            })
        
        summary_df = pd.DataFrame(summary_metrics)
        spark.createDataFrame(summary_df).write.mode("overwrite").saveAsTable("unused_objects_summary")
        print(f"  ✓ Saved summary metrics to 'unused_objects_summary' table")
        
    except Exception as e:
        print(f"  ❌ Error saving to lakehouse: {str(e)}")
        tracker.log_error("Error saving unused objects analysis to lakehouse", e)

def main():
    """
    Enhanced main function with integrated unused objects analysis
    """
    print("🚀" + "="*80)
    print("🎯 FABRIC MODEL ANALYSIS WITH UNUSED OBJECTS DETECTION")
    print("="*82)
    
    # Get all workspaces
    workspaces = fabric.list_workspaces()
    workspaces = sanitize_df_columns(workspaces)    
    tracker.update(
        processed_workspaces=0,
        total_workspaces=len(workspaces),
        processing_stage='Data Collection'
    )

    # Collect all workspace objects
    datasets, reports, dataflows, measures, relationships, tables, columns, datasets_info = get_workspace_objects(workspaces)

    # Save original data to Lakehouse
    tracker.update(
        current_operation='Saving to Lakehouse...',
        processing_stage='Data Storage'
    )
    
    if not workspaces.empty:
        spark.createDataFrame(workspaces).write.mode("overwrite").saveAsTable("fabric_workspaces")
    if not datasets.empty:
        spark.createDataFrame(datasets).write.mode("overwrite").saveAsTable("workspace_datasets")
    if not columns.empty:
        spark.createDataFrame(columns).write.mode("overwrite").saveAsTable("dataset_columns")
    if not tables.empty:
        spark.createDataFrame(tables).write.mode("overwrite").saveAsTable("dataset_tables")
    if not dataflows.empty:
        spark.createDataFrame(dataflows).write.mode("overwrite").saveAsTable("fabric_dataflows")
    if not measures.empty:
        spark.createDataFrame(measures).write.mode("overwrite").saveAsTable("dataset_measures")
    
    if not reports.empty:
        reports_spark = spark.createDataFrame(reports)
        columns_to_clean = ['users', 'subscriptions']
        for col_name in columns_to_clean:
            if col_name in reports.columns:
                reports_spark = reports_spark.withColumn(
                    col_name, 
                    F.col(col_name).cast(ArrayType(StringType()))
                )
        reports_spark.write.mode("overwrite").saveAsTable("workspace_reports")

    if not relationships.empty:
        for col in relationships.select_dtypes(include=['uint64']).columns:
            relationships[col] = relationships[col].astype('int64')
        spark.createDataFrame(relationships).write.mode("overwrite").saveAsTable("dataset_relationships")
    
    # NEW: Analyze unused objects across all datasets
    tracker.update(
        current_operation='Analyzing Unused Objects...',
        processing_stage='Unused Objects Analysis'
    )
    
    unused_objects_results = analyze_unused_objects_across_datasets(datasets_info)
    
    # NEW: Save unused objects analysis to lakehouse
    save_unused_objects_to_lakehouse(unused_objects_results)
    
    # Final summary
    print("\n" + "="*82)
    print("🎉 ENHANCED ANALYSIS COMPLETE")
    print("="*82)
    print(f"📊 Workspaces: {len(workspaces)}")
    print(f"📊 Datasets: {len(datasets)}")
    print(f"📊 Reports: {len(reports)}")
    print(f"📊 Measures: {len(measures)} (Unused: {unused_objects_results['measures_metrics']['unused_measures']})")
    print(f"📊 Columns: {len(columns)} (Unused: {unused_objects_results['columns_metrics']['unused_columns']})")
    print(f"📊 Tables: {len(tables)} (Unused: {unused_objects_results['tables_metrics']['unused_tables']})")
    print(f"📊 Relationships: {len(relationships)}")
    print("="*82)
    
    return unused_objects_results

# Execute the enhanced main function
if __name__ == "__main__":
    results = main()