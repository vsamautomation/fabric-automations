# # ============================================================
# # WORKSPACE SCAN FUNCTIONS
# # ============================================================

# class PowerBIMetadataExtractor:
#     """Gets columns, tables, and measures from Power BI report metadata (PBIR-Legacy Mode)"""

#     def __init__(self):
#         self.tables = set()
#         self.columns = set()
#         self.measures = set()
#         self.visual_details = []
#         self.filter_details = []

#     def _reset(self):
#         """Reset all collections for new getion"""
#         self.tables.clear()
#         self.columns.clear()
#         self.measures.clear()
#         self.visual_details.clear()
#         self.filter_details.clear()

#     def _get_from_expression(self, expression: Dict):
#         """get from expression object"""
#         if 'Column' in expression:
#             #Get table 
#             col = expression['Column']
#             src = col.get('Expression', {}).get('SourceRef',{})
#             table = src.get('Entity','')
#             if table:
#                 self.tables.add(entity)

#             #Get Column
#             column = col.get('Property','')
#             if table and column:
#                 self.columns.add(f"{table}.{column}")

#     def _get_from_where_condition(self, where_item: Dict):
#         """Get additional columns and tables"""
#         condition = where_item.get('Condition',{})
#         if 'In' in condition:
#             expr = condition['In'].get('Expressions', [])
#             for e in expr:
#                 self._get_from_expression(e)

#     def _get_from_filters(self, filters: List[Dict], filter_type: str, context: str):
#         """get from filters array"""
#         for f_idx, f_obj in enumerate(filters):
#             f_name = f_obj.get('name', f'Filter_{f_idx}')

#             #get from expression
#             expression = f_obj.get('expression', {})
#             self._get_from_expression(expression)

#             #Get from filter object (might have other referenced columns here)
#             filter_def = f_obj.get('filter',{})
#             if filter_def:
#                 #Get tables from 'From'
#                 from_clause = filter_def.get('From',[])
#                 for from_item in from_clause:
#                     table = from_item.get('Entity','')
#                     if table:
#                         self.tables.add(table)
                
#                 #Get more tables and columns from 'Where' clause
#                 where_clause = filter_def.get('Where', [])
#                 for where_item in where_clause:
#                     self._get_from_where_condition(where_item)
            
#             #Store filter details
#             self.filter_details.append({
#                 'filter_type':filter_type,
#                 'context': context,
#                 'filter_name': filter_name,
#                 'has_expression': bool(expression),
#                 'has_filter_def': bool(filter_def)
#             })

#     def _parse_query_ref(self, query_ref: str):
#         """Parse queryRef format (e.g., 'table.column' or 'table.measure')"""
#         if '.' in query_ref:
#             table, field = query_ref.split('.', 1)
#             self.tables.add(table)
#             #field type (Column | Measure) will be determined from prototype query

#     def _get_from_prototype_query(self, prototype_query: Dict):
#         """Get from prototypeQuery object"""
#         # Get table from 'From' Clause
#         from_clause = prototype_query.get('From', [])
#         for from_item in from_clause:
#             table = from_item.get('Entity','')
#             if table:
#                 self.tables.add(table)
        
#         #Get columns and measures from 'Select' clause
#         select_clause = prototype_query.get('Select', [])
#         for s in select_clause:
#             name = s.get('Name', '')  #returns table.field[Column | Measure]

#             #Check if Column
#             if 'Column' in s:
#                 column = s['Column'].get('Property', '')
#                 if column and name:
#                     self.columns.add(name) #qualified column name (table.column)
#                     #Get table
#                     if '.' in name:
#                         table = name.split('.')[0]
#                         self.tables.add(table)
#             #Check if Measure
#             elif 'Measure' in s:
#                 measure = s['Measure'].get('Property', '')
#                 if measure and name:
#                     self.measures.add(name) #qualified measure name (table.measure)

#                     #Get table
#                     if '.' in name:
#                         table = name.split('.')[0]
#                         self.tables.add(table)

#     def _get_from_actual_visual(self, visual: Dict, page_name: str, visual_name: str):
#         """Get from singleVisual object"""
#         vis_type = visual.get('visualType', 'unknown')

#         #Get from projecions (has query references in table.column format)
#         projections = visual.get('projections', {})
#         projection_refs = []

#         for _, proj_list in projections.items():
#             for p in proj_list:
#                 query_ref = p.get('queryRef', '')
#                 if query_ref:
#                     projection_refs.append(query_ref)
#                     self._parse_query_ref(query_ref)
        
#         #Get from prototypeQuery (provides extra info on type of fields referenced [Column | Measure] )
#         prototype_query = visual.get('prototyoeQuery', {})
#         self._get_from_prototype_query(prototype_query)

#         #Store visual details
#         self.visual_details.append({
#             'page': page_name,
#             'visual_name': visual_name,
#             'visual_type': vis_type,
#             'projection_refs': projection_refs,
#             'has_prototype_query': bool(prototype_query)
#         })
 
#     def _get_from_visual_containers(self, visual_containers: List[Dict], page_name: str):
#         """Get from visualContainers array"""
        
#         for vis_idx, vis_container in enumerate(visual_containers):
#             vis_config = vis_container.get('config', {})
#             vis_name = vis_config.get('name', f'Visual_{vis_idx}')

#             #Get from visual_level filters
#             vis_filter = vis_container.get('filters', [])

#             if isinstance(vis_filter, str):
#                 vis_filter = json.loads(vis_filter)

#             self._get_from_filters(vis_filter, 'visual', f"{page_name}->{vis_name}")

#             #Get from actual visual
#             vis = vis_config.get('singeVisual', {})
#             if vis:
#                 self._get_from_actual_visual(vis, page_name, vis_name)


#     def get_from_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         """get metadata from JSON data"""
#         self._reset()
        
#         # get from sections
#         sections = data.get('sections', [])

#         for page_idx, page in enumerate(sections):
#             page_name = page.get('displayName', f'Page_{page_idx}')
            
#             # get from page-level filters
#             page_filters = page.get('filters',[])
#             if isinstance(page_filters, str):
#                 page_filters = json.loads(page_filters)
            
#             self._get_from_filters(page_filters, 'page', page_name)

#             #Get from visual containers
#             visual_containers = page.get('visualContainers',[])
#             self._get_from_visual_containers(visual_containers, page_name)

#         return {
#             'tables': sorted(list(self.tables)),
#             'columns': sorted(list(self.columns)),
#             'measures': sorted(list(self.measures)),
#             'summary': {
#                 'total_tables': len(self.tables),
#                 'total_columns': len(self.columns),
#                 'total_measures': len(self.measures)
#             },
#             'visual_details': self.visual_details,
#             'filter_details': self.filter_details
#         }


# class FabricWorkspaceAnalyzer:
#     """Main analyzer class implementing the complete workflow"""
#     def __init__(self):
#         self.workspaces_df = pd.DataFrame()
#         self.datasets_df = pd.DataFrame()
#         self.reports_df = pd.DataFrame()
#         self.pbi_reports_df = pd.DataFrame()
#         self.all_dataset_info = {}
#         self.report_metadata_list = []
#         self.report_objects_used = []

#     def sanitize_df_columns(df, extra_columns=False, ws_id=None, ds_id=None, ws_name=None, ds_name=None):
#         """
#         Replaces spaces in column names with underscore to prevent errors during Spark Dataframe Creation
#         """
#         if df.empty:
#             return df
            
#         df.columns = [
#             re.sub(r'\W+', "_", col.strip().lower())
#             for col in df.columns
#         ]

#         if extra_columns:
#             df['workspace_id'] = ws_id
#             df['dataset_id'] = ds_id
#             df['workspace_name'] = ws_name
#             df['dataset_name'] = ds_name
            
#         return df

#     def save_to_lakehouse(df, table_name, description=""):
#         """
#         Save DataFrame to lakehouse using Spark
#         """
#         try:
#             if df.empty:
#                 print(f"  ⚠️ Skipping empty DataFrame for table: {table_name}")
#                 return
                
#             # Add analysis timestamp
#             df_with_timestamp = df.copy()
#             df_with_timestamp['analysis_date'] = datetime.now()
            
#             # Convert to Spark DataFrame and save
#             spark_df = spark.createDataFrame(df_with_timestamp)
#             spark_df.write.mode("overwrite").saveAsTable(table_name)
            
#             print(f"  ✅ Saved {len(df)} records to '{table_name}' table")
#             if description:
#                 print(f"     📝 {description}")
                
#         except Exception as e:
#             print(f"  ❌ Error saving to {table_name}: {str(e)}")

    
#     def get_workspaces(self):
#         """Step 1: Get Workspaces"""
#         print("🔍 STEP 1: Discovering workspaces...")
        
#         self.workspaces_df = fabric.list_workspaces()
#         self.workspaces_df = self.sanitize_df_columns(self.workspaces_df)
#         self.workspaces_df = self.workspaces_df[['id', 'name', 'type']]
        
#         print(f"  ✅ Found {len(self.workspaces_df)} workspaces")
#         return self.workspaces_df

    
#     def get_datasets_and_reports(self):
#         """Step 2: Get Datasets and Reports in parallel"""
#         print("\n🔍 STEP 2: Getting datasets and reports...")
        
#         datasets_all, reports_all = [], []
        
#         for _, ws in self.workspaces_df.iterrows():
#             ws_id = ws['id']
#             ws_name = ws['name']
#             ws_type = ws['type']
            
#             if ws_type == "AdminInsights":
#                 continue
                
#             print(f"  📦 Scanning workspace: {ws_name}")
            
#             # Get Datasets
#             try:
#                 ds = fabric.list_datasets(workspace=ws_id)
#                 if not ds.empty:
#                     ds['workspace_id'] = ws_id
#                     ds['workspace_name'] = ws_name
#                     datasets_all.append(ds)
#             except Exception as e:
#                 print(f"    ⚠️ Datasets error in {ws_name}: {e}")
            
#             # Get Reports
#             try:
#                 rep = fabric.list_reports(workspace=ws_id)
#                 if not rep.empty:
#                     rep['workspace_id'] = ws_id
#                     rep['workspace_name'] = ws_name
#                     reports_all.append(rep)
#             except Exception as e:
#                 print(f"    ⚠️ Reports error in {ws_name}: {e}")
        
#         # Combine results
#         self.datasets_df = self.sanitize_df_columns(pd.concat(datasets_all, ignore_index=True) if datasets_all else pd.DataFrame())
#         self.reports_df = self.sanitize_df_columns(pd.concat(reports_all, ignore_index=True) if reports_all else pd.DataFrame())
        
#         # Filter PowerBI reports
#         if not self.reports_df.empty and "report_type" in self.reports_df.columns:
#             self.pbi_reports_df = self.reports_df[self.reports_df["report_type"] == "PowerBIReport"].copy()
#         else:
#             self.pbi_reports_df = self.reports_df
        
#         print(f"  ✅ Found {len(self.datasets_df)} datasets and {len(self.reports_df)} reports ({len(self.pbi_reports_df)} PowerBI reports)")
#         return self.datasets_df, self.reports_df
    

# def get_workspace_objects():
#     print("🔍 Discovering workspaces...")

#     workspaces = fabric.list_workspaces()
#     workspaces = sanitize_df_columns(workspaces)

#     datasets, reports, dataflows = [], [], []

#     for _, ws in workspaces.iterrows():
#         ws_id = ws['id']
#         ws_name = ws['name']
#         ws_type = ws['type']

#         if ws_type == "AdminInsights":
#             continue

#         print(f"\n📦 Scanning workspace: {ws_name}")

#         # --Get Datases

#         try:
#             ds = fabric.list_datasets(workspace=ws_id)
#             if not ds.empty:
#                 ds = sanitize_df_columns(ds)
#                 ds['workspace_id'] = ws_id
#                 ds['workspace_name'] = ws_name
#                 datasets.append(ds)
#         except Exception as e:
#             print(f"  ⚠️ Datasets error in {ws_name}: {e}")
        
#         # -- Reports
#         try:
#             rep = fabric.list_reports(workspace=ws_id)
#             if not rep.empty:
#                 rep = sanitize_df_columns(rep)
#                 rep['report_workspace_id'] = ws_id
#                 rep['report_workspace_name'] = ws_name
#                 reports.append(rep)
#         except Exception as e:
#             print(f"  ⚠️ Reports error in {ws_name}: {e}")

#         # --- Dataflows
#         try:
#             dfs = sempy_labs.list_dataflows(workspace=ws_id)
#             if not dfs.empty:
#                 dfs = sanitize_df_columns(dfs)
#                 dfs['workspace_id'] = ws_id
#                 dataflows_all.append(dfs)
#         except Exception as e:
#             print(f"  ⚠️ Dataflows error in {ws_name}: {e}")

#     datasets = pd.concat(datasets, ignore_index =True) if datasets else pd.DataFrame()
#     reports = pd.concat(reports, ignore_index = True) if reports else pd.DataFrame()
#     dataflows = pd.concat(dataflows, ignore_index = True) if dataflows else pd.DataFrame()

#     if not reports.empty and 'report_type' in reports.columns:
#         pbi_reports = reports[reports['report_type'] == "PowerBIReport"].copy()
#         paginated_reports = reports[reports['report_type'] == "PaginatedReport"].copy()
#     else:
#         pbi_reports = reports_df
#         paginated_reports = pd.DataFrame()
    
    
#     print("\n✅ Object discovery complete")
#     print(f"  Workspaces: {len(workspaces_df)}")
#     print(f"  Datasets:   {len(datasets_df)}")
#     print(f"  Reports:    {len(reports_df)} (PBI: {len(pbi_reports_df)}, Paginated: {len(paginated_reports_df)})")
#     print(f"  Dataflows:  {len(dataflows_df)}")

#     return datasets, reports, dataflows, pbi_reports, paginated_reports

# def collect_report_info(pbi_reports):

#     report_metadata = []
#     report_objects = []

#     if not pbi_reports.empty:
#         print(f"\n🖼️ Processing {len(pbi_reports_df)} PowerBI reports...")

#         for idx, report in pbi_reports.iterrows():
#             reportId = report.get('id', '')
#             reportName = report.get('name', f'Report_{idx}')
#             workspaceId = report.get('report_workspace_id', '')

# def collect_dataset_info(ds_id: str, ds_name: str, ws_id: str, ws_name: str) -> DatasetInfo:
    
#     print(f"🔹 Processing dataset: {ds_name} (Workspace: {ws_name})")

#     dataset_info = DatasetInfo(ds_id, ds_name, ws_id, ws_name)

#     #Get DataModel Dependencies. This will be used later to compare with report dependecies to find unused columns.
#     try:
#         deps = fabric.get_model_calc_dependencies(dataset=ds_id, workspace=ws_id)
#         with deps as calc_deps:
#             dependencies_df = getattr(calc_deps, "dependencies_df", None)
        
#         if dependencies_df is not None and not dependencies_df.empty:
#             dependencies_df = sanitize_df_columns(
#                 df = dependencies_df, 
#                 extra_columns= True,
#                 ws_id = ws_id, 
#                 ds_id= ds_id,
#                 ws_name= ws_name,
#                 ds_name= ds_name
#             )
#             dataset_info.dependencies_df = dependencies_df
#             print(f"  Found {len(dependencies_df)} dependencies")
#         else:
#             dataset_info.dependencies_df = pd.DataFrame()
#             print(f"  No dependencies found for {ds_name}")
#     except Exception as e:
#         print(f"  ⚠️ Dependencies unavailable for {ds_name}: {e}")
#         dataset_info.dependencies_df = pd.DataFrame()

#     # Get all tables in the dataset.
#     try:
#         tables = fabric.list_tables(dataset=ds_id, workspace=ws_id)
#         if not tables.empty:
#             tables = sanitize_df_columns(
#                 df = tables, 
#                 extra_columns = True,
#                 ws_id = ws_id, 
#                 ds_id = ds_id,
#                 ws_name = ws_name,
#                 ds_name= ds_name
#             )
#             dataset_info.tables_df = tables
#             print(f"  Found {len(tables)} tables")
#     except Exception as e:
#         print(f"  ⚠️ Tables unavailable for {ds_name}: {e}")

                
#     # Get all relationships in the dataset.
#     try:
#         relationships = fabric.list_relationships(dataset=ds_id, workspace=ws_id, extended=True)
#         if not relationships.empty:
#             relationships = sanitize_df_columns(df = relationships)
#             relationships['qualified_from'] = "'" + relationships['from_table'] + "'[" + relationships['from_column'] + "]"
#             relationships['qualified_to'] = "'" + relationships['to_table'] + "'[" + relationships['to_column'] + "]"
#             dataset_info.relationships_df = relationships
#             print(f"  Found {len(relationships)} relationships")
#     except Exception as e:
#         print(f"  ⚠️ Relationships unavailable for {ds_name}: {e}")

#     # Get all measures in the dataset.
#     try:
#         measures = fabric.list_measures(dataset=ds_id, workspace=ws_id)
#         if not measures.empty:
#             measures = sanitize_df_columns(df = measures)
#             dataset_info.measures_df = measures
#             print(f"  Found {len(measures)} measures")
#     except Exception as e:
#         print(f"  ⚠️ Measures unavailable for {ds_name}: {e}")

#     # Get all columns in the dataset.
#     try:
#         columns = fabric.list_columns(dataset=ds_id, workspace=ws_id, extended=True)
#         if not columns.empty:
#             columns = sanitize_df_columns(
#                 df = columns,
#                 extra_columns= True,
#                 ws_id = ws_id, 
#                 ds_id= ds_id,
#                 ws_name= ws_name,
#                 ds_name= ds_name
#             )
#             columns['qualified_name'] = "'" + columns['table_name'] + "'[" + columns['column_name'] + ']'
#             dataset_info.columns_df = columns
#             print(f"  Found {len(columns)} columns")
#     except Exception as e:
#         print(f"  ⚠️ Columns unavailable for {ds_name}: {e}")
    
#     return dataset_info


# print("✅ Workspace Functions defined")