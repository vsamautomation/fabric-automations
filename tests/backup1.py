class FabricWorkspaceAnalyzer:
    """Main analyzer class implementing the complete workflow"""
    
    def __init__(self):
        self.workspaces_df = pd.DataFrame()
        self.datasets_df = pd.DataFrame()
        self.reports_df = pd.DataFrame()
        self.pbi_reports_df = pd.DataFrame()
        self.all_dataset_info = {}
        self.report_metadata_list = []
        self.report_objects_used = []
        
    def sanitize_df_columns(self, df, extra_columns=False, ws_id=None, ds_id=None, ws_name=None, ds_name=None):
        """Replaces spaces in column names with underscore to prevent errors during Spark Dataframe Creation"""
        if df.empty:
            return df
            
        df.columns = [
            re.sub(r'\W+', "_", col.strip().lower())
            for col in df.columns
        ]

        if extra_columns:
            df['workspace_id'] = ws_id
            df['dataset_id'] = ds_id
            df['workspace_name'] = ws_name
            df['dataset_name'] = ds_name
            
        return df

    def save_to_lakehouse(self, df, table_name, description=""):
        """Save DataFrame to lakehouse using Spark"""
        try:
            if df.empty:
                print(f"  ⚠️ Skipping empty DataFrame for table: {table_name}")
                return
                
            # Add analysis timestamp
            df_with_timestamp = df.copy()
            df_with_timestamp['analysis_date'] = datetime.now()
            
            # Convert to Spark DataFrame and save
            spark_df = spark.createDataFrame(df_with_timestamp)
            spark_df.write.mode("overwrite").saveAsTable(table_name)
            
            print(f"  ✅ Saved {len(df)} records to '{table_name}' table")
            if description:
                print(f"     📝 {description}")
                
        except Exception as e:
            print(f"  ❌ Error saving to {table_name}: {str(e)}")
    
    def get_workspaces(self):
        """Step 1: Get Workspaces"""
        print("🔍 STEP 1: Discovering workspaces...")
        
        self.workspaces_df = fabric.list_workspaces()
        self.workspaces_df = self.sanitize_df_columns(self.workspaces_df)
        self.workspaces_df = self.workspaces_df[['id', 'name', 'type']]
        
        print(f"  ✅ Found {len(self.workspaces_df)} workspaces")
        return self.workspaces_df
    
    def get_datasets_and_reports(self):
        """Step 2: Get Datasets and Reports in parallel"""
        print("\n🔍 STEP 2: Getting datasets and reports...")
        
        datasets_all, reports_all = [], []
        
        for _, ws in self.workspaces_df.iterrows():
            ws_id = ws['id']
            ws_name = ws['name']
            ws_type = ws['type']
            
            if ws_type == "AdminInsights":
                continue
                
            print(f"  📦 Scanning workspace: {ws_name}")
            
            # Get Datasets
            try:
                ds = fabric.list_datasets(workspace=ws_id)
                if not ds.empty:
                    ds['workspace_id'] = ws_id
                    ds['workspace_name'] = ws_name
                    datasets_all.append(ds)
            except Exception as e:
                print(f"    ⚠️ Datasets error in {ws_name}: {e}")
            
            # Get Reports
            try:
                rep = fabric.list_reports(workspace=ws_id)
                if not rep.empty:
                    rep['workspace_id'] = ws_id
                    rep['workspace_name'] = ws_name
                    reports_all.append(rep)
            except Exception as e:
                print(f"    ⚠️ Reports error in {ws_name}: {e}")
        
        # Combine results
        self.datasets_df = self.sanitize_df_columns(pd.concat(datasets_all, ignore_index=True) if datasets_all else pd.DataFrame())
        self.reports_df = self.sanitize_df_columns(pd.concat(reports_all, ignore_index=True) if reports_all else pd.DataFrame())
        
        # Filter PowerBI reports
        if not self.reports_df.empty and "report_type" in self.reports_df.columns:
            self.pbi_reports_df = self.reports_df[self.reports_df["report_type"] == "PowerBIReport"].copy()
        else:
            self.pbi_reports_df = self.reports_df
        
        print(f"  ✅ Found {len(self.datasets_df)} datasets and {len(self.reports_df)} reports ({len(self.pbi_reports_df)} PowerBI reports)")
        return self.datasets_df, self.reports_df
    
    def process_all_datasets(self):
        """Step 3: Process all datasets and aggregate all objects (tables, columns, measures, dependencies)"""
        print("\n🔍 STEP 3: Processing all datasets and aggregating objects...")
        
        all_columns_list = []
        all_tables_list = []
        all_measures_list = []
        all_dependencies_list = []
        all_relationships_list = []
        
        for _, ds_row in self.datasets_df.iterrows():
            ds_id = ds_row['dataset_id']
            ds_name = ds_row['dataset_name']
            ws_id = ds_row['workspace_id']
            ws_name = ds_row['workspace_name']
            
            print(f"  📊 Processing dataset: {ds_name}")
            
            # Collect comprehensive dataset info
            dataset_info = self.collect_dataset_info(ds_id, ds_name, ws_id, ws_name)
            self.all_dataset_info[ds_id] = dataset_info
            
            # Aggregate columns
            if dataset_info.columns_df is not None and not dataset_info.columns_df.empty:
                all_columns_list.append(dataset_info.columns_df)
            
            # Aggregate tables
            if dataset_info.tables_df is not None and not dataset_info.tables_df.empty:
                all_tables_list.append(dataset_info.tables_df)
            
            # Aggregate measures
            if dataset_info.measures_df is not None and not dataset_info.measures_df.empty:
                # Add additional context that might not be in the measures_df
                measures_with_context = dataset_info.measures_df.copy()
                if 'dataset_id' not in measures_with_context.columns:
                    measures_with_context['dataset_id'] = ds_id
                if 'dataset_name' not in measures_with_context.columns:
                    measures_with_context['dataset_name'] = dataset_info.ds_name
                if 'workspace_id' not in measures_with_context.columns:
                    measures_with_context['workspace_id'] = dataset_info.ws_id
                if 'workspace_name' not in measures_with_context.columns:
                    measures_with_context['workspace_name'] = dataset_info.ws_name
                all_measures_list.append(measures_with_context)
            
            # Aggregate dependencies
            if dataset_info.dependencies_df is not None and not dataset_info.dependencies_df.empty:
                all_dependencies_list.append(dataset_info.dependencies_df)
            
            # Aggregate relationships
            if dataset_info.relationships_df is not None and not dataset_info.relationships_df.empty:
                relationships_with_context = dataset_info.relationships_df.copy()
                relationships_with_context['dataset_id'] = ds_id
                relationships_with_context['dataset_name'] = dataset_info.ds_name
                relationships_with_context['workspace_id'] = dataset_info.ws_id
                relationships_with_context['workspace_name'] = dataset_info.ws_name
                all_relationships_list.append(relationships_with_context)
        
        # Combine all aggregated data
        all_columns_df = pd.concat(all_columns_list, ignore_index=True) if all_columns_list else pd.DataFrame()
        all_tables_df = pd.concat(all_tables_list, ignore_index=True) if all_tables_list else pd.DataFrame()
        all_measures_df = pd.concat(all_measures_list, ignore_index=True) if all_measures_list else pd.DataFrame()
        all_dependencies_df = pd.concat(all_dependencies_list, ignore_index=True) if all_dependencies_list else pd.DataFrame()
        all_relationships_df = pd.concat(all_relationships_list, ignore_index=True) if all_relationships_list else pd.DataFrame()
        
        print(f"  ✅ Processed {len(self.all_dataset_info)} datasets")
        print(f"    📋 Aggregated: {len(all_columns_df)} columns, {len(all_tables_df)} tables, {len(all_measures_df)} measures")
        print(f"    🔗 Aggregated: {len(all_dependencies_df)} dependencies, {len(all_relationships_df)} relationships")
        
        return all_columns_df, all_tables_df, all_measures_df, all_dependencies_df, all_relationships_df
    
    def get_reports_metadata(self):
        """Step 4: Get Reports metadata (what objects they use)"""
        print("\n🔍 STEP 4: Extracting report metadata...")
        
        if self.pbi_reports_df.empty:
            print("  ⚠️ No PowerBI reports found")
            return []
        
        for idx, report_row in self.pbi_reports_df.iterrows():
            report_id = report_row.get('id', '')
            report_name = report_row.get('name', f'Report_{idx}')
            workspace_id = report_row.get('workspace_id', '')
            workspace_name = report_row.get('workspace_name', '')
            dataset_id = report_row.get('dataset_id', '')
            
            print(f"  📊 Processing report {idx+1}/{len(self.pbi_reports_df)+1}: {report_name}")
            
            # Extract metadata
            report_metadata = self.extract_report_metadata(
                report_id, report_name, workspace_id, workspace_name, dataset_id
            )
            
            self.report_metadata_list.append(report_metadata)
            
            # Create detailed records for each object used by this report
            if report_metadata.extraction_success:
                # Add table records
                for table in report_metadata.tables:
                    self.report_objects_used.append({
                        'report_id': report_id,
                        'report_name': report_name,
                        'workspace_id': workspace_id,
                        'workspace_name': workspace_name,
                        'dataset_id': dataset_id,
                        'object_type': 'Table',
                        'object_name': table,
                        'full_reference': table,
                        'extraction_method': report_metadata.extraction_method
                    })
                
                # Add column records
                for column in report_metadata.columns:
                    table_name = column.split("'")[1]
                    column_name = column.split("'")[2].strip("[]")
                    self.report_objects_used.append({
                        'report_id': report_id,
                        'report_name': report_name,
                        'workspace_id': workspace_id,
                        'workspace_name': workspace_name,
                        'dataset_id': dataset_id,
                        'object_type': 'Column',
                        'object_name': column_name,
                        'full_reference': column,
                        'table_name': table_name,
                        'extraction_method': report_metadata.extraction_method
                    })
                
                # Add measure records
                for measure in report_metadata.measures:
                    table_name = measure.split("'")[1]
                    measure_name = measure.split("'")[2].strip("[]")
                    self.report_objects_used.append({
                        'report_id': report_id,
                        'report_name': report_name,
                        'workspace_id': workspace_id,
                        'workspace_name': workspace_name,
                        'dataset_id': dataset_id,
                        'object_type': 'Measure',
                        'object_name': measure_name,
                        'full_reference': measure,
                        'table_name': table_name,
                        'extraction_method': report_metadata.extraction_method
                    })
        
        print(f"  ✅ Processed {len(self.report_metadata_list)+1} reports, extracted {len(self.report_objects_used)} object references")
        return self.report_metadata_list
    
    def check_dependencies(self, all_columns_df, all_tables_df, all_measures_df):
        """Step 5: Check for dependencies between objects"""
        print("\n🔍 STEP 5: Checking for dependencies...")
        
        # Convert report objects to DataFrame for easier analysis
        report_objects_df = pd.DataFrame(self.report_objects_used) if self.report_objects_used else pd.DataFrame()
        
        # Get all used objects from reports
        used_tables = set()
        used_columns = set()
        used_measures = set()
        
        if not report_objects_df.empty:
            used_tables.update(report_objects_df[report_objects_df['object_type'] == 'Table']['full_reference'].tolist())
            used_columns.update(report_objects_df[report_objects_df['object_type'] == 'Column']['full_reference'].tolist())
            used_measures.update(report_objects_df[report_objects_df['object_type'] == 'Measure']['full_reference'].tolist())
        
        print(f"  📋 Initial objects from reports: {len(used_tables)} tables, {len(used_columns)} columns, {len(used_measures)} measures")
        
        # Check for dependencies within datasets (relationships and transitive dependencies)
        for ds_id, dataset_info in self.all_dataset_info.items():
            # Check relationships - columns used in relationships are required
            if dataset_info.relationships_df is not None and not dataset_info.relationships_df.empty:
                for _, rel in dataset_info.relationships_df.iterrows():
                    if 'qualified_from' in rel:
                        used_columns.add(rel['qualified_from'])
                    if 'qualified_to' in rel:
                        used_columns.add(rel['qualified_to'])
        
        print(f"  🔗 After adding relationship columns: {len(used_columns)} columns")
        
        # Transitive dependency resolution: find what the used objects depend on
        # Keep iterating until no new dependencies are found
        iteration = 0
        max_iterations = 10  # Prevent infinite loops
        
        while iteration < max_iterations:
            iteration += 1
            initial_tables_count = len(used_tables)
            initial_columns_count = len(used_columns)
            initial_measures_count = len(used_measures)
            
            print(f"  🔄 Dependency resolution iteration {iteration}...")
            
            # Check dependencies for all used objects
            for ds_id, dataset_info in self.all_dataset_info.items():
                if dataset_info.dependencies_df is None or dataset_info.dependencies_df.empty:
                    continue
                
                # Iterate through each dependency row
                for _, dep in dataset_info.dependencies_df.iterrows():
                    # Get the full_object_name (the object that has the dependency)
                    full_object_name = dep.get('full_object_name', '')
                    
                    # Check if this object is in our used sets
                    if full_object_name in used_columns or full_object_name in used_measures:
                        # This object is used, so we need to mark its dependencies as used too
                        ref_object_type = dep.get('referenced_object_type', '')
                        referenced_full_object_name = dep.get('referenced_full_object_name', '')
                        
                        if ref_object_type == 'Table':
                            # The used object depends on a table
                            table_name = dep.get('referenced_table', '')
                            if table_name:
                                used_tables.add(table_name)
                        
                        elif ref_object_type == 'Column':
                            # The used object depends on a column
                            if referenced_full_object_name:
                                used_columns.add(referenced_full_object_name)
                        
                        elif ref_object_type == 'Measure':
                            # The used object depends on a measure
                            if referenced_full_object_name:
                                used_measures.add(referenced_full_object_name)
            
            # Check if we found any new dependencies
            new_tables = len(used_tables) - initial_tables_count
            new_columns = len(used_columns) - initial_columns_count
            new_measures = len(used_measures) - initial_measures_count
            
            print(f"    ➕ Added: {new_tables} tables, {new_columns} columns, {new_measures} measures")
            
            # If no new dependencies were found, we're done
            if new_tables == 0 and new_columns == 0 and new_measures == 0:
                print(f"  ✅ Dependency resolution converged after {iteration} iteration(s)")
                break
        
        print(f"  ✅ Final dependencies: {len(used_tables)} tables, {len(used_columns)} columns, {len(used_measures)} measures")
        
        return {
            'used_tables': used_tables,
            'used_columns': used_columns,
            'used_measures': used_measures,
            'report_objects_df': report_objects_df
        }
    
    def filter_results(self, all_columns_df, all_tables_df, all_measures_df, dependencies):
        """Step 6: Filter results to identify used vs unused objects"""
        print("\n🔍 STEP 6: Filtering results to identify used vs unused objects...")
        
        used_tables = dependencies['used_tables']
        used_columns = dependencies['used_columns']
        used_measures = dependencies['used_measures']
        
        # Filter columns
        if not all_columns_df.empty:
            if 'qualified_name' in all_columns_df.columns:
                all_columns_df['is_used'] = all_columns_df['qualified_name'].isin(used_columns)
            else:
                # Create qualified name if it doesn't exist
                all_columns_df['qualified_name'] = "'" + all_columns_df['table_name'] + "'[" + all_columns_df['column_name'] + ']'
                all_columns_df['is_used'] = all_columns_df['qualified_name'].isin(used_columns)
            
            used_columns_df = all_columns_df[all_columns_df['is_used'] == True].copy()
            unused_columns_df = all_columns_df[all_columns_df['is_used'] == False].copy()
        else:
            used_columns_df = pd.DataFrame()
            unused_columns_df = pd.DataFrame()
        
        # Filter tables
        if not all_tables_df.empty:
            all_tables_df['is_used'] = all_tables_df['name'].isin(used_tables)
            used_tables_df = all_tables_df[all_tables_df['is_used'] == True].copy()
            unused_tables_df = all_tables_df[all_tables_df['is_used'] == False].copy()
        else:
            used_tables_df = pd.DataFrame()
            unused_tables_df = pd.DataFrame()
        
        # Filter measures
        if not all_measures_df.empty:
            # Create qualified measure name for comparison
            all_measures_df['qualified_name'] = "'" + all_measures_df['table_name'] + "'[" + all_measures_df['measure_name'] + "]"
            all_measures_df['is_used'] = all_measures_df['qualified_name'].isin(used_measures)
            used_measures_df = all_measures_df[all_measures_df['is_used'] == True].copy()
            unused_measures_df = all_measures_df[all_measures_df['is_used'] == False].copy()
        else:
            used_measures_df = pd.DataFrame()
            unused_measures_df = pd.DataFrame()
        
        print(f"  ✅ Results filtered:")
        print(f"    Used: {len(used_tables_df)} tables, {len(used_columns_df)} columns, {len(used_measures_df)} measures")
        print(f"    Unused: {len(unused_tables_df)} tables, {len(unused_columns_df)} columns, {len(unused_measures_df)} measures")
        
        return {
            'used_tables': used_tables_df,
            'used_columns': used_columns_df,
            'used_measures': used_measures_df,
            'unused_tables': unused_tables_df,
            'unused_columns': unused_columns_df,
            'unused_measures': unused_measures_df
        }
    
    def collect_dataset_info(self, ds_id: str, ds_name: str, ws_id: str, ws_name: str) -> DatasetInfo:
        """Centralized function to collect all dataset-related information"""
        dataset_info = DatasetInfo(ds_id, ds_name, ws_id, ws_name)
        
        # Get model dependencies
        try:
            deps = fabric.get_model_calc_dependencies(dataset=ds_id, workspace=ws_id)
            with deps as calc_deps:
                dependencies_df = getattr(calc_deps, "dependencies_df", None)
            
            if dependencies_df is not None and not dependencies_df.empty:
                dependencies_df = self.sanitize_df_columns(
                    df=dependencies_df, 
                    extra_columns=True,
                    ws_id=ws_id, 
                    ds_id=ds_id,
                    ws_name=ws_name,
                    ds_name=ds_name
                )
                dataset_info.dependencies_df = dependencies_df
            else:
                dataset_info.dependencies_df = pd.DataFrame()
        except Exception as e:
            print(f"    ⚠️ Dependencies unavailable for {ds_name}: {e}")
            dataset_info.dependencies_df = pd.DataFrame()

        # Get tables
        try:
            tables = fabric.list_tables(dataset=ds_id, workspace=ws_id)
            if not tables.empty:
                tables = self.sanitize_df_columns(
                    df=tables, 
                    extra_columns=True,
                    ws_id=ws_id, 
                    ds_id=ds_id,
                    ws_name=ws_name,
                    ds_name=ds_name
                )
                dataset_info.tables_df = tables
        except Exception as e:
            print(f"    ⚠️ Tables unavailable for {ds_name}: {e}")
            
        # Get relationships
        try:
            relationships = fabric.list_relationships(dataset=ds_id, workspace=ws_id, extended=True)
            if not relationships.empty:
                relationships = self.sanitize_df_columns(df=relationships)
                relationships['qualified_from'] = "'" + relationships['from_table'] + "'[" + relationships['from_column'] + "]"
                relationships['qualified_to'] = "'" + relationships['to_table'] + "'[" + relationships['to_column'] + "]"
                dataset_info.relationships_df = relationships
        except Exception as e:
            print(f"    ⚠️ Relationships unavailable for {ds_name}: {e}")

        # Get measures
        try:
            measures = fabric.list_measures(dataset=ds_id, workspace=ws_id)
            if not measures.empty:
                measures = self.sanitize_df_columns(df=measures)
                dataset_info.measures_df = measures
        except Exception as e:
            print(f"    ⚠️ Measures unavailable for {ds_name}: {e}")

        # Get columns
        try:
            columns = fabric.list_columns(dataset=ds_id, workspace=ws_id, extended=True)
            if not columns.empty:
                columns = self.sanitize_df_columns(
                    df=columns,
                    extra_columns=True,
                    ws_id=ws_id, 
                    ds_id=ds_id,
                    ws_name=ws_name,
                    ds_name=ds_name
                )
                columns['qualified_name'] = "'" + columns['table_name'] + "'[" + columns['column_name'] + ']'
                dataset_info.columns_df = columns
        except Exception as e:
            print(f"    ⚠️ Columns unavailable for {ds_name}: {e}")
        
        return dataset_info
    
    def extract_report_metadata(self, report_id: str, report_name: str, workspace_id: str, workspace_name: str, dataset_id: str) -> ReportMetadata:
        """Extract metadata from Power BI reports using dual approach"""
        
        # Initialize result object
        result = ReportMetadata(
            report_id=report_id,
            report_name=report_name,
            workspace_id=workspace_id,
            workspace_name=workspace_name,
            dataset_id=dataset_id,
            report_format="Unknown",
            extraction_method="None",
            tables=[],
            columns=[],
            measures=[],
            visuals_count=0,
            filters_count=0,
            extraction_success=False
        )
        
        try:
            # Step 1: Try to determine report format
            report = ReportWrapper(report=report_id, workspace=workspace_id)
            rep_format = report.format
            result.report_format = rep_format
            print(f"  📑 Report Type: {rep_format}")
            if rep_format == "PBIR":
                # Method 1: Use sempy_labs.report.list_all_semantic_model_objects() for PBIR format
                try:
                    objects = report.list_semantic_model_objects()
                    
                    if objects is not None and not objects.empty:
                        # Process the objects DataFrame
                        tables = objects['Table Name'].unique().tolist()
                        columns = (
                            objects[objects['Object Type'] == 'Column']
                            .assign(qualified=lambda df: "'" + df['Table Name'].fillna('') + "'[" + df['Object Name'] + "]")['qualified'] #build 'table'[column]
                            .unique().tolist()
                        )
                        measures = (
                            objects[objects['Object Type'] == 'Measure']
                            .assign(qualified = lambda df: "'" + df['Table Name'].fillna('') + "'[" + df['Object Name'] + "]")["qualified"] #build 'table'[measure]
                            .unique().tolist()
                        )
                        
                        result.tables = tables
                        result.columns = columns
                        result.measures = measures
                        result.extraction_method = "sempy_labs_objects"
                        result.extraction_success = True
                        
                        print(f"    ✅ Extracted via sempy_labs: {len(tables)} tables, {len(columns)} columns, {len(measures)} measures")
                        return result
                        
                except NotImplementedError as e:
                    print(f"    ⚠️ sempy_labs method not supported: {str(e)}")
                except Exception as e:
                    print(f"    ⚠️ sempy_labs method failed: {str(e)}")
            
            # Method 2: Fall back to JSON parsing
            report_json = sempy_labs.report.get_report_json(report=report_id, workspace=workspace_id)
            
            if report_json:
                # Use our custom extractor
                extractor = PowerBIMetadataExtractor()
                extraction_results = extractor.extract_from_json_data(report_json)
                
                result.tables = extraction_results.get('tables', [])
                result.columns = extraction_results.get('columns', [])
                result.measures = extraction_results.get('measures', [])
                result.visuals_count = len(extraction_results.get('visual_details', []))
                result.filters_count = len(extraction_results.get('filter_details', []))
                result.extraction_method = "json_parsing"
                result.extraction_success = True
                
                print(f"    ✅ Extracted via JSON: {len(result.tables)} tables, {len(result.columns)} columns, {len(result.measures)} measures")
                return result
            else:
                result.error_message = "Could not retrieve report JSON"
                
        except Exception as e:
            result.error_message = f"Extraction failed: {str(e)}"
            print(f"    ❌ Error extracting metadata: {str(e)}")
        
        return result
    
    def save_all_results(self, all_columns_df, all_tables_df, all_measures_df, all_dependencies_df, all_relationships_df, filtered_results, dependencies):
        """Save all results to lakehouse"""
        print("\n💾 STEP 7: Saving all results to lakehouse...")
        
        # Save workspace information
        self.save_to_lakehouse(self.workspaces_df, "workspace_analysis", "Workspace information")
        
        # Save dataset information
        self.save_to_lakehouse(self.datasets_df, "dataset_analysis", "Dataset information with workspace context")
        
        # Save all objects
        self.save_to_lakehouse(all_columns_df, "all_columns_analysis", "All columns from all datasets")
        self.save_to_lakehouse(all_tables_df, "all_tables_analysis", "All tables from all datasets")
        self.save_to_lakehouse(all_measures_df, "all_measures_analysis", "All measures from all datasets")
        self.save_to_lakehouse(all_dependencies_df, "all_dependencies_analysis", "All dependencies from all datasets")
        self.save_to_lakehouse(all_relationships_df, "all_relationships_analysis", "All relationships from all datasets")
        
        # Save filtered results
        self.save_to_lakehouse(filtered_results['used_columns'], "used_columns", "Columns that are used by reports or dependencies")
        self.save_to_lakehouse(filtered_results['used_tables'], "used_tables", "Tables that are used by reports or dependencies")
        self.save_to_lakehouse(filtered_results['used_measures'], "used_measures", "Measures that are used by reports or dependencies")
        
        self.save_to_lakehouse(filtered_results['unused_columns'], "unused_columns", "Columns that are NOT used by reports or dependencies")
        self.save_to_lakehouse(filtered_results['unused_tables'], "unused_tables", "Tables that are NOT used by reports or dependencies")
        self.save_to_lakehouse(filtered_results['unused_measures'], "unused_measures", "Measures that are NOT used by reports or dependencies")
        
        # Save report metadata
        if self.report_metadata_list:
            report_metadata_records = []
            for metadata in self.report_metadata_list:
                record = {
                    'report_id': metadata.report_id,
                    'report_name': metadata.report_name,
                    'workspace_id': metadata.workspace_id,
                    'workspace_name': metadata.workspace_name,
                    'dataset_id': metadata.dataset_id,
                    'report_format': metadata.report_format,
                    'extraction_method': metadata.extraction_method,
                    'tables_count': len(metadata.tables),
                    'columns_count': len(metadata.columns),
                    'measures_count': len(metadata.measures),
                    'visuals_count': metadata.visuals_count,
                    'filters_count': metadata.filters_count,
                    'extraction_success': metadata.extraction_success,
                    'error_message': metadata.error_message,
                    'tables_list': ','.join(metadata.tables) if metadata.tables else '',
                    'columns_list': ','.join(metadata.columns) if metadata.columns else '',
                    'measures_list': ','.join(metadata.measures) if metadata.measures else ''
                }
                report_metadata_records.append(record)
            
            report_metadata_df = pd.DataFrame(report_metadata_records)
            self.save_to_lakehouse(report_metadata_df, "report_metadata_analysis", "PowerBI report metadata extraction results")
        
        # Save detailed report objects usage
        if self.report_objects_used:
            report_objects_df = pd.DataFrame(self.report_objects_used)
            self.save_to_lakehouse(report_objects_df, "report_objects_used", "Detailed breakdown of objects used by each PowerBI report")
        
        print("  ✅ All results saved to lakehouse!")
    
    def run_complete_analysis(self):
        """Run the complete analysis workflow"""
        print("🚀 STARTING COMPLETE FABRIC WORKSPACE ANALYSIS")
        print("=" * 80)
        
        start_time = time.time()
        
        # Step 1: Get Workspaces
        self.get_workspaces()
        
        # Step 2: Get Datasets and Reports
        self.get_datasets_and_reports()
        
        # Step 3: Process all datasets and aggregate all objects
        all_columns_df, all_tables_df, all_measures_df, all_dependencies_df, all_relationships_df = self.process_all_datasets()
        
        # Step 4: Get report metadata
        self.get_reports_metadata()
        
        # Step 5: Check dependencies
        dependencies = self.check_dependencies(all_columns_df, all_tables_df, all_measures_df)
        
        # Step 6: Filter results
        filtered_results = self.filter_results(all_columns_df, all_tables_df, all_measures_df, dependencies)
        
        # Step 7: Save all results
        self.save_all_results(all_columns_df, all_tables_df, all_measures_df, all_dependencies_df, all_relationships_df, filtered_results, dependencies)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 FABRIC WORKSPACE ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"⏱️ Total execution time: {duration:.2f} seconds")
        print(f"\n📊 Summary:")
        print(f"  Workspaces analyzed: {len(self.workspaces_df)}")
        print(f"  Datasets processed: {len(self.datasets_df)}")
        print(f"  Reports analyzed: {len(self.report_metadata_list)}")
        print(f"  Total objects found: {len(all_columns_df)} columns, {len(all_tables_df)} tables, {len(all_measures_df)} measures")
        print(f"  Used objects: {len(filtered_results['used_columns'])} columns, {len(filtered_results['used_tables'])} tables, {len(filtered_results['used_measures'])} measures")
        print(f"  Unused objects: {len(filtered_results['unused_columns'])} columns, {len(filtered_results['unused_tables'])} tables, {len(filtered_results['unused_measures'])} measures")
        print("\n💾 All results saved to lakehouse tables!")
        print("=" * 80)
