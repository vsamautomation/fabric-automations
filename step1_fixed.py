# ------------------------------------------------------------
# STEP 1: Object Discovery (ENHANCED with workspace object counts) - FIXED
# ------------------------------------------------------------

print("🔍 Discovering workspaces...")

workspaces_df = fabric.list_workspaces()
workspaces_df = sanitize_df_columns(workspaces_df)
workspaces_df = workspaces_df[['id', 'name', 'type']]
display(workspaces_df)

datasets_all, reports_all, paginated_all, dataflows_all = [], [], [], []

for _, ws in workspaces_df.iterrows():
    ws_id = ws['id']
    ws_name = ws['name']
    ws_type = ws['type']
    if ws_type == "AdminInsights":
        continue
    print(f"\n📦 Scanning workspace: {ws_name}")

   # --- Datasets
    try:
        ds = fabric.list_datasets(workspace=ws_id)
        if not ds.empty:
            ds['workspace_id'] = ws_id
            ds['workspace_name'] = ws_name
            datasets_all.append(ds)
    except Exception as e:
        print(f"  ⚠️ Datasets error in {ws_name}: {e}")

    # --- Reports (includes both Power BI and Paginated)
    try:
        rep = fabric.list_reports(workspace=ws_id)
        if not rep.empty:
            rep['workspace_id'] = ws_id
            rep['workspace_name'] = ws_name
            reports_all.append(rep)
    except Exception as e:
        print(f"  ⚠️ Reports error in {ws_name}: {e}")

    # --- Dataflows
    try:
        dfs = fabric.list_items(type='Dataflow',workspace=ws_id)
        if not dfs.empty:
            dfs['workspace_id'] = ws_id
            dfs['workspace_name'] = ws_name
            dataflows_all.append(dfs)
    except Exception as e:
        print(f"  ⚠️ Dataflows error in {ws_name}: {e}")

# Combine results
datasets_df  = sanitize_df_columns(pd.concat(datasets_all, ignore_index=True) if datasets_all else pd.DataFrame())
reports_df   = sanitize_df_columns(pd.concat(reports_all, ignore_index=True) if reports_all else pd.DataFrame())
dataflows_df = sanitize_df_columns(pd.concat(dataflows_all, ignore_index=True) if dataflows_all else pd.DataFrame())

# Fix duplicate columns issue
def fix_duplicate_columns(df, df_name):
    """Remove duplicate columns if they exist"""
    if df.empty:
        return df
    
    # Check for duplicate columns
    if df.columns.duplicated().any():
        print(f"⚠️ Found duplicate columns in {df_name}, removing duplicates...")
        print(f"   Original columns: {list(df.columns)}")
        # Keep only the first occurrence of each column name
        df = df.loc[:, ~df.columns.duplicated()]
        print(f"   After removing duplicates: {list(df.columns)}")
    
    return df

# Apply fix to all dataframes
datasets_df = fix_duplicate_columns(datasets_df, "datasets")
reports_df = fix_duplicate_columns(reports_df, "reports") 
dataflows_df = fix_duplicate_columns(dataflows_df, "dataflows")

# Split report types for clarity
if not reports_df.empty and "report_type" in reports_df.columns:
    pbi_reports_df = reports_df[reports_df["report_type"] == "PowerBIReport"].copy()
    paginated_reports_df = reports_df[reports_df["report_type"] == "PaginatedReport"].copy()
else:
    pbi_reports_df = reports_df
    paginated_reports_df = pd.DataFrame()

# 🆕 ADD OBJECT COUNTS TO WORKSPACE DATAFRAME
print("\n📊 Adding object counts to workspace dataframe...")

# Initialize count columns
workspaces_df['dataset_count'] = 0
workspaces_df['total_reports'] = 0
workspaces_df['pbi_reports'] = 0
workspaces_df['paginated_reports'] = 0
workspaces_df['dataflows'] = 0

# Count objects per workspace with error handling
try:
    if not datasets_df.empty and 'workspace_id' in datasets_df.columns:
        dataset_counts = datasets_df['workspace_id'].value_counts().to_dict()
        workspaces_df['dataset_count'] = workspaces_df['id'].map(dataset_counts).fillna(0).astype(int)
except Exception as e:
    print(f"⚠️ Error counting datasets: {e}")

try:
    if not reports_df.empty and 'workspace_id' in reports_df.columns:
        # Total reports count
        total_report_counts = reports_df['workspace_id'].value_counts().to_dict()
        workspaces_df['total_reports'] = workspaces_df['id'].map(total_report_counts).fillna(0).astype(int)
        
        # PBI reports count
        if not pbi_reports_df.empty and 'workspace_id' in pbi_reports_df.columns:
            pbi_counts = pbi_reports_df['workspace_id'].value_counts().to_dict()
            workspaces_df['pbi_reports'] = workspaces_df['id'].map(pbi_counts).fillna(0).astype(int)
        
        # Paginated reports count
        if not paginated_reports_df.empty and 'workspace_id' in paginated_reports_df.columns:
            paginated_counts = paginated_reports_df['workspace_id'].value_counts().to_dict()
            workspaces_df['paginated_reports'] = workspaces_df['id'].map(paginated_counts).fillna(0).astype(int)
except Exception as e:
    print(f"⚠️ Error counting reports: {e}")

try:
    if not dataflows_df.empty and 'workspace_id' in dataflows_df.columns:
        print(f"   Dataflows columns: {list(dataflows_df.columns)}")
        print(f"   Dataflows shape: {dataflows_df.shape}")
        dataflow_counts = dataflows_df['workspace_id'].value_counts().to_dict()
        workspaces_df['dataflows'] = workspaces_df['id'].map(dataflow_counts).fillna(0).astype(int)
except Exception as e:
    print(f"⚠️ Error counting dataflows: {e}")
    print(f"   Dataflows columns: {list(dataflows_df.columns) if not dataflows_df.empty else 'Empty DataFrame'}")

print("\n✅ Object discovery complete with enhanced workspace context.")
print(f"  Workspaces: {len(workspaces_df)}")
print(f"  Datasets:   {len(datasets_df)}")
print(f"  Reports:    {len(reports_df)} (PBI: {len(pbi_reports_df)}, Paginated: {len(paginated_reports_df)})")
print(f"  Dataflows:  {len(dataflows_df)}")

# Display enhanced workspace summary
print("\n📋 Workspace Object Summary:")
workspace_summary = workspaces_df[['name', 'dataset_count', 'total_reports', 'pbi_reports', 'paginated_reports', 'dataflows']]
display(workspace_summary)

# Save to Lakehouse - Enhanced Workspaces
print("\n💾 Saving enhanced workspace data to lakehouse...")
save_to_lakehouse(workspaces_df, "workspace_analysis", "Workspace information with object counts")