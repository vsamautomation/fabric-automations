# ------------------------------------------------------------
# STEP 1: Object Discovery
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

# Count objects per workspace
if not datasets_df.empty:
    dataset_counts = datasets_df['workspace_id'].value_counts().to_dict()
    workspaces_df['dataset_count'] = workspaces_df['id'].map(dataset_counts).fillna(0).astype(int)

if not reports_df.empty:
    # Total reports count
    total_report_counts = reports_df['workspace_id'].value_counts().to_dict()
    workspaces_df['total_reports'] = workspaces_df['id'].map(total_report_counts).fillna(0).astype(int)
    
    # PBI reports count
    if not pbi_reports_df.empty:
        pbi_counts = pbi_reports_df['workspace_id'].value_counts().to_dict()
        workspaces_df['pbi_reports'] = workspaces_df['id'].map(pbi_counts).fillna(0).astype(int)
    
    # Paginated reports count
    if not paginated_reports_df.empty:
        paginated_counts = paginated_reports_df['workspace_id'].value_counts().to_dict()
        workspaces_df['paginated_reports'] = workspaces_df['id'].map(paginated_counts).fillna(0).astype(int)

if not dataflows_df.empty:
    dataflow_counts = dataflows_df['workspace_id'].value_counts().to_dict()
    workspaces_df['dataflows'] = workspaces_df['id'].map(dataflow_counts).fillna(0).astype(int)

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