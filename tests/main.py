import requests
import msal
import time
import json

# -----------------------
# Configuration - Update these values
# -----------------------
AZURE_CLIENT_ID = "ee72cedb-edfb-421e-b82e-b5c9e1126d33"
AZURE_TENANT_ID = "bf42f62c-baed-4cc3-bc32-e00602a34f67"

SCOPE = ["https://analysis.windows.net/powerbi/api/.default"]
BASE_URL = "https://api.powerbi.com/v1.0/myorg/admin"
OUTPUT_FILE = "ws_report_metadata_results.json"

# -----------------------
# Initialize MSAL App
# -----------------------
app = msal.PublicClientApplication(
    AZURE_CLIENT_ID,
    authority=f"https://login.microsoftonline.com/{AZURE_TENANT_ID}"
)


def get_access_token():
    """Acquire token interactively or silently"""
    result = None
    accounts = app.get_accounts()

    if accounts:
        print("Found cached account, trying silent authentication...")
        result = app.acquire_token_silent(SCOPE, account=accounts[0])

    if not result or "access_token" not in result:
        print("Opening browser for authentication...")
        result = app.acquire_token_interactive(scopes=SCOPE, prompt="select_account")

    if "access_token" in result:
        print("✅ Authentication successful")
        return result["access_token"]
    else:
        raise Exception(f"Authentication failed: {result.get('error_description', 'Unknown error')}")


def get_all_workspaces(access_token):
    """Fetch all available workspaces"""
    url = "https://api.powerbi.com/v1.0/myorg/groups"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    workspaces = []
    while url:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        workspaces.extend(data.get("value", []))
        url = data.get("@odata.nextLink")  # pagination support
    return workspaces



def start_scan(access_token, workspace_id):
    """Start workspace scan job and return scanId"""
    url = f"{BASE_URL}/workspaces/getInfo?lineage=True&datasourceDetails=True&datasetSchema=True&datasetExpressions=True"
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {"workspaces": [workspace_id]}

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    scan_id = data["id"]
    print(f"🚀 Scan started for Workspace {workspace_id}. Scan ID: {scan_id}")
    return scan_id


def poll_scan_status(access_token, scan_id, interval=30):
    """Poll scan status until it's succeeded or failed"""
    url = f"{BASE_URL}/workspaces/scanStatus/{scan_id}"
    headers = {"Authorization": f"Bearer {access_token}"}

    while True:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")

        print(f"⏳ Current scan status: {status}")
        if status == "Succeeded":
            print("✅ Scan completed successfully")
            return True
        elif status == "Failed":
            print("❌ Scan failed")
            return False

        time.sleep(interval)


def get_scan_result(access_token, scan_id):
    """Retrieve scan results after completion"""
    url = f"{BASE_URL}/workspaces/scanResult/{scan_id}"
    headers = {"Authorization": f"Bearer {access_token}"}

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    # Step 1: Authenticate
    token = get_access_token()

    # Step 2: Get all workspaces
    workspaces = get_all_workspaces(token)
    print(f"🔍 Found {len(workspaces)} workspaces.")

    all_results = []

    # Step 3: Loop through and scan each workspace
    for ws in workspaces:
        ws_id = ws["id"]
        ws_name = ws.get("name", "Unnamed")
        print(f"\n=== Processing Workspace: {ws_name} ({ws_id}) ===")

        try:
            # Start scan
            scan_id = start_scan(token, ws_id)

            # Poll until completed
            if poll_scan_status(token, scan_id, interval=10):
                # Get results
                results = get_scan_result(token, scan_id)

                # Attach workspace info
                workspace_result = {
                    "workspaceId": ws_id,
                    "workspaceName": ws_name,
                    "scanResult": results
                }
                all_results.append(workspace_result)

                # Quick summary in console
                reports = results["workspaces"][0].get("reports", [])
                datasets = results["workspaces"][0].get("datasets", [])
                print(f"📊 {ws_name}: {len(reports)} reports, {len(datasets)} datasets")

        except Exception as e:
            print(f"⚠️ Failed to process workspace {ws_name} ({ws_id}): {e}")

    # Step 4: Save all results to JSON
    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    # print(f"\n✅ All scan results saved to {OUTPUT_FILE}")

    all_reports = []
    all_datasets = []


    for item in all_results:  # loop through data[0], data[1], ...
        try:
            ws = item["scanResult"]["workspaces"][0]
            ws_id = ws.get("id")
            ws_name = ws.get("name")

            reports = ws.get("reports", [])
            for report in reports:
                report_copy = dict(report)  # avoid mutating original
                # prepend ws fields
                enriched_report = {"report_ws_id": ws_id, "report_ws_name": ws_name, **report_copy}
                all_reports.append(enriched_report)

        except (KeyError, IndexError):
            # In case scanResult/workspaces/reports is missing
            continue 
    
    for item in all_results:  # loop through data[0], data[1], ...
        try:
            ws = item["scanResult"]["workspaces"][0]
            ws_id = ws.get("id")
            ws_name = ws.get("name")

            datasets = ws.get("datasets", [])
            for dataset in datasets:
                dataset_copy = dict(dataset)  # avoid mutating original
                enriched_dataset = {"dataset_ws_id": ws_id, "dataset_ws_name": ws_name, **dataset_copy}
                all_datasets.append(enriched_dataset)

        except (KeyError, IndexError):
            # In case scanResult/workspaces/datasets is missing
            continue 
    
        # Build a quick lookup dict for datasets by id
    dataset_lookup = {ds["id"]: ds for ds in all_datasets}

    # Merge tables into reports
    enriched_reports = []
    for report in all_reports:
        dataset_id = report.get("datasetId")
        dataset = dataset_lookup.get(dataset_id)

        # Copy the report to avoid mutating original
        report_copy = dict(report)
        if dataset:
            report_copy["dataset_ws_id"] = dataset.get("dataset_ws_id", [])
            report_copy["dataset_ws_name"] = dataset.get("dataset_ws_name", [])
            report_copy["dataset_name"] = dataset.get("name", [])
            report_copy["tables"] = dataset.get("tables", [])

        else:
            report_copy["tables"] = []  # no dataset match
            report_copy["dataset_ws_id"] = []
            report_copy["dataset_ws_name"] = []
        
        enriched_reports.append(report_copy)
    
        # Step 4: Save all results to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(enriched_reports, f, indent=4)

    print(f"\n✅ All scan results saved to {OUTPUT_FILE}")