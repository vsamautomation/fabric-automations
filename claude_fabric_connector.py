"""
Fabric Notebook Compatible Claude Connector
Uses direct HTTP requests instead of the anthropic library to avoid dependency issues
"""

import json
import requests
from typing import Dict, Any, Optional
from datetime import datetime
import os

class FabricClaudeConnector:
    """Claude connector that works in Microsoft Fabric environments"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key"""
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set ANTHROPIC_API_KEY or pass directly.")
        
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        print("✅ Fabric Claude connector initialized")
    
    def _make_request(self, messages: list, model: str = "claude-3-haiku-20240307", max_tokens: int = 1000) -> dict:
        """Make direct HTTP request to Claude API"""
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise Exception("❌ Authentication failed. Check your API key.")
            elif response.status_code == 429:
                raise Exception("⚠️ Rate limit exceeded. Please wait and try again.")
            else:
                raise Exception(f"❌ API request failed: {response.status_code} - {response.text}")
                
        except requests.RequestException as e:
            raise Exception(f"❌ Network error: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test basic connection to Claude API"""
        try:
            print("🔄 Testing Claude API connection...")
            
            messages = [{
                "role": "user", 
                "content": "Hello! This is a connection test. Please respond with 'Connection successful' and today's date."
            }]
            
            result = self._make_request(messages, max_tokens=100)
            
            if 'content' in result and len(result['content']) > 0:
                response_text = result['content'][0]['text']
                print("✅ Connection successful!")
                print(f"📝 Claude's response: {response_text}")
                return True
            else:
                print("❌ Unexpected response format")
                return False
                
        except Exception as e:
            print(f"❌ Connection test failed: {str(e)}")
            return False
    
    def analyze_fabric_metadata(self, metadata: Dict[str, Any]) -> str:
        """Analyze Fabric metadata and suggest schemas"""
        try:
            print("🔄 Analyzing Fabric metadata with Claude...")
            
            prompt = f"""
            As a data architecture expert, analyze this Microsoft Fabric workspace metadata and provide specific recommendations for Lakehouse and Data Warehouse implementation:

            Metadata: {json.dumps(metadata, indent=2)}

            Please provide structured recommendations in the following format:

            ## LAKEHOUSE SCHEMAS (Delta Tables)
            [Suggest Delta table schemas with columns, data types, and partitioning]

            ## DATA WAREHOUSE DESIGN
            [Suggest dimensional model with fact and dimension tables]

            ## STORED PROCEDURES
            [Essential T-SQL stored procedures for data processing]

            ## DATA PIPELINES
            [Recommended data pipeline structure and flow]

            ## PERFORMANCE OPTIMIZATIONS
            [Indexing, partitioning, and optimization recommendations]

            Keep recommendations practical and implementable in Microsoft Fabric.
            """
            
            messages = [{"role": "user", "content": prompt}]
            
            result = self._make_request(
                messages, 
                model="claude-3-sonnet-20240229",  # More capable model for analysis
                max_tokens=3000
            )
            
            if 'content' in result and len(result['content']) > 0:
                analysis = result['content'][0]['text']
                print("✅ Metadata analysis complete!")
                return analysis
            else:
                return "❌ Failed to get analysis from Claude"
                
        except Exception as e:
            return f"❌ Analysis failed: {str(e)}"
    
    def generate_sql_scripts(self, schema_analysis: str) -> str:
        """Generate actual SQL scripts based on schema analysis"""
        try:
            prompt = f"""
            Based on this schema analysis, generate actual T-SQL scripts for Microsoft Fabric:

            {schema_analysis}

            Please provide:
            1. CREATE TABLE statements for Lakehouse (Delta format)
            2. CREATE TABLE statements for Data Warehouse
            3. CREATE PROCEDURE statements for common operations
            4. Sample INSERT/MERGE statements for data loading

            Format as executable T-SQL code blocks.
            """
            
            messages = [{"role": "user", "content": prompt}]
            
            result = self._make_request(
                messages,
                model="claude-3-sonnet-20240229",
                max_tokens=2500
            )
            
            if 'content' in result and len(result['content']) > 0:
                return result['content'][0]['text']
            else:
                return "❌ Failed to generate SQL scripts"
                
        except Exception as e:
            return f"❌ SQL generation failed: {str(e)}"

# Test function for Fabric Notebook
def test_fabric_claude_connection(api_key: str = None):
    """Test function specifically for Fabric notebooks"""
    
    print("🚀 Testing Claude Connection in Fabric Notebook")
    print("=" * 55)
    
    # Get API key from parameter or environment
    if not api_key:
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("❌ No API key provided. Usage:")
            print("   test_fabric_claude_connection('your-api-key')")
            print("   or set ANTHROPIC_API_KEY environment variable")
            return None
    
    try:
        # Initialize connector
        connector = FabricClaudeConnector(api_key)
        
        # Test connection
        if not connector.test_connection():
            print("❌ Connection test failed")
            return None
        
        print("\n" + "=" * 55)
        print("📊 TESTING WITH SAMPLE FABRIC METADATA")
        
        # Sample metadata similar to your Fabric discovery
        sample_metadata = {
            "discovery_info": {
                "timestamp": datetime.now().isoformat(),
                "environment": "Microsoft Fabric"
            },
            "workspaces": [
                {
                    "id": "ws-analytics-001",
                    "name": "Corporate Analytics",
                    "type": "Workspace", 
                    "dataset_count": 12,
                    "total_reports": 25,
                    "pbi_reports": 20,
                    "paginated_reports": 5,
                    "dataflows": 4
                },
                {
                    "id": "ws-sales-002", 
                    "name": "Sales Intelligence",
                    "type": "Workspace",
                    "dataset_count": 8,
                    "total_reports": 15,
                    "pbi_reports": 15,
                    "paginated_reports": 0,
                    "dataflows": 2
                }
            ],
            "datasets": [
                {
                    "id": "ds-sales-001",
                    "name": "Sales Performance Dataset",
                    "workspace_id": "ws-sales-002",
                    "configured_by": "analytics-team@company.com",
                    "refresh_schedule": "Daily"
                },
                {
                    "id": "ds-finance-001", 
                    "name": "Financial Reporting Dataset",
                    "workspace_id": "ws-analytics-001",
                    "configured_by": "finance-team@company.com", 
                    "refresh_schedule": "Hourly"
                }
            ],
            "summary": {
                "total_workspaces": 2,
                "total_datasets": 20,
                "total_reports": 40,
                "total_dataflows": 6
            }
        }
        
        # Get analysis
        analysis = connector.analyze_fabric_metadata(sample_metadata)
        
        print("\n📋 AI-GENERATED ANALYSIS:")
        print(analysis)
        
        # Save results
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "connection_successful": True,
            "sample_metadata": sample_metadata,
            "claude_analysis": analysis
        }
        
        # In Fabric, you might save this differently
        with open("fabric_claude_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to 'fabric_claude_test_results.json'")
        print("\n🎉 Test completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Use your actual Fabric metadata with connector.analyze_fabric_metadata()")
        print("   2. Generate SQL scripts with connector.generate_sql_scripts()")
        print("   3. Implement schemas in your Lakehouse and Data Warehouse")
        
        return connector
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return None

# Usage example for Fabric Notebook:
# connector = test_fabric_claude_connection("your-api-key-here")