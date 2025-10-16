#!/usr/bin/env python3
"""
Test script to connect to Claude (Anthropic) API
This script tests the connection and basic functionality with Claude's API.
"""

import os
import sys
from typing import Optional, Dict, Any
import json
from datetime import datetime

# Try to import required libraries
try:
    import anthropic
except ImportError:
    print("❌ Anthropic library not found. Installing...")
    os.system(f"{sys.executable} -m pip install anthropic")
    import anthropic

try:
    import requests
except ImportError:
    print("❌ Requests library not found. Installing...")
    os.system(f"{sys.executable} -m pip install requests")
    import requests

class ClaudeConnector:
    """Simple connector class for Claude API testing"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude connector with API key"""
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "❌ API key not found. Set ANTHROPIC_API_KEY environment variable or pass it directly."
            )
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        print("✅ Claude client initialized successfully")
    
    def test_connection(self) -> bool:
        """Test basic connection to Claude API"""
        try:
            print("\n🔄 Testing Claude API connection...")
            
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Using Haiku for faster/cheaper testing
                max_tokens=100,
                messages=[
                    {"role": "user", "content": "Hello! This is a connection test. Please respond with 'Connection successful' and today's date."}
                ]
            )
            
            print("✅ Connection successful!")
            print(f"📝 Claude's response: {response.content[0].text}")
            return True
            
        except anthropic.AuthenticationError:
            print("❌ Authentication failed. Please check your API key.")
            return False
        except anthropic.RateLimitError:
            print("⚠️ Rate limit exceeded. Please wait and try again.")
            return False
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False
    
    def test_data_analysis_capability(self) -> bool:
        """Test Claude's data analysis capabilities"""
        try:
            print("\n🔄 Testing Claude's data analysis capabilities...")
            
            # Sample data that might be similar to your Fabric metadata
            test_data = {
                "workspaces": [
                    {"name": "Analytics Workspace", "dataset_count": 5, "report_count": 12},
                    {"name": "Sales Dashboard", "dataset_count": 3, "report_count": 8},
                    {"name": "Finance Reports", "dataset_count": 2, "report_count": 4}
                ]
            }
            
            prompt = f"""
            Analyze this sample workspace data and suggest a simple data model structure:
            {json.dumps(test_data, indent=2)}
            
            Please provide:
            1. Suggested table schema for a data warehouse
            2. Key relationships
            3. Recommended indexes
            
            Keep the response concise (under 200 words).
            """
            
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            print("✅ Data analysis test successful!")
            print("📊 Claude's analysis:")
            print(response.content[0].text)
            return True
            
        except Exception as e:
            print(f"❌ Data analysis test failed: {str(e)}")
            return False
    
    def generate_schema_suggestion(self, metadata: Dict[str, Any]) -> str:
        """Generate schema suggestions based on metadata"""
        try:
            prompt = f"""
            Based on this Microsoft Fabric workspace metadata, suggest database schemas, stored procedures, and pipeline structures for a Lakehouse and Data Warehouse:

            Metadata: {json.dumps(metadata, indent=2)}

            Please provide:
            1. Lakehouse table schemas (Delta format)
            2. Data Warehouse dimension/fact tables
            3. Essential stored procedures
            4. Data pipeline recommendations

            Format as structured JSON where possible.
            """
            
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",  # Using Sonnet for more complex analysis
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error generating schema suggestions: {str(e)}"

def main():
    """Main test function"""
    print("🚀 Claude API Connection Test")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("⚠️ ANTHROPIC_API_KEY environment variable not set.")
        print("💡 You can set it by running:")
        print("   $env:ANTHROPIC_API_KEY='your-api-key-here'  # PowerShell")
        print("   set ANTHROPIC_API_KEY=your-api-key-here     # CMD")
        
        # Option to enter API key manually for testing
        manual_key = input("\n🔑 Enter your Anthropic API key (or press Enter to skip): ").strip()
        if manual_key:
            api_key = manual_key
        else:
            print("❌ Cannot proceed without API key. Exiting.")
            return
    
    try:
        # Initialize connector
        connector = ClaudeConnector(api_key)
        
        # Run tests
        results = {
            "connection_test": False,
            "data_analysis_test": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Test 1: Basic connection
        results["connection_test"] = connector.test_connection()
        
        # Test 2: Data analysis capability
        if results["connection_test"]:
            results["data_analysis_test"] = connector.test_data_analysis_capability()
        
        # Save test results
        with open("claude_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print(f"✅ Connection Test: {'PASSED' if results['connection_test'] else 'FAILED'}")
        print(f"✅ Data Analysis Test: {'PASSED' if results['data_analysis_test'] else 'FAILED'}")
        
        if all(results.values()) if isinstance(list(results.values())[0], bool) else results["connection_test"] and results["data_analysis_test"]:
            print("\n🎉 All tests passed! Claude is ready for your Fabric metadata analysis.")
            print("\n📝 Next steps:")
            print("   1. Load your workspace metadata")
            print("   2. Use connector.generate_schema_suggestion(metadata) for AI-powered analysis")
            print("   3. Implement the suggested schemas in your Lakehouse/DW")
        else:
            print("\n⚠️ Some tests failed. Please check the errors above.")
        
        return connector if results["connection_test"] else None
        
    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        return None

if __name__ == "__main__":
    connector = main()
    
    # If tests passed, offer to test with actual metadata
    if connector:
        print("\n" + "=" * 50)
        test_with_metadata = input("🔬 Test with sample Fabric metadata? (y/n): ").strip().lower()
        
        if test_with_metadata == 'y':
            # Sample metadata structure based on your Fabric scripts
            sample_metadata = {
                "workspaces": [
                    {
                        "id": "sample-ws-1",
                        "name": "Analytics Hub",
                        "type": "Workspace",
                        "dataset_count": 8,
                        "total_reports": 15,
                        "pbi_reports": 12,
                        "paginated_reports": 3,
                        "dataflows": 2
                    }
                ],
                "datasets": [
                    {
                        "id": "ds-1",
                        "name": "Sales Analytics",
                        "workspace_id": "sample-ws-1",
                        "configured_by": "user@company.com"
                    }
                ],
                "discovery_timestamp": datetime.now().isoformat()
            }
            
            print("\n🔄 Generating schema suggestions...")
            suggestions = connector.generate_schema_suggestion(sample_metadata)
            print("\n📋 AI-Generated Schema Suggestions:")
            print(suggestions)
            
            # Save suggestions to file
            with open("claude_schema_suggestions.txt", "w") as f:
                f.write(f"Claude Schema Suggestions - {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n\n")
                f.write(suggestions)
            
            print(f"\n💾 Suggestions saved to 'claude_schema_suggestions.txt'")