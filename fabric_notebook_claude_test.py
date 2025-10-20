# FABRIC NOTEBOOK - CLAUDE API TEST
# Copy and paste this entire cell into your Fabric notebook

# ==============================================
# CELL 1: Claude Connector Setup
# ==============================================

import json
import requests
from datetime import datetime
import os

class FabricClaudeConnector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        print("✅ Claude connector ready")
    
    def _make_request(self, messages, model="claude-3-haiku-20240307", max_tokens=1000):
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages
        }
        
        response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            raise Exception("❌ Authentication failed. Check your API key.")
        else:
            raise Exception(f"❌ API request failed: {response.status_code}")
    
    def test_connection(self):
        print("🔄 Testing Claude API connection...")
        messages = [{"role": "user", "content": "Hello! Please respond with 'Connection successful'."}]
        
        result = self._make_request(messages, max_tokens=100)
        response_text = result['content'][0]['text']
        print("✅ Connection successful!")
        print(f"📝 Claude says: {response_text}")
        return True
    
    def analyze_metadata(self, metadata):
        print("🔄 Analyzing metadata with Claude...")
        
        prompt = f"""
        As a Microsoft Fabric expert, analyze this workspace metadata and suggest:
        
        1. Lakehouse table schemas (Delta format)
        2. Data Warehouse dimensional model
        3. Essential stored procedures
        4. Data pipeline recommendations
        
        Metadata: {json.dumps(metadata, indent=2)}
        
        Keep suggestions practical and implementable in Fabric.
        """
        
        messages = [{"role": "user", "content": prompt}]
        result = self._make_request(messages, model="claude-3-sonnet-20240229", max_tokens=2500)
        
        analysis = result['content'][0]['text']
        print("✅ Analysis complete!")
        return analysis

# ==============================================
# CELL 2: Test with Your API Key
# ==============================================

# REPLACE 'your-api-key-here' with your actual Anthropic API key
API_KEY = "your-api-key-here"  # ⚠️ CHANGE THIS!

try:
    # Initialize connector
    claude = FabricClaudeConnector(API_KEY)
    
    # Test connection
    claude.test_connection()
    
    print("\n" + "="*50)
    print("🎉 SUCCESS! Claude is connected and ready.")
    print("You can now use claude.analyze_metadata() with your Fabric data!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Check that:")
    print("   1. Your API key is correct")
    print("   2. You have internet access")
    print("   3. Your Anthropic account has credits")

# ==============================================
# CELL 3: Test with Sample Data (Optional)
# ==============================================

# Uncomment and run this cell to test with sample Fabric metadata
"""
sample_metadata = {
    "workspaces": [
        {
            "name": "Sales Analytics",
            "dataset_count": 5,
            "report_count": 12,
            "dataflows": 2
        }
    ],
    "total_objects": 19
}

print("\n🔬 Testing with sample metadata...")
analysis = claude.analyze_metadata(sample_metadata)
print("\n📋 Claude's Analysis:")
print(analysis)
"""

print("\n📝 Next steps:")
print("   1. Replace 'your-api-key-here' with your actual API key")
print("   2. Run this cell to test the connection")
print("   3. Use claude.analyze_metadata(your_fabric_data) with real data")
print("\nReady to analyze your Fabric workspace metadata! 🚀")