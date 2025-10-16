# Claude API Integration Setup

This guide helps you set up Claude (Anthropic) API integration for AI-powered metadata analysis in your Fabric Automations project.

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements_claude.txt
```

### 2. Get Your API Key
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign up/Login to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (it starts with `sk-ant-...`)

### 3. Set Environment Variable
```powershell
# PowerShell
$env:ANTHROPIC_API_KEY='your-api-key-here'

# Or create a .env file
cp .env.template .env
# Edit .env file with your actual API key
```

### 4. Test Connection
```powershell
python test_claude_connection.py
```

## 🧪 What the Test Script Does

The `test_claude_connection.py` script performs:

1. **Connection Test**: Verifies your API key works
2. **Data Analysis Test**: Tests Claude's ability to analyze sample data
3. **Schema Generation**: Creates AI-powered schema suggestions for your metadata

## 📊 Integration with Your Fabric Metadata

Once connected, you can use Claude to:

### Generate Data Model Schemas
```python
from test_claude_connection import ClaudeConnector

# Initialize connector
connector = ClaudeConnector()

# Load your actual workspace metadata (from your existing scripts)
metadata = {
    "workspaces": workspaces_df.to_dict('records'),
    "datasets": datasets_df.to_dict('records'),
    "reports": reports_df.to_dict('records'),
    "discovery_timestamp": datetime.now().isoformat()
}

# Get AI-powered schema suggestions
suggestions = connector.generate_schema_suggestion(metadata)
print(suggestions)
```

### Expected Output
Claude will provide suggestions for:
- **Lakehouse Tables**: Delta format schemas
- **Data Warehouse**: Dimension and fact table designs  
- **Stored Procedures**: Essential sprocs for data processing
- **Pipelines**: Data flow recommendations

## 🏗️ Next Steps After Testing

1. **Successful Connection**: Run the test and verify all tests pass
2. **Load Real Metadata**: Use your existing Fabric discovery scripts to collect metadata
3. **Generate Schemas**: Use Claude to analyze your metadata and suggest optimal schemas
4. **Implement in Fabric**: Create the suggested tables, procedures, and pipelines in your Lakehouse/DW

## 🛠️ Available Claude Models

- **Haiku** (`claude-3-haiku-20240307`): Fast, cost-effective for simple tasks
- **Sonnet** (`claude-3-sonnet-20240229`): Balanced performance and capability
- **Opus** (`claude-3-opus-20240229`): Most powerful for complex analysis

## 📝 File Structure

```
Fabric Automations/
├── test_claude_connection.py     # Main test script
├── requirements_claude.txt       # Python dependencies
├── .env.template                # Configuration template
├── CLAUDE_SETUP.md              # This guide
│
# Generated files after running tests:
├── claude_test_results.json     # Test results
├── claude_schema_suggestions.txt # AI-generated suggestions
```

## 🔒 Security Notes

- Never commit your actual API key to version control
- Use environment variables or `.env` files for sensitive data
- The `.env` file should be added to your `.gitignore`

## 🆘 Troubleshooting

**Authentication Error**: 
- Double-check your API key
- Ensure the key is properly set in environment variable

**Rate Limit Error**:
- Wait a few moments and try again
- Consider using Haiku model for testing (cheaper/faster)

**Import Error**:
- Run `pip install anthropic requests` manually
- Check your Python environment

## 💡 Integration Ideas

Once Claude is working, you can enhance your Fabric automation by:

1. **Auto-generating optimized schemas** based on your actual workspace structure
2. **Creating intelligent data pipelines** that understand your data relationships  
3. **Generating documentation** for your data models
4. **Suggesting performance optimizations** for your tables and queries

Ready to test? Run `python test_claude_connection.py` and let Claude analyze your Fabric metadata!