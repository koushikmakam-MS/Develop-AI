"""Quick test script for the Kusto MCP server."""
import json
import urllib.request

from brain_ai.kusto.client import KustoMCPClient


def main():
    client = KustoMCPClient()

    # 1. Health check
    print("=== Health Check ===")
    health = client.health_check()
    for k, v in health.items():
        print(f"  {k}: {v}")

    # 2. Tools discovery
    print("\n=== Tools Discovery ===")
    try:
        resp = urllib.request.urlopen("http://127.0.0.1:8701/tools", timeout=5)
        tools = json.loads(resp.read())
        for t in tools:
            print(f"  Tool: {t['name']} - {t['description'][:80]}")
    except Exception as e:
        print(f"  Error listing tools: {e}")

    # 3. Test KQL query
    print("\n=== Test KQL Query ===")
    result = client.execute_kql('print message="Hello from Kusto MCP"')
    print(f"  Success: {result['success']}")
    print(f"  Rows: {result['row_count']}")
    print(f"  Output: {result['formatted']}")
    if result.get("error"):
        print(f"  Error: {result['error']}")


if __name__ == "__main__":
    main()
