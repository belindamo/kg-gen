import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from tests.utils.resolution import KGAssistedRAG
import json

# Test with the existing 100k corpus KG
kg_path = "kg_100000/kg.json"
output_dir = "kg_100000/test_output"

if os.path.exists(kg_path):
    print(f"Testing resolution with {kg_path}")
    
    # Load KG to check its contents
    with open(kg_path, 'r') as f:
        kg_data = json.load(f)
    
    print(f"KG contains:")
    print(f"  - {len(kg_data.get('entities', []))} entities")
    print(f"  - {len(kg_data.get('edges', []))} edges")  
    print(f"  - {len(kg_data.get('relations', []))} relations")
    
    # Try to run resolution
    try:
        rag = KGAssistedRAG(kg_path, output_dir)
        print("Successfully initialized KGAssistedRAG")
        
        # Try clustering
        print("\nAttempting clustering...")
        rag.cluster()
        print("Clustering completed successfully!")
        
    except Exception as e:
        print(f"Error during resolution: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"KG file not found at {kg_path}")
    print("Please run the cost scaling experiment first to generate the KG") 