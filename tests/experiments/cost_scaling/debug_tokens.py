import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import dspy
from dotenv import load_dotenv
load_dotenv()

# Initialize a simple LM to test history structure
lm = dspy.LM(model="gemini/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
dspy.configure(lm=lm)

print("Testing dspy token tracking...")
print("-" * 50)

# Clear any existing history
if hasattr(lm, 'history'):
    lm.history.clear()
    print(f"Cleared LM history. Initial length: {len(lm.history)}")

# Make a simple test call
class TestSignature(dspy.Signature):
    """Extract one entity from the text."""
    text: str = dspy.InputField()
    entity: str = dspy.OutputField()

predictor = dspy.Predict(TestSignature)
result = predictor(text="The cat sat on the mat.")

print(f"Test result: {result}")

# Examine the history structure
print("\nExamining LM history structure:")
print(f"History type: {type(lm.history)}")
print(f"History length: {len(lm.history)}")

if len(lm.history) > 0:
    print("\nFirst history entry:")
    entry = lm.history[0]
    print(f"Entry type: {type(entry)}")
    print(f"Entry keys: {list(entry.keys()) if isinstance(entry, dict) else 'Not a dict'}")
    
    # Look for usage information
    if isinstance(entry, dict):
        for key, value in entry.items():
            print(f"  {key}: {type(value)} = {value}")
            
            # If value is an object, examine its attributes
            if hasattr(value, '__dict__'):
                print(f"    {key} attributes: {list(value.__dict__.keys())}")
                if key == 'usage':
                    usage = value
                    if hasattr(usage, 'prompt_tokens'):
                        print(f"      prompt_tokens: {usage.prompt_tokens}")
                    if hasattr(usage, 'completion_tokens'):
                        print(f"      completion_tokens: {usage.completion_tokens}")
                    if hasattr(usage, 'total_tokens'):
                        print(f"      total_tokens: {usage.total_tokens}")

# Also check global history if it exists
try:
    if hasattr(dspy, 'GLOBAL_HISTORY'):
        print(f"\nGlobal history length: {len(dspy.GLOBAL_HISTORY)}")
        if len(dspy.GLOBAL_HISTORY) > 0:
            print("First global history entry:")
            global_entry = dspy.GLOBAL_HISTORY[0]
            print(f"Global entry type: {type(global_entry)}")
            print(f"Global entry: {global_entry}")
except Exception as e:
    print(f"No global history or error accessing it: {e}")

print("\n" + "="*50)
print("RECOMMENDATIONS:")
print("="*50)
print("Based on this output, we can fix the token extraction logic.")
print("The cost tracking works, so the issue is just with parsing the usage data.") 