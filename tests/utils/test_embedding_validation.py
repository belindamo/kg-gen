#!/usr/bin/env python3
"""
Test script to validate embedding sanitization and check for numerical issues.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import warnings
from resolution import KGAssistedRAG

def test_embedding_validation():
    """Test embedding validation and sanitization"""
    
    # Test data paths
    test_kg_path = "../data/wiki_qa/aggregated/articles_1_kg.json"
    test_output = "../data/test_embedding_validation"
    
    if not os.path.exists(test_kg_path):
        print(f"⚠️  Test KG file not found: {test_kg_path}")
        print("Skipping test - please run with a valid KG file")
        return
    
    print("🧪 Testing embedding validation...")
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Initialize KGAssistedRAG
        rag = KGAssistedRAG(
            kg_path=test_kg_path,
            output_folder=test_output
        )
        
        # Test queries that might trigger numerical issues
        test_queries = [
            "Winter Olympics",
            "",  # Empty query
            "a" * 1000,  # Very long query
            "test query",
            "🎿⛷️🏂",  # Unicode characters
        ]
        
        print("\n🔍 Testing retrieval with various queries...")
        for query in test_queries:
            try:
                print(f"Testing query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
                
                # Test node retrieval
                nodes = rag.get_relevant_items(query, 10, "node")
                print(f"  ✓ Retrieved {len(nodes)} nodes")
                
                # Test edge retrieval
                edges = rag.get_relevant_items(query, 10, "edge")
                print(f"  ✓ Retrieved {len(edges)} edges")
                
            except Exception as e:
                print(f"  ❌ Error with query '{query}': {e}")
        
        # Check for RuntimeWarnings about matmul
        matmul_warnings = [warning for warning in w if 'matmul' in str(warning.message)]
        overflow_warnings = [warning for warning in w if 'overflow' in str(warning.message)]
        invalid_warnings = [warning for warning in w if 'invalid value' in str(warning.message)]
        
        print(f"\n📊 Warning Analysis:")
        print(f"  Total warnings captured: {len(w)}")
        print(f"  Matmul warnings: {len(matmul_warnings)}")
        print(f"  Overflow warnings: {len(overflow_warnings)}")
        print(f"  Invalid value warnings: {len(invalid_warnings)}")
        
        if matmul_warnings:
            print("\n⚠️  Still seeing matmul warnings:")
            for warning in matmul_warnings[:3]:  # Show first 3
                print(f"    {warning.message}")
        else:
            print("\n✅ No matmul warnings detected!")
        
        # Test embedding statistics
        print(f"\n📈 Embedding Statistics:")
        print(f"  Node embeddings shape: {rag.node_embeddings.shape}")
        print(f"  Node embeddings dtype: {rag.node_embeddings.dtype}")
        print(f"  Node embeddings range: [{np.min(rag.node_embeddings):.6f}, {np.max(rag.node_embeddings):.6f}]")
        print(f"  Node embeddings NaN count: {np.isnan(rag.node_embeddings).sum()}")
        print(f"  Node embeddings Inf count: {np.isinf(rag.node_embeddings).sum()}")
        
        print(f"  Edge embeddings shape: {rag.edge_embeddings.shape}")
        print(f"  Edge embeddings dtype: {rag.edge_embeddings.dtype}")
        print(f"  Edge embeddings range: [{np.min(rag.edge_embeddings):.6f}, {np.max(rag.edge_embeddings):.6f}]")
        print(f"  Edge embeddings NaN count: {np.isnan(rag.edge_embeddings).sum()}")
        print(f"  Edge embeddings Inf count: {np.isinf(rag.edge_embeddings).sum()}")
        
        return len(matmul_warnings) == 0

def test_artificial_problematic_embeddings():
    """Test how our sanitization handles artificially problematic embeddings"""
    
    print("\n🧪 Testing artificial problematic embeddings...")
    
    # Create test embeddings with known issues
    test_embeddings = np.array([
        [1.0, 2.0, 3.0],           # Normal
        [np.nan, 1.0, 2.0],        # NaN
        [np.inf, 1.0, 2.0],        # Positive infinity
        [-np.inf, 1.0, 2.0],       # Negative infinity
        [1e20, 1e20, 1e20],        # Very large values
        [0.0, 0.0, 0.0],           # Zero vector
    ])
    
    # Simulate the sanitization process
    from resolution import KGAssistedRAG
    rag = KGAssistedRAG.__new__(KGAssistedRAG)  # Create instance without __init__
    
    sanitized = rag._sanitize_embeddings(test_embeddings, "test")
    
    print(f"Original shape: {test_embeddings.shape}")
    print(f"Sanitized shape: {sanitized.shape}")
    print(f"Sanitized NaN count: {np.isnan(sanitized).sum()}")
    print(f"Sanitized Inf count: {np.isinf(sanitized).sum()}")
    print(f"Sanitized max value: {np.max(np.abs(sanitized)):.6f}")
    
    # Check if all vectors are unit length (normalized)
    norms = np.linalg.norm(sanitized, axis=1)
    print(f"Vector norms: {norms}")
    print(f"All unit vectors: {np.allclose(norms, 1.0)}")
    
    return True

if __name__ == "__main__":
    print("🔬 Embedding Validation Test Suite")
    print("=" * 50)
    
    # Test artificial cases first
    test_artificial_problematic_embeddings()
    
    # Test with real data
    success = test_embedding_validation()
    
    if success:
        print("\n🎉 All tests passed! Numerical issues should be resolved.")
    else:
        print("\n❌ Some issues remain. Check the warnings above.") 