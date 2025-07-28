import numpy as np
import faiss
import os

# Test the actual embeddings that were generated
embeddings_path = "kg_100000/test_output/node_embeddings.npy"

if os.path.exists(embeddings_path):
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    print(f"Embeddings min: {embeddings.min()}, max: {embeddings.max()}")
    print(f"Contains NaN: {np.any(np.isnan(embeddings))}")
    print(f"Contains Inf: {np.any(np.isinf(embeddings))}")
    print(f"Is C-contiguous: {embeddings.flags['C_CONTIGUOUS']}")
    print(f"Is F-contiguous: {embeddings.flags['F_CONTIGUOUS']}")
    
    # Try different approaches
    print("\n1. Testing direct FAISS clustering...")
    try:
        embeddings_float32 = embeddings.astype(np.float32)
        embeddings_contiguous = np.ascontiguousarray(embeddings_float32)
        
        d = embeddings_contiguous.shape[1]
        num_clusters = 5
        
        kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=True, gpu=False)
        kmeans.train(embeddings_contiguous)
        print("✓ Direct FAISS clustering successful")
    except Exception as e:
        print(f"✗ Direct FAISS clustering failed: {e}")
    
    print("\n2. Testing with copy...")
    try:
        # Make a fresh copy
        embeddings_copy = embeddings.copy()
        embeddings_float32 = embeddings_copy.astype(np.float32)
        embeddings_contiguous = np.ascontiguousarray(embeddings_float32)
        
        kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=True, gpu=False)
        kmeans.train(embeddings_contiguous)
        print("✓ FAISS clustering with copy successful")
    except Exception as e:
        print(f"✗ FAISS clustering with copy failed: {e}")
        
    print("\n3. Testing with smaller subset...")
    try:
        # Try with first 100 embeddings
        subset = embeddings[:100].copy()
        subset_float32 = subset.astype(np.float32)
        subset_contiguous = np.ascontiguousarray(subset_float32)
        
        kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=True, gpu=False)
        kmeans.train(subset_contiguous)
        print("✓ FAISS clustering with subset successful")
    except Exception as e:
        print(f"✗ FAISS clustering with subset failed: {e}")
        
else:
    print(f"Embeddings file not found at {embeddings_path}")
    print("Run the minimal test first to generate embeddings") 