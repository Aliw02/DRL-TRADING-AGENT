# utils/accelerator.py
import torch

# --- Check for GPU availability ---
IS_GPU_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if IS_GPU_AVAILABLE else "cpu"

# --- Conditionally Import Libraries ---
if IS_GPU_AVAILABLE:
    try:
        print("✅ GPU detected. Using cuDF and CuPy for acceleration.")
        import cudf as pd  # Use cuDF and alias it as pd
        import cupy as np  # Use CuPy and alias it as np
        
        # Add a function to convert pandas DataFrames to cudf
        def to_gpu(df_cpu):
            return pd.from_pandas(df_cpu)
    except ImportError:
        print("⚠️ Failed to import GPU libraries. Falling back to CPU.")
        IS_GPU_AVAILABLE = False # Correct the status
        DEVICE = "cpu"
        import pandas as pd
        import numpy as np
        def to_gpu(df_cpu):
            return df_cpu
else:
    print("⚠️ No GPU detected. Using standard Pandas and NumPy on CPU.")
    import pandas as pd # Use standard Pandas
    import numpy as np  # Use standard NumPy
    # In CPU mode, this function does nothing
    def to_gpu(df_cpu):
        return df_cpu