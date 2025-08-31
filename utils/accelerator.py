# utils/accelerator.py (CORRECTED VERSION WITH PROPER CUPY HANDLING)
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
        import numpy as np_cpu  # Keep regular numpy available for conversions
        
        # Add a function to convert pandas DataFrames to cudf
        def to_gpu(df_cpu):
            return pd.from_pandas(df_cpu)
            
        # Add a function to convert CuPy arrays to NumPy arrays
        def to_numpy(cupy_array):
            """Convert CuPy array to NumPy array."""
            if hasattr(cupy_array, 'get'):
                return cupy_array.get()
            else:
                return np.asnumpy(cupy_array)
                
    except ImportError:
        print("⚠️ Failed to import GPU libraries. Falling back to CPU.")
        IS_GPU_AVAILABLE = False # Correct the status
        DEVICE = "cpu"
        import pandas as pd
        import numpy as np
        import numpy as np_cpu
        def to_gpu(df_cpu):
            return df_cpu
        def to_numpy(array):
            return np.asarray(array)
else:
    print("⚠️ No GPU detected. Using standard Pandas and NumPy on CPU.")
    import pandas as pd # Use standard Pandas
    import numpy as np  # Use standard NumPy
    import numpy as np_cpu  # Keep consistent naming
    # In CPU mode, these functions do nothing special
    def to_gpu(df_cpu):
        return df_cpu
    def to_numpy(array):
        return np.asarray(array)
