import os
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort

def export_model_to_onnx(model: nn.Module, output_path: str, input_dim: int = 1536) -> None:
    """
    Converts PyTorch weights to an ONNX graph with dynamic batch dimensions.
    """
    model.eval()
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"  [Export] Saved ONNX model: {os.path.abspath(output_path)}")

def verify_onnx(output_path: str, input_dim: int = 1536) -> None:
    """
    Loads exported ONNX file via onnxruntime and runs a sanity check inference pass.
    """
    session = ort.InferenceSession(output_path)
    sample_tensor = np.random.randn(1, input_dim).astype(np.float32)
    result = session.run(None, {"input": sample_tensor})
    test_score = float(result[0][0][0])
    print(f"  [Verify] ONNX inference operational. Baseline test score: {test_score:.4f}")