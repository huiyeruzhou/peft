import safetensors
import json

def read_safetensors_metadata(file_path):
    """Read metadata from a safetensors file"""
    try:
        with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
            # Get metadata
            metadata = f.metadata()
            print("Metadata:")
            if metadata:
                print(json.dumps(metadata, indent=2))
            else:
                print("No metadata found")
            
            # Get tensor names
            tensor_names = f.keys()
            print("\nTensor names:")
            for name in tensor_names:
                print(f"  {name}")
                
            # Try to get tensor shapes and dtypes
            print("\nTensor info:")
            for name in tensor_names:
                tensor = f.get_tensor(name)
                print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
                
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    file_path = "/Users/bytedance/Documents/peft/run/svdlora_addition_model/adapter_model.safetensors"
    read_safetensors_metadata(file_path)