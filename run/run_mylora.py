import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from mylora import inject_hybrid_lora

def test_hybrid_lora_with_gpt2():
    print("Testing Hybrid LoRA implementation with GPT-2...")
    
    
    # Test 2: GPT-2 model test
    print("\n=== Test 2: GPT-2 model ===")
    try:
        # Load pre-trained GPT-2 model
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        print(model)
        # Apply hybrid LoRA to the language model
        model = inject_hybrid_lora(
            model, 
            rank=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"],  # Target attention layers
            svd_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"] # Use SVD for c_attn layers
        )
        
        print("GPT-2 model with Hybrid LoRA created successfully!")
        
        # Test forward pass
        inputs = torch.randint(0, 100, (4, 10))  # batch_size=4, seq_len=10
        with torch.no_grad():
            outputs = model(inputs)
            
        print(f"GPT-2 model input shape: {inputs.shape}")
        print(f"GPT-2 model output shape: {outputs.logits.shape}")
        print("GPT-2 model forward pass successful!")
        
    except Exception as e:
        print(f"Error in Test 2: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_hybrid_lora_with_gpt2()