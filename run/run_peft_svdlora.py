import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

def test_svdlora():
    print("Testing SVD LoRA implementation...")

    # Test 2: Language model test
    print("\n=== Test Language model Forward===")
    try:
        # Create a small test model instead of downloading from HF
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        
        # Create a LoRA config for the language model
        lora_config = LoraConfig(
            r=4,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_svdlora=True,
            target_modules='all-linear',
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to the language model
        peft_model = get_peft_model(base_model, lora_config)
        
        print("Small model with SVD LoRA created successfully!")
        
        # Test forward pass
        inputs = torch.randint(0, 100, (4, 10))  # batch_size=4, seq_len=10
        with torch.no_grad():
            outputs = peft_model(inputs)
            
        print(f"Small model input shape: {inputs.shape}")
        print(f"Small model output shape: {outputs.logits.shape}")

        print("========= Test trainable =========")
        # breakpoint()
        # test trainable

        # test trainable
        peft_model.train()        
        trainable_paras = list(n for n, p in peft_model.named_parameters() if p.requires_grad)
        trainable = len(trainable_paras)
        lora = 0
        for name, param in peft_model.named_parameters():
            if "coeffs_" in name:
                lora += 1
                assert param.requires_grad, f"Parameter {name} should require gradients"
        print("svd", trainable, lora)
        if trainable != lora:
            print("Not all SVD LoRA parameters are trainable")
            breakpoint()


        base_model = peft_model.unload()
        lora_config.use_svdlora=False
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.train()        
        trainable = sum(p.requires_grad for p in peft_model.parameters())
        lora = 0
        # breakpoint()
        for name, param in peft_model.named_parameters():
            if "lora" in name:
                lora += 1
                assert param.requires_grad, f"Parameter {name} should require gradients"
        print("normal", trainable, lora)
        assert trainable == lora, "Not all LoRA parameters are trainable"
    



    except Exception as e:
        print(f"Error in Test 2: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_svdlora()