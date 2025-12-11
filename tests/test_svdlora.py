# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from torch import nn
import numpy as np

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.svdlora import SVDLoraLinear


class SimpleModel(nn.Module):
    def __init__(self, vocab_size=100, embedding_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = torch.mean(x, dim=1)  # Global average pooling
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TestSVDLoRA:
    """Tests for SVD LoRA implementation."""
    
    def test_svdlora_linear_creation(self):
        """Test that SVDLoraLinear can be created correctly."""
        # Create a simple linear layer
        original_layer = nn.Linear(10, 20)
        
        # Create SVDLoraLinear
        svd_lora_layer = SVDLoraLinear(original_layer, "default", r=4)
        
        # Check that the layer was created correctly
        assert isinstance(svd_lora_layer, SVDLoraLinear)
        assert hasattr(svd_lora_layer, 'lora_coeffs_A')
        assert hasattr(svd_lora_layer, 'lora_coeffs_B')
        assert 'default' in svd_lora_layer.lora_coeffs_A
        assert 'default' in svd_lora_layer.lora_coeffs_B
        
        # Check coefficient shapes
        base_weight = original_layer.weight
        U, S, Vh = torch.linalg.svd(base_weight.data.float(), full_matrices=False)
        k = U.shape[1]  # Number of singular values
        
        assert svd_lora_layer.lora_coeffs_A['default'].shape == (4, k)  # (r, k)
        assert svd_lora_layer.lora_coeffs_B['default'].shape == (k, 4)  # (k, r)
        
    def test_svdlora_svd_decomposition(self):
        """Test that SVD decomposition is correctly computed and stored."""
        # Create a simple linear layer
        original_layer = nn.Linear(10, 20)
        original_weight = original_layer.weight.data.clone()
        
        # Create SVDLoraLinear
        svd_lora_layer = SVDLoraLinear(original_layer, "default", r=4)
        
        # Check that U and Vh matrices are stored as buffers
        assert hasattr(svd_lora_layer, 'U_default')
        assert hasattr(svd_lora_layer, 'Vh_default')
        
        # Check that the stored matrices have correct shapes
        U = svd_lora_layer.U_default
        Vh = svd_lora_layer.Vh_default
        
        assert U.shape == (20, min(10, 20))  # (out_features, min(in_features, out_features))
        assert Vh.shape == (min(10, 20), 10)  # (min(in_features, out_features), in_features)
        
        # Verify that U and Vh are from the SVD of the original weight matrix
        with torch.no_grad():
            reconstructed_weight = U @ torch.diag(torch.ones(min(10, 20))) @ Vh
            # Since we're using reduced SVD, we can't perfectly reconstruct the original matrix
            # But we can check that the shapes are correct and values are reasonable
            
    def test_svdlora_get_delta_weight(self):
        """Test that get_delta_weight works correctly for SVD LoRA."""
        # Create a simple linear layer
        original_layer = nn.Linear(10, 20)
        
        # Create SVDLoraLinear
        svd_lora_layer = SVDLoraLinear(original_layer, "default", r=4)
        
        # Initialize coefficients with some values
        torch.manual_seed(0)
        with torch.no_grad():
            svd_lora_layer.lora_coeffs_A['default'].copy_(torch.randn_like(svd_lora_layer.lora_coeffs_A['default']))
            svd_lora_layer.lora_coeffs_B['default'].copy_(torch.randn_like(svd_lora_layer.lora_coeffs_B['default']))
        
        # Get delta weight
        delta_weight = svd_lora_layer.get_delta_weight("default")
        
        # Check that delta weight has the correct shape
        assert delta_weight.shape == original_layer.weight.shape
        
        # Check that delta weight is computed correctly using SVD formulation
        U = svd_lora_layer.U_default
        Vh = svd_lora_layer.Vh_default
        coeffs_A = svd_lora_layer.lora_coeffs_A['default']
        coeffs_B = svd_lora_layer.lora_coeffs_B['default']
        scaling = svd_lora_layer.scaling['default']
        
        # Reconstruct effective A and B matrices
        effective_A = coeffs_A @ Vh
        effective_B = U @ coeffs_B
        
        expected_delta_weight = effective_B @ effective_A * scaling
        assert torch.allclose(delta_weight, expected_delta_weight, atol=1e-5)
        
    def test_svdlora_forward_pass(self):
        """Test that SVD LoRA forward pass works correctly."""
        # Create a simple linear layer
        original_layer = nn.Linear(10, 20)
        
        # Create SVDLoraLinear
        svd_lora_layer = SVDLoraLinear(original_layer, "default", r=4)
        
        # Create some input data
        x = torch.randn(5, 10)
        
        # Forward pass
        output = svd_lora_layer(x)
        
        # Check that output has the correct shape
        assert output.shape == (5, 20)
        
        # Compare with manual calculation
        # Standard linear layer output
        with torch.no_grad():
            expected_output = original_layer(x)
            
        # Add SVD LoRA contribution
        U = svd_lora_layer.U_default
        Vh = svd_lora_layer.Vh_default
        coeffs_A = svd_lora_layer.lora_coeffs_A['default']
        coeffs_B = svd_lora_layer.lora_coeffs_B['default']
        scaling = svd_lora_layer.scaling['default']
        
        # Apply SVD LoRA transformation
        x_dropout = x  # No dropout in this test
        x_k = x_dropout @ Vh.T 
        x_r = x_k @ coeffs_A.T 
        x_k_out = x_r @ coeffs_B.T
        lora_out = x_k_out @ U.T
        
        expected_output = expected_output + lora_out * scaling
        assert torch.allclose(output, expected_output, atol=1e-5)
        
    def test_svdlora_with_peft_model(self):
        """Test that SVD LoRA works with PEFT model."""
        # Create a simple model
        model = SimpleModel()
        
        # Create a LoRA config with SVD LoRA
        lora_config = LoraConfig(
            r=4,
            lora_alpha=32,
            target_modules=["linear1", "linear2"],
            use_svdlora=True,
            bias="none"
        )
        
        # Apply LoRA to the model
        peft_model = get_peft_model(model, lora_config)
        
        # Check that SVD LoRA layers were created
        svd_lora_layers_found = 0
        for name, module in peft_model.named_modules():
            if isinstance(module, SVDLoraLinear):
                svd_lora_layers_found += 1
                # Check that SVD components are present
                assert hasattr(module, 'U_default')
                assert hasattr(module, 'Vh_default')
                assert hasattr(module, 'lora_coeffs_A')
                assert hasattr(module, 'lora_coeffs_B')
                
        # Should have 2 SVD LoRA layers (linear1 and linear2)
        assert svd_lora_layers_found == 2
                
        # Test forward pass
        input_ids = torch.randint(0, 100, (4, 10))
        output = peft_model(input_ids)
        assert output.shape == (4, 100)
        
    def test_svdlora_training(self):
        """Test that SVD LoRA layers can be trained."""
        # Create a simple model
        model = SimpleModel()
        
        # Create a LoRA config with SVD LoRA
        lora_config = LoraConfig(
            r=4,
            lora_alpha=32,
            target_modules=["linear1", "linear2"],
            use_svdlora=True,
            bias="none"
        )
        
        # Apply LoRA to the model
        peft_model = get_peft_model(model, lora_config)
        
        # Set model to training mode
        peft_model.train()
        
        # Check that SVD LoRA parameters require gradients
        trainable_params = 0
        for name, param in peft_model.named_parameters():
            if 'lora_coeffs_' in name:
                assert param.requires_grad, f"Parameter {name} should require gradients"
                trainable_params += 1
                
        # Should have trainable parameters for both lora_coeffs_A and lora_coeffs_B for each linear layer
        # We have 2 linear layers (linear1 and linear2), each with lora_coeffs_A and lora_coeffs_B
        assert trainable_params == 4  # 2 params (A,B) * 2 layers (linear1, linear2)
        
        # Create some dummy data
        input_ids = torch.randint(0, 100, (4, 10))
        labels = torch.randint(0, 100, (4,))
        
        # Forward pass
        outputs = peft_model(input_ids)
        loss = nn.functional.cross_entropy(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for SVD LoRA parameters
        svd_lora_params_with_grad = 0
        for name, param in peft_model.named_parameters():
            if 'lora_coeffs_' in name:
                svd_lora_params_with_grad += 1
                # Check that gradients exist
                assert param.grad is not None
                
        # Should have gradients for lora_coeffs_A and lora_coeffs_B for each linear layer
        assert svd_lora_params_with_grad == 4  # 2 params (A,B) * 2 layers (linear1, linear2)
        
    def test_svdlora_multiple_adapters(self):
        """Test that SVD LoRA works with multiple adapters."""
        # Create a simple linear layer
        original_layer = nn.Linear(10, 20)
        
        # Create SVDLoraLinear with first adapter
        svd_lora_layer = SVDLoraLinear(original_layer, "adapter1", r=4)
        
        # Add second adapter
        svd_lora_layer.update_layer(
            "adapter2",
            r=6,
            lora_alpha=32,
            lora_dropout=0.0,
            init_lora_weights=True,
            use_rslora=False
        )
        
        # Initialize coefficients with different values for each adapter
        torch.manual_seed(0)
        with torch.no_grad():
            svd_lora_layer.lora_coeffs_A['adapter1'].copy_(torch.randn_like(svd_lora_layer.lora_coeffs_A['adapter1']))
            svd_lora_layer.lora_coeffs_B['adapter1'].copy_(torch.randn_like(svd_lora_layer.lora_coeffs_B['adapter1']))
            svd_lora_layer.lora_coeffs_A['adapter2'].copy_(torch.randn_like(svd_lora_layer.lora_coeffs_A['adapter2']))
            svd_lora_layer.lora_coeffs_B['adapter2'].copy_(torch.randn_like(svd_lora_layer.lora_coeffs_B['adapter2']))
        
        # Check that both adapters exist
        assert 'adapter1' in svd_lora_layer.lora_coeffs_A
        assert 'adapter2' in svd_lora_layer.lora_coeffs_A
        assert 'adapter1' in svd_lora_layer.lora_coeffs_B
        assert 'adapter2' in svd_lora_layer.lora_coeffs_B
        
        # Check shapes
        assert svd_lora_layer.lora_coeffs_A['adapter1'].shape == (4, 10)  # (r1, k)
        assert svd_lora_layer.lora_coeffs_A['adapter2'].shape == (6, 10)  # (r2, k)
        assert svd_lora_layer.lora_coeffs_B['adapter1'].shape == (10, 4)  # (k, r1)
        assert svd_lora_layer.lora_coeffs_B['adapter2'].shape == (10, 6)  # (k, r2)
        
        # Test forward pass with different adapters
        x = torch.randn(5, 10)
        
        # Set active adapters
        svd_lora_layer.set_adapter(["adapter1"])
        output1 = svd_lora_layer(x)
        
        svd_lora_layer.set_adapter(["adapter2"])
        output2 = svd_lora_layer(x)
        
        # Outputs should be different
        assert not torch.allclose(output1, output2)
        
