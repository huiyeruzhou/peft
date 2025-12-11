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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import LoraLayer

class SVDLoraLinear(nn.Module, LoraLayer):
    """SVD LoRA implemented in a dense layer using SVD decomposition."""
    
    # Override adapter_layer_names to include SVD-specific parameters
    adapter_layer_names: tuple[str, ...] = ("lora_coeffs_A", "lora_coeffs_B", "lora_U", "lora_Vh")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        # Initialize the base layer first
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        del self.lora_A
        del self.lora_B
        del self.lora_embedding_A
        del self.lora_embedding_B
        self.fan_in_fan_out = fan_in_fan_out
        # Initialize ParameterDict for lo r a and lora_Vh matrices
        self.lora_U = nn.ParameterDict({})
        self.lora_Vh = nn.ParameterDict({})

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            **kwargs,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        use_alora: bool = False,
        use_qalora: bool = False,
        lora_bias: bool = False,
        arrow_config = None,
        qalora_group_size: int = 32,
        inference_mode: bool = False,
        use_bdlora=None,
        **kwargs,
    ):
        # Store SVD dimensions first to determine correct shapes
        base_weight = self.get_base_layer().weight
        with torch.no_grad():
            W = base_weight.data.float()
            lora_U, S, lora_Vh = torch.linalg.svd(W, full_matrices=False)
            
            dtype = base_weight.dtype
            # Store lora_U and lora_Vh matrices as parameters without gradients in ParameterDict
            self.lora_U[adapter_name] = nn.Parameter(lora_U.to(dtype).contiguous(), requires_grad=False)   # Frozen Output Basis
            self.lora_Vh[adapter_name] = nn.Parameter(lora_Vh.to(dtype).contiguous(), requires_grad=False) # Frozen Input Basis
            
            # Store SVD dimensions
            k = lora_U.shape[1]  # Number of singular values
            
        # Create coefficient parameters specifically for SVD LoRA
        # Coefficients for A (Input side) - shape (r, k)
        if not hasattr(self, 'lora_coeffs_A'):
            self.lora_coeffs_A = nn.ModuleDict()
        # Using Linear layer without bias to store coefficients as trainable parameters
        lora_A_layer = nn.Linear(k, r, bias=False)
        # Initialize with zeros
        with torch.no_grad():
            lora_A_layer.weight.zero_()
        self.lora_coeffs_A[adapter_name] = lora_A_layer

        # Coefficients for B (Output side) - shape (k, r)
        if not hasattr(self, 'lora_coeffs_B'):
            self.lora_coeffs_B = nn.ModuleDict()
        # Using Linear layer without bias to store coefficients as trainable parameters
        lora_B_layer = nn.Linear(r, k, bias=False)
        # Initialize with zeros
        with torch.no_grad():
            lora_B_layer.weight.zero_()
        self.lora_coeffs_B[adapter_name] = lora_B_layer
        
        # Set other parameters
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        self.use_rslora[adapter_name] = use_rslora
        self.use_dora[adapter_name] = use_dora
        self.lora_bias[adapter_name] = lora_bias

        # Initialize the weights
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
            
        # Move adapter to device of base layer
        self._move_adapter_to_device_of_base_layer(adapter_name)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_coeffs_A.keys():
            # Initialize coefficient matrices for SVD LoRA
            lora_A_layer = self.lora_coeffs_A[adapter_name]
            lora_B_layer = self.lora_coeffs_B[adapter_name]
            # Initialize A with Kaiming uniform and B with zeros
            nn.init.kaiming_uniform_(lora_A_layer.weight, a=5**0.5)
            nn.init.zeros_(lora_B_layer.weight)

    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        raise NotImplementedError("SVD LoRA merging is not implemented yet.")

    def unmerge(self) -> None:
        raise NotImplementedError("SVD LoRA unmerging is not implemented yet.")

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter using SVD decomposition.
        
        Args:
            adapter_name (str): The name of the adapter for which the delta weight should be computed.
        """
        lora_U = self.lora_U[adapter_name]
        lora_Vh = self.lora_Vh[adapter_name]
        
        # Get weights from the Linear layers using proper parameter access
        lora_coeffs_A = self.lora_coeffs_A[adapter_name].weight.data  # Shape: (r, k)
        lora_coeffs_B = self.lora_coeffs_B[adapter_name].weight.data  # Shape: (k, r)
        
        # Reconstruct effective A and B matrices
        effective_A = lora_coeffs_A @ lora_Vh  # Shape: (r, n)
        effective_B = lora_U @ lora_coeffs_B   # Shape: (m, r)
        
        device = effective_B.device
        dtype = effective_B.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        if cast_to_fp32:
            effective_A = effective_A.float()
            effective_B = effective_B.float()

        output_tensor = effective_B @ effective_A * self.scaling[adapter_name]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_coeffs_A.keys():
                    continue

                lora_U = self.lora_U[active_adapter]
                lora_Vh = self.lora_Vh[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                # Apply dropout
                x_dropout = dropout(x)
                
                # SVD Path: x -> lora_Vh.T -> lora_coeffs_A.T -> lora_coeffs_B.T -> lora_U.T
                x_k = x_dropout @ lora_Vh.T  # Shape: (*, k)
                # x_r = x_k @ lora_coeffs_A.T  # Shape: (*, r)
                x_r = self.lora_coeffs_A[active_adapter](x_k)
                x_k_out = self.lora_coeffs_B[active_adapter](x_r)  # Shape: (*, k)
                lora_out = x_k_out @ lora_U.T  # Shape: (*, m)
                
                result = result + lora_out * scaling

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep