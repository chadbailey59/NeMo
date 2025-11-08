# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Custom streaming ASR service that patches the model config to include prompt_dictionary.
This is needed for models that require prompt conditioning (e.g., hybrid_rnnt_ctc_bpe_models_prompt).
"""

import torch
from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr
from nemo.agents.voice_agent.pipecat.services.nemo.streaming_asr import NemoStreamingASRService


class PatchedNemoStreamingASRService(NemoStreamingASRService):
    """
    Extended streaming ASR service that adds prompt_dictionary support.

    This class patches the model config before loading to add the required
    prompt_dictionary for prompt-conditioned ASR models.
    """

    def _load_model(self, model: str):
        """Load model with prompt_dictionary patch for prompt-conditioned models."""
        # Load the model using the standard approach
        # On Jetson platforms, load to CPU first to avoid CUDA memory allocation issues
        load_device = 'cpu'  # Always load to CPU first for Jetson compatibility
        target_device = torch.device(self.device)

        if model.endswith(".nemo"):
            asr_model = nemo_asr.models.ASRModel.restore_from(
                model,
                map_location=load_device,
                override_config_path=None
            )
        else:
            # For pretrained models, we need to check if they require prompt_dictionary
            # First, try loading normally
            try:
                asr_model = nemo_asr.models.ASRModel.from_pretrained(
                    model,
                    map_location=load_device
                )
            except ValueError as e:
                if "No prompt_dictionary found in config" in str(e):
                    # Model requires prompt_dictionary - patch the config
                    print(f"Model requires prompt_dictionary. Creating patched config...")

                    # Create a minimal prompt dictionary with common languages
                    # This can be extended based on your needs
                    prompt_dict = {
                        'en-US': 0,
                        'en-GB': 1,
                        'de-DE': 2,
                        'es-ES': 3,
                        'fr-FR': 4,
                        'it-IT': 5,
                        'pt-BR': 6,
                        'ja-JP': 7,
                        'zh-CN': 8,
                        'ko-KR': 9,
                        # Add more as needed, up to num_prompts (default 128)
                    }

                    # Download and get the config
                    from nemo.utils.model_utils import import_class_by_path
                    from nemo.collections.asr.models import ASRModel

                    # Get the model config without loading weights to GPU
                    pretrained_cfg = ASRModel.from_pretrained(
                        model,
                        return_config=True,
                        map_location='cpu'  # Load to CPU to avoid CUDA errors
                    )

                    # Patch the config with prompt_dictionary
                    with open_dict(pretrained_cfg):
                        if 'model_defaults' not in pretrained_cfg:
                            pretrained_cfg.model_defaults = {}
                        pretrained_cfg.model_defaults.prompt_dictionary = prompt_dict
                        pretrained_cfg.model_defaults.num_prompts = 128
                        pretrained_cfg.model_defaults.initialize_prompt_feature = False

                    # Save the patched config temporarily
                    import tempfile
                    import os

                    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                        OmegaConf.save(pretrained_cfg, f.name)
                        temp_config_path = f.name

                    try:
                        # Load with the patched config to CPU first
                        asr_model = nemo_asr.models.ASRModel.from_pretrained(
                            model,
                            override_config_path=temp_config_path,
                            map_location=load_device  # Load to CPU first
                        )
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_config_path):
                            os.unlink(temp_config_path)
                else:
                    # Different error, re-raise
                    raise

        # Move model to target device (CUDA) after loading
        if self.device != 'cpu':
            print(f"Moving model from CPU to {self.device}...")
            # Clear CUDA cache and run garbage collection before moving model (helps on Jetson)
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            asr_model = asr_model.to(target_device)

        # Apply decoder type changes if specified
        if self.decoder_type is not None and hasattr(asr_model, "cur_decoder"):
            asr_model.change_decoding_strategy(decoder_type=self.decoder_type)
        elif isinstance(asr_model, nemo_asr.models.EncDecCTCModel):
            self.decoder_type = "ctc"
        elif isinstance(asr_model, nemo_asr.models.EncDecRNNTModel):
            self.decoder_type = "rnnt"
        else:
            raise ValueError("Decoder type not supported for this model.")

        # Apply attention context size if specified
        if self.att_context_size is not None:
            if hasattr(asr_model.encoder, "set_default_att_context_size"):
                asr_model.encoder.set_default_att_context_size(att_context_size=self.att_context_size)
            else:
                raise ValueError("Model does not support multiple lookaheads.")
        else:
            self.att_context_size = asr_model.cfg.encoder.att_context_size

        # Configure decoding strategy
        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.compute_timestamps = False
            decoding_cfg.preserve_alignments = True
            if hasattr(asr_model, 'joint'):  # if an RNNT model
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.fused_batch_size = -1
            asr_model.change_decoding_strategy(decoding_cfg)

        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=self.att_context_size)

        # Setup streaming parameters if chunk_size is specified
        if self.chunk_size > 0:
            if self.shift_size < 0:
                shift_size = self.chunk_size
            else:
                shift_size = self.shift_size
            asr_model.encoder.setup_streaming_params(
                chunk_size=self.chunk_size, left_chunks=self.left_chunks, shift_size=shift_size
            )

        asr_model.eval()
        return asr_model
