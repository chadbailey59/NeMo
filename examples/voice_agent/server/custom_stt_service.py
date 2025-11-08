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
Custom STT service that uses the patched streaming ASR service.
"""

from nemo.agents.voice_agent.pipecat.services.nemo.stt import NemoSTTService
from custom_streaming_asr import PatchedNemoStreamingASRService


class CustomNemoSTTService(NemoSTTService):
    """
    Custom STT service that uses PatchedNemoStreamingASRService to support
    prompt-conditioned ASR models.
    """

    def _load_model(self):
        """Load model using the patched streaming ASR service."""
        if self._backend == "legacy":
            self._model = PatchedNemoStreamingASRService(
                self._model_name, device=self._device, decoder_type=self._decoder_type
            )
        else:
            raise ValueError(f"Invalid ASR backend: {self._backend}")
