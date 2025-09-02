
import torch
import psutil
from typing import Tuple
from dataclasses import dataclass


@dataclass
class OptimizationConfig:
    use_amp: bool = True
    use_channels_last: bool = True


class DeviceManager:
    
    def __init__(self):
        self.device, self.dtype, self.config = self._detect_device()
        self.whisper_device = "cuda" if self.device == "cuda" else "cpu"
        
    def _detect_device(self) -> Tuple[str, torch.dtype, OptimizationConfig]:
        config = OptimizationConfig()

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            dtype = (
                torch.bfloat16
                if torch.cuda.get_device_capability()[0] >= 8
                else torch.float16
            )
            return "cuda", dtype, config

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            config.use_channels_last = False
            return "mps", torch.float32, config

        torch.set_num_threads(psutil.cpu_count(logical=False))
        config.use_amp = False
        config.use_channels_last = False
        return "cpu", torch.float32, config

