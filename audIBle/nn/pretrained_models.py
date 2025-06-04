import torch
import torch.nn as nn
import torchaudio
import numpy as np
from transformers import (
    Wav2Vec2Model, Wav2Vec2FeatureExtractor,
    HubertModel,
    ASTModel, ASTFeatureExtractor,
    AutoModel, AutoFeatureExtractor
)
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore")

class BaseAudioEncoder(nn.Module):
    """Base class for audio encoder modules"""
    
    def __init__(self, model_name: str, freeze_encoder: bool = True):
        super().__init__()
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        self.feature_extractor = None
        self.encoder = None
        
    def _freeze_encoder(self):
        """Freeze encoder parameters"""
        if self.freeze_encoder and self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
    def preprocess_audio(self, audio: torch.Tensor, sampling_rate: int = 16000) -> Dict[str, torch.Tensor]:
        """Preprocess audio using the model's feature extractor"""
        # Convert to numpy for feature extractor
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
            
        # Handle batch dimension
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)
        elif audio_np.ndim == 3:
            audio_np = audio_np.squeeze(1)  # Remove channel dimension if present
            
        inputs = self.feature_extractor(
            audio_np, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        )
        
        return inputs

class WavLMEncoder(BaseAudioEncoder):
    """WavLM encoder module"""
    
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", freeze_encoder: bool = True):
        super().__init__(model_name, freeze_encoder)
        
        print(f"Loading WavLM: {model_name}")
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        self.hidden_size = self.encoder.config.hidden_size
        self.num_layers = self.encoder.config.num_hidden_layers + 1  # +1 for input embeddings
        
        self._freeze_encoder()
        
    def forward(self, audio: torch.Tensor, output_hidden_states: bool = True, 
                layer_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through WavLM
        
        Args:
            audio: Input audio tensor [batch_size, sequence_length] or [batch_size, 1, sequence_length]
            output_hidden_states: Whether to return all hidden states
            layer_indices: Specific layer indices to return (if None, returns all)
            
        Returns:
            Dictionary containing representations
        """
        # Preprocess audio
        inputs = self.preprocess_audio(audio)
        
        # Move inputs to same device as model
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.encoder(
            inputs['input_values'], 
            output_hidden_states=output_hidden_states
        )
        
        result = {
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output if hasattr(outputs, 'pooler_output') else None
        }
        
        if output_hidden_states:
            hidden_states = outputs.hidden_states
            if layer_indices is not None:
                hidden_states = [hidden_states[i] for i in layer_indices if i < len(hidden_states)]
            result['hidden_states'] = hidden_states
            
        return result

class HuBERTEncoder(BaseAudioEncoder):
    """HuBERT encoder module"""
    
    def __init__(self, model_name: str = "facebook/hubert-base-ls960", freeze_encoder: bool = True):
        super().__init__(model_name, freeze_encoder)
        
        print(f"Loading HuBERT: {model_name}")
        self.encoder = HubertModel.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        self.hidden_size = self.encoder.config.hidden_size
        self.num_layers = self.encoder.config.num_hidden_layers + 1
        
        self._freeze_encoder()
        
    def forward(self, audio: torch.Tensor, output_hidden_states: bool = True, 
                layer_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through HuBERT"""
        # Preprocess audio
        inputs = self.preprocess_audio(audio)
        
        # Move inputs to same device as model
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.encoder(
            inputs['input_values'], 
            output_hidden_states=output_hidden_states
        )
        
        result = {
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output if hasattr(outputs, 'pooler_output') else None
        }
        
        if output_hidden_states:
            hidden_states = outputs.hidden_states
            if layer_indices is not None:
                hidden_states = [hidden_states[i] for i in layer_indices if i < len(hidden_states)]
            result['hidden_states'] = hidden_states
            
        return result

class ASTEncoder(BaseAudioEncoder):
    """Audio Spectrogram Transformer encoder module"""
    
    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593", freeze_encoder: bool = True):
        super().__init__(model_name, freeze_encoder)
        
        print(f"Loading AST: {model_name}")
        self.encoder = ASTModel.from_pretrained(model_name)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        
        self.hidden_size = self.encoder.config.hidden_size
        self.num_layers = self.encoder.config.num_hidden_layers + 1
        
        self._freeze_encoder()
        
    def forward(self, audio: torch.Tensor, output_hidden_states: bool = True, 
                layer_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through AST"""
        # Preprocess audio
        inputs = self.preprocess_audio(audio)
        
        # Move inputs to same device as model
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.encoder(
            inputs['input_values'], 
            output_hidden_states=output_hidden_states
        )
        
        result = {
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output if hasattr(outputs, 'pooler_output') else None
        }
        
        if output_hidden_states:
            hidden_states = outputs.hidden_states
            if layer_indices is not None:
                hidden_states = [hidden_states[i] for i in layer_indices if i < len(hidden_states)]
            result['hidden_states'] = hidden_states
            
        return result

class BEATsEncoder(BaseAudioEncoder):
    """BEATs encoder module"""
    
    def __init__(self, model_name: str = "microsoft/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2", freeze_encoder: bool = True):
        super().__init__(model_name, freeze_encoder)
        
        try:
            print(f"Loading BEATs: {model_name}")
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
            
            # Try to get config info
            if hasattr(self.encoder, 'config'):
                self.hidden_size = getattr(self.encoder.config, 'hidden_size', 768)
                self.num_layers = getattr(self.encoder.config, 'num_hidden_layers', 12) + 1
            else:
                self.hidden_size = 768  # Default
                self.num_layers = 13    # Default
                
            self._freeze_encoder()
            
        except Exception as e:
            print(f"Error loading BEATs: {e}")
            print("BEATs might require specific installation or model access")
            self.encoder = None
            self.feature_extractor = None
            
    def forward(self, audio: torch.Tensor, output_hidden_states: bool = True, 
                layer_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through BEATs"""
        if self.encoder is None:
            raise ValueError("BEATs encoder not properly loaded")
            
        # Preprocess audio
        inputs = self.preprocess_audio(audio)
        
        # Move inputs to same device as model
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.encoder(
            **inputs, 
            output_hidden_states=output_hidden_states
        )
        
        result = {
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output if hasattr(outputs, 'pooler_output') else None
        }
        
        if output_hidden_states:
            hidden_states = outputs.hidden_states
            if layer_indices is not None:
                hidden_states = [hidden_states[i] for i in layer_indices if i < len(hidden_states)]
            result['hidden_states'] = hidden_states
            
        return result

class MultiAudioEncoder(nn.Module):
    """Multi-encoder module that combines multiple audio encoders"""
    
    def __init__(self, encoder_configs: Dict[str, Dict], freeze_encoders: bool = True):
        """
        Initialize multi-encoder
        
        Args:
            encoder_configs: Dict mapping encoder names to their configs
                Example: {
                    'wavlm': {'model_name': 'microsoft/wavlm-base-plus'},
                    'hubert': {'model_name': 'facebook/hubert-base-ls960'},
                    'ast': {'model_name': 'MIT/ast-finetuned-audioset-10-10-0.4593'}
                }
            freeze_encoders: Whether to freeze encoder parameters
        """
        super().__init__()
        
        self.encoders = nn.ModuleDict()
        self.encoder_info = {}
        
        # Available encoder classes
        encoder_classes = {
            'wavlm': WavLMEncoder,
            'hubert': HuBERTEncoder,
            'ast': ASTEncoder,
            'beats': BEATsEncoder
        }
        
        # Initialize encoders
        for encoder_name, config in encoder_configs.items():
            if encoder_name.lower() in encoder_classes:
                encoder_class = encoder_classes[encoder_name.lower()]
                model_name = config.get('model_name', None)
                
                try:
                    if model_name:
                        encoder = encoder_class(model_name=model_name, freeze_encoder=freeze_encoders)
                    else:
                        encoder = encoder_class(freeze_encoder=freeze_encoders)
                        
                    self.encoders[encoder_name] = encoder
                    self.encoder_info[encoder_name] = {
                        'hidden_size': encoder.hidden_size,
                        'num_layers': encoder.num_layers,
                        'model_name': encoder.model_name
                    }
                    print(f"✓ {encoder_name} loaded successfully")
                    
                except Exception as e:
                    print(f"✗ Failed to load {encoder_name}: {e}")
            else:
                print(f"✗ Unknown encoder type: {encoder_name}")
                
    def forward(self, audio: torch.Tensor, encoder_names: Optional[List[str]] = None, 
                output_hidden_states: bool = True, layer_indices: Optional[Dict[str, List[int]]] = None) -> Dict[str, Dict]:
        """
        Forward pass through selected encoders
        
        Args:
            audio: Input audio tensor
            encoder_names: List of encoder names to use (if None, use all)
            output_hidden_states: Whether to return hidden states
            layer_indices: Dict mapping encoder names to layer indices
            
        Returns:
            Dict mapping encoder names to their outputs
        """
        if encoder_names is None:
            encoder_names = list(self.encoders.keys())
            
        results = {}
        
        for encoder_name in encoder_names:
            if encoder_name in self.encoders:
                try:
                    # Get layer indices for this encoder
                    enc_layer_indices = None
                    if layer_indices and encoder_name in layer_indices:
                        enc_layer_indices = layer_indices[encoder_name]
                        
                    # Forward pass
                    output = self.encoders[encoder_name](
                        audio, 
                        output_hidden_states=output_hidden_states,
                        layer_indices=enc_layer_indices
                    )
                    results[encoder_name] = output
                    
                except Exception as e:
                    print(f"Error in {encoder_name} forward pass: {e}")
                    results[encoder_name] = {}
            else:
                print(f"Encoder {encoder_name} not available")
                
        return results
    
    def get_encoder_info(self) -> Dict[str, Dict]:
        """Get information about loaded encoders"""
        return self.encoder_info
    
    def get_combined_hidden_size(self, encoder_names: Optional[List[str]] = None) -> int:
        """Get combined hidden size of specified encoders"""
        if encoder_names is None:
            encoder_names = list(self.encoders.keys())
            
        total_size = 0
        for name in encoder_names:
            if name in self.encoder_info:
                total_size += self.encoder_info[name]['hidden_size']
                
        return total_size

# Utility functions for downstream tasks
class AudioClassifier(nn.Module):
    """Example classifier using audio encoder representations"""
    
    def __init__(self, encoder: Union[BaseAudioEncoder, MultiAudioEncoder], 
                 num_classes: int, hidden_size: Optional[int] = None,
                 use_layer_indices: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                 pooling_method: str = 'mean'):
        """
        Audio classifier with pretrained encoder
        
        Args:
            encoder: Audio encoder module
            num_classes: Number of output classes
            hidden_size: Hidden size (auto-detected if None)
            use_layer_indices: Which layers to use for classification
            pooling_method: How to pool sequence representations ('mean', 'max', 'cls')
        """
        super().__init__()
        
        self.encoder = encoder
        self.use_layer_indices = use_layer_indices
        self.pooling_method = pooling_method
        
        # Determine input size for classifier
        if hidden_size is None:
            if isinstance(encoder, MultiAudioEncoder):
                hidden_size = encoder.get_combined_hidden_size()
            else:
                hidden_size = encoder.hidden_size
                
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def pool_representations(self, representations: torch.Tensor) -> torch.Tensor:
        """Pool sequence representations"""
        if self.pooling_method == 'mean':
            return representations.mean(dim=1)
        elif self.pooling_method == 'max':
            return representations.max(dim=1)[0]
        elif self.pooling_method == 'cls':
            return representations[:, 0]  # Use CLS token
        else:
            return representations.mean(dim=1)  # Default to mean
            
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Get encoder outputs
        if isinstance(self.encoder, MultiAudioEncoder):
            outputs = self.encoder(audio, output_hidden_states=True, layer_indices=self.use_layer_indices)
            
            # Combine representations from all encoders
            combined_reprs = []
            for encoder_name, encoder_output in outputs.items():
                if 'last_hidden_state' in encoder_output:
                    repr_tensor = encoder_output['last_hidden_state']
                    pooled = self.pool_representations(repr_tensor)
                    combined_reprs.append(pooled)
                    
            if combined_reprs:
                features = torch.cat(combined_reprs, dim=-1)
            else:
                raise ValueError("No valid representations found")
                
        else:
            outputs = self.encoder(audio, output_hidden_states=True, layer_indices=self.use_layer_indices)
            
            # Use specified layer or last hidden state
            if self.use_layer_indices and 'hidden_states' in outputs:
                # Use specified layer (take first if multiple)
                layer_idx = self.use_layer_indices[0] if isinstance(self.use_layer_indices, list) else self.use_layer_indices
                features = self.pool_representations(outputs['hidden_states'][layer_idx])
            else:
                features = self.pool_representations(outputs['last_hidden_state'])
                
        # Classify
        logits = self.classifier(features)
        return logits

# Example usage and helper functions
def load_audio_file(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load and preprocess audio file"""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample if necessary
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
        
    return waveform.squeeze()

# Example usage
if __name__ == "__main__":
    # Example 1: Single encoder
    print("=== Single Encoder Example ===")
    wavlm = WavLMEncoder(freeze_encoder=True)
    
    # Create dummy audio (batch_size=2, sequence_length=16000)
    dummy_audio = torch.randn(2, 16000)
    
    # Forward pass
    outputs = wavlm(dummy_audio, output_hidden_states=True)
    print(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")
    print(f"Number of hidden states: {len(outputs['hidden_states'])}")
    
    # Example 2: Multi-encoder
    print("\n=== Multi-Encoder Example ===")
    encoder_configs = {
        'wavlm': {'model_name': 'microsoft/wavlm-base-plus'},
        'hubert': {'model_name': 'facebook/hubert-base-ls960'},
        'ast': {'model_name': 'MIT/ast-finetuned-audioset-10-10-0.4593'}
    }
    
    multi_encoder = MultiAudioEncoder(encoder_configs, freeze_encoders=True)
    
    # Forward pass with specific layers
    layer_indices = {
        'wavlm': [6, 8, 10],  # Use layers 6, 8, 10
        'hubert': [5, 7, 9],  # Use layers 5, 7, 9
        'ast': [4, 6, 8]      # Use layers 4, 6, 8
    }
    
    multi_outputs = multi_encoder(dummy_audio, layer_indices=layer_indices)
    
    for encoder_name, output in multi_outputs.items():
        if 'last_hidden_state' in output:
            print(f"{encoder_name} - Last hidden state: {output['last_hidden_state'].shape}")
            print(f"{encoder_name} - Selected layers: {len(output['hidden_states'])}")
    
    # Example 3: Classification task
    print("\n=== Classification Example ===")
    classifier = AudioClassifier(
        encoder=multi_encoder,
        num_classes=10,
        use_layer_indices={'wavlm': [8], 'hubert': [7], 'ast': [6]},
        pooling_method='mean'
    )
    
    # Forward pass for classification
    logits = classifier(dummy_audio)
    print(f"Classification logits shape: {logits.shape}")
    
    # Get encoder info
    print(f"\nEncoder info: {multi_encoder.get_encoder_info()}")
    print(f"Combined hidden size: {multi_encoder.get_combined_hidden_size()}")