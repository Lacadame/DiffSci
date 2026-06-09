# flake8: noqa

from .mlp import MLPUncond, MLPCond
from .hfnet import HFNetCond, HFNetUncond
from .difftransformer import DiffusionTransformer

from .punetg_config import PUNetGConfig
from .punetg_encdec import PUNetGEncoder, PUNetGDecoder
from .punetg import PUNetG, PUNetGConfig, PUNetGCond
from .adm import ADM, ADMConfig

from .embedder import (
    PositionalEncoding1d,
    ScalarEmbedder,
    VectorEmbedder,
    FunctionEmbedder,
    SequenceTransformer,
    CompositeEmbedder,
)

from .vaenet import VAENet, VAENetConfig
from .vaenet_mp import VAENetMP, VAENetMPConfig
from . import commonlayers
from . import localattn

from .normedlayers import (
    # CONFIG-D / CONFIG-E learned layers (forced WN + weight norm on use).
    normalize,
    MagnitudePreservingLinear,
    MagnitudePreservingConv2d,
    MagnitudePreservingConv3d,
    # CONFIG-G fixed-function and lightweight learned layers.
    PixelNorm,
    Gain,
    mp_silu,
    mp_sum,
)

from .patched_conv import (
    patch_conv_1d,
    patch_conv_2d,
    patch_conv_3d,
    get_patch_conv,
)

from .convit import ConVit, ConVitBlock, ConVitConfig

from .enhanced_conditioning import (
    FiLMLayer,
    SpatialConditionEncoder,
    ConditionAmplifier,
    EnhancedConditioningWrapper,
    wrap_model_with_enhanced_conditioning,
)
