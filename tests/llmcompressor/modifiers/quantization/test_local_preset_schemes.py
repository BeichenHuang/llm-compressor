"""
Tests for local preset schemes (W3A16, W2A16) defined in llm-compressor.

These schemes extend the preset schemes available in compressed_tensors
to support additional bit-widths like 3-bit and 2-bit quantization.
"""

import pytest
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from llmcompressor.modifiers.quantization.quantization.mixin import (
    LOCAL_PRESET_SCHEMES,
    is_preset_scheme,
    preset_name_to_scheme,
)


class TestLocalPresetSchemes:
    """Test local preset schemes functionality."""

    def test_local_schemes_defined(self):
        """Verify that local schemes are defined."""
        assert "W3A16" in LOCAL_PRESET_SCHEMES
        assert "W3A16_ASYM" in LOCAL_PRESET_SCHEMES
        assert "W2A16" in LOCAL_PRESET_SCHEMES

    def test_w3a16_scheme_config(self):
        """Test W3A16 scheme configuration."""
        w3a16 = LOCAL_PRESET_SCHEMES["W3A16"]
        assert "weights" in w3a16
        weights = w3a16["weights"]
        assert isinstance(weights, QuantizationArgs)
        assert weights.num_bits == 3
        assert weights.symmetric is True
        assert weights.group_size == 128

    def test_w3a16_asym_scheme_config(self):
        """Test W3A16_ASYM scheme configuration."""
        w3a16_asym = LOCAL_PRESET_SCHEMES["W3A16_ASYM"]
        assert "weights" in w3a16_asym
        weights = w3a16_asym["weights"]
        assert isinstance(weights, QuantizationArgs)
        assert weights.num_bits == 3
        assert weights.symmetric is False
        assert weights.group_size == 128

    def test_w2a16_scheme_config(self):
        """Test W2A16 scheme configuration."""
        w2a16 = LOCAL_PRESET_SCHEMES["W2A16"]
        assert "weights" in w2a16
        weights = w2a16["weights"]
        assert isinstance(weights, QuantizationArgs)
        assert weights.num_bits == 2
        assert weights.symmetric is True
        assert weights.group_size == 128


class TestIsPresetScheme:
    """Test is_preset_scheme function."""

    @pytest.mark.parametrize(
        "scheme_name",
        ["W3A16", "w3a16", "W3A16_ASYM", "w3a16_asym", "W2A16", "w2a16"],
    )
    def test_local_schemes_recognized(self, scheme_name):
        """Test that local schemes are recognized as valid presets."""
        assert is_preset_scheme(scheme_name) is True

    @pytest.mark.parametrize(
        "scheme_name",
        ["W4A16", "W8A8", "FP8", "FP8_DYNAMIC"],
    )
    def test_compressed_tensors_schemes_recognized(self, scheme_name):
        """Test that compressed_tensors schemes are still recognized."""
        assert is_preset_scheme(scheme_name) is True

    def test_invalid_scheme_not_recognized(self):
        """Test that invalid scheme names are not recognized."""
        assert is_preset_scheme("INVALID_SCHEME") is False
        assert is_preset_scheme("W5A16") is False


class TestPresetNameToScheme:
    """Test preset_name_to_scheme function."""

    def test_w3a16_creates_valid_scheme(self):
        """Test that W3A16 creates a valid QuantizationScheme."""
        targets = ["Linear"]
        scheme = preset_name_to_scheme("W3A16", targets)

        assert isinstance(scheme, QuantizationScheme)
        assert scheme.targets == targets
        assert scheme.weights is not None
        assert scheme.weights.num_bits == 3
        assert scheme.weights.symmetric is True

    def test_w3a16_asym_creates_valid_scheme(self):
        """Test that W3A16_ASYM creates a valid QuantizationScheme."""
        targets = ["Linear"]
        scheme = preset_name_to_scheme("W3A16_ASYM", targets)

        assert isinstance(scheme, QuantizationScheme)
        assert scheme.targets == targets
        assert scheme.weights is not None
        assert scheme.weights.num_bits == 3
        assert scheme.weights.symmetric is False

    def test_w2a16_creates_valid_scheme(self):
        """Test that W2A16 creates a valid QuantizationScheme."""
        targets = ["Linear"]
        scheme = preset_name_to_scheme("W2A16", targets)

        assert isinstance(scheme, QuantizationScheme)
        assert scheme.targets == targets
        assert scheme.weights is not None
        assert scheme.weights.num_bits == 2

    def test_compressed_tensors_scheme_still_works(self):
        """Test that compressed_tensors presets still work through this function."""
        targets = ["Linear"]
        scheme = preset_name_to_scheme("W4A16", targets)

        assert isinstance(scheme, QuantizationScheme)
        assert scheme.targets == targets
        assert scheme.weights is not None
        assert scheme.weights.num_bits == 4

    def test_case_insensitive(self):
        """Test that scheme names are case-insensitive for local schemes."""
        targets = ["Linear"]
        scheme_lower = preset_name_to_scheme("w3a16", targets)
        scheme_upper = preset_name_to_scheme("W3A16", targets)

        assert scheme_lower.weights.num_bits == scheme_upper.weights.num_bits
        assert scheme_lower.weights.symmetric == scheme_upper.weights.symmetric

    def test_invalid_scheme_raises_error(self):
        """Test that invalid scheme names raise KeyError."""
        with pytest.raises(KeyError):
            preset_name_to_scheme("INVALID_SCHEME", ["Linear"])

    def test_scheme_deepcopy(self):
        """Test that returned schemes are independent copies."""
        targets1 = ["Linear"]
        targets2 = ["Conv2d"]
        scheme1 = preset_name_to_scheme("W3A16", targets1)
        scheme2 = preset_name_to_scheme("W3A16", targets2)

        # Verify they have different targets
        assert scheme1.targets != scheme2.targets

        # Verify modifying one doesn't affect the other
        scheme1.targets.append("Embedding")
        assert "Embedding" not in scheme2.targets
