# core/module/autoregressive/factory.py
"""
NeuroBrix Autoregressive Generator Factory - ZERO FALLBACK.

Complete registry of autoregressive generator implementations.
Unknown generator type = crash, not silent fallback.

IMPORTANT: This factory is for AUTOREGRESSIVE generation only.
Diffusion schedulers use core.module.scheduler.factory.
"""

from typing import Dict, Any, Type


class AutoregressiveConfigError(Exception):
    """Error in autoregressive generator configuration."""
    pass


class AutoregressiveFactory:
    """
    Factory to instantiate autoregressive generators from NBX config.

    ZERO FALLBACK: Unknown generator type = crash, not silent fallback.

    Supports:
    - AutoregressiveGenerator: Generic token-by-token generation
    - VQImageGenerator: VQ codebook token generation for images

    NOT for diffusion schedulers - use SchedulerFactory instead.
    """

    # Registry of supported generators
    # Maps config class name / alias to (module_path, class_name)
    _REGISTRY: Dict[str, tuple] = {
        # =========================================================================
        # Autoregressive Generators (token-by-token generation)
        # =========================================================================
        "AutoregressiveGenerator": ("generator", "AutoregressiveGenerator"),
        "autoregressive": ("generator", "AutoregressiveGenerator"),
        "ar": ("generator", "AutoregressiveGenerator"),

        "VQImageGenerator": ("generator", "VQImageGenerator"),
        "vq_image": ("generator", "VQImageGenerator"),
        "janus": ("generator", "VQImageGenerator"),  # Alias for Janus models

        # =========================================================================
        # LLM Samplers (can be used standalone)
        # =========================================================================
        "greedy": ("samplers", "GreedySampler"),
        "GreedySampler": ("samplers", "GreedySampler"),

        "top_k": ("samplers", "TopKSampler"),
        "TopKSampler": ("samplers", "TopKSampler"),

        "top_p": ("samplers", "TopPSampler"),
        "TopPSampler": ("samplers", "TopPSampler"),
        "nucleus": ("samplers", "TopPSampler"),

        "temperature": ("samplers", "TemperatureSampler"),
        "TemperatureSampler": ("samplers", "TemperatureSampler"),

        "combined": ("samplers", "CombinedSampler"),
        "CombinedSampler": ("samplers", "CombinedSampler"),
    }

    # Cache for imported classes
    _CLASS_CACHE: Dict[str, Type] = {}

    @classmethod
    def _get_class(cls, module_path: str, class_name: str) -> Type:
        """Lazily import and return generator class."""
        cache_key = f"{module_path}.{class_name}"

        if cache_key in cls._CLASS_CACHE:
            return cls._CLASS_CACHE[cache_key]

        # Import the module
        import importlib
        full_module = f"neurobrix.core.module.autoregressive.{module_path}"

        try:
            module = importlib.import_module(full_module)
            generator_class = getattr(module, class_name)
            cls._CLASS_CACHE[cache_key] = generator_class
            return generator_class
        except ImportError as e:
            raise AutoregressiveConfigError(
                f"ZERO FALLBACK: Failed to import generator module.\n"
                f"Module: {full_module}\n"
                f"Class: {class_name}\n"
                f"Error: {e}"
            )
        except AttributeError:
            raise AutoregressiveConfigError(
                f"ZERO FALLBACK: Generator class not found in module.\n"
                f"Module: {full_module}\n"
                f"Class: {class_name}"
            )

    @classmethod
    def create(cls, config: Dict[str, Any]):
        """
        Create a generator instance from NBX config.

        ZERO FALLBACK: Crashes if generator type unknown or missing.

        Args:
            config: Generator configuration from NBX container

        Returns:
            Generator instance

        Raises:
            AutoregressiveConfigError: If generator type missing or unknown
        """
        # 1. Get generator type - REQUIRED
        generator_type = config.get("_class_name")

        if not generator_type:
            raise AutoregressiveConfigError(
                f"\n{'='*70}\n"
                f"ZERO FALLBACK VIOLATION: Generator config missing '_class_name'\n"
                f"{'='*70}\n"
                f"Config keys: {sorted(config.keys())}\n"
                f"\n"
                f"The NBX container MUST specify which generator to use.\n"
                f"Check runtime/defaults.json in the .nbx file.\n"
                f"{'='*70}"
            )

        # 2. Look up implementation - REQUIRED
        if generator_type not in cls._REGISTRY:
            available = sorted(set(cls._REGISTRY.keys()))
            raise AutoregressiveConfigError(
                f"\n{'='*70}\n"
                f"ZERO FALLBACK VIOLATION: Unknown generator type\n"
                f"{'='*70}\n"
                f"Requested: '{generator_type}'\n"
                f"Available: {available}\n"
                f"\n"
                f"To add support for this generator:\n"
                f"1. Implement the generator in core/module/autoregressive/\n"
                f"2. Register it in AutoregressiveFactory._REGISTRY\n"
                f"{'='*70}"
            )

        # 3. Get class and create instance
        module_path, class_name = cls._REGISTRY[generator_type]
        generator_class = cls._get_class(module_path, class_name)

        return generator_class.from_config(config)

    @classmethod
    def create_by_name(cls, name: str, config: Dict[str, Any] | None = None):
        """
        Create generator by name (convenience method).

        Args:
            name: Generator name/alias (e.g., "vq_image", "autoregressive")
            config: Optional config dict

        Returns:
            Generator instance
        """
        config = config or {}
        config["_class_name"] = name
        return cls.create(config)

    @classmethod
    def list_available(cls) -> list:
        """List all available generator types (unique class names only)."""
        # Return unique generator types (not aliases)
        unique_types = set()
        for key, (module, classname) in cls._REGISTRY.items():
            unique_types.add(classname)
        return sorted(unique_types)

    @classmethod
    def list_aliases(cls) -> Dict[str, str]:
        """List all aliases and their target generators."""
        return {
            key: f"{module}.{classname}"
            for key, (module, classname) in cls._REGISTRY.items()
        }

    @classmethod
    def get_generator_info(cls, generator_type: str) -> Dict[str, Any]:
        """Get information about a generator type."""
        if generator_type not in cls._REGISTRY:
            return {"error": f"Unknown generator: {generator_type}"}

        module_path, class_name = cls._REGISTRY[generator_type]
        return {
            "name": generator_type,
            "class": class_name,
            "module": f"neurobrix.core.module.autoregressive.{module_path}",
            "aliases": [
                k for k, v in cls._REGISTRY.items()
                if v == (module_path, class_name)
            ],
        }
