"""
Resolution Package

Provides input resolution, synthesis, and extraction for component execution.

Usage:
    from neurobrix.core.runtime.resolution import InputResolver, InputSynthesizer, OutputExtractor

    resolver = InputResolver(variable_resolver, connections_index, topology, loop_id)
    inputs = resolver.resolve_component_inputs("transformer")

    synthesizer = InputSynthesizer(topology)
    inputs = synthesizer.synthesize_inputs(component_name, variable_resolver)

    extractor = OutputExtractor(topology, variable_resolver)
    extractor.extract_outputs(component_name, outputs)
"""

from .input_resolver import InputResolver
from .input_synthesizer import InputSynthesizer
from .output_extractor import OutputExtractor

__all__ = ["InputResolver", "InputSynthesizer", "OutputExtractor"]
