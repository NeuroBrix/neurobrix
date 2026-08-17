"""Weight-storage encoding support registry.

A build variant may store eligible weight matrices in a packed
encoding instead of a dense floating dtype (the manifest declares it
under `weight_encoding.scheme`; the encoded component's weights_index
`dtype` carries the same name, and each encoded tensor rides as a
`.qweight` / `.scales` / `.qmins` triplet). This registry is the
single authority on which schemes THIS engine version can execute —
the runtime loader refuses any build declaring a scheme outside it,
loudly and before any weight I/O.

int4-g128-asym: weight-only int4, groups of 128 along in_features,
asymmetric (fp16 scale + fp16 min per group), 8 nibbles per int32
little-nibble-first, packed [K//8, N]; dequant is q*scale+min in fp32
(the dequant-GEMV kernel family's canonical form).
"""

SUPPORTED_WEIGHT_ENCODINGS = {"int4-g128-asym"}
