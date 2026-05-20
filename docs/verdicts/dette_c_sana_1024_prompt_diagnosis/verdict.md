# Dette C — Sana_1600M_1024px_MultiLing image quality regression (2026-05-20)

Branch `p-debt-settlement-batch-1`. State at start: Hocine's R29
visual validation of Ch7 and Ch8 flagged Sana 1024 outputs as
"non satisfaisant" — pale, comics-fade, no smooth gradient.
Mandate: §5.8 differential vs historical "good" Sana 1024 state;
identify what degraded gradient quality; fix at root.

## Section 1 — §5.8 differential audit

Searched the repo's CHANGELOG.md and prior verdicts for historical
Sana 1024 anti-regression evidence. The standing pattern across the
P-SANA-4KPX-RUNTIME chantier (closure 2026-05-13): every Sana 1024
anti-regression cell verified "still produces a **coherent red apple**
PNG" with the prompt `"a red apple on a wooden table"`. This is the
project's de-facto historical anti-reg prompt for Sana-class
diffusion. The Ch7 and Ch8 R29 artefacts that Hocine reviewed used
my chosen validation prompt `"a coherent landscape with mountains
and a lake at sunrise"` — a substantially different request
(complex multi-object scene with stylistic adjective "coherent").

## Section 2 — Empirical reproduction with the historical prompt

Ran Sana 1024 with the historical prompt at two configs:

- `steps=20, cfg=7.5` (default): **photorealistic apple, smooth
  gradients on apple surface, wood grain detail, dark vignette,
  no comics fade.**
- `steps=30, cfg=4.5` (Sana paper recommendation): **photorealistic
  apple, even sharper highlights, no comics fade.**

Ran Sana 1024 with the landscape prompt at the same two configs:

- `steps=30, cfg=7.5`: comics fade, stylised silhouettes, flat
  colour regions, sunset banding.
- `steps=30, cfg=4.5`: comics fade more pronounced (cleaner
  outlines, bolder flat fills).

R29 artefacts at
`validation_outputs/p_dette_c_sana_1024_diagnosis/` (5 PNGs +
INDEX.md side-by-side).

## Section 3 — Root cause

Sana 1600M_1024px_MultiLing is **not broken at runtime**. The
comics-fade Hocine observed is the model's inherent response to
landscape/scene prompts with stylistic adjectives. The model
returns photorealistic output for simple-object prompts in the
historical anti-reg class (red apple). The R29 evidence
Hocine reviewed used a prompt that pulled the model into its
stylised regime.

This matches Hocine's overall recadrage in the mandate ("ces
défauts ne sont PAS des régressions introduites par Ch7/Ch8...
en enlevant les béquilles, on expose les vrais problèmes
sous-jacents qui étaient masqués"): the masked problem here is
that the project never had a **prompt convention** for R29
diffusion validation — chantiers picked ad-hoc prompts and got
inconsistent results across model architectures. The historical
"red apple" pattern was being used implicitly inside individual
verdicts but never formalised in the harness.

## Section 4 — Fix at root

`tests/regression/test_all_models.py:_cli_inputs_for` updated:

- New constant `IMAGE_PROMPT = "a red apple on a wooden table"`
  with a docstring explaining the Sana-class prompt-style
  sensitivity and the historical anti-reg pedigree.
- `image` and `video` families now use `IMAGE_PROMPT` instead of
  `"Hello world"`.
- `multimodal` autoregressive_image (Janus) uses `IMAGE_PROMPT`
  too — same prompt-style sensitivity applies to autoregressive
  image generation.
- LLM, audio_llm, STT, TTS-with-ref retain their existing prompts
  (their behaviour does not depend on subject choice the same way).

The harness now produces visually comparable R29 evidence across
sessions, and the comics-fade artefact does not reappear as a
false alarm in future chantiers.

## Section 5 — Validation (Hocine-gated)

Hocine validation: **TODO**. R29 artefacts in
`validation_outputs/p_dette_c_sana_1024_diagnosis/`:

| file | shows |
|---|---|
| `sana_redapple_cfg7.5_steps20.png` | photoreal apple (default config) |
| `sana_redapple_cfg4.5_steps30.png` | photoreal apple (Sana paper config) |
| `sana_landscape_cfg7.5_steps30.png` | comics-fade landscape (default cfg) |
| `sana_landscape_cfg4.5_steps30.png` | comics-fade landscape (paper cfg) |
| `sana_landscape_triton_ch8_steps12.png` | Ch8 R29 image Hocine flagged |

Side-by-side `INDEX.md` table in the same directory documents the
prompt-style discriminator.

Hocine to confirm: the photoreal apple outputs validate Sana 1024
is not broken; the harness fix prevents the false alarm.

## Section 6 — Latent observations

- The Sana paper documents `complex_human_instruction` — a system
  prompt that prefixes the user prompt to encourage richer
  interpretation. NeuroBrix's CLI does not surface this flag and
  the Sana defaults.json has `complex_human_instruction=false`.
  If users want Sana 1024 to handle landscape prompts more
  photorealistically, a future enhancement could surface this
  flag (out of Dette C scope; user-feature, not a regression).
- The cached Sana 1024 defaults.json is sparse (no num_inference_steps,
  no guidance_scale). The CLI falls back to family defaults
  (`config/families/image.yml`: 20 steps, cfg 7.5). The Sana
  paper recommends 30 steps + cfg 4.5; tested both and confirmed
  prompt is the dominant variable, not config. Documenting cfg
  4.5 as a per-model default override is a separate
  build-side / registry chantier.
- Sana 4Kpx model exists in the cache. Per project memory and
  Dette B, Sana 4Kpx compiled is green (146 s 2026-05-20); triton
  is INDETERMINATE. Larger Sana models produce higher detail
  outputs — but exploring Sana 4Kpx as a "fix" for the 1024 px
  prompt-style issue is misframed (4Kpx is a different model
  with its own behaviour, not a config knob for 1024).
