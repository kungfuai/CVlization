This is a fun problem! Style transfer from clean synthetic sheet music to vintage/aged renderings is well-suited to a few approaches depending on how much control and quality you need.
Foundation Models / Architectures Worth Considering
Diffusion-based (best quality)
	∙	ControlNet + Stable Diffusion is probably the strongest starting point. You’d use the synthetic sheet music as a structural condition (via a “canny edge” or “scribble” ControlNet), then guide generation toward vintage aesthetics via prompt or a trained LoRA. This gives you tight structural fidelity while allowing heavy stylistic transformation.
	∙	InstructPix2Pix (or similar instruction-tuned diffusion models) can work if you frame it as an editing task (“make this look like aged 19th century engraved sheet music”), though structural fidelity can drift.
GAN-based (faster, more controllable)
	∙	CycleGAN is the classic unpaired image-to-image choice — great if you don’t have paired (synthetic → vintage) examples, which you probably don’t. It’s well-studied and there’s prior work on document style transfer with it.
	∙	Pix2Pix if you can construct paired examples (e.g., by scanning vintage scores and computationally “cleaning” them to create synthetic counterparts, then training in reverse).
Texture/neural style transfer
	∙	Classic Gatys-style NST or more modern variants can apply vintage paper texture, ink bleed, and aging artifacts while mostly preserving structure — simpler to implement but less photorealistic than diffusion.
Practical Recommendations
For your specific use case, I’d suggest:
ControlNet + SD with a vintage-tuned LoRA is the most promising path. Train a LoRA on a dataset of scanned vintage scores (IMSLP is a goldmine here), then condition on edges/structure from your synthetic input. You get both style fidelity and structural preservation.
Two-stage pipeline: First do a structural extraction (binarize, extract staff lines and notation), then run diffusion-based stylization, then optionally composite aging artifacts (paper texture, yellowing, foxing) as a post-processing pass. This prevents notation from being garbled.
Domain-specific fine-tuning matters a lot — general “vintage” LoRAs won’t capture the specific ink spread patterns, plate engraving artifacts, or lithographic textures of historical scores. Curating a dataset by era/printing technique (e.g., copper engraving vs. lithography vs. early typeset) will dramatically improve results.
Dataset Resources
	∙	IMSLP has thousands of public domain scanned scores across centuries of printing styles
	∙	MUSCIMA++ for synthetic/clean score data
	∙	OpenScore for clean engraved MusicXML-derived images
Are you aiming for a specific era of vintage rendering (baroque engraving, romantic-era lithography, early 20th century print)? That would shape the approach quite a bit.
