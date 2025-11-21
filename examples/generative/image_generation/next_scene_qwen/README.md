# Next-Scene Qwen Image LoRA

Cinematic scene progression using Qwen-Image-Edit with Next-Scene LoRA for creating natural visual transitions and narrative continuity in sequential storytelling.

## Features

- **Camera Dynamics**: Dolly shots, tracking moves, angle shifts, and reframing
- **Environmental Progression**: Character reveals, expanded scenery, spatial evolution
- **Atmospheric Transitions**: Lighting changes, weather shifts, time-of-day evolution
- **Narrative Continuity**: Preserves spatial relationships and emotional resonance across frames

## Model Details

- **Base Model**: Qwen-Image-Edit (build 2509)
- **LoRA Version**: V2 (enhanced quality, improved prompt responsiveness)
- **Type**: Low-Rank Adaptation for sequential image generation
- **Strength**: 0.7-0.8 recommended
- **License**: MIT

## Requirements

- NVIDIA GPU with 10GB+ VRAM
- Docker with NVIDIA Container Toolkit
- 25GB disk space for model weights

## Quick Start

### 1. Build the Docker image

```bash
./build.sh
```

### 2. Generate a single scene

```bash
./predict.sh "The camera moves forward revealing a mysterious doorway"
```

Output will be saved to `outputs/scene.png`

### 3. Generate a multi-scene narrative

```bash
./sequence.sh "Wide shot of a city,Camera zooms to a window,Inside view of a character,Close-up of their hands"
```

Or use a JSON file:

```bash
./sequence.sh scenes_example.json
```

Output will be saved to `outputs/sequence/scene_001.png`, `scene_002.png`, etc.

## Usage with CVL CLI

```bash
# Build the container
cvl run next-scene-qwen build

# Generate single scene
cvl run next-scene-qwen predict "Camera pans across a landscape"

# Generate sequence
cvl run next-scene-qwen sequence scenes.json
```

## Scene Description Format

All prompts are automatically prefixed with "Next Scene:" for optimal results with the LoRA.

### Best Practices

1. **Include camera movement**: "Camera dollies forward", "Tracking shot following", "Angle shifts up"
2. **Describe progression**: "revealing", "transitioning to", "expanding view"
3. **Set atmosphere**: "soft morning light", "shadows growing longer", "mist rolling in"
4. **Maintain continuity**: Reference elements from previous scenes for coherent narrative

### Example Prompts

```
"Wide establishing shot of a coastal town at sunset, camera slowly pushes in"
"Camera moves closer, revealing cobblestone streets and warm window lights"
"Tracking shot following a lone figure walking toward the harbor"
"Camera tilts down to focus on footsteps in wet pavement, reflecting streetlamps"
"Angle shifts up as the figure stops at the dock, looking out at the ocean"
```

## Advanced Usage

### Single Scene Generation

```bash
python3 generate_scene.py \
    --prompt "Camera pans revealing a hidden garden" \
    --input-image previous_scene.png \
    --output outputs/next_scene.png \
    --lora-scale 0.75 \
    --strength 0.6 \
    --width 1024 \
    --height 768 \
    --seed 42
```

**Parameters**:
- `--prompt`: Scene description (auto-prefixed with "Next Scene:")
- `--input-image`: Previous scene for continuity (optional)
- `--output`: Output image path
- `--lora-scale`: LoRA strength (0.7-0.8 recommended, default: 0.75)
- `--strength`: How much to transform from input image (default: 0.6)
- `--width/height`: Output dimensions (default: 1024x768)
- `--negative-prompt`: What to avoid (default includes "black bars")
- `--seed`: Random seed for reproducibility

### Multi-Scene Sequence

```bash
python3 generate_sequence.py \
    --scenes scenes.json \
    --output-dir outputs/my_sequence \
    --output-video outputs/sequence.mp4 \
    --lora-scale 0.75 \
    --fps 2 \
    --seed 42
```

**Parameters**:
- `--scenes`: JSON file or comma-separated prompts
- `--output-dir`: Directory for scene images
- `--output-video`: Optional video output (combines scenes)
- `--fps`: Frames per second for video (default: 2)
- `--strength`: Scene-to-scene transformation strength (default: 0.6)
- Other parameters same as single scene generation

### JSON Scene Format

```json
{
  "scenes": [
    "First scene description with camera movement",
    "Second scene building on the first",
    "Third scene continuing the narrative",
    "Final scene completing the sequence"
  ]
}
```

## Tips for Best Results

1. **LoRA Strength**: Start with 0.75 and adjust
   - Lower (0.6-0.7): More subtle cinematic progression
   - Higher (0.8-0.9): Stronger next-scene transitions

2. **Scene-to-Scene Strength**: Controls continuity
   - Lower (0.4-0.5): Maintains more from previous scene
   - Higher (0.7-0.8): Allows more dramatic transitions

3. **Prompt Style**: Works best with:
   - Landscape/establishing shots
   - Camera movement descriptions
   - Atmospheric and lighting details
   - Directional language ("revealing", "moving toward")

4. **Avoid**:
   - Abrupt scene changes without transition language
   - Conflicting spatial descriptions
   - The model may create black bars - use negative prompt to reduce

## Creating Videos from Sequences

The sequence generator can optionally create videos:

```bash
./sequence.sh scenes.json --output-video outputs/narrative.mp4 --fps 3
```

This creates a slideshow-style video with each scene as a frame. Adjust `--fps` for pacing:
- **1-2 fps**: Slow, contemplative pacing
- **3-4 fps**: Standard narrative flow
- **6+ fps**: Faster transitions

## Example Workflows

### Cinematic Story Sequence

1. Create `story.json`:
```json
{
  "scenes": [
    "Wide aerial shot of a cyberpunk city at night, neon lights glowing",
    "Camera descends into the streets, revealing crowded markets",
    "Tracking shot following a hooded figure through the crowd",
    "Camera moves closer as figure enters a dark alley",
    "Low angle looking up as figure approaches a hidden door",
    "Door opens revealing bright light, camera pushes through"
  ]
}
```

2. Generate:
```bash
./sequence.sh story.json --output-video story.mp4 --fps 2 --seed 123
```

### Progressive Environment Reveal

```bash
./sequence.sh \
  "Foggy morning, barely visible silhouette of a building,\
   Fog begins to lift revealing gothic architecture,\
   Camera pans revealing intricate stone details,\
   Warm sunrise light illuminates the entire cathedral,\
   Wide establishing shot with cathedral towering against clear sky" \
  --output-dir outputs/cathedral_reveal \
  --lora-scale 0.8
```

## Model Source

- **HuggingFace**: [lovis93/next-scene-qwen-image-lora-2509](https://huggingface.co/lovis93/next-scene-qwen-image-lora-2509)
- **Version**: V2 (improved quality and prompt responsiveness)
- **License**: MIT License
- **Base**: Qwen-Image-Edit build 2509

## Troubleshooting

**Black bars appearing in images**:
- Use negative prompt: "black bars, letterbox, borders"
- Adjust `--lora-scale` to 0.7 or lower
- Ensure aspect ratio matches typical cinematic formats

**Inconsistent scene progression**:
- Use `--input-image` to feed previous scene
- Lower `--strength` to maintain more continuity
- Include references to previous scene in prompt

**Out of Memory**:
- Reduce `--width` and `--height`
- Close other GPU applications
- Enable gradient checkpointing (modify script)

**LoRA not loading**:
- Verify model ID is correct
- Check HuggingFace authentication if model requires access
- Ensure sufficient disk space for download

## Future Enhancements

Potential improvements for this example:
- Support for ComfyUI integration
- Automatic scene interpolation
- Style consistency across sequences
- Audio synchronization for video output
