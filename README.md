# Origami Aesthetics MCP Server

A Model Context Protocol (MCP) server that translates origami-inspired aesthetic concepts into concrete visual parameters. Built on a categorical framework that preserves structural relationships across visual domains.

## Categorical Framework

This server implements a four-layer hierarchy:

```
Intent Layer → Aesthetic Layer → Visual Parameters → Execution Layer
```

### Why Origami?

Origami provides exceptionally rich visual vocabulary because:

1. **Structural Precision**: Geometric relationships are mathematically defined
2. **Dimensional Transformation**: The 2D→3D functor is origami's defining characteristic
3. **Universal Recognition**: Cross-cultural aesthetic with immediate visual associations
4. **Compositional Logic**: Fold sequences create emergent complexity from simple rules

## Core Concepts

### Aesthetic Concepts
- **precise**: Sharp folds, geometric accuracy, crisp edges
- **organic**: Curved creases, natural flow, soft transitions
- **minimal**: Few folds, clean lines, extensive negative space
- **complex**: Intricate detail, multiple layers, high fold count
- **geometric**: Mathematical patterns, angular precision, structural emphasis
- **elegant**: Refined simplicity, economical use of folds, graceful form
- **architectural**: Structural logic, monumental scale implication, engineered precision
- **meditative**: Contemplative repetition, rhythmic patterns, calm presence

### Fold Types
- **Mountain/Valley**: Sharp directional folds (convex/concave)
- **Curved Crease**: Organic, flowing transitions
- **Reverse/Sink/Petal**: Complex structural transformations
- **Pleat**: Rhythmic alternating folds

### Pattern Types
- **Traditional**: Classic origami forms (crane, box, etc.)
- **Tessellation**: Repeating geometric patterns
- **Modular**: Unit-based assembly, interlocking components
- **Recursive**: Self-similar patterns, fractal-like nesting
- **Computational**: Algorithmically generated, extreme precision
- **Organic Curved**: Wet-folding, sculptural forms

## Tools

### 1. Intent Analysis
```python
analyze_origami_intent(
    intent_description="precise geometric tessellation with architectural presence",
    desired_complexity="complex",
    emphasis=["geometric", "architectural"]
)
```

Extracts aesthetic concepts from natural language (Intent → Aesthetic layer).

### 2. Fold Parameters
```python
generate_fold_parameters(
    aesthetic_concepts=["precise", "geometric", "complex"],
    complexity="complex",
    pattern_type="tessellation"
)
```

Generates concrete fold characteristics (Aesthetic → Visual Parameters layer).

### 3. Lighting Parameters
```python
generate_lighting_parameters(
    aesthetic_concepts=["elegant", "minimal"],
    lighting_style="backlit",
    material_type="washi"
)
```

Creates lighting that reveals origami structure effectively.

### 4. Composition Parameters
```python
generate_composition_parameters(
    aesthetic_concepts=["architectural", "geometric"],
    symmetry="radial",
    view_angle="three_quarter"
)
```

Generates framing and perspective parameters.

### 5. Complete Specification
```python
generate_complete_specification(
    intent_description="elegant minimalist form with soft lighting",
    complexity="intermediate",
    pattern_type="traditional",
    lighting_style="ambient_soft",
    material_type="washi",
    symmetry="bilateral",
    view_angle="three_quarter"
)
```

Chains all deterministic layers into execution-ready specification.

### 6. Vocabulary Access
```python
get_origami_vocabulary()
```

Returns complete taxonomy of concepts, mappings, and properties.

### 7. Concept Comparison
```python
compare_aesthetic_profiles(
    concept_a="precise",
    concept_b="organic"
)
```

Analyzes differences between aesthetic concepts.

## Cross-Domain Functors

### Grid Dynamics Mapping
```python
map_to_grid_dynamics(
    pattern_type="tessellation",
    complexity="complex"
)
```

Maps origami crease patterns to spatial arrangement structures. Preserves:
- Geometric relationships
- Focal point logic
- Propagation patterns

Example: A radial origami pattern (crane) maps to center-focused grid dynamics with outward radiation, similar to nuclear blast visualization.

### Narrative Structure Mapping
```python
map_to_narrative_structure(
    aesthetic_concepts=["precise", "meditative"],
    complexity="intermediate",
    pattern_type="traditional"
)
```

Maps origami aesthetics to story structure. Preserves:
- Tension progression (fold sequence → plot development)
- Revelation patterns (unfolding form → character revelation)
- Structural logic (geometric precision → narrative precision)

Example: Complex recursive origami maps to nested story-within-story narrative structure.

## Usage Examples

### Example 1: Architectural Visualization
```python
# Generate spec for precise architectural origami aesthetic
spec = generate_complete_specification(
    intent_description="architectural precision with strong geometric presence",
    complexity="complex",
    pattern_type="computational",
    lighting_style="dramatic_side",
    material_type="foil",
    symmetry="bilateral",
    view_angle="three_quarter"
)

# Result includes:
# - Edge sharpness: 0.93 (architectural precision)
# - Fold accuracy: 0.97
# - Strong directional shadows
# - Metallic material with high reflectivity
# - Structural emphasis in composition
```

### Example 2: Meditative Simplicity
```python
spec = generate_complete_specification(
    intent_description="calm meditative simplicity with organic flow",
    complexity="simple",
    pattern_type="traditional",
    lighting_style="ambient_soft",
    material_type="washi",
    symmetry="bilateral",
    view_angle="three_quarter"
)

# Result includes:
# - Edge sharpness: 0.8 (softer, contemplative)
# - Fold count: 5-15 (minimal)
# - Gentle gradient shadows
# - Translucent washi with fibrous texture
# - Extensive negative space
```

### Example 3: Complex Tessellation
```python
spec = generate_complete_specification(
    intent_description="intricate repeating geometric pattern",
    complexity="super_complex",
    pattern_type="tessellation",
    lighting_style="backlit",
    material_type="tissue_foil",
    symmetry="radial",
    view_angle="flat_overhead"
)

# Result includes:
# - Fold count: 100-300 (extreme detail)
# - Hexagonal/triangular grid base
# - Backlit to reveal internal pattern structure
# - Tissue foil for color depth with minimal translucency
# - Overhead view to show pattern clarity
```

### Example 4: Cross-Modal Workflow
```python
# 1. Generate origami aesthetic
origami_spec = generate_complete_specification(
    intent_description="dangerous precision with sharp angles",
    complexity="complex",
    pattern_type="geometric"
)

# 2. Map to grid dynamics for spatial composition
grid_params = map_to_grid_dynamics(
    pattern_type="geometric",
    complexity="complex"
)

# 3. Map to narrative for story generation
narrative_params = map_to_narrative_structure(
    aesthetic_concepts=["precise", "geometric"],
    complexity="complex",
    pattern_type="geometric"
)

# Result: Coherent aesthetic across visual and narrative domains
# - Visual: Sharp angular composition with geometric precision
# - Narrative: Character with meticulous, logical mindset
#   in precisely structured plot with mathematical progression
```

## Architecture Notes

### Deterministic vs. Creative Layers

This server implements **deterministic mappings** only. All tools perform:
- Taxonomy lookups
- Parameter calculations
- Structural mappings

For **creative synthesis** (final prompt generation, artistic decisions), use an LLM with domain-specific system instructions:

```python
# Recommended LLM system instruction for origami synthesis:
"""
You translate origami aesthetic concepts into visual composition parameters. 
Focus on:
- Geometric precision and fold logic
- Layer relationships and depth
- The transformation from flat to dimensional
- How light reveals folded structure
- Balance between simplicity and complexity

Consider both traditional origami (crane, box) and modern computational 
origami (curved creases, tessellations).
"""
```

### Cost Optimization

Following the hybrid architecture pattern:

1. **Deterministic (this MCP)**: ~60-85% of token savings
   - Intent extraction
   - Taxonomy mapping
   - Parameter lookup
   - Structural transformations

2. **Creative synthesis (LLM)**: Focus on genuinely creative decisions
   - Artistic interpretation within constraints
   - Novel combinations
   - Context-sensitive refinements

### Compositional Patterns

Origami brick composes cleanly with other aesthetic bricks:

- **Origami + Material Texture**: "Brushed metal with precise geometric folds"
- **Origami + Grid Dynamics**: "Radial tessellation with explosive propagation"
- **Origami + Lighting**: "Backlit translucent layers revealing internal structure"
- **Origami + Color Palette**: "Monochromatic with sharp contrast at fold lines"

Each brick operates in its categorical domain, preserving independent aspects of the final composition.

## Testing Framework

### Validation Tests

1. **Structural Preservation**: Does "precise geometric" produce expected parameters?
2. **Compositional Coherence**: Do multi-brick workflows maintain aesthetic consistency?
3. **Cross-Domain Functors**: Does origami→grid dynamics preserve geometric relationships?
4. **Parameter Sensitivity**: Do small changes produce expected visual differences?

### Example Test
```python
# Test: "precise" should yield higher edge_sharpness than "organic"
precise_params = generate_fold_parameters(["precise"], "intermediate", "traditional")
organic_params = generate_fold_parameters(["organic"], "intermediate", "organic_curved")

assert precise_params["edge_sharpness"] > organic_params["edge_sharpness"]
assert precise_params["fold_accuracy"] > organic_params["fold_accuracy"]
# Expected: precise=0.95, organic=0.4
```

## Installation

```bash
# Clone or download server
cd origami-aesthetics-mcp

# Install FastMCP (if not already installed)
pip install fastmcp --break-system-packages

# Run server
python server.py
```

## Integration with Lushy

This MCP server provides:
- **Origami Brick**: Complete aesthetic system as packageable brick
- **Cross-modal functors**: Origami → other domains (grid, narrative, etc.)
- **Material/lighting bricks**: Specific aspects extracted as standalone bricks
- **Compositional logic**: How origami aesthetics combine with other bricks

Users can:
1. Use origami brick directly for origami-inspired visuals
2. Compose with other bricks (materials, lighting, grid dynamics)
3. Map origami structure to narrative or other modalities
4. Create origami-specific sub-bricks (tessellation, wet-folding, etc.)

## Categorical Theory Notes

### Functors Implemented

1. **Intent → Aesthetic**: Natural language → Concept extraction
2. **Aesthetic → Parameters**: Concepts → Concrete values
3. **Origami → Grid Dynamics**: Crease pattern → Spatial arrangement
4. **Origami → Narrative**: Fold structure → Story structure

### Preservation Properties

Each functor preserves specific structural relationships:
- **Geometric functors**: Angles, proportions, symmetry
- **Progression functors**: Sequential logic, build-up patterns
- **Complexity functors**: Detail density, layer depth, intricacy
- **Tension functors**: Contrast, balance, focal points

### Composition Laws

Origami brick composes via:
- **Product composition**: Independent aspects (material × lighting × origami)
- **Sequential composition**: Each layer feeds the next (intent → aesthetic → params)
- **Functorial composition**: Cross-domain mappings preserve structure

## License

MIT License - See LICENSE file for details

## Credits

Built on categorical framework developed for aesthetic composition systems.
Implements four-layer hierarchy: Intent → Aesthetic → Visual Parameters → Execution.

## Contributing

Contributions welcome! Key areas:
- Additional fold type mappings
- New pattern characteristics
- Cross-domain functors to other modalities
- Validation test cases
- Integration examples
