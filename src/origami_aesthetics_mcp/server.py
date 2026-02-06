"""
Origami Aesthetics MCP Server - Phase 1A Enhanced
==================================================

Translates origami-inspired aesthetic concepts into visual composition parameters.
Focuses on geometric precision, dimensional transformation, and fold logic.

PHASE 1A ENHANCEMENT:
- Trajectory computation between aesthetic states
- Flow field navigation in origami morphospace
- RK4 integration for smooth transitions
- Convergence analysis and validation

Categorical framework:
- Intent Layer: High-level aesthetic goals (precise, organic, complex, minimal)
- Aesthetic Layer: Origami-specific concepts (fold types, patterns, techniques)
- Visual Parameters Layer: Concrete visual specs (edge sharpness, layering, geometry)
- Execution Layer: Ready for image generation or 3D modeling
"""

from fastmcp import FastMCP
from typing import Literal, Optional
from enum import Enum
import numpy as np


# Phase 1A: Import aesthetic-dynamics-core with graceful degradation
try:
    from aesthetic_dynamics_core import (
        _integrate_trajectory_impl,
        _compute_gradient_field_impl,
        _analyze_convergence_impl
    )
    DYNAMICS_AVAILABLE = True
except ImportError:
    # Graceful degradation if aesthetic-dynamics-core not installed
    DYNAMICS_AVAILABLE = False
    _integrate_trajectory_impl = None
    _compute_gradient_field_impl = None
    _analyze_convergence_impl = None

mcp = FastMCP("origami-aesthetics-mcp")

# Version tracking for Phase 2.7
SERVER_VERSION = "1.3.0-phase2.7"
VALIDATION_DATE = "2026-02-06"

# =============================================================================
# DETERMINISTIC TAXONOMY - Core origami vocabulary
# =============================================================================

class FoldType(str, Enum):
    """Basic origami fold types with their geometric properties"""
    MOUNTAIN = "mountain_fold"
    VALLEY = "valley_fold" 
    REVERSE = "reverse_fold"
    SINK = "sink_fold"
    PETAL = "petal_fold"
    SQUASH = "squash_fold"
    CURVED_CREASE = "curved_crease"
    PLEAT = "pleat"

class CreasePattern(str, Enum):
    """Origami crease pattern styles"""
    TRADITIONAL = "traditional"
    TESSELLATION = "tessellation"
    MODULAR = "modular"
    RECURSIVE = "recursive"
    COMPUTATIONAL = "computational"
    ORGANIC_CURVED = "organic_curved"

class ComplexityLevel(str, Enum):
    """Fold complexity levels"""
    SIMPLE = "simple"           # 5-15 folds, basic forms
    INTERMEDIATE = "intermediate"  # 15-40 folds, recognizable complexity
    COMPLEX = "complex"         # 40-100 folds, intricate detail
    SUPER_COMPLEX = "super_complex"  # 100+ folds, extreme detail

class SymmetryType(str, Enum):
    """Symmetry patterns in origami"""
    BILATERAL = "bilateral"
    RADIAL = "radial"
    ROTATIONAL = "rotational"
    ASYMMETRIC = "asymmetric"
    FRACTAL = "fractal"

# Deterministic fold property mappings
FOLD_PROPERTIES = {
    FoldType.MOUNTAIN: {
        "convexity": 1.0,
        "sharpness": 0.9,
        "light_interaction": "catches_light",
        "visual_prominence": "high",
        "shadow_cast": "strong"
    },
    FoldType.VALLEY: {
        "convexity": -1.0,
        "sharpness": 0.9,
        "light_interaction": "receives_shadow",
        "visual_prominence": "medium",
        "shadow_cast": "minimal"
    },
    FoldType.CURVED_CREASE: {
        "convexity": 0.5,
        "sharpness": 0.3,
        "light_interaction": "gradual_transition",
        "visual_prominence": "medium",
        "shadow_cast": "soft"
    },
    FoldType.PLEAT: {
        "convexity": 0.0,
        "sharpness": 0.95,
        "light_interaction": "alternating",
        "visual_prominence": "high",
        "shadow_cast": "rhythmic"
    }
}

# Aesthetic concept to fold characteristic mappings
# PHASE 1A: These define the origami aesthetic morphospace
AESTHETIC_STATES = {
    "precise": {
        "name": "Precise Origami",
        "coordinates": {
            "edge_sharpness": 0.95,
            "fold_accuracy": 0.98,
            "complexity_level": 0.6,
            "organic_quality": 0.1,
            "dimensional_depth": 0.45
        },
        "description": "Sharp edges, perfect folds, geometric precision"
    },
    "organic": {
        "name": "Organic Origami",
        "coordinates": {
            "edge_sharpness": 0.4,
            "fold_accuracy": 0.7,
            "complexity_level": 0.5,
            "organic_quality": 0.9,
            "dimensional_depth": 0.75
        },
        "description": "Soft edges, curved creases, natural forms"
    },
    "minimal": {
        "name": "Minimal Origami",
        "coordinates": {
            "edge_sharpness": 0.85,
            "fold_accuracy": 0.95,
            "complexity_level": 0.2,
            "organic_quality": 0.3,
            "dimensional_depth": 0.30
        },
        "description": "Few folds, clean lines, essential form"
    },
    "complex": {
        "name": "Complex Origami",
        "coordinates": {
            "edge_sharpness": 0.8,
            "fold_accuracy": 0.9,
            "complexity_level": 0.95,
            "organic_quality": 0.4,
            "dimensional_depth": 0.85
        },
        "description": "Many folds, intricate layers, elaborate detail"
    },
    "geometric": {
        "name": "Geometric Origami",
        "coordinates": {
            "edge_sharpness": 0.92,
            "fold_accuracy": 0.96,
            "complexity_level": 0.65,
            "organic_quality": 0.15,
            "dimensional_depth": 0.55
        },
        "description": "Mathematical patterns, angular forms, symmetry"
    },
    "elegant": {
        "name": "Elegant Origami",
        "coordinates": {
            "edge_sharpness": 0.88,
            "fold_accuracy": 0.94,
            "complexity_level": 0.55,
            "organic_quality": 0.45,
            "dimensional_depth": 0.60
        },
        "description": "Balanced complexity, refined execution, understated beauty"
    },
    "architectural": {
        "name": "Architectural Origami",
        "coordinates": {
            "edge_sharpness": 0.93,
            "fold_accuracy": 0.97,
            "complexity_level": 0.75,
            "organic_quality": 0.2,
            "dimensional_depth": 0.90
        },
        "description": "Structural logic, monumental scale, engineered precision"
    },
    "meditative": {
        "name": "Meditative Origami",
        "coordinates": {
            "edge_sharpness": 0.8,
            "fold_accuracy": 0.9,
            "complexity_level": 0.5,
            "organic_quality": 0.5,
            "dimensional_depth": 0.50
        },
        "description": "Rhythmic repetition, contemplative pace, balanced presence"
    }
}

# PHASE 1A: Define parameter names for trajectory computation
# Phase 2.7: Expanded to 5D for framework-standard morphospace
ORIGAMI_PARAMETER_NAMES = [
    "edge_sharpness",
    "fold_accuracy", 
    "complexity_level",
    "organic_quality",
    "dimensional_depth"     # 0.0 = flat/overhead, 1.0 = highly sculptural 3D
]

# PHASE 1A: Define bounds for origami morphospace
ORIGAMI_BOUNDS = [0.0, 1.0]  # All parameters normalized to [0, 1]

# Original mappings preserved (used by existing tools)
AESTHETIC_MAPPINGS = {
    "precise": {
        "edge_sharpness": 0.95,
        "fold_accuracy": 0.98,
        "corner_treatment": "sharp",
        "line_consistency": "uniform",
        "preferred_folds": [FoldType.MOUNTAIN, FoldType.VALLEY],
        "surface_quality": "crisp"
    },
    "organic": {
        "edge_sharpness": 0.4,
        "fold_accuracy": 0.7,
        "corner_treatment": "rounded",
        "line_consistency": "variable",
        "preferred_folds": [FoldType.CURVED_CREASE],
        "surface_quality": "soft"
    },
    "minimal": {
        "edge_sharpness": 0.85,
        "fold_accuracy": 0.95,
        "corner_treatment": "clean",
        "line_consistency": "sparse",
        "fold_count": "low",
        "negative_space": "high",
        "surface_quality": "pristine"
    },
    "complex": {
        "edge_sharpness": 0.8,
        "fold_accuracy": 0.9,
        "corner_treatment": "sharp",
        "line_consistency": "dense",
        "fold_count": "high",
        "layer_depth": "multiple",
        "surface_quality": "intricate"
    },
    "geometric": {
        "edge_sharpness": 0.92,
        "fold_accuracy": 0.96,
        "corner_treatment": "angular",
        "line_consistency": "mathematical",
        "symmetry_emphasis": "strong",
        "pattern_regularity": "high",
        "surface_quality": "technical"
    },
    "elegant": {
        "edge_sharpness": 0.88,
        "fold_accuracy": 0.94,
        "corner_treatment": "refined",
        "line_consistency": "balanced",
        "fold_efficiency": "high",
        "form_simplicity": "understated",
        "surface_quality": "refined"
    },
    "architectural": {
        "edge_sharpness": 0.93,
        "fold_accuracy": 0.97,
        "corner_treatment": "structured",
        "line_consistency": "engineered",
        "scale_implication": "monumental",
        "structural_logic": "apparent",
        "surface_quality": "precise"
    },
    "meditative": {
        "edge_sharpness": 0.8,
        "fold_accuracy": 0.9,
        "corner_treatment": "intentional",
        "line_consistency": "rhythmic",
        "repetition_quality": "contemplative",
        "pace_implication": "slow",
        "surface_quality": "calm"
    }
}

# Crease pattern characteristics
PATTERN_CHARACTERISTICS = {
    CreasePattern.TRADITIONAL: {
        "grid_base": "square",
        "complexity": "moderate",
        "recognizability": "high",
        "cultural_association": "japanese_classical",
        "fold_sequence": "linear",
        "final_form": "representational"
    },
    CreasePattern.TESSELLATION: {
        "grid_base": ["triangular", "hexagonal", "square"],
        "complexity": "high",
        "recognizability": "pattern_based",
        "repetition": "extensive",
        "modularity": "high",
        "final_form": "geometric_surface"
    },
    CreasePattern.MODULAR: {
        "grid_base": "unit_based",
        "complexity": "variable",
        "recognizability": "structural",
        "assembly_logic": "interlocking",
        "scalability": "high",
        "final_form": "constructed_whole"
    },
    CreasePattern.RECURSIVE: {
        "grid_base": "fractal",
        "complexity": "very_high",
        "recognizability": "mathematical",
        "self_similarity": "strong",
        "scale_variation": "multiple",
        "final_form": "nested_patterns"
    },
    CreasePattern.COMPUTATIONAL: {
        "grid_base": "algorithmic",
        "complexity": "extreme",
        "recognizability": "technical",
        "design_method": "computer_generated",
        "precision_requirement": "very_high",
        "final_form": "impossible_by_hand"
    },
    CreasePattern.ORGANIC_CURVED: {
        "grid_base": "freeform",
        "complexity": "moderate_high",
        "recognizability": "natural",
        "curvature": "extensive",
        "technique": "wet_folding",
        "final_form": "sculptural_organic"
    }
}

# Material properties for origami paper
MATERIAL_PROPERTIES = {
    "traditional_kami": {
        "thickness": 0.07,  # mm
        "stiffness": 0.4,
        "texture": "smooth",
        "translucency": 0.3,
        "color_saturation": 0.8
    },
    "washi": {
        "thickness": 0.12,
        "stiffness": 0.3,
        "texture": "fibrous",
        "translucency": 0.5,
        "color_saturation": 0.6
    },
    "foil": {
        "thickness": 0.05,
        "stiffness": 0.7,
        "texture": "metallic",
        "translucency": 0.0,
        "reflectivity": 0.9
    },
    "tissue_foil": {
        "thickness": 0.03,
        "stiffness": 0.5,
        "texture": "fine",
        "translucency": 0.1,
        "color_depth": 0.9
    },
    "wet_fold_paper": {
        "thickness": 0.15,
        "stiffness": 0.2,
        "texture": "slightly_rough",
        "translucency": 0.2,
        "curvature_capacity": 0.95
    }
}

# Lighting interaction patterns
LIGHTING_INTERACTIONS = {
    "backlit": {
        "translucency_emphasis": "high",
        "layer_visibility": "internal_structure_revealed",
        "edge_treatment": "glowing_rims",
        "shadow_character": "soft_diffused",
        "depth_perception": "layered_translucent"
    },
    "dramatic_side": {
        "translucency_emphasis": "low",
        "layer_visibility": "surface_only",
        "edge_treatment": "sharp_highlights",
        "shadow_character": "strong_directional",
        "depth_perception": "geometric_planes"
    },
    "ambient_soft": {
        "translucency_emphasis": "medium",
        "layer_visibility": "subtle_depth",
        "edge_treatment": "soft_definition",
        "shadow_character": "gentle_gradients",
        "depth_perception": "organic_form"
    },
    "studio_clean": {
        "translucency_emphasis": "controlled",
        "layer_visibility": "technical_clarity",
        "edge_treatment": "precise_definition",
        "shadow_character": "minimal_neutral",
        "depth_perception": "architectural_clear"
    }
}

# =============================================================================
# PHASE 1A: TRAJECTORY DYNAMICS
# =============================================================================

def _compute_trajectory_between_aesthetic_states_impl(
    start_aesthetic_id: str,
    end_aesthetic_id: str,
    num_steps: int = 30,
    return_analysis: bool = True
) -> dict:
    """
    Core implementation of trajectory computation between origami aesthetic states.
    
    PHASE 1A ENHANCEMENT
    Layer 2: Zero-cost deterministic trajectory integration
    
    Args:
        start_aesthetic_id: Starting aesthetic (e.g., "precise", "organic")
        end_aesthetic_id: Target aesthetic
        num_steps: Number of integration steps
        return_analysis: Include convergence analysis
    
    Returns:
        Complete trajectory with convergence metrics and path analysis
    """
    if not DYNAMICS_AVAILABLE:
        return {
            "error": "aesthetic-dynamics-core not installed",
            "message": "Install with: pip install aesthetic-dynamics-core --break-system-packages",
            "fallback": "Use compare_aesthetic_profiles for static comparison"
        }
    
    # Validate aesthetic IDs
    if start_aesthetic_id not in AESTHETIC_STATES:
        return {
            "error": f"Unknown start aesthetic: {start_aesthetic_id}",
            "available": list(AESTHETIC_STATES.keys())
        }
    
    if end_aesthetic_id not in AESTHETIC_STATES:
        return {
            "error": f"Unknown end aesthetic: {end_aesthetic_id}",
            "available": list(AESTHETIC_STATES.keys())
        }
    
    # Get state coordinates
    start_state = AESTHETIC_STATES[start_aesthetic_id]
    end_state = AESTHETIC_STATES[end_aesthetic_id]
    
    start_coords = start_state["coordinates"]
    end_coords = end_state["coordinates"]
    
    # Compute trajectory using aesthetic-dynamics-core
    trajectory_result = _integrate_trajectory_impl(
        start_state=start_coords,
        target_state=end_coords,
        parameter_names=ORIGAMI_PARAMETER_NAMES,
        num_steps=num_steps,
        bounds=ORIGAMI_BOUNDS,
        convergence_threshold=0.01
    )
    
    # Prepare response
    response = {
        "start_aesthetic": {
            "id": start_aesthetic_id,
            "name": start_state["name"],
            "description": start_state["description"],
            "coordinates": start_coords
        },
        "end_aesthetic": {
            "id": end_aesthetic_id,
            "name": end_state["name"],
            "description": end_state["description"],
            "coordinates": end_coords
        },
        "trajectory": {
            "states": trajectory_result["trajectory"],
            "num_steps": trajectory_result["num_steps"],
            "parameter_names": ORIGAMI_PARAMETER_NAMES
        },
        "convergence": {
            "converged": trajectory_result["converged"],
            "convergence_step": trajectory_result["convergence_step"],
            "final_distance": trajectory_result["final_distance"],
            "convergence_threshold": 0.01
        },
        "path_metrics": {
            "geodesic_length": trajectory_result["path_length"],
            "euclidean_distance": trajectory_result["initial_distance"],
            "path_efficiency": trajectory_result["initial_distance"] / trajectory_result["path_length"] 
                              if trajectory_result["path_length"] > 0 else 1.0
        },
        "dynamics_info": {
            "integration_method": "RK4 (Runge-Kutta 4th order)",
            "bounds": str(ORIGAMI_BOUNDS),
            "cost": "0 tokens (pure Layer 2)",
            "morphospace": "4D origami aesthetic space"
        }
    }
    
    # Add convergence analysis if requested
    if return_analysis and trajectory_result["converged"]:
        analysis = _analyze_convergence_impl(
            trajectory=trajectory_result["trajectory"],
            target_state=end_coords,
            parameter_names=ORIGAMI_PARAMETER_NAMES,
            threshold=0.01
        )
        
        response["convergence_analysis"] = {
            "monotonic_decrease": analysis["monotonic_decrease"],
            "oscillation_count": analysis["oscillation_count"],
            "convergence_rate": analysis["convergence_rate"],
            "distance_reduction": analysis["distance_reduction"]
        }
    
    return response


@mcp.tool()
def compute_trajectory_between_aesthetic_states(
    start_aesthetic_id: str,
    end_aesthetic_id: str,
    num_steps: int = 30,
    return_analysis: bool = True
) -> dict:
    """
    Compute smooth trajectory between two origami aesthetic states.
    
    NEW PHASE 1A TOOL: Uses aesthetic-dynamics-core for zero-cost trajectory
    integration via RK4. Enables visualization of smooth aesthetic transitions,
    understanding of intermediate states, and validation of morphospace structure.
    
    This answers questions like:
    - "What's the smoothest transition from precise to organic?"
    - "What intermediate aesthetics exist between minimal and complex?"
    - "How do we smoothly evolve from geometric to meditative?"
    
    Args:
        start_aesthetic_id: Starting aesthetic ("precise", "organic", "minimal", 
                           "complex", "geometric", "elegant", "architectural", "meditative")
        end_aesthetic_id: Target aesthetic (same options)
        num_steps: Number of integration steps (default: 30)
        return_analysis: Include convergence analysis (default: True)
    
    Returns:
        Dictionary with trajectory data, convergence metrics, and path analysis
    
    Cost: 0 tokens (pure Layer 2 deterministic computation)
    
    Example:
        >>> compute_trajectory_between_aesthetic_states(
        ...     "precise",
        ...     "organic",
        ...     num_steps=20
        ... )
        {
            "start_aesthetic": {
                "id": "precise",
                "name": "Precise Origami",
                "description": "Sharp edges, perfect folds, geometric precision"
            },
            "end_aesthetic": {
                "id": "organic", 
                "name": "Organic Origami",
                "description": "Soft edges, curved creases, natural forms"
            },
            "trajectory": [...],  # 21 intermediate states
            "converged": true,
            "path_metrics": {
                "geodesic_length": 0.847,
                "euclidean_distance": 0.831,
                "path_efficiency": 0.981
            }
        }
    """
    return _compute_trajectory_between_aesthetic_states_impl(
        start_aesthetic_id, end_aesthetic_id, num_steps, return_analysis
    )


# =============================================================================
# EXISTING CORE TOOLS - Intent to Parameters (Preserved)
# =============================================================================

@mcp.tool()
def analyze_origami_intent(
    intent_description: str,
    desired_complexity: Literal["simple", "intermediate", "complex", "super_complex"] = "intermediate",
    emphasis: list[str] = None
) -> dict:
    """
    Analyze user intent and extract origami aesthetic concepts.
    
    This is the Intent → Aesthetic layer mapping (deterministic extraction).
    
    Args:
        intent_description: Natural language description of desired aesthetic
        desired_complexity: Target complexity level
        emphasis: Optional list of aspects to emphasize (geometric, organic, minimal, etc.)
    
    Returns:
        Extracted aesthetic concepts with confidence scores
    """
    # Extract keywords from intent description
    intent_lower = intent_description.lower()
    
    # Map keywords to aesthetic concepts
    concept_keywords = {
        "precise": ["precise", "sharp", "crisp", "exact", "accurate", "perfect"],
        "organic": ["organic", "natural", "flowing", "curved", "soft", "smooth"],
        "minimal": ["minimal", "simple", "clean", "essential", "sparse", "few"],
        "complex": ["complex", "intricate", "detailed", "elaborate", "layered"],
        "geometric": ["geometric", "angular", "mathematical", "symmetrical", "pattern"],
        "elegant": ["elegant", "refined", "graceful", "balanced", "understated"],
        "architectural": ["architectural", "structural", "monumental", "engineered"],
        "meditative": ["meditative", "calm", "rhythmic", "contemplative", "patient"]
    }
    
    # Score each concept based on keyword matches
    concept_scores = {}
    for concept, keywords in concept_keywords.items():
        score = sum(1 for keyword in keywords if keyword in intent_lower)
        if score > 0:
            concept_scores[concept] = score / len(keywords)
    
    # Apply emphasis if specified
    if emphasis:
        for emph in emphasis:
            if emph in concept_scores:
                concept_scores[emph] *= 1.5
    
    # Normalize scores
    if concept_scores:
        max_score = max(concept_scores.values())
        concept_scores = {k: v/max_score for k, v in concept_scores.items()}
    
    # Filter low-confidence concepts
    filtered_concepts = {k: v for k, v in concept_scores.items() if v > 0.3}
    
    if not filtered_concepts:
        # Default to minimal if no clear match
        filtered_concepts = {"minimal": 0.5}
    
    return {
        "extracted_concepts": list(filtered_concepts.keys()),
        "confidence_scores": filtered_concepts,
        "desired_complexity": desired_complexity,
        "emphasis": emphasis or []
    }


@mcp.tool()
def generate_fold_parameters(
    aesthetic_concepts: list[str],
    complexity: Literal["simple", "intermediate", "complex", "super_complex"] = "intermediate",
    pattern_type: Literal["traditional", "tessellation", "modular", "recursive", 
                         "computational", "organic_curved"] = "traditional"
) -> dict:
    """
    Generate concrete origami fold parameters from aesthetic concepts.
    
    This is the Aesthetic → Visual Parameters layer (deterministic mapping).
    
    Args:
        aesthetic_concepts: List of aesthetic concepts (precise, organic, minimal, complex, etc.)
        complexity: Complexity level for fold count and detail
        pattern_type: Type of crease pattern to use
    
    Returns:
        Concrete visual parameters for fold characteristics
    """
    # Start with default parameters
    params = {
        "edge_sharpness": 0.8,
        "fold_accuracy": 0.85,
        "corner_treatment": "defined",
        "line_consistency": "moderate",
        "surface_quality": "standard"
    }
    
    # Aggregate aesthetic concept parameters
    for concept in aesthetic_concepts:
        if concept in AESTHETIC_MAPPINGS:
            concept_params = AESTHETIC_MAPPINGS[concept]
            
            # Average numerical parameters
            if "edge_sharpness" in concept_params:
                params["edge_sharpness"] = (params["edge_sharpness"] + concept_params["edge_sharpness"]) / 2
            if "fold_accuracy" in concept_params:
                params["fold_accuracy"] = (params["fold_accuracy"] + concept_params["fold_accuracy"]) / 2
            
            # Take most recent categorical parameters
            for key in ["corner_treatment", "line_consistency", "surface_quality"]:
                if key in concept_params:
                    params[key] = concept_params[key]
    
    # Map complexity to fold count
    fold_count_map = {
        "simple": {"min": 5, "max": 15, "typical": 10},
        "intermediate": {"min": 15, "max": 40, "typical": 25},
        "complex": {"min": 40, "max": 100, "typical": 65},
        "super_complex": {"min": 100, "max": 300, "typical": 180}
    }
    
    params["fold_count"] = fold_count_map[complexity]
    params["complexity_level"] = complexity
    params["pattern_type"] = pattern_type
    params["pattern_characteristics"] = PATTERN_CHARACTERISTICS[CreasePattern(pattern_type)]
    
    return params


@mcp.tool()
def generate_lighting_parameters(
    aesthetic_concepts: list[str],
    lighting_style: Literal["backlit", "dramatic_side", "ambient_soft", "studio_clean"] = "ambient_soft",
    material_type: Literal["traditional_kami", "washi", "foil", "tissue_foil", 
                          "wet_fold_paper"] = "traditional_kami"
) -> dict:
    """
    Generate lighting parameters that reveal origami structure effectively.
    
    Args:
        aesthetic_concepts: Aesthetic concepts influencing lighting choice
        lighting_style: Primary lighting approach
        material_type: Paper material type (affects translucency, reflection)
    
    Returns:
        Lighting parameters optimized for origami characteristics
    """
    # Get base lighting interaction
    lighting_params = LIGHTING_INTERACTIONS[lighting_style].copy()
    
    # Get material properties
    material_params = MATERIAL_PROPERTIES[material_type].copy()
    
    # Adjust based on aesthetic concepts
    adjustments = {
        "precise": {"edge_treatment": "enhanced", "shadow_character": "sharp"},
        "organic": {"shadow_character": "soft_gradients", "depth_perception": "flowing"},
        "minimal": {"translucency_emphasis": "low", "shadow_character": "subtle"},
        "complex": {"layer_visibility": "enhanced", "depth_perception": "multilayered"}
    }
    
    for concept in aesthetic_concepts:
        if concept in adjustments:
            lighting_params.update(adjustments[concept])
    
    return {
        "lighting_style": lighting_style,
        "lighting_interactions": lighting_params,
        "material_type": material_type,
        "material_properties": material_params,
        "combined_effect": {
            "translucency": material_params.get("translucency", 0.3),
            "reflectivity": material_params.get("reflectivity", 0.0),
            "edge_visibility": lighting_params["edge_treatment"],
            "shadow_quality": lighting_params["shadow_character"]
        }
    }


@mcp.tool()
def generate_composition_parameters(
    aesthetic_concepts: list[str],
    symmetry: Literal["bilateral", "radial", "rotational", "asymmetric", "fractal"] = "bilateral",
    view_angle: Literal["flat_overhead", "three_quarter", "profile", "detail_macro"] = "three_quarter"
) -> dict:
    """
    Generate compositional parameters for framing origami subjects.
    
    Args:
        aesthetic_concepts: Aesthetic concepts influencing composition
        symmetry: Symmetry type of the origami form
        view_angle: Camera/viewing perspective
    
    Returns:
        Composition parameters for optimal presentation
    """
    # Base composition parameters
    comp_params = {
        "symmetry": symmetry,
        "view_angle": view_angle,
        "framing": "centered",
        "negative_space": "balanced",
        "depth_emphasis": "moderate"
    }
    
    # Adjust based on view angle
    view_adjustments = {
        "flat_overhead": {
            "depth_emphasis": "minimal",
            "pattern_emphasis": "maximum",
            "lighting_suggestion": "even_diffused"
        },
        "three_quarter": {
            "depth_emphasis": "strong",
            "dimensionality": "three_dimensional",
            "lighting_suggestion": "directional"
        },
        "profile": {
            "depth_emphasis": "silhouette",
            "edge_emphasis": "maximum",
            "lighting_suggestion": "side_or_backlit"
        },
        "detail_macro": {
            "depth_emphasis": "shallow_focus",
            "texture_emphasis": "maximum",
            "lighting_suggestion": "directional_close"
        }
    }
    
    comp_params.update(view_adjustments[view_angle])
    
    # Adjust based on aesthetic concepts
    if "minimal" in aesthetic_concepts:
        comp_params["negative_space"] = "abundant"
        comp_params["framing"] = "generous"
    
    if "complex" in aesthetic_concepts:
        comp_params["negative_space"] = "minimal"
        comp_params["detail_visibility"] = "prioritized"
    
    if symmetry == "fractal" or "recursive" in str(aesthetic_concepts):
        comp_params["scale_indication"] = "important"
        comp_params["reference_element"] = "recommended"
    
    return comp_params


@mcp.tool()
def generate_complete_specification(
    intent_description: str,
    complexity: Literal["simple", "intermediate", "complex", "super_complex"] = "intermediate",
    pattern_type: Literal["traditional", "tessellation", "modular", "recursive", 
                         "computational", "organic_curved"] = "traditional",
    lighting_style: Literal["backlit", "dramatic_side", "ambient_soft", "studio_clean"] = "ambient_soft",
    material_type: Literal["traditional_kami", "washi", "foil", "tissue_foil", 
                          "wet_fold_paper"] = "traditional_kami",
    symmetry: Literal["bilateral", "radial", "rotational", "asymmetric", "fractal"] = "bilateral",
    view_angle: Literal["flat_overhead", "three_quarter", "profile", "detail_macro"] = "three_quarter"
) -> dict:
    """
    Generate complete origami-inspired visual specification from intent.
    
    This chains all deterministic layers: Intent → Aesthetic → Parameters → Specification
    Ready for LLM creative synthesis or direct execution.
    
    Args:
        intent_description: Natural language aesthetic intent
        complexity: Fold complexity level
        pattern_type: Crease pattern style
        lighting_style: Lighting approach
        material_type: Paper material simulation
        symmetry: Compositional symmetry
        view_angle: Camera perspective
    
    Returns:
        Complete specification ready for synthesis or execution
    """
    # Layer 1: Intent Analysis
    intent_analysis = analyze_origami_intent(intent_description, complexity)
    
    aesthetic_concepts = intent_analysis["extracted_concepts"]
    
    # Layer 2: Generate Parameters
    fold_params = generate_fold_parameters(aesthetic_concepts, complexity, pattern_type)
    lighting_params = generate_lighting_parameters(aesthetic_concepts, lighting_style, material_type)
    comp_params = generate_composition_parameters(aesthetic_concepts, symmetry, view_angle)
    
    # Synthesize complete specification
    return {
        "intent_analysis": intent_analysis,
        "aesthetic_concepts": aesthetic_concepts,
        "fold_parameters": fold_params,
        "lighting_parameters": lighting_params,
        "composition_parameters": comp_params,
        "execution_ready": True,
        "synthesis_guidance": {
            "primary_aesthetic": aesthetic_concepts[0] if aesthetic_concepts else "balanced",
            "complexity_target": complexity,
            "key_characteristics": [
                f"edge_sharpness: {fold_params['edge_sharpness']:.2f}",
                f"fold_accuracy: {fold_params['fold_accuracy']:.2f}",
                f"pattern: {pattern_type}",
                f"symmetry: {symmetry}"
            ]
        }
    }


@mcp.tool()
def get_origami_vocabulary() -> dict:
    """
    Retrieve the complete origami aesthetic vocabulary and taxonomies.
    
    Useful for understanding available concepts, mappings, and relationships.
    
    Returns:
        Complete vocabulary including aesthetic mappings, fold properties, patterns
    """
    return {
        "aesthetic_concepts": list(AESTHETIC_MAPPINGS.keys()),
        "aesthetic_mappings": AESTHETIC_MAPPINGS,
        "fold_types": [e.value for e in FoldType],
        "fold_properties": FOLD_PROPERTIES,
        "crease_patterns": [e.value for e in CreasePattern],
        "pattern_characteristics": PATTERN_CHARACTERISTICS,
        "complexity_levels": [e.value for e in ComplexityLevel],
        "symmetry_types": [e.value for e in SymmetryType],
        "material_types": list(MATERIAL_PROPERTIES.keys()),
        "material_properties": MATERIAL_PROPERTIES,
        "lighting_styles": list(LIGHTING_INTERACTIONS.keys()),
        "lighting_interactions": LIGHTING_INTERACTIONS,
        # PHASE 1A additions
        "phase_1a_aesthetic_states": list(AESTHETIC_STATES.keys()),
        "phase_1a_parameter_names": ORIGAMI_PARAMETER_NAMES,
        "phase_1a_morphospace_dimensions": len(ORIGAMI_PARAMETER_NAMES)
    }


@mcp.tool()
def compare_aesthetic_profiles(
    concept_a: str,
    concept_b: str
) -> dict:
    """
    Compare two aesthetic concepts to understand their differences.
    
    Useful for understanding relationships between concepts and making choices.
    
    Args:
        concept_a: First aesthetic concept (precise, organic, minimal, etc.)
        concept_b: Second aesthetic concept
    
    Returns:
        Detailed comparison showing differences and similarities
    """
    
    if concept_a not in AESTHETIC_MAPPINGS or concept_b not in AESTHETIC_MAPPINGS:
        return {
            "error": "One or both concepts not found",
            "available_concepts": list(AESTHETIC_MAPPINGS.keys())
        }
    
    profile_a = AESTHETIC_MAPPINGS[concept_a]
    profile_b = AESTHETIC_MAPPINGS[concept_b]
    
    # Find numerical differences
    numerical_diff = {}
    for key in ["edge_sharpness", "fold_accuracy"]:
        if key in profile_a and key in profile_b:
            diff = profile_a[key] - profile_b[key]
            numerical_diff[key] = {
                f"{concept_a}": profile_a[key],
                f"{concept_b}": profile_b[key],
                "difference": diff,
                "interpretation": "sharper" if diff > 0 else "softer" if diff < 0 else "equal"
            }
    
    # Find categorical differences
    categorical_diff = {}
    for key in ["corner_treatment", "line_consistency", "surface_quality"]:
        if key in profile_a and key in profile_b:
            categorical_diff[key] = {
                f"{concept_a}": profile_a[key],
                f"{concept_b}": profile_b[key],
                "same": profile_a[key] == profile_b[key]
            }
    
    # PHASE 1A: Add morphospace distance
    if concept_a in AESTHETIC_STATES and concept_b in AESTHETIC_STATES:
        import math
        coords_a = AESTHETIC_STATES[concept_a]["coordinates"]
        coords_b = AESTHETIC_STATES[concept_b]["coordinates"]
        
        # Compute Euclidean distance in morphospace
        distance_squared = sum(
            (coords_a[p] - coords_b[p])**2 
            for p in ORIGAMI_PARAMETER_NAMES
        )
        morphospace_distance = math.sqrt(distance_squared)
        
        numerical_diff["morphospace_distance"] = {
            "distance": morphospace_distance,
            "interpretation": (
                "very_similar" if morphospace_distance < 0.3 else
                "moderately_different" if morphospace_distance < 0.6 else
                "significantly_different"
            )
        }
    
    return {
        "concepts_compared": [concept_a, concept_b],
        "numerical_differences": numerical_diff,
        "categorical_differences": categorical_diff,
        "full_profiles": {
            concept_a: profile_a,
            concept_b: profile_b
        }
    }


# =============================================================================
# CROSS-DOMAIN FUNCTORS (Preserved)
# =============================================================================

@mcp.tool()
def map_to_grid_dynamics(
    pattern_type: Literal["traditional", "tessellation", "modular", "recursive", "computational", "organic_curved"],
    complexity: Literal["simple", "intermediate", "complex", "super_complex"]
) -> dict:
    """
    Map origami pattern to grid dynamics brick parameters.
    
    Functor: Origami crease pattern → Spatial arrangement structure
    Preserves: Geometric relationships, focal points, propagation patterns
    
    Args:
        pattern_type: Origami crease pattern type
        complexity: Complexity level
    
    Returns:
        Grid dynamics parameters that preserve origami structure
    """
    
    pattern = CreasePattern(pattern_type)
    pattern_chars = PATTERN_CHARACTERISTICS[pattern]
    
    # Map origami patterns to grid dynamics
    if pattern == CreasePattern.TESSELLATION:
        grid_type = pattern_chars["grid_base"][0] if isinstance(pattern_chars["grid_base"], list) else pattern_chars["grid_base"]
        return {
            "grid_type": grid_type,
            "focal_point": "distributed",  # No single center
            "intensity_pattern": "uniform_modular",
            "propagation": "tiling_repetition",
            "structural_logic": "tessellation",
            "complexity_distribution": "evenly_distributed"
        }
    
    elif pattern == CreasePattern.TRADITIONAL:
        # Traditional patterns often have radial/center focus (like crane)
        return {
            "grid_type": "radial",
            "focal_point": "center_strong",
            "intensity_pattern": "emanating_from_center",
            "propagation": "outward_radiation",
            "structural_logic": "center_to_edge",
            "complexity_distribution": "center_concentrated"
        }
    
    elif pattern == CreasePattern.RECURSIVE:
        return {
            "grid_type": "fractal",
            "focal_point": "multiple_nested",
            "intensity_pattern": "self_similar_scales",
            "propagation": "recursive_subdivision",
            "structural_logic": "nested_repetition",
            "complexity_distribution": "scale_invariant"
        }
    
    elif pattern == CreasePattern.MODULAR:
        return {
            "grid_type": "unit_based",
            "focal_point": "connection_points",
            "intensity_pattern": "modular_assembly",
            "propagation": "interlocking_growth",
            "structural_logic": "unit_composition",
            "complexity_distribution": "per_unit_local"
        }
    
    else:  # ORGANIC_CURVED or COMPUTATIONAL
        return {
            "grid_type": "freeform",
            "focal_point": "organic_flow",
            "intensity_pattern": "natural_gradient",
            "propagation": "curved_paths",
            "structural_logic": "continuous_surface",
            "complexity_distribution": "flowing_variation"
        }


@mcp.tool()
def map_to_narrative_structure(
    aesthetic_concepts: list[str],
    complexity: Literal["simple", "intermediate", "complex", "super_complex"],
    pattern_type: Literal["traditional", "tessellation", "modular", "recursive", "computational", "organic_curved"]
) -> dict:
    """
    Map origami aesthetics to narrative story structure.
    
    Functor: Origami composition → Narrative dramatic structure
    Preserves: Tension progression, revelation patterns, structural logic
    
    Args:
        aesthetic_concepts: Origami aesthetic concepts
        complexity: Fold complexity
        pattern_type: Crease pattern type
    
    Returns:
        Narrative parameters that embody origami structure
    """
    
    # Map complexity to narrative complexity
    narrative_complexity = {
        "simple": "linear_progression",
        "intermediate": "layered_revelation",
        "complex": "interwoven_threads",
        "super_complex": "nested_narratives"
    }
    
    # Map aesthetic concepts to character/setting qualities
    character_mapping = {
        "precise": "meticulous, methodical, perfectionist",
        "organic": "intuitive, flowing, adaptive",
        "minimal": "restrained, essential, disciplined",
        "complex": "multilayered, sophisticated, intricate mind",
        "geometric": "logical, mathematical, structured thinker",
        "elegant": "refined, graceful, economical in action",
        "architectural": "builder, visionary, structural",
        "meditative": "contemplative, patient, present"
    }
    
    # Map pattern type to plot structure
    pattern_to_plot = {
        "traditional": "classical_three_act",
        "tessellation": "repeating_motifs",
        "modular": "interconnected_vignettes",
        "recursive": "nested_stories_within_stories",
        "computational": "precise_logical_progression",
        "organic_curved": "flowing_character_arc"
    }
    
    # Build narrative parameters
    narrative_params = {
        "plot_structure": pattern_to_plot[pattern_type],
        "narrative_complexity": narrative_complexity[complexity],
        "character_qualities": [character_mapping[c] for c in aesthetic_concepts if c in character_mapping],
        "pacing": "measured" if "meditative" in aesthetic_concepts else "dynamic",
        "revelation_pattern": "gradual_unfolding",
        "structural_logic": f"{pattern_type}_inspired"
    }
    
    return narrative_params

def _generate_oscillation_pattern(
    pattern_type: str,
    num_samples: int,
    phase_offset: float = 0.0,
    num_cycles: float = 1.0  # ADD THIS PARAMETER
) -> np.ndarray:
    """
    Generate normalized oscillation pattern [0, 1].
    
    Args:
        pattern_type: "sinusoidal", "triangular", or "square"
        num_samples: Number of points in pattern
        phase_offset: Starting phase (0.0 to 1.0)
        num_cycles: Number of complete cycles to generate  # NEW
    
    Returns:
        Array of values oscillating between 0 and 1
    """
    # Generate time array spanning num_cycles
    t = np.linspace(0, num_cycles, num_samples)  # CHANGE: 0 to num_cycles
    
    # Apply phase offset (in cycles)
    t = t + phase_offset
    
    if pattern_type == "sinusoidal":
        # Smooth wave: 0 → 1 → 0 per cycle
        pattern = (np.sin(2 * np.pi * t - np.pi/2) + 1) / 2
        
    elif pattern_type == "triangular":
        # Linear ramps: 0 → 1 → 0 per cycle
        t_mod = t % 1.0  # Use modulo for each cycle
        pattern = 1 - 2 * np.abs(t_mod - 0.5)
        
    elif pattern_type == "square":
        # Abrupt transitions: 0 or 1
        pattern = (np.sin(2 * np.pi * t) > 0).astype(float)
        
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    return pattern


def _interpolate_between_states(
    state_a: dict,
    state_b: dict,
    parameter_names: list,
    alpha: float
) -> dict:
    """
    Linearly interpolate between two parameter states.
    
    Args:
        state_a: Starting state coordinates
        state_b: Ending state coordinates
        parameter_names: List of parameter names
        alpha: Interpolation factor (0.0 = state_a, 1.0 = state_b)
    
    Returns:
        Interpolated state dictionary
    """
    interpolated = {}
    for param in parameter_names:
        a_val = state_a[param]
        b_val = state_b[param]
        interpolated[param] = a_val + alpha * (b_val - a_val)
    
    return interpolated


def _generate_rhythmic_origami_sequence_impl(
    state_a_id: str,
    state_b_id: str,
    oscillation_pattern: str,
    num_cycles: int,
    steps_per_cycle: int,
    phase_offset: float
) -> dict:
    """
    Core implementation of rhythmic origami sequence generation.
    
    REUSES existing trajectory computation infrastructure.
    Generates oscillating transitions between two aesthetic states.
    """
    if not DYNAMICS_AVAILABLE:
        return {
            "error": "aesthetic-dynamics-core not installed",
            "message": "Install with: pip install aesthetic-dynamics-core",
            "fallback": "Use compute_trajectory_between_aesthetic_states for single transitions"
        }
    
    # Validate state IDs
    if state_a_id not in AESTHETIC_STATES:
        return {
            "error": f"Unknown start state: {state_a_id}",
            "available_states": list(AESTHETIC_STATES.keys())
        }
    
    if state_b_id not in AESTHETIC_STATES:
        return {
            "error": f"Unknown end state: {state_b_id}",
            "available_states": list(AESTHETIC_STATES.keys())
        }
    
    # Get state coordinates
    state_a = AESTHETIC_STATES[state_a_id]["coordinates"]
    state_b = AESTHETIC_STATES[state_b_id]["coordinates"]
    
    # Generate oscillation pattern
    total_samples = num_cycles * steps_per_cycle
    oscillation = _generate_oscillation_pattern(
        oscillation_pattern,
        total_samples,
        phase_offset,
        num_cycles
    )
    
    # Generate sequence by interpolating between states
    sequence = []
    for alpha in oscillation:
        state = _interpolate_between_states(
            state_a,
            state_b,
            ORIGAMI_PARAMETER_NAMES,
            alpha
        )
        sequence.append(state)
    
    # Identify key phase points (peaks, troughs, crossings)
    phase_points = []
    
    # Add start point
    phase_points.append({
        "step": 0,
        "type": "start",
        "state": state_a_id,
        "alpha": 0.0
    })
    
    # Find peaks (maximum toward state_b)
    for i in range(1, len(oscillation) - 1):
        if oscillation[i] > oscillation[i-1] and oscillation[i] > oscillation[i+1]:
            if oscillation[i] > 0.8:  # Relaxed threshold (was 0.9)
                phase_points.append({
                    "step": i,
                    "type": "peak",
                    "state": state_b_id,
                    "alpha": float(oscillation[i])
                })

    # Find troughs (minimum toward state_a)
    for i in range(1, len(oscillation) - 1):
        if oscillation[i] < oscillation[i-1] and oscillation[i] < oscillation[i+1]:
            if oscillation[i] < 0.2:  # Relaxed threshold (was 0.1)
                phase_points.append({
                    "step": i,
                    "type": "trough",
                    "state": state_a_id,
                    "alpha": float(oscillation[i])
                })
                
    # Add end point
    phase_points.append({
        "step": len(oscillation) - 1,
        "type": "end",
        "state": state_a_id if oscillation[-1] < 0.5 else state_b_id,
        "alpha": float(oscillation[-1])
    })
    
    # Sort by step
    phase_points.sort(key=lambda p: p["step"])
    
    # Describe aesthetic flow
    if oscillation_pattern == "sinusoidal":
        flow_description = f"Smooth, continuous oscillation between {state_a_id} and {state_b_id}. Natural rhythm like breathing or day/night cycles."
    elif oscillation_pattern == "triangular":
        flow_description = f"Linear transitions between {state_a_id} and {state_b_id}. Mechanical rhythm with constant rate of change."
    elif oscillation_pattern == "square":
        flow_description = f"Abrupt alternation between {state_a_id} and {state_b_id}. Punctuated rhythm with sharp phase transitions."
    else:
        flow_description = f"Oscillating pattern between {state_a_id} and {state_b_id}."
    
    return {
        "sequence": sequence,
        "pattern_type": oscillation_pattern,
        "num_cycles": num_cycles,
        "steps_per_cycle": steps_per_cycle,
        "total_steps": len(sequence),
        "state_a": {
            "id": state_a_id,
            "name": AESTHETIC_STATES[state_a_id]["name"],
            "coordinates": state_a
        },
        "state_b": {
            "id": state_b_id,
            "name": AESTHETIC_STATES[state_b_id]["name"],
            "coordinates": state_b
        },
        "phase_points": phase_points,
        "aesthetic_flow": flow_description,
        "frequency": num_cycles / total_samples,
        "oscillation_profile": oscillation.tolist(),
        "parameter_names": ORIGAMI_PARAMETER_NAMES,
        "dynamics_info": {
            "method": "Rhythmic interpolation with pattern functions",
            "cost": "0 tokens (pure Layer 2)",
            "phase_offset": phase_offset
        }
    }


# =============================================================================
# PHASE 2.6: RHYTHMIC COMPOSITION PRESETS
# =============================================================================

ORIGAMI_RHYTHMIC_PRESETS = {
    "daily_fold_cycle": {
        "state_a": "precise",
        "state_b": "organic",
        "pattern": "sinusoidal",
        "num_cycles": 1,
        "steps_per_cycle": 24,  # 24 hours
        "description": "Day/night aesthetic cycle: morning precision evolves to evening organic flow",
        "use_case": "Visualize temporal progression from sharp geometric forms to soft natural curves"
    },
    
    "seasonal_complexity": {
        "state_a": "minimal",
        "state_b": "complex",
        "pattern": "sinusoidal",
        "num_cycles": 4,
        "steps_per_cycle": 30,  # ~30 days per season
        "description": "Yearly cycle through origami complexity: winter simplicity to summer abundance",
        "use_case": "Represent seasonal variation in design complexity"
    },
    
    "meditative_breathing": {
        "state_a": "geometric",
        "state_b": "meditative",
        "pattern": "triangular",
        "num_cycles": 10,
        "steps_per_cycle": 8,  # Breath cycle: 4 in, 4 out
        "description": "Breath-like rhythm: tension (geometric) to release (meditative)",
        "use_case": "Create contemplative visual rhythm mimicking breathing pattern"
    },
    
    "architectural_pulse": {
        "state_a": "architectural",
        "state_b": "elegant",
        "pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 20,
        "description": "Structural rigidity oscillating with refined elegance",
        "use_case": "Balance engineering precision with aesthetic refinement"
    },
    
    "precision_flow_toggle": {
        "state_a": "precise",
        "state_b": "organic",
        "pattern": "square",
        "num_cycles": 5,
        "steps_per_cycle": 10,
        "description": "Sharp toggle between precision and organic flow",
        "use_case": "Emphasize contrast through abrupt aesthetic shifts"
    }
}


def _apply_origami_rhythmic_preset_impl(
    preset_name: str,
    override_params: Optional[dict]
) -> dict:
    """
    Core implementation of preset application.
    """
    if preset_name not in ORIGAMI_RHYTHMIC_PRESETS:
        return {
            "error": f"Unknown preset: {preset_name}",
            "available_presets": list(ORIGAMI_RHYTHMIC_PRESETS.keys())
        }
    
    preset = ORIGAMI_RHYTHMIC_PRESETS[preset_name]
    
    # Start with preset defaults
    params = {
        "state_a_id": preset["state_a"],
        "state_b_id": preset["state_b"],
        "oscillation_pattern": preset["pattern"],
        "num_cycles": preset["num_cycles"],
        "steps_per_cycle": preset["steps_per_cycle"],
        "phase_offset": 0.0
    }
    
    # Apply overrides if provided
    if override_params:
        params.update(override_params)
    
    # Generate sequence using core implementation
    result = _generate_rhythmic_origami_sequence_impl(**params)
    
    # Add preset metadata
    if "error" not in result:
        result["preset_info"] = {
            "name": preset_name,
            "description": preset["description"],
            "use_case": preset["use_case"],
            "overrides_applied": override_params is not None
        }
    
    return result


# =============================================================================
# PHASE 2.6: MCP TOOL WRAPPERS
# =============================================================================

@mcp.tool()
def generate_rhythmic_origami_sequence(
    state_a_id: str,
    state_b_id: str,
    oscillation_pattern: Literal["sinusoidal", "triangular", "square"] = "sinusoidal",
    num_cycles: int = 2,
    steps_per_cycle: int = 20,
    phase_offset: float = 0.0
) -> dict:
    """
    Generate rhythmic oscillation between two origami aesthetic states.
    
    PHASE 2.6 ENHANCEMENT: Adds temporal/rhythmic composition to origami aesthetics.
    Creates periodic transitions that cycle between aesthetic configurations.
    
    Building on Phase 1A trajectory computation, this tool enables:
    - Day/night aesthetic cycles (precise ↔ organic)
    - Seasonal complexity variation (minimal ↔ complex)
    - Breath-like rhythms (geometric ↔ meditative)
    - Structural pulses (architectural ↔ elegant)
    
    Pure Layer 2 deterministic operation - 0 tokens.
    
    Args:
        state_a_id: Starting aesthetic state (see AESTHETIC_STATES)
        state_b_id: Alternating aesthetic state
        oscillation_pattern: Wave shape
            - "sinusoidal": Smooth, continuous (natural rhythms)
            - "triangular": Linear ramps (mechanical rhythms)
            - "square": Abrupt changes (punctuated rhythms)
        num_cycles: Number of complete A→B→A cycles
        steps_per_cycle: Samples per cycle (higher = smoother)
        phase_offset: Starting phase (0.0 = start at A, 0.5 = start at B)
    
    Returns:
        sequence: List of origami parameter states
        pattern_type: Echo of oscillation pattern
        num_cycles: Number of cycles completed
        phase_points: Key transition moments (peaks, troughs)
        aesthetic_flow: Human-readable flow description
        frequency: Cycles per total duration
        oscillation_profile: Raw oscillation values [0, 1]
    
    Cost: 0 tokens (pure Layer 2 computation)
    
    Example:
        >>> generate_rhythmic_origami_sequence(
        ...     "precise",
        ...     "organic",
        ...     oscillation_pattern="sinusoidal",
        ...     num_cycles=2,
        ...     steps_per_cycle=20
        ... )
        {
            "sequence": [
                {"edge_sharpness": 0.95, "fold_accuracy": 0.98, ...},
                {"edge_sharpness": 0.92, "fold_accuracy": 0.96, ...},
                ...
            ],
            "pattern_type": "sinusoidal",
            "num_cycles": 2,
            "total_steps": 40,
            "phase_points": [
                {"step": 0, "type": "start", "state": "precise"},
                {"step": 10, "type": "peak", "state": "organic"},
                {"step": 20, "type": "trough", "state": "precise"},
                ...
            ],
            "aesthetic_flow": "Smooth, continuous oscillation..."
        }
    """
    return _generate_rhythmic_origami_sequence_impl(
        state_a_id=state_a_id,
        state_b_id=state_b_id,
        oscillation_pattern=oscillation_pattern,
        num_cycles=num_cycles,
        steps_per_cycle=steps_per_cycle,
        phase_offset=phase_offset
    )


@mcp.tool()
def apply_origami_rhythmic_preset(
    preset_name: Literal[
        "daily_fold_cycle",
        "seasonal_complexity",
        "meditative_breathing",
        "architectural_pulse",
        "precision_flow_toggle"
    ],
    override_params: Optional[dict] = None
) -> dict:
    """
    Apply a curated rhythmic origami pattern preset.
    
    PHASE 2.6 CONVENIENCE TOOL: Pre-configured rhythmic compositions
    for common origami aesthetic use cases.
    
    Available Presets:
    
    1. **daily_fold_cycle**
       - precise ↔ organic over 24 steps
       - Morning clarity → evening organic flow
       - Sinusoidal pattern
    
    2. **seasonal_complexity**
       - minimal ↔ complex over 4 cycles (30 steps each)
       - Winter simplicity → summer abundance
       - Sinusoidal pattern
    
    3. **meditative_breathing**
       - geometric ↔ meditative over 10 cycles (8 steps each)
       - Tension (sharp) → release (flowing)
       - Triangular pattern (linear ramps)
    
    4. **architectural_pulse**
       - architectural ↔ elegant over 3 cycles (20 steps each)
       - Engineering rigor → aesthetic refinement
       - Sinusoidal pattern
    
    5. **precision_flow_toggle**
       - precise ↔ organic over 5 cycles (10 steps each)
       - Sharp aesthetic contrast
       - Square wave (abrupt transitions)
    
    Args:
        preset_name: Name of preset configuration
        override_params: Optional dict to override preset defaults
            Keys: state_a_id, state_b_id, oscillation_pattern,
                  num_cycles, steps_per_cycle, phase_offset
    
    Returns:
        Same as generate_rhythmic_origami_sequence, plus:
        preset_info: Metadata about applied preset
    
    Cost: 0 tokens (pure Layer 2)
    
    Example:
        >>> apply_origami_rhythmic_preset("daily_fold_cycle")
        # Returns 24-step cycle: precise → organic → precise
        
        >>> apply_origami_rhythmic_preset(
        ...     "seasonal_complexity",
        ...     override_params={"num_cycles": 2}  # Just 2 seasons
        ... )
        # Returns modified preset with 2 cycles instead of 4
    """
    return _apply_origami_rhythmic_preset_impl(preset_name, override_params)


@mcp.tool()
def list_origami_rhythmic_presets() -> dict:
    """
    List all available rhythmic origami presets with descriptions.
    
    Returns detailed information about each preset including:
    - State transitions
    - Pattern type
    - Number of cycles
    - Use cases
    
    Cost: 0 tokens (pure lookup)
    """
    presets_info = {}
    
    for preset_name, preset_data in ORIGAMI_RHYTHMIC_PRESETS.items():
        presets_info[preset_name] = {
            "states": f"{preset_data['state_a']} ↔ {preset_data['state_b']}",
            "pattern": preset_data["pattern"],
            "cycles": preset_data["num_cycles"],
            "steps_per_cycle": preset_data["steps_per_cycle"],
            "total_steps": preset_data["num_cycles"] * preset_data["steps_per_cycle"],
            "description": preset_data["description"],
            "use_case": preset_data["use_case"]
        }
    
    return {
        "available_presets": list(ORIGAMI_RHYTHMIC_PRESETS.keys()),
        "total_presets": len(ORIGAMI_RHYTHMIC_PRESETS),
        "presets": presets_info,
        "usage": "Call apply_origami_rhythmic_preset(preset_name) to use"
    }


# =============================================================================
# PHASE 2.7: ATTRACTOR VISUALIZATION PROMPT GENERATION
# =============================================================================
# Visual vocabulary types for translating origami parameter coordinates into
# image generation keywords. Follows DOMAIN_EXPANSION_GUIDE.md pattern.

ORIGAMI_VISUAL_VOCABULARY = {
    "traditional_crane": {
        "coords": {
            "edge_sharpness": 0.90,
            "fold_accuracy": 0.95,
            "complexity_level": 0.35,
            "organic_quality": 0.20,
            "dimensional_depth": 0.50
        },
        "keywords": [
            "traditional origami crane",
            "crisp paper folds",
            "clean mountain and valley creases",
            "bilateral symmetry",
            "minimal fold count",
            "elegant simplicity",
            "Japanese paper craft"
        ]
    },
    "modular_kusudama": {
        "coords": {
            "edge_sharpness": 0.88,
            "fold_accuracy": 0.92,
            "complexity_level": 0.80,
            "organic_quality": 0.25,
            "dimensional_depth": 0.85
        },
        "keywords": [
            "modular origami assembly",
            "interlocking geometric units",
            "radial symmetry kusudama",
            "complex multi-unit construction",
            "precise angular connections",
            "three-dimensional paper polyhedron",
            "mathematical paper sculpture"
        ]
    },
    "wet_fold_organic": {
        "coords": {
            "edge_sharpness": 0.35,
            "fold_accuracy": 0.60,
            "complexity_level": 0.55,
            "organic_quality": 0.95,
            "dimensional_depth": 0.80
        },
        "keywords": [
            "wet-folded sculptural origami",
            "smooth curved paper surfaces",
            "organic flowing creases",
            "soft rounded contours",
            "naturalistic paper form",
            "gentle gradual folds",
            "three-dimensional paper sculpture"
        ]
    },
    "tessellation_surface": {
        "coords": {
            "edge_sharpness": 0.92,
            "fold_accuracy": 0.96,
            "complexity_level": 0.75,
            "organic_quality": 0.10,
            "dimensional_depth": 0.25
        },
        "keywords": [
            "origami tessellation pattern",
            "repeating geometric fold grid",
            "precise pleated paper surface",
            "rhythmic crease pattern",
            "mathematical tiling folds",
            "flat geometric paper relief",
            "hexagonal or triangular paper grid"
        ]
    },
    "architectural_sculptural": {
        "coords": {
            "edge_sharpness": 0.93,
            "fold_accuracy": 0.97,
            "complexity_level": 0.70,
            "organic_quality": 0.15,
            "dimensional_depth": 0.95
        },
        "keywords": [
            "architectural origami structure",
            "engineered paper construction",
            "monumental fold geometry",
            "structural paper engineering",
            "sharp planar intersections",
            "dramatic angular paper form",
            "large-scale folded architecture"
        ]
    }
}


def _extract_origami_visual_vocabulary(
    state: dict,
    strength: float = 1.0
) -> dict:
    """
    Extract visual vocabulary keywords by nearest-neighbor matching
    in origami parameter space.
    
    Args:
        state: Dict of origami parameter values
        strength: Weight for this domain in multi-domain composition [0, 1]
    
    Returns:
        Dict with nearest_type, distance, keywords, strength
    """
    # Build state vector from available parameters
    param_names = ORIGAMI_PARAMETER_NAMES
    state_vec = np.array([state.get(p, 0.5) for p in param_names])
    
    min_distance = float('inf')
    nearest_type = None
    nearest_keywords = []
    
    for type_name, type_data in ORIGAMI_VISUAL_VOCABULARY.items():
        type_vec = np.array([type_data["coords"].get(p, 0.5) for p in param_names])
        distance = float(np.linalg.norm(state_vec - type_vec))
        
        if distance < min_distance:
            min_distance = distance
            nearest_type = type_name
            nearest_keywords = type_data["keywords"]
    
    return {
        "domain": "origami",
        "nearest_type": nearest_type,
        "distance": round(min_distance, 4),
        "keywords": nearest_keywords,
        "strength": strength,
        "weighted_keywords": nearest_keywords[:max(1, int(len(nearest_keywords) * strength))]
    }


def _generate_origami_attractor_prompt(
    attractor_state: dict,
    mode: str = "composite",
    strength: float = 1.0,
    style_modifiers: list = None
) -> dict:
    """
    Generate image generation prompt from origami attractor coordinates.
    
    Translates parameter-space coordinates into visual vocabulary suitable
    for text-to-image models. Follows DOMAIN_EXPANSION_GUIDE.md pattern.
    
    Args:
        attractor_state: Dict of origami parameter values at attractor point
        mode: "composite" (blended prompt) or "split_view" (standalone prompt)
        strength: Domain weight in multi-domain composition
        style_modifiers: Optional list of additional style keywords
    
    Returns:
        Dict with prompt text, keywords, metadata
    """
    # Extract vocabulary via nearest-neighbor
    vocab = _extract_origami_visual_vocabulary(attractor_state, strength)
    
    # Determine paper material from organic_quality
    organic = attractor_state.get("organic_quality", 0.5)
    if organic > 0.7:
        material = "thick handmade washi paper"
        technique = "wet-folding technique"
    elif organic < 0.3:
        material = "crisp geometric card stock"
        technique = "precise machine-scored creases"
    else:
        material = "traditional kami paper"
        technique = "hand-folded"
    
    # Determine lighting from dimensional_depth
    depth = attractor_state.get("dimensional_depth", 0.5)
    if depth > 0.7:
        lighting = "dramatic directional side lighting revealing fold depth"
    elif depth < 0.3:
        lighting = "even diffused overhead lighting emphasizing pattern"
    else:
        lighting = "soft studio lighting with gentle shadows"
    
    # Determine complexity description
    complexity = attractor_state.get("complexity_level", 0.5)
    if complexity > 0.75:
        complexity_desc = "highly intricate multi-layered"
    elif complexity < 0.3:
        complexity_desc = "minimal essential"
    else:
        complexity_desc = "moderately detailed"
    
    # Determine edge quality
    sharpness = attractor_state.get("edge_sharpness", 0.5)
    if sharpness > 0.85:
        edge_desc = "razor-sharp defined edges"
    elif sharpness < 0.5:
        edge_desc = "soft rounded edges"
    else:
        edge_desc = "clean well-defined edges"
    
    # Build prompt components
    keywords = vocab["weighted_keywords"]
    
    # Assemble prompt based on mode
    if mode == "composite":
        # For multi-domain blending: contribute keywords weighted by strength
        prompt_parts = keywords.copy()
        prompt_parts.extend([material, technique, lighting])
        if style_modifiers:
            prompt_parts.extend(style_modifiers)
        
        prompt = ", ".join(prompt_parts)
        
    elif mode == "split_view":
        # Standalone origami prompt with full description
        prompt = (
            f"{complexity_desc} origami form, {edge_desc}, "
            f"{material}, {technique}, "
            f"{', '.join(keywords[:4])}, "
            f"{lighting}"
        )
        if style_modifiers:
            prompt += ", " + ", ".join(style_modifiers)
    
    elif mode == "sequence":
        # For temporal sequences: compact per-frame prompt
        prompt = (
            f"{keywords[0]}, {edge_desc}, {material}, {lighting}"
        )
    
    else:
        prompt = ", ".join(keywords)
    
    return {
        "domain": "origami",
        "prompt": prompt,
        "mode": mode,
        "nearest_visual_type": vocab["nearest_type"],
        "type_distance": vocab["distance"],
        "keywords_used": keywords,
        "material": material,
        "technique": technique,
        "lighting": lighting,
        "strength": strength,
        "parameters": {
            "edge_sharpness": attractor_state.get("edge_sharpness", 0.5),
            "fold_accuracy": attractor_state.get("fold_accuracy", 0.5),
            "complexity_level": attractor_state.get("complexity_level", 0.5),
            "organic_quality": attractor_state.get("organic_quality", 0.5),
            "dimensional_depth": attractor_state.get("dimensional_depth", 0.5)
        },
        "cost": "0 tokens (pure Layer 2 lookup)"
    }


# =============================================================================
# PHASE 2.7: PRESET ATTRACTOR STATES
# =============================================================================
# These represent discovered attractor configurations for origami in
# multi-domain compositions. Coordinates from Tier 4D analysis.

ORIGAMI_PRESET_ATTRACTORS = {
    "period_30_universal_sync": {
        "name": "Period 30 - Universal Sync (Origami Component)",
        "description": "Origami contribution to the dominant 3-domain LCM synchronization",
        "basin_size": 0.116,
        "state": {
            "edge_sharpness": 0.86,
            "fold_accuracy": 0.92,
            "complexity_level": 0.58,
            "organic_quality": 0.38,
            "dimensional_depth": 0.62
        }
    },
    "period_19_gap_flow": {
        "name": "Period 19 - Gap Flow (Origami Component)",
        "description": "Resilient gap-filler attractor; balanced between geometric and organic",
        "basin_size": 0.074,
        "state": {
            "edge_sharpness": 0.78,
            "fold_accuracy": 0.88,
            "complexity_level": 0.50,
            "organic_quality": 0.52,
            "dimensional_depth": 0.55
        }
    },
    "period_60_harmonic_hub": {
        "name": "Period 60 - Harmonic Hub (Origami Component)",
        "description": "Complex multi-domain synchronization hub",
        "basin_size": 0.040,
        "state": {
            "edge_sharpness": 0.90,
            "fold_accuracy": 0.94,
            "complexity_level": 0.68,
            "organic_quality": 0.28,
            "dimensional_depth": 0.72
        }
    },
    "origami_meditative_loop": {
        "name": "Meditative Breathing Loop",
        "description": "Self-contained origami attractor: geometric tension ↔ meditative release",
        "basin_size": None,
        "state": {
            "edge_sharpness": 0.85,
            "fold_accuracy": 0.93,
            "complexity_level": 0.55,
            "organic_quality": 0.35,
            "dimensional_depth": 0.52
        }
    },
    "origami_complexity_crest": {
        "name": "Complexity Crest",
        "description": "Peak complexity state from seasonal preset; intricate sculptural form",
        "basin_size": None,
        "state": {
            "edge_sharpness": 0.82,
            "fold_accuracy": 0.92,
            "complexity_level": 0.88,
            "organic_quality": 0.42,
            "dimensional_depth": 0.82
        }
    }
}


# =============================================================================
# PHASE 2.7: DOMAIN REGISTRY INTEGRATION
# =============================================================================

def get_origami_domain_registry_config() -> dict:
    """
    Return domain registration data for the emergent attractor discovery system.
    
    Follows ADDING_NEW_DOMAINS.md plugin architecture pattern.
    Call this from domain_registry.py to register origami-aesthetics.
    
    Returns:
        Dict with coordinates, presets, vocabulary, parameter_names
    """
    # Extract state coordinates in registry format
    coordinates = {}
    for state_id, state_data in AESTHETIC_STATES.items():
        coordinates[state_id] = state_data["coordinates"]
    
    # Extract presets in registry format
    presets = {}
    for preset_name, preset_data in ORIGAMI_RHYTHMIC_PRESETS.items():
        presets[preset_name] = {
            "period": preset_data["steps_per_cycle"],
            "state_a_id": preset_data["state_a"],
            "state_b_id": preset_data["state_b"],
            "pattern": preset_data["pattern"],
            "description": preset_data["description"]
        }
    
    # Extract vocabulary categories
    vocabulary = {
        "form": [
            "crisp paper folds", "mountain and valley creases",
            "interlocking geometric units", "curved paper surfaces",
            "pleated surface relief", "angular paper planes"
        ],
        "material": [
            "traditional kami paper", "handmade washi",
            "metallic foil paper", "wet-folded stock",
            "tissue-foil composite"
        ],
        "technique": [
            "precise hand-folding", "wet-folding technique",
            "modular assembly", "tessellation pleating",
            "reverse fold sequences", "sink fold compression"
        ],
        "aesthetic": [
            "geometric precision", "organic sculptural form",
            "minimal essential beauty", "complex layered intricacy",
            "meditative repetition", "architectural monumentality"
        ]
    }
    
    return {
        "domain_id": "origami",
        "display_name": "Origami Aesthetics",
        "description": "Origami-inspired aesthetics with fold logic and dimensional transformation",
        "mcp_server": "origami-aesthetics",
        "parameter_names": ORIGAMI_PARAMETER_NAMES,
        "state_coordinates": coordinates,
        "presets": presets,
        "periods": sorted(set(p["steps_per_cycle"] for p in ORIGAMI_RHYTHMIC_PRESETS.values())),
        "vocabulary": vocabulary,
        "visual_types": list(ORIGAMI_VISUAL_VOCABULARY.keys()),
        "visual_vocabulary": ORIGAMI_VISUAL_VOCABULARY
    }


# =============================================================================
# PHASE 2.7: MCP TOOL WRAPPERS
# =============================================================================

@mcp.tool()
def generate_attractor_visualization_prompt(
    attractor_name: str = "",
    custom_state: Optional[dict] = None,
    mode: Literal["composite", "split_view", "sequence"] = "composite",
    strength: float = 1.0,
    style_modifiers: Optional[list] = None
) -> dict:
    """
    Generate image generation prompt from origami attractor coordinates.
    
    PHASE 2.7 TOOL: Translates mathematical attractor positions in origami
    parameter space into visual vocabulary suitable for text-to-image models.
    
    Supports three generation modes:
    - **composite**: Keywords for blending into multi-domain prompts
    - **split_view**: Standalone origami prompt with full description
    - **sequence**: Compact per-frame prompt for temporal animations
    
    Uses nearest-neighbor matching against 5 visual vocabulary types:
    - traditional_crane: Classic simple forms
    - modular_kusudama: Multi-unit geometric assemblies
    - wet_fold_organic: Curved sculptural forms
    - tessellation_surface: Repeating geometric patterns
    - architectural_sculptural: Structural paper engineering
    
    Args:
        attractor_name: Name of preset attractor (see list_origami_preset_attractors)
                        Leave empty if using custom_state
        custom_state: Custom parameter dict with origami coordinates
                      Keys: edge_sharpness, fold_accuracy, complexity_level,
                            organic_quality, dimensional_depth
        mode: Prompt generation mode
        strength: Domain weight for multi-domain composition [0.0, 1.0]
        style_modifiers: Optional additional style keywords to append
    
    Returns:
        prompt: Generated image prompt text
        mode: Echo of generation mode
        nearest_visual_type: Matched vocabulary type
        keywords_used: Visual keywords incorporated
        material: Inferred paper material
        technique: Inferred folding technique
        lighting: Inferred lighting setup
    
    Cost: 0 tokens (pure Layer 2 lookup + interpolation)
    
    Example:
        >>> generate_attractor_visualization_prompt(
        ...     attractor_name="period_30_universal_sync",
        ...     mode="split_view"
        ... )
        {
            "prompt": "moderately detailed origami form, clean well-defined edges, ...",
            "nearest_visual_type": "traditional_crane",
            ...
        }
        
        >>> generate_attractor_visualization_prompt(
        ...     custom_state={
        ...         "edge_sharpness": 0.35,
        ...         "fold_accuracy": 0.60,
        ...         "complexity_level": 0.55,
        ...         "organic_quality": 0.95,
        ...         "dimensional_depth": 0.80
        ...     },
        ...     mode="composite",
        ...     strength=0.6
        ... )
    """
    # Resolve state from attractor name or custom
    if custom_state:
        state = custom_state
    elif attractor_name and attractor_name in ORIGAMI_PRESET_ATTRACTORS:
        state = ORIGAMI_PRESET_ATTRACTORS[attractor_name]["state"]
    elif attractor_name:
        return {
            "error": f"Unknown attractor: {attractor_name}",
            "available_attractors": list(ORIGAMI_PRESET_ATTRACTORS.keys()),
            "hint": "Use custom_state for arbitrary coordinates"
        }
    else:
        return {
            "error": "Provide either attractor_name or custom_state",
            "available_attractors": list(ORIGAMI_PRESET_ATTRACTORS.keys())
        }
    
    result = _generate_origami_attractor_prompt(
        attractor_state=state,
        mode=mode,
        strength=strength,
        style_modifiers=style_modifiers
    )
    
    # Add attractor metadata if using preset
    if attractor_name and attractor_name in ORIGAMI_PRESET_ATTRACTORS:
        attractor_info = ORIGAMI_PRESET_ATTRACTORS[attractor_name]
        result["attractor_info"] = {
            "name": attractor_info["name"],
            "description": attractor_info["description"],
            "basin_size": attractor_info["basin_size"]
        }
    
    return result


@mcp.tool()
def list_origami_preset_attractors() -> dict:
    """
    List all preset attractor states available for visualization.
    
    PHASE 2.7 TOOL: Shows discovered attractor configurations with their
    origami parameter coordinates and metadata.
    
    Returns:
        Catalog of preset attractors with coordinates and descriptions
    
    Cost: 0 tokens (pure lookup)
    """
    attractors_info = {}
    for name, data in ORIGAMI_PRESET_ATTRACTORS.items():
        attractors_info[name] = {
            "display_name": data["name"],
            "description": data["description"],
            "basin_size": data["basin_size"],
            "coordinates": data["state"],
            "nearest_visual_type": _extract_origami_visual_vocabulary(
                data["state"]
            )["nearest_type"]
        }
    
    return {
        "total_attractors": len(ORIGAMI_PRESET_ATTRACTORS),
        "attractors": attractors_info,
        "visual_types": list(ORIGAMI_VISUAL_VOCABULARY.keys()),
        "parameter_names": ORIGAMI_PARAMETER_NAMES,
        "usage": "Call generate_attractor_visualization_prompt(attractor_name) to generate prompts"
    }


@mcp.tool()
def list_origami_visual_types() -> dict:
    """
    List all visual vocabulary types used for prompt generation.
    
    PHASE 2.7 TOOL: Shows the 5 canonical origami visual types that
    form the basis for nearest-neighbor keyword matching.
    
    Each type has parameter coordinates and 7 image generation keywords.
    
    Returns:
        Complete visual vocabulary catalog
    
    Cost: 0 tokens (pure lookup)
    """
    types_info = {}
    for type_name, type_data in ORIGAMI_VISUAL_VOCABULARY.items():
        types_info[type_name] = {
            "coordinates": type_data["coords"],
            "keywords": type_data["keywords"],
            "keyword_count": len(type_data["keywords"])
        }
    
    return {
        "total_types": len(ORIGAMI_VISUAL_VOCABULARY),
        "types": types_info,
        "parameter_names": ORIGAMI_PARAMETER_NAMES,
        "matching_method": "Euclidean nearest-neighbor in 5D parameter space",
        "usage": "Types are matched automatically by generate_attractor_visualization_prompt"
    }


@mcp.tool()
def generate_attractor_sequence_prompts(
    preset_name: str,
    num_keyframes: int = 5,
    mode: Literal["composite", "split_view", "sequence"] = "sequence",
    style_modifiers: Optional[list] = None
) -> dict:
    """
    Generate a sequence of image prompts from a rhythmic preset's trajectory.
    
    PHASE 2.7 TOOL: Combines Phase 2.6 rhythmic presets with Phase 2.7
    attractor visualization to produce temporal prompt sequences.
    
    Extracts evenly-spaced keyframes from the oscillation trajectory and
    generates an image prompt for each keyframe position.
    
    Use cases:
    - Animated origami aesthetic transitions
    - Storyboard generation for fold sequences
    - Multi-frame compositional studies
    
    Args:
        preset_name: Phase 2.6 preset to sample from
        num_keyframes: Number of keyframes to extract (3-20)
        mode: Prompt generation mode per keyframe
        style_modifiers: Optional style keywords applied to all frames
    
    Returns:
        keyframes: List of prompts with step index and coordinates
        preset_info: Metadata about the source preset
        trajectory_summary: Min/max parameter ranges across sequence
    
    Cost: 0 tokens (pure Layer 2)
    
    Example:
        >>> generate_attractor_sequence_prompts(
        ...     "daily_fold_cycle",
        ...     num_keyframes=6,
        ...     mode="sequence"
        ... )
    """
    if preset_name not in ORIGAMI_RHYTHMIC_PRESETS:
        return {
            "error": f"Unknown preset: {preset_name}",
            "available_presets": list(ORIGAMI_RHYTHMIC_PRESETS.keys())
        }
    
    # Clamp keyframes
    num_keyframes = max(3, min(20, num_keyframes))
    
    # Generate the full rhythmic sequence
    preset = ORIGAMI_RHYTHMIC_PRESETS[preset_name]
    sequence_result = _generate_rhythmic_origami_sequence_impl(
        state_a_id=preset["state_a"],
        state_b_id=preset["state_b"],
        oscillation_pattern=preset["pattern"],
        num_cycles=preset["num_cycles"],
        steps_per_cycle=preset["steps_per_cycle"],
        phase_offset=0.0
    )
    
    if "error" in sequence_result:
        return sequence_result
    
    sequence = sequence_result["sequence"]
    total_steps = len(sequence)
    
    # Extract evenly-spaced keyframes
    if num_keyframes >= total_steps:
        keyframe_indices = list(range(total_steps))
    else:
        keyframe_indices = [
            int(i * (total_steps - 1) / (num_keyframes - 1))
            for i in range(num_keyframes)
        ]
    
    # Generate prompt for each keyframe
    keyframes = []
    for idx in keyframe_indices:
        state = sequence[idx]
        prompt_result = _generate_origami_attractor_prompt(
            attractor_state=state,
            mode=mode,
            strength=1.0,
            style_modifiers=style_modifiers
        )
        
        keyframes.append({
            "step_index": idx,
            "step_fraction": round(idx / max(1, total_steps - 1), 3),
            "coordinates": state,
            "prompt": prompt_result["prompt"],
            "nearest_visual_type": prompt_result["nearest_visual_type"],
            "material": prompt_result["material"],
            "lighting": prompt_result["lighting"]
        })
    
    # Compute trajectory parameter ranges
    param_ranges = {}
    for param in ORIGAMI_PARAMETER_NAMES:
        values = [s.get(param, 0.5) for s in sequence]
        param_ranges[param] = {
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "range": round(max(values) - min(values), 3)
        }
    
    return {
        "preset_name": preset_name,
        "preset_description": preset["description"],
        "num_keyframes": len(keyframes),
        "total_trajectory_steps": total_steps,
        "keyframes": keyframes,
        "trajectory_summary": {
            "parameter_ranges": param_ranges,
            "states": f"{preset['state_a']} ↔ {preset['state_b']}",
            "pattern": preset["pattern"],
            "cycles": preset["num_cycles"]
        },
        "cost": "0 tokens (pure Layer 2)"
    }


# =============================================================================
# STRATEGY ANALYSIS (Preserved)
# =============================================================================

def analyze_strategy_document(strategy_text: str) -> dict:
    """
    Layer 2 deterministic pattern matching against origami aesthetic taxonomy.
    
    Detects structural patterns through origami vocabulary:
    - Structural clarity (fold definition quality)
    - Complexity management (fold count appropriate to form)
    - Precision vs flexibility (edge treatment)
    - Layering logic (information depth)
    - Pattern regularity (consistency of approach)
    """
    
    findings = []
    text_lower = strategy_text.lower()
    
    # Dimension 1: Structural clarity
    clarity_markers = {
        "clear": ["clearly defined", "explicit", "unambiguous", "specific roles", "well-defined"],
        "vague": ["unclear", "ambiguous", "undefined", "vague", "loosely defined"]
    }
    
    clear_count = sum(1 for marker in clarity_markers["clear"] if marker in text_lower)
    vague_count = sum(1 for marker in clarity_markers["vague"] if marker in text_lower)
    
    if clear_count > vague_count and clear_count >= 2:
        findings.append({
            "dimension": "structural_clarity",
            "pattern": "clear_structure",
            "confidence": min(0.7 + (clear_count * 0.05), 0.95),
            "evidence": [f"Clear structure markers: {clarity_markers['clear'][:clear_count]}"],
            "categorical_family": "constraints",
            "origami_analogy": "Sharp, well-defined folds with clear mountain/valley distinction"
        })
    elif vague_count > clear_count and vague_count >= 2:
        findings.append({
            "dimension": "structural_clarity",
            "pattern": "ambiguous_structure",
            "confidence": min(0.7 + (vague_count * 0.05), 0.95),
            "evidence": [f"Vague structure markers: {clarity_markers['vague'][:vague_count]}"],
            "categorical_family": "constraints",
            "origami_analogy": "Soft, curved creases without clear definition"
        })
    
    # Dimension 2: Complexity appropriate to purpose
    complexity_markers = {
        "simple": ["straightforward", "simple", "minimal", "essential", "streamlined"],
        "complex": ["complex", "multifaceted", "intricate", "comprehensive", "elaborate"],
        "overwhelming": ["overwhelming", "convoluted", "overly complex", "unnecessarily complicated"]
    }
    
    simple_count = sum(1 for marker in complexity_markers["simple"] if marker in text_lower)
    complex_count = sum(1 for marker in complexity_markers["complex"] if marker in text_lower)
    overwhelming_count = sum(1 for marker in complexity_markers["overwhelming"] if marker in text_lower)
    
    if simple_count > complex_count:
        findings.append({
            "dimension": "complexity_management",
            "pattern": "minimal_elegant",
            "confidence": min(0.7 + (simple_count * 0.05), 0.95),
            "evidence": [f"Simplicity emphasis: {simple_count} markers found"],
            "categorical_family": "objects",
            "origami_analogy": "Few, well-chosen folds creating elegant form"
        })
    elif overwhelming_count >= 2:
        findings.append({
            "dimension": "complexity_management",
            "pattern": "excessive_complexity",
            "confidence": min(0.7 + (overwhelming_count * 0.05), 0.95),
            "evidence": [f"Complexity overload: {overwhelming_count} warning markers"],
            "categorical_family": "constraints",
            "origami_analogy": "Too many folds obscuring the intended form"
        })
    elif complex_count >= 3:
        findings.append({
            "dimension": "complexity_management",
            "pattern": "sophisticated_layering",
            "confidence": min(0.7 + (complex_count * 0.05), 0.95),
            "evidence": [f"Intentional complexity: {complex_count} markers"],
            "categorical_family": "objects",
            "origami_analogy": "Multiple layers revealing deeper structure"
        })
    
    # Dimension 3: Precision vs flexibility
    precision_markers = ["precise", "exact", "specific", "measured", "calibrated"]
    flexibility_markers = ["flexible", "adaptive", "responsive", "organic", "fluid"]
    
    precision_count = sum(1 for marker in precision_markers if marker in text_lower)
    flexibility_count = sum(1 for marker in flexibility_markers if marker in text_lower)
    
    if precision_count > flexibility_count and precision_count >= 2:
        findings.append({
            "dimension": "execution_approach",
            "pattern": "precise_execution",
            "confidence": min(0.7 + (precision_count * 0.05), 0.95),
            "evidence": [f"Precision emphasis: {precision_count} markers"],
            "categorical_family": "morphisms",
            "origami_analogy": "Sharp mountain and valley folds, high fold accuracy"
        })
    elif flexibility_count > precision_count and flexibility_count >= 2:
        findings.append({
            "dimension": "execution_approach",
            "pattern": "organic_flexibility",
            "confidence": min(0.7 + (flexibility_count * 0.05), 0.95),
            "evidence": [f"Flexibility emphasis: {flexibility_count} markers"],
            "categorical_family": "morphisms",
            "origami_analogy": "Curved creases, wet-folding technique for organic forms"
        })
    
    # Dimension 4: Pattern regularity
    regularity_markers = ["consistent", "uniform", "systematic", "standardized", "regular"]
    variability_markers = ["varied", "diverse", "customized", "tailored", "unique"]
    
    regularity_count = sum(1 for marker in regularity_markers if marker in text_lower)
    variability_count = sum(1 for marker in variability_markers if marker in text_lower)
    
    if regularity_count > variability_count and regularity_count >= 2:
        findings.append({
            "dimension": "pattern_consistency",
            "pattern": "tessellation_logic",
            "confidence": min(0.7 + (regularity_count * 0.05), 0.95),
            "evidence": [f"Pattern regularity: {regularity_count} markers"],
            "categorical_family": "morphisms",
            "origami_analogy": "Tessellation pattern with repeating geometric units"
        })
    elif variability_count >= 2:
        findings.append({
            "dimension": "pattern_consistency",
            "pattern": "modular_adaptation",
            "confidence": min(0.7 + (variability_count * 0.05), 0.95),
            "evidence": [f"Adaptive variability: {variability_count} markers"],
            "categorical_family": "morphisms",
            "origami_analogy": "Modular origami with context-specific unit selection"
        })
    
    # Dimension 5: Layering depth
    layering_markers = ["foundational", "builds upon", "layers", "hierarchical", "nested"]
    flat_markers = ["flat", "single-level", "non-hierarchical", "peer-based"]
    
    layering_count = sum(1 for marker in layering_markers if marker in text_lower)
    flat_count = sum(1 for marker in flat_markers if marker in text_lower)
    
    if layering_count >= 2:
        findings.append({
            "dimension": "information_architecture",
            "pattern": "layered_depth",
            "confidence": min(0.7 + (layering_count * 0.05), 0.95),
            "evidence": [f"Layering structure: {layering_count} markers"],
            "categorical_family": "objects",
            "origami_analogy": "Multiple paper layers creating dimensional depth"
        })
    elif flat_count >= 2:
        findings.append({
            "dimension": "information_architecture",
            "pattern": "flat_surface",
            "confidence": min(0.7 + (flat_count * 0.05), 0.95),
            "evidence": [f"Flat structure: {flat_count} markers"],
            "categorical_family": "objects",
            "origami_analogy": "Single-sheet design viewed overhead, minimal dimensional variation"
        })
    
    return {
        "domain": "origami_aesthetics",
        "findings": findings,
        "total_findings": len(findings),
        "methodology": "deterministic_pattern_matching",
        "llm_cost_tokens": 0,
    }


@mcp.tool()
def analyze_strategy_document_tool(strategy_text: str) -> dict:
    """
    Analyze a strategy document through origami aesthetics structural lens.
    
    This is the tomographic domain projection tool - it projects strategic
    text through origami vocabulary to detect structural patterns.
    
    Zero LLM cost - pure deterministic pattern matching.
    
    Args:
        strategy_text: Full text of the strategy document to analyze
    
    Returns:
        Dictionary with structural findings
    """
    return analyze_strategy_document(strategy_text)


# =============================================================================
# SERVER INFO
# =============================================================================

@mcp.tool()
def get_server_info() -> dict:
    """
    Get information about the Origami Aesthetics MCP server.
    
    Returns server metadata, capabilities, and Phase 1A/2.6 enhancements.
    """
    return {
        "name": "Origami Aesthetics MCP",
        "version": SERVER_VERSION,
        "validation_date": VALIDATION_DATE,
        "description": "Translates origami-inspired aesthetic concepts into visual parameters",
        "layer": "Layer 1 (Taxonomy) + Layer 2 (Deterministic)",
        "cost_profile": {
            "layer_1": "0 tokens (pure lookup)",
            "layer_2": "0 tokens (deterministic computation + RK4 integration)",
            "layer_3": "~100-200 tokens (Claude synthesis)"
        },
        "capabilities": {
            "layer_1_taxonomy": [
                "Aesthetic concept definitions",
                "Fold type properties",
                "Crease pattern characteristics",
                "Material properties",
                "Lighting interactions"
            ],
            "layer_2_structure": [
                "Intent to aesthetic concept extraction",
                "Aesthetic to visual parameter mapping",
                "Lighting parameter generation",
                "Composition parameter generation",
                "Complete specification synthesis",
                "compute_trajectory_between_aesthetic_states - RK4 trajectory integration (Phase 1A)",
                "generate_rhythmic_origami_sequence - Rhythmic oscillation (Phase 2.6)",
                "apply_origami_rhythmic_preset - Curated rhythmic patterns (Phase 2.6)",
                "list_origami_rhythmic_presets - Preset catalog (Phase 2.6)",
                "generate_attractor_visualization_prompt - Attractor to image prompt (Phase 2.7)",
                "generate_attractor_sequence_prompts - Temporal prompt sequences (Phase 2.7)",
                "list_origami_preset_attractors - Attractor catalog (Phase 2.7)",
                "list_origami_visual_types - Visual vocabulary catalog (Phase 2.7)"
            ],
            "cross_domain": [
                "Map to grid dynamics",
                "Map to narrative structure",
                "Domain registry integration (get_origami_domain_registry_config)"
            ],
            "tomographic": [
                "Strategy document analysis"
            ]
        },
        "morphospace": {
            "dimensions": len(ORIGAMI_PARAMETER_NAMES),
            "parameter_names": ORIGAMI_PARAMETER_NAMES,
            "bounds": ORIGAMI_BOUNDS,
            "aesthetic_states": list(AESTHETIC_STATES.keys()),
            "state_count": len(AESTHETIC_STATES)
        },
        "phase_enhancements": {
            "phase_1a": {
                "dynamics_available": DYNAMICS_AVAILABLE,
                "integration_method": "RK4 (Runge-Kutta 4th order)" if DYNAMICS_AVAILABLE else "Not available",
                "trajectory_features": [
                    "Zero-cost aesthetic transition computation",
                    "Smooth morphospace navigation",
                    "Convergence analysis",
                    "Path efficiency metrics",
                    "Intermediate state discovery"
                ] if DYNAMICS_AVAILABLE else [],
                "morphospace_structure": {
                    "dimensions": "5D (edge_sharpness, fold_accuracy, complexity_level, organic_quality, dimensional_depth)",
                    "states_defined": len(AESTHETIC_STATES),
                    "example_transitions": [
                        "precise → organic (geometric to natural)",
                        "minimal → complex (simple to elaborate)",
                        "geometric → meditative (logical to contemplative)"
                    ]
                }
            },
            "phase_2_6": {
                "available": True,
                "oscillation_patterns": ["sinusoidal", "triangular", "square"],
                "presets_count": len(ORIGAMI_RHYTHMIC_PRESETS),
                "preset_names": list(ORIGAMI_RHYTHMIC_PRESETS.keys()),
                "periods": sorted(set(p["steps_per_cycle"] for p in ORIGAMI_RHYTHMIC_PRESETS.values())),
                "features": [
                    "Periodic aesthetic transitions",
                    "Multiple oscillation patterns",
                    "Curated domain-specific presets",
                    "Phase point detection (peaks, troughs)",
                    "Temporal rhythm visualization"
                ],
                "use_cases": [
                    "Day/night aesthetic cycles",
                    "Seasonal complexity variation",
                    "Breath-like contemplative rhythms",
                    "Structural pulse patterns",
                    "Time-based origami evolution"
                ]
            },
            "phase_2_7": {
                "available": True,
                "visual_vocabulary_types": list(ORIGAMI_VISUAL_VOCABULARY.keys()),
                "preset_attractors": list(ORIGAMI_PRESET_ATTRACTORS.keys()),
                "prompt_modes": ["composite", "split_view", "sequence"],
                "features": [
                    "Attractor-to-prompt translation",
                    "5 visual vocabulary types (nearest-neighbor matching)",
                    "5 preset attractor states from Tier 4D discovery",
                    "Temporal sequence prompt generation",
                    "Multi-domain composition support (strength weighting)",
                    "Domain registry integration for emergent attractor system"
                ],
                "matching_method": "Euclidean nearest-neighbor in 5D parameter space"
            }
        },
        "compatible_bricks": [
            "aesthetic-dynamics-core - Phase 1A trajectory computation (required)",
            "grid-dynamics-mcp - Spatial arrangement mapping",
            "narrative-structure-mcp - Story structure mapping",
            "composition-graph-mcp - Multi-domain composition + Phase 2.7 orchestration"
        ],
        "validation_tests": [
            "Convergence: trajectories reach targets",
            "Monotonicity: distances decrease over time",
            "Round-trip: A→B→A returns to start",
            "Bounds: parameters stay within [0, 1]",
            "Aesthetic coherence: intermediate states are meaningful",
            "Vocabulary extraction: canonical states match at distance ≈ 0.0",
            "Prompt generation: all modes produce valid output"
        ],
        "usage_example": """
# Phase 1A: Compute trajectory from precise to organic
result = compute_trajectory_between_aesthetic_states(
    "precise", 
    "organic", 
    num_steps=20
)

# Phase 2.6: Generate rhythmic sequence
result = generate_rhythmic_origami_sequence(
    "precise",
    "organic", 
    oscillation_pattern="sinusoidal",
    num_cycles=2
)

# Phase 2.6: Apply preset
result = apply_origami_rhythmic_preset("daily_fold_cycle")

# Phase 2.7: Generate attractor visualization prompt
result = generate_attractor_visualization_prompt(
    attractor_name="period_30_universal_sync",
    mode="split_view"
)

# Phase 2.7: Generate temporal sequence of prompts
result = generate_attractor_sequence_prompts(
    "daily_fold_cycle",
    num_keyframes=6,
    mode="sequence"
)
        """
    }


# =============================================================================
# Server Entry Point
# =============================================================================

if __name__ == "__main__":
    mcp.run()


def main():
    """Entry point for script execution"""
    mcp.run()
