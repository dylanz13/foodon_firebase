from typing import TypedDict, List, Dict, Any, Optional, Literal
import firebase_admin
from firebase_admin import credentials, firestore
from rdflib import Graph, Namespace, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.runnables import RunnableSequence
from langgraph.graph import StateGraph, START, END
import os
import logging
import asyncio
from functools import lru_cache
import json
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from enum import Enum

logging.basicConfig(level=logging.INFO)
load_dotenv()

# ----------------------------------------
# Enhanced Types and Validation
# ----------------------------------------
class SafetyFlag(str, Enum):
    SAFE = "safe"
    NOT_SAFE = "not_safe"
    UNSURE = "unsure"

class IngredientInfo(BaseModel):
    incompatible_allergens: List[str] = Field(default_factory=list)  # Changed from "allergens"
    incompatible_diets: List[str] = Field(default_factory=list)      # Changed from "diets"
    chosen_uri: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

class IngredientValidation(BaseModel):
    is_valid: bool
    validation_errors: List[str] = Field(default_factory=list)
    corrected_incompatible_allergens: List[str] = Field(default_factory=list)
    corrected_incompatible_diets: List[str] = Field(default_factory=list)

class UserProfile(BaseModel):
    user_id: str
    allergens: List[str] = Field(default_factory=list)
    dietary_restrictions: List[str] = Field(default_factory=list)
    custom_restrictions: Dict[str, List[str]] = Field(default_factory=dict)

class SanityCheck(BaseModel):
    recipe_ingredients_valid: bool
    issues_found: List[str] = Field(default_factory=list)
    corrected_ingredients: List[str] = Field(default_factory=list)

class MissingIngredientsAnalysis(BaseModel):
    sanity_check: SanityCheck
    suggested_ingredients: List[str] = Field(default_factory=list)
    reasoning: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

class SafetyAnalysis(BaseModel):
    overall_safety: SafetyFlag
    flagged_ingredients: Dict[str, SafetyFlag] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    explanations: Dict[str, str] = Field(default_factory=dict)

class IngredientState(TypedDict, total=False):
    # Original fields
    raw_text: str
    dish_name: Optional[str]
    user_id: Optional[str]
    ingredients: List[str]
    resolved: Dict[str, IngredientInfo]
    unresolved: List[str]
    kg_candidates: Dict[str, List[tuple]]
    errors: List[str]
    
    # New fields for enhanced analysis
    user_profile: Optional[UserProfile]
    missing_ingredients_analysis: Optional[MissingIngredientsAnalysis]
    all_ingredients: List[str]  # Core + suggested ingredients
    safety_analysis: Optional[SafetyAnalysis]

# ----------------------------------------
# Lazy Resource Loading
# ----------------------------------------
@lru_cache(maxsize=1)
def get_foodon_graph():
    """Lazy load FoodOn graph to improve startup time."""
    try:
        graph = Graph()
        graph.parse("config/foodon.owl", format="xml")
        logging.info("FoodOn ontology loaded successfully")
        return graph
    except Exception as e:
        logging.error(f"Failed to load FoodOn ontology: {e}")
        return None

@lru_cache(maxsize=1)
def get_firebase_db():
    """Lazy Firebase initialization with error handling."""
    try:
        if not firebase_admin._apps:
            firestore_path = os.getenv("FIRESTORE_PATH")
            cred = credentials.Certificate(firestore_path)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        logging.error(f"Firebase initialization failed: {e}")
        return None

@lru_cache(maxsize=1)
def get_llm():
    """Initialize LLM with proper configuration."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_api_key=api_key,
        temperature=0.1,
        max_retries=3
    )

# ----------------------------------------
# Enhanced Prompts
# ----------------------------------------
MISSING_INGREDIENTS_PROMPT = PromptTemplate(
    input_variables=["dish_name", "known_ingredients", "ocr_text"],
    template="""Analyze this restaurant dish and suggest missing ingredients:

Dish Name: {dish_name}

Known Ingredients (from recipe database):
{known_ingredients}

Original OCR Text (may contain additional clues):
{ocr_text}

IMPORTANT: First, perform a sanity check on the recipe database ingredients:
1. Do the listed ingredients actually make sense for this dish?
2. Are there any obviously incorrect or unrelated ingredients?
3. Are the ingredients appropriate for the cooking style/cuisine?

Then, suggest common ingredients that are likely missing from the known ingredients list. Focus on:
1. Common toppings and garnishes
2. Seasonings and spices typically used
3. Cooking ingredients often omitted from menus
4. Preparation components (oils, vinegars, etc.)
5. Side accompaniments commonly served with this dish

Consider the restaurant context and typical preparation methods.

Respond with JSON in this exact format:
{{
  "sanity_check": {{
    "recipe_ingredients_valid": true,
    "issues_found": ["list any problematic ingredients from the database"],
    "corrected_ingredients": ["list of ingredients that should replace invalid ones"]
  }},
  "suggested_ingredients": ["ingredient1", "ingredient2", ...],
  "reasoning": "Brief explanation of why these ingredients are likely included",
  "confidence": 0.8
}}

Keep ingredients as simple names (e.g., "olive oil", "garlic", "parsley").
Confidence should be between 0.0 and 1.0."""
)

EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["raw_text"],
    template="""You are a food ingredient extraction expert. Parse this menu text into structured JSON.

RULES:
- Extract only actual food ingredients, not cooking methods or descriptions
- Convert to singular form and normalize names
- Split compound ingredients when clear
- Include seasonings and allergens if mentioned

OUTPUT FORMAT: {{"ingredients": ["ingredient1", "ingredient2", ...]}}

EXAMPLES:
Input: "Grilled chicken breast with roasted vegetables and garlic butter"
Output: {{"ingredients": ["chicken breast", "mixed vegetables", "garlic", "butter"]}}

Input: "Pasta with tomato sauce and parmesan cheese"  
Output: {{"ingredients": ["pasta", "tomato", "parmesan cheese"]}}

TEXT TO PARSE: '''{raw_text}'''"""
)

REASONING_PROMPT = PromptTemplate(
    input_variables=["ingredient", "candidates"],
    template="""You are an expert in the FoodOn food ontology. Analyze these ontology candidates for ingredient '{ingredient}':

CANDIDATES: {candidates}

TASK: Select the best matching food concept and identify INCOMPATIBLE allergens/dietary restrictions.

IMPORTANT RULES:
- Only list allergens that this ingredient DEFINITELY CONTAINS
- Only list diets that this ingredient is DEFINITELY NOT suitable for
- Common allergens: milk, eggs, fish, shellfish, nuts, peanuts, soy, wheat, sesame
- Common diet restrictions: vegetarian, vegan, gluten-free, dairy-free, nut-free, halal, kosher
- Be conservative - if unsure, don't include it
- Rate your confidence (0.0-1.0)

EXAMPLES:
- "pork" → incompatible_diets: ["vegetarian", "vegan", "halal", "kosher"]
- "wheat flour" → incompatible_allergens: ["wheat"], incompatible_diets: ["gluten-free"]
- "milk" → incompatible_allergens: ["milk"], incompatible_diets: ["vegan", "dairy-free"]

OUTPUT FORMAT:
{{{{
    "chosen_uri": "best_matching_uri_or_null",
    "incompatible_allergens": ["allergen1", "allergen2"],
    "incompatible_diets": ["diet1", "diet2"], 
    "confidence": 0.85
}}}}

RESPOND WITH VALID JSON ONLY:""")

SAFETY_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["ingredients", "user_allergens", "user_restrictions", "dish_name"],
    template="""You are a food safety expert analyzing ingredients for dietary compatibility.

DISH: {dish_name}
INGREDIENTS TO ANALYZE: {ingredients}

USER PROFILE:
- Allergens: {user_allergens}
- Dietary Restrictions: {user_restrictions}

ANALYSIS RULES:
1. SAFE: No conflicts with user's dietary needs
2. NOT_SAFE: Direct conflict with allergens or restrictions
3. UNSURE: Potential hidden ingredients, cross-contamination risks, or preparation uncertainties

Consider these common hidden risks:
- Cross-contamination in kitchens
- Shared fryers/cooking surfaces
- Hidden ingredients in sauces/seasonings
- Processing aids not listed on menus
- Common preparation methods that might introduce allergens

For each flagged ingredient, provide a clear explanation of the concern.

OUTPUT FORMAT:
{{
  "overall_safety": "safe|not_safe|unsure",
  "flagged_ingredients": {{
    "ingredient_name": "safe|not_safe|unsure"
  }},
  "warnings": ["warning1", "warning2"],
  "explanations": {{
    "ingredient_name": "detailed explanation of the safety concern"
  }}
}}

RESPOND WITH VALID JSON ONLY:"""
)

# ----------------------------------------
# User Profile Management
# ----------------------------------------
async def load_user_profile(state: IngredientState) -> IngredientState:
    """Load user dietary profile from Firebase."""
    try:
        user_id = state.get("user_id")
        if not user_id:
            state["user_profile"] = None
            return state
            
        db = get_firebase_db()
        if not db:
            state["user_profile"] = None
            return state
            
        user_doc = db.collection("preferences").document(user_id).get()
        
        if user_doc.exists:
            profile_data = user_doc.to_dict()
            print(profile_data)
            state["user_profile"] = UserProfile(
                user_id=user_id,
                allergens=profile_data.get("allergens", []),
                dietary_restrictions=profile_data.get("dietary_restrictions", []),
                custom_restrictions=profile_data.get("custom_restrictions", {})
            )
            logging.info(f"Loaded profile for user {user_id}")
        else:
            state["user_profile"] = UserProfile(user_id=user_id)
            logging.info(f"No profile found for user {user_id}, using defaults")
            
        return state
        
    except Exception as e:
        error_msg = f"User profile loading failed: {str(e)}"
        logging.error(error_msg)
        state.setdefault("errors", []).append(error_msg)
        state["user_profile"] = None
        return state

# ----------------------------------------
# Enhanced Node Implementations
# ----------------------------------------
async def analyze_missing_ingredients(state: IngredientState) -> IngredientState:
    """Analyze dish for missing ingredients that might pose risks."""
    try:
        llm = get_llm()
        chain = RunnableSequence(MISSING_INGREDIENTS_PROMPT, llm, JsonOutputParser())
        
        dish_name = state.get("dish_name", "Unknown dish")
        known_ingredients = state.get("ingredients", [])
        raw_text = state.get("raw_text", "")
        
        known_ingredients_str = ', '.join(known_ingredients) if known_ingredients else 'None found'
        
        result = await chain.ainvoke({
            "dish_name": dish_name,
            "known_ingredients": known_ingredients_str,
            "ocr_text": raw_text
        })
        
        analysis = MissingIngredientsAnalysis(**result)
        state["missing_ingredients_analysis"] = analysis
        
        # Combine core ingredients with suggested ones
        all_ingredients = known_ingredients.copy()
        if analysis.sanity_check.corrected_ingredients:
            # Replace problematic ingredients with corrected ones
            for issue in analysis.sanity_check.issues_found:
                if issue in all_ingredients:
                    all_ingredients.remove(issue)
            all_ingredients.extend(analysis.sanity_check.corrected_ingredients)
        
        all_ingredients.extend(analysis.suggested_ingredients)
        state["all_ingredients"] = list(set(all_ingredients))  # Remove duplicates
        
        logging.info(f"Missing ingredients analysis: {len(analysis.suggested_ingredients)} suggested, confidence: {analysis.confidence}")
        return state
        
    except Exception as e:
        error_msg = f"Missing ingredients analysis failed: {str(e)}"
        logging.error(error_msg)
        state.setdefault("errors", []).append(error_msg)
        state["all_ingredients"] = state.get("ingredients", [])
        return state

async def llm_extract_ingredients(state: IngredientState) -> IngredientState:
    """Extract ingredients using LLM with error handling."""
    try:
        llm = get_llm()
        chain = RunnableSequence(EXTRACTION_PROMPT, llm, JsonOutputParser())
        
        raw = state.get("raw_text", "")
        if not raw.strip():
            state["ingredients"] = []
            state["unresolved"] = []
            state["resolved"] = {}
            return state
            
        result = await chain.ainvoke({"raw_text": raw})
        ingredients = result.get("ingredients", [])
        
        # Validate and clean ingredients
        cleaned_ingredients = [
            ing.strip().lower() for ing in ingredients
            if ing and isinstance(ing, str) and len(ing.strip()) > 1
        ]
        
        state["ingredients"] = cleaned_ingredients
        state["unresolved"] = cleaned_ingredients.copy()
        state["resolved"] = {}
        state["errors"] = state.get("errors", [])
        
        logging.info(f"Extracted {len(cleaned_ingredients)} ingredients")
        return state
        
    except Exception as e:
        error_msg = f"LLM extraction failed: {str(e)}"
        logging.error(error_msg)
        state.setdefault("errors", []).append(error_msg)
        state["ingredients"] = []
        state["unresolved"] = []
        state["resolved"] = {}
        return state

def check_firebase_cache(state: IngredientState) -> IngredientState:
    """Check Firebase cache with error handling and batch operations."""
    try:
        db = get_firebase_db()
        if not db:
            return state
            
        coll = db.collection("ingredients_cache")
        new_unresolved = []
        
        # Use all_ingredients if available, otherwise fall back to ingredients
        ingredients_to_check = state.get("all_ingredients", state.get("unresolved", []))
        
        # Batch read for better performance
        docs_to_read = [coll.document(ing.lower().strip()) for ing in ingredients_to_check]
        docs = db.get_all(docs_to_read)
        
        for i, doc in enumerate(docs):
            ingredient = ingredients_to_check[i]
            if doc.exists:
                try:
                    data = doc.to_dict()
                    validated_info = IngredientInfo(**data)
                    state["resolved"][ingredient] = validated_info
                    logging.info(f"Cache hit for ingredient: {ingredient}")
                except Exception as e:
                    logging.warning(f"Invalid cache data for {ingredient}: {e}")
                    new_unresolved.append(ingredient)
            else:
                new_unresolved.append(ingredient)
        
        state["unresolved"] = new_unresolved
        logging.info(f"Cache resolved {len(state['resolved'])} ingredients, {len(new_unresolved)} remain")
        return state
        
    except Exception as e:
        error_msg = f"Firebase cache check failed: {str(e)}"
        logging.error(error_msg)
        state.setdefault("errors", []).append(error_msg)
        return state

def sparql_query_foodon(state: IngredientState) -> IngredientState:
    """Query FoodOn ontology with SPARQL injection prevention."""
    try:
        graph = get_foodon_graph()
        if not graph:
            return state
            
        FOODON = Namespace("http://purl.obolibrary.org/obo/FOODON_")
        candidates = {}
        
        for ing in state["unresolved"]:
            # Sanitize input to prevent SPARQL injection
            safe_ingredient = ing.replace("'", "").replace('"', '').replace('\\', '')
            
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
            SELECT DISTINCT ?concept ?label WHERE {{
              VALUES ?pred {{ rdfs:label oboInOwl:hasExactSynonym oboInOwl:hasRelatedSynonym }}
              ?concept ?pred ?label .
              FILTER(
                LANG(?label) = "en" && 
                (CONTAINS(LCASE(str(?label)), LCASE("{safe_ingredient}")) ||
                 CONTAINS(LCASE("{safe_ingredient}"), LCASE(str(?label))))
              )
              FILTER(STRSTARTS(str(?concept), "http://purl.obolibrary.org/obo/FOODON_"))
            }} 
            ORDER BY STRLEN(str(?label))
            LIMIT 15"""
            
            try:
                results = graph.query(query)
                candidates[ing] = [(str(r.concept), str(r.label)) for r in results]
                logging.info(f"Found {len(candidates[ing])} candidates for '{ing}'")
            except Exception as query_error:
                logging.warning(f"SPARQL query failed for '{ing}': {query_error}")
                candidates[ing] = []
        
        state["kg_candidates"] = candidates
        return state
        
    except Exception as e:
        error_msg = f"SPARQL querying failed: {str(e)}"
        logging.error(error_msg)
        state.setdefault("errors", []).append(error_msg)
        state["kg_candidates"] = {}
        return state

async def llm_reason_over_kg(state: IngredientState) -> IngredientState:
    """Use LLM to reason over knowledge graph candidates."""
    try:
        llm = get_llm()
        chain = RunnableSequence(REASONING_PROMPT, llm, JsonOutputParser())
        
        for ing, candidates in state.get("kg_candidates", {}).items():
            if not candidates:
                fallback_info = IngredientInfo(
                    allergens=[],
                    diets=[],
                    chosen_uri=None,
                    confidence=0.1
                )
                state["resolved"][ing] = fallback_info
                continue
                
            try:
                response = await chain.ainvoke({
                    "ingredient": ing,
                    "candidates": candidates[:10]
                })
                
                validated_info = IngredientInfo(**response)
                state["resolved"][ing] = validated_info
                
                logging.info(f"Resolved '{ing}' with confidence {validated_info.confidence}")
                
            except Exception as reason_error:
                logging.warning(f"LLM reasoning failed for '{ing}': {reason_error}")
                fallback_info = IngredientInfo(confidence=0.1)
                state["resolved"][ing] = fallback_info
        
        return state
        
    except Exception as e:
        error_msg = f"LLM reasoning failed: {str(e)}"
        logging.error(error_msg)
        state.setdefault("errors", []).append(error_msg)
        return state

async def analyze_safety_for_user(state: IngredientState) -> IngredientState:
    """Analyze safety of all ingredients against user profile."""
    try:
        user_profile = state.get("user_profile")
        if not user_profile:
            # No user profile, mark as unsure
            state["safety_analysis"] = SafetyAnalysis(
                overall_safety=SafetyFlag.UNSURE,
                warnings=["No user dietary profile available for safety analysis"]
            )
            return state
        
        llm = get_llm()
        chain = RunnableSequence(SAFETY_ANALYSIS_PROMPT, llm, JsonOutputParser())
        
        all_ingredients = state.get("all_ingredients", [])
        dish_name = state.get("dish_name", "Unknown dish")
        
        result = await chain.ainvoke({
            "ingredients": ', '.join(all_ingredients),
            "user_allergens": ', '.join(user_profile.allergens) if user_profile.allergens else 'None',
            "user_restrictions": ', '.join(user_profile.dietary_restrictions) if user_profile.dietary_restrictions else 'None',
            "dish_name": dish_name
        })
        
        # Additional logic: If ANY ingredient is "not_safe", overall should be "not_safe"
        flagged_ingredients = result.get("flagged_ingredients", {})
        if any(flag == "not_safe" for flag in flagged_ingredients.values()):
            result["overall_safety"] = "not_safe"
        elif any(flag == "unsure" for flag in flagged_ingredients.values()) and result["overall_safety"] == "safe":
            result["overall_safety"] = "unsure"
        
        state["safety_analysis"] = SafetyAnalysis(**result)
        
        logging.info(f"Safety analysis complete: {result['overall_safety']}")
        return state
        
    except Exception as e:
        error_msg = f"Safety analysis failed: {str(e)}"
        logging.error(error_msg)
        state.setdefault("errors", []).append(error_msg)
        state["safety_analysis"] = SafetyAnalysis(
            overall_safety=SafetyFlag.UNSURE,
            warnings=[f"Safety analysis failed: {str(e)}"]
        )
        return state

def update_firebase_cache(state: IngredientState) -> IngredientState:
    """Update Firebase cache with batch operations."""
    try:
        db = get_firebase_db()
        if not db:
            return state
            
        coll = db.collection("ingredients_cache")
        batch = db.batch()
        
        updates = 0
        for ing, info in state["resolved"].items():
            if hasattr(info, 'confidence') and info.confidence > 0.5:
                key = ing.lower().strip()
                doc_ref = coll.document(key)
                batch.set(doc_ref, info.dict() if hasattr(info, 'dict') else info.__dict__)
                updates += 1
        
        if updates > 0:
            batch.commit()
            logging.info(f"Cached {updates} ingredient mappings")
            
        return state
        
    except Exception as e:
        error_msg = f"Cache update failed: {str(e)}"
        logging.error(error_msg)
        state.setdefault("errors", []).append(error_msg)
        return state

def return_result(state: IngredientState) -> IngredientState:
    """Final processing and result preparation."""
    # Ensure all ingredients have entries
    all_ingredients = state.get("all_ingredients", [])
    for ing in all_ingredients:
        if ing not in state.get("resolved", {}):
            state.setdefault("resolved", {})[ing] = IngredientInfo(confidence=0.0)
    
    state["unresolved"] = []
    
    # Log final statistics
    total = len(all_ingredients)
    resolved = len(state.get("resolved", {}))
    errors = len(state.get("errors", []))
    safety = state.get("safety_analysis", {})
    
    logging.info(f"Processing complete: {resolved}/{total} ingredients resolved, {errors} errors, safety: {getattr(safety, 'overall_safety', 'unknown')}")
    return state

VALIDATION_PROMPT = PromptTemplate(
    input_variables=["ingredient", "incompatible_allergens", "incompatible_diets"],
    template="""You are a food safety validator. Review this ingredient analysis for logical errors:

INGREDIENT: {ingredient}
REPORTED INCOMPATIBLE ALLERGENS: {incompatible_allergens}
REPORTED INCOMPATIBLE DIETS: {incompatible_diets}

VALIDATION RULES:
1. Check if allergens make sense (e.g., pork should NOT contain wheat allergen)
2. Check if diet restrictions are logical (e.g., pork is NOT vegetarian/vegan)
3. Look for common sense violations
4. Be strict - flag anything that seems wrong

COMMON VALIDATION CHECKS:
- Meat products: incompatible with vegetarian/vegan, but don't contain grain allergens
- Dairy products: contain milk allergen, incompatible with vegan/dairy-free
- Grain products: may contain gluten allergens, but not meat-related restrictions
- Nuts: contain nut allergens, but not necessarily incompatible with vegetarian diets

OUTPUT FORMAT:
{{
    "is_valid": true/false,
    "validation_errors": ["error1", "error2"],
    "corrected_incompatible_allergens": ["corrected_list"],
    "corrected_incompatible_diets": ["corrected_list"]
}}

RESPOND WITH VALID JSON ONLY:""")

async def validate_ingredient_analysis(state: IngredientState) -> IngredientState:
    """Validate ingredient analysis to prevent hallucinations."""
    try:
        llm = get_llm()
        chain = RunnableSequence(VALIDATION_PROMPT, llm, JsonOutputParser())
        
        validated_resolved = {}
        
        for ing, info in state.get("resolved", {}).items():
            try:
                # Skip validation for very low confidence results
                if hasattr(info, 'confidence') and info.confidence < 0.3:
                    validated_resolved[ing] = info
                    continue
                
                incompatible_allergens = getattr(info, 'incompatible_allergens', [])
                incompatible_diets = getattr(info, 'incompatible_diets', [])
                
                validation_result = await chain.ainvoke({
                    "ingredient": ing,
                    "incompatible_allergens": ', '.join(incompatible_allergens) if incompatible_allergens else 'None',
                    "incompatible_diets": ', '.join(incompatible_diets) if incompatible_diets else 'None'
                })
                
                validation = IngredientValidation(**validation_result)
                
                if not validation.is_valid:
                    # Use corrected values
                    corrected_info = IngredientInfo(
                        incompatible_allergens=validation.corrected_incompatible_allergens,
                        incompatible_diets=validation.corrected_incompatible_diets,
                        chosen_uri=getattr(info, 'chosen_uri', None),
                        confidence=max(0.1, getattr(info, 'confidence', 0.0) - 0.2)  # Reduce confidence for corrected items
                    )
                    validated_resolved[ing] = corrected_info
                    
                    logging.warning(f"Corrected analysis for '{ing}': {validation.validation_errors}")
                else:
                    validated_resolved[ing] = info
                    
            except Exception as validation_error:
                logging.warning(f"Validation failed for '{ing}': {validation_error}")
                # Keep original if validation fails
                validated_resolved[ing] = info
        
        state["resolved"] = validated_resolved
        logging.info(f"Validation complete for {len(validated_resolved)} ingredients")
        return state
        
    except Exception as e:
        error_msg = f"Ingredient validation failed: {str(e)}"
        logging.error(error_msg)
        state.setdefault("errors", []).append(error_msg)
        return state

def should_query_kg(state: IngredientState) -> str:
    """Conditional routing based on unresolved ingredients."""
    return "sparql_foodon" if state.get("unresolved") else "safety_analysis"

def has_user_profile(state: IngredientState) -> str:
    """Check if we need to load user profile."""
    return "missing_ingredients" if state.get("user_id") else "missing_ingredients"

# ----------------------------------------
# Enhanced Graph Construction
# ----------------------------------------
def create_ingredient_graph() -> StateGraph:
    """Create the enhanced ingredient analysis workflow graph."""
    builder = StateGraph(IngredientState)
    
    # Add nodes
    builder.add_node('load_user_profile', load_user_profile)
    builder.add_node('llm_extract', llm_extract_ingredients)
    builder.add_node('missing_ingredients', analyze_missing_ingredients)
    builder.add_node('firebase_check', check_firebase_cache)
    builder.add_node('sparql_foodon', sparql_query_foodon)
    builder.add_node('reason_kg', llm_reason_over_kg)
    builder.add_node('validate_analysis', validate_ingredient_analysis)  # NEW NODE
    builder.add_node('safety_analysis', analyze_safety_for_user)
    builder.add_node('update_cache', update_firebase_cache)
    builder.add_node('return', return_result)

    # Define workflow
    builder.set_entry_point('load_user_profile')
    builder.set_finish_point('return')
    
    # Add edges (modified to include validation)
    builder.add_edge('load_user_profile', 'llm_extract')
    builder.add_edge('llm_extract', 'missing_ingredients')
    builder.add_edge('missing_ingredients', 'firebase_check')
    builder.add_conditional_edges(
        'firebase_check',
        should_query_kg,
        {
            'sparql_foodon': 'sparql_foodon',
            'safety_analysis': 'validate_analysis'  # CHANGED: Go to validation first
        }
    )
    builder.add_edge('sparql_foodon', 'reason_kg')
    builder.add_edge('reason_kg', 'validate_analysis')    # NEW EDGE
    builder.add_edge('validate_analysis', 'safety_analysis')  # NEW EDGE
    builder.add_edge('safety_analysis', 'update_cache')
    builder.add_edge('update_cache', 'return')

    return builder.compile()

# ----------------------------------------
# Usage Functions
# ----------------------------------------
async def analyze_ingredients_for_user(
    raw_text: str,
    user_id: Optional[str] = None,
    dish_name: Optional[str] = None
) -> Dict[str, Any]:
    """Main entry point for ingredient analysis with user safety checking."""
    graph = create_ingredient_graph()
    
    initial_state: IngredientState = {
        "raw_text": raw_text,
        "dish_name": dish_name,
        "user_id": user_id,
        "ingredients": [],
        "resolved": {},
        "unresolved": [],
        "all_ingredients": [],
        "errors": []
    }
    
    result = await graph.ainvoke(initial_state)
    
    # Format comprehensive output
    safety_analysis = result.get("safety_analysis")
    missing_analysis = result.get("missing_ingredients_analysis")
    
    return {
        "dish_name": result.get("dish_name"),
        "core_ingredients": result.get("ingredients", []),
        "suggested_ingredients": missing_analysis.suggested_ingredients if missing_analysis else [],
        "all_ingredients": result.get("all_ingredients", []),
        "ingredient_analysis": {
            ingredient: {
                "incompatible_allergens": info.incompatible_allergens if hasattr(info, 'incompatible_allergens') else [],
                "incompatible_diets": info.incompatible_diets if hasattr(info, 'incompatible_diets') else [],
                "confidence": info.confidence if hasattr(info, 'confidence') else 0.0,
                "foodon_uri": info.chosen_uri if hasattr(info, 'chosen_uri') else None
            }
            for ingredient, info in result.get("resolved", {}).items()
        },
        "safety_analysis": {
            "overall_safety": safety_analysis.overall_safety if safety_analysis else "unsure",
            "flagged_ingredients": safety_analysis.flagged_ingredients if safety_analysis else {},
            "warnings": safety_analysis.warnings if safety_analysis else [],
            "explanations": safety_analysis.explanations if safety_analysis else {}
        } if safety_analysis else None,
        "missing_ingredients_analysis": {
            "sanity_check": missing_analysis.sanity_check.__dict__ if missing_analysis and missing_analysis.sanity_check else None,
            "suggested_ingredients": missing_analysis.suggested_ingredients if missing_analysis else [],
            "reasoning": missing_analysis.reasoning if missing_analysis else "",
            "confidence": missing_analysis.confidence if missing_analysis else 0.0
        } if missing_analysis else None,
        "user_profile": {
            "allergens": result["user_profile"].allergens if result.get("user_profile") else [],
            "dietary_restrictions": result["user_profile"].dietary_restrictions if result.get("user_profile") else []
        } if result.get("user_profile") else None,
        "errors": result.get("errors", []),
        "statistics": {
            "total_ingredients": len(result.get("all_ingredients", [])),
            "core_ingredients": len(result.get("ingredients", [])),
            "suggested_ingredients": len(missing_analysis.suggested_ingredients) if missing_analysis else 0,
            "successfully_analyzed": len([
                info for info in result.get("resolved", {}).values()
                if hasattr(info, 'confidence') and info.confidence > 0.5
            ])
        }
    }

# Helper functions for user profile management
async def update_user_dietary_preferences(
    user_id: str,
    allergens: List[str] = None,
    dietary_restrictions: List[str] = None,
    custom_restrictions: Dict[str, List[str]] = None
) -> bool:
    """Update user dietary preferences in Firebase."""
    try:
        db = get_firebase_db()
        if not db:
            return False
            
        doc_ref = db.collection("user_profiles").document(user_id)
        
        update_data = {}
        if allergens is not None:
            update_data["allergens"] = allergens
        if dietary_restrictions is not None:
            update_data["dietary_restrictions"] = dietary_restrictions
        if custom_restrictions is not None:
            update_data["custom_restrictions"] = custom_restrictions
        
        doc_ref.set(update_data, merge=True)
        logging.info(f"Updated dietary preferences for user {user_id}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to update user preferences: {e}")
        return False

# Example usage:
if __name__ == "__main__":
    # Example with user safety analysis
    async def main():
        result = await analyze_ingredients_for_user(
            raw_text="Black tonkatsu sesame ramen",
            user_id="ih9K8vqe14MJKr60Bzhp4MZToE53"
        )
        print(json.dumps(result, indent=2))
    
    asyncio.run(main())
