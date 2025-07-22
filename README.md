# Ingredient Analysis API

A comprehensive food ingredient analysis service that extracts ingredients from menu text, identifies allergens and dietary incompatibilities, and provides personalized safety assessments for users.

## Features

- **Intelligent Ingredient Extraction**: Uses AI to parse menu descriptions and extract ingredients
- **Missing Ingredient Detection**: Identifies commonly omitted ingredients in restaurant dishes
- **Allergen & Diet Analysis**: Identifies incompatible allergens and dietary restrictions
- **User Safety Assessment**: Personalized safety analysis based on user dietary profiles
- **Knowledge Graph Integration**: Leverages FoodOn ontology for accurate food classification
- **Validation Layer**: Built-in validation to prevent AI hallucinations and logical errors

## API Endpoints

### Main Analysis
```http
POST /analyze
```

Analyzes ingredient text and provides comprehensive food safety information.

**Request Body:**
```json
{
  "raw_text": "Grilled chicken breast with roasted vegetables and garlic butter",
  "dish_name": "Chicken Dinner", // optional
  "user_id": "user123" // optional, for personalized safety analysis
}
```

**Response:**
```json
{
  "dish_name": "Chicken Dinner",
  "core_ingredients": ["chicken breast", "vegetables", "garlic", "butter"],
  "suggested_ingredients": ["olive oil", "salt", "black pepper"],
  "all_ingredients": ["chicken breast", "vegetables", "garlic", "butter", "olive oil", "salt", "black pepper"],
  "ingredient_analysis": {
    "chicken breast": {
      "incompatible_allergens": [],
      "incompatible_diets": ["vegetarian", "vegan"],
      "confidence": 0.95,
      "foodon_uri": "http://purl.obolibrary.org/obo/FOODON_12345"
    },
    "butter": {
      "incompatible_allergens": ["milk"],
      "incompatible_diets": ["vegan", "dairy-free"],
      "confidence": 0.92,
      "foodon_uri": "http://purl.obolibrary.org/obo/FOODON_67890"
    }
  },
  "safety_analysis": {
    "overall_safety": "safe", // "safe", "not_safe", "unsure"
    "flagged_ingredients": {
      "butter": "not_safe"
    },
    "warnings": ["Contains dairy - not suitable for lactose intolerant users"],
    "explanations": {
      "butter": "Contains milk allergen which conflicts with user's dairy allergy"
    }
  },
  "user_profile": {
    "allergens": ["milk"],
    "dietary_restrictions": ["vegetarian"]
  },
  "statistics": {
    "total_ingredients": 7,
    "core_ingredients": 4,
    "suggested_ingredients": 3,
    "successfully_analyzed": 6
  }
}
```

### Batch Processing
```http
POST /analyze/batch
```

Processes multiple dishes at once.

**Request Body:**
```json
{
  "dishes": [
    {
      "raw_text": "Caesar salad with croutons",
      "dish_name": "Caesar Salad",
      "user_id": "user123"
    },
    {
      "raw_text": "Grilled salmon with rice",
      "dish_name": "Salmon Dinner"
    }
  ]
}
```

### User Profile Management

#### Get User Profile
```http
GET /users/{user_id}/profile
```

**Response:**
```json
{
  "user_id": "user123",
  "allergens": ["milk", "nuts"],
  "dietary_restrictions": ["vegetarian"],
  "custom_restrictions": {
    "avoid_ingredients": ["msg", "artificial_colors"]
  }
}
```

#### Update User Preferences
```http
POST /users/{user_id}/preferences
```

**Request Body:**
```json
{
  "allergens": ["milk", "nuts"],
  "dietary_restrictions": ["vegetarian", "gluten-free"],
  "custom_restrictions": {
    "avoid_ingredients": ["msg"]
  }
}
```

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-22T10:30:00Z",
  "services": {
    "database": "connected",
    "ai_model": "available",
    "knowledge_graph": "loaded"
  }
}
```

## Common Allergens Detected

- Milk/Dairy
- Eggs
- Fish
- Shellfish
- Tree nuts
- Peanuts
- Soy
- Wheat/Gluten
- Sesame

## Dietary Restrictions Supported

- Vegetarian
- Vegan
- Gluten-free
- Dairy-free
- Nut-free
- Halal
- Kosher

## Safety Levels

- **`safe`**: No known conflicts with user's dietary restrictions
- **`not_safe`**: Direct conflict with user's allergens or dietary restrictions
- **`unsure`**: Potential hidden ingredients, cross-contamination risks, or missing information

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid input)
- `404`: Resource not found (user profile)
- `500`: Internal server error

**Error Response Format:**
```json
{
  "error": "Invalid input format",
  "details": "Missing required field 'raw_text'",
  "timestamp": "2025-07-22T10:30:00Z"
}
```

## Rate Limits

- Analysis endpoints: 100 requests/hour per IP
- User management: 1000 requests/hour per user
- Health check: No limit

## Example Use Cases

### Restaurant Menu Analysis
```bash
curl -X POST http://api.example.com/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "raw_text": "Pad Thai with shrimp, peanuts, and bean sprouts",
    "dish_name": "Pad Thai",
    "user_id": "allergic_user_123"
  }'
```

### Updating Dietary Preferences
```bash
curl -X POST http://api.example.com/users/user123/preferences \
  -H "Content-Type: application/json" \
  -d '{
    "allergens": ["peanuts", "shellfish"],
    "dietary_restrictions": ["gluten-free"]
  }'
```

## Integration Notes

- Include `user_id` in analysis requests for personalized safety assessments
- The API automatically suggests common missing ingredients (seasonings, cooking oils, etc.)
- Validation layer prevents common AI errors (e.g., meat being marked as vegetarian)
- Results are cached for faster repeated analysis of common ingredients

## Support

For questions or issues, please contact the development team or check the API health endpoint for system status.