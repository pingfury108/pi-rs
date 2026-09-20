//! Constrained sampling support, port of `packages/ai/src/api/constrained-sampling.ts`:
//! convert tool JSON Schemas to the strict subset expected by provider
//! structured-output modes (all properties required, additionalProperties
//! false, optional non-null properties wrapped in anyOf null).

use serde_json::Value;

/// Keys unsupported by strict schemas.
const UNSUPPORTED_KEYS: &[&str] = &[
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
];

/// Convert a tool schema to the strict subset. Returns an error message when
/// the schema cannot be represented strictly (caller falls back to the
/// original schema with `strict: false`).
pub fn make_strict_json_schema(schema: &Value) -> Result<Value, String> {
    let mut cloned = schema.clone();
    make_node_strict(&mut cloned)?;
    Ok(cloned)
}

/// Resolve strict sampling for a tool: `Some(strict_schema)` when the schema
/// converts cleanly, `None` to fall back (`strict: false`, original schema).
pub fn resolve_strict(schema: &Value, supports_strict: bool) -> Option<Value> {
    if !supports_strict {
        return None;
    }
    make_strict_json_schema(schema).ok()
}

fn is_object(v: &Value) -> bool {
    v.is_object()
}

fn schema_allows_null(schema: &Value) -> bool {
    if !is_object(schema) {
        return false;
    }
    match schema.get("type") {
        Some(Value::String(t)) if t == "null" => return true,
        Some(Value::Array(types)) => {
            if types.iter().any(|t| t == "null") {
                return true;
            }
        }
        _ => {}
    }
    if schema.get("const") == Some(&Value::Null) {
        return true;
    }
    if let Some(Value::Array(e)) = schema.get("enum") {
        if e.contains(&Value::Null) {
            return true;
        }
    }
    schema
        .get("anyOf")
        .and_then(Value::as_array)
        .is_some_and(|variants| variants.iter().any(schema_allows_null))
}

fn make_node_strict(schema: &mut Value) -> Result<(), String> {
    if !is_object(schema) {
        return Err("boolean schemas are unsupported".into());
    }
    for key in UNSUPPORTED_KEYS {
        if schema.get(key).is_some() {
            return Err(format!("{key} schemas are unsupported"));
        }
    }

    // anyOf: recurse but reject structured unions
    if let Some(variants) = schema.get_mut("anyOf").and_then(Value::as_array_mut) {
        if variants.is_empty() {
            return Err("anyOf must contain at least one schema".into());
        }
        for variant in variants {
            // pi: structured = union member is an object/array schema
            let structured = match variant.get("type") {
                Some(Value::String(t)) => t == "object" || t == "array",
                Some(Value::Array(types)) => types
                    .iter()
                    .any(|t| t == "object" || t == "array"),
                _ => variant.get("properties").is_some() || variant.get("items").is_some(),
            };
            if structured {
                return Err("object and array unions are unsupported".into());
            }
            make_node_strict(variant)?;
        }
    }

    if let Some(items) = schema.get_mut("items") {
        if items.is_array() {
            return Err("tuple schemas are unsupported".into());
        }
        make_node_strict(items)?;
    }

    let is_object_schema = schema.get("type").and_then(Value::as_str) == Some("object");
    if schema.get("properties").is_some() && !is_object_schema {
        return Err("properties require type object".into());
    }
    if !is_object_schema {
        return Ok(());
    }
    match schema.get("additionalProperties") {
        Some(v) if v != &Value::Bool(false) => {
            return Err("schema-valued or true additionalProperties is unsupported".into())
        }
        _ => {}
    }
    if schema
        .get("required")
        .is_some_and(|r| !r.is_array() || r.as_array().unwrap().iter().any(|k| !k.is_string()))
    {
        return Err("object required must be a string array".into());
    }

    let property_names: Vec<String> = schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|p| p.keys().cloned().collect())
        .unwrap_or_default();
    let required: Vec<String> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|r| {
            r.iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();
    if required.iter().any(|key| !property_names.contains(key)) {
        return Err("required contains an unknown property".into());
    }

    if let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut) {
        for (key, property) in properties.iter_mut() {
            make_node_strict(property)?;
            if !required.contains(key) && !schema_allows_null(property) {
                *property = serde_json::json!({"anyOf": [property, {"type": "null"}]});
            }
        }
    }
    let obj = schema.as_object_mut().unwrap();
    obj.insert("required".into(), Value::Array(
        property_names.into_iter().map(Value::String).collect(),
    ));
    obj.insert("additionalProperties".into(), Value::Bool(false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn makes_properties_required_and_closes_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": ["path"],
        });
        let strict = make_strict_json_schema(&schema).unwrap();
        // property order follows the serde map; compare as sets
        let required = strict["required"].as_array().unwrap();
        let required: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(required.contains(&"path") && required.contains(&"limit"));
        assert_eq!(strict["additionalProperties"], json!(false));
        // optional non-null property wrapped in anyOf null
        assert_eq!(strict["properties"]["limit"]["anyOf"][1]["type"], "null");
        // required property untouched
        assert_eq!(strict["properties"]["path"]["type"], "string");
    }

    #[test]
    fn rejects_unsupported_keys() {
        let schema = json!({"type": "object", "properties": {}, "$ref": "#/x"});
        assert!(make_strict_json_schema(&schema).is_err());
        let schema = json!({"type": "object", "properties": {}, "allOf": []});
        assert!(make_strict_json_schema(&schema).is_err());
    }

    #[test]
    fn rejects_open_additional_properties() {
        let schema = json!({"type": "object", "properties": {}, "additionalProperties": true});
        assert!(make_strict_json_schema(&schema).is_err());
    }

    #[test]
    fn allows_null_optional_properties() {
        let schema = json!({
            "type": "object",
            "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        });
        let strict = make_strict_json_schema(&schema).unwrap();
        // already nullable: no double wrapping
        assert_eq!(strict["properties"]["x"]["anyOf"][0]["type"], "string");
    }

    #[test]
    fn resolve_falls_back_on_unsupported() {
        let schema = json!({"type": "object", "properties": {}, "not": {}});
        assert!(resolve_strict(&schema, true).is_none());
        assert!(resolve_strict(&schema, false).is_none());
    }
}
