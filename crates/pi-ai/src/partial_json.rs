//! Incremental JSON parsing for streaming tool-call arguments.
//!
//! Port of the behavior in `packages/ai/src/utils/json-parse.ts`
//! (`parseStreamingJson`): parse a possibly-incomplete JSON string, always
//! returning a usable `Value` (empty object as last resort).

use serde_json::Value;

/// Parse a possibly-incomplete JSON object from a stream.
pub fn parse_partial_json(input: &str) -> Value {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Value::Object(Default::default());
    }
    if let Ok(v) = serde_json::from_str(trimmed) {
        return v;
    }
    let repaired = repair(trimmed);
    serde_json::from_str(&repaired).unwrap_or_else(|_| Value::Object(Default::default()))
}

/// Close unclosed strings/containers and strip dangling partial tokens.
fn repair(input: &str) -> String {
    let (stack, in_string, escape) = scan(input);
    let mut out = String::from(input);

    if in_string {
        // A trailing backslash opens an escape with no follower: drop it.
        if escape {
            debug_assert!(out.ends_with('\\'));
            out.pop();
        }
        out.push('"');
    }

    loop {
        let trimmed = out.trim_end();
        if trimmed.ends_with(':') {
            // Dangling key with no value: insert null.
            out = format!("{trimmed} null");
            break;
        }
        if trimmed.ends_with(',') {
            // Dangling separator: drop it, then re-check.
            out = trimmed.trim_end_matches(',').to_string();
            continue;
        }
        if let Some(cut) = partial_token_start(trimmed) {
            // Partial literal (tru/fals/nu) or incomplete number: strip it.
            out = trimmed[..cut].to_string();
            continue;
        }
        break;
    }

    for c in stack.iter().rev() {
        out.push(match c {
            '{' => '}',
            _ => ']',
        });
    }
    out
}

/// Single pass: track string state and the stack of open containers.
fn scan(input: &str) -> (Vec<char>, bool, bool) {
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escape = false;
    for c in input.chars() {
        if escape {
            escape = false;
            continue;
        }
        match c {
            '"' if in_string => in_string = false,
            '"' => in_string = true,
            '\\' if in_string => escape = true,
            '{' | '[' if !in_string => stack.push(c),
            '}' | ']' if !in_string => {
                stack.pop();
            }
            _ => {}
        }
    }
    (stack, in_string, escape)
}

/// If the text ends with an incomplete literal or number, return the index
/// where that token starts.
fn partial_token_start(s: &str) -> Option<usize> {
    let cut = s
        .rfind(|c| matches!(c, '{' | '[' | ',' | ':'))
        .map(|i| i + 1)
        .unwrap_or(0);
    let token = s[cut..].trim();
    if token.is_empty() {
        return None;
    }
    // Partial true/false/null.
    if ["true", "false", "null"]
        .iter()
        .any(|lit| lit.starts_with(token) && *lit != token)
    {
        return Some(cut);
    }
    // Incomplete number such as `12.`, `1e`, `1e+`, `-`.
    if token.bytes().all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-'))
        && token.bytes().any(|b| b.is_ascii_digit() || matches!(b, b'-' | b'.'))
        && token.ends_with(['.', 'e', 'E', '+', '-'])
    {
        return Some(cut);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn complete_json_passes_through() {
        assert_eq!(parse_partial_json(r#"{"a": 1}"#), json!({"a": 1}));
    }

    #[test]
    fn empty_input_yields_empty_object() {
        assert_eq!(parse_partial_json(""), json!({}));
        assert_eq!(parse_partial_json("   "), json!({}));
    }

    #[test]
    fn unclosed_object() {
        assert_eq!(parse_partial_json(r#"{"a": 1"#), json!({"a": 1}));
        assert_eq!(parse_partial_json(r#"{"a""#), json!({}));
    }

    #[test]
    fn unterminated_string_value() {
        assert_eq!(
            parse_partial_json(r#"{"path": "src/ma"#),
            json!({"path": "src/ma"})
        );
    }

    #[test]
    fn dangling_key() {
        assert_eq!(parse_partial_json(r#"{"path":"#), json!({"path": null}));
        assert_eq!(parse_partial_json(r#"{"a": 1, "b": "#), json!({"a": 1, "b": null}));
    }

    #[test]
    fn dangling_comma() {
        assert_eq!(parse_partial_json(r#"{"a": 1,"#), json!({"a": 1}));
        assert_eq!(parse_partial_json("[1, 2,"), json!([1, 2]));
    }

    #[test]
    fn partial_literals() {
        assert_eq!(parse_partial_json(r#"{"flag": tru"#), json!({"flag": null}));
        assert_eq!(parse_partial_json(r#"{"x": nul"#), json!({"x": null}));
        assert_eq!(parse_partial_json(r#"{"y": fals"#), json!({"y": null}));
    }

    #[test]
    fn incomplete_numbers() {
        assert_eq!(parse_partial_json(r#"{"n": 12."#), json!({"n": null}));
        assert_eq!(parse_partial_json(r#"{"n": 1e"#), json!({"n": null}));
        // complete numbers are kept
        assert_eq!(parse_partial_json(r#"{"n": 12"#), json!({"n": 12}));
    }

    #[test]
    fn nested_containers() {
        assert_eq!(
            parse_partial_json(r#"{"a": [1, {"b": "x""#),
            json!({"a": [1, {"b": "x"}]})
        );
    }

    #[test]
    fn escaped_quotes() {
        assert_eq!(
            parse_partial_json(r#"{"a": "say \"hi"#),
            json!({"a": "say \"hi"})
        );
        // trailing escaped backslash inside string
        assert_eq!(parse_partial_json(r#"{"a": "x\"#), json!({"a": "x"}));
        // legitimate double backslash stays
        assert_eq!(parse_partial_json(r#"{"a": "x\\"#), json!({"a": "x\\"}));
    }

    #[test]
    fn tool_args_streaming_scenario() {
        // typical accumulation across deltas
        let chunks = [r#"{"#, r#""path": "#, r#""README"#, r#".md", "lim"#, r#"it": 100}"#];
        let mut acc = String::new();
        let mut last = Value::Object(Default::default());
        for c in chunks {
            acc.push_str(c);
            last = parse_partial_json(&acc);
        }
        assert_eq!(
            last,
            json!({"path": "README.md", "limit": 100})
        );
    }
}
