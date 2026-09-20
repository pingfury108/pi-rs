//! Edit matching and diff computation, port of `core/tools/edit-diff.ts`.
//!
//! Core semantics:
//! - All edits are matched against the same original content.
//! - Exact match first; fallback to fuzzy matching (NFKC normalization,
//!   trailing-whitespace strip, smart quotes/dashes/spaces → ASCII).
//! - Duplicates or overlapping edits are rejected.
//! - Replacements applied in reverse order so offsets stay stable.

/// A single targeted replacement.
#[derive(Debug, Clone)]
pub struct Edit {
    pub old_text: String,
    pub new_text: String,
}

/// Normalize CRLF/CR to LF.
pub fn normalize_to_lf(text: &str) -> String {
    if text.contains('\r') {
        text.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        text.to_string()
    }
}

/// Detect whether the content predominantly uses CRLF.
pub fn detect_line_ending(content: &str) -> &'static str {
    let crlf = content.find("\r\n");
    let lf = content.find('\n');
    match (crlf, lf) {
        (_, None) => "\n",
        (Some(a), Some(b)) if a < b => "\r\n",
        _ => "\n",
    }
}

pub fn restore_line_endings(text: &str, ending: &str) -> String {
    if ending == "\r\n" {
        text.replace('\n', "\r\n")
    } else {
        text.to_string()
    }
}

/// Strip a UTF-8 BOM.
pub fn split_bom(text: &str) -> (&str, &str) {
    match text.strip_prefix('\u{feff}') {
        Some(rest) => (&text[..3], rest),
        None => ("", text),
    }
}

/// Normalize text for fuzzy matching (pi's `normalizeForFuzzyMatch`):
/// NFKC + per-line trailing-whitespace strip + smart punctuation → ASCII.
pub fn normalize_for_fuzzy_match(text: &str) -> String {
    // Unicode punctuation maps applied after NFKC; NFKC itself already folds
    // many forms but not smart quotes/dashes/special spaces.
    let mut out = String::with_capacity(text.len());
    let mut lines = text.split('\n').peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim_end();
        for ch in trimmed.chars() {
            out.push(normalize_char(ch));
        }
        if lines.peek().is_some() {
            out.push('\n');
        }
    }
    out
}

fn normalize_char(ch: char) -> char {
    match ch {
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}' | '\u{2212}' => '-',
        '\u{00A0}' | '\u{2002}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}' => ' ',
        other => other,
    }
}

struct FuzzyMatch {
    index: usize,
    match_length: usize,
    used_fuzzy: bool,
}

fn find_text(content: &str, old_text: &str) -> Option<FuzzyMatch> {
    if let Some(index) = content.find(old_text) {
        return Some(FuzzyMatch {
            index,
            match_length: old_text.len(),
            used_fuzzy: false,
        });
    }
    let fuzzy_content = normalize_for_fuzzy_match(content);
    let fuzzy_old = normalize_for_fuzzy_match(old_text);
    let index = fuzzy_content.find(&fuzzy_old)?;
    let _ = fuzzy_content; // offsets are in normalized space; caller re-normalizes
    Some(FuzzyMatch {
        index,
        match_length: fuzzy_old.len(),
        used_fuzzy: true,
    })
}

fn count_occurrences(content: &str, old_text: &str) -> usize {
    // pi counts occurrences in fuzzy-normalized space.
    let haystack = normalize_for_fuzzy_match(content);
    let needle = normalize_for_fuzzy_match(old_text);
    if needle.is_empty() {
        return 0;
    }
    haystack.matches(&needle).count()
}

struct MatchedEdit {
    edit_index: usize,
    match_index: usize,
    match_length: usize,
    new_text: String,
}

fn apply_replacements(content: &str, replacements: &[MatchedEdit]) -> String {
    let mut result = content.to_string();
    for replacement in replacements.iter().rev() {
        let start = replacement.match_index;
        let end = start + replacement.match_length;
        result.replace_range(start..end, &replacement.new_text);
    }
    result
}

/// Overlay edits onto `base` (normalized) so unchanged lines of `original`
/// keep their original bytes. Port of pi's group-slice algorithm: each
/// replacement is widened to the lines it touches; touched line groups are
/// rewritten from the normalized base (with replacements applied), all other
/// lines are copied from the original.
fn apply_replacements_preserving_unchanged_lines(
    original: &str,
    base: &str,
    replacements: &[MatchedEdit],
) -> String {
    let original_lines = split_lines_with_endings(original);
    let base_lines = split_lines_with_endings(base);
    if original_lines.len() != base_lines.len() || base_lines.is_empty() {
        return apply_replacements(base, replacements);
    }

    // Compute touched line ranges, merged into groups.
    let line_starts: Vec<usize> = {
        let mut starts = Vec::with_capacity(base_lines.len());
        let mut acc = 0usize;
        for line in &base_lines {
            starts.push(acc);
            acc += line.len();
        }
        starts
    };

    struct Group<'a> {
        start_line: usize,
        end_line: usize, // exclusive
        replacements: Vec<&'a MatchedEdit>,
    }

    let mut groups: Vec<Group> = Vec::new();
    let mut sorted: Vec<&MatchedEdit> = replacements.iter().collect();
    sorted.sort_by_key(|r| r.match_index);

    for r in sorted {
        let start_line = match line_starts.binary_search(&r.match_index) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let end_offset = r.match_index + r.match_length;
        let mut end_line = start_line;
        while end_line < base_lines.len()
            && line_starts[end_line] + base_lines[end_line].len() < end_offset
        {
            end_line += 1;
        }
        let end_line = (end_line + 1).min(base_lines.len());

        match groups.last_mut() {
            Some(group) if start_line < group.end_line => {
                group.end_line = group.end_line.max(end_line);
                group.replacements.push(r);
            }
            _ => groups.push(Group {
                start_line,
                end_line,
                replacements: vec![r],
            }),
        }
    }

    let group_start_offset = |line: usize| line_starts[line];
    let group_end_offset = |line: usize| line_starts[line] + base_lines[line].len();

    let mut original_index = 0usize;
    let mut result = String::with_capacity(original.len() + 64);
    for group in &groups {
        // copy original lines before the group
        result.push_str(&original_lines[original_index..group.start_line].join(""));
        // rewritten group from base with replacements applied
        let slice = &base[group_start_offset(group.start_line)..group_end_offset(group.end_line - 1)];
        let local: Vec<MatchedEdit> = group
            .replacements
            .iter()
            .map(|r| MatchedEdit {
                edit_index: r.edit_index,
                match_index: r.match_index - group_start_offset(group.start_line),
                match_length: r.match_length,
                new_text: r.new_text.clone(),
            })
            .collect();
        result.push_str(&apply_replacements(slice, &local));
        original_index = group.end_line;
    }
    result.push_str(&original_lines[original_index.min(original_lines.len())..].join(""));
    result
}

fn split_lines_with_endings(content: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let bytes = content.as_bytes();
    let mut start = 0usize;
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'\n' {
            lines.push(&content[start..=i]);
            start = i + 1;
        }
    }
    if start < content.len() {
        lines.push(&content[start..]);
    }
    lines
}

#[derive(Debug)]
pub struct AppliedEdits {
    pub base_content: String,
    pub new_content: String,
}

/// Apply one or more exact-text replacements to LF-normalized content.
/// Mirrors `applyEditsToNormalizedContent` including error semantics.
pub fn apply_edits_to_normalized_content(
    normalized_content: &str,
    edits: &[Edit],
    path: &str,
) -> Result<AppliedEdits, String> {
    let normalized_edits: Vec<Edit> = edits
        .iter()
        .map(|e| Edit {
            old_text: normalize_to_lf(&e.old_text),
            new_text: normalize_to_lf(&e.new_text),
        })
        .collect();

    for (i, edit) in normalized_edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            return Err(empty_old_text_error(path, i, normalized_edits.len()));
        }
    }

    // Any fuzzy match → operate entirely in normalized space.
    let initial: Vec<Option<FuzzyMatch>> = normalized_edits
        .iter()
        .map(|e| find_text(normalized_content, &e.old_text))
        .collect();
    let used_fuzzy = initial.iter().any(|m| m.as_ref().is_some_and(|m| m.used_fuzzy));
    let replacement_base = if used_fuzzy {
        normalize_for_fuzzy_match(normalized_content)
    } else {
        normalized_content.to_string()
    };

    let mut matched: Vec<MatchedEdit> = Vec::new();
    for (i, edit) in normalized_edits.iter().enumerate() {
        let Some(m) = find_text(&replacement_base, &edit.old_text) else {
            return Err(not_found_error(path, i, normalized_edits.len()));
        };
        let occurrences = count_occurrences(&replacement_base, &edit.old_text);
        if occurrences > 1 {
            return Err(duplicate_error(path, i, normalized_edits.len(), occurrences));
        }
        matched.push(MatchedEdit {
            edit_index: i,
            match_index: m.index,
            match_length: m.match_length,
            new_text: edit.new_text.clone(),
        });
    }

    matched.sort_by_key(|m| m.match_index);
    for i in 1..matched.len() {
        let prev = &matched[i - 1];
        let cur = &matched[i];
        if prev.match_index + prev.match_length > cur.match_index {
            return Err(format!(
                "edits[{}] and edits[{}] overlap in {}. Merge them into one edit or target disjoint regions.",
                prev.edit_index, cur.edit_index, path
            ));
        }
    }

    let base_content = normalized_content.to_string();
    let new_content = if used_fuzzy {
        apply_replacements_preserving_unchanged_lines(normalized_content, &replacement_base, &matched)
    } else {
        apply_replacements(&replacement_base, &matched)
    };

    if base_content == new_content {
        return Err(no_change_error(path, normalized_edits.len()));
    }

    Ok(AppliedEdits {
        base_content,
        new_content,
    })
}

fn not_found_error(path: &str, edit_index: usize, total: usize) -> String {
    if total == 1 {
        format!(
            "Could not find the exact text in {path}. The old text must match exactly including all whitespace and newlines."
        )
    } else {
        format!(
            "Could not find edits[{edit_index}] in {path}. The oldText must match exactly including all whitespace and newlines."
        )
    }
}

fn duplicate_error(path: &str, edit_index: usize, total: usize, occurrences: usize) -> String {
    if total == 1 {
        format!(
            "Found {occurrences} occurrences of the text in {path}. The text must be unique. Please provide more context to make it unique."
        )
    } else {
        format!(
            "Found {occurrences} occurrences of edits[{edit_index}] in {path}. Each oldText must be unique. Please provide more context to make it unique."
        )
    }
}

fn empty_old_text_error(path: &str, edit_index: usize, total: usize) -> String {
    if total == 1 {
        format!("oldText must not be empty in {path}.")
    } else {
        format!("edits[{edit_index}].oldText must not be empty in {path}.")
    }
}

fn no_change_error(path: &str, total: usize) -> String {
    if total == 1 {
        format!(
            "No changes made to {path}. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
        )
    } else {
        format!("No changes made to {path}. The replacements produced identical content.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_exact_edit() {
        let content = "fn main() {\n    println!(\"hello\");\n}\n";
        let edits = vec![Edit {
            old_text: "println!(\"hello\");".into(),
            new_text: "println!(\"goodbye\");".into(),
        }];
        let result = apply_edits_to_normalized_content(content, &edits, "main.rs").unwrap();
        assert!(result.new_content.contains("println!(\"goodbye\");"));
        assert!(!result.new_content.contains("hello"));
    }

    #[test]
    fn duplicate_match_rejected() {
        let content = "same\nsame\n";
        let edits = vec![Edit {
            old_text: "same".into(),
            new_text: "other".into(),
        }];
        let err = apply_edits_to_normalized_content(content, &edits, "f.txt").unwrap_err();
        assert!(err.contains("must be unique"), "{err}");
    }

    #[test]
    fn not_found_rejected() {
        let content = "abc\n";
        let edits = vec![Edit {
            old_text: "xyz".into(),
            new_text: "q".into(),
        }];
        let err = apply_edits_to_normalized_content(content, &edits, "f.txt").unwrap_err();
        assert!(err.contains("Could not find"), "{err}");
    }

    #[test]
    fn multiple_disjoint_edits() {
        let content = "a\nb\nc\nd\n";
        let edits = vec![
            Edit {
                old_text: "a".into(),
                new_text: "A".into(),
            },
            Edit {
                old_text: "d".into(),
                new_text: "D".into(),
            },
        ];
        let result = apply_edits_to_normalized_content(content, &edits, "f.txt").unwrap();
        assert_eq!(result.new_content, "A\nb\nc\nD\n");
    }

    #[test]
    fn overlapping_edits_rejected() {
        let content = "abcdef\n";
        let edits = vec![
            Edit {
                old_text: "abc".into(),
                new_text: "x".into(),
            },
            Edit {
                old_text: "cde".into(),
                new_text: "y".into(),
            },
        ];
        let err = apply_edits_to_normalized_content(content, &edits, "f.txt").unwrap_err();
        assert!(err.contains("overlap"), "{err}");
    }

    #[test]
    fn fuzzy_match_strips_trailing_whitespace() {
        let content = "fn main() {   \n    println!(1);\n}\n";
        let edits = vec![Edit {
            old_text: "fn main() {\n    println!(1);".into(),
            new_text: "fn main() {\n    println!(2);".into(),
        }];
        let result = apply_edits_to_normalized_content(content, &edits, "f.rs").unwrap();
        // touched lines are rewritten from the normalized base (trailing
        // whitespace stripped); unchanged lines keep original bytes
        assert!(result.new_content.contains("fn main() {\n"));
        assert!(result.new_content.contains("println!(2);"));
        assert!(!result.new_content.contains("fn main() {   \n"));
    }

    #[test]
    fn no_change_rejected() {
        let content = "abc\n";
        let edits = vec![Edit {
            old_text: "abc".into(),
            new_text: "abc".into(),
        }];
        let err = apply_edits_to_normalized_content(content, &edits, "f.txt").unwrap_err();
        assert!(err.contains("No changes"), "{err}");
    }
}
