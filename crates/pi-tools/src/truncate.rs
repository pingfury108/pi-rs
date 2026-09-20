//! Shared truncation utilities, port of `core/tools/truncate.ts`.
//!
//! Two independent limits - whichever is hit first wins:
//! - Line limit (default: 2000 lines)
//! - Byte limit (default: 50KB)
//! Never returns partial lines (except bash tail-truncation edge case).

/// Max chars per grep match line.
pub const GREP_MAX_LINE_LENGTH: usize = 500;
pub const DEFAULT_MAX_LINES: usize = 2000;
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;

/// Result of a truncation pass.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TruncationResult {
    pub content: String,
    pub truncated: bool,
    /// Which limit was hit: "lines" | "bytes" | null.
    pub truncated_by: Option<&'static str>,
    pub total_lines: usize,
    pub total_bytes: usize,
    pub output_lines: usize,
    pub output_bytes: usize,
    /// Only for tail truncation edge case.
    pub last_line_partial: bool,
    /// For head truncation when the first line alone exceeds the byte limit.
    pub first_line_exceeds_limit: bool,
    pub max_lines: usize,
    pub max_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TruncationOptions {
    pub max_lines: usize,
    pub max_bytes: usize,
}

impl Default for TruncationOptions {
    fn default() -> Self {
        Self {
            max_lines: DEFAULT_MAX_LINES,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

/// Format bytes as human-readable size (pi's `formatSize`).
pub fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn split_lines_for_counting(content: &str) -> Vec<&str> {
    if content.is_empty() {
        return vec![];
    }
    let mut lines: Vec<&str> = content.split('\n').collect();
    if content.ends_with('\n') {
        lines.pop();
    }
    lines
}

fn no_truncation(content: &str, options: TruncationOptions) -> TruncationResult {
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();
    let total_bytes = content.len();
    TruncationResult {
        content: content.to_string(),
        truncated: false,
        truncated_by: None,
        total_lines,
        total_bytes,
        output_lines: total_lines,
        output_bytes: total_bytes,
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines: options.max_lines,
        max_bytes: options.max_bytes,
    }
}

/// Truncate from the head (keep first N lines/bytes). Never partial lines.
pub fn truncate_head(content: &str, options: impl Into<Option<TruncationOptions>>) -> TruncationResult {
    let options = options.into().unwrap_or_default();
    let total_bytes = content.len();
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();

    if total_lines <= options.max_lines && total_bytes <= options.max_bytes {
        return no_truncation(content, options);
    }

    // First line alone exceeds byte limit.
    if lines[0].len() > options.max_bytes {
        return TruncationResult {
            content: String::new(),
            truncated: true,
            truncated_by: Some("bytes"),
            total_lines,
            total_bytes,
            output_lines: 0,
            output_bytes: 0,
            last_line_partial: false,
            first_line_exceeds_limit: true,
            max_lines: options.max_lines,
            max_bytes: options.max_bytes,
        };
    }

    let mut output: Vec<&str> = Vec::new();
    let mut output_bytes = 0usize;
    let mut truncated_by = "lines";

    for (i, line) in lines.iter().enumerate().take(options.max_lines) {
        let line_bytes = line.len() + usize::from(i > 0);
        if output_bytes + line_bytes > options.max_bytes {
            truncated_by = "bytes";
            break;
        }
        output_bytes += line_bytes;
        output.push(line);
    }

    if output.len() >= options.max_lines && output_bytes <= options.max_bytes {
        truncated_by = "lines";
    }

    let output_content = output.join("\n");
    let final_bytes = output_content.len();
    TruncationResult {
        content: output_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: output.len(),
        output_bytes: final_bytes,
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines: options.max_lines,
        max_bytes: options.max_bytes,
    }
}

/// Truncate from the tail (keep last N lines/bytes). May return a partial
/// first line if the last line of the original content exceeds the byte limit.
pub fn truncate_tail(content: &str, options: impl Into<Option<TruncationOptions>>) -> TruncationResult {
    let options = options.into().unwrap_or_default();
    let total_bytes = content.len();
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();

    if total_lines <= options.max_lines && total_bytes <= options.max_bytes {
        return no_truncation(content, options);
    }

    let mut output: Vec<String> = Vec::new();
    let mut output_bytes = 0usize;
    let mut truncated_by = "lines";
    let mut last_line_partial = false;

    for line in lines.iter().rev() {
        if output.len() >= options.max_lines {
            break;
        }
        let line_bytes = line.len() + usize::from(!output.is_empty());
        if output_bytes + line_bytes > options.max_bytes {
            truncated_by = "bytes";
            if output.is_empty() {
                // Edge case: single line longer than the byte limit; keep its end.
                let truncated_line = truncate_bytes_from_end(line, options.max_bytes);
                output_bytes = truncated_line.len();
                output.insert(0, truncated_line);
                last_line_partial = true;
            }
            break;
        }
        output_bytes += line_bytes;
        output.insert(0, (*line).to_string());
    }

    if output.len() >= options.max_lines && output_bytes <= options.max_bytes {
        truncated_by = "lines";
    }

    let output_content = output.join("\n");
    let final_bytes = output_content.len();
    TruncationResult {
        content: output_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: output.len(),
        output_bytes: final_bytes,
        last_line_partial,
        first_line_exceeds_limit: false,
        max_lines: options.max_lines,
        max_bytes: options.max_bytes,
    }
}

/// Truncate a string to fit within a byte limit, keeping the end and
/// respecting UTF-8 boundaries.
fn truncate_bytes_from_end(s: &str, max_bytes: usize) -> String {
    let buf = s.as_bytes();
    if buf.len() <= max_bytes {
        return s.to_string();
    }
    let mut start = buf.len() - max_bytes;
    while start < buf.len() && (buf[start] & 0xc0) == 0x80 {
        start += 1;
    }
    String::from_utf8_lossy(&buf[start..]).into_owned()
}

/// Truncate a single line to max chars with a `[truncated]` suffix.
pub fn truncate_line(line: &str, max_chars: impl Into<Option<usize>>) -> (String, bool) {
    let max_chars = max_chars.into().unwrap_or(GREP_MAX_LINE_LENGTH);
    if line.chars().count() <= max_chars {
        return (line.to_string(), false);
    }
    let cut: String = line.chars().take(max_chars).collect();
    (format!("{cut}... [truncated]"), true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_no_truncation() {
        let r = truncate_head("a\nb\nc", None);
        assert!(!r.truncated);
        assert_eq!(r.total_lines, 3);
        assert_eq!(r.content, "a\nb\nc");
    }

    #[test]
    fn head_by_lines() {
        let content = (0..10).map(|i| i.to_string()).collect::<Vec<_>>().join("\n");
        let r = truncate_head(&content, TruncationOptions { max_lines: 5, max_bytes: 1000 });
        assert!(r.truncated);
        assert_eq!(r.truncated_by, Some("lines"));
        assert_eq!(r.output_lines, 5);
        assert_eq!(r.content, "0\n1\n2\n3\n4");
    }

    #[test]
    fn head_by_bytes() {
        // "bbbb" costs 5 bytes with its newline (4+1): 4+5 > 8 stops after one line
        let content = "aaaa\nbbbb\ncccc";
        let r = truncate_head(content, TruncationOptions { max_lines: 100, max_bytes: 8 });
        assert!(r.truncated);
        assert_eq!(r.truncated_by, Some("bytes"));
        assert_eq!(r.output_lines, 1);
        assert_eq!(r.content, "aaaa");
    }

    #[test]
    fn head_first_line_exceeds() {
        let r = truncate_head("long-line-exceeds\nshort", TruncationOptions { max_lines: 100, max_bytes: 5 });
        assert!(r.first_line_exceeds_limit);
        assert_eq!(r.content, "");
    }

    #[test]
    fn tail_keeps_end() {
        let content = (0..10).map(|i| i.to_string()).collect::<Vec<_>>().join("\n");
        let r = truncate_tail(&content, TruncationOptions { max_lines: 3, max_bytes: 1000 });
        assert!(r.truncated);
        assert_eq!(r.truncated_by, Some("lines"));
        assert_eq!(r.content, "7\n8\n9");
    }

    #[test]
    fn tail_last_line_partial() {
        let content = "x\n0123456789";
        let r = truncate_tail(content, TruncationOptions { max_lines: 100, max_bytes: 6 });
        assert!(r.last_line_partial);
        assert_eq!(r.content, "456789");
    }

    #[test]
    fn line_truncation() {
        let (text, was) = truncate_line("short", None);
        assert_eq!(text, "short");
        assert!(!was);
        let (text, was) = truncate_line(&"x".repeat(600), None);
        assert!(was);
        assert!(text.ends_with("... [truncated]"));
    }

    #[test]
    fn format_sizes() {
        assert_eq!(format_size(500), "500B");
        assert_eq!(format_size(2048), "2.0KB");
        assert_eq!(format_size(3 * 1024 * 1024), "3.0MB");
    }
}
