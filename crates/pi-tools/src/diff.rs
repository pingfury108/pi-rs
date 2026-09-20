//! Diff generation for the edit tool, port of the diff parts of
//! `core/tools/edit-diff.ts` (display diff + unified patch).

/// Generate a display-oriented diff string with line numbers and context.
/// Returns the diff and the first changed line number (in the new file).
pub fn generate_diff_string(
    old_content: &str,
    new_content: &str,
    context_lines: usize,
) -> (String, Option<usize>) {
    let old_lines = split_for_diff(old_content);
    let new_lines = split_for_diff(new_content);

    // Line-level diff opcodes.
    let ops = similar::TextDiff::configure()
        .newline_terminated(false)
        .diff_lines(old_content, new_content)
        .ops()
        .to_vec();

    let max_line_num = old_lines.len().max(new_lines.len());
    let width = max_line_num.to_string().len();
    let mut output: Vec<String> = Vec::new();
    let mut old_line_num = 1usize;
    let mut new_line_num = 1usize;
    let mut last_was_change = false;
    let mut first_changed_line: Option<usize> = None;

    for (i, op) in ops.iter().enumerate() {
        let (tag, old_range, new_range) = (op.tag(), op.old_range(), op.new_range());
        let is_change = tag != similar::DiffTag::Equal;

        if is_change {
            if first_changed_line.is_none() {
                first_changed_line = Some(new_line_num);
            }
            // removed lines
            for old_index in old_range.clone() {
                let num = pad(old_line_num, width);
                output.push(format!("-{num} {}", old_lines[old_index]));
                old_line_num += 1;
            }
            // added lines
            for new_index in new_range.clone() {
                let num = pad(new_line_num, width);
                output.push(format!("+{num} {}", new_lines[new_index]));
                new_line_num += 1;
            }
            last_was_change = true;
        } else {
            let raw: Vec<&str> = old_range.clone().map(|i| old_lines[i]).collect();
            let next_is_change = ops
                .get(i + 1)
                .is_some_and(|next| next.tag() != similar::DiffTag::Equal);
            let has_leading = last_was_change;
            let has_trailing = next_is_change;

            if has_leading && has_trailing {
                if raw.len() <= context_lines * 2 {
                    for line in raw {
                        let num = pad(old_line_num, width);
                        output.push(format!(" {num} {line}"));
                        old_line_num += 1;
                        new_line_num += 1;
                    }
                } else {
                    for line in raw.iter().take(context_lines) {
                        let num = pad(old_line_num, width);
                        output.push(format!(" {num} {line}"));
                        old_line_num += 1;
                        new_line_num += 1;
                    }
                    let skipped = raw.len() - context_lines * 2;
                    output.push(format!(" {} ...", " ".repeat(width)));
                    old_line_num += skipped;
                    new_line_num += skipped;
                    for line in raw.iter().skip(raw.len() - context_lines) {
                        let num = pad(old_line_num, width);
                        output.push(format!(" {num} {line}"));
                        old_line_num += 1;
                        new_line_num += 1;
                    }
                }
            } else if has_leading {
                let shown = context_lines.min(raw.len());
                for line in raw.iter().take(shown) {
                    let num = pad(old_line_num, width);
                    output.push(format!(" {num} {line}"));
                    old_line_num += 1;
                    new_line_num += 1;
                }
                let skipped = raw.len() - shown;
                if skipped > 0 {
                    output.push(format!(" {} ...", " ".repeat(width)));
                    old_line_num += skipped;
                    new_line_num += skipped;
                }
            } else if has_trailing {
                let skipped = raw.len().saturating_sub(context_lines);
                if skipped > 0 {
                    output.push(format!(" {} ...", " ".repeat(width)));
                    old_line_num += skipped;
                    new_line_num += skipped;
                }
                for line in raw.iter().skip(skipped) {
                    let num = pad(old_line_num, width);
                    output.push(format!(" {num} {line}"));
                    old_line_num += 1;
                    new_line_num += 1;
                }
            } else {
                old_line_num += raw.len();
                new_line_num += raw.len();
            }
            last_was_change = false;
        }
    }

    (output.join("\n"), first_changed_line)
}

/// Generate a standard unified patch (context = 4, matching pi).
pub fn generate_unified_patch(path: &str, old_content: &str, new_content: &str) -> String {
    similar::TextDiff::configure()
        .diff_lines(old_content, new_content)
        .unified_diff()
        .context_radius(4)
        .header(path, path)
        .to_string()
}

fn pad(num: usize, width: usize) -> String {
    let s = num.to_string();
    format!("{:>width$}", s, width = width)
}

fn split_for_diff(content: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = content.split('\n').collect();
    if content.ends_with('\n') {
        lines.pop();
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shows_changes_with_context() {
        let old = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\n";
        let mut new_lines: Vec<&str> = old.lines().collect();
        new_lines[7] = "H";
        let new = format!("{}\n", new_lines.join("\n"));

        let (diff, first_changed) = generate_diff_string(old, &new, 2);
        assert_eq!(first_changed, Some(8));
        assert!(diff.contains("- 8 h"));
        assert!(diff.contains("+ 8 H"));
        assert!(diff.contains("..."), "long context should be elided");
    }
}
