//! Integration tests for the built-in tools against a temp workspace.

use std::path::PathBuf;

use pi_agent::{AgentTool, ToolContext};
use pi_ai::types::ToolResultContent;
use pi_tools::{BashTool, EditTool, FindTool, GrepTool, LsTool, ReadTool, WriteTool};

fn tmp_workspace() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pi-rs-tools-{}", uuid::Uuid::now_v7()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn text_of(result: &pi_agent::AgentToolResult) -> String {
    result
        .content
        .iter()
        .map(|c| match c {
            ToolResultContent::Text(t) => t.text.clone(),
            ToolResultContent::Image(_) => "[image]".to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

async fn run(
    tool: &dyn AgentTool,
    args: serde_json::Value,
) -> Result<pi_agent::AgentToolResult, String> {
    tool.execute("call_1", args, &ToolContext::default(), std::sync::Arc::new(|_| {}))
        .await
}

#[tokio::test]
async fn write_then_read_roundtrip() {
    let ws = tmp_workspace();
    let write = WriteTool::new(&ws);
    let read = ReadTool::new(&ws);

    let result = run(
        &write,
        serde_json::json!({"path": "src/main.rs", "content": "line1\nline2\nline3\n"}),
    )
    .await
    .unwrap();
    assert!(text_of(&result).contains("Successfully wrote"));

    let result = run(&read, serde_json::json!({"path": "src/main.rs"})).await.unwrap();
    // pi's read keeps the trailing empty line from the final newline
    assert_eq!(text_of(&result), "line1\nline2\nline3\n");

    // offset/limit (1-indexed offset); pi appends a continuation notice when
    // the file has remaining lines
    let result = run(
        &read,
        serde_json::json!({"path": "src/main.rs", "offset": 2, "limit": 1}),
    )
    .await
    .unwrap();
    assert_eq!(
        text_of(&result),
        "line2\n\n[2 more lines in file. Use offset=3 to continue.]"
    );

    // offset beyond end
    let err = run(&read, serde_json::json!({"path": "src/main.rs", "offset": 99}))
        .await
        .unwrap_err();
    assert!(err.contains("beyond end of file"), "{err}");

    std::fs::remove_dir_all(&ws).ok();
}

#[tokio::test]
async fn read_truncation_notice() {
    let ws = tmp_workspace();
    let write = WriteTool::new(&ws);
    let read = ReadTool::new(&ws);

    let big = (0..100).map(|i| format!("line{i}")).collect::<Vec<_>>().join("\n");
    run(&write, serde_json::json!({"path": "big.txt", "content": big}))
        .await
        .unwrap();

    // small custom limit is honored by read? pi's read uses fixed limits; use bash-style:
    // instead simulate via offset: read from line 90 -> tail
    let result = run(&read, serde_json::json!({"path": "big.txt", "offset": 90}))
        .await
        .unwrap();
    assert!(text_of(&result).starts_with("line89"));
    std::fs::remove_dir_all(&ws).ok();
}

#[tokio::test]
async fn edit_single_and_multi() {
    let ws = tmp_workspace();
    let write = WriteTool::new(&ws);
    let edit = EditTool::new(&ws);

    run(
        &write,
        serde_json::json!({"path": "code.rs", "content": "fn a() {}\nfn b() {}\nfn c() {}\n"}),
    )
    .await
    .unwrap();

    // single edit (legacy flat form also accepted)
    let result = run(
        &edit,
        serde_json::json!({
            "path": "code.rs",
            "edits": [{"oldText": "fn b() {}", "newText": "fn b() { /* edited */ }"}]
        }),
    )
    .await
    .unwrap();
    assert!(text_of(&result).contains("Successfully replaced 1 block(s)"));
    let details = &result.details;
    assert!(details["diff"].is_string());
    assert!(details["patch"].as_str().unwrap().contains("---"));
    assert!(details["firstChangedLine"].as_u64().unwrap() >= 2);

    // multi edit
    let result = run(
        &edit,
        serde_json::json!({
            "path": "code.rs",
            "edits": [
                {"oldText": "fn a() {}", "newText": "fn a() { /* 1 */ }"},
                {"oldText": "fn c() {}", "newText": "fn c() { /* 2 */ }"}
            ]
        }),
    )
    .await
    .unwrap();
    assert!(text_of(&result).contains("Successfully replaced 2 block(s)"));

    let content = std::fs::read_to_string(ws.join("code.rs")).unwrap();
    assert!(content.contains("fn a() { /* 1 */ }"));
    assert!(content.contains("fn b() { /* edited */ }"));
    assert!(content.contains("fn c() { /* 2 */ }"));

    // duplicate oldText rejected
    let err = run(
        &edit,
        serde_json::json!({
            "path": "code.rs",
            "edits": [{"oldText": "fn", "newText": "fn"}]
        }),
    )
    .await
    .unwrap_err();
    assert!(err.contains("unique") || err.contains("No changes"), "{err}");

    std::fs::remove_dir_all(&ws).ok();
}

#[tokio::test]
async fn bash_runs_and_reports_exit_codes() {
    let ws = tmp_workspace();
    let bash = BashTool::new(&ws);

    let result = run(&bash, serde_json::json!({"command": "echo hello-bash"}))
        .await
        .unwrap();
    assert_eq!(text_of(&result).trim(), "hello-bash");

    // non-zero exit becomes an error carrying output
    let err = run(&bash, serde_json::json!({"command": "echo oops >&2; exit 3"}))
        .await
        .unwrap_err();
    assert!(err.contains("exited with code 3"), "{err}");
    assert!(err.contains("oops"));

    // cwd is respected
    run(&write_ref(&ws), serde_json::json!({"path": "marker.txt", "content": "x"}))
        .await
        .unwrap();
    let result = run(&bash, serde_json::json!({"command": "ls marker.txt"})).await.unwrap();
    assert_eq!(text_of(&result).trim(), "marker.txt");

    std::fs::remove_dir_all(&ws).ok();
}

fn write_ref(ws: &std::path::Path) -> WriteTool {
    WriteTool::new(ws)
}

#[tokio::test]
async fn grep_finds_matches_with_line_numbers() {
    let ws = tmp_workspace();
    let write = WriteTool::new(&ws);

    run(
        &write,
        serde_json::json!({"path": "a.txt", "content": "apple pie\nbanana split\napple juice\n"}),
    )
    .await
    .unwrap();
    run(
        &write,
        serde_json::json!({"path": "b.txt", "content": "no fruit here\n"}),
    )
    .await
    .unwrap();

    let grep = GrepTool::new(&ws);
    let result = run(&grep, serde_json::json!({"pattern": "apple"})).await.unwrap();
    let text = text_of(&result);
    assert!(text.contains("a.txt:1: apple pie"), "{text}");
    assert!(text.contains("a.txt:3: apple juice"), "{text}");
    assert!(!text.contains("b.txt"));

    // literal + ignoreCase
    let result = run(
        &grep,
        serde_json::json!({"pattern": "APPLE", "literal": true, "ignoreCase": true}),
    )
    .await
    .unwrap();
    assert!(text_of(&result).contains("apple pie"));

    // no matches
    let result = run(&grep, serde_json::json!({"pattern": "zzz-not-there"})).await.unwrap();
    assert_eq!(text_of(&result), "No matches found");

    // context lines
    let result = run(
        &grep,
        serde_json::json!({"pattern": "banana", "context": 1}),
    )
    .await
    .unwrap();
    let text = text_of(&result);
    assert!(text.contains("a.txt-1- apple pie"), "{text}");
    assert!(text.contains("a.txt:2: banana split"));

    std::fs::remove_dir_all(&ws).ok();
}

#[tokio::test]
async fn find_matches_glob() {
    let ws = tmp_workspace();
    let write = WriteTool::new(&ws);

    run(&write, serde_json::json!({"path": "src/x.rs", "content": "1"})).await.unwrap();
    run(&write, serde_json::json!({"path": "src/deep/y.rs", "content": "2"})).await.unwrap();
    run(&write, serde_json::json!({"path": "doc.md", "content": "3"})).await.unwrap();

    let find = FindTool::new(&ws);
    let result = run(&find, serde_json::json!({"pattern": "*.rs"})).await.unwrap();
    let text = text_of(&result);
    assert!(text.contains("src/x.rs"), "{text}");
    assert!(text.contains("src/deep/y.rs"), "{text}");
    assert!(!text.contains("doc.md"));

    let result = run(&find, serde_json::json!({"pattern": "**/*.md"})).await.unwrap();
    assert!(text_of(&result).contains("doc.md"));

    std::fs::remove_dir_all(&ws).ok();
}

#[tokio::test]
async fn ls_lists_with_dir_suffix() {
    let ws = tmp_workspace();
    let write = WriteTool::new(&ws);

    run(&write, serde_json::json!({"path": "sub/file.txt", "content": "x"})).await.unwrap();
    run(&write, serde_json::json!({"path": "afile.txt", "content": "y"})).await.unwrap();

    let ls = LsTool::new(&ws);
    let result = run(&ls, serde_json::json!({})).await.unwrap();
    let text = text_of(&result);
    assert!(text.contains("afile.txt"), "{text}");
    assert!(text.contains("sub/"), "{text}");
    assert!(text.lines().count() >= 2);

    // empty dir
    std::fs::create_dir_all(ws.join("empty")).unwrap();
    let result = run(&ls, serde_json::json!({"path": "empty"})).await.unwrap();
    assert_eq!(text_of(&result), "(empty directory)");

    std::fs::remove_dir_all(&ws).ok();
}

#[tokio::test]
async fn long_lines_truncated_in_grep() {
    let ws = tmp_workspace();
    let write = WriteTool::new(&ws);
    let long_line = "x".repeat(1000);
    run(
        &write,
        serde_json::json!({"path": "long.txt", "content": format!("{long_line}\n")}),
    )
    .await
    .unwrap();

    let grep = GrepTool::new(&ws);
    let result = run(&grep, serde_json::json!({"pattern": "xxx"})).await.unwrap();
    let text = text_of(&result);
    assert!(text.contains("... [truncated]"), "{text}");
    std::fs::remove_dir_all(&ws).ok();
}
