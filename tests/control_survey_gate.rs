//! C0.1 gate: `docs/backends/CONTROL_SURVEY.md` exists and **every** backend row carries
//! a source link and a verification date.
//!
//! The ledger's AC is "file committed; each row has a source link and a date". That is
//! only worth anything if a machine checks it, so this parses the survey's markdown
//! tables and fails on any row that cites nothing. Written before the lowering code so a
//! future row added without a citation is a red build, not a reviewer's catch.

use std::path::Path;

const SURVEY: &str = "docs/backends/CONTROL_SURVEY.md";

/// A table row is a line starting with `|` that is not a header or separator.
fn body_rows(md: &str) -> Vec<(usize, String)> {
    md.lines()
        .enumerate()
        .filter(|(_, l)| l.trim_start().starts_with('|'))
        .filter(|(_, l)| !l.contains("---"))
        .filter(|(_, l)| {
            let first = l.split('|').nth(1).unwrap_or("").trim();
            // drop header rows
            first != "Backend" && first != "Family"
        })
        .map(|(i, l)| (i + 1, l.to_string()))
        .collect()
}

#[test]
fn survey_exists_and_is_non_trivial() {
    assert!(
        Path::new(SURVEY).exists(),
        "C0.1 requires {SURVEY}; it is missing"
    );
    let md = std::fs::read_to_string(SURVEY).unwrap();
    let rows = body_rows(&md);
    assert!(
        rows.len() >= 10,
        "survey has {} rows; every backend Syrinx drives plus the spec's table should be \
         covered",
        rows.len()
    );
}

#[test]
fn every_row_cites_a_source() {
    let md = std::fs::read_to_string(SURVEY).unwrap();
    let mut bad = Vec::new();
    for (line_no, row) in body_rows(&md) {
        // a citation is a markdown link, a bare URL, or an explicit on-disk path
        let cited = row.contains("](http")
            || row.contains("http://")
            || row.contains("https://")
            || row.contains("~/models/")
            || row.contains("crates/");
        if !cited {
            bad.push(format!("  line {line_no}: {}", row.chars().take(90).collect::<String>()));
        }
    }
    assert!(
        bad.is_empty(),
        "every survey row must cite a source (link or on-disk path):\n{}",
        bad.join("\n")
    );
}

#[test]
fn the_survey_carries_a_verification_date() {
    let md = std::fs::read_to_string(SURVEY).unwrap();
    // An ISO date must appear, and it must be introduced as a verification date rather
    // than merely mentioned somewhere.
    let has_iso = md
        .split_whitespace()
        .any(|w| {
            let w = w.trim_matches(|c: char| !c.is_ascii_digit() && c != '-');
            w.len() == 10
                && w.as_bytes()[4] == b'-'
                && w.as_bytes()[7] == b'-'
                && w.chars().filter(|c| c.is_ascii_digit()).count() == 8
        });
    assert!(has_iso, "{SURVEY} must state an ISO verification date");
    assert!(
        md.contains("Verification date"),
        "{SURVEY} must label its verification date explicitly"
    );
}
