//! Frozen RED tests for T-01.02 — numeric verbalization over normalized text.
//!
//! These pin criteria C1 (USD currency + singular/plural boundary), C2 (date vs
//! decimal disambiguation), C3 (ordinal suffix selection st/nd/rd/th), and C4
//! (bare cardinal + non-numeric passthrough) against the real public API the
//! green phase must build:
//!
//!   * `syrinx_frontend::expand::expand_numbers(&str) -> String`
//!
//! Contract (DESIGN / list.md): take a `&str` that may contain currency, dates,
//! decimals, ordinals, and cardinals; return a `String` with each numeric token
//! replaced by its spoken English form and non-numeric spans passed through
//! verbatim. Singular vs plural currency, date vs decimal, and ordinal-suffix
//! selection are pinned on both sides; an out-of-range date component yields a
//! cardinal fallback rather than a panic. Invariant: a token with no numeric
//! content is byte-identical in the output.
//!
//! RED: `syrinx-frontend` exposes no `expand` module yet, so the symbol does not
//! resolve and the test target fails to build — every criterion is unmet. GREEN
//! implements `expand_numbers` so each assertion below holds.

use syrinx_frontend::expand::expand_numbers;

// ----------------------------------------------------------------------------
// C1 — USD currency with the singular/plural boundary pinned on both sides.
// ----------------------------------------------------------------------------

/// `$1,200` reads as a cardinal amount with the comma stripped and a *plural*
/// "dollars" suffix because the amount exceeds one (criterion C1).
#[test]
fn currency_thousands_is_plural_dollars() {
    assert_eq!(expand_numbers("$1,200"), "one thousand two hundred dollars");
}

/// `$1` reads with the *singular* "dollar" suffix — the low side of the plural
/// boundary, amount == 1 (criterion C1).
#[test]
fn currency_one_is_singular_dollar() {
    assert_eq!(expand_numbers("$1"), "one dollar");
}

/// `$2` reads with the *plural* "dollars" suffix — the high side of the plural
/// boundary, amount just past one. Pins that the suffix switches at exactly one
/// (criterion C1).
#[test]
fn currency_two_is_plural_dollars() {
    let out = expand_numbers("$2");
    assert_eq!(out, "two dollars");
    assert!(out.ends_with("dollars"), "amount > 1 is plural");
    assert_ne!(out, "two dollar", "must not be singular for amount > 1");
}

// ----------------------------------------------------------------------------
// C2 — date vs decimal disambiguation, pinned on both sides.
// ----------------------------------------------------------------------------

/// `1/2/26` is read as a month/day/year date: month name, ordinal day, and the
/// year spoken in pairs (criterion C2, the date side of the disambiguation).
#[test]
fn date_mdy_reads_month_ordinal_day_and_year() {
    assert_eq!(expand_numbers("1/2/26"), "January second twenty twenty-six");
}

/// `3.14` is read as a decimal: the integer part as a cardinal, then "point",
/// then the fractional digits read *individually* — "one four", not "fourteen"
/// (criterion C2, the decimal side of the disambiguation).
#[test]
fn decimal_reads_digits_individually() {
    let out = expand_numbers("3.14");
    assert_eq!(out, "three point one four");
    assert!(!out.contains("fourteen"), "fraction is read digit-by-digit");
}

/// `10.5` confirms the *integer* part of a decimal is a cardinal ("ten", not
/// "one zero") while the fraction stays digit-wise — distinguishing the two
/// halves of the decimal reading (criterion C2).
#[test]
fn decimal_integer_part_is_cardinal() {
    assert_eq!(expand_numbers("10.5"), "ten point five");
}

/// An out-of-range date component (month 13) falls back to a cardinal reading
/// rather than panicking or inventing a month name — 13 is spoken "thirteen"
/// and no month name appears (criterion C2, the invalid-date side; contract
/// edge "out-of-range date component yields the cardinal fallback").
#[test]
fn out_of_range_date_falls_back_to_cardinal_without_panic() {
    let out = expand_numbers("13/2/26");
    assert!(out.contains("thirteen"), "out-of-range month read as a cardinal");
    assert!(!out.contains("January"), "13 is not a valid month name");
}

// ----------------------------------------------------------------------------
// C3 — ordinal suffix selection across st / nd / rd / th.
// ----------------------------------------------------------------------------

/// The `st` suffix maps to the first ordinal (criterion C3).
#[test]
fn ordinal_st_suffix() {
    assert_eq!(expand_numbers("1st"), "first");
}

/// The `nd` suffix maps to the second ordinal (criterion C3).
#[test]
fn ordinal_nd_suffix() {
    assert_eq!(expand_numbers("2nd"), "second");
}

/// The `rd` suffix on a two-digit value yields a hyphenated compound ordinal —
/// "twenty-third", not "twenty-three" (criterion C3).
#[test]
fn ordinal_rd_suffix_two_digit_hyphenated() {
    let out = expand_numbers("23rd");
    assert_eq!(out, "twenty-third");
    assert_ne!(out, "twenty-three", "ordinal, not cardinal");
}

/// The `th` suffix maps to the corresponding ordinal — completing the st/nd/rd/th
/// suffix coverage (criterion C3).
#[test]
fn ordinal_th_suffix() {
    assert_eq!(expand_numbers("4th"), "fourth");
}

// ----------------------------------------------------------------------------
// C4 — bare cardinal (hyphenation) and non-numeric passthrough.
// ----------------------------------------------------------------------------

/// A bare integer in 21..=99 with a nonzero unit is a *hyphenated* cardinal
/// (criterion C4).
#[test]
fn bare_integer_is_hyphenated_cardinal() {
    let out = expand_numbers("42");
    assert_eq!(out, "forty-two");
    assert!(out.contains('-'), "compound tens are hyphenated");
}

/// A round multiple of ten has *no* hyphen — the low side of the hyphenation
/// rule, distinguishing "twenty" from "twenty-something" (criterion C4).
#[test]
fn round_ten_cardinal_has_no_hyphen() {
    let out = expand_numbers("20");
    assert_eq!(out, "twenty");
    assert!(!out.contains('-'), "a round ten carries no hyphen");
}

/// Text with no numeric token is returned byte-identical — the passthrough
/// invariant (criterion C4).
#[test]
fn non_numeric_text_passes_through_unchanged() {
    assert_eq!(expand_numbers("hello"), "hello");
}

/// Empty input passes through to the empty string (edge of the passthrough
/// invariant; criterion C4).
#[test]
fn empty_input_passes_through() {
    assert_eq!(expand_numbers(""), "");
}
