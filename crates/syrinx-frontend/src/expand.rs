//! Numeric verbalization over already-normalized text (T-01.02).
//!
//! [`expand_numbers`] takes a `&str` that may carry a single numeric token —
//! USD currency (`$1,200`), a month/day/year date (`1/2/26`), a decimal
//! (`3.14`), an ordinal (`23rd`), or a bare cardinal (`42`) — and returns a
//! `String` with that token rewritten in spoken English. A token with no
//! numeric content is returned byte-identical (the passthrough invariant).
//!
//! Disambiguation: dates use `/`, decimals use `.`, ordinals carry an
//! `st`/`nd`/`rd`/`th` suffix, currency a leading `$`; a bare run of digits is a
//! cardinal. An out-of-range month yields a cardinal fallback rather than a
//! panic. Decimal fractions are read digit-by-digit; the integer part as a
//! cardinal. Currency switches between the singular "dollar" (amount one) and
//! plural "dollars" (any other amount).
//!
//! Out of scope (per the task): non-USD currency, localized number formats,
//! Roman-numeral and phone-number expansion.

/// Cardinal words for a single digit, indexed by its value 0..=9.
const ONES: [&str; 10] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
];

/// Cardinal words for 10..=19, indexed by the units digit 0..=9.
const TEENS: [&str; 10] = [
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
];

/// Tens words indexed by the tens digit; 0 and 1 are unused (teens cover 10..19).
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// Cardinal-suffix → ordinal-suffix replacements, applied to the trailing word
/// of a cardinal spelling. Longest/most-specific words come first so a compound
/// like "twenty-three" replaces its final "three", not an earlier substring.
const ORDINALS: [(&str, &str); 28] = [
    ("thousand", "thousandth"),
    ("hundred", "hundredth"),
    ("twenty", "twentieth"),
    ("thirty", "thirtieth"),
    ("forty", "fortieth"),
    ("fifty", "fiftieth"),
    ("sixty", "sixtieth"),
    ("seventy", "seventieth"),
    ("eighty", "eightieth"),
    ("ninety", "ninetieth"),
    ("eleven", "eleventh"),
    ("twelve", "twelfth"),
    ("thirteen", "thirteenth"),
    ("fourteen", "fourteenth"),
    ("fifteen", "fifteenth"),
    ("sixteen", "sixteenth"),
    ("seventeen", "seventeenth"),
    ("eighteen", "eighteenth"),
    ("nineteen", "nineteenth"),
    ("ten", "tenth"),
    ("one", "first"),
    ("two", "second"),
    ("three", "third"),
    ("four", "fourth"),
    ("five", "fifth"),
    ("six", "sixth"),
    ("seven", "seventh"),
    ("eight", "eighth"),
];

/// Rewrite the numeric token in `input` as spoken English, or pass it through
/// verbatim when it carries no numeric content.
pub fn expand_numbers(input: &str) -> String {
    if let Some(rest) = input.strip_prefix('$') {
        return currency(rest);
    }
    if let Some(words) = try_date(input) {
        return words;
    }
    if let Some(words) = try_decimal(input) {
        return words;
    }
    if let Some(words) = try_ordinal(input) {
        return words;
    }
    if let Some(words) = try_cardinal(input) {
        return words;
    }
    input.to_string()
}

/// Spell a USD amount: strip grouping commas, read the integer as a cardinal,
/// and pick the singular suffix for exactly one dollar, the plural otherwise.
fn currency(rest: &str) -> String {
    let digits: String = rest.chars().filter(|c| *c != ',').collect();
    let n: u32 = digits.parse().unwrap();
    let amount = cardinal(n);
    match n {
        1 => format!("{} dollar", amount),
        _ => format!("{} dollars", amount),
    }
}

/// Read `m/d/y` as month name + ordinal day + paired year. An unrecognized
/// month falls back to reading all three components as cardinals (no panic).
fn try_date(input: &str) -> Option<String> {
    let parts: Vec<&str> = input.split('/').collect();
    let [m, d, y] = parts.as_slice() else {
        return None;
    };
    let month: u32 = m.parse().ok()?;
    let day: u32 = d.parse().ok()?;
    let year: u32 = y.parse().ok()?;
    match month_name(month) {
        Some(name) => Some(format!("{} {} {}", name, ordinal(day), say_year(year))),
        None => Some(format!(
            "{} {} {}",
            cardinal(month),
            cardinal(day),
            cardinal(year)
        )),
    }
}

/// Read `int.frac` as a cardinal integer, "point", then each fraction digit
/// spoken individually. Returns `None` if there is no `.` or a part is invalid.
fn try_decimal(input: &str) -> Option<String> {
    let (int_part, frac_part) = input.split_once('.')?;
    let n: u32 = int_part.parse().ok()?;
    let mut frac: Vec<&'static str> = Vec::new();
    for c in frac_part.chars() {
        frac.push(digit_word(c)?);
    }
    Some(format!("{} point {}", cardinal(n), frac.join(" ")))
}

/// Read a trailing `st`/`nd`/`rd`/`th` ordinal token as its spoken ordinal.
fn try_ordinal(input: &str) -> Option<String> {
    for suffix in ["st", "nd", "rd", "th"] {
        if let Some(num) = input.strip_suffix(suffix) {
            let n: u32 = num.parse().ok()?;
            return Some(ordinal(n));
        }
    }
    None
}

/// Read a bare run of digits as a cardinal; `None` for anything non-numeric.
fn try_cardinal(input: &str) -> Option<String> {
    let n: u32 = input.parse().ok()?;
    Some(cardinal(n))
}

/// Spoken month name for 1..=12, or `None` for any out-of-range value.
fn month_name(m: u32) -> Option<&'static str> {
    match m {
        1 => Some("January"),
        2 => Some("February"),
        3 => Some("March"),
        4 => Some("April"),
        5 => Some("May"),
        6 => Some("June"),
        7 => Some("July"),
        8 => Some("August"),
        9 => Some("September"),
        10 => Some("October"),
        11 => Some("November"),
        12 => Some("December"),
        _ => None,
    }
}

/// Read a two-digit year as a 20xx century pair: "twenty" then the cardinal.
fn say_year(y: u32) -> String {
    format!("twenty {}", cardinal(y))
}

/// Cardinal word for a single decimal digit char, or `None` if not `0`..=`9`.
fn digit_word(c: char) -> Option<&'static str> {
    match c {
        '0' => Some("zero"),
        '1' => Some("one"),
        '2' => Some("two"),
        '3' => Some("three"),
        '4' => Some("four"),
        '5' => Some("five"),
        '6' => Some("six"),
        '7' => Some("seven"),
        '8' => Some("eight"),
        '9' => Some("nine"),
        _ => None,
    }
}

/// Spell `n` as an English cardinal (supports the 0..=9999 range this frontend
/// needs). Digits are split positionally so the spelling structure is selected
/// by slice patterns rather than magnitude comparisons.
fn cardinal(n: u32) -> String {
    let s = n.to_string();
    let digits: Vec<u32> = s.chars().map(|c| c.to_digit(10).unwrap()).collect();
    match digits.as_slice() {
        [u] => ONES[*u as usize].to_string(),
        [t, u] => two_digit(*t, *u),
        [h, t, u] => spell_group(*h, *t, *u),
        [th, h, t, u] => {
            let head = format!("{} thousand", ONES[*th as usize]);
            match (*h, *t, *u) {
                (0, 0, 0) => head,
                _ => format!("{} {}", head, spell_group(*h, *t, *u)),
            }
        }
        _ => s,
    }
}

/// Spell a two-digit value from its tens digit `t` and units digit `u`:
/// teens for `t == 1`, a bare tens word when `u == 0`, else a hyphenated pair.
fn two_digit(t: u32, u: u32) -> String {
    match (t, u) {
        (1, _) => TEENS[u as usize].to_string(),
        (_, 0) => TENS[t as usize].to_string(),
        (_, _) => format!("{}-{}", TENS[t as usize], ONES[u as usize]),
    }
}

/// Spell a three-digit group from its hundreds/tens/units digits, tolerating a
/// zero hundreds digit (used for the lower group of a four-digit number).
fn spell_group(h: u32, t: u32, u: u32) -> String {
    match (h, t, u) {
        (0, 0, _) => ONES[u as usize].to_string(),
        (0, _, _) => two_digit(t, u),
        (_, 0, 0) => format!("{} hundred", ONES[h as usize]),
        (_, 0, _) => format!("{} hundred {}", ONES[h as usize], ONES[u as usize]),
        (_, _, _) => format!("{} hundred {}", ONES[h as usize], two_digit(t, u)),
    }
}

/// Convert `n` to its spoken ordinal by replacing the trailing cardinal word
/// with its ordinal form ("twenty-three" → "twenty-third").
fn ordinal(n: u32) -> String {
    let c = cardinal(n);
    for (card, ord) in ORDINALS {
        if let Some(stem) = c.strip_suffix(card) {
            return format!("{}{}", stem, ord);
        }
    }
    c
}
