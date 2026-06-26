//! Faithful (common-case) port of CosyVoice2's `frontend.text_normalize` — the
//! **wetext**-based zh+en text normalizer the model actually runs before tokenizing.
//!
//! [`normalize_text`] reproduces `CosyVoiceFrontEnd.text_normalize(text,
//! split=False, text_frontend=True)` (the normalized *string*, pre-sentence-split):
//!
//! ```text
//!   if '<|' in text and '|>' in text: return text          # SSML bypass
//!   if text == '': return text
//!   text = text.strip()
//!   if contains_chinese(text):
//!       text = zh_tn_model.normalize(text)                 # wetext zh WFST
//!       text = text.replace("\n", "")
//!       text = replace_blank(text)
//!       text = replace_corner_mark(text)                   # ² -> 平方, ³ -> 立方
//!       text = text.replace(".", "。")
//!       text = text.replace(" - ", "，")
//!       text = remove_bracket(text)                        # （）【】 ` —— ...
//!       text = re.sub(r'[，,、]+$', '。', text)
//!   else:
//!       text = en_tn_model.normalize(text)                 # wetext en WFST
//!       text = spell_out_number(text, inflect)             # any leftover digits
//! ```
//!
//! ## Honest scope
//!
//! wetext is a large WFST grammar; this is a hand-written **common-case** match,
//! not full WFST parity. It covers, for both languages, the high-frequency cases:
//! integers, decimals, percentages, `$`/`￥` currency, ordinals (en), HH:MM times
//! (en), `N-M` ranges (en), 4-digit years, plus the zh post-processing
//! replacements (`replace_blank` / `replace_corner_mark` / bracket removal / `.` ->
//! `。` / trailing-comma -> `。`) and zh `年/月/日/点/分/第N/到` number reading.
//!
//! Known misses vs. the wetext reference (reported by the root parity test, not
//! hidden): the full en **date** grammar (`2024-01-15` -> "the fifteenth of
//! january ..."), unit/abbreviation expansion beyond a small map (`kg`, `MB`,
//! `Dr.`, `Mr.`), the `U.S.A.` -> `USA` acronym collapse, the `2.0` -> "two point
//! oh" version reading, and wetext's own range artifacts. These remain verbatim.
//!
//! Pure Rust, no external crates — gated behind the crate `tn` feature so the
//! default/CI build is unchanged.

/// Normalize `text` the way CosyVoice2's wetext frontend does (common cases),
/// returning the normalized string (the `split=False` branch).
pub fn normalize_text(text: &str) -> String {
    // SSML bypass: text_frontend is disabled, returned unchanged (no strip).
    if text.contains("<|") && text.contains("|>") {
        return text.to_string();
    }
    if text.is_empty() {
        return String::new();
    }
    let text = text.trim();
    if contains_chinese(text) {
        let mut t = zh_read_numbers(text);
        t = t.replace('\n', "");
        t = replace_blank(&t);
        t = replace_corner_mark(&t);
        t = t.replace('.', "。");
        t = t.replace(" - ", "，");
        t = remove_bracket(&t);
        t = trim_trailing_commas(&t);
        t
    } else {
        en_normalize(text)
    }
}

/// Does `text` contain a CJK ideograph (`一`-`鿿`)? (frontend_utils
/// `contains_chinese`.)
fn contains_chinese(text: &str) -> bool {
    text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
}

// ===========================================================================
// Chinese post-processing helpers (verbatim ports of frontend_utils).
// ===========================================================================

/// `replace_blank`: keep an interior space only when BOTH neighbours are ASCII
/// non-space; drop it otherwise (it sits between/next to CJK text).
fn replace_blank(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if c == ' ' {
            let prev_ok = i > 0 && chars[i - 1].is_ascii() && chars[i - 1] != ' ';
            let next_ok =
                i + 1 < chars.len() && chars[i + 1].is_ascii() && chars[i + 1] != ' ';
            if prev_ok && next_ok {
                out.push(c);
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// `replace_corner_mark`: ² -> 平方, ³ -> 立方.
fn replace_corner_mark(text: &str) -> String {
    text.replace('²', "平方").replace('³', "立方")
}

/// `remove_bracket`: strip （）【】 and backticks, and turn `——` into a space.
fn remove_bracket(text: &str) -> String {
    text.replace('（', "")
        .replace('）', "")
        .replace('【', "")
        .replace('】', "")
        .replace('`', "")
        .replace("——", " ")
}

/// `re.sub(r'[，,、]+$', '。', text)`: a trailing run of `，` / `,` / `、` becomes a
/// single `。`.
fn trim_trailing_commas(text: &str) -> String {
    let trimmed = text.trim_end_matches(['，', ',', '、']);
    if trimmed.len() == text.len() {
        text.to_string()
    } else {
        format!("{trimmed}。")
    }
}

// ===========================================================================
// Chinese number reading (the wetext zh WFST, common cases).
// ===========================================================================

const ZH_DIGITS: [&str; 10] = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];

/// Scan `s`, replacing each Arabic-number token with its Chinese reading. Context
/// taken into account: a leading `￥` (-> `元` suffix), a trailing `%` (-> `百分之`
/// prefix), and a 4-digit run immediately before `年` (read digit-by-digit as a
/// year). Everything else is read as a cardinal (`点` for decimals).
fn zh_read_numbers(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    let mut pending_currency = false;
    while i < chars.len() {
        let c = chars[i];
        // `￥` directly before a digit: drop it, mark the next number as currency.
        if c == '￥' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            pending_currency = true;
            i += 1;
            continue;
        }
        if c.is_ascii_digit() {
            let mut int_digits = String::new();
            while i < chars.len() && chars[i].is_ascii_digit() {
                int_digits.push(chars[i]);
                i += 1;
            }
            let mut frac_digits = String::new();
            if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
                i += 1; // skip '.'
                while i < chars.len() && chars[i].is_ascii_digit() {
                    frac_digits.push(chars[i]);
                    i += 1;
                }
            }
            let next = chars.get(i).copied();

            // percent: `N%` / `N.M%` -> 百分之… (consume the '%').
            if next == Some('%') {
                out.push_str("百分之");
                out.push_str(&zh_read_value(&int_digits, &frac_digits));
                i += 1;
                continue;
            }
            // year: a bare 4-digit run before 年, read digit-by-digit.
            if frac_digits.is_empty() && int_digits.len() == 4 && next == Some('年') {
                out.push_str(&zh_read_year(&int_digits));
                continue;
            }
            // ordinary number.
            out.push_str(&zh_read_value(&int_digits, &frac_digits));
            if pending_currency {
                out.push_str("元");
                pending_currency = false;
            }
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Read an integer (`int_digits`) plus optional fractional digits as Chinese:
/// cardinal for the integer, `点` then digit-by-digit for the fraction.
fn zh_read_value(int_digits: &str, frac_digits: &str) -> String {
    let mut s = zh_cardinal(int_digits);
    if !frac_digits.is_empty() {
        s.push('点');
        for ch in frac_digits.chars() {
            let d = ch.to_digit(10).unwrap_or(0) as usize;
            s.push_str(ZH_DIGITS[d]);
        }
    }
    s
}

/// Read each digit of a 4-digit year individually (`2024` -> 二零二四).
fn zh_read_year(digits: &str) -> String {
    digits
        .chars()
        .map(|ch| ZH_DIGITS[ch.to_digit(10).unwrap_or(0) as usize])
        .collect()
}

/// Chinese cardinal reading of a non-negative integer string (up to 兆), with
/// inter-group zeroing and the leading `一十` -> `十` contraction.
fn zh_cardinal(digits: &str) -> String {
    let n: u128 = digits.parse().unwrap_or(0);
    if n == 0 {
        return "零".to_string();
    }
    let small_units = ["", "十", "百", "千"];
    let big_units = ["", "万", "亿", "兆"];
    // 4-digit groups, least-significant first.
    let mut groups: Vec<usize> = Vec::new();
    let mut m = n;
    while m > 0 {
        groups.push((m % 10000) as usize);
        m /= 10000;
    }
    let mut result = String::new();
    for gi in (0..groups.len()).rev() {
        let g = groups[gi];
        if g == 0 {
            // a zero group between non-zero groups contributes a single 零.
            if !result.is_empty()
                && groups[..gi].iter().any(|&x| x != 0)
                && !result.ends_with('零')
            {
                result.push('零');
            }
            continue;
        }
        // a short group (<1000) after a higher group needs a connecting 零.
        if !result.is_empty() && g < 1000 && !result.ends_with('零') {
            result.push('零');
        }
        result.push_str(&zh_group(g, &small_units));
        result.push_str(big_units[gi]);
    }
    if let Some(rest) = result.strip_prefix("一十") {
        result = format!("十{rest}");
    }
    result
}

/// Read a 1..=9999 group with 千/百/十 units and interior zeroing.
fn zh_group(g: usize, units: &[&str; 4]) -> String {
    let d = [g / 1000 % 10, g / 100 % 10, g / 10 % 10, g % 10];
    let unit_idx = [3usize, 2, 1, 0];
    let mut s = String::new();
    let mut started = false;
    let mut zero_pending = false;
    for k in 0..4 {
        let dv = d[k];
        if dv == 0 {
            if started {
                zero_pending = true;
            }
        } else {
            if zero_pending {
                s.push('零');
                zero_pending = false;
            }
            s.push_str(ZH_DIGITS[dv]);
            s.push_str(units[unit_idx[k]]);
            started = true;
        }
    }
    s
}

// ===========================================================================
// English number reading (the wetext en WFST + spell_out_number, common cases).
// ===========================================================================

const EN_ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
];
const EN_TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// Normalize an English (non-Chinese) string: expand numbers/money/percent/time/
/// ordinals/years/ranges + a small unit & abbreviation map; lower-cased number
/// words, space-separated, British "and".
fn en_normalize(text: &str) -> String {
    // Abbreviations first (verbatim wetext expansions for the common titles).
    let pre = text.replace("Dr.", "doctor").replace("Mr.", "Mister");
    let scanned = en_scan_numbers(&pre);
    // Unit expansion (whole-word).
    replace_word(&replace_word(&scanned, "kg", "kilograms"), "MB", "megabytes")
}

/// Replace a whole-word `from` token with `to` (ASCII word boundaries).
fn replace_word(text: &str, from: &str, to: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        if text[i..].starts_with(from) {
            let before_ok = i == 0 || !is_word_byte(bytes[i - 1]);
            let after = i + from.len();
            let after_ok = after >= text.len() || !is_word_byte(bytes[after]);
            if before_ok && after_ok {
                out.push_str(to);
                i = after;
                continue;
            }
        }
        // advance one char.
        let ch = text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Scan English text, replacing Arabic-number tokens with spelled-out words.
fn en_scan_numbers(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    let mut pending_dollar = false;
    while i < chars.len() {
        let c = chars[i];
        if c == '$' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            pending_dollar = true;
            i += 1;
            continue;
        }
        if c.is_ascii_digit() {
            let start = i;
            let prev = if start > 0 { Some(chars[start - 1]) } else { None };
            let mut int_digits = String::new();
            while i < chars.len() && chars[i].is_ascii_digit() {
                int_digits.push(chars[i]);
                i += 1;
            }

            // time: HH:MM (only when not currency).
            if !pending_dollar
                && i + 2 < chars.len()
                && chars[i] == ':'
                && chars[i + 1].is_ascii_digit()
                && chars[i + 2].is_ascii_digit()
                && !(i + 3 < chars.len() && chars[i + 3].is_ascii_digit())
            {
                let mm: String = [chars[i + 1], chars[i + 2]].iter().collect();
                out.push_str(&en_read_time(&int_digits, &mm));
                i += 3;
                continue;
            }

            // optional decimal fraction.
            let mut frac_digits = String::new();
            if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    frac_digits.push(chars[i]);
                    i += 1;
                }
            }
            let next = chars.get(i).copied();

            // percent.
            if next == Some('%') {
                out.push_str(&en_read_value(&int_digits, &frac_digits, false));
                out.push_str(" percent");
                i += 1;
                continue;
            }
            // currency ($N / $N.MM): drop trailing fractional zeros, append dollars.
            if pending_dollar {
                out.push_str(&en_read_value(&int_digits, &frac_digits, true));
                out.push_str(" dollars");
                pending_dollar = false;
                continue;
            }
            // ordinal: digits followed by st/nd/rd/th (word boundary).
            if frac_digits.is_empty() {
                if let Some(suf) = en_ordinal_suffix(&chars, i) {
                    out.push_str(&en_ordinal(&int_digits));
                    i += suf;
                    continue;
                }
            }
            // range: N-M -> "N to M" (cardinal both sides; the right side is read
            // by the loop). Checked before the year reading so range parts stay
            // cardinal.
            if frac_digits.is_empty()
                && next == Some('-')
                && chars.get(i + 1).is_some_and(|c| c.is_ascii_digit())
            {
                out.push_str(&en_cardinal_str(&int_digits));
                out.push_str(" to ");
                i += 1; // consume '-'
                continue;
            }
            // year: a standalone 4-digit run (not a range part).
            if frac_digits.is_empty()
                && int_digits.len() == 4
                && prev != Some('-')
                && !prev.is_some_and(|c| c.is_ascii_digit())
            {
                out.push_str(&en_read_year(&int_digits));
                continue;
            }
            // ordinary cardinal / decimal.
            out.push_str(&en_read_value(&int_digits, &frac_digits, false));
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// If `chars[i..]` begins with an ordinal suffix at a word boundary, return its
/// length (always 2). Otherwise `None`.
fn en_ordinal_suffix(chars: &[char], i: usize) -> Option<usize> {
    if i + 1 >= chars.len() {
        return None;
    }
    let suf: String = [chars[i], chars[i + 1]].iter().collect();
    let is_suf = matches!(suf.as_str(), "st" | "nd" | "rd" | "th");
    if !is_suf {
        return None;
    }
    // must be a word boundary after the suffix.
    if chars.get(i + 2).is_some_and(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    Some(2)
}

/// Render integer + optional fraction. `currency_strip` drops trailing fractional
/// zeros (so `$5.50` reads "five point five").
fn en_read_value(int_digits: &str, frac_digits: &str, currency_strip: bool) -> String {
    let frac = if currency_strip {
        frac_digits.trim_end_matches('0')
    } else {
        frac_digits
    };
    let mut s = en_cardinal_str(int_digits);
    if !frac.is_empty() {
        s.push_str(" point");
        for ch in frac.chars() {
            let d = ch.to_digit(10).unwrap_or(0) as usize;
            s.push(' ');
            s.push_str(EN_ONES[d]);
        }
    }
    s
}

/// HH:MM -> spoken time ("seven thirty", "twelve o'clock", "nine oh five").
fn en_read_time(hh: &str, mm: &str) -> String {
    let hour: u128 = hh.parse().unwrap_or(0);
    let minute: u128 = mm.parse().unwrap_or(0);
    let h = en_cardinal(hour);
    if minute == 0 {
        format!("{h} o'clock")
    } else if minute < 10 {
        format!("{h} oh {}", en_cardinal(minute))
    } else {
        format!("{h} {}", en_cardinal(minute))
    }
}

/// 4-digit year reading (`2024` -> "twenty twenty four").
fn en_read_year(digits: &str) -> String {
    let hi: u128 = digits[..2].parse().unwrap_or(0);
    let lo: u128 = digits[2..].parse().unwrap_or(0);
    if lo == 0 {
        format!("{} hundred", en_cardinal(hi))
    } else if lo < 10 {
        format!("{} oh {}", en_cardinal(hi), en_cardinal(lo))
    } else {
        format!("{} {}", en_cardinal(hi), en_cardinal(lo))
    }
}

fn en_cardinal_str(digits: &str) -> String {
    en_cardinal(digits.parse::<u128>().unwrap_or(0))
}

/// English cardinal: space-separated words, British "and" before a trailing 1..99.
fn en_cardinal(n: u128) -> String {
    if n < 1000 {
        return en_below_thousand(n);
    }
    for (val, name) in [(1_000_000_000u128, "billion"), (1_000_000, "million"), (1_000, "thousand")] {
        if n >= val {
            let high = n / val;
            let rem = n % val;
            let mut s = format!("{} {name}", en_cardinal(high));
            if rem > 0 {
                if rem < 100 {
                    s.push_str(" and ");
                    s.push_str(&en_below_hundred(rem));
                } else {
                    s.push(' ');
                    s.push_str(&en_cardinal(rem));
                }
            }
            return s;
        }
    }
    en_below_thousand(n)
}

/// 0..=999 with internal "and" (`250` -> "two hundred and fifty").
fn en_below_thousand(n: u128) -> String {
    if n < 100 {
        return en_below_hundred(n);
    }
    let h = EN_ONES[(n / 100) as usize];
    let rem = n % 100;
    if rem == 0 {
        format!("{h} hundred")
    } else {
        format!("{h} hundred and {}", en_below_hundred(rem))
    }
}

/// 0..=99 ("forty two", "twenty", "nineteen").
fn en_below_hundred(n: u128) -> String {
    if n < 20 {
        return EN_ONES[n as usize].to_string();
    }
    let t = EN_TENS[(n / 10) as usize];
    let u = n % 10;
    if u == 0 {
        t.to_string()
    } else {
        format!("{t} {}", EN_ONES[u as usize])
    }
}

/// Ordinal reading: cardinal, then the last word made ordinal (`21` -> "twenty
/// first").
fn en_ordinal(digits: &str) -> String {
    let card = en_cardinal_str(digits);
    match card.rsplit_once(' ') {
        Some((head, last)) => format!("{head} {}", ordinal_word(last)),
        None => ordinal_word(&card),
    }
}

/// Map a cardinal word to its ordinal form.
fn ordinal_word(w: &str) -> String {
    match w {
        "zero" => "zeroth",
        "one" => "first",
        "two" => "second",
        "three" => "third",
        "four" => "fourth",
        "five" => "fifth",
        "six" => "sixth",
        "seven" => "seventh",
        "eight" => "eighth",
        "nine" => "ninth",
        "ten" => "tenth",
        "eleven" => "eleventh",
        "twelve" => "twelfth",
        "twenty" => "twentieth",
        "thirty" => "thirtieth",
        "forty" => "fortieth",
        "fifty" => "fiftieth",
        "sixty" => "sixtieth",
        "seventy" => "seventieth",
        "eighty" => "eightieth",
        "ninety" => "ninetieth",
        "hundred" => "hundredth",
        "thousand" => "thousandth",
        "million" => "millionth",
        "billion" => "billionth",
        other => return format!("{other}th"),
    }
    .to_string()
}
