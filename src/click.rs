//! Click-captcha helpers: parse the instruction prompt and match its
//! characters against detected glyph boxes.
//!
//! Supported prompt shapes (examples):
//! - Chinese "click in order" prompts with bracketed targets
//! - Chinese "please click" prompts with a quoted target
//! - `click in order: A B C`
//! - `Please click the letters D, F, K`
//!
//! The prompt may also be omitted entirely: `parse_prompt` then returns an
//! empty list and the caller is expected to return the detected boxes so the
//! client can apply its own matching logic.

/// Result of matching prompt characters against detected boxes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClickMatch {
    /// Box indices in prompt order (only successful matches, one per target).
    pub order: Vec<usize>,

    /// Prompt characters that no unused box could satisfy, in prompt order.
    pub unmatched: Vec<char>,
}

/// Instruction phrases removed when the prompt has no bracketed targets.
const CN_PHRASES: &[&str] = &[
    "请依次点击",
    "请按顺序点击",
    "按顺序点击",
    "依次点击",
    "请点击",
    "请点选",
    "点击",
    "点选",
    "请按",
    "顺序",
    "依次",
    "图中",
    "图片中",
    "验证码",
    "文字",
    "如下",
    "按钮",
    "图标",
    "请",
];

/// English instruction words (matched case-insensitively, punctuation trimmed).
const EN_WORDS: &[&str] = &[
    "click",
    "please",
    "in",
    "order",
    "the",
    "letters",
    "letter",
    "characters",
    "character",
    "chars",
    "captcha",
    "image",
    "select",
    "tap",
    "press",
    "and",
    "then",
    "on",
    "below",
    "shown",
];

/// Parse a click prompt into the ordered target characters.
///
/// Bracketed/quoted segments win when present (so surrounding page
/// instructions are ignored); otherwise instruction words are stripped and
/// the remaining alphanumeric characters are used, in order.
pub fn parse_prompt(text: &str) -> Vec<char> {
    let chars: Vec<char> = text.chars().collect();
    let mut quoted = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        if let Some(close) = matching_bracket(chars[i]) {
            let mut j = i + 1;

            while j < chars.len() && chars[j] != close {
                j += 1;
            }

            if j < chars.len() {
                quoted.extend(
                    chars[i + 1..j]
                        .iter()
                        .copied()
                        .filter(|c| is_target_char(*c)),
                );
                i = j + 1;
                continue;
            }
        }

        i += 1;
    }

    if !quoted.is_empty() {
        return quoted;
    }

    let mut cleaned = text.to_string();

    for phrase in CN_PHRASES {
        cleaned = cleaned.replace(phrase, " ");
    }

    let mut out = Vec::new();

    for token in cleaned.split_whitespace() {
        let trimmed: String = token.chars().filter(|c| c.is_alphanumeric()).collect();

        if trimmed.is_empty() {
            continue;
        }

        if EN_WORDS.iter().any(|w| trimmed.eq_ignore_ascii_case(w)) {
            continue;
        }

        out.extend(trimmed.chars().filter(|c| is_target_char(*c)));
    }

    out
}

/// Match prompt characters against per-box recognized text.
///
/// For each target (in prompt order) the first unused box whose recognized
/// text contains the character wins; an exact match is preferred over an
/// ASCII-case-insensitive one, so `D` still picks `D` over `d` when both
/// exist.
pub fn match_prompt(prompt: &[char], hits: &[String]) -> ClickMatch {
    let mut used = vec![false; hits.len()];
    let mut order = Vec::new();
    let mut unmatched = Vec::new();

    for &target in prompt {
        let mut found = None;

        for (i, hit) in hits.iter().enumerate() {
            if !used[i] && hit.chars().any(|c| c == target) {
                found = Some(i);
                break;
            }
        }

        if found.is_none() {
            for (i, hit) in hits.iter().enumerate() {
                if !used[i] && hit.chars().any(|c| c.eq_ignore_ascii_case(&target)) {
                    found = Some(i);
                    break;
                }
            }
        }

        match found {
            Some(i) => {
                used[i] = true;
                order.push(i);
            }
            None => unmatched.push(target),
        }
    }

    ClickMatch { order, unmatched }
}

/// Match prompt characters against per-box, per-target scores.
///
/// `box_scores[box][target]` is the model's score for the target character in
/// that box. Pairs are assigned globally by descending score, so an early
/// prompt character cannot take the box a later character needs.
pub fn match_prompt_scores(prompt: &[char], box_scores: &[Vec<f32>]) -> ClickMatch {
    let mut pairs = Vec::new();

    for (box_idx, scores) in box_scores.iter().enumerate() {
        for (target_idx, &score) in scores.iter().enumerate().take(prompt.len()) {
            if score > 0.0 && score.is_finite() {
                pairs.push((score, target_idx, box_idx));
            }
        }
    }

    pairs.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    let mut used_boxes = vec![false; box_scores.len()];
    let mut assigned: Vec<Option<usize>> = vec![None; prompt.len()];

    for (_, target_idx, box_idx) in pairs {
        if assigned[target_idx].is_none() && !used_boxes[box_idx] {
            assigned[target_idx] = Some(box_idx);
            used_boxes[box_idx] = true;
        }
    }

    let mut order = Vec::new();
    let mut unmatched = Vec::new();

    for (target_idx, &target) in prompt.iter().enumerate() {
        match assigned[target_idx] {
            Some(box_idx) => order.push(box_idx),
            None => unmatched.push(target),
        }
    }

    ClickMatch { order, unmatched }
}

/// Closing bracket for an opening bracket/quote, if any.
fn matching_bracket(open: char) -> Option<char> {
    Some(match open {
        '【' => '】',
        '[' => ']',
        '（' => '）',
        '(' => ')',
        '「' => '」',
        '『' => '』',
        '“' | '”' => '”',
        '‘' | '’' => '’',
        '"' => '"',
        '\'' => '\'',
        _ => return None,
    })
}

/// Characters that can be click targets (letters/digits/CJK, ...).
fn is_target_char(c: char) -> bool {
    c.is_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(text: &str) -> Vec<char> {
        text.chars().collect()
    }

    #[test]
    fn parses_bracketed_chinese_prompts() {
        assert_eq!(parse_prompt("请依次点击【明,天,会】"), chars("明天会"));
        assert_eq!(
            parse_prompt("请依次点击【明 天 会】的按钮"),
            chars("明天会")
        );
        assert_eq!(parse_prompt("请点击“天”"), chars("天"));
        assert_eq!(parse_prompt("请点击「甲乙」"), chars("甲乙"));
    }

    #[test]
    fn parses_unbracketed_chinese_prompts() {
        assert_eq!(parse_prompt("请依次点击明天会"), chars("明天会"));
        assert_eq!(parse_prompt("请点击图中文字验证码"), chars(""));
        assert_eq!(parse_prompt("依次点击：甲乙丙"), chars("甲乙丙"));
    }

    #[test]
    fn parses_english_prompts() {
        assert_eq!(parse_prompt("click in order: A B C"), chars("ABC"));
        assert_eq!(
            parse_prompt("Please click the letters D, F, K"),
            chars("DFK")
        );
        assert_eq!(parse_prompt("select: x7"), chars("x7"));
    }

    #[test]
    fn empty_prompt_yields_no_targets() {
        assert!(parse_prompt("").is_empty());
        assert!(parse_prompt("请依次点击").is_empty());
        assert!(parse_prompt("click in order").is_empty());
    }

    #[test]
    fn matches_in_prompt_order() {
        let hits = vec!["天".to_string(), "明".to_string(), "会".to_string()];
        let m = match_prompt(&chars("明天会"), &hits);

        assert_eq!(m.order, vec![1, 0, 2]);
        assert!(m.unmatched.is_empty());
    }

    #[test]
    fn duplicates_consume_distinct_boxes() {
        let hits = vec!["天".to_string(), "天".to_string(), "明".to_string()];
        let m = match_prompt(&chars("天天明"), &hits);

        assert_eq!(m.order, vec![0, 1, 2]);
        assert!(m.unmatched.is_empty());
    }

    #[test]
    fn reports_unmatched_characters() {
        let hits = vec!["天".to_string()];
        let m = match_prompt(&chars("明天"), &hits);

        assert_eq!(m.order, vec![0]);
        assert_eq!(m.unmatched, chars("明"));
    }

    #[test]
    fn prefers_exact_case_then_falls_back() {
        let hits = vec!["d".to_string(), "D".to_string()];
        let m = match_prompt(&chars("D"), &hits);

        assert_eq!(m.order, vec![1], "prefer the exact-case box");

        let hits = vec!["d".to_string()];
        let m = match_prompt(&chars("D"), &hits);

        assert_eq!(m.order, vec![0], "fall back to case-insensitive");
    }

    #[test]
    fn hits_with_multiple_characters_still_match() {
        let hits = vec!["AB".to_string()];
        let m = match_prompt(&chars("B"), &hits);

        assert_eq!(m.order, vec![0]);
    }

    #[test]
    fn score_assignment_avoids_early_target_stealing() {
        // Box 0 is the best match for both targets, but B scores higher.
        // Global assignment gives box 0 to B; A has no other box.
        let scores = vec![vec![0.90, 0.95], vec![0.10, 0.20]];
        let m = match_prompt_scores(&chars("AB"), &scores);

        assert_eq!(m.order, vec![1, 0]);
        assert!(m.unmatched.is_empty());
    }

    #[test]
    fn score_assignment_uses_the_best_available_box() {
        let scores = vec![vec![0.80, 0.70], vec![0.60, 0.90]];
        let m = match_prompt_scores(&chars("AB"), &scores);

        assert_eq!(m.order, vec![0, 1]);
        assert!(m.unmatched.is_empty());
    }
}
