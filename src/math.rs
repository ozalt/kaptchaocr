//! Arithmetic captcha parsing and solving.
//!
//! Normalizes full-width and common OCR aliases (`＋－×÷＝`, `x`, `X`, `*` …)
//! plus Chinese numerals and operator words,
//! then either evaluates the expression or solves an equation with a single
//! unknown (`?`).
//!
//! ```
//! use kaptchowmin::math::solve;
//!
//! assert_eq!(solve("5+?=8").unwrap().answer, "3");
//! assert_eq!(solve("９÷３＝").unwrap().answer, "3");
//! assert_eq!(solve("九乘六等于？").unwrap().answer, "54");
//! ```

use serde::{Deserialize, Serialize};

/// Result of parsing and solving an arithmetic captcha.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MathSolution {
    /// Normalized expression (keeps `=` and `?`).
    pub expression: String,

    /// Computed value, rounded to at most 2 decimals.
    pub result: f64,

    /// Value formatted for direct typing, e.g. `"3"` or `"3.5"`.
    pub answer: String,
}

const EPS: f64 = 1e-9;
const MAX_INPUT_LEN: usize = 64;
const MAX_MAGNITUDE: f64 = 1e9;
const REL_EPS: f64 = 1e-6;

/// Try to parse `text` as an arithmetic captcha and solve it.
///
/// Returns `None` when the text is not a supported arithmetic expression
/// (plain captcha text, two unknowns, division by zero, unknown without an
/// equation, ...).
pub fn solve(text: &str) -> Option<MathSolution> {
    let expression = normalize(text);

    if expression.is_empty()
        || expression.len() > MAX_INPUT_LEN
        || !expression.chars().any(|c| c.is_ascii_digit())
        || !expression
            .chars()
            .any(|c| matches!(c, '+' | '-' | '*' | '/' | '?'))
        || !expression
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '+' | '-' | '*' | '/' | '=' | '?'))
        || expression.matches('=').count() > 1
        || expression.matches('?').count() > 1
    {
        return None;
    }

    // A trailing `=` (e.g. `3+5=`) is just a prompt marker: drop the rhs.
    let (lhs_src, rhs_src) = match expression.split_once('=') {
        Some((lhs, rhs)) if !rhs.is_empty() => (lhs, Some(rhs)),
        Some((lhs, _)) => (lhs, None),
        None => (expression.as_str(), None),
    };

    let lhs = parse_all(lhs_src)?;
    let rhs = match rhs_src {
        Some(src) => Some(parse_all(src)?),
        None => None,
    };

    match (has_unknown(&lhs), rhs.as_ref().is_some_and(has_unknown)) {
        // Unknown on both sides: unsupported.
        (true, true) => None,

        // Unknown on the left: `5 + ? = 8`.
        (true, false) => {
            let target = eval_const(rhs.as_ref()?)?;
            solve_and_finish(&expression, &lhs, target)
        }

        // Unknown on the right: `3 + 5 = ?`.
        (false, true) => {
            let target = eval_const(&lhs)?;
            solve_and_finish(&expression, rhs.as_ref()?, target)
        }

        // No unknown: evaluate the side that contains an operator.
        (false, false) => {
            let node = if contains_op(&lhs) {
                &lhs
            } else {
                rhs.as_ref().unwrap_or(&lhs)
            };
            finish(&expression, eval_const(node)?)
        }
    }
}

/// Solve `node == target` for the unknown, then verify and finish.
fn solve_and_finish(expression: &str, node: &Node, target: f64) -> Option<MathSolution> {
    let x = solve_for(node, target)?;
    let value = eval(node, x)?;
    if (value - target).abs() > REL_EPS * (1.0 + value.abs().max(target.abs())) {
        return None;
    }
    finish(expression, x)
}

fn finish(expression: &str, value: f64) -> Option<MathSolution> {
    if !value.is_finite() || value.abs() > MAX_MAGNITUDE {
        return None;
    }

    let result = (value * 100.0).round() / 100.0;

    Some(MathSolution {
        expression: expression.to_string(),
        result,
        answer: format_number(result),
    })
}

/// Normalize full-width characters, common OCR aliases and Chinese numeral
/// words to an ASCII expression, dropping all whitespace.
fn normalize(text: &str) -> String {
    // Word-level replacements first, longest words before their prefixes.
    let mut text = text.to_string();
    for (from, to) in [
        ("除以", "/"),
        ("乘以", "*"),
        ("加上", "+"),
        ("减去", "-"),
        ("等于", "="),
        ("等於", "="),
        ("多少", "?"),
    ] {
        text = text.replace(from, to);
    }

    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(chars.len());
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        // Chinese numerals can span several chars.
        if is_chinese_numeral(c) {
            let start = i;
            while i < chars.len() && is_chinese_numeral(chars[i]) {
                i += 1;
            }
            if let Some(value) = parse_chinese_number(&chars[start..i]) {
                out.push_str(&format!("{value}"));
            }
            continue;
        }

        i += 1;

        match c {
            '＋' => out.push('+'),
            '－' | '−' | '–' | '—' | '減' => out.push('-'),
            '×' | '✕' | '✖' | '＊' | '*' | 'x' | 'X' => out.push('*'),
            '÷' | '／' => out.push('/'),
            '＝' => out.push('='),
            '？' | '幾' => out.push('?'),
            '加' => out.push('+'),
            '减' => out.push('-'),
            '乘' => out.push('*'),
            '除' => out.push('/'),
            '等' => out.push('='),
            '几' => out.push('?'),
            '０'..='９' => {
                out.push((b'0' + (c as u32 - '０' as u32) as u8) as char);
            }
            '。' | '，' | '、' | ',' => {}
            c if c.is_whitespace() => {}
            c => out.push(c),
        }
    }

    // Collapse repeated unknowns (Chinese question words add a second `?`).
    while out.contains("??") {
        out = out.replace("??", "?");
    }

    out
}

/// Whether `c` participates in a Chinese numeral.
fn is_chinese_numeral(c: char) -> bool {
    matches!(
        c,
        '零' | '〇'
            | '○'
            | '一'
            | '壹'
            | '二'
            | '两'
            | '兩'
            | '贰'
            | '貳'
            | '三'
            | '叁'
            | '四'
            | '肆'
            | '五'
            | '伍'
            | '六'
            | '陆'
            | '陸'
            | '七'
            | '柒'
            | '八'
            | '捌'
            | '九'
            | '玖'
            | '十'
            | '拾'
            | '百'
            | '佰'
            | '千'
            | '仟'
    )
}

/// Parse a Chinese numeral, which can span several characters.
fn parse_chinese_number(chars: &[char]) -> Option<f64> {
    let mut total = 0.0;
    let mut current = 0.0;
    let mut seen = false;

    for &c in chars {
        match c {
            '零' | '〇' | '○' => current = 0.0,
            '一' | '壹' => current = 1.0,
            '二' | '两' | '兩' | '贰' | '貳' => current = 2.0,
            '三' | '叁' => current = 3.0,
            '四' | '肆' => current = 4.0,
            '五' | '伍' => current = 5.0,
            '六' | '陆' | '陸' => current = 6.0,
            '七' | '柒' => current = 7.0,
            '八' | '捌' => current = 8.0,
            '九' | '玖' => current = 9.0,
            '十' | '拾' => {
                total += if current == 0.0 { 1.0 } else { current } * 10.0;
                current = 0.0;
            }
            '百' | '佰' => {
                total += if current == 0.0 { 1.0 } else { current } * 100.0;
                current = 0.0;
            }
            '千' | '仟' => {
                total += if current == 0.0 { 1.0 } else { current } * 1000.0;
                current = 0.0;
            }
            _ => return None,
        }

        seen = true;
    }

    seen.then_some(total + current)
}

/// Format a value with at most 2 decimals, trimming trailing zeros.
fn format_number(value: f64) -> String {
    if (value - value.round()).abs() < EPS {
        return format!("{}", value.round() as i64);
    }

    let text = format!("{value:.2}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[derive(Debug, Clone)]
enum Node {
    Num(f64),
    Unknown,
    Neg(Box<Node>),
    Bin {
        op: Op,
        lhs: Box<Node>,
        rhs: Box<Node>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

fn has_unknown(node: &Node) -> bool {
    match node {
        Node::Num(_) => false,
        Node::Unknown => true,
        Node::Neg(inner) => has_unknown(inner),
        Node::Bin { lhs, rhs, .. } => has_unknown(lhs) || has_unknown(rhs),
    }
}

fn contains_op(node: &Node) -> bool {
    match node {
        Node::Num(_) | Node::Unknown => false,
        Node::Neg(inner) => contains_op(inner),
        Node::Bin { .. } => true,
    }
}

/// Evaluate `node` with the unknown bound to `x`.
fn eval(node: &Node, x: f64) -> Option<f64> {
    let value = match node {
        Node::Num(n) => *n,
        Node::Unknown => x,
        Node::Neg(inner) => -eval(inner, x)?,
        Node::Bin { op, lhs, rhs } => {
            let a = eval(lhs, x)?;
            let b = eval(rhs, x)?;

            match op {
                Op::Add => a + b,
                Op::Sub => a - b,
                Op::Mul => a * b,
                Op::Div => {
                    if b.abs() < EPS {
                        return None;
                    }
                    a / b
                }
            }
        }
    };

    value.is_finite().then_some(value)
}

/// Evaluate a constant node (no unknown).
fn eval_const(node: &Node) -> Option<f64> {
    if has_unknown(node) {
        None
    } else {
        eval(node, 0.0)
    }
}

/// Invert `node` so that `solve_for(node, target)` returns the `x` with
/// `node(x) == target` (when the unknown appears exactly once).
fn solve_for(node: &Node, target: f64) -> Option<f64> {
    match node {
        Node::Unknown => Some(target),
        Node::Num(n) => ((n - target).abs() <= EPS).then_some(target),
        Node::Neg(inner) => solve_for(inner, -target),

        Node::Bin { op, lhs, rhs } => {
            if has_unknown(lhs) {
                let r = eval_const(rhs)?;

                let inner = match op {
                    Op::Add => target - r,
                    Op::Sub => target + r,
                    Op::Mul => target / r,
                    Op::Div => target * r,
                };

                solve_for(lhs, inner)
            } else if has_unknown(rhs) {
                let l = eval_const(lhs)?;

                let inner = match op {
                    Op::Add => target - l,
                    Op::Sub => l - target,
                    Op::Mul => {
                        if l.abs() < EPS {
                            return None;
                        }
                        target / l
                    }
                    Op::Div => {
                        if target.abs() < EPS {
                            return None;
                        }
                        l / target
                    }
                };

                solve_for(rhs, inner)
            } else {
                None
            }
        }
    }
}

/// Parse a complete expression, rejecting trailing garbage.
fn parse_all(src: &str) -> Option<Node> {
    let mut parser = Parser {
        chars: src.chars().collect(),
        pos: 0,
    };

    let node = parser.expr()?;

    (parser.pos == parser.chars.len()).then_some(node)
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    /// `expr := term (('+' | '-') term)*`
    fn expr(&mut self) -> Option<Node> {
        let mut node = self.term()?;

        while let Some(op) = self.peek() {
            let op = match op {
                '+' => Op::Add,
                '-' => Op::Sub,
                _ => break,
            };

            self.pos += 1;
            let rhs = self.term()?;
            node = Node::Bin {
                op,
                lhs: Box::new(node),
                rhs: Box::new(rhs),
            };
        }

        Some(node)
    }

    /// `term := factor (('*' | '/') factor)*`
    fn term(&mut self) -> Option<Node> {
        let mut node = self.factor()?;

        while let Some(op) = self.peek() {
            let op = match op {
                '*' => Op::Mul,
                '/' => Op::Div,
                _ => break,
            };

            self.pos += 1;
            let rhs = self.factor()?;
            node = Node::Bin {
                op,
                lhs: Box::new(node),
                rhs: Box::new(rhs),
            };
        }

        Some(node)
    }

    /// `factor := ('+' | '-')? primary`
    fn factor(&mut self) -> Option<Node> {
        match self.peek() {
            Some('+') => {
                self.pos += 1;
                self.factor()
            }
            Some('-') => {
                self.pos += 1;
                self.factor().map(|node| Node::Neg(Box::new(node)))
            }
            _ => self.primary(),
        }
    }

    /// `primary := number | '?'`
    fn primary(&mut self) -> Option<Node> {
        match self.peek()? {
            '?' => {
                self.pos += 1;
                Some(Node::Unknown)
            }
            c if c.is_ascii_digit() => {
                let start = self.pos;

                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.pos += 1;
                }

                let digits: String = self.chars[start..self.pos].iter().collect();
                digits.parse::<f64>().ok().map(Node::Num)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn answer(text: &str) -> String {
        solve(text)
            .map(|s| s.answer)
            .unwrap_or_else(|| "None".to_string())
    }

    #[test]
    fn evaluates_plain_expressions() {
        assert_eq!(answer("3+5=?"), "8");
        assert_eq!(answer("3+5="), "8");
        assert_eq!(answer("7-2="), "5");
        assert_eq!(answer("4×6="), "24");
        assert_eq!(answer("4x6"), "24");
        assert_eq!(answer("4X6"), "24");
        assert_eq!(answer("9÷3="), "3");
        assert_eq!(answer("9/3"), "3");
    }

    #[test]
    fn solves_single_unknown() {
        assert_eq!(answer("5+?=8"), "3");
        assert_eq!(answer("?×6=42"), "7");
        assert_eq!(answer("12/?=4"), "3");
        assert_eq!(answer("5+2=?"), "7");
        assert_eq!(answer("?=5"), "5");
    }

    #[test]
    fn normalizes_full_width() {
        assert_eq!(answer("３＋５＝"), "8");
        assert_eq!(answer("８－２＝"), "6");
        assert_eq!(answer("２×３＝"), "6");
        assert_eq!(answer("１２÷４＝"), "3");
    }

    #[test]
    fn respects_precedence() {
        assert_eq!(answer("1+2*3"), "7");
        assert_eq!(answer("2*3+4"), "10");
        assert_eq!(answer("10-6/2"), "7");
    }

    #[test]
    fn handles_fractions_and_negatives() {
        assert_eq!(answer("10/4"), "2.5");
        assert_eq!(answer("10/3"), "3.33");
        assert_eq!(answer("-3+5="), "2");
        assert_eq!(answer("5-8="), "-3");
    }

    #[test]
    fn rejects_non_math() {
        assert!(solve("").is_none());
        assert!(solve("42").is_none());
        assert!(solve("K9").is_none());
        assert!(solve("今天天气不错").is_none());
        assert!(solve("5+?").is_none(), "unknown without an equation");
        assert!(solve("?+?=4").is_none(), "two unknowns");
        assert!(solve("1/0").is_none(), "division by zero");
        assert!(solve("1+2=3=4").is_none(), "two equals signs");
        assert!(solve("(1+2)*3").is_none(), "parentheses unsupported");
    }

    #[test]
    fn solves_chinese_numeral_captchas() {
        assert_eq!(answer("九乘六等于？"), "54");
        assert_eq!(answer("三加五等于多少"), "8");
        assert_eq!(answer("十二减四等于几"), "8");
        assert_eq!(answer("十除以二等于？"), "5");
        assert_eq!(answer("五加几等于八"), "3");
        assert_eq!(answer("一百零三减三等于"), "100");
        assert_eq!(answer("二十乘以三等于？"), "60");
        assert_eq!(answer("5加3等于多少"), "8");
        assert_eq!(answer("十減三等于"), "7");
        assert_eq!(answer("两千除以四等于"), "500");
        assert_eq!(answer("一加二"), "3");
    }

    #[test]
    fn keeps_normalized_expression() {
        let solved = solve(" ５ ＋ ? ＝ １３ ").unwrap();
        assert_eq!(solved.expression, "5+?=13");
        assert_eq!(solved.answer, "8");
        assert_eq!(solved.result, 8.0);
    }
}
