//! Time formatting functions (strftime/strptime).

use super::struct_time::StructTime;
use std::fmt::Write;

const WEEKDAY_ABBR: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const WEEKDAY_FULL: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];
const MONTH_ABBR: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
const MONTH_FULL: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// Format a struct_time according to the given format string.
pub fn strftime(format: &str, time: &StructTime) -> Result<String, FormatError> {
    let mut result = String::with_capacity(format.len() * 2);
    let mut chars = format.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '%' {
            match chars.next() {
                Some('%') => result.push('%'),
                Some('a') => {
                    let wday = (time.tm_wday + 1) % 7;
                    result.push_str(WEEKDAY_ABBR[wday as usize]);
                }
                Some('A') => {
                    let wday = (time.tm_wday + 1) % 7;
                    result.push_str(WEEKDAY_FULL[wday as usize]);
                }
                Some('w') => {
                    let wday = (time.tm_wday + 1) % 7;
                    write!(result, "{}", wday).unwrap();
                }
                Some('d') => write!(result, "{:02}", time.tm_mday).unwrap(),
                Some('b') | Some('h') => {
                    if time.tm_mon >= 1 && time.tm_mon <= 12 {
                        result.push_str(MONTH_ABBR[(time.tm_mon - 1) as usize]);
                    }
                }
                Some('B') => {
                    if time.tm_mon >= 1 && time.tm_mon <= 12 {
                        result.push_str(MONTH_FULL[(time.tm_mon - 1) as usize]);
                    }
                }
                Some('m') => write!(result, "{:02}", time.tm_mon).unwrap(),
                Some('y') => write!(result, "{:02}", time.tm_year % 100).unwrap(),
                Some('Y') => write!(result, "{}", time.tm_year).unwrap(),
                Some('H') => write!(result, "{:02}", time.tm_hour).unwrap(),
                Some('I') => {
                    let h12 = match time.tm_hour {
                        0 => 12,
                        h if h > 12 => h - 12,
                        h => h,
                    };
                    write!(result, "{:02}", h12).unwrap();
                }
                Some('p') => result.push_str(if time.tm_hour < 12 { "AM" } else { "PM" }),
                Some('M') => write!(result, "{:02}", time.tm_min).unwrap(),
                Some('S') => write!(result, "{:02}", time.tm_sec).unwrap(),
                Some('f') => result.push_str("000000"),
                Some('z') => {
                    if let Some(off) = time.tm_gmtoff {
                        let s = if off >= 0 { '+' } else { '-' };
                        let a = off.abs();
                        write!(result, "{}{:02}{:02}", s, a / 3600, (a % 3600) / 60).unwrap();
                    }
                }
                Some('Z') => {
                    if let Some(z) = time.tm_zone {
                        result.push_str(z);
                    }
                }
                Some('j') => write!(result, "{:03}", time.tm_yday).unwrap(),
                Some('U') => {
                    let w = (time.tm_yday + 6 - (time.tm_wday + 1) % 7) / 7;
                    write!(result, "{:02}", w).unwrap();
                }
                Some('W') => {
                    let w = (time.tm_yday + 6 - time.tm_wday) / 7;
                    write!(result, "{:02}", w).unwrap();
                }
                Some('c') => {
                    let wd = (time.tm_wday + 1) % 7;
                    let mn = if time.tm_mon >= 1 && time.tm_mon <= 12 {
                        MONTH_ABBR[(time.tm_mon - 1) as usize]
                    } else {
                        "???"
                    };
                    write!(
                        result,
                        "{} {} {:2} {:02}:{:02}:{:02} {}",
                        WEEKDAY_ABBR[wd as usize],
                        mn,
                        time.tm_mday,
                        time.tm_hour,
                        time.tm_min,
                        time.tm_sec,
                        time.tm_year
                    )
                    .unwrap();
                }
                Some('x') => write!(
                    result,
                    "{:02}/{:02}/{:02}",
                    time.tm_mon,
                    time.tm_mday,
                    time.tm_year % 100
                )
                .unwrap(),
                Some('X') => write!(
                    result,
                    "{:02}:{:02}:{:02}",
                    time.tm_hour, time.tm_min, time.tm_sec
                )
                .unwrap(),
                Some('e') => write!(result, "{:2}", time.tm_mday).unwrap(),
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some(ch) => {
                    result.push('%');
                    result.push(ch);
                }
                None => result.push('%'),
            }
        } else {
            result.push(c);
        }
    }
    Ok(result)
}

/// Parse a time string according to the given format.
pub fn strptime(string: &str, format: &str) -> Result<StructTime, FormatError> {
    let mut tm_year = 1900;
    let mut tm_mon = 1;
    let mut tm_mday = 1;
    let mut tm_hour = 0;
    let mut tm_min = 0;
    let mut tm_sec = 0;
    let mut tm_wday = 0;
    let mut tm_yday = 0;
    let tm_isdst = -1;

    let mut str_chars = string.chars().peekable();
    let mut fmt_chars = format.chars().peekable();

    while let Some(fc) = fmt_chars.next() {
        if fc == '%' {
            match fmt_chars.next() {
                Some('%') => {
                    if str_chars.next() != Some('%') {
                        return Err(FormatError::UnexpectedCharacter);
                    }
                }
                Some('Y') => tm_year = parse_int(&mut str_chars, 4)?,
                Some('y') => {
                    let y = parse_int(&mut str_chars, 2)?;
                    tm_year = if y >= 69 { 1900 + y } else { 2000 + y };
                }
                Some('m') => tm_mon = parse_int(&mut str_chars, 2)?,
                Some('d') => tm_mday = parse_int(&mut str_chars, 2)?,
                Some('H') => tm_hour = parse_int(&mut str_chars, 2)?,
                Some('I') => tm_hour = parse_int(&mut str_chars, 2)?,
                Some('M') => tm_min = parse_int(&mut str_chars, 2)?,
                Some('S') => tm_sec = parse_int(&mut str_chars, 2)?,
                Some('j') => tm_yday = parse_int(&mut str_chars, 3)?,
                Some('p') => {
                    let mut ap = String::new();
                    while str_chars.peek().map(|c| c.is_alphabetic()).unwrap_or(false)
                        && ap.len() < 2
                    {
                        ap.push(str_chars.next().unwrap());
                    }
                    if ap.eq_ignore_ascii_case("PM") && tm_hour < 12 {
                        tm_hour += 12;
                    } else if ap.eq_ignore_ascii_case("AM") && tm_hour == 12 {
                        tm_hour = 0;
                    }
                }
                Some('b') | Some('B') | Some('h') => {
                    let mut ms = String::new();
                    while str_chars.peek().map(|c| c.is_alphabetic()).unwrap_or(false) {
                        ms.push(str_chars.next().unwrap());
                    }
                    tm_mon = parse_month_name(&ms)?;
                }
                Some('a') | Some('A') => {
                    let mut ws = String::new();
                    while str_chars.peek().map(|c| c.is_alphabetic()).unwrap_or(false) {
                        ws.push(str_chars.next().unwrap());
                    }
                    tm_wday = parse_weekday_name(&ws)?;
                }
                Some('w') => {
                    let w = parse_int(&mut str_chars, 1)?;
                    tm_wday = if w == 0 { 6 } else { w - 1 };
                }
                Some('Z') => {
                    while str_chars
                        .peek()
                        .map(|c| c.is_alphabetic() || *c == '/' || *c == '_')
                        .unwrap_or(false)
                    {
                        str_chars.next();
                    }
                }
                Some('z') => {
                    if str_chars.peek() == Some(&'+') || str_chars.peek() == Some(&'-') {
                        str_chars.next();
                        for _ in 0..4 {
                            if str_chars
                                .peek()
                                .map(|c| c.is_ascii_digit())
                                .unwrap_or(false)
                            {
                                str_chars.next();
                            }
                        }
                    }
                }
                Some('f') => {
                    for _ in 0..6 {
                        if str_chars
                            .peek()
                            .map(|c| c.is_ascii_digit())
                            .unwrap_or(false)
                        {
                            str_chars.next();
                        }
                    }
                }
                Some(c) => return Err(FormatError::UnsupportedSpecifier(c)),
                None => return Err(FormatError::IncompleteFormat),
            }
        } else if fc.is_whitespace() {
            while str_chars.peek().map(|c| c.is_whitespace()).unwrap_or(false) {
                str_chars.next();
            }
        } else if str_chars.next() != Some(fc) {
            return Err(FormatError::UnexpectedCharacter);
        }
    }
    if tm_yday == 0 && tm_mon > 0 && tm_mday > 0 {
        tm_yday = super::struct_time::day_of_year(tm_year, tm_mon, tm_mday);
    }
    Ok(StructTime::new(
        tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst,
    ))
}

fn parse_int(
    chars: &mut std::iter::Peekable<std::str::Chars>,
    max_digits: usize,
) -> Result<i32, FormatError> {
    while chars.peek().map(|c| c.is_whitespace()).unwrap_or(false) {
        chars.next();
    }
    let mut s = String::new();
    for _ in 0..max_digits {
        if let Some(&c) = chars.peek() {
            if c.is_ascii_digit() {
                s.push(chars.next().unwrap());
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if s.is_empty() {
        return Err(FormatError::ExpectedNumber);
    }
    s.parse::<i32>().map_err(|_| FormatError::InvalidNumber)
}

fn parse_month_name(s: &str) -> Result<i32, FormatError> {
    let sl = s.to_lowercase();
    for (i, &a) in MONTH_ABBR.iter().enumerate() {
        if sl.starts_with(&a.to_lowercase()) {
            return Ok((i + 1) as i32);
        }
    }
    for (i, &f) in MONTH_FULL.iter().enumerate() {
        if sl == f.to_lowercase() {
            return Ok((i + 1) as i32);
        }
    }
    Err(FormatError::InvalidMonthName)
}

fn parse_weekday_name(s: &str) -> Result<i32, FormatError> {
    let sl = s.to_lowercase();
    for (i, &a) in WEEKDAY_ABBR.iter().enumerate() {
        if sl.starts_with(&a.to_lowercase()) {
            return Ok(if i == 0 { 6 } else { (i - 1) as i32 });
        }
    }
    for (i, &f) in WEEKDAY_FULL.iter().enumerate() {
        if sl == f.to_lowercase() {
            return Ok(if i == 0 { 6 } else { (i - 1) as i32 });
        }
    }
    Err(FormatError::InvalidWeekdayName)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    UnexpectedCharacter,
    ExpectedNumber,
    InvalidNumber,
    InvalidMonthName,
    InvalidWeekdayName,
    UnsupportedSpecifier(char),
    IncompleteFormat,
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatError::UnexpectedCharacter => write!(f, "unexpected character"),
            FormatError::ExpectedNumber => write!(f, "expected number"),
            FormatError::InvalidNumber => write!(f, "invalid number"),
            FormatError::InvalidMonthName => write!(f, "invalid month name"),
            FormatError::InvalidWeekdayName => write!(f, "invalid weekday name"),
            FormatError::UnsupportedSpecifier(c) => write!(f, "unsupported: %{}", c),
            FormatError::IncompleteFormat => write!(f, "incomplete format"),
        }
    }
}

impl std::error::Error for FormatError {}
