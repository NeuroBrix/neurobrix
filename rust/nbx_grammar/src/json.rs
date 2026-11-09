//! A small JSON reader for schemas (std only): objects keep their key order, numbers keep their source text.

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(String),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self { Some(s) } else { None }
    }
    /// The value written back as compact JSON (keys in their order): what a `const` or an `enum` entry must produce.
    pub fn to_compact(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }
    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Number(n) => out.push_str(n),
            Json::Str(s) => write_str(s, out),
            Json::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 { out.push(','); }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Object(kv) => {
                out.push('{');
                for (i, (k, v)) in kv.iter().enumerate() {
                    if i > 0 { out.push(','); }
                    write_str(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

pub fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn parse(text: &str) -> Result<Json, String> {
    let mut p = Parser { b: text.as_bytes(), i: 0 };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing bytes at offset {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

const MAX_DEPTH: usize = 128;

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\n' | b'\r' | b'\t') {
            self.i += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }
    fn expect(&mut self, lit: &[u8]) -> Result<(), String> {
        if self.b[self.i..].starts_with(lit) {
            self.i += lit.len();
            Ok(())
        } else {
            Err(format!("expected {:?} at offset {}", std::str::from_utf8(lit).unwrap(), self.i))
        }
    }
    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return Err(format!("nesting deeper than {}", MAX_DEPTH));
        }
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.expect(b"true").map(|_| Json::Bool(true)),
            Some(b'f') => self.expect(b"false").map(|_| Json::Bool(false)),
            Some(b'n') => self.expect(b"null").map(|_| Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(format!("unexpected byte {:?} at offset {}", c as char, self.i)),
            None => Err("unexpected end".into()),
        }
    }
    fn object(&mut self, depth: usize) -> Result<Json, String> {
        self.i += 1;
        let mut kv = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Object(kv));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err(format!("expected a key at offset {}", self.i));
            }
            let k = self.string()?;
            self.ws();
            self.expect(b":")?;
            self.ws();
            let v = self.value(depth + 1)?;
            kv.push((k, v));
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => { self.i += 1; return Ok(Json::Object(kv)); }
                _ => return Err(format!("expected ',' or '}}' at offset {}", self.i)),
            }
        }
    }
    fn array(&mut self, depth: usize) -> Result<Json, String> {
        self.i += 1;
        let mut a = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Array(a));
        }
        loop {
            self.ws();
            a.push(self.value(depth + 1)?);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => { self.i += 1; return Ok(Json::Array(a)); }
                _ => return Err(format!("expected ',' or ']' at offset {}", self.i)),
            }
        }
    }
    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() {
            return Err("short \\u escape".into());
        }
        let s = std::str::from_utf8(&self.b[self.i..self.i + 4]).map_err(|_| "bad \\u escape")?;
        let v = u32::from_str_radix(s, 16).map_err(|_| format!("bad \\u escape at offset {}", self.i))?;
        self.i += 4;
        Ok(v)
    }
    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out = Vec::new();
        loop {
            let c = *self.b.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).map_err(|_| "invalid UTF-8 in a string".into()),
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) {
                                self.expect(b"\\u")?;
                                let lo = self.hex4()?;
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            }
                            let ch = char::from_u32(cp).ok_or("invalid \\u code point")?;
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return Err(format!("bad escape at offset {}", self.i - 1)),
                    }
                }
                c if c < 0x20 => return Err(format!("control byte in a string at offset {}", self.i - 1)),
                c => out.push(c),
            }
        }
    }
    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.peek() == Some(b'-') { self.i += 1; }
        match self.peek() {
            Some(b'0') => self.i += 1,
            Some(c) if c.is_ascii_digit() => while self.peek().map_or(false, |c| c.is_ascii_digit()) { self.i += 1; },
            _ => return Err(format!("bad number at offset {}", start)),
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            if !self.peek().map_or(false, |c| c.is_ascii_digit()) { return Err(format!("bad fraction at offset {}", self.i)); }
            while self.peek().map_or(false, |c| c.is_ascii_digit()) { self.i += 1; }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) { self.i += 1; }
            if !self.peek().map_or(false, |c| c.is_ascii_digit()) { return Err(format!("bad exponent at offset {}", self.i)); }
            while self.peek().map_or(false, |c| c.is_ascii_digit()) { self.i += 1; }
        }
        Ok(Json::Number(String::from_utf8(self.b[start..self.i].to_vec()).unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_a_schema_and_writes_it_back_compact() {
        let s = r#" {"type": "object", "properties": {"a": {"enum": ["x\"y", 1.5e3, null, true]}}, "required": ["a"]} "#;
        let v = parse(s).unwrap();
        assert_eq!(v.get("type").unwrap().as_str(), Some("object"));
        assert_eq!(v.to_compact(), r#"{"type":"object","properties":{"a":{"enum":["x\"y",1.5e3,null,true]}},"required":["a"]}"#);
    }
    #[test]
    fn refuses_what_is_not_json() {
        for bad in ["{", "[1,]", "{\"a\" 1}", "01", "1.", "\"\\x\"", "tru", "{} x", "\"a\nb\""] {
            assert!(parse(bad).is_err(), "{bad}");
        }
        assert_eq!(parse("\"\\ud83c\\udf4e\"").unwrap(), Json::Str("🍎".into()));
    }
}
