//! A JSON Schema of the supported subset compiled to a node arena (std only). Rigor first: a keyword we do not enforce
//! REFUSES to compile, by name; the grammar may be stricter than the schema (an object generates only its declared
//! properties), never looser.
use crate::json::Json;

pub const MAX_KEYS: usize = 64; // an object's properties, an enum's values: one u64 bitset each
pub const MAX_NUMBER: usize = 32; // bytes of a generated number

#[derive(Debug, Clone)]
pub enum Node {
    Integer,
    Number,
    Str,
    /// enum / const: each value's compact JSON encoding
    Literals(Vec<Vec<u8>>),
    Array { items: usize, min: u32, max: u32 },
    /// keys as their JSON encodings (quotes included: no key is a prefix of another)
    Object { keys: Vec<Vec<u8>>, values: Vec<usize>, required: u64 },
    /// alternatives whose first bytes are disjoint (decided at the first byte, no backtracking)
    AnyOf(Vec<usize>),
    /// exact text of a description (no surrounding whitespace allowed)
    Text(Vec<u8>),
    /// the parts one after the other
    Seq(Vec<usize>),
    /// one of `then[i]`, chosen by the literal `prefixes[i]` read first (no prefix is a prefix of another)
    Choice { prefixes: Vec<Vec<u8>>, then: Vec<usize> },
    /// `node` repeated `min`..=`max` times, `sep` between two
    Repeat { node: usize, min: u32, max: u32, sep: Vec<u8> },
    /// free text (UTF-8, no control byte but newline and tab) ended by the literal `term`, consumed, or by the token
    /// `token` when the template writes that end as one id (a special token has no bytes to spell); `fail` term's KMP table
    Until { term: Vec<u8>, fail: Vec<u16>, token: Option<u32> },
    /// exactly the token `id` (a template's dialect marker, special or plain): no bytes, the id itself
    Token(u32),
}

#[derive(Debug)]
pub struct Schema {
    pub nodes: Vec<Node>,
    pub root: usize,
}

const IGNORED: &[&str] = &["title", "description", "$schema", "$id", "$comment", "default", "examples", "deprecated",
                           "readOnly", "writeOnly", "strict"];

pub fn compile(schema: &Json) -> Result<Schema, String> {
    let mut c = Compiler { nodes: Vec::new(), defs: schema.get("$defs").or(schema.get("definitions")).cloned(), depth: 0 };
    let root = c.node(schema)?;
    Ok(Schema { nodes: c.nodes, root })
}

/// A grammar DESCRIPTION (JSON): {"lit": text} | {"schema": <JSON Schema>} | {"seq": [part...]} |
/// {"choice": [{"lit": prefix, "then": part}...]} | {"repeat": part, "min": m, "max": n, "sep": text} |
/// {"until": terminator}. The tool-call envelopes are written in it by the engine, per template dialect.
pub fn compile_description(d: &Json) -> Result<Schema, String> {
    let mut c = Compiler { nodes: Vec::new(), defs: None, depth: 0 };
    let root = c.part(d)?;
    Ok(Schema { nodes: c.nodes, root })
}

fn kmp(t: &[u8]) -> Vec<u16> {
    let mut f = vec![0u16; t.len()];
    let mut k = 0usize;
    for i in 1..t.len() {
        while k > 0 && t[i] != t[k] { k = f[k - 1] as usize; }
        if t[i] == t[k] { k += 1; }
        f[i] = k as u16;
    }
    f
}

struct Compiler {
    nodes: Vec<Node>,
    defs: Option<Json>,
    depth: usize,
}

impl Compiler {
    fn part(&mut self, d: &Json) -> Result<usize, String> {
        let text = |k: &str| -> Result<Vec<u8>, String> {
            match d.get(k) { Some(Json::Str(s)) if !s.is_empty() => Ok(s.as_bytes().to_vec()),
                             _ => Err(format!("description: {k:?} must be a non-empty string")) }
        };
        if d.get("lit").is_some() && d.get("then").is_none() {
            return Ok(self.push(Node::Text(text("lit")?)));
        }
        if let Some(s) = d.get("schema") {
            let saved = self.defs.take();
            self.defs = s.get("$defs").or(s.get("definitions")).cloned();
            let r = self.node(s);
            self.defs = saved;
            return r;
        }
        if let Some(Json::Array(parts)) = d.get("seq") {
            if parts.is_empty() { return Err("description: an empty seq".into()); }
            let ids = parts.iter().map(|p| self.part(p)).collect::<Result<Vec<_>, _>>()?;
            return Ok(self.push(Node::Seq(ids)));
        }
        if let Some(Json::Array(alts)) = d.get("choice") {
            let mut prefixes = Vec::new();
            let mut then = Vec::new();
            for a in alts {
                let p = match a.get("lit") { Some(Json::Str(s)) if !s.is_empty() => s.as_bytes().to_vec(),
                                             _ => return Err("description: a choice without its literal".into()) };
                prefixes.push(p);
                then.push(self.part(a.get("then").ok_or("description: a choice without then")?)?);
            }
            for i in 0..prefixes.len() { for j in 0..prefixes.len() {
                if i != j && prefixes[j].starts_with(&prefixes[i]) {
                    return Err("description: a choice whose literal is a prefix of another's".into());
                }
            }}
            if prefixes.is_empty() || prefixes.len() > MAX_KEYS { return Err("description: a choice of 1..=64".into()); }
            return Ok(self.push(Node::Choice { prefixes, then }));
        }
        if let Some(r) = d.get("repeat") {
            let num = |k: &str, dflt: u32| -> Result<u32, String> { match d.get(k) {
                None => Ok(dflt), Some(Json::Number(n)) => n.parse().map_err(|_| format!("description: {k}")),
                _ => Err(format!("description: {k}")) } };
            let (min, max) = (num("min", 0)?, num("max", u32::MAX)?);
            if min > max || max == 0 { return Err("description: repeat min..=max with max >= 1".into()); }
            let sep = match d.get("sep") { None => Vec::new(), Some(Json::Str(s)) => s.as_bytes().to_vec(),
                                           _ => return Err("description: sep".into()) };
            let node = self.part(r)?;
            return Ok(self.push(Node::Repeat { node, min, max, sep }));
        }
        if d.get("until").is_some() {
            let term = text("until")?;
            let fail = kmp(&term);
            let token = match d.get("end_token") {
                None => None,
                Some(Json::Number(n)) => Some(n.parse().map_err(|_| "description: end_token id")?),
                _ => return Err("description: end_token must be an id".into()),
            };
            return Ok(self.push(Node::Until { term, fail, token }));
        }
        if let Some(t) = d.get("token") {
            return match t { Json::Number(n) => Ok(self.push(Node::Token(n.parse().map_err(|_| "description: token id")?))),
                             _ => Err("description: token must be an id".into()) };
        }
        Err("description: a part that is none of lit, schema, seq, choice, repeat, until, token".into())
    }

    fn push(&mut self, n: Node) -> usize {
        self.nodes.push(n);
        self.nodes.len() - 1
    }

    fn node(&mut self, s: &Json) -> Result<usize, String> {
        self.depth += 1;
        if self.depth > 32 {
            return Err("unsupported: a $ref chain or nesting deeper than 32 (recursive schemas are not supported)".into());
        }
        let r = self.node_inner(s);
        self.depth -= 1;
        r
    }

    fn node_inner(&mut self, s: &Json) -> Result<usize, String> {
        let kv = match s {
            Json::Object(kv) => kv,
            Json::Bool(true) => return Err("unsupported: the schema `true` (any value)".into()),
            _ => return Err("unsupported: a schema that is not an object".into()),
        };
        for (k, v) in kv {
            let known = matches!(k.as_str(), "type" | "properties" | "required" | "additionalProperties" | "items" |
                "minItems" | "maxItems" | "enum" | "const" | "anyOf" | "$ref" | "$defs" | "definitions") || IGNORED.contains(&k.as_str());
            if !known {
                return Err(format!("unsupported: the keyword {k:?}"));
            }
            if k == "additionalProperties" && *v != Json::Bool(false) {
                return Err("unsupported: additionalProperties other than false".into());
            }
        }
        // the keywords each branch enforces: one beside it that the branch would not apply is refused, never dropped
        let enforced: Vec<&str> = kv.iter().map(|(k, _)| k.as_str())
            .filter(|k| !IGNORED.contains(k) && !matches!(*k, "$defs" | "definitions")).collect();
        let alone = |key: &str, allowed: &[&str]| -> Result<(), String> {
            match enforced.iter().find(|k| **k != key && !allowed.contains(k)) {
                Some(other) => Err(format!("unsupported: the keyword {other:?} beside {key:?} (it would not be applied)")),
                None => Ok(()),
            }
        };
        if let Some(r) = s.get("$ref") {
            alone("$ref", &[])?;
            let path = r.as_str().ok_or("unsupported: a non-string $ref")?;
            let name = path.strip_prefix("#/$defs/").or(path.strip_prefix("#/definitions/"))
                .ok_or_else(|| format!("unsupported: $ref {path:?} (only #/$defs/<name>)"))?;
            let target = self.defs.as_ref().and_then(|d| d.get(name)).cloned()
                .ok_or_else(|| format!("unsupported: $ref {path:?} names no definition"))?;
            return self.node(&target);
        }
        if let Some(alts) = s.get("anyOf") {
            alone("anyOf", &[])?;
            let alts = match alts { Json::Array(a) if !a.is_empty() => a, _ => return Err("unsupported: an empty anyOf".into()) };
            let ids = alts.iter().map(|a| self.node(a)).collect::<Result<Vec<_>, _>>()?;
            return self.any_of(ids);
        }
        if s.get("const").is_some() || s.get("enum").is_some() {
            let key = if s.get("const").is_some() { "const" } else { "enum" };
            alone(key, &["type"])?;
            let vals: Vec<Json> = match (s.get("const"), s.get("enum")) {
                (Some(c), _) => vec![c.clone()],
                (None, Some(Json::Array(a))) if !a.is_empty() => a.clone(),
                _ => return Err("unsupported: an empty or non-array enum".into()),
            };
            if vals.len() > MAX_KEYS {
                return Err(format!("unsupported: an enum of {} values (at most {MAX_KEYS})", vals.len()));
            }
            if let Some(t) = s.get("type") {
                let types: Vec<&str> = match t {
                    Json::Str(x) => vec![x.as_str()],
                    Json::Array(xs) => xs.iter().filter_map(|x| x.as_str()).collect(),
                    _ => return Err("unsupported: a non-string type".into()),
                };
                for v in &vals {
                    let kind = match v {
                        Json::Null => "null", Json::Bool(_) => "boolean", Json::Str(_) => "string",
                        Json::Array(_) => "array", Json::Object(_) => "object",
                        Json::Number(n) => if n.contains(['.', 'e', 'E']) { "number" } else { "integer" },
                    };
                    if !types.iter().any(|t| *t == kind || (*t == "number" && kind == "integer")) {
                        return Err(format!("unsupported: the {key} value {} is not of the type {t:?}", v.to_compact()));
                    }
                }
            }
            return Ok(self.push(Node::Literals(vals.iter().map(|v| v.to_compact().into_bytes()).collect())));
        }
        match s.get("type") {
            Some(Json::Str(t)) => self.typed(t, s),
            Some(Json::Array(ts)) => {
                let mut ids = Vec::new();
                for t in ts {
                    ids.push(self.typed(t.as_str().ok_or("unsupported: a non-string type")?, s)?);
                }
                self.any_of(ids)
            }
            _ => Err("unsupported: a schema without a type, enum, const, anyOf or $ref".into()),
        }
    }

    fn any_of(&mut self, ids: Vec<usize>) -> Result<usize, String> {
        let mut seen = [false; 256];
        for &i in &ids {
            for b in self.first_bytes(i) {
                if seen[b as usize] {
                    return Err("unsupported: anyOf alternatives that can start with the same byte (they must differ by JSON type or first character)".into());
                }
                seen[b as usize] = true;
            }
        }
        Ok(self.push(Node::AnyOf(ids)))
    }

    pub fn first_bytes(&self, i: usize) -> Vec<u8> {
        match &self.nodes[i] {
            Node::Integer | Node::Number => b"-0123456789".to_vec(),
            Node::Str => vec![b'"'],
            Node::Literals(l) => { let mut v: Vec<u8> = l.iter().map(|x| x[0]).collect(); v.sort(); v.dedup(); v }
            Node::Array { .. } => vec![b'['],
            Node::Object { .. } => vec![b'{'],
            Node::AnyOf(a) => a.iter().flat_map(|&j| self.first_bytes(j)).collect(),
            Node::Text(t) => vec![t[0]],
            Node::Seq(parts) => parts.first().map_or(vec![], |&j| self.first_bytes(j)),
            Node::Choice { prefixes, .. } => { let mut v: Vec<u8> = prefixes.iter().map(|x| x[0]).collect(); v.sort(); v.dedup(); v }
            Node::Repeat { node, .. } => self.first_bytes(*node),
            Node::Until { .. } => (0u8..=255).collect(),
            Node::Token(_) => vec![],
        }
    }

    fn typed(&mut self, t: &str, s: &Json) -> Result<usize, String> {
        Ok(match t {
            "null" => self.push(Node::Literals(vec![b"null".to_vec()])),
            "boolean" => self.push(Node::Literals(vec![b"true".to_vec(), b"false".to_vec()])),
            "integer" => self.push(Node::Integer),
            "number" => self.push(Node::Number),
            "string" => self.push(Node::Str),
            "array" => {
                let items = match s.get("items") {
                    Some(i) => self.node(i)?,
                    None => return Err("unsupported: an array without items".into()),
                };
                let num = |k: &str| -> Result<Option<u32>, String> {
                    match s.get(k) {
                        None => Ok(None),
                        Some(Json::Number(n)) => n.parse::<u32>().map(Some).map_err(|_| format!("unsupported: {k} {n}")),
                        Some(_) => Err(format!("unsupported: a non-number {k}")),
                    }
                };
                let (min, max) = (num("minItems")?.unwrap_or(0), num("maxItems")?.unwrap_or(u32::MAX));
                if min > max {
                    return Err("unsupported: minItems above maxItems".into());
                }
                self.push(Node::Array { items, min, max })
            }
            "object" => {
                let props = match s.get("properties") {
                    Some(Json::Object(p)) => p.clone(),
                    None => Vec::new(),
                    _ => return Err("unsupported: non-object properties".into()),
                };
                if props.len() > MAX_KEYS {
                    return Err(format!("unsupported: an object of {} properties (at most {MAX_KEYS})", props.len()));
                }
                let mut keys = Vec::new();
                let mut values = Vec::new();
                for (k, v) in &props {
                    keys.push(Json::Str(k.clone()).to_compact().into_bytes());
                    values.push(self.node(v)?);
                }
                let mut required = 0u64;
                if let Some(r) = s.get("required") {
                    let names = match r { Json::Array(a) => a, _ => return Err("unsupported: non-array required".into()) };
                    for n in names {
                        let n = n.as_str().ok_or("unsupported: a non-string required name")?;
                        let i = props.iter().position(|(k, _)| k == n)
                            .ok_or_else(|| format!("unsupported: required {n:?} is not a declared property"))?;
                        required |= 1 << i;
                    }
                }
                self.push(Node::Object { keys, values, required })
            }
            other => return Err(format!("unsupported: the type {other:?}")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;
    #[test]
    fn every_unenforced_construct_refuses_to_compile_by_name() {
        for (s, word) in [
            (r#"{"type":"string","pattern":"a+"}"#, "pattern"), (r#"{"type":"string","format":"date"}"#, "format"),
            (r#"{"type":"integer","minimum":3}"#, "minimum"), (r#"{"type":"object","additionalProperties":true}"#, "additionalProperties"),
            (r#"{"oneOf":[{"type":"string"}]}"#, "oneOf"), (r#"{"type":"array"}"#, "without items"),
            (r#"{"anyOf":[{"type":"integer"},{"type":"number"}]}"#, "same byte"), (r#"{"type":"string","maxLength":4}"#, "maxLength"),
            (r##"{"$ref":"#/$defs/n","$defs":{"n":{"type":"object","properties":{"c":{"$ref":"#/$defs/n"}}}}}"##, "deeper"),
            (r#"{"anyOf":[{"type":"string"},{"type":"null"}],"type":"object","required":["a"]}"#, "beside \"anyOf\""),
            (r##"{"$ref":"#/$defs/s","maxLength":3,"$defs":{"s":{"type":"string"}}}"##, "maxLength"),
            (r##"{"$ref":"#/$defs/s","type":"integer","$defs":{"s":{"type":"string"}}}"##, "beside \"$ref\""),
            (r#"{"type":"string","enum":["a",1]}"#, "not of the type"),
            (r#"{"const":3,"items":{"type":"string"}}"#, "beside \"const\""),
        ] {
            let e = compile(&parse(s).unwrap()).unwrap_err();
            assert!(e.contains(word), "{s}: {e}");
        }
    }
    #[test]
    fn the_supported_subset_compiles() {
        let s = r##"{"type":"object","properties":{"name":{"type":"string"},"tags":{"type":"array","items":{"enum":["a","b"]},"maxItems":3},
            "n":{"type":["integer","null"]},"o":{"$ref":"#/$defs/p"}},"required":["name"],"additionalProperties":false,
            "$defs":{"p":{"type":"object","properties":{"x":{"type":"boolean"}}}}}"##;
        compile(&parse(s).unwrap()).unwrap();
    }
}
