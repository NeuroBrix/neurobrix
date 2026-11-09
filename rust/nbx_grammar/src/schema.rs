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

struct Compiler {
    nodes: Vec<Node>,
    defs: Option<Json>,
    depth: usize,
}

impl Compiler {
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
        if let Some(r) = s.get("$ref") {
            let path = r.as_str().ok_or("unsupported: a non-string $ref")?;
            let name = path.strip_prefix("#/$defs/").or(path.strip_prefix("#/definitions/"))
                .ok_or_else(|| format!("unsupported: $ref {path:?} (only #/$defs/<name>)"))?;
            let target = self.defs.as_ref().and_then(|d| d.get(name)).cloned()
                .ok_or_else(|| format!("unsupported: $ref {path:?} names no definition"))?;
            return self.node(&target);
        }
        for (k, v) in kv {
            let known = matches!(k.as_str(), "type" | "properties" | "required" | "additionalProperties" | "items" |
                "minItems" | "maxItems" | "enum" | "const" | "anyOf" | "$defs" | "definitions") || IGNORED.contains(&k.as_str());
            if !known {
                return Err(format!("unsupported: the keyword {k:?}"));
            }
            if k == "additionalProperties" && *v != Json::Bool(false) {
                return Err("unsupported: additionalProperties other than false".into());
            }
        }
        if let Some(c) = s.get("const") {
            return Ok(self.push(Node::Literals(vec![c.to_compact().into_bytes()])));
        }
        if let Some(e) = s.get("enum") {
            let vals = match e { Json::Array(a) if !a.is_empty() => a, _ => return Err("unsupported: an empty or non-array enum".into()) };
            if vals.len() > MAX_KEYS {
                return Err(format!("unsupported: an enum of {} values (at most {MAX_KEYS})", vals.len()));
            }
            return Ok(self.push(Node::Literals(vals.iter().map(|v| v.to_compact().into_bytes()).collect())));
        }
        if let Some(alts) = s.get("anyOf") {
            let alts = match alts { Json::Array(a) if !a.is_empty() => a, _ => return Err("unsupported: an empty anyOf".into()) };
            let ids = alts.iter().map(|a| self.node(a)).collect::<Result<Vec<_>, _>>()?;
            return self.any_of(ids);
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
