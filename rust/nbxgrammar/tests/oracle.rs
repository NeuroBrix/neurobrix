//! The engine against two oracles: (1) the mask equals the brute force (every token's bytes fed to a fresh copy of the
//! matcher; EOS only where the output may end); (2) every output a random walk completes parses as JSON and validates
//! against its schema (an independent validator, here only).
use nbxgrammar::json::{parse, Json};
use nbxgrammar::matcher::Matcher;
use nbxgrammar::schema::{compile, compile_description};
use nbxgrammar::vocab::Vocab;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

fn vocab(rng: &mut Rng) -> Vocab {
    let mut toks: Vec<Option<Vec<u8>>> = (0u16..256).map(|b| Some(vec![b as u8])).collect();
    for s in ["{\"", "\":", "\": ", "\",", "\"}", "},", "}", "]", "[", "[\"", "\"]", " \"", "\"name", "name", "tags", "\"tags\":",
              "true", "false", "null", "nul", "12", "0", "-1", ".5", "e+", "1e3", "é", "日本", "🍎", "\\n", "\\\"", "\\u00e9",
              "a\"b", "\", \"", ", ", "\n  ", "   ", "\"a\"", "\"b\"", "\"x\"", "\"n\": 1", "\"o\": {", "\"ok\"", "abc"] {
        toks.push(Some(s.as_bytes().to_vec()));
    }
    toks.push(Some(vec![0xE6, 0x97])); // a partial UTF-8 character
    toks.push(None); // a special token
    for _ in 0..300 {
        let n = 1 + rng.below(5);
        let pool = b"{}[]\":, \nabnxyz01-.e\\";
        toks.push(Some((0..n).map(|_| pool[rng.below(pool.len())]).collect()));
    }
    let eos = toks.len() as u32;
    toks.push(None);
    Vocab::new(toks, vec![eos])
}

fn brute(v: &Vocab, m: &Matcher) -> Vec<u32> {
    let mut out = vec![0u32; v.words()];
    for (id, t) in v.tokens.iter().enumerate() {
        if let Some(t) = t {
            let mut m2 = m.clone();
            if !t.is_empty() && m2.feed_all(t) { out[id >> 5] |= 1 << (id & 31); }
        }
    }
    for t in m.token_steps() { out[(t >> 5) as usize] |= 1 << (t & 31); }
    if m.accepting() { for &e in &v.eos { out[(e >> 5) as usize] |= 1 << (e & 31); } }
    out
}

fn valid(s: &Json, v: &Json, defs: &Json) -> bool {
    if let Some(Json::Str(r)) = s.get("$ref") {
        return valid(defs.get(r.trim_start_matches("#/$defs/")).unwrap(), v, defs);
    }
    if let Some(c) = s.get("const") { return c == v; }
    if let Some(Json::Array(e)) = s.get("enum") { return e.contains(v); }
    if let Some(Json::Array(a)) = s.get("anyOf") { return a.iter().any(|x| valid(x, v, defs)); }
    let types: Vec<&str> = match s.get("type") {
        Some(Json::Str(t)) => vec![t.as_str()],
        Some(Json::Array(ts)) => ts.iter().map(|t| t.as_str().unwrap()).collect(),
        _ => return false,
    };
    types.iter().any(|t| match (*t, v) {
        ("null", Json::Null) | ("boolean", Json::Bool(_)) | ("string", Json::Str(_)) => true,
        ("number", Json::Number(_)) => true,
        ("integer", Json::Number(n)) => !n.contains(['.', 'e', 'E']),
        ("array", Json::Array(a)) => {
            let num = |k| s.get(k).map(|x| if let Json::Number(n) = x { n.parse::<usize>().unwrap() } else { 0 });
            a.len() >= num("minItems").unwrap_or(0) && a.len() <= num("maxItems").unwrap_or(usize::MAX)
                && a.iter().all(|x| valid(s.get("items").unwrap(), x, defs))
        }
        ("object", Json::Object(kv)) => {
            let props = s.get("properties");
            let req_ok = match s.get("required") {
                Some(Json::Array(r)) => r.iter().all(|n| kv.iter().any(|(k, _)| Some(k.as_str()) == n.as_str())),
                _ => true,
            };
            req_ok && kv.iter().all(|(k, x)| props.and_then(|p| p.get(k)).map_or(false, |ps| valid(ps, x, defs)))
                && (1..kv.len()).all(|i| kv[..i].iter().all(|(k, _)| *k != kv[i].0))
        }
        _ => false,
    })
}

const SCHEMAS: &[&str] = &[
    r#"{"type":"object","properties":{"name":{"type":"string"},"n":{"type":"integer"},"ok":{"type":"boolean"}},"required":["name","n"]}"#,
    r#"{"type":"array","items":{"enum":["a","b",12,1,null]},"minItems":1,"maxItems":4}"#,
    r#"{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"}},"x":{"type":["number","null"]},"o":{"type":"object","properties":{"a":{"const":true}},"required":["a"]}},"required":["tags"]}"#,
    r#"{"type":"number"}"#,
    r#"{"anyOf":[{"type":"string"},{"type":"array","items":{"type":"integer"},"maxItems":2},{"type":"null"}]}"#,
];

#[test]
fn the_mask_equals_the_brute_force_and_every_output_validates() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let v = vocab(&mut rng);
    let (mut checked, mut completed) = (0, 0);
    for src in SCHEMAS {
        let sj = parse(src).unwrap();
        let schema = compile(&sj).unwrap();
        for _ in 0..40 {
            let mut m = Matcher::new(&schema);
            let mut out: Vec<u8> = Vec::new();
            for _step in 0..60 {
                let mut mask = vec![0u32; v.words()];
                v.mask(&m, &mut mask);
                assert_eq!(mask, brute(&v, &m), "schema {src} after {:?}", String::from_utf8_lossy(&out));
                checked += 1;
                let allowed: Vec<u32> = (0..v.tokens.len() as u32).filter(|&t| mask[(t >> 5) as usize] >> (t & 31) & 1 == 1).collect();
                assert!(!allowed.is_empty(), "a dead end after {:?}", String::from_utf8_lossy(&out));
                let eos = v.eos[0];
                let t = if allowed.contains(&eos) && rng.below(3) == 0 { eos } else {
                    let non: Vec<u32> = allowed.iter().copied().filter(|&t| t != eos).collect();
                    if non.is_empty() { eos } else { non[rng.below(non.len())] }
                };
                v.advance(&mut m, t).unwrap();
                if t == eos {
                    let text = String::from_utf8(out.clone()).expect("UTF-8");
                    let val = parse(&text).unwrap_or_else(|e| panic!("{src}: {text:?} is not JSON: {e}"));
                    assert!(valid(&sj, &val, sj.get("$defs").unwrap_or(&Json::Null)), "{src}: {text} does not validate");
                    completed += 1;
                    break;
                }
                out.extend_from_slice(v.tokens[t as usize].as_ref().unwrap());
            }
        }
    }
    assert!(checked > 2000 && completed > 50, "checked {checked} masks, completed {completed} outputs");
}


/// A hermes-style tool-call envelope written as a grammar description: the markers as token steps (ids 600 and 601 here,
/// tokens without bytes), the function chosen by its name's literal, its arguments by their schema, one or two calls.
fn hermes(tool_open: u32, tool_close: u32) -> String {
    let call = |name: &str, params: &str| format!(r#"{{"lit": "\"{name}\"", "then": {{"seq": [{{"lit": ", \"arguments\": "}}, {{"schema": {params}}}, {{"lit": "}}\n"}}, {{"token": {tool_close}}}]}}}}"#);
    format!(r#"{{"repeat": {{"seq": [{{"token": {tool_open}}}, {{"lit": "\n{{\"name\": "}}, {{"choice": [{}, {}]}}]}}, "min": 1, "max": 2, "sep": "\n"}}"#,
            call("get_weather", r#"{"type":"object","properties":{"city":{"type":"string"},"unit":{"enum":["c","f"]}},"required":["city"]}"#),
            call("get_time", r#"{"type":"object","properties":{"tz":{"type":"string"}},"required":["tz"]}"#))
}

#[test]
fn an_envelope_with_token_steps_equals_the_brute_force_and_yields_calls_the_parser_reads() {
    let mut rng = Rng(0xD1B54A32D192ED03);
    let mut v = vocab(&mut rng);
    let open = v.tokens.len() as u32;          // two marker tokens without bytes (as a special token)
    let mut toks = v.tokens.clone();
    toks.extend([None, None]);
    for s in ["get_weather", "\"get_", "weather\"", "\"get_time\"", ", \"arguments\": ", "\n{\"name\": ", "\"city\": \"", "}\n"] {
        toks.push(Some(s.as_bytes().to_vec()));
    }
    let eos = toks.len() as u32;
    toks.push(None);
    v = Vocab::new(toks, vec![eos]);
    let desc = parse(&hermes(open, open + 1)).unwrap();
    let g = compile_description(&desc).unwrap();
    let mut completed = 0;
    for _ in 0..60 {
        let mut m = Matcher::new(&g);
        let mut out: Vec<u8> = Vec::new();
        for _ in 0..200 {
            let mut mask = vec![0u32; v.words()];
            v.mask(&m, &mut mask);
            assert_eq!(mask, brute(&v, &m), "after {:?}", String::from_utf8_lossy(&out));
            let allowed: Vec<u32> = (0..v.tokens.len() as u32).filter(|&t| mask[(t >> 5) as usize] >> (t & 31) & 1 == 1).collect();
            assert!(!allowed.is_empty(), "a dead end after {:?}", String::from_utf8_lossy(&out));
            let t = if allowed.contains(&eos) && rng.below(2) == 0 { eos } else {
                let non: Vec<u32> = allowed.iter().copied().filter(|&t| t != eos).collect();
                if non.is_empty() { eos } else { non[rng.below(non.len())] }
            };
            v.advance(&mut m, t).unwrap();
            if t == eos { completed += 1; break; }
            out.extend_from_slice(match t { x if x == open => b"<tool_call>".as_slice(), x if x == open + 1 => b"</tool_call>",
                                            _ => v.tokens[t as usize].as_ref().unwrap() });
        }
        if !m.accepting() { continue; }       // the walk's step cap reached before the end
        let text = String::from_utf8(out).expect("the output is UTF-8");
        // every call: <tool_call>\n{"name": NAME, "arguments": ARGS}\n</tool_call>, ARGS valid for NAME
        let calls: Vec<&str> = text.split("<tool_call>").skip(1).collect();
        assert!((1..=2).contains(&calls.len()), "{text}");
        for c in calls {
            let body = c.trim_end_matches(['\n']).trim_end_matches("</tool_call>").trim();
            let j = parse(body).unwrap_or_else(|e| panic!("{text}: {e}"));
            let name = j.get("name").and_then(|n| n.as_str()).unwrap();
            assert!(name == "get_weather" || name == "get_time", "{text}");
            let args = j.get("arguments").unwrap();
            assert!(args.get(if name == "get_weather" { "city" } else { "tz" }).is_some(), "{text}");
        }
    }
    assert!(completed > 20, "completed {completed}");
}

#[test]
fn until_reads_free_text_up_to_its_terminator() {
    let d = parse(r#"{"seq": [{"lit": "<p>"}, {"until": "</p>"}, {"lit": "!"}]}"#).unwrap();
    let g = compile_description(&d).unwrap();
    let mut m = Matcher::new(&g);
    assert!(m.feed_all("<p>a </ b </p".as_bytes()) && !m.accepting());
    assert!(m.feed_all(b">!") && m.accepting());
    let mut m2 = Matcher::new(&g);
    assert!(m2.feed_all(b"<p>x</p>") && !m2.clone().feed(b'?'));
}


#[test]
fn whitespace_is_one_space_inside_a_value_and_none_around_the_root() {
    let g = compile(&parse(r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#).unwrap()).unwrap();
    let ok = |t: &str| { let mut m = Matcher::new(&g); m.feed_all(t.as_bytes()) && m.accepting() };
    assert!(ok(r#"{"a": 1}"#) && ok(r#"{"a":1}"#) && ok(r#"{ "a" : 1 }"#));
    for bad in ["\n{\"a\": 1}", " {\"a\": 1}", "{\"a\":  1}", "{\n\"a\": 1}", "{\"a\":\t1}"] {
        assert!(!ok(bad), "{bad:?} accepted");
    }
    let mut m = Matcher::new(&g);
    assert!(m.feed_all(br#"{"a": 1}"#) && !m.clone().feed(b' ') && !m.feed(b'\n'));      // nothing after the root
}

#[test]
fn a_reasoning_closes_by_its_end_token_and_the_cached_until_mask_equals_the_brute_force() {
    // the reasoning of a Thinking template, closed by `</think>` written as ONE special id (no bytes to spell it) or as
    // its text, then the answer; the mask taken from the until cache equals the brute force at every step
    let mut rng = Rng(0xA0761D6478BD642F);
    let mut toks = vocab(&mut rng).tokens;
    for s in ["</", "think", ">", "</think>", "<", "x<y", "\n\n", "</thi", "nk>\n\n{", "Okay, ", " the user", "日"] {
        toks.push(Some(s.as_bytes().to_vec()));
    }
    let end = toks.len() as u32;
    toks.push(None);                                   // `</think>` as a special token
    let eos = toks.len() as u32;
    toks.push(None);
    let v = Vocab::new(toks, vec![eos]);
    let desc = format!(r#"{{"seq": [{{"until": "</think>", "end_token": {end}}}, {{"lit": "\n\n"}},
                         {{"schema": {{"type": "object", "properties": {{"a": {{"type": "integer"}}}}, "required": ["a"]}}}}]}}"#);
    let g = compile_description(&parse(&desc).unwrap()).unwrap();
    let mut closed_by_token = 0;
    for walk in 0..80 {
        let mut m = Matcher::new(&g);
        let mut in_reasoning = true;
        for step in 0..120 {
            let mut mask = vec![0u32; v.words()];
            v.mask(&m, &mut mask);
            assert_eq!(mask, brute(&v, &m), "walk {walk} step {step}");
            let allowed: Vec<u32> = (0..v.tokens.len() as u32).filter(|&t| mask[(t >> 5) as usize] >> (t & 31) & 1 == 1).collect();
            if in_reasoning && m.until_clean().is_some() { assert!(allowed.contains(&end), "walk {walk} step {step}"); }
            let t = if allowed.contains(&end) && step > 0 && rng.below(25) == 0 { end } else { allowed[rng.below(allowed.len())] };
            if t == eos { break; }
            v.advance(&mut m, t).unwrap();
            if t == end { closed_by_token += 1; in_reasoning = false; }
        }
    }
    assert!(closed_by_token > 20, "closed by the token {closed_by_token} times");
    let plain = compile_description(&parse(r#"{"seq": [{"until": "</think>"}, {"lit": "!"}]}"#).unwrap()).unwrap();
    let mut m = Matcher::new(&plain);
    assert!(m.feed_all(b"hmm ") && m.token_steps().is_empty());   // without end_token the text alone closes it
}


#[test]
fn a_schema_inside_a_description_takes_no_space_before_its_root() {
    // D5 (the rack's Thinking json_schema answer: "\n\n {"): the schema part of a sequence is a JSON root too
    let d = parse(r#"{"seq": [{"lit": "x\n\n"}, {"schema": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}]}"#).unwrap();
    let g = compile_description(&d).unwrap();
    let ok = |t: &str| { let mut m = Matcher::new(&g); m.feed_all(t.as_bytes()) && m.accepting() };
    assert!(ok("x\n\n{\"a\": 1}") && ok("x\n\n{ \"a\" : 1 }"));
    assert!(!ok("x\n\n {\"a\": 1}"), "a space before the part's root");
}
