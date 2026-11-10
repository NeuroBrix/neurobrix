//! Mask cost per step on a real vocabulary (the owner's gate: < 1 ms CPU on ~150 k tokens): random walks over the
//! oracle's schemas; median / p99 / max per step, and the schema compile time. Usage: bench <vocab file>
use nbxgrammar::{json::parse, matcher::Matcher, schema::compile, vocab::Vocab};
use std::time::Instant;

fn read_vocab(p: &str) -> Vocab {
    let b = std::fs::read(p).unwrap();
    let u = |i: &mut usize| { let v = u32::from_le_bytes(b[*i..*i + 4].try_into().unwrap()); *i += 4; v };
    let mut i = 0;
    let n = u(&mut i) as usize;
    let mut toks = Vec::with_capacity(n);
    for _ in 0..n {
        let l = u(&mut i);
        if l == u32::MAX { toks.push(None) } else { toks.push(Some(b[i..i + l as usize].to_vec())); i += l as usize; }
    }
    let ne = u(&mut i) as usize;
    let eos = (0..ne).map(|_| u(&mut i)).collect();
    Vocab::new(toks, eos)
}

const SCHEMAS: &[&str] = &[
    r#"{"type":"object","properties":{"name":{"type":"string"},"n":{"type":"integer"},"ok":{"type":"boolean"}},"required":["name","n"]}"#,
    r#"{"type":"array","items":{"enum":["alpha","beta",12,1,null]},"minItems":1,"maxItems":8}"#,
    r#"{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"}},"x":{"type":["number","null"]},"o":{"type":"object","properties":{"a":{"const":true}},"required":["a"]}},"required":["tags"]}"#,
    r#"{"type":"object","properties":{"location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]},"days":{"type":"integer"}},"required":["location","unit"],"additionalProperties":false}"#,
];

fn main() {
    let path = std::env::args().nth(1).expect("vocab file");
    let t0 = Instant::now();
    let v = read_vocab(&path);
    println!("vocabulary {} tokens, built (trie + string cache) in {:.0} ms", v.tokens.len(), t0.elapsed().as_secs_f64() * 1e3);
    let mut rng = 0x2545F4914F6CDD1Du64;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let mut times: Vec<f64> = Vec::new();
    let mut compiles: Vec<f64> = Vec::new();
    for src in SCHEMAS {
        let t = Instant::now();
        let schema = compile(&parse(src).unwrap()).unwrap();
        compiles.push(t.elapsed().as_secs_f64() * 1e6);
        for _ in 0..30 {
            let mut m = Matcher::new(&schema);
            let mut mask = vec![0u32; v.words()];
            for _ in 0..80 {
                let t = Instant::now();
                v.mask(&m, &mut mask);
                times.push(t.elapsed().as_secs_f64() * 1e6);
                let allowed: Vec<u32> = (0..v.tokens.len() as u32).filter(|&t| mask[(t >> 5) as usize] >> (t & 31) & 1 == 1).collect();
                let pick = allowed[(next() % allowed.len() as u64) as usize];
                if v.eos.contains(&pick) { break; }
                v.advance(&mut m, pick).unwrap();
            }
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| times[((times.len() as f64 - 1.0) * p) as usize];
    println!("mask per step over {} steps: median {:.1} us, p99 {:.1} us, max {:.1} us", times.len(), q(0.5), q(0.99), times[times.len() - 1]);
    compiles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("schema compile: median {:.1} us, max {:.1} us", compiles[compiles.len() / 2], compiles[compiles.len() - 1]);
}
