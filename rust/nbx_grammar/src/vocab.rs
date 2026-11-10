//! The tokenizer's tokens as bytes, in a flat trie, with the masks precomputed once per vocabulary: XGrammar's split of
//! context-independent tokens (decided by the lexical state alone, cached) from the rest (walked at each step).
use crate::matcher::{Frame, Matcher};

struct Trie {
    byte: Vec<u8>,
    first_child: Vec<u32>,
    next_sibling: Vec<u32>,
    tok_start: Vec<u32>,
    tok_len: Vec<u32>,
    toks: Vec<u32>,
}

const NONE: u32 = u32::MAX;

impl Trie {
    fn build(tokens: &[Option<Vec<u8>>], keep: impl Fn(usize) -> bool) -> Trie {
        // children as sorted vectors while building, flattened after
        let mut children: Vec<Vec<(u8, u32)>> = vec![Vec::new()];
        let mut ends: Vec<Vec<u32>> = vec![Vec::new()];
        for (id, t) in tokens.iter().enumerate() {
            let Some(t) = t else { continue };
            if t.is_empty() || !keep(id) { continue; }
            let mut n = 0usize;
            for &b in t {
                let next = match children[n].binary_search_by_key(&b, |&(c, _)| c) {
                    Ok(k) => children[n][k].1 as usize,
                    Err(k) => {
                        children.push(Vec::new());
                        ends.push(Vec::new());
                        let m = children.len() - 1;
                        children[n].insert(k, (b, m as u32));
                        m
                    }
                };
                n = next;
            }
            ends[n].push(id as u32);
        }
        let n = children.len();
        let mut tr = Trie { byte: vec![0; n], first_child: vec![NONE; n], next_sibling: vec![NONE; n],
                            tok_start: vec![0; n], tok_len: vec![0; n], toks: Vec::new() };
        for i in 0..n {
            let ch = &children[i];
            if let Some(&(_, f)) = ch.first() { tr.first_child[i] = f; }
            for w in ch.windows(2) { tr.next_sibling[w[0].1 as usize] = w[1].1; }
            for &(b, c) in ch { tr.byte[c as usize] = b; }
            tr.tok_start[i] = tr.toks.len() as u32;
            tr.tok_len[i] = ends[i].len() as u32;
            tr.toks.extend_from_slice(&ends[i]);
        }
        tr
    }

    fn walk(&self, node: u32, m: &Matcher, out: &mut [u32]) {
        let mut c = self.first_child[node as usize];
        while c != NONE {
            let mut m2 = m.clone();
            if m2.feed(self.byte[c as usize]) {
                let (s, l) = (self.tok_start[c as usize] as usize, self.tok_len[c as usize] as usize);
                for &t in &self.toks[s..s + l] {
                    out[(t >> 5) as usize] |= 1 << (t & 31);
                }
                self.walk(c, &m2, out);
            }
            c = self.next_sibling[c as usize];
        }
    }
}

pub struct Vocab {
    pub tokens: Vec<Option<Vec<u8>>>,
    pub eos: Vec<u32>,
    all: Trie,
    /// tokens that leave a plain string body plain (bytes >= 0x20, no '"' or '\', whole UTF-8 characters)
    string_safe: Vec<u32>,
    /// every other token, walked when the matcher is in a plain string body
    string_rest: Trie,
    /// per first byte of an until terminator, built on first use: the tokens that keep free text clean (whole
    /// characters, no control byte but newline and tab, never that byte), and the trie of the others, walked
    until: Vec<std::sync::OnceLock<(Vec<u32>, Trie)>>,
}

fn until_safe(t: &[u8], first: u8) -> bool {
    // free text fed `t` from a clean point stays clean and never starts the terminator
    let mut i = 0;
    while i < t.len() {
        let b = t[i];
        if b == first { return false; }
        let n = match b {
            b'\n' | b'\t' | 0x20..=0x7F => 1,
            0xC2..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF4 => 4,
            _ => return false,
        };
        if i + n > t.len() || std::str::from_utf8(&t[i..i + n]).is_err() { return false; }
        i += n;
    }
    true
}

fn string_safe(t: &[u8]) -> bool {
    // a plain string body fed `t` must end plain, never having left the string
    let mut i = 0;
    while i < t.len() {
        let b = t[i];
        let n = match b {
            0x20..=0x7F if b != b'"' && b != b'\\' => 1,
            0xC2..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF4 => 4,
            _ => return false,
        };
        if i + n > t.len() || std::str::from_utf8(&t[i..i + n]).is_err() { return false; }
        i += n;
    }
    true
}

impl Vocab {
    pub fn new(tokens: Vec<Option<Vec<u8>>>, eos: Vec<u32>) -> Vocab {
        let words = (tokens.len() + 31) / 32;
        let mut safe = vec![0u32; words];
        for (id, t) in tokens.iter().enumerate() {
            if let Some(t) = t {
                if !t.is_empty() && string_safe(t) { safe[id >> 5] |= 1 << (id & 31); }
            }
        }
        let all = Trie::build(&tokens, |_| true);
        let rest = Trie::build(&tokens, |id| safe[id >> 5] >> (id & 31) & 1 == 0);
        let until = (0..256).map(|_| std::sync::OnceLock::new()).collect();
        Vocab { tokens, eos, all, string_safe: safe, string_rest: rest, until }
    }

    pub fn words(&self) -> usize {
        (self.tokens.len() + 31) / 32
    }

    /// The tokens allowed after what `m` has consumed, into `out` (ceil(vocab / 32) int32 words).
    pub fn mask(&self, m: &Matcher, out: &mut [u32]) {
        out.iter_mut().for_each(|w| *w = 0);
        if matches!(m.stack.last(), Some(Frame::Str { esc: 0, need: 0, .. })) {
            out.copy_from_slice(&self.string_safe);
            self.string_rest.walk(0, m, out);
        } else if let Some(first) = m.until_clean() {
            let (safe, rest) = self.until[first as usize].get_or_init(|| {
                let mut safe = vec![0u32; self.words()];
                for (id, t) in self.tokens.iter().enumerate() {
                    if let Some(t) = t { if !t.is_empty() && until_safe(t, first) { safe[id >> 5] |= 1 << (id & 31); } }
                }
                let rest = Trie::build(&self.tokens, |id| safe[id >> 5] >> (id & 31) & 1 == 0);
                (safe, rest)
            });
            out.copy_from_slice(safe);
            rest.walk(0, m, out);
        } else {
            self.all.walk(0, m, out);
        }
        for t in m.token_steps() {
            if (t as usize) < self.tokens.len() { out[(t >> 5) as usize] |= 1 << (t & 31); }
        }
        if m.accepting() {
            for &e in &self.eos { out[(e >> 5) as usize] |= 1 << (e & 31); }
        }
    }

    /// The token chosen: its bytes consumed (an EOS when the output may end); Err when the grammar refuses it.
    pub fn advance(&self, m: &mut Matcher, token: u32) -> Result<(), String> {
        if self.eos.contains(&token) {
            return if m.accepting() { m.stack.clear(); m.stack.push(Frame::Done); Ok(()) }
                   else { Err(format!("token {token} (end of sequence) before the output may end")) };
        }
        if m.token_steps().contains(&token) {
            let mut m2 = m.clone();
            return if m2.feed_token(token) { *m = m2; Ok(()) } else { Err(format!("token {token} is not allowed here")) };
        }
        let t = self.tokens.get(token as usize).and_then(|t| t.as_ref())
            .ok_or_else(|| format!("token {token} has no bytes a grammar may emit"))?;
        let mut m2 = m.clone();
        if m2.feed_all(t) { *m = m2; Ok(()) } else { Err(format!("token {token} is not allowed here")) }
    }
}
