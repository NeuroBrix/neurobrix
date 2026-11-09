//! The byte-level matcher: a stack of small Copy frames fed one byte at a time (cheap to clone for the trie walk).
use crate::schema::{Node, Schema, MAX_NUMBER};

/// whitespace bytes allowed at each place JSON allows them (bounded: an output never runs forever on blanks)
pub const WS_MAX: u8 = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Frame {
    /// the end of the output: nothing more may come
    Done,
    /// a value of `node` may start here
    Value { node: u32, ws: u8 },
    /// inside one of `node`'s literals: `cand` the ones still matching after `pos` bytes
    Lit { node: u32, pos: u16, cand: u64 },
    /// inside a string body: `esc` 0 none, 1 after '\', 2..=5 inside \uXXXX; `need` UTF-8 continuation bytes left,
    /// the first of them in lo..=hi
    Str { esc: u8, need: u8, lo: u8, hi: u8 },
    /// a number: `phase` of the JSON number grammar, `len` bytes so far
    Num { int: bool, phase: u8, len: u8 },
    /// an array: phase 0 after '[', 1 after an item, 2 after ','
    Arr { node: u32, count: u32, phase: u8, ws: u8 },
    /// an object: phase 0 after '{', 1 in a key (pos, cand), 2 after a key, 4 after a value, 5 after ','
    Obj { node: u32, seen: u64, phase: u8, ws: u8, pos: u16, cand: u64, cur: u8 },
}

#[derive(Clone)]
pub struct Matcher<'s> {
    pub schema: &'s Schema,
    pub stack: Vec<Frame>,
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\r' | b'\t')
}

// number phases: 0 after '-', 1 after a leading 0, 2 integer digits, 3 after '.', 4 fraction digits, 5 after e/E,
// 6 after the exponent's sign, 7 exponent digits. End-capable: 1, 2, 4, 7.
fn num_end(phase: u8) -> bool {
    matches!(phase, 1 | 2 | 4 | 7)
}

fn num_step(int: bool, phase: u8, b: u8) -> Option<u8> {
    let d = b.is_ascii_digit();
    match (phase, b) {
        (0, b'0') => Some(1),
        (0, _) if d => Some(2),
        (1, b'.') | (2, b'.') if !int => Some(3),
        (1, b'e') | (1, b'E') | (2, b'e') | (2, b'E') | (4, b'e') | (4, b'E') if !int => Some(5),
        (2, _) if d => Some(2),
        (3, _) | (4, _) if d => Some(4),
        (5, b'+') | (5, b'-') => Some(6),
        (5, _) | (6, _) | (7, _) if d => Some(7),
        _ => None,
    }
}

impl<'s> Matcher<'s> {
    pub fn new(schema: &'s Schema) -> Self {
        Matcher { schema, stack: vec![Frame::Done, Frame::Value { node: schema.root as u32, ws: 0 }] }
    }

    /// The output may end here (its root value is complete, or a number / literal at the root could stop).
    pub fn accepting(&self) -> bool {
        match self.stack.as_slice() {
            [Frame::Done] => true,
            [Frame::Done, Frame::Num { phase, .. }] => num_end(*phase),
            [Frame::Done, Frame::Lit { node, pos, cand }] => self.lit_complete(*node, *pos, *cand),
            _ => false,
        }
    }

    fn lits(&self, node: u32) -> &'s [Vec<u8>] {
        match &self.schema.nodes[node as usize] {
            Node::Literals(l) => l,
            _ => unreachable!(),
        }
    }

    fn lit_complete(&self, node: u32, pos: u16, cand: u64) -> bool {
        let l = self.lits(node);
        (0..l.len()).any(|i| cand >> i & 1 == 1 && l[i].len() == pos as usize)
    }

    fn start_value(&mut self, node: u32, ws: u8, b: u8) -> bool {
        let top = self.stack.len() - 1;
        match &self.schema.nodes[node as usize] {
            _ if is_ws(b) => {
                if ws >= WS_MAX { return false; }
                self.stack[top] = Frame::Value { node, ws: ws + 1 };
                true
            }
            Node::Literals(l) => {
                let mut cand = 0u64;
                for (i, x) in l.iter().enumerate() {
                    if x[0] == b { cand |= 1 << i; }
                }
                if cand == 0 { return false; }
                self.stack[top] = Frame::Lit { node, pos: 1, cand };
                true
            }
            Node::Str => {
                if b != b'"' { return false; }
                self.stack[top] = Frame::Str { esc: 0, need: 0, lo: 0, hi: 0 };
                true
            }
            Node::Integer | Node::Number => {
                let int = matches!(self.schema.nodes[node as usize], Node::Integer);
                let phase = if b == b'-' { 0 } else { match num_step(int, 0, b) { Some(p) => p, None => return false } };
                self.stack[top] = Frame::Num { int, phase, len: 1 };
                true
            }
            Node::Array { min: _, max, .. } => {
                if b != b'[' { return false; }
                let _ = max;
                self.stack[top] = Frame::Arr { node, count: 0, phase: 0, ws: 0 };
                true
            }
            Node::Object { .. } => {
                if b != b'{' { return false; }
                self.stack[top] = Frame::Obj { node, seen: 0, phase: 0, ws: 0, pos: 0, cand: 0, cur: 0 };
                true
            }
            Node::AnyOf(alts) => {
                for &a in alts {
                    if Self::starts(self.schema, a, b) {
                        self.stack[top] = Frame::Value { node: a as u32, ws };
                        return self.feed(b);
                    }
                }
                false
            }
        }
    }

    fn starts(s: &Schema, n: usize, b: u8) -> bool {
        match &s.nodes[n] {
            Node::Literals(l) => l.iter().any(|x| x[0] == b),
            Node::Str => b == b'"',
            Node::Integer | Node::Number => b == b'-' || b.is_ascii_digit(),
            Node::Array { .. } => b == b'[',
            Node::Object { .. } => b == b'{',
            Node::AnyOf(a) => a.iter().any(|&j| Self::starts(s, j, b)),
        }
    }

    /// One byte; false (the matcher unusable after) when the grammar refuses it.
    pub fn feed(&mut self, b: u8) -> bool {
        loop {
            let top = self.stack.len() - 1;
            match self.stack[top] {
                Frame::Done => return false,
                Frame::Value { node, ws } => return self.start_value(node, ws, b),
                Frame::Lit { node, pos, cand } => {
                    let l = self.lits(node);
                    let mut next = 0u64;
                    for i in 0..l.len() {
                        if cand >> i & 1 == 1 && l[i].len() > pos as usize && l[i][pos as usize] == b {
                            next |= 1 << i;
                        }
                    }
                    if next != 0 {
                        self.stack[top] = Frame::Lit { node, pos: pos + 1, cand: next };
                        return true;
                    }
                    if self.lit_complete(node, pos, cand) {
                        self.stack.pop();
                        continue;
                    }
                    return false;
                }
                Frame::Str { esc, need, lo, hi } => {
                    let f = if need > 0 {
                        if b < lo || b > hi { return false; }
                        Frame::Str { esc: 0, need: need - 1, lo: 0x80, hi: 0xBF }
                    } else if esc == 1 {
                        match b {
                            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => Frame::Str { esc: 0, need: 0, lo: 0, hi: 0 },
                            b'u' => Frame::Str { esc: 2, need: 0, lo: 0, hi: 0 },
                            _ => return false,
                        }
                    } else if esc >= 2 {
                        if !b.is_ascii_hexdigit() { return false; }
                        Frame::Str { esc: if esc == 5 { 0 } else { esc + 1 }, need: 0, lo: 0, hi: 0 }
                    } else {
                        match b {
                            b'"' => { self.stack.pop(); return true; }
                            b'\\' => Frame::Str { esc: 1, need: 0, lo: 0, hi: 0 },
                            0x00..=0x1F => return false,
                            0x20..=0x7F => return true,
                            0xC2..=0xDF => Frame::Str { esc: 0, need: 1, lo: 0x80, hi: 0xBF },
                            0xE0 => Frame::Str { esc: 0, need: 2, lo: 0xA0, hi: 0xBF },
                            0xED => Frame::Str { esc: 0, need: 2, lo: 0x80, hi: 0x9F },
                            0xE1..=0xEF => Frame::Str { esc: 0, need: 2, lo: 0x80, hi: 0xBF },
                            0xF0 => Frame::Str { esc: 0, need: 3, lo: 0x90, hi: 0xBF },
                            0xF4 => Frame::Str { esc: 0, need: 3, lo: 0x80, hi: 0x8F },
                            0xF1..=0xF3 => Frame::Str { esc: 0, need: 3, lo: 0x80, hi: 0xBF },
                            _ => return false,
                        }
                    };
                    self.stack[top] = f;
                    return true;
                }
                Frame::Num { int, phase, len } => {
                    if let Some(p) = num_step(int, phase, b) {
                        if (len as usize) < MAX_NUMBER {
                            self.stack[top] = Frame::Num { int, phase: p, len: len + 1 };
                            return true;
                        }
                    }
                    if num_end(phase) {
                        self.stack.pop();
                        continue;
                    }
                    return false;
                }
                Frame::Arr { node, count, phase, ws } => {
                    let (items, min, max) = match self.schema.nodes[node as usize] {
                        Node::Array { items, min, max } => (items as u32, min, max),
                        _ => unreachable!(),
                    };
                    if is_ws(b) {
                        if ws >= WS_MAX { return false; }
                        self.stack[top] = Frame::Arr { node, count, phase, ws: ws + 1 };
                        return true;
                    }
                    match (phase, b) {
                        (0, b']') | (1, b']') => {
                            if count < min { return false; }
                            self.stack.pop();
                            return true;
                        }
                        (1, b',') => {
                            if count >= max { return false; }
                            self.stack[top] = Frame::Arr { node, count, phase: 2, ws: 0 };
                            return true;
                        }
                        (0, _) | (2, _) => {
                            if count >= max { return false; }
                            self.stack[top] = Frame::Arr { node, count: count + 1, phase: 1, ws: 0 };
                            self.stack.push(Frame::Value { node: items, ws: 0 });
                            if !self.feed(b) { return false; }
                            return true;
                        }
                        _ => return false,
                    }
                }
                Frame::Obj { node, seen, phase, ws, pos, cand, cur } => {
                    let (keys, values, required) = match &self.schema.nodes[node as usize] {
                        Node::Object { keys, values, required } => (keys, values, *required),
                        _ => unreachable!(),
                    };
                    let all = if keys.len() == 64 { u64::MAX } else { (1u64 << keys.len()) - 1 };
                    if phase != 1 && is_ws(b) {
                        if ws >= WS_MAX { return false; }
                        self.stack[top] = Frame::Obj { node, seen, phase, ws: ws + 1, pos, cand, cur };
                        return true;
                    }
                    match (phase, b) {
                        (0, b'}') | (4, b'}') => {
                            if required & !seen != 0 { return false; }
                            self.stack.pop();
                            return true;
                        }
                        (0, b'"') | (5, b'"') => {
                            let unseen = all & !seen;
                            if unseen == 0 { return false; }
                            self.stack[top] = Frame::Obj { node, seen, phase: 1, ws: 0, pos: 1, cand: unseen, cur };
                            return true;
                        }
                        (1, _) => {
                            let mut next = 0u64;
                            let mut done: Option<usize> = None;
                            for i in 0..keys.len() {
                                if cand >> i & 1 == 1 && keys[i].len() > pos as usize && keys[i][pos as usize] == b {
                                    next |= 1 << i;
                                    if keys[i].len() == pos as usize + 1 { done = Some(i); }
                                }
                            }
                            if next == 0 { return false; }
                            self.stack[top] = match done {
                                Some(i) => Frame::Obj { node, seen: seen | 1 << i, phase: 2, ws: 0, pos: 0, cand: 0, cur: i as u8 },
                                None => Frame::Obj { node, seen, phase: 1, ws: 0, pos: pos + 1, cand: next, cur },
                            };
                            return true;
                        }
                        (2, b':') => {
                            self.stack[top] = Frame::Obj { node, seen, phase: 4, ws: 0, pos: 0, cand: 0, cur };
                            self.stack.push(Frame::Value { node: values[cur as usize] as u32, ws: 0 });
                            return true;
                        }
                        (4, b',') => {
                            if all & !seen == 0 { return false; }
                            self.stack[top] = Frame::Obj { node, seen, phase: 5, ws: 0, pos: 0, cand: 0, cur };
                            return true;
                        }
                        _ => return false,
                    }
                }
            }
        }
    }

    /// Every byte of `bytes`; false at the first refused one.
    pub fn feed_all(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&b| self.feed(b))
    }
}
