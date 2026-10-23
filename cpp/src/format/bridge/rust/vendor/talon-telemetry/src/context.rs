//! Bounded W3C carrier. Parsing and cloning never allocate.

/// Owned, validated W3C context; never includes baggage or credentials.
#[derive(Clone, PartialEq, Eq)]
pub struct TraceContext {
    parent: [u8; 55],
    state: [u8; 512],
    state_len: u16,
}

impl std::fmt::Debug for TraceContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TraceContext")
            .field("traceparent", &self.traceparent())
            .finish_non_exhaustive()
    }
}

fn hex(bytes: &[u8]) -> bool {
    bytes
        .iter()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
}

impl TraceContext {
    #[cfg(feature = "recording")]
    pub(crate) fn from_ids(trace: [u8; 16], span: [u8; 8], flags: u8) -> Self {
        let mut result = Self {
            parent: *b"00-00000000000000000000000000000000-0000000000000000-00",
            state: [0; 512],
            state_len: 0,
        };
        let hex = b"0123456789abcdef";
        for (i, b) in trace.iter().enumerate() {
            result.parent[3 + 2 * i] = hex[(b >> 4) as usize];
            result.parent[4 + 2 * i] = hex[(b & 15) as usize];
        }
        for (i, b) in span.iter().enumerate() {
            result.parent[36 + 2 * i] = hex[(b >> 4) as usize];
            result.parent[37 + 2 * i] = hex[(b & 15) as usize];
        }
        result.parent[53] = hex[(flags >> 4) as usize];
        result.parent[54] = hex[(flags & 15) as usize];
        result
    }
    /// Invalid parent returns None. Invalid state is discarded independently.
    /// Future versions accept an extension after the mandatory 55-byte prefix.
    pub fn from_w3c(parent: &str, state: Option<&str>) -> Option<Self> {
        let p = parent.as_bytes();
        if p.len() < 55
            || p.len() > 1024
            || p[2] != b'-'
            || p[35] != b'-'
            || p[52] != b'-'
            || !hex(&p[..2])
            || &p[..2] == b"ff"
            || !hex(&p[3..35])
            || !hex(&p[36..52])
            || !hex(&p[53..55])
            || p[3..35].iter().all(|b| *b == b'0')
            || p[36..52].iter().all(|b| *b == b'0')
            || (&p[..2] == b"00" && p.len() != 55)
            || (p.len() > 55 && (p[55] != b'-' || !p[56..].iter().all(u8::is_ascii_graphic)))
        {
            return None;
        }
        let mut result = Self {
            parent: [0; 55],
            state: [0; 512],
            state_len: 0,
        };
        result.parent.copy_from_slice(&p[..55]);
        // Forward the understood fields using version 00, as required by W3C.
        result.parent[..2].copy_from_slice(b"00");
        if let Some(s) = state.filter(|s| valid_state(s)) {
            result.state[..s.len()].copy_from_slice(s.as_bytes());
            result.state_len = s.len() as u16;
        }
        Some(result)
    }

    /// W3C traceparent text.
    pub fn traceparent(&self) -> &str {
        std::str::from_utf8(&self.parent).unwrap()
    }
    /// Validated tracestate text, possibly empty.
    pub fn tracestate(&self) -> &str {
        std::str::from_utf8(&self.state[..self.state_len as usize]).unwrap()
    }
    /// Parent-based sampling decision.
    pub fn sampled(&self) -> bool {
        self.parent[54].to_digit16() & 1 == 1
    }
}

trait HexDigit {
    fn to_digit16(self) -> u8;
}
impl HexDigit for u8 {
    fn to_digit16(self) -> u8 {
        if self.is_ascii_digit() {
            self - b'0'
        } else {
            self - b'a' + 10
        }
    }
}

fn valid_state(state: &str) -> bool {
    if state.is_empty() || state.len() > 512 {
        return false;
    }
    let mut keys = [""; 32];
    for (i, entry) in state.split(',').enumerate() {
        if i == 32 {
            return false;
        }
        let Some((key, value)) = entry.trim_matches([' ', '\t']).split_once('=') else {
            return false;
        };
        if key.is_empty()
            || key.len() > 256
            || keys[..i].contains(&key)
            || value.is_empty()
            || value.len() > 256
            || value.ends_with(' ')
            || !value
                .bytes()
                .all(|b| (0x20..=0x7e).contains(&b) && b != b',' && b != b'=')
        {
            return false;
        }
        let valid_tail = |s: &str| {
            s.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"_-*/".contains(&b))
        };
        let valid_key = if let Some((tenant, system)) = key.split_once('@') {
            !tenant.is_empty()
                && tenant.len() <= 241
                && tenant.as_bytes()[0].is_ascii_alphanumeric()
                && valid_tail(tenant)
                && !system.is_empty()
                && system.len() <= 14
                && system.as_bytes()[0].is_ascii_lowercase()
                && valid_tail(system)
        } else {
            key.as_bytes()[0].is_ascii_lowercase() && valid_tail(key)
        };
        if !valid_key {
            return false;
        }
        keys[i] = key;
    }
    true
}

/// Parent selection is request-local. Explicit and Root never inherit ambient context.
#[derive(Clone, Copy, Default)]
pub enum TraceParent<'a> {
    #[default]
    Inherit,
    Explicit(&'a TraceContext),
    Root,
}

/// Additional request options; old Rust entrypoints use Inherit.
#[derive(Clone, Copy, Default)]
pub struct RequestOptions<'a> {
    pub parent: TraceParent<'a>,
}

#[cfg(test)]
mod tests {
    use super::*;
    const P: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    #[test]
    fn w3c_validation() {
        assert!(TraceContext::from_w3c(P, Some("vendor=value"))
            .unwrap()
            .sampled());
        assert_eq!(
            TraceContext::from_w3c(P, Some("a=1,a=2"))
                .unwrap()
                .tracestate(),
            ""
        );
        assert!(TraceContext::from_w3c(&P.to_uppercase(), None).is_none());
        assert!(TraceContext::from_w3c(
            &P.replace(
                "4bf92f3577b34da6a3ce929d0e0e4736",
                "00000000000000000000000000000000"
            ),
            None
        )
        .is_none());
        assert!(TraceContext::from_w3c(&format!("{P}-extension"), None).is_none());
        assert!(TraceContext::from_w3c(&format!("01{}-extension", &P[2..]), None).is_some());
        assert!(TraceContext::from_w3c("", None).is_none());
    }
}
