//! The byte level: every byte as a character, and back.
//!
//! A byte-level vocabulary holds characters, not bytes, so that a token is
//! text and a file of tokens is readable. The mapping is the one the published
//! byte-level tokenisers use: the bytes that are already printable stand for
//! themselves, and the rest are moved up into a range that is printable, in
//! order, so that no byte is lost and none collides.
//!
//! It is a bijection over all 256 bytes, which is why any text at all can be
//! encoded: there is no byte without a character and no unknown token.

/// The character each byte is written as.
pub(crate) fn alphabet() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut moved = 0u32;
    for byte in 0..=255u8 {
        out[byte as usize] = if printable(byte) {
            char::from(byte)
        } else {
            // The next free place above the 256 bytes, in the order the bytes
            // that needed moving were met.
            let moved_to = char::from_u32(256 + moved).unwrap_or('\u{fffd}');
            moved += 1;
            moved_to
        };
    }
    out
}

/// Is this byte one the mapping leaves where it is?
///
/// The three runs are the printable ASCII, and the two runs of printable
/// Latin-1 either side of the soft hyphen, which is not printable.
const fn printable(byte: u8) -> bool {
    matches!(byte, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_byte_has_its_own_character() {
        let alphabet = alphabet();
        let mut seen = alloc_set();
        for byte in 0..=255u8 {
            assert!(
                seen.insert(alphabet[byte as usize]),
                "two bytes share a character"
            );
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn the_printable_bytes_stand_for_themselves() {
        let alphabet = alphabet();
        assert_eq!(alphabet[usize::from(b'A')], 'A');
        assert_eq!(alphabet[usize::from(b' ')], 'Ġ');
        assert_eq!(alphabet[usize::from(b'\n')], 'Ċ');
        assert_eq!(alphabet[0xad], 'Ń');
    }

    fn alloc_set() -> std::collections::BTreeSet<char> {
        std::collections::BTreeSet::new()
    }
}
