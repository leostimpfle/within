#![no_main]

use libfuzzer_sys::fuzz_target;
use within::Preconditioner;

// A `Preconditioner` is picklable and reused across processes, so its bytes can
// originate outside the process that produced them (a cache, another machine, a
// truncated or tampered file). Deserializing arbitrary input must never panic
// or read out of bounds: it either reconstructs a valid preconditioner or
// returns a typed postcard error.
fuzz_target!(|data: &[u8]| {
    let _ = postcard::from_bytes::<Preconditioner>(data);
});
