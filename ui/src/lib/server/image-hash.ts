// Perceptual image hashing for dataset / suggestion dedup. Uses BlockHash
// (blockhash.io) — chosen over pHash/dHash for an OSS-distributed tool
// because:
//   - Public spec (auditable), public-domain algorithm, MIT lib
//   - Spatial mean-per-block ≈ better tolerance to mild crops than DCT-based
//     pHash or gradient-based dHash
//   - 256-bit hash leaves comfortable threshold headroom
//   - blockhash-core has ~50k weekly DL — stable + battle-tested
//
// We treat hashes as fixed-length lowercase hex strings (64 chars = 256 bits).
// Storage in SQLite is TEXT; comparison is via Hamming distance in TS
// (SQLite doesn't ship a popcount).
import sharp from 'sharp';
import { bmvbhash } from 'blockhash-core';

/** Bits per row of the BlockHash grid. `bits=16` → 16×16 blocks → 256-bit
 * hash. Higher bits = more discriminating but more storage; 16 is the
 * blockhash.io recommended default for general-purpose dedup. */
const HASH_BITS = 16;
/** Pre-resize ceiling before BlockHash. The algorithm splits the image
 * into bits×bits blocks; on a 4000×3000 source each block is 250×187
 * pixels, which is fine but slow. Resizing to a canonical 256×256 keeps
 * the hash invariant to the original resolution and bounds compute time
 * to ~5 ms per image. */
const RESIZE_TO = 256;

/** Default Hamming distance below which two images are treated as the same.
 * The live value is the `dedup_hamming_threshold` setting; this is the
 * fallback and the seed default.
 *
 * 10/256 is the BlockHash literature "near-identical" floor (recompress,
 * resize, light edits) — but on real libraries that only catches byte-level
 * re-saves. Burst frames of the same moment, where the subject moves a
 * little, sit around 30-50 bits apart, while genuinely-unrelated photos
 * cluster above ~64 (measured: a clear bimodal gap). 32 catches typical
 * bursts with margin below that valley. */
export const DEDUP_HAMMING_THRESHOLD = 32;

/** Compute a 256-bit perceptual hash from raw image bytes. The image is
 * decoded by Sharp, normalized to RGBA at a fixed canonical size, then
 * hashed by blockhash-core's recommended `bmvbhash` (non-overlapping
 * mean-block hash).
 *
 * Returns null when Sharp can't decode the input (corrupt / unsupported
 * format) — callers should treat null as "no dedup signal for this image",
 * never as an error. */
export async function computeImageHash(bytes: Buffer | Uint8Array): Promise<string | null> {
    try {
        const { data, info } = await sharp(bytes)
            // Honor the EXIF Orientation tag BEFORE hashing. Sharp reads
            // stored-orientation pixels by default; without this, the same
            // photo saved once with orientation baked in and once with the
            // EXIF flag set hashes to two near-opposite values (Hamming ~150)
            // and never dedups. `.rotate()` with no args auto-orients.
            .rotate()
            .ensureAlpha()
            .resize(RESIZE_TO, RESIZE_TO, { fit: 'inside' })
            .raw()
            .toBuffer({ resolveWithObject: true });
        const imageData = {
            width: info.width,
            height: info.height,
            // blockhash-core accepts plain Uint8Array, but Uint8ClampedArray
            // matches the browser ImageData shape its spec doc references.
            data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength)
        };
        return bmvbhash(imageData, HASH_BITS);
    } catch {
        return null;
    }
}

/** Packed hash: a hex hash chopped into N × 32-bit words. Parse once with
 * `parseHashPacked`, then compare with `hashHammingPacked` — avoids the
 * BigInt parse/XOR/popcount overhead that dominated `hashHamming` on hot
 * paths (suggestion dedup did 600k+ comparisons per page load). */
export type PackedHash = Uint32Array;

/** Parse a hex hash into 32-bit words. Returns null for malformed input
 * (length not a multiple of 8 hex chars, or non-hex chars). */
export function parseHashPacked(hex: string): PackedHash | null {
    if (hex.length === 0 || (hex.length & 7) !== 0) return null;
    const n = hex.length >>> 3;
    const out = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        const v = parseInt(hex.slice(i * 8, i * 8 + 8), 16);
        if (!Number.isFinite(v)) return null;
        out[i] = v >>> 0;
    }
    return out;
}

/** SWAR popcount on a 32-bit unsigned int. Constant-time, no branches. */
function popcount32(x: number): number {
    x = x - ((x >>> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
    x = (x + (x >>> 4)) & 0x0f0f0f0f;
    return Math.imul(x, 0x01010101) >>> 24;
}

/** Hamming distance over two pre-parsed hashes. Returns Infinity if word
 * counts differ. ~50-100× faster than the string/BigInt path. */
export function hashHammingPacked(a: PackedHash, b: PackedHash): number {
    if (a.length !== b.length) return Infinity;
    let total = 0;
    for (let i = 0; i < a.length; i++) total += popcount32(a[i] ^ b[i]);
    return total;
}

/** Hamming distance between two same-length hex hash strings. Convenience
 * wrapper around the packed path — DO NOT use in tight loops, parse once
 * with `parseHashPacked` and reuse. Returns Infinity on length mismatch
 * or malformed input (so a bad hash is treated as "never a dup"). */
export function hashHamming(a: string, b: string): number {
    if (a.length !== b.length) return Infinity;
    const pa = parseHashPacked(a);
    const pb = parseHashPacked(b);
    if (!pa || !pb) return Infinity;
    return hashHammingPacked(pa, pb);
}

/** True iff the candidate's hash is within `DEDUP_HAMMING_THRESHOLD` of
 * ANY hash in the reference set. O(|references|) per call — for very
 * large reference sets (10k+) consider a BK-tree, but for our use case
 * (a few hundred dataset images per scope) the linear scan is fine. */
export function isLikelyDuplicate(
    candidateHash: string,
    referenceHashes: Iterable<string>
): boolean {
    for (const ref of referenceHashes) {
        if (hashHamming(candidateHash, ref) <= DEDUP_HAMMING_THRESHOLD) return true;
    }
    return false;
}
