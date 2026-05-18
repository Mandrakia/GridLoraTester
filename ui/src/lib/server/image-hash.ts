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

/** Hamming distance below which two images are considered the same. Per
 * BlockHash literature, ≤ 10 / 256 is the standard "near-identical" floor
 * (WhatsApp recompress, sharp resize, light edits, mild crop all under). */
export const DEDUP_HAMMING_THRESHOLD = 10;

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

/** Hamming distance between two same-length hex hash strings. Returns
 * Infinity for any non-matching length (so a malformed hash is treated
 * as "never a dup"). Implementation chunks at 16 hex chars (64 bits) and
 * uses BigInt popcount — fast enough for ~1000 hash comparisons per
 * suggestion render. */
export function hashHamming(a: string, b: string): number {
    if (a.length !== b.length) return Infinity;
    let total = 0;
    for (let i = 0; i < a.length; i += 16) {
        const aa = BigInt('0x' + a.slice(i, i + 16));
        const bb = BigInt('0x' + b.slice(i, i + 16));
        let x = aa ^ bb;
        while (x > 0n) {
            // Brian Kernighan's popcount: clears the lowest set bit each
            // iteration, so loop count = popcount, not bit width.
            x &= x - 1n;
            total++;
        }
    }
    return total;
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
