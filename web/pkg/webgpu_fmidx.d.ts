/* tslint:disable */
/* eslint-disable */

/**
 * A builder that accumulates DNA sequences and constructs an FM-index.
 *
 * ```js
 * const builder = new FmIndexBuilder(32);
 * builder.add_sequence("ACGTACGT");
 * const handle = await builder.build_gpu();  // GPU path
 * // or: const handle = builder.build_cpu(); // CPU path (sync)
 * ```
 */
export class FmIndexBuilder {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Add sequences from a FASTA string or newline-separated plain sequences.
     */
    add_fasta(fasta: string): void;
    /**
     * Add a single DNA sequence (ACGT characters, case-insensitive).
     */
    add_sequence(seq: string): void;
    /**
     * Build the FM-index using the CPU (synchronous).
     */
    build_cpu(): FmIndexHandle;
    /**
     * Build the FM-index using the GPU (asynchronous, returns a Promise).
     *
     * Falls back gracefully: if GPU initialization fails the promise rejects
     * with the error message.
     */
    build_gpu(): Promise<FmIndexHandle>;
    /**
     * Clear all staged sequences.
     */
    clear(): void;
    /**
     * Create a new builder.
     *
     * `sa_sample_rate`: controls the trade-off between locate speed and memory.
     * Higher = less memory, slower locate queries. Default: 32.
     */
    constructor(sa_sample_rate?: number | null);
    /**
     * Number of sequences currently staged.
     */
    sequence_count(): number;
}

/**
 * A built FM-index, ready for queries.
 *
 * Obtain one from `FmIndexBuilder.build_cpu()` or `await FmIndexBuilder.build_gpu()`.
 */
export class FmIndexHandle {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Count occurrences of `pattern` (ACGT string) in the indexed sequences.
     *
     * Returns 0 if the pattern is not found.
     */
    count(pattern: string): number;
    /**
     * Deserialize an FM-index from bytes previously produced by `to_bytes()`.
     */
    static from_bytes(data: Uint8Array): FmIndexHandle;
    /**
     * Locate all occurrence positions of `pattern`.
     *
     * Returns a `Uint32Array` of text positions (0-based, in concatenated text).
     */
    locate(pattern: string): Uint32Array;
    /**
     * Number of sequences in the index.
     */
    num_sequences(): number;
    /**
     * Total length of the indexed text (including per-sequence sentinel characters).
     */
    text_len(): number;
    /**
     * Serialize the FM-index to a `Uint8Array` for storage or transfer.
     */
    to_bytes(): Uint8Array;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_fmindexbuilder_free: (a: number, b: number) => void;
    readonly __wbg_fmindexhandle_free: (a: number, b: number) => void;
    readonly fmindexbuilder_add_fasta: (a: number, b: number, c: number) => [number, number];
    readonly fmindexbuilder_add_sequence: (a: number, b: number, c: number) => [number, number];
    readonly fmindexbuilder_build_cpu: (a: number) => [number, number, number];
    readonly fmindexbuilder_build_gpu: (a: number) => any;
    readonly fmindexbuilder_clear: (a: number) => void;
    readonly fmindexbuilder_new: (a: number) => number;
    readonly fmindexbuilder_sequence_count: (a: number) => number;
    readonly fmindexhandle_count: (a: number, b: number, c: number) => [number, number, number];
    readonly fmindexhandle_from_bytes: (a: number, b: number) => [number, number, number];
    readonly fmindexhandle_locate: (a: number, b: number, c: number) => [number, number, number, number];
    readonly fmindexhandle_num_sequences: (a: number) => number;
    readonly fmindexhandle_text_len: (a: number) => number;
    readonly fmindexhandle_to_bytes: (a: number) => [number, number, number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h465b7f6cc7174e01: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h23411f3a1b5b82b2: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h0360c65602824ded: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
