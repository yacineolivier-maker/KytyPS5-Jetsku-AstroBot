#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_NATIVELEARN_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_NATIVELEARN_H_

// Astro Bot PC (KYTY_NATIVE_MODE=s0, stage S0 of the standalone native renderer,
// docs\CONCEPTION-RENDU-NATIF.md, loader/gameCapture.h "KYTY_NATIVE_MODE=s0").
//
// Learning only: every native draw still runs through the emulator's draw path (N1 splice). Right
// after its pipeline selection (RenderExecutor::ExecutePreparedDraw), the draw is looked up in a
// native pipeline table keyed by NativeKey:
//   (ES program, PS program, record state hash, target formats, primitive, specialization signature)
// where the programs are the record's (SPI_SHADER_PGM_LO_ES/PS from the game's register shadow, else
// the engine shader objects' code), the state hash is the record's (programs + fixed function), the
// target formats are the emulator's resolved ones (a native segment's first draw always takes the
// emulator path, which establishes them) and the specialization signature hashes the
// ResourceSpecialization the emulator chose for each stage (the permutation; plus the plain pixel
// variant choice). A miss inserts an entry: pipeline (layout, set layout, push-descriptor flag),
// both CompiledShaderInfo pointers, mesh info, push-data starts, and per stage a resource recipe.
//
// Recipe: for every V# (info.buffers), T# (info.images), S# (info.samplers) and flat SRT dword
// (srt_reads), where each dword comes from, deduced statically from the permutation's sealed
// ResourcePlan (descriptor_sources / srt_reads IR values):
//   - Const: an immediate;
//   - User: GetUserData(s[n]) (index relative to the plan's user_data_base);
//   - Load: one LoadAddressU32 (kind ScalarAddress) whose pointer halves are two User dwords and
//     whose offset is an immediate: dword at ((hi << 32 | lo) & 48 bits & ~3) + (imm & ~3) + (ofs & ~3),
//     as SrtWalker::ComputeRawAddress does for a non-buffer read; a ReadConst is followed to its
//     flat read; BitCasts are transparent.
// Anything deeper (a pointer loaded from memory, arithmetic, selects, ReadConstBuffer, indirect
// descriptors) marks the program not native-eligible (reason and the stopping opcode reported).
//
// KYTY_NATIVE_VERIFY=1: 1 eligible native draw in 64 is rebuilt from the record + recipe (user data
// from the record: the game's register shadow where known, else the engine's slot SGPR copy) and
// compared with what the normal path committed for the same draw: the snapshot's V#/T#/S# and flat
// SRT, the shader-data user dwords, the push constants (captured in CommitBindings) and the mesh draw
// dwords (captured at the first drawMeshTasks). Guest memory is read through
// LibKernel::Memory::TryReadGpuCleanBacking (never through a protected view); a dword it refuses is
// skipped, not compared.
//
// Table: append-only, open addressing over atomic entry pointers (one writer under the render
// mutex; lock-free readers by acquire loads). Entries are never freed.
//
// S0b (key conflicts): the S0 key missed inputs of the program selection. GetGraphicsPrograms
// derives each stage's program key from registers the record's state hash does not cover: the
// vertex/mesh stage's static state (PA_CL_VS_OUT_CNTL, GE_CNTL and GE_MAX_OUTPUT_PER_SUBGROUP group
// sizes, VGT_GS_MAX_VERT_OUT, GE_NGG_SUBGRP_CNTL, VGT_SHADER_STAGES_EN wave size, the clip-space
// viewport transform baked into the SPIR-V when PA_CL_CLIP_CNTL.CLIP_DISABLE is set (viewport 0
// scale/offset), the GS back half SPI_SHADER_PGM_LO_GS, RSRC1/RSRC2), the pixel stage's (SPI_PS_*,
// SPI_BARYC_CNTL, DB_SHADER_CONTROL, CB_SHADER_MASK, export formats, blend 0 dual source / alpha
// remap, the target export mapping). The key now also mixes a register signature: the emulator's
// DrawRun "targets key" (every context register and the user-config registers but the index offset
// and object id, i.e. DrawRun::StructureKey's register part, which includes ShaderRegisters), the
// stage program registers (HW::Shader: ES/GS/PS program addresses, RSRC1/2, start-instance SGPR)
// and the pixel stage's target export mapping. A conflict that remains is diagnosed from the two
// permutations' identities (PipelineCache::NativePermutationIdentity: source code hash, user-data
// count, code size, first differing static-state word, push start, specialization), counted by
// cause in "Native 10s", and marks the entry conflicted (never emitted natively by S1).
//
// S1 (KYTY_NATIVE_MODE=s1, first subset, renderer/renderDraw.cpp RenderExecutor::NativeS1Emit):
// PredictS1 rebuilds a native op's vertex stage inputs without the emulator's program preparation:
// key base from the record + the bound registers + the segment's target formats, the base key's
// representative entry's recipe evaluated on the bound user SGPRs (loads through
// TryReadGpuCleanBacking), the buffer specialization derived from the rebuilt V#s
// (IR::SpecializeBuffersForNative, null V#s zeroed), the full key looked up. Covered entries
// (s1_covered): one vertex stage and no pixel stage, simple recipes (no control flow), buffers
// only (no images, samplers, GDS, shared memory, uniform fill), no buffer or address writes, no
// embedded vertex fetch, no user clip planes, user data without a prefix or with the GS user-data
// address prefix. KYTY_NATIVE_VERIFY=1: a sampled prediction (1 in 64, and every segment head) is
// not emitted; the normal path runs and Begin compares it (key, entry, program, every snapshot
// dword, user data, flat SRT, specialization, layout); a difference is logged and stops the
// emulator.
//
// S1b (key cost, conflicts, pixel stage):
//  - The register part of the key is no longer the DrawRun targets key (every context register):
//    per-object registers (stencil reference, blend colour, scissors, depth clear...) made a new
//    key for almost every draw of the Geometry passes, i.e. a learning attempt (recipe build,
//    entry) per draw on Thread_Gpu once the table was full. It is now the selection signature
//    (renderDraw.cpp NativeSelectionHash): the context registers program and pipeline selection
//    read (ShaderRegisters, clip control, viewport 0 when CLIP_DISABLE bakes it into the program,
//    VGT_SHADER_STAGES_EN, mode control, AA config / sample control, EQAA, depth bounds, blend 0)
//    plus the user-config GE registers and the stage program registers: a few hundred bytes.
//  - The remaining "vs-static" conflicts (static words 13/16/22/25/28/31) are the embedded vertex
//    fetch words of BuildStageStaticKey: with no LDS word, clip space off and no mesh, the vertex
//    key is 11 words then 3 per fetch resource (registers, attribute, V# dword 3 & 0x7ffff =
//    dst_sel / formats), so words 13, 16, 19, ... are fetch V# formats, read from guest memory
//    through the fetch tables, which no register shows. The key mixes their hash (fetch_hash,
//    with the vertex input strides), outside the base key; a prediction (no fetch: not covered)
//    mixes the no-fetch value.
//  - The pixel stage is covered (images, samplers, buffers; plain variant): the prediction
//    rebuilds both stages' snapshots and their whole specialization (IR::SpecializeForNative);
//    the emission resolves T#/S# through the emulator's PrepareBindings (RepeatStageTextures,
//    TextureBindingMemo certificate, ResolveTexture) and RebindImages, exactly as the normal path.
//  - kS1Shape is broken down by first uncovered property and stage (NativeS1Shape).
//
// S1c (embedded vertex fetch, control flow):
//  - Embedded fetch (a non-NGG vertex program whose AGC metadata names a vertex attribute table
//    and a vertex buffer table, ShaderGetStaticVertexInputInfo): the fetch V#s are not descriptor
//    sources of the resource plan. The program preparation reads, per input semantic, the
//    attribute word attrib[semantic] (V# index, attribute format, byte offset, instance flag) and
//    the V# buffer[index * 4 .. + 4] through the two table pointers in the GS user SGPRs, applies
//    the format to dword 3 and the offset to the base, then merges the attributes into vertex
//    buffers (ShaderDetectBuffers: same stride and instance flag, bases less than a stride apart).
//    The recompiled program reads them as Vulkan vertex attributes (GetAttribute): the pipeline's
//    vertex input state (strides, attribute offsets, formats) and the program key (dword 3 &
//    0x7ffff per resource) depend on them, the draw binds the merged buffers through the
//    BufferCache (AcquireVertexBuffers: one ObtainBuffer per merged guest range, then
//    bindVertexBuffers2 with the guest sizes) and takes its vertex / instance offsets from the user
//    SGPRs the fetch code names (ResolveDrawOffsets). The entry learns the static part (table
//    SGPRs, per resource its destination registers and semantic: FetchTemplate); a prediction
//    rebuilds the rest from the bound SGPRs and guest memory (ShaderRebuildEmbeddedFetch, clean
//    backing reads), mixes its fetch hash (FetchHash, the S1b key part) into the full key, and the
//    emission acquires and binds the vertex buffers and resolves the offsets with the emulator's
//    own functions. Verify compares every rebuilt V#, destination and merged buffer.
//  - Control flow: a plan with control flow activates descriptor sources per block; a source
//    reached by no taken edge is inactive and materializes as zero dwords (a condition that does
//    not evaluate follows both edges). The prediction runs the emulator's own active-source walk
//    (SrtWalker::FindActiveSources over the clean runtime, as MaterializeResources does) on the
//    rebuilt user data, zeroes the inactive sources and skips their reads. A condition read that
//    fails (bytes not clean) falls back (kS1Read): the walk would have followed both edges where the
//    normal path's read may succeed.

// S1d (full census, user-data layout, buffer writes, eligibility cache):
//  - User-data layout. snapshot.user_data is ShaderParams::user_data[0, user_data_count): the
//    stage's user SGPRs copied at an offset (GetShaderParams' user_data_base argument, shader.cpp
//    PrepareProgram): 8 for a merged / NGG vertex program (logical stage Mesh), whose dwords 0..1
//    are the GS user-data address when the program has a GS back half (kGsFront), else 0; 0 for a
//    legacy vertex program and for the pixel stage. The plan's user_data_base (8 for the legacy
//    vertex stage, 0 for Mesh and pixel) is the SGPR number of user_data[0], not a prefix length:
//    S1b/S1c took it for one, which made every legacy vertex program "prefix" (its first SGPRs are
//    not zero) and evaluated the recipes 8 dwords off. The entry now learns the prefix length as
//    user_data size minus the bound SGPR count (DrawView::user_sgpr_count), checks its content
//    (zero with a nonzero GS address bound, or the GS address then zeros; ambiguous otherwise), and
//    the prediction rebuilds prefix + bound SGPRs and evaluates the recipes on the whole user data
//    (recipe indices are user_data indices: SGPR - user_data_base).
//  - Census: every uncovered property of an entry is a bit of s1_mask (GC::S1ShapeBit), reported
//    per slot with the draws whose entry is fully covered (the predicted emitted share before the
//    dynamic checks).
//  - Buffer writes (V# written / atomic) are covered: the emission binds them through the
//    emulator's binding functions (ObtainBuffer / ObtainWrittenBuffer with the precise write ranges
//    from the rebuilt snapshot) and records the normal path's post-draw ShaderWriteBarrier
//    (HasShaderBufferWrites per stage; KYTY_DRAW_WRITE_SINK off: EndRendering first). Image writes,
//    atomics, address writes and GDS stay uncovered (kShWrites / kShBinding).
//  - Eligibility cache: a direct-mapped table keyed by the selection signature (NativeRegsHash,
//    which includes the stage program addresses) and the formats remembers a base entry that is not
//    covered: such a draw returns before the record lookup.

// S1e (cost of the S1 path, permutation variants):
//  - "vs:variant" is a permutation variant, not the plain-pixel variant: the base key (programs,
//    state, formats, registers) has several learned entries, and the one the full key finds (with
//    the rebuilt fetch hash and specialization) has other S1 stages than the base's representative:
//    another resource plan (each program source has its own: an embedded fetch with other V#
//    formats is another vertex program key, hence another source), other recipes, user-data count
//    or fetch template. S1d predicted with the representative's stages only, so these draws ran
//    the whole prediction and then fell back. The prediction now rebuilds with a model entry: the
//    base's last found entry (hint), else the representative; when the found entry has other
//    stages, once more with that entry as the model, and it stands only when that rebuild's key
//    finds the model again (a fixed point: the plan, recipes and template are those of the program
//    the normal path selects for these inputs; verify compares the whole prediction as before). A
//    hint whose key nobody learned retries with the representative. KYTY_NATIVE_S1_VARIANT=0: S1d.
//  - Guest reads: the recipe loads of a stage are grouped at learning time by pointer halves into
//    spans of at most 256 bytes (LoadGroup); the prediction reads a group once, on first use, in one
//    clean backing read (S1d: one TryReadGpuCleanBacking per dword, e.g. 8 per T#), and falls back
//    to the per-dword reads when the span read is refused, so every dword is the per-dword answer.
//    The embedded fetch rebuild reads the attribute words and the V# table by span the same way.
//    KYTY_NATIVE_S1_GROUPED=0: per dword.
//  - KYTY_NATIVE_S1_PROFILE=1: Prof (below).
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "loader/gameCapture.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics::NativeLearn {

struct DwordRecipe {
	enum class Kind : uint8_t { Const, User, Load };
	Kind     kind   = Kind::Const;
	uint8_t  lo     = 0; // Load: user-data index of the pointer's low half
	uint8_t  hi     = 0; // Load: and of its high half
	uint32_t value  = 0; // Const: the value; User: the user-data index
	int64_t  offset = 0; // Load: byte offset added to the dword-aligned pointer
	// S1e (derived from the fields above when the entry is learned; not part of the recipe's
	// identity): the stage's load group of a Load (Entry::S1Stage::groups) and its byte offset in it.
	uint16_t group      = 0xffffu;
	uint16_t group_byte = 0;
};

// S1e: a run of a stage's Load dwords through the same pointer halves, read from guest memory in
// one TryReadGpuCleanBacking call (bytes [offset, offset + bytes) after the dword-aligned pointer).
// A group read that fails (or a range touching a never-mapped address) falls back to the per-dword
// reads, so every evaluated dword is exactly the one the per-dword path gives.
struct LoadGroup {
	uint8_t  lo     = 0;
	uint8_t  hi     = 0;
	int64_t  offset = 0;
	uint32_t bytes  = 0;
	uint32_t at     = 0; // byte position of the group in the prediction's group buffer
};

struct ResourceRecipe {
	uint32_t                   dword_count = 0;
	std::array<DwordRecipe, 8> dwords {};
};

struct StageRecipe {
	std::vector<ResourceRecipe> buffers, images, samplers;
	struct Flat {
		uint32_t    flat_offset = 0;
		DwordRecipe recipe;
	};
	std::vector<Flat> flat;
	uint32_t          reason      = Loader::GameCapture::kS0Simple;
	uint32_t          deep_opcode = 0; // kS0Deep / kS0ConstBuffer: ValueOpcode + 1
};

// S1c: the static part of an embedded vertex fetch (ShaderVertexInputInfo of the learned draw).
struct FetchTemplate {
	int      attrib_reg = -1;
	int      buffer_reg = -1;
	uint32_t count      = 0; // resources_num
	std::array<ShaderVertexDestination, ShaderVertexInputInfo::RES_MAX> dst {}; // register_start, registers_num, attr_id
};

// S1b/S1c: the embedded fetch part of the NativeKey (0 without embedded fetch).
uint64_t FetchHash(const ShaderVertexInputInfo& info);

struct Entry {
	Entry()                        = default;
	Entry(const Entry&)            = delete;
	Entry& operator=(const Entry&) = delete;
	uint64_t                                        key  = 0;
	uint64_t                                        base = 0; // key without the specialization signature
	const PipelineCache::Pipeline*                  pipeline        = nullptr;
	vk::Pipeline                                    vk_pipeline     = nullptr;
	vk::PipelineLayout                              layout          = nullptr;
	vk::DescriptorSetLayout                         set_layout      = nullptr;
	bool                                            push_descriptors = false;
	const ShaderRecompiler::IR::CompiledShaderInfo* vs = nullptr;
	const ShaderRecompiler::IR::CompiledShaderInfo* ps = nullptr;
	uint64_t                                        vs_id = 0, ps_id = 0;
	bool                                            mesh_active = false;
	ShaderMeshInputInfo                             mesh;
	uint32_t                                        push_start[2] {};
	std::array<StageRecipe, 2>                      recipe;
	uint32_t                                        stage_count = 0;
	uint32_t                                        reason      = Loader::GameCapture::kS0Simple;
	uint32_t                                        deep_opcode = 0;
	// S0b: the permutations' identities when learned (conflict diagnosis) and whether a conflict was
	// seen for the key (S1 never emits a conflicted entry).
	std::array<PipelineCache::NativeProgramIdentity, 2> identity {};
	bool                                                plain_pixel     = false;
	bool                                                plain_available = false; // S1b: the pixel permutation has a plain variant
	mutable bool                                        conflicted      = false;
	// The newest pipeline object the normal path selected for the key (a fast-first build is
	// replaced by its optimized one; old objects stay valid). Thread_Gpu only.
	mutable const PipelineCache::Pipeline* current            = nullptr;
	mutable uint64_t                       current_generation = 0; // PipelineGeneration before that selection
	// S1: per stage (vertex/mesh, pixel), the inputs a native rebuild needs; whether the entry is covered.
	struct S1Stage {
		const ShaderRecompiler::IR::ResourcePlan* plan = nullptr;
		uint32_t ud_count    = 0; // snapshot.user_data size (user SGPR base + count)
		uint32_t ud_base     = 0; // the plan's user SGPR base
		uint32_t prefix_mode = 0; // 0: prefix dwords 0; 1: dwords 0..1 the GS user-data address; 2: other / ambiguous
		uint32_t ud_prefix   = 0; // S1d: user_data dwords before the bound user SGPRs (0 or 8)
		uint32_t flat_size   = 0; // flattened SRT dwords
		std::vector<LoadGroup> groups; // S1e: the recipe's loads by pointer (see LoadGroup)
		uint32_t               group_bytes = 0;
	};
	std::array<S1Stage, 2> s1 {};
	bool     viewport_index = false; // the vertex program writes the viewport index
	std::unique_ptr<FetchTemplate> fetch; // S1c: embedded vertex fetch (null: none)
	bool     s1_covered     = false;
	uint32_t s1_shape       = 0; // first uncovered property (Loader::GameCapture::NativeS1Shape)
	uint32_t s1_shape_stage = 0; // and its stage
	uint64_t s1_mask        = 0; // S1d: every uncovered property (GC::S1ShapeBit), 0: covered
	uint32_t write_kinds    = 0; // S1d: GC::NativeS1WriteKind bits of both stages
};

// S1: a native op's rebuilt stages (PredictS1), owned by the session.
struct S1Prediction {
	const Entry*                                                entry = nullptr;
	uint64_t                                                    key   = 0;
	uint32_t                                                    stage_count = 0;
	bool                                                        plain       = false; // S1b: the plain pixel variant
	const PipelineCache::Pipeline*                              pipeline    = nullptr; // entry->current when predicted
	uint64_t                                                    generation  = 0;
	std::array<ShaderRecompiler::IR::ResourceSnapshot, 2>       snapshot;
	std::array<ShaderRecompiler::IR::ResourceSpecialization, 2> specialization;
	// S1c: the rebuilt embedded fetch (fetch: valid; only the fetch fields and `stage`, which the
	// emission sets, are meaningful).
	bool                  fetch      = false;
	uint64_t              fetch_hash = 0;
	ShaderVertexInputInfo vertex;
};
// S1: what PredictS1 reads besides the record.
struct S1Inputs {
	uint32_t        op_flags          = 0;
	uint64_t        regs_hash         = 0; // NativeRegsHash of the bound registers (selection signature)
	uint64_t        formats_hash      = 0; // the segment's resolved target formats (its last normal draw)
	uint32_t        index_count       = 0;
	uint64_t        es_addr           = 0; // bound SPI_SHADER_PGM_LO/HI_ES
	const uint32_t* user_sgpr         = nullptr; // bound GS user SGPRs
	uint32_t        user_sgpr_count   = 0;
	uint32_t        user_sgpr_max     = 0; // S1c: entries of the user_sgpr array (HW::UserSgprInfo::SGPRS_MAX)
	uint64_t        gs_user_data_addr = 0;
	// S1b: the pixel stage.
	bool            pixel              = false;
	uint64_t        ps_addr            = 0;
	const uint32_t* ps_user_sgpr       = nullptr;
	uint32_t        ps_user_sgpr_count = 0;
	uint32_t        lod_plain          = 0; // LodStatsCounter::Plain (0 off, 1 on, 2 verify)
};

// What ExecutePreparedDraw knows about one stage of the draw.
struct StageView {
	const ShaderRecompiler::IR::CompiledShaderInfo*     program     = nullptr;
	const ShaderRecompiler::IR::ResourceSnapshot*       snapshot    = nullptr;
	const ShaderRecompiler::IR::ResourcePlan*           plan        = nullptr;
	const ShaderRecompiler::IR::ResourceSpecialization* spec        = nullptr;
	const std::vector<uint32_t>*                        shader_data = nullptr;
	uint64_t                                            program_id  = 0;
};

struct DrawView {
	uint32_t                       op_flags = 0;
	std::array<StageView, 2>       stages {}; // vertex (ES / mesh), pixel
	uint32_t                       stage_count        = 0;
	uint32_t                       vertex_stage_count = 1;
	bool                           mesh_active = false;
	bool                           plain_pixel = false;
	bool                           indexed     = false;
	const PipelineCache::Pipeline* pipeline    = nullptr;
	const ShaderMeshInputInfo*     mesh        = nullptr;
	uint64_t                       formats_hash = 0;
	uint64_t                       cp_es_addr   = 0; // the emulator's SPI_SHADER_PGM_LO/HI_ES
	uint32_t                       index_count  = 0; // the executed draw's count
	uint64_t                       regs_hash    = 0; // S0b: program-selection register signature
	uint64_t                       pipeline_generation = 0; // the pipeline cache generation before the selection
	uint64_t                       gs_user_data_addr = 0; // bound GS user-data address (prefix check)
	bool                           fetch_embedded    = false; // S1 coverage: vertex stage inputs
	uint32_t                       vertex_buffers    = 0;
	uint32_t                       user_clip_planes  = 0;
	std::array<PipelineCache::NativeProgramIdentity, 2> identity {}; // S0b: the stages' permutations
	uint64_t                       fetch_hash        = 0; // S1b: embedded fetch formats/strides (0: none)
	bool                           plain_available   = false; // S1b: the pixel permutation has a plain variant
	bool                           ps_mip_stats      = false; // S1b: the pixel bindings record mip statistics
	const ShaderVertexInputInfo*   vertex_info       = nullptr; // S1c: the draw's vertex input info
	std::array<uint32_t, 2>        user_sgpr_count {};          // S1d: the bound user SGPR count per stage
	std::array<bool, 2>            buffer_writes {};            // S1d: HasShaderBufferWrites per stage (verify)
};

// S1d: KYTY_NATIVE_S1_WRITES (default 1): buffer-writing programs are emitted natively.
[[nodiscard]] bool S1WritesEnabled();

// S1e: KYTY_NATIVE_S1_PROFILE=1 (diagnostic): TSC sub-step timing of the S1 path on Thread_Gpu,
// per engine thread slot, printed every 10 s ("GameNative s1 profile"): the native emissions (the
// prediction's parts and the emission's), and what S1 adds to the normal-path draws of the slots it
// emits for (the emission attempt that fell back, with its prediction parts when it ran, the
// learning hook, the segment state), with those draws' DrawIndex totals and DrawRun continuations.
// Thread_Gpu only (single writer, the report is printed by that thread).
namespace Prof {
enum Step : uint32_t {
	kChecks,     // NativeS1Emit: segment, kind and register checks before the prediction
	kPredict,    // PredictS1 (whole; its parts below)
	kPLookup,    //   eligibility cache, record lookup, base entry, hint
	kPUserData,  //   user data rebuild
	kPFlow,      //   active-source walk (control flow)
	kPEval,      //   recipe evaluation (guest reads)
	kPSpec,      //   specialization and its hash
	kPFetch,     //   embedded fetch rebuild and hash
	kPKey,       //   full key lookup, coverage, variant check
	kEPre,       // emission: pipeline / mesh / restart checks
	kEBindVs,    // PrepareBindings, vertex stage
	kEBindPs,    // PrepareBindings, pixel stage, feedback check
	kEGfx,       // PrepareGraphicsBindings
	kEVertex,    // vertex and index buffers, write stages, late check
	kECommit,    // DrawScope, CommitBindings, dynamic state, BeginRendering, pipeline, draw, barrier
	kETail,      // bookkeeping after the draw
	kNLearn,     // normal path: NativeS0Learn (hashes, Begin, verify prediction)
	kNSegment,   // normal path: NativeS1SegmentDraw
	kNFinish,    // normal path: NativeS0Finish
	kStepCount
};
[[nodiscard]] bool Enabled();
// The slot the following Add / Draw calls are charged to (the native op's engine thread slot).
void SetSlot(uint32_t op_flags);
void Add(uint32_t step, uint64_t ticks);
// One draw of the slot: emitted natively, or taken by the normal path (after an attempt: attempted;
// continued: a DrawRun continuation), its DrawIndex ticks, and the ticks of a failed attempt.
void Draw(bool native, bool attempted, bool continued, uint64_t ticks, uint64_t attempt_ticks);
// Prints the window's report when 10 s passed (Thread_Gpu).
void Tick();
// Sub-step marker: each call charges the ticks since the previous mark (or construction) to a step.
struct Marker {
	bool     on;
	uint64_t t;
	Marker(): on(Enabled()), t(on ? Now() : 0) {}
	void operator()(uint32_t step) {
		if (on) [[unlikely]] {
			const uint64_t n = Now();
			Add(step, n - t);
			t = n;
		}
	}
	void Skip() {
		if (on) [[unlikely]] t = Now();
	}
	static uint64_t Now();
};
} // namespace Prof

class Session {
public:
	Session();
	~Session();
	Session(const Session&)            = delete;
	Session& operator=(const Session&) = delete;

	// Right after the pipeline selection. False: this is not the op's draw (an earlier pending
	// draw committed under the op), the caller keeps the op for a later draw. True: consumed.
	bool Begin(const DrawView& view);
	// The op ends: an op whose draw never reached Begin is reported (skipped / wrong draw).
	void EndOp(uint32_t op_flags);
	// Verify sample in progress: the commit's push constants and mesh draw dwords are captured.
	[[nodiscard]] bool Capturing() const { return m_capturing; }
	void               CapturePush(const uint32_t* dwords, bool valid);
	void               CaptureMesh(const uint32_t* dwords, uint32_t count);
	// After the draw command: comparisons of the captures, report.
	void Finish();
	// Whether the last Begin took the op's own draw (Finish not yet called), and the formats hash
	// it used (S1 segment state).
	[[nodiscard]] bool     Active() const { return m_active; }
	[[nodiscard]] uint64_t ActiveFormats() const { return m_formats; }
	[[nodiscard]] uint32_t ActiveOp() const { return m_op; }
	[[nodiscard]] bool     S1Armed() const { return m_s1_armed; }

	// S1: rebuilds the op's vertex stage (see the header). Returns Loader::GameCapture::NativeS1Reason
	// (kS1Emitted: Prediction() is valid until the next call).
	uint32_t PredictS1(const S1Inputs& in);
	[[nodiscard]] const S1Prediction& Prediction() const { return m_prediction; }
	// S1c: the emission sets the rebuilt vertex info's stage (ResolveDrawOffsets).
	[[nodiscard]] S1Prediction& MutablePrediction() { return m_prediction; }
	// S1b: the last kS1Shape's first uncovered property and stage.
	[[nodiscard]] uint32_t S1ShapeCode() const { return m_s1_shape; }
	[[nodiscard]] uint32_t S1ShapeStage() const { return m_s1_shape_stage; }
	// S1d: the census of the last PredictS1 (census: a learned base entry was found; mask: every
	// uncovered property met, 0 when covered) and the entry's write kinds.
	[[nodiscard]] bool     S1Census() const { return m_s1_census; }
	[[nodiscard]] uint64_t S1Mask() const { return m_s1_mask; }
	[[nodiscard]] uint32_t S1WriteKinds() const { return m_s1_write_kinds; }
	// S1 verify: the prediction is compared by the op's own draw on the normal path (Begin).
	void ArmS1Check() {
		m_s1_armed    = true;
		m_s1_armed_op = m_s1_op;
	}
	// S1: the op's draw was emitted natively (it never reaches Begin): reported as consumed.
	void NoteS1Emitted(uint32_t op_flags);
	// S1 verify sampling (1 in 64).
	[[nodiscard]] bool S1Sample() { return (++m_s1_samples & 63u) == 0; }

private:
	struct Table;
	void Report();
	void Mismatch(const char* what, uint32_t index, uint32_t dword, uint64_t expected, uint64_t got);
	void VerifyStage(uint32_t stage, const StageRecipe& recipe, const StageView& view, const uint32_t* ud,
	                 uint32_t ud_count, uint32_t push_start);
	void Conflict(const Entry& entry, const DrawView& view);
	void CheckS1(const DrawView& view, uint64_t key, const Entry* entry);

	std::unique_ptr<Table>             m_table;
	Loader::GameCapture::NativeDraw    m_record {};
	Loader::GameCapture::NativeS0Draw  m_result {};
	uint32_t                           m_op         = 0;
	bool                               m_active     = false; // between Begin and Finish
	bool                               m_capturing  = false;
	bool                               m_wrong_seen = false; // the op committed another draw first
	bool                               m_consumed   = false; // Begin took the op's draw
	uint32_t                           m_samples    = 0;
	uint64_t                           m_formats    = 0;
	// S1.
	S1Prediction m_prediction;
	uint32_t     m_s1_op       = 0;
	bool         m_s1_armed    = false;
	uint32_t     m_s1_armed_op = 0;
	uint32_t     m_s1_samples  = 0;
	uint32_t     m_s1_shape       = 0;
	uint32_t     m_s1_shape_stage = 0;
	bool         m_s1_census      = false;
	uint64_t     m_s1_mask        = 0;
	uint32_t     m_s1_write_kinds = 0;
	// S1d: eligibility cache (see the header): tag = Mix(regs_hash, formats_hash).
	struct S1Ineligible {
		uint64_t tag   = 0;
		uint64_t mask  = 0;
		uint32_t shape = 0, stage = 0, write_kinds = 0, hits = 0;
	};
	static constexpr uint32_t kS1CacheSize = 2048; // power of two
	std::unique_ptr<std::array<S1Ineligible, kS1CacheSize>> m_s1_cache;
	std::vector<uint8_t> m_s1_active; // S1c: active descriptor sources of the stage being predicted
	// S1e: the stage's load group bytes and states (0 unread, 1 read, 2 refused: per-dword reads).
	std::vector<uint8_t> m_s1_group_data;
	std::vector<uint8_t> m_s1_group_state;
	std::vector<uint32_t> m_s1_mip_scratch; // S1f: WriteMipStatsFields' shader data (plain choice)
	// S1e ("vs:variant"): per base key, the entry the last successful prediction found; the next
	// prediction of the base rebuilds with its stages first (one pass when the permutation repeats).
	struct S1Hint {
		uint64_t     base  = 0;
		const Entry* entry = nullptr;
	};
	static constexpr uint32_t kS1HintSize = 1024; // power of two
	std::unique_ptr<std::array<S1Hint, kS1HintSize>> m_s1_hints;
	// S1e: the stage rebuild of PredictS1 with a model entry's stages (key out; reason on failure).
	// S1f: `plain` in: the plain pixel variant is wanted and available (KYTY_LOD_STATS_PLAIN_VARIANT
	// on); out: the normal path's choice, i.e. also no mip-statistics counter on the rebuilt pixel
	// T#s (WriteMipStatsFields on the rebuilt snapshot, as RebindBuffers computes it).
	uint32_t BuildS1(const Entry& model, const S1Inputs& in, uint64_t base, bool& plain, uint64_t& key);
	// Verify sample: what the native path would push (expected) and what the commit pushed.
	std::array<uint32_t, 32> m_push_expected {};
	uint32_t                 m_push_mask = 0; // dwords of m_push_expected to compare
	std::array<uint32_t, 32> m_push_got {};
	bool                     m_push_valid = false;
	std::array<uint32_t, 8>  m_mesh_expected {};
	uint32_t                 m_mesh_mask = 0;
	std::array<uint32_t, 8>  m_mesh_got {};
	bool                     m_mesh_valid = false;
};

} // namespace Libs::Graphics::NativeLearn

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_NATIVELEARN_H_
