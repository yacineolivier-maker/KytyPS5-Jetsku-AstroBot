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
	bool                                                plain_pixel = false;
	mutable bool                                        conflicted  = false;
	// The newest pipeline object the normal path selected for the key (a fast-first build is
	// replaced by its optimized one; old objects stay valid). Thread_Gpu only.
	mutable const PipelineCache::Pipeline* current            = nullptr;
	mutable uint64_t                       current_generation = 0; // PipelineGeneration before that selection
	// S1: the stage-0 inputs a native rebuild needs, and whether the entry is covered.
	const ShaderRecompiler::IR::ResourcePlan* plan0 = nullptr;
	uint32_t ud_count       = 0;     // snapshot.user_data size (user SGPR base + count)
	uint32_t ud_base        = 0;     // the plan's user SGPR base
	uint32_t prefix_mode    = 0;     // 0: prefix dwords 0; 1: dwords 0..1 the GS user-data address; 2: other
	uint32_t flat_size      = 0;     // flattened SRT dwords
	bool     viewport_index = false; // the vertex program writes the viewport index
	bool     s1_covered     = false;
	uint32_t s1_shape       = 0;     // first uncovered property (diagnostic)
};

// S1: a native op's rebuilt vertex stage (PredictS1), owned by the session.
struct S1Prediction {
	const Entry*                                 entry = nullptr;
	uint64_t                                     key   = 0;
	ShaderRecompiler::IR::ResourceSnapshot       snapshot;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
};
// S1: what PredictS1 reads besides the record.
struct S1Inputs {
	uint32_t        op_flags          = 0;
	uint64_t        regs_hash         = 0; // NativeRegsHash of the bound registers
	uint64_t        formats_hash      = 0; // the segment's resolved target formats (its last normal draw)
	uint32_t        index_count       = 0;
	uint64_t        es_addr           = 0; // bound SPI_SHADER_PGM_LO/HI_ES
	const uint32_t* user_sgpr         = nullptr; // bound GS user SGPRs
	uint32_t        user_sgpr_count   = 0;
	uint64_t        gs_user_data_addr = 0;
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
};

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
