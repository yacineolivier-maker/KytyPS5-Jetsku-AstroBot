// Astro Bot PC (KYTY_NATIVE_MODE=s0): native pipeline learning and draw rebuild verification
// (nativeLearn.h).

#include "graphics/host_gpu/renderer/nativeLearn.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "kernel/memory.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fmt/format.h>
#include <string>
#include <xxhash.h>

namespace Libs::Graphics::NativeLearn {

namespace IR = ShaderRecompiler::IR;
namespace GC = Loader::GameCapture;

namespace {

constexpr uint64_t kAddressMask = 0x0000ffffffffffffull; // SrtWalker AddressMask

uint64_t Mix(uint64_t h, uint64_t v) {
	h ^= v + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
	return h * 0x100000001b3ull;
}

uint64_t SpecHash(const IR::ResourceSpecialization& s) {
	uint64_t h = 0xcbf29ce484222325ull;
	h          = Mix(h, s.buffers.size());
	for (const auto& b: s.buffers) {
		h = Mix(h, b.packed_stride);
		h = Mix(h, static_cast<uint64_t>(b.descriptor_format));
		h = Mix(h, b.descriptor_swizzle);
		h = Mix(h, b.zero_stride_oob ? 1u : 0u);
	}
	h = Mix(h, s.images.size());
	for (const auto& i: s.images) {
		h = Mix(h, static_cast<uint64_t>(i.numeric_class));
		h = Mix(h, static_cast<uint64_t>(i.dimension));
		h = Mix(h, i.mip_count);
		h = Mix(h, static_cast<uint64_t>(i.conversion_format));
		h = Mix(h, i.shader_swizzle);
		h = Mix(h, i.indirect_root);
		h = Mix(h, i.indirect_mapping_offset);
		h = Mix(h, i.indirect_search_iterations);
		h = Mix(h, (i.cube ? 1u : 0u) | (i.fmask ? 2u : 0u));
	}
	return h;
}

// Static classification of one IR value (see nativeLearn.h, "Recipe").
struct Classifier {
	const IR::ResourcePlan& plan;
	uint32_t                reason      = GC::kS0Simple;
	uint32_t                deep_opcode = 0;

	bool Fail(uint32_t why, const IR::Inst* inst) {
		if (reason == GC::kS0Simple || reason == GC::kS0Flow) {
			reason      = why;
			deep_opcode = inst != nullptr ? static_cast<uint32_t>(inst->GetOpcode()) + 1u : 0u;
		}
		return false;
	}

	bool Classify(IR::Value value, DwordRecipe& out, uint32_t depth = 0) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			out.kind = DwordRecipe::Kind::Const;
			switch (value.GetType()) {
				case IR::Type::U1: out.value = value.U1() ? 1u : 0u; return true;
				case IR::Type::U8: out.value = value.U8(); return true;
				case IR::Type::U16: out.value = value.U16(); return true;
				case IR::Type::U32: out.value = value.U32(); return true;
				case IR::Type::U64: out.value = static_cast<uint32_t>(value.U64()); return true;
				case IR::Type::F32: out.value = std::bit_cast<uint32_t>(value.F32Value()); return true;
				default: return Fail(GC::kS0Deep, nullptr);
			}
		}
		const IR::Inst* inst = value.TryInstruction();
		if (inst == nullptr || depth > 4) {
			return Fail(GC::kS0Deep, inst);
		}
		switch (inst->GetOpcode()) {
			case IR::ValueOpcode::GetUserData: {
				const auto arg = inst->Arg(0);
				if (!arg.IsImmediate() || arg.GetType() != IR::Type::ScalarReg) return Fail(GC::kS0Deep, inst);
				const auto reg = IR::RegIndex(arg.ScalarRegister());
				if (reg < plan.user_data_base || reg - plan.user_data_base >= 32u) return Fail(GC::kS0Deep, inst);
				out.kind  = DwordRecipe::Kind::User;
				out.value = reg - plan.user_data_base;
				return true;
			}
			case IR::ValueOpcode::BitCastU32F32:
			case IR::ValueOpcode::BitCastF32U32: return Classify(inst->Arg(0), out, depth + 1);
			case IR::ValueOpcode::ReadConst: {
				const auto slot = inst->Arg(1).Resolve();
				if (!slot.IsImmediate() || slot.GetType() != IR::Type::U32 || slot.U32() >= plan.srt_reads.size()) {
					return Fail(GC::kS0Deep, inst);
				}
				return Classify(plan.srt_reads[slot.U32()].value, out, depth + 1);
			}
			case IR::ValueOpcode::LoadAddressU32: {
				const auto flags = inst->Flags<IR::MemoryFlags>();
				if (flags.index >= plan.memory_info.size() || plan.memory_info[flags.index].kind != IR::ResourceKind::ScalarAddress) {
					return Fail(GC::kS0Deep, inst);
				}
				const auto* handle = inst->Arg(0).ResolveInstruction();
				if (handle == nullptr || handle->NumArgs() < 2) return Fail(GC::kS0Deep, inst);
				DwordRecipe lo;
				DwordRecipe hi;
				// The pointer halves must be user data themselves (one level only).
				const auto user = [&](IR::Value v, DwordRecipe& r) {
					v = v.Resolve();
					const auto* i = v.TryInstruction();
					while (i != nullptr && (i->GetOpcode() == IR::ValueOpcode::BitCastU32F32 ||
					                        i->GetOpcode() == IR::ValueOpcode::BitCastF32U32)) {
						v = i->Arg(0).Resolve();
						i = v.TryInstruction();
					}
					if (i == nullptr || i->GetOpcode() != IR::ValueOpcode::GetUserData) {
						return Fail(GC::kS0Deep, i != nullptr ? i : inst);
					}
					return Classify(v, r, depth + 1) && r.kind == DwordRecipe::Kind::User;
				};
				if (!user(handle->Arg(0), lo) || !user(handle->Arg(1), hi)) return false;
				const auto ofs = inst->Arg(1).Resolve();
				if (!ofs.IsImmediate() || ofs.GetType() != IR::Type::U32) return Fail(GC::kS0Deep, inst);
				const auto immediate = static_cast<int64_t>(static_cast<int32_t>(plan.memory_info[flags.index].offset));
				out.kind   = DwordRecipe::Kind::Load;
				out.lo     = static_cast<uint8_t>(lo.value);
				out.hi     = static_cast<uint8_t>(hi.value);
				out.offset = (immediate & ~int64_t {3}) + static_cast<int64_t>(ofs.U32() & ~3u);
				return true;
			}
			case IR::ValueOpcode::ReadConstBuffer: return Fail(GC::kS0ConstBuffer, inst);
			default: return Fail(GC::kS0Deep, inst);
		}
	}

	void Resource(uint32_t source, std::vector<ResourceRecipe>& out) {
		ResourceRecipe recipe;
		if (source >= plan.descriptor_sources.size()) {
			Fail(GC::kS0Deep, nullptr);
			out.push_back(recipe);
			return;
		}
		const auto& d = plan.descriptor_sources[source];
		if (d.indirect_descriptor.has_value()) {
			if (reason == GC::kS0Simple || reason == GC::kS0Flow) reason = GC::kS0Indirect;
			out.push_back(recipe);
			return;
		}
		recipe.dword_count = std::min<uint32_t>(d.dword_count, 8u);
		for (uint32_t k = 0; k < recipe.dword_count; k++) {
			(void)Classify(d.dwords[k], recipe.dwords[k]);
		}
		out.push_back(recipe);
	}
};

StageRecipe BuildRecipe(const IR::ResourcePlan& plan, const IR::CompiledShaderInfo& program) {
	StageRecipe recipe;
	Classifier  c {plan};
	for (const auto& b: program.info.buffers) c.Resource(b.source, recipe.buffers);
	for (const auto& i: program.info.images) c.Resource(i.source, recipe.images);
	for (const auto& s: program.info.samplers) c.Resource(s.source, recipe.samplers);
	for (const auto& read: plan.srt_reads) {
		StageRecipe::Flat flat;
		flat.flat_offset = read.flat_offset;
		(void)c.Classify(read.value, flat.recipe);
		recipe.flat.push_back(flat);
	}
	if (c.reason == GC::kS0Simple && !plan.control_flow.empty()) {
		c.reason = GC::kS0Flow;
	}
	recipe.reason      = c.reason;
	recipe.deep_opcode = c.deep_opcode;
	return recipe;
}

// The reason of a draw from its stages' (the first not-simple one; flow below every other).
void CombineReason(uint32_t& reason, uint32_t& opcode, const StageRecipe& recipe) {
	if (recipe.reason == GC::kS0Simple) return;
	if (reason == GC::kS0Simple || (reason == GC::kS0Flow && recipe.reason != GC::kS0Flow)) {
		reason = recipe.reason;
		opcode = recipe.deep_opcode;
	}
}

// One dword the native path would produce from the record's user data. False: not available
// (the guest bytes are not clean for a backing read).
bool Evaluate(const DwordRecipe& r, const uint32_t* ud, uint32_t& out) {
	switch (r.kind) {
		case DwordRecipe::Kind::Const: out = r.value; return true;
		case DwordRecipe::Kind::User: out = r.value < 32u ? ud[r.value] : 0u; return true;
		case DwordRecipe::Kind::Load: {
			const uint64_t base    = (((static_cast<uint64_t>(ud[r.hi & 31u]) << 32u) | ud[r.lo & 31u]) & kAddressMask) & ~uint64_t {3};
			const uint64_t address = base + static_cast<uint64_t>(r.offset);
			if (IR::NeverMappedAddress(address, sizeof(uint32_t))) {
				out = 0; // SrtWalker::InPlaceReadable: the GPU reads 0 there
				return true;
			}
			uint32_t word = 0;
			if (!LibKernel::Memory::TryReadGpuCleanBacking(address, &word, sizeof(word))) {
				return false;
			}
			out = word;
			return true;
		}
	}
	return false;
}

// S1: one dword from the bound user data (`ud`, `count` dwords after the plan's user SGPR base).
// False: an index outside the bound user data, or guest bytes not clean for a backing read.
bool EvaluateBound(const DwordRecipe& r, const uint32_t* ud, uint32_t count, uint32_t& out) {
	switch (r.kind) {
		case DwordRecipe::Kind::Const: out = r.value; return true;
		case DwordRecipe::Kind::User:
			if (r.value >= count) return false;
			out = ud[r.value];
			return true;
		case DwordRecipe::Kind::Load: {
			if (r.hi >= count || r.lo >= count) return false;
			const uint64_t base    = (((static_cast<uint64_t>(ud[r.hi]) << 32u) | ud[r.lo]) & kAddressMask) & ~uint64_t {3};
			const uint64_t address = base + static_cast<uint64_t>(r.offset);
			if (IR::NeverMappedAddress(address, sizeof(uint32_t))) {
				out = 0;
				return true;
			}
			return LibKernel::Memory::TryReadGpuCleanBacking(address, &out, sizeof(out));
		}
	}
	return false;
}

bool SameDword(const DwordRecipe& a, const DwordRecipe& b) {
	return a.kind == b.kind && a.lo == b.lo && a.hi == b.hi && a.value == b.value && a.offset == b.offset;
}

bool SameResources(const std::vector<ResourceRecipe>& a, const std::vector<ResourceRecipe>& b) {
	if (a.size() != b.size()) return false;
	for (size_t i = 0; i < a.size(); i++) {
		if (a[i].dword_count != b[i].dword_count) return false;
		for (uint32_t k = 0; k < a[i].dword_count; k++) {
			if (!SameDword(a[i].dwords[k], b[i].dwords[k])) return false;
		}
	}
	return true;
}

bool SameRecipe(const StageRecipe& a, const StageRecipe& b) {
	if (!SameResources(a.buffers, b.buffers) || !SameResources(a.images, b.images) || !SameResources(a.samplers, b.samplers) ||
	    a.flat.size() != b.flat.size()) {
		return false;
	}
	for (size_t i = 0; i < a.flat.size(); i++) {
		if (a.flat[i].flat_offset != b.flat[i].flat_offset || !SameDword(a.flat[i].recipe, b.flat[i].recipe)) return false;
	}
	return true;
}

// The NativeKey without the specialization signature, from the record and the draw's register
// signature and resolved target formats.
uint64_t BaseKey(const GC::NativeDraw& record, uint64_t formats_hash, uint64_t regs_hash) {
	const bool     es_known = (record.known & GC::kNkPgmEs) != 0;
	const uint64_t es_id    = es_known ? record.pgm_es : record.code_geom;
	const uint64_t ps_id    = (record.known & GC::kNkPgmPs) != 0 ? record.pgm_ps : record.code_ps;
	uint64_t       base     = 0xcbf29ce484222325ull;
	base                    = Mix(base, es_id);
	base                    = Mix(base, ps_id);
	base                    = Mix(base, record.state_hash);
	base                    = Mix(base, formats_hash);
	base                    = Mix(base, record.prim);
	base                    = Mix(base, regs_hash); // S0b
	return base;
}

// S1 coverage of one stage of a learned entry (0: the vertex/mesh stage, 1: the pixel stage);
// GC::kShCovered, else the first uncovered property (GC::NativeS1Shape).
uint32_t S1StageShape(const Entry& e, uint32_t s, const DrawView& view) {
	using Kind          = IR::DescriptorBindingKind;
	const auto& program = s == 0 ? *e.vs : *e.ps;
	const auto& st      = e.s1[s];
	const auto& recipe  = e.recipe[s];
	if (s == 0 && view.vertex_stage_count != 1) return GC::kShTess;
	// S1c: a simple recipe with control flow is covered (active-source walk at prediction).
	if (recipe.reason != GC::kS0Simple && recipe.reason != GC::kS0Flow) return GC::kShDeep; // deep, indirect descriptor, const buffer
	if (program.has_address_writes ||
	    std::any_of(program.info.buffers.begin(), program.info.buffers.end(), [](const auto& b) { return b.written || b.atomic; }) ||
	    std::any_of(program.info.images.begin(), program.info.images.end(), [](const auto& i) { return i.written || i.atomic; })) {
		return GC::kShWrites;
	}
	if (std::any_of(program.info.images.begin(), program.info.images.end(),
	                [](const auto& i) { return i.resource_class != IR::ImageResourceClass::Sampled || !i.indirect_resources.empty() ||
	                                       i.indirect_root != IR::ImageResource::NoIndirectImage;
	                })) {
		return GC::kShImageClass;
	}
	for (const auto& binding: program.bindings.descriptors) {
		switch (binding.kind) {
			case Kind::Buffers:
			case Kind::UniformBuffers:
			case Kind::BdaPagetable:
			case Kind::FaultBuffer:
			case Kind::FlattenedSrt:
			case Kind::Samplers:
			case Kind::ShaderData: break;
			case Kind::MipStats:
				if (s == 1) break; // the pixel stage's mip statistics (bound by the emulator's functions)
				return GC::kShBinding;
			default:
				if (IR::ImageBindingResourceClass(binding.kind) == IR::ImageResourceClass::Sampled) break;
				return GC::kShBinding;
		}
	}
	if (st.plan == nullptr) return GC::kShFlow;
	// S1c: the flow walk evaluates block conditions only (not the uniform fill's values).
	if (st.plan->uniform_fill.fill.words != 0) return GC::kShUniform;
	// S1c: embedded fetch is covered on the vertex (non-mesh) path with a learned template.
	if (s == 0 && (view.fetch_embedded || view.vertex_buffers != 0) &&
	    (e.mesh_active || e.fetch == nullptr || !view.fetch_embedded)) {
		return GC::kShFetch;
	}
	if (s == 0 && view.user_clip_planes != 0) return GC::kShClip;
	if (st.prefix_mode > 1 || st.ud_count < st.ud_base) return GC::kShPrefix;
	for (const auto* list: {&recipe.buffers, &recipe.images, &recipe.samplers}) {
		for (const auto& r: *list) {
			if (r.dword_count == 0) return GC::kShIndirect;
		}
	}
	if (view.stages[s].snapshot->flattened_srt.size() != st.flat_size) return GC::kShFlat;
	return GC::kShCovered;
}

// S1 coverage of a learned entry: every stage covered (shape/stage: the first uncovered one).
void S1Shape(Entry& e, const DrawView& view) {
	e.s1_shape       = GC::kShCovered;
	e.s1_shape_stage = 0;
	for (uint32_t s = 0; s < e.stage_count && e.s1_shape == GC::kShCovered; s++) {
		e.s1_shape       = S1StageShape(e, s, view);
		e.s1_shape_stage = s;
	}
	e.s1_covered = e.s1_shape == GC::kShCovered;
}

bool SameFetch(const Entry& a, const Entry& b) {
	if ((a.fetch == nullptr) != (b.fetch == nullptr)) return false;
	if (a.fetch == nullptr) return true;
	const auto& x = *a.fetch;
	const auto& y = *b.fetch;
	if (x.attrib_reg != y.attrib_reg || x.buffer_reg != y.buffer_reg || x.count != y.count) return false;
	for (uint32_t i = 0; i < x.count; i++) {
		if (x.dst[i].register_start != y.dst[i].register_start || x.dst[i].registers_num != y.dst[i].registers_num ||
		    x.dst[i].attr_id != y.dst[i].attr_id) {
			return false;
		}
	}
	return true;
}

bool SameS1Stages(const Entry& a, const Entry& b) {
	if (a.stage_count != b.stage_count || !SameFetch(a, b)) return false;
	for (uint32_t s = 0; s < a.stage_count; s++) {
		const auto& x = a.s1[s];
		const auto& y = b.s1[s];
		if (x.plan != y.plan || x.ud_count != y.ud_count || x.ud_base != y.ud_base || x.prefix_mode != y.prefix_mode ||
		    x.flat_size != y.flat_size || !SameRecipe(a.recipe[s], b.recipe[s])) {
			return false;
		}
	}
	return true;
}

// S1c: the clean reader of the active-source walk: ReadShaderGuestMemory's answers (clean backing,
// or 0 for an address the GPU cannot read) without its retry bookkeeping; a refused read is noted
// (the prediction then falls back: the walk follows both edges of an unevaluated condition, where
// the normal path's read may succeed).
struct FlowReads {
	bool failed = false;
};
bool FlowRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (values.empty()) return false;
	if (LibKernel::Memory::TryReadGpuCleanBacking(address, values.data(), values.size_bytes())) return true;
	const auto size = values.size_bytes();
	if (IR::NeverMappedAddress(address, size) ||
	    (ShaderRecompiler::GetCodegenOptions().srt_variant_reads && !LibKernel::Memory::IsGpuMapped(address, size))) {
		std::fill(values.begin(), values.end(), 0u); // ReadsUnmappedGuestMemory
		return true;
	}
	static_cast<FlowReads*>(userdata)->failed = true;
	return false;
}

// S1c: the plan's active descriptor sources for the user data (MaterializeResources' clean walk).
// False: a condition read refused. `active` empty: every source is active.
bool ActiveSources(const IR::ResourcePlan& plan, std::span<const uint32_t> user_data, uint64_t shader_base,
                   std::vector<uint8_t>& active) {
	active.clear();
	if (plan.control_flow.empty()) return true;
	FlowReads       reads;
	IR::SrtRuntime  runtime {
	     .user_data                  = user_data,
	     .shader_base                = shader_base,
	     .userdata                   = &reads,
	     .read_specialization_memory = FlowRead,
	     .is_guest_mapped = ShaderRecompiler::GetCodegenOptions().srt_variant_reads ? LibKernel::Memory::IsGpuMapped : nullptr,
    };
	IR::SrtWalker clean(plan, IR::CleanRuntime(runtime));
	const auto    found = clean.FindActiveSources();
	if (reads.failed) return false;
	active.assign(found.begin(), found.end());
	return true;
}

} // namespace

// S1b/S1c (nativeLearn.h): one XXH3 over the packed words (was one call per buffer and attribute
// list): the embedded fetch inputs of the vertex program key (per resource the destination
// registers, attribute and V# dword 3 & 0x7ffff) and of the pipeline's vertex input state
// (strides, instance flags, attribute offsets).
uint64_t FetchHash(const ShaderVertexInputInfo& info) {
	if (!info.fetch_embedded && info.resources_num == 0 && info.buffers_num == 0) return 0;
	constexpr int Max = ShaderVertexInputInfo::RES_MAX;
	std::array<uint32_t, 3 + Max * 3 + Max * (3 + 2 * ShaderVertexInputBuffer::ATTR_MAX)> words; // filled up to n
	uint32_t n = 0;
	words[n++] = info.fetch_embedded ? 1u : 0u;
	words[n++] = static_cast<uint32_t>(info.resources_num);
	words[n++] = static_cast<uint32_t>(info.buffers_num);
	for (int i = 0; i < info.resources_num && i < Max; i++) {
		words[n++] = static_cast<uint32_t>(info.resources_dst[i].registers_num);
		words[n++] = static_cast<uint32_t>(info.resources_dst[i].attr_id);
		words[n++] = info.resources[i].fields[3] & 0x7ffffu;
	}
	for (int i = 0; i < info.buffers_num && i < Max; i++) {
		const auto& b = info.buffers[i];
		words[n++]    = b.stride;
		words[n++]    = b.fetch_index;
		words[n++]    = static_cast<uint32_t>(b.attr_num);
		const auto attrs = std::clamp(b.attr_num, 0, ShaderVertexInputBuffer::ATTR_MAX);
		for (int a = 0; a < attrs; a++) words[n++] = static_cast<uint32_t>(b.attr_indices[a]);
		for (int a = 0; a < attrs; a++) words[n++] = b.attr_offsets[a];
	}
	const uint64_t h = XXH3_64bits(words.data(), n * sizeof(uint32_t));
	return h != 0 ? h : 1;
}

// Append-only open-addressing table (one writer, lock-free readers).
struct Session::Table {
	static constexpr uint32_t Slots      = 16384; // power of two
	static constexpr uint32_t MaxEntries = 8192;
	std::array<std::atomic<Entry*>, Slots> slots {};
	std::array<uint64_t, Slots>            base_keys {}; // writer only
	std::array<const Entry*, Slots>        base_entries {}; // S1: the first entry of each base key (Thread_Gpu)
	std::vector<std::unique_ptr<Entry>>    owned;
	uint32_t                               base_count = 0;

	// S1 (Thread_Gpu, the writer's thread): the first entry learned for a base key.
	[[nodiscard]] const Entry* FindBase(uint64_t base) const {
		base = base == 0 ? 1 : base;
		for (uint32_t p = 0, b = static_cast<uint32_t>(base >> 20u); p < 64; p++) {
			const auto k = base_keys[(b + p) & (Slots - 1u)];
			if (k == 0) return nullptr;
			if (k == base) return base_entries[(b + p) & (Slots - 1u)];
		}
		return nullptr;
	}
	[[nodiscard]] const Entry* Find(uint64_t key) const {
		for (uint32_t probe = 0, h = static_cast<uint32_t>(key >> 20u); probe < 64; probe++) {
			const Entry* e = slots[(h + probe) & (Slots - 1u)].load(std::memory_order_acquire);
			if (e == nullptr) return nullptr;
			if (e->key == key) return e;
		}
		return nullptr;
	}
	// Null when full.
	const Entry* Insert(std::unique_ptr<Entry> entry) {
		if (owned.size() >= MaxEntries) return nullptr;
		const uint64_t key = entry->key;
		for (uint32_t probe = 0, h = static_cast<uint32_t>(key >> 20u); probe < 64; probe++) {
			auto& slot = slots[(h + probe) & (Slots - 1u)];
			if (slot.load(std::memory_order_relaxed) == nullptr) {
				// Distinct base keys (writer only, same probing scheme over its own array).
				const uint64_t base = entry->base == 0 ? 1 : entry->base;
				for (uint32_t p = 0, b = static_cast<uint32_t>(base >> 20u); p < 64; p++) {
					auto& k = base_keys[(b + p) & (Slots - 1u)];
					if (k == base) break;
					if (k == 0) {
						k                                    = base;
						base_entries[(b + p) & (Slots - 1u)] = entry.get();
						base_count++;
						break;
					}
				}
				const Entry* published = entry.get();
				owned.push_back(std::move(entry));
				slot.store(const_cast<Entry*>(published), std::memory_order_release);
				return published;
			}
		}
		return nullptr;
	}
};

Session::Session(): m_table(std::make_unique<Table>()) {}
Session::~Session() = default;

void Session::Mismatch(const char* what, uint32_t index, uint32_t dword, uint64_t expected, uint64_t got) {
	m_result.mismatches++;
	if (m_result.mismatches <= 2) {
		const auto text = fmt::format("{} {} dword {}: record+recipe 0x{:x}, committed 0x{:x} (pgm es 0x{:x} ps 0x{:x})", what, index, dword,
		                              expected, got, m_record.pgm_es, m_record.pgm_ps);
		GC::NoteNativeS0Mismatch(m_op, text.c_str());
	}
}

void Session::VerifyStage(uint32_t stage, const StageRecipe& recipe, const StageView& view, const uint32_t* ud, uint32_t ud_count,
                          uint32_t push_start) {
	const auto& snapshot = *view.snapshot;
	const auto  check    = [&](const char* what, const std::vector<ResourceRecipe>& recipes, const std::vector<IR::DescriptorValue>& values) {
        for (uint32_t i = 0; i < recipes.size() && i < values.size(); i++) {
            const auto& r = recipes[i];
            const auto& v = values[i];
            if (r.dword_count == 0) continue; // indirect / unknown source
            std::array<uint32_t, 8> expected {};
            bool                    available = true;
            bool                    zero      = true;
            for (uint32_t k = 0; k < r.dword_count; k++) {
                available = available && Evaluate(r.dwords[k], ud, expected[k]);
                zero      = zero && v.dwords[k] == 0;
            }
            if (!available) continue;
            uint32_t differ = 0;
            for (uint32_t k = 0; k < r.dword_count && k < v.dword_count; k++) {
                differ += expected[k] != v.dwords[k] ? 1u : 0u;
            }
            if (differ != 0 && zero) {
                m_result.nulled++; // inactive source or nulled descriptor: the committed value is 0
                continue;
            }
            m_result.compared += r.dword_count;
            for (uint32_t k = 0; k < r.dword_count && k < v.dword_count; k++) {
                if (expected[k] != v.dwords[k]) Mismatch(what, i, k, expected[k], v.dwords[k]);
            }
        }
	};
	check(stage == 0 ? "vs V#" : "ps V#", recipe.buffers, snapshot.buffers);
	check(stage == 0 ? "vs T#" : "ps T#", recipe.images, snapshot.images);
	check(stage == 0 ? "vs S#" : "ps S#", recipe.samplers, snapshot.samplers);
	for (const auto& flat: recipe.flat) {
		uint32_t expected = 0;
		if (flat.flat_offset >= snapshot.flattened_srt.size() || !Evaluate(flat.recipe, ud, expected)) continue;
		m_result.compared++;
		if (expected != snapshot.flattened_srt[flat.flat_offset]) {
			Mismatch(stage == 0 ? "vs flat-srt" : "ps flat-srt", flat.flat_offset, 0, expected, snapshot.flattened_srt[flat.flat_offset]);
		}
	}
	// Shader data: the user dwords first (AppendUserShaderData), then host-derived dwords (buffer
	// offsets, mip-statistics fields), which a native emitter computes from its own bindings.
	const auto& program = *view.program;
	const auto& regs    = program.bindings.user_data_registers;
	for (uint32_t k = 0; k < regs.size(); k++) {
		const uint32_t index    = regs[k] - program.user_data_base;
		const uint32_t expected = index < ud_count ? ud[index] : 0u;
		if (view.shader_data != nullptr && k < view.shader_data->size()) {
			m_result.compared++;
			if ((*view.shader_data)[k] != expected) Mismatch(stage == 0 ? "vs shader-data" : "ps shader-data", k, 0, expected, (*view.shader_data)[k]);
		}
		if (program.bindings.UsesPushData() && push_start + k < m_push_expected.size()) {
			m_push_expected[push_start + k] = expected;
			m_push_mask |= 1u << (push_start + k);
		}
	}
}

bool Session::Begin(const DrawView& view) {
	m_active    = false;
	m_capturing = false;
	m_result    = {};
	m_op        = view.op_flags;
	if (!GC::LookupNativeRecordForOp(view.op_flags, m_record)) {
		m_result.reason = GC::kS0NoRecord;
		m_consumed      = true;
		Report();
		return true;
	}
	// The executed draw must be the record's: an op may first commit earlier pending draws.
	const bool es_known = (m_record.known & GC::kNkPgmEs) != 0;
	if (m_record.count != view.index_count || (es_known && m_record.pgm_es != view.cp_es_addr)) {
		m_wrong_seen = true;
		return false;
	}
	m_wrong_seen = false;
	m_consumed   = true;
	m_active     = true;
	if (view.vertex_stage_count != 1 || view.stage_count == 0) {
		m_result.reason = GC::kS0Tess;
		return true;
	}
	for (uint32_t s = 0; s < view.stage_count; s++) {
		if (view.stages[s].plan == nullptr || view.stages[s].program == nullptr || view.stages[s].snapshot == nullptr) {
			m_result.reason = GC::kS0NoPlan;
			return true;
		}
	}
	// NativeKey (S0b: with the register signature).
	m_formats            = view.formats_hash;
	const uint64_t base  = BaseKey(m_record, view.formats_hash, view.regs_hash);
	uint64_t spec        = Mix(0x84222325cbf29ce4ull, view.plain_pixel ? 1u : 0u);
	for (uint32_t s = 0; s < view.stage_count; s++) {
		spec = Mix(spec, view.stages[s].spec != nullptr ? SpecHash(*view.stages[s].spec) : 0u);
	}
	// S1b: the embedded fetch words of the vertex program key (from guest memory), outside the base.
	uint64_t key = Mix(Mix(base, view.fetch_hash), spec);
	key          = key == 0 ? 1 : key;
	const auto* pixel   = view.stage_count > 1 ? &view.stages[1] : nullptr;
	const Entry* entry  = m_table->Find(key);
	std::array<StageRecipe, 2> fresh;
	const std::array<StageRecipe, 2>* recipes = nullptr;
	if (entry != nullptr) {
		m_result.found = true;
		const bool same_programs = entry->vs == view.stages[0].program && entry->ps == (pixel != nullptr ? pixel->program : nullptr) &&
		                           entry->vs_id == view.stages[0].program_id &&
		                           entry->ps_id == (pixel != nullptr ? pixel->program_id : 0u) &&
		                           entry->layout == view.pipeline->pipeline_layout;
		if (!same_programs) {
			m_result.conflict = true;
			Conflict(*entry, view);
		} else {
			if (entry->vk_pipeline != view.pipeline->pipeline) {
				m_result.replaced = true; // e.g. a linked or fast-first pipeline swapped for its optimized build
			}
			entry->current            = view.pipeline; // S1 emits with the newest object the normal path chose
			entry->current_generation = view.pipeline_generation;
		}
		m_result.reason      = entry->reason;
		m_result.deep_opcode = entry->deep_opcode;
		recipes              = &entry->recipe;
	} else if (m_table->owned.size() >= Table::MaxEntries) {
		// S1b: a full table learns nothing: no recipe is built for nothing (was one per draw).
		m_result.reason = GC::kS0TableFull;
	} else {
		auto e              = std::make_unique<Entry>();
		e->key              = key;
		e->base             = base;
		e->pipeline         = view.pipeline;
		e->vk_pipeline      = view.pipeline->pipeline;
		e->layout           = view.pipeline->pipeline_layout;
		e->set_layout       = view.pipeline->descriptor_set_layout;
		e->push_descriptors = view.pipeline->uses_push_descriptors;
		e->vs               = view.stages[0].program;
		e->vs_id            = view.stages[0].program_id;
		e->ps               = pixel != nullptr ? pixel->program : nullptr;
		e->ps_id            = pixel != nullptr ? pixel->program_id : 0u;
		e->mesh_active      = view.mesh_active;
		e->current          = view.pipeline;
		e->current_generation = view.pipeline_generation;
		e->identity         = view.identity;
		e->plain_pixel      = view.plain_pixel;
		e->plain_available  = view.plain_available;
		if (view.mesh != nullptr) e->mesh = *view.mesh;
		e->stage_count = view.stage_count;
		uint32_t reason = GC::kS0Simple;
		uint32_t opcode = 0;
		for (uint32_t s = 0; s < view.stage_count; s++) {
			e->push_start[s] = view.stages[s].program->bindings.push_data_start_dword;
			e->recipe[s]     = BuildRecipe(*view.stages[s].plan, *view.stages[s].program);
			CombineReason(reason, opcode, e->recipe[s]);
		}
		e->reason      = reason;
		e->deep_opcode = opcode;
		// S1: each stage's inputs of a native rebuild, and the coverage.
		for (uint32_t s = 0; s < view.stage_count; s++) {
			const auto& snapshot = *view.stages[s].snapshot;
			auto&       st       = e->s1[s];
			st.plan              = view.stages[s].plan;
			st.ud_base           = st.plan->user_data_base;
			st.ud_count          = static_cast<uint32_t>(snapshot.user_data.size());
			st.flat_size         = static_cast<uint32_t>(snapshot.flattened_srt.size());
			bool zero = true;
			for (uint32_t i = 0; i < st.ud_base && i < st.ud_count; i++) zero = zero && snapshot.user_data[i] == 0;
			bool front = s == 0 && st.ud_base >= 2 && st.ud_count >= 2 &&
			             snapshot.user_data[0] == static_cast<uint32_t>(view.gs_user_data_addr) &&
			             snapshot.user_data[1] == static_cast<uint32_t>(view.gs_user_data_addr >> 32u);
			for (uint32_t i = 2; front && i < st.ud_base && i < st.ud_count; i++) front = snapshot.user_data[i] == 0;
			st.prefix_mode = zero ? 0u : front ? 1u : 2u;
		}
		// S1c: the embedded fetch's static part.
		if (view.fetch_embedded && view.vertex_info != nullptr && !view.mesh_active) {
			const auto& vi = *view.vertex_info;
			if (vi.resources_num >= 0 && vi.resources_num <= ShaderVertexInputInfo::RES_MAX) {
				auto tpl        = std::make_unique<FetchTemplate>();
				tpl->attrib_reg = vi.fetch_attrib_reg;
				tpl->buffer_reg = vi.fetch_buffer_reg;
				tpl->count      = static_cast<uint32_t>(vi.resources_num);
				for (uint32_t i = 0; i < tpl->count; i++) {
					tpl->dst[i]             = {};
					tpl->dst[i].register_start = vi.resources_dst[i].register_start;
					tpl->dst[i].registers_num  = vi.resources_dst[i].registers_num;
					tpl->dst[i].attr_id        = vi.resources_dst[i].attr_id;
				}
				e->fetch = std::move(tpl);
			}
		}
		{
			const auto& outputs = e->vs->info.outputs;
			e->viewport_index   = std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
                return output.kind == IR::StageOutputKind::ViewportIndex;
            });
			S1Shape(*e, view);
		}
		m_result.reason      = reason;
		m_result.deep_opcode = opcode;
		fresh                = e->recipe;
		if (m_table->Insert(std::move(e)) != nullptr) {
			m_result.learned = true;
			GC::NoteNativeS0Table(static_cast<uint32_t>(m_table->owned.size()), m_table->base_count);
		} else {
			m_result.reason = GC::kS0TableFull;
		}
		recipes = &fresh;
		if (m_result.learned) recipes = &m_table->Find(key)->recipe;
	}
	// S1 verify: a native prediction made for this op is compared with the normal path's draw.
	if (m_s1_armed) {
		CheckS1(view, key, m_table->Find(key));
	}
	// Verify sample: 1 eligible draw in 64.
	const bool eligible = m_result.reason == GC::kS0Simple || m_result.reason == GC::kS0Flow;
	if (!GC::NativeVerifyEnabled() || !eligible || (++m_samples & 63u) != 0) {
		return true;
	}
	m_result.verified = true;
	m_push_mask       = 0;
	m_mesh_mask       = 0;
	m_push_valid      = false;
	m_mesh_valid      = false;
	for (uint32_t s = 0; s < view.stage_count; s++) {
		const auto&    st       = view.stages[s];
		const uint32_t* shadow  = s == 0 ? m_record.user_gs : m_record.user_ps;
		const uint32_t  known   = s == 0 ? m_record.user_gs_known : m_record.user_ps_known;
		const uint32_t* copy    = s == 0 ? m_record.sgpr_geom : m_record.sgpr_ps;
		uint32_t        ud[32] {};
		const auto&     user    = st.snapshot->user_data;
		const uint32_t  ud_size = std::min<uint32_t>(static_cast<uint32_t>(user.size()), 32u);
		for (uint32_t i = 0; i < 32; i++) {
			ud[i] = (known & (1u << i)) != 0 ? shadow[i] : copy[i];
		}
		// User-data mapping: the user data the emulator's stage used vs the record's two sources.
		for (uint32_t i = 0; i < ud_size; i++) {
			m_result.ud_compared++;
			if ((known & (1u << i)) == 0) {
				m_result.ud_shadow_unk++;
			} else if (shadow[i] == user[i]) {
				m_result.ud_shadow_eq++;
			} else {
				m_result.ud_shadow_ne++;
			}
			(copy[i] == user[i] ? m_result.ud_copy_eq : m_result.ud_copy_ne)++;
		}
		VerifyStage(s, (*recipes)[s], st, ud, 32u, st.program->bindings.push_data_start_dword);
	}
	// Mesh draw dwords the native path would push from the record (first dispatch of the draw):
	// count, vertex offset, first instance, index size, index address lo/hi, first group.
	if (view.mesh_active) {
		const bool     indexed = m_record.kind == 0 || m_record.kind == 1;
		const uint32_t size    = indexed ? (m_record.ib_32 != 0 ? 4u : 2u) : 0u;
		const uint64_t address = indexed ? m_record.ib_addr + static_cast<uint64_t>(m_record.first) * size : 0u;
		m_mesh_expected        = {m_record.count, m_record.vertex_offset, 0u, size, static_cast<uint32_t>(address),
                                  static_cast<uint32_t>(address >> 32u), 0u, 0u};
		m_mesh_mask            = 0x1u | 0x2u | 0x40u;
		if (m_record.kind != 3) m_mesh_mask |= 0x4u;
		if (indexed) m_mesh_mask |= 0x8u | 0x10u | 0x20u;
	}
	m_capturing = true;
	return true;
}

void Session::CapturePush(const uint32_t* dwords, bool valid) {
	if (!m_capturing) return;
	m_push_valid = valid;
	if (valid) std::memcpy(m_push_got.data(), dwords, sizeof(m_push_got));
}

void Session::CaptureMesh(const uint32_t* dwords, uint32_t count) {
	if (!m_capturing || m_mesh_valid) return;
	m_mesh_valid = true;
	m_mesh_got   = {};
	std::memcpy(m_mesh_got.data(), dwords, std::min<uint32_t>(count, 8u) * sizeof(uint32_t));
}

void Session::Finish() {
	if (!m_active) return;
	if (m_capturing) {
		if (m_push_valid) {
			for (uint32_t k = 0; k < 32; k++) {
				if ((m_push_mask & (1u << k)) == 0) continue;
				m_result.compared++;
				if (m_push_expected[k] != m_push_got[k]) Mismatch("push-constant", k, 0, m_push_expected[k], m_push_got[k]);
			}
		}
		if (m_mesh_valid) {
			for (uint32_t k = 0; k < 8; k++) {
				if ((m_mesh_mask & (1u << k)) == 0) continue;
				m_result.compared++;
				if (m_mesh_expected[k] != m_mesh_got[k]) Mismatch("mesh-draw", k, 0, m_mesh_expected[k], m_mesh_got[k]);
			}
		}
	}
	Report();
}

void Session::EndOp(uint32_t op_flags) {
	if (m_active) {
		Finish(); // returned between Begin and the draw command
	} else if (op_flags != 0 && !m_consumed) {
		// The op's draw never reached the pipeline selection.
		m_result        = {};
		m_op            = op_flags;
		m_result.reason = m_wrong_seen ? GC::kS0WrongDraw : GC::kS0Skipped;
		Report();
	}
	m_wrong_seen = false;
	m_consumed   = false;
	m_op         = 0;
	m_s1_armed   = false;
}

// S0b: why two permutations a key maps to differ (the first differing component).
void Session::Conflict(const Entry& entry, const DrawView& view) {
	entry.conflicted = true;
	uint32_t cause   = GC::kCfOther;
	uint32_t word    = 0;
	uint64_t was = 0, now = 0;
	for (uint32_t s = 0; s < 2 && cause == GC::kCfOther; s++) {
		const auto& a = entry.identity[s];
		const auto& b = view.identity[s];
		if (a.permutation == b.permutation) continue;
		const uint32_t first = s == 0 ? GC::kCfVsHash : GC::kCfPsHash;
		if (a.hash != b.hash) {
			cause = first, was = a.hash, now = b.hash;
		} else if (a.user_data_count != b.user_data_count) {
			cause = first + 1u, was = a.user_data_count, now = b.user_data_count;
		} else if (a.code_size != b.code_size) {
			cause = first + 2u, was = a.code_size, now = b.code_size;
		} else if (a.static_state != b.static_state && (a.static_state == nullptr || b.static_state == nullptr ||
		                                                *a.static_state != *b.static_state)) {
			cause = first + 3u;
			if (a.static_state != nullptr && b.static_state != nullptr) {
				const auto& x = *a.static_state;
				const auto& y = *b.static_state;
				const auto  n = std::min(x.size(), y.size());
				word          = static_cast<uint32_t>(n);
				for (size_t k = 0; k < n; k++) {
					if (x[k] != y[k]) {
						word = static_cast<uint32_t>(k);
						break;
					}
				}
				was = word < x.size() ? x[word] : 0xffffffffu;
				now = word < y.size() ? y[word] : 0xffffffffu;
			}
		} else if (a.push_start != b.push_start) {
			cause = first + 4u, was = a.push_start, now = b.push_start;
		} else if (a.specialization != nullptr && b.specialization != nullptr && !(*a.specialization == *b.specialization)) {
			cause = first + 5u;
		}
	}
	if (cause == GC::kCfOther && entry.plain_pixel != view.plain_pixel) cause = GC::kCfPlain;
	if (cause == GC::kCfOther && entry.layout != view.pipeline->pipeline_layout && entry.identity[0].permutation == view.identity[0].permutation &&
	    entry.identity[1].permutation == view.identity[1].permutation) {
		cause = GC::kCfLayout;
	}
	m_result.conflict_cause = cause;
	m_result.conflict_word  = word;
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		static constexpr const char* names[GC::kCfCount] = {"none",    "layout",  "vs-hash", "vs-udc",  "vs-code", "vs-static",
		                                                    "vs-push", "vs-spec", "ps-hash", "ps-udc",  "ps-code", "ps-static",
		                                                    "ps-push", "ps-spec", "plain",   "other"};
		GC::NoteNativeS0Mismatch(view.op_flags,
		                         fmt::format("NativeKey conflict: key 0x{:x} learned 0x{:x}/0x{:x}, now 0x{:x}/0x{:x}: {} word {} 0x{:x} -> 0x{:x}",
		                                     entry.key, entry.vs_id, entry.ps_id, view.stages[0].program_id,
		                                     view.stage_count > 1 ? view.stages[1].program_id : 0u, names[cause], word, was, now)
		                             .c_str());
	}
}

uint32_t Session::PredictS1(const S1Inputs& in) {
	m_s1_op            = in.op_flags;
	m_s1_armed         = false;
	m_s1_shape         = GC::kShCovered;
	m_s1_shape_stage   = 0;
	m_prediction.entry = nullptr;
	const auto shape   = [&](uint32_t code, uint32_t stage) {
        m_s1_shape       = code;
        m_s1_shape_stage = stage;
        return GC::kS1Shape;
	};
	if (!GC::LookupNativeRecordForOp(in.op_flags, m_record)) {
		return GC::kS1Record;
	}
	if (m_record.count != in.index_count || ((m_record.known & GC::kNkPgmEs) != 0 && m_record.pgm_es != in.es_addr) ||
	    (in.pixel && (m_record.known & GC::kNkPgmPs) != 0 && m_record.pgm_ps != in.ps_addr)) {
		return GC::kS1Record;
	}
	const uint64_t base = BaseKey(m_record, in.formats_hash, in.regs_hash);
	const Entry*   rep  = m_table->FindBase(base);
	if (rep == nullptr) return GC::kS1Miss;
	if (!rep->s1_covered) return shape(rep->s1_shape, rep->s1_shape_stage);
	const uint32_t stages = in.pixel ? 2u : 1u;
	if (rep->stage_count != stages) return shape(GC::kShVariant, stages - 1u);
	// The plain pixel variant (KYTY_LOD_STATS_PLAIN_VARIANT): verify mode binds a canary the native
	// path does not handle; on, the variant is taken unless the bindings record mip statistics,
	// which the emission checks after resolving them (kS1Plain).
	if (in.pixel && in.lod_plain == 2u) return shape(GC::kShLodVerify, 1);
	const bool plain = in.pixel && in.lod_plain == 1u && rep->plain_available;
	uint64_t   spec  = Mix(0x84222325cbf29ce4ull, plain ? 1u : 0u);
	for (uint32_t s = 0; s < stages; s++) {
		const auto&     st     = rep->s1[s];
		const uint32_t* sgpr   = s == 0 ? in.user_sgpr : in.ps_user_sgpr;
		const uint32_t  sgpr_n = s == 0 ? in.user_sgpr_count : in.ps_user_sgpr_count;
		if (sgpr_n + st.ud_base != st.ud_count || sgpr_n > 32u) return shape(GC::kShUdCount, s);
		auto& snap = m_prediction.snapshot[s];
		snap.user_data.assign(st.ud_count, 0u);
		if (st.prefix_mode == 1u) {
			snap.user_data[0] = static_cast<uint32_t>(in.gs_user_data_addr);
			snap.user_data[1] = static_cast<uint32_t>(in.gs_user_data_addr >> 32u);
		}
		std::copy_n(sgpr, sgpr_n, snap.user_data.begin() + st.ud_base);
		const uint32_t* ud     = snap.user_data.data() + st.ud_base;
		const auto&     recipe = rep->recipe[s];
		const auto&     info   = (s == 0 ? rep->vs : rep->ps)->info;
		// S1c: the plan's active sources (control flow): an inactive source materializes as zeros.
		if (!ActiveSources(*st.plan, snap.user_data, s == 0 ? in.es_addr : in.ps_addr, m_s1_active)) return GC::kS1Read;
		const auto build = [&](const std::vector<ResourceRecipe>& recipes, const auto& resources, std::vector<IR::DescriptorValue>& values) {
			values.resize(recipes.size());
			for (size_t i = 0; i < recipes.size(); i++) {
				const auto& r = recipes[i];
				auto&       v = values[i];
				v             = {};
				v.dword_count = r.dword_count;
				if (!m_s1_active.empty() && i < resources.size()) {
					const auto source = resources[i].source;
					if (source >= m_s1_active.size()) return false;
					if (m_s1_active[source] == 0u) {
						v.dword_count = st.plan->descriptor_sources[source].dword_count;
						continue;
					}
				}
				for (uint32_t k = 0; k < r.dword_count; k++) {
					if (!EvaluateBound(r.dwords[k], ud, sgpr_n, v.dwords[k])) return false;
				}
			}
			return true;
		};
		if (!build(recipe.buffers, info.buffers, snap.buffers) || !build(recipe.images, info.images, snap.images) ||
		    !build(recipe.samplers, info.samplers, snap.samplers)) {
			return GC::kS1Read;
		}
		snap.uniform_fill = {};
		snap.flattened_srt.assign(st.flat_size, 0u);
		for (const auto& flat: recipe.flat) {
			if (flat.flat_offset >= st.flat_size) return shape(GC::kShFlat, s);
			if (!EvaluateBound(flat.recipe, ud, sgpr_n, snap.flattened_srt[flat.flat_offset])) return GC::kS1Read;
		}
		// The whole specialization as materialization derives it (null V#s and invalid T#s zeroed,
		// the image remap applied to the snapshot).
		if (!IR::SpecializeForNative(*st.plan, snap, m_prediction.specialization[s])) return GC::kS1Spec;
		spec = Mix(spec, SpecHash(m_prediction.specialization[s]));
	}
	// S1c: the embedded fetch, rebuilt from the bound SGPRs and the guest tables.
	m_prediction.fetch      = rep->fetch != nullptr;
	m_prediction.fetch_hash = 0;
	if (rep->fetch != nullptr) {
		const auto& tpl = *rep->fetch;
		auto&       vi  = m_prediction.vertex;
		vi.resources_num = static_cast<int>(tpl.count);
		for (uint32_t i = 0; i < tpl.count; i++) {
			vi.resources_dst[i].register_start = tpl.dst[i].register_start;
			vi.resources_dst[i].registers_num  = tpl.dst[i].registers_num;
			vi.resources_dst[i].attr_id        = tpl.dst[i].attr_id;
		}
		vi.fetch_attrib_reg = tpl.attrib_reg;
		vi.fetch_buffer_reg = tpl.buffer_reg;
		if (!ShaderRebuildEmbeddedFetch(tpl.attrib_reg, tpl.buffer_reg, in.user_sgpr, in.user_sgpr_max, vi)) return GC::kS1Read;
		m_prediction.fetch_hash = FetchHash(vi);
	}
	uint64_t key = Mix(Mix(base, m_prediction.fetch_hash), spec);
	key          = key == 0 ? 1 : key;
	const Entry* entry = m_table->Find(key);
	if (entry == nullptr || entry->conflicted) return GC::kS1Miss;
	if (!entry->s1_covered) return shape(entry->s1_shape, entry->s1_shape_stage);
	if (!SameS1Stages(*entry, *rep)) return shape(GC::kShVariant, 0);
	m_prediction.entry       = entry;
	m_prediction.key         = key;
	m_prediction.stage_count = stages;
	m_prediction.plain       = plain;
	m_prediction.pipeline    = entry->current;
	m_prediction.generation  = entry->current_generation;
	return GC::kS1Emitted;
}

void Session::NoteS1Emitted(uint32_t op_flags) {
	m_result         = {};
	m_op             = op_flags;
	m_result.found   = true;
	m_result.reason  = m_prediction.entry != nullptr ? m_prediction.entry->reason : GC::kS0Simple;
	Report();
	m_consumed = true;
}

// S1 verify: the prediction made for this op against the normal path's preparation of its draw.
void Session::CheckS1(const DrawView& view, uint64_t key, const Entry* entry) {
	m_s1_armed = false;
	if (m_s1_armed_op != view.op_flags) return;
	const auto& pred = m_prediction;
	// S1b: a predicted plain pixel variant whose draw records mip statistics is not emitted
	// (kS1Plain): the normal path's instrumented program is the expected outcome, not compared.
	if (pred.plain && view.ps_mip_stats && !view.plain_pixel) return;
	uint32_t    bad  = 0;
	std::string what;
	const auto  fail = [&](const std::string& text) {
        bad++;
        if (what.size() < 400) what += (what.empty() ? "" : "; ") + text;
	};
	if (key != pred.key) fail(fmt::format("key 0x{:x} predicted 0x{:x}", key, pred.key));
	if (entry != pred.entry) fail("entry");
	if (view.stage_count != pred.stage_count) fail(fmt::format("stages {} predicted {}", view.stage_count, pred.stage_count));
	if (view.plain_pixel != pred.plain) fail(fmt::format("plain pixel {} predicted {}", view.plain_pixel, pred.plain));
	if (pred.entry != nullptr && view.stages[0].program != pred.entry->vs) fail("vs program");
	if (pred.entry != nullptr && view.stage_count > 1 && view.stages[1].program != pred.entry->ps) fail("ps program");
	if (pred.entry != nullptr && view.pipeline->pipeline_layout != pred.entry->layout) fail("layout");
	// S1b: the pipeline object, unless one was replaced since the prediction's selection.
	if (pred.pipeline != nullptr && view.pipeline_generation == pred.generation && view.pipeline->pipeline != pred.pipeline->pipeline) {
		fail("pipeline object");
	}
	const auto words = [&](const char* what_name, const std::vector<IR::DescriptorValue>& a, const std::vector<IR::DescriptorValue>& b) {
		if (a.size() != b.size()) {
			fail(fmt::format("{} count {} predicted {}", what_name, a.size(), b.size()));
			return;
		}
		for (size_t i = 0; i < a.size(); i++) {
			bool same = a[i].dword_count == b[i].dword_count;
			for (uint32_t k = 0; same && k < a[i].dword_count && k < 8u; k++) same = a[i].dwords[k] == b[i].dwords[k];
			if (!same) {
				fail(fmt::format("{} {}: 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x} predicted 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x}", what_name, i,
				                 a[i].dwords[0], a[i].dwords[1], a[i].dwords[2], a[i].dwords[3], b[i].dwords[0], b[i].dwords[1],
				                 b[i].dwords[2], b[i].dwords[3]));
			}
		}
	};
	for (uint32_t s = 0; s < view.stage_count && s < pred.stage_count; s++) {
		const char* stage = s == 0 ? "vs" : "ps";
		const auto* snap  = view.stages[s].snapshot;
		const auto& p     = pred.snapshot[s];
		if (snap == nullptr) {
			fail(fmt::format("{} no snapshot", stage));
			continue;
		}
		words(s == 0 ? "vs V#" : "ps V#", snap->buffers, p.buffers);
		words(s == 0 ? "vs T#" : "ps T#", snap->images, p.images);
		words(s == 0 ? "vs S#" : "ps S#", snap->samplers, p.samplers);
		if (snap->flattened_srt != p.flattened_srt) fail(fmt::format("{} flat SRT ({} dwords)", stage, snap->flattened_srt.size()));
		if (snap->user_data != p.user_data) fail(fmt::format("{} user data ({} dwords)", stage, snap->user_data.size()));
		if (!(snap->uniform_fill == p.uniform_fill)) fail(fmt::format("{} uniform fill", stage));
		if (view.stages[s].spec == nullptr || !(*view.stages[s].spec == pred.specialization[s])) fail(fmt::format("{} specialization", stage));
	}
	// S1c: the rebuilt embedded fetch against the normal path's vertex input info.
	const bool view_fetch = view.vertex_info != nullptr && view.vertex_info->fetch_embedded;
	if (pred.fetch != view_fetch) {
		fail(fmt::format("embedded fetch {} predicted {}", view_fetch, pred.fetch));
	} else if (pred.fetch) {
		const auto& a = *view.vertex_info;
		const auto& b = pred.vertex;
		if (a.resources_num != b.resources_num || a.buffers_num != b.buffers_num) {
			fail(fmt::format("fetch resources {} buffers {} predicted {} {}", a.resources_num, a.buffers_num, b.resources_num, b.buffers_num));
		} else {
			for (int i = 0; i < a.resources_num && i < ShaderVertexInputInfo::RES_MAX; i++) {
				const auto& x = a.resources_dst[i];
				const auto& y = b.resources_dst[i];
				if (std::memcmp(a.resources[i].fields, b.resources[i].fields, sizeof(a.resources[i].fields)) != 0 ||
				    x.register_start != y.register_start || x.registers_num != y.registers_num || x.attr_id != y.attr_id ||
				    x.fetch_index != y.fetch_index) {
					fail(fmt::format("fetch V# {}: 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x} v{}x{} a{} i{} predicted 0x{:08x} 0x{:08x} 0x{:08x} "
					                 "0x{:08x} v{}x{} a{} i{}",
					                 i, a.resources[i].fields[0], a.resources[i].fields[1], a.resources[i].fields[2], a.resources[i].fields[3],
					                 x.register_start, x.registers_num, x.attr_id, x.fetch_index, b.resources[i].fields[0],
					                 b.resources[i].fields[1], b.resources[i].fields[2], b.resources[i].fields[3], y.register_start,
					                 y.registers_num, y.attr_id, y.fetch_index));
				}
			}
			for (int i = 0; i < a.buffers_num && i < ShaderVertexInputInfo::RES_MAX; i++) {
				const auto& x    = a.buffers[i];
				const auto& y    = b.buffers[i];
				bool        same = x.addr == y.addr && x.stride == y.stride && x.num_records == y.num_records &&
				            x.fetch_index == y.fetch_index && x.attr_num == y.attr_num;
				for (int k = 0; same && k < x.attr_num && k < ShaderVertexInputBuffer::ATTR_MAX; k++) {
					same = x.attr_indices[k] == y.attr_indices[k] && x.attr_offsets[k] == y.attr_offsets[k];
				}
				if (!same) {
					fail(fmt::format("fetch buffer {}: 0x{:x} stride {} records {} attrs {} predicted 0x{:x} stride {} records {} attrs {}", i,
					                 x.addr, x.stride, x.num_records, x.attr_num, y.addr, y.stride, y.num_records, y.attr_num));
				}
			}
		}
	}
	m_result.s1_checked    = true;
	m_result.s1_mismatches = bad;
	if (bad != 0) {
		GC::NoteNativeS1Mismatch(view.op_flags, what.c_str());
		if (GC::NativeVerifyEnabled()) {
			EXIT("GameNative s1 verify: native prediction differs from the normal path: %s\n", what.c_str());
		}
	}
}

void Session::Report() {
	GC::NoteNativeS0(m_op, m_result);
	m_active    = false;
	m_capturing = false;
}

} // namespace Libs::Graphics::NativeLearn
