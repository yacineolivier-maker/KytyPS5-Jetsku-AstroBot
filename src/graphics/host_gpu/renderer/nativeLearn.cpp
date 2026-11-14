// Astro Bot PC (KYTY_NATIVE_MODE=s0): native pipeline learning and draw rebuild verification
// (nativeLearn.h).

#include "graphics/host_gpu/renderer/nativeLearn.h"

#include "common/logging/log.h"
#include "kernel/memory.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fmt/format.h>
#include <string>

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

} // namespace

// Append-only open-addressing table (one writer, lock-free readers).
struct Session::Table {
	static constexpr uint32_t Slots      = 16384; // power of two
	static constexpr uint32_t MaxEntries = 8192;
	std::array<std::atomic<Entry*>, Slots> slots {};
	std::array<uint64_t, Slots>            base_keys {}; // writer only
	std::vector<std::unique_ptr<Entry>>    owned;
	uint32_t                               base_count = 0;

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
						k = base;
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
	// NativeKey.
	const uint64_t es_id = es_known ? m_record.pgm_es : m_record.code_geom;
	const uint64_t ps_id = (m_record.known & GC::kNkPgmPs) != 0 ? m_record.pgm_ps : m_record.code_ps;
	uint64_t       base  = 0xcbf29ce484222325ull;
	base                 = Mix(base, es_id);
	base                 = Mix(base, ps_id);
	base                 = Mix(base, m_record.state_hash);
	base                 = Mix(base, view.formats_hash);
	base                 = Mix(base, m_record.prim);
	uint64_t spec        = Mix(0x84222325cbf29ce4ull, view.plain_pixel ? 1u : 0u);
	for (uint32_t s = 0; s < view.stage_count; s++) {
		spec = Mix(spec, view.stages[s].spec != nullptr ? SpecHash(*view.stages[s].spec) : 0u);
	}
	uint64_t key = Mix(base, spec);
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
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
				GC::NoteNativeS0Mismatch(view.op_flags, fmt::format("NativeKey conflict: key 0x{:x} learned vs 0x{:x}/0x{:x}, now 0x{:x}/0x{:x}", key,
				                                                    entry->vs_id, entry->ps_id, view.stages[0].program_id,
				                                                    pixel != nullptr ? pixel->program_id : 0u)
				                                            .c_str());
			}
		} else if (entry->vk_pipeline != view.pipeline->pipeline) {
			m_result.replaced = true; // e.g. a linked or fast-first pipeline swapped for its optimized build
		}
		m_result.reason      = entry->reason;
		m_result.deep_opcode = entry->deep_opcode;
		recipes              = &entry->recipe;
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
}

void Session::Report() {
	GC::NoteNativeS0(m_op, m_result);
	m_active    = false;
	m_capturing = false;
}

} // namespace Libs::Graphics::NativeLearn
