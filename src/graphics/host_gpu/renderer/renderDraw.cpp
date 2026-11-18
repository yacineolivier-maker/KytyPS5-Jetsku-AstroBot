#include "graphics/host_gpu/renderer/passStats.h"
#include "graphics/host_gpu/renderer/renderDraw.h"
#include "graphics/host_gpu/renderer/nativeLearn.h" // Astro Bot PC (KYTY_NATIVE_MODE=s0)

#include "graphics/host_gpu/renderer/drawPrep/ownedCounters.h"

#include "common/assert.h"
#include "common/liveSwitch.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/hangTrace.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/rendererBatch.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/gpuBreadcrumbs.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/cpCommit.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawPrep/bindingPlan.h"
#include "graphics/host_gpu/renderer/drawPrep/commitStats.h"
#include "graphics/host_gpu/renderer/drawPrep/drawPrep.h"
#include "graphics/host_gpu/renderer/drawPrep/drawRun.h"
#include "graphics/host_gpu/renderer/drawPrep/fastCheck.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/meshIndirect.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

static bool MeshRestartEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
		const auto* value = std::getenv("KYTY_MESH_RESTART");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}));
	return enabled;
}

struct MeshDrawSegment {
	uint32_t first;
	uint32_t count;
	uint32_t groups;
};

// Instances one mesh dispatch of `groups` workgroups per instance can carry: instances go in Y,
// so the Y limit and the total workgroup limit both apply. Zero when one instance does not fit.
[[nodiscard]] static uint32_t MeshInstancesPerDispatch(uint32_t groups,
                                                       const vk::PhysicalDeviceMeshShaderPropertiesEXT& limits) {
	if (groups == 0 || groups > limits.maxMeshWorkGroupCount[0]) {
		return 0;
	}
	return std::min(limits.maxMeshWorkGroupCount[1], limits.maxMeshWorkGroupTotalCount / groups);
}

// Workgroups one mesh dispatch takes in X: all of them, or with split_groups (a device whose X
// limit is below its total, RADV) at most that limit; the program adds each part's first group
// (draw dword IR::PushData::MeshFirstGroupDword).
[[nodiscard]] static uint32_t MeshGroupsPerDispatch(uint32_t groups, const ShaderMeshInputInfo& mesh,
                                                    const vk::PhysicalDeviceMeshShaderPropertiesEXT& limits) {
	if (mesh.split_groups == 0) {
		return groups;
	}
	return std::min({groups, limits.maxMeshWorkGroupCount[0], limits.maxMeshWorkGroupTotalCount});
}

static void SplitMeshRestartIndices(uint64_t address, uint32_t count, uint32_t element_size,
                                    uint32_t marker, const ShaderMeshInputInfo& mesh,
                                    std::vector<MeshDrawSegment>& segments) {
	EXIT_IF(address == 0 || (element_size != 1 && element_size != 2 && element_size != 4));
	segments.clear();
	std::array<uint8_t, 4096> bytes;
	uint32_t start = 0;
	uint64_t markers = 0;
	const auto add = [&](uint32_t end) {
		const auto primitives = mesh.InputPrimitiveCount(end - start);
		if (primitives != 0) segments.push_back({start, end - start,
		    (primitives - 1u) / mesh.primitives_per_group + 1u});
	};
	for (uint32_t base = 0; base < count;) {
		const auto chunk = std::min<uint32_t>(count - base, bytes.size() / element_size);
		const auto size = static_cast<size_t>(chunk) * element_size;
		const auto source = address + static_cast<uint64_t>(base) * element_size;
		if (!LibKernel::Memory::TryReadGpuCleanBacking(source, bytes.data(), size)) {
			// The ordinary mapped read follows the existing page-fault/readback path.
			std::memcpy(bytes.data(), reinterpret_cast<const void*>(source), size);
		}
		for (uint32_t i = 0; i < chunk; ++i) {
			uint32_t value = 0;
			std::memcpy(&value, bytes.data() + i * element_size, element_size);
			if (value == marker) {
				add(base + i);
				start = base + i + 1u;
				++markers;
			}
		}
		base += chunk;
	}
	add(count);
	Profiler::CountFrameEvent(Profiler::FrameEvent::MeshRestartMarkers, markers);
	Profiler::CountFrameEvent(Profiler::FrameEvent::MeshRestartSegments, segments.size());
}
std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.fetch_embedded) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
	    static_cast<int>(!depth.desc.info.data.Empty()) +
	        static_cast<int>(depth.desc.info.HasStencil()),
	    ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.program->info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u attr_num=%d\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index, b.attr_num);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < b.attr_num; ai++) {
					const auto  res_index = b.attr_indices[ai];
					const auto& r         = vs_input_info.resources[res_index];
					const auto& rd        = vs_input_info.resources_dst[res_index];
					const auto  offset    = b.attr_offsets[ai];
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < b.attr_num; ai++) {
			const auto  res_index = b.attr_indices[ai];
			const auto& r         = vs_input_info.resources[res_index];
			const auto& rd        = vs_input_info.resources_dst[res_index];
			LOGF("DrawInputState[%u]: attr[%d] res=%d offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, res_index, b.attr_offsets[ai], rd.register_start, rd.registers_num,
			     rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

static bool DynamicStateShadowEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
		const auto* value = std::getenv("KYTY_DYNAMIC_STATE_SHADOW");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}));
	return enabled;
}

// Dynamic-state commands recorded by draws (GPU thread; KYTY_DRAW_RUN=verify reads the difference
// around a continuation's dynamic state). Astro Bot PC (P3 prerequisites, W8): per thread; the
// difference is always taken inside one draw, on its thread, so the values compared are the same.
static thread_local uint64_t g_dynamic_state_emitted = 0;

// Records one dynamic-state group unless the shadow proves the command buffer already holds
// exactly these values (see GraphicsDynamicStateShadow for when that holds).
class DynamicStateRecorder {
public:
	DynamicStateRecorder(GraphicsDynamicStateShadow& shadow, const CommandBuffer& buffer,
	                     vk::CommandBuffer vk_buffer /* identity */)
	    : m_shadow(shadow) {
		m_reuse = DynamicStateShadowEnabled() && shadow.valid && shadow.command == vk_buffer &&
		          shadow.pipeline != nullptr &&
		          buffer.BoundPipeline(vk::PipelineBindPoint::eGraphics) == shadow.pipeline;
		if (!m_reuse) {
			shadow         = {};
			shadow.command = vk_buffer;
		}
	}
	~DynamicStateRecorder() {
		m_shadow.valid = DynamicStateShadowEnabled();
		if (m_emitted != 0) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DynamicStateCommandsEmitted, m_emitted);
			g_dynamic_state_emitted += m_emitted;
		}
		if (m_avoided != 0) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DynamicStateCommandsAvoided, m_avoided);
		}
	}
	DynamicStateRecorder(const DynamicStateRecorder&)            = delete;
	DynamicStateRecorder& operator=(const DynamicStateRecorder&) = delete;

	// Whether the group must be recorded; `current` is the shadow's value for the group and
	// `valid` whether the shadow holds it at all. Stores the new value when recording. Values
	// are compared bitwise (types without padding), so a skipped command would have set
	// bit-identical state.
	template <typename T>
	bool Update(T& current, const T& value, bool valid = true) {
		static_assert(std::is_trivially_copyable_v<T>);
		if (m_reuse && valid && std::memcmp(&current, &value, sizeof(T)) == 0) {
			++m_avoided;
			return false;
		}
		current = value;
		++m_emitted;
		return true;
	}
	[[nodiscard]] bool Reuse() const { return m_reuse; }
	void Emitted(uint64_t count) { m_emitted += count; }
	void Avoided(uint64_t count) { m_avoided += count; }

private:
	GraphicsDynamicStateShadow& m_shadow;
	bool                        m_reuse   = false;
	uint64_t                    m_emitted = 0;
	uint64_t                    m_avoided = 0;
};

// The union of the scissors a draw rasterizes with (SetGraphicsDynamicParams), not yet clamped to
// the framebuffer: every pixel the draw can write lies inside it (KYTY_ALIAS_BYTES claims only
// the render-target blocks under it).
// quiet: on a draw-prep thread, where calc_final_scissor must not log: false for a clip-rect rule
// it does not support (the command processor computes the union then, logging it).
static bool ScissorUnion(const HW::Context& ctx, bool indexed_viewports, bool quiet,
                         vk::Rect2D& result) {
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	const auto&        vp             = ctx.GetScreenViewport();
	const vk::Extent2D unbounded {16384, 16384};
	int64_t            left   = INT64_MAX;
	int64_t            top    = INT64_MAX;
	int64_t            right  = INT64_MIN;
	int64_t            bottom = INT64_MIN;
	for (uint32_t i = 0; i < (indexed_viewports ? viewport_slots : 1u); i++) {
		if (!ctx.GetClipControl().clip_disable && vp.viewports[i].xscale == 0.0f) {
			continue; // an empty slot: zero scissor
		}
		ScissorRect scissor;
		if (quiet) {
			if (!calc_scissor_unclamped(vp, ctx.GetScanModeControl(), i, scissor)) {
				return false;
			}
			scissor = clamp_scissor(scissor, unbounded);
		} else {
			scissor = calc_final_scissor(vp, ctx.GetScanModeControl(), unbounded, i);
		}
		if (scissor.right <= scissor.left || scissor.bottom <= scissor.top) {
			continue;
		}
		left   = std::min<int64_t>(left, scissor.left);
		top    = std::min<int64_t>(top, scissor.top);
		right  = std::max<int64_t>(right, scissor.right);
		bottom = std::max<int64_t>(bottom, scissor.bottom);
	}
	if (right <= left || bottom <= top) {
		result = {};
		return true;
	}
	result = {{static_cast<int32_t>(left), static_cast<int32_t>(top)},
	          {static_cast<uint32_t>(right - left), static_cast<uint32_t>(bottom - top)}};
	return true;
}

vk::Rect2D DrawScissorUnion(const HW::Context& ctx, bool indexed_viewports) {
	vk::Rect2D result {};
	(void)ScissorUnion(ctx, indexed_viewports, false, result);
	return result;
}

bool DrawScissorUnionQuiet(const HW::Context& ctx, bool indexed_viewports, vk::Rect2D& result) {
	return ScissorUnion(ctx, indexed_viewports, true, result);
}

// SetGraphicsDynamicParams' viewport for slot `i`, before an empty slot's width is made positive.
static vk::Viewport MakeDynamicViewport(const HW::Context& ctx, const vk::PhysicalDeviceLimits& limits,
                                        uint32_t i) {
	const auto&  guest = ctx.GetScreenViewport().viewports[i];
	vk::Viewport viewport {};
	if (ctx.GetClipControl().clip_disable) {
		viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
		viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
	} else {
		viewport.x      = guest.xoffset - guest.xscale;
		viewport.y      = guest.yoffset - guest.yscale;
		viewport.width  = guest.xscale * 2.0f;
		viewport.height = guest.yscale * 2.0f;
	}
	viewport.minDepth = guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
	viewport.maxDepth = guest.zscale + guest.zoffset;
	return viewport;
}

bool DrawPrep::PlanDynamicViewports(const HW::Context& ctx, const vk::PhysicalDeviceLimits& limits,
                                    bool indexed_viewports, DynamicViewportPlan& plan) {
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	static_assert(viewport_slots <= DynamicViewportPlan::MaxViewports);
	plan.valid      = false;
	plan.count      = indexed_viewports ? viewport_slots : 1;
	plan.empty_mask = 0;
	for (uint32_t i = 0; i < plan.count; i++) {
		auto& viewport = plan.viewports[i];
		viewport       = MakeDynamicViewport(ctx, limits, i);
		if (!calc_scissor_unclamped(ctx.GetScreenViewport(), ctx.GetScanModeControl(), i,
		                            plan.scissors[i])) {
			return false;
		}
		if (viewport.width == 0.0f) {
			viewport.width = 1.0f;
			plan.empty_mask |= 1u << i;
		}
	}
	plan.valid = true;
	return true;
}

// planned: the viewports and unclamped scissors of the draw's binding plan (KYTY_DRAW_PREP_BINDINGS
// dynamic, computed from the same registers and vertex program), or null.
static void SetGraphicsDynamicParams(const CommandBuffer& buffer, const CommandSink& vk_buffer,
                                     const ShaderVertexInputInfo& vs_input_info,
                                     const RenderDepthInfo& depth, const RenderState& rendering,
                                     GraphicsDynamicStateShadow&            shadow,
                                     const DrawPrep::DynamicViewportPlan* planned) {
	KYTY_PROFILER_FUNCTION();
	DynamicStateRecorder recorder(shadow, buffer, vk_buffer.Identity());

	const auto& ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const auto scissor_rect = [](const ScissorRect& final_scissor) {
		return vk::Rect2D {{final_scissor.left, final_scissor.top},
		                   {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                    static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)}};
	};
	const bool verify = planned != nullptr && DrawPrep::BindingsVerifyMode() != 0;
	uint32_t   viewport_count = 0;
	if (planned != nullptr && !verify) {
		viewport_count = planned->count;
		for (uint32_t i = 0; i < viewport_count; i++) {
			viewports[i] = planned->viewports[i];
			scissors[i]  = scissor_rect(clamp_scissor(planned->scissors[i], framebuffer_extent));
			if ((planned->empty_mask & (1u << i)) != 0) {
				scissors[i].extent = vk::Extent2D {0, 0};
			}
		}
		OwnedCounters::Add(DrawPrep::GetBindingTotals().viewports_used, 1);
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingViewportsUsed);
	} else {
		const auto& outputs = vs_input_info.stage.program->info.outputs;
		const bool  indexed_viewports =
		    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
			    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
		    });
		viewport_count = indexed_viewports ? viewport_slots : 1;
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		for (uint32_t i = 0; i < viewport_count; i++) {
			auto& viewport = viewports[i];
			viewport       = MakeDynamicViewport(ctx, limits, i);

			const auto final_scissor =
			    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
			auto& scissor = scissors[i];
			scissor       = scissor_rect(final_scissor);
			if (viewport.width == 0.0f) {
				// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
				viewport.width = 1.0f;
				scissor.extent = vk::Extent2D {0, 0};
			}
		}
		if (verify) {
			DrawPrep::CountBindingVerifyCheck();
			bool same = planned->count == viewport_count;
			for (uint32_t i = 0; same && i < viewport_count; i++) {
				auto scissor = scissor_rect(clamp_scissor(planned->scissors[i], framebuffer_extent));
				if ((planned->empty_mask & (1u << i)) != 0) {
					scissor.extent = vk::Extent2D {0, 0};
				}
				same = std::memcmp(&planned->viewports[i], &viewports[i], sizeof(vk::Viewport)) == 0 &&
				       std::memcmp(&scissor, &scissors[i], sizeof(vk::Rect2D)) == 0;
			}
			if (!same) {
				DrawPrep::ReportBindingMismatch("dynamic viewports", viewport_count);
			}
		}
	}
	static_assert(viewport_slots <= GraphicsDynamicStateShadow::MaxViewports);
	// Bitwise comparisons: a skipped command must leave bit-identical state.
	const bool same_count = recorder.Reuse() && shadow.viewport_count == viewport_count;
	if (same_count && std::memcmp(shadow.viewports.data(), viewports.data(),
	                              sizeof(vk::Viewport) * viewport_count) == 0) {
		recorder.Avoided(1);
	} else {
		vk_buffer.setViewportWithCount(viewport_count, viewports.data());
		std::copy_n(viewports.begin(), viewport_count, shadow.viewports.begin());
		recorder.Emitted(1);
	}
	if (same_count && std::memcmp(shadow.scissors.data(), scissors.data(),
	                              sizeof(vk::Rect2D) * viewport_count) == 0) {
		recorder.Avoided(1);
	} else {
		vk_buffer.setScissorWithCount(viewport_count, scissors.data());
		std::copy_n(scissors.begin(), viewport_count, shadow.scissors.begin());
		recorder.Emitted(1);
	}
	shadow.viewport_count = viewport_count;

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	if (recorder.Update(shadow.line_width, line_width)) {
		vk_buffer.setLineWidth(line_width);
	}
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	if (recorder.Update(shadow.blend_constants, blend_constants)) {
		vk_buffer.setBlendConstants(blend_constants.data());
	}
	const vk::Bool32 depth_test_enable  = depth.depth_test_enable ? VK_TRUE : VK_FALSE;
	const vk::Bool32 depth_write_enable = depth.depth_write_enable ? VK_TRUE : VK_FALSE;
	if (recorder.Update(shadow.depth_test_enable, depth_test_enable)) {
		vk_buffer.setDepthTestEnable(depth_test_enable);
	}
	if (recorder.Update(shadow.depth_write_enable, depth_write_enable)) {
		vk_buffer.setDepthWriteEnable(depth_write_enable);
	}
	if (recorder.Update(shadow.depth_compare_op, depth.depth_compare_op)) {
		vk_buffer.setDepthCompareOp(depth.depth_compare_op);
	}

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	const vk::Bool32 depth_bias_flag = depth_bias_enable ? VK_TRUE : VK_FALSE;
	if (recorder.Update(shadow.depth_bias_enable, depth_bias_flag)) {
		vk_buffer.setDepthBiasEnable(depth_bias_flag);
	}
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		const std::array<float, 3> bias {constant_factor, poly_offset.clamp, slope_factor};
		if (recorder.Update(shadow.depth_bias, bias, shadow.depth_bias_valid)) {
			vk_buffer.setDepthBias(constant_factor, poly_offset.clamp, slope_factor);
			shadow.depth_bias_valid = true;
		}
	}

	const vk::Bool32 stencil_test_enable = depth.stencil_test_enable ? VK_TRUE : VK_FALSE;
	if (recorder.Update(shadow.stencil_test_enable, stencil_test_enable)) {
		vk_buffer.setStencilTestEnable(stencil_test_enable);
	}
	if (depth.stencil_test_enable) {
		const auto set_stencil = [&](vk::StencilFaceFlagBits face, const vk::StencilOpState& state) {
			vk_buffer.setStencilOp(face, state.failOp, state.passOp, state.depthFailOp, state.compareOp);
			vk_buffer.setStencilCompareMask(face, state.compareMask);
			vk_buffer.setStencilWriteMask(face, state.writeMask);
			vk_buffer.setStencilReference(face, state.reference);
		};
		// Each face is recorded as its group of four commands.
		const bool valid = shadow.stencil_valid;
		if (recorder.Update(shadow.stencil_front, depth.stencil_front, valid)) {
			set_stencil(vk::StencilFaceFlagBits::eFront, depth.stencil_front);
		}
		if (recorder.Update(shadow.stencil_back, depth.stencil_back, valid)) {
			set_stencil(vk::StencilFaceFlagBits::eBack, depth.stencil_back);
		}
		shadow.stencil_valid = true;
	}

	if (PipelineDynamicRasterStateEnabled()) {
		// Formerly pipeline-key fields (PipelineCache::GetGraphicsPipeline), from the same
		// registers: rect lists are drawn without culling.
		const bool        rect_list = Prospero::IsRectList(buffer.GetUserConfig().GetPrimType());
		vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
		if (!rect_list && mode.cull_back) {
			cull_mode |= vk::CullModeFlagBits::eBack;
		}
		if (!rect_list && mode.cull_front) {
			cull_mode |= vk::CullModeFlagBits::eFront;
		}
		if (recorder.Update(shadow.cull_mode, cull_mode)) {
			vk_buffer.setCullMode(cull_mode);
		}
		const auto front_face =
		    mode.face ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise;
		if (recorder.Update(shadow.front_face, front_face)) {
			vk_buffer.setFrontFace(front_face);
		}
#if !defined(__APPLE__)
		const vk::Bool32 bounds_test = depth.depth_bounds_test_enable ? VK_TRUE : VK_FALSE;
		if (recorder.Update(shadow.depth_bounds_test_enable, bounds_test)) {
			vk_buffer.setDepthBoundsTestEnable(bounds_test);
		}
		if (depth.depth_bounds_test_enable) {
			const std::array<float, 2> bounds {depth.depth_min_bounds, depth.depth_max_bounds};
			if (recorder.Update(shadow.depth_bounds, bounds, shadow.depth_bounds_valid)) {
				vk_buffer.setDepthBounds(bounds[0], bounds[1]);
				shadow.depth_bounds_valid = true;
			}
		}
#endif
	}

	// Without VK_EXT_color_write_enable (MoltenVK, older drivers) the pipeline is created without
	// the eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead; an
	// attachment without an image view discards its writes.
	if (buffer.GetGraphics().color_write_enable_enabled && rendering.num_color_attachments != 0) {
		std::array<vk::Bool32, RENDER_COLOR_ATTACHMENTS_MAX> enable {};
		for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
			enable[slot] = rendering.color_attachments[slot].image_view != nullptr;
		}
		const bool same = recorder.Reuse() && shadow.color_write_valid &&
		                  shadow.color_write_count == rendering.num_color_attachments &&
		                  std::memcmp(shadow.color_write.data(), enable.data(),
		                              sizeof(vk::Bool32) * rendering.num_color_attachments) == 0;
		if (same) {
			recorder.Avoided(1);
		} else {
			vk_buffer.setColorWriteEnableEXT(rendering.num_color_attachments, enable.data());
			shadow.color_write       = enable;
			shadow.color_write_count = rendering.num_color_attachments;
			shadow.color_write_valid = true;
			recorder.Emitted(1);
		}
	}
}

// uc_check and hw_check of DrawIndex/DrawAuto. KYTY_DRAW_PREP_BINDINGS hwcheck: skipped when the
// committed draw's binding plan found they would neither stop the emulator nor log for these
// registers, the draw's snapshot bound for its commit (the verdict needs no validated
// preparation: it is a function of the registers and of log state that only moves one way).
static void RunDrawChecks(const CommandBuffer& buffer, const HW::UserConfig& ucfg,
                          const DrawPrep::BindingPlan* plan) {
	if (plan == nullptr || !plan->hw_checks_quiet || DrawPrep::BindingsVerifyMode() != 0) {
		FastCheck::Fail(FastCheck::Cause::PlanNotQuiet); // Astro Bot PC (KYTY_CP_PAR_FASTCHECK)
	}
	if (plan != nullptr && plan->hw_checks_quiet) {
		if (DrawPrep::BindingsVerifyMode() == 0) {
			OwnedCounters::Add(DrawPrep::GetBindingTotals().hw_checks_skipped, 1);
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingHwChecksSkipped);
			return;
		}
		DrawPrep::CountBindingVerifyCheck();
		if (!hw_checks_quiet(buffer.GetRegisters(), ucfg)) {
			DrawPrep::ReportBindingMismatch("hw checks");
		}
	}
	uc_check(ucfg);

	hw_check(buffer);
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

static bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	       db.shader_dual_export_enable || db.shader_execute_on_noop || sh_regs.AlphaToMask();
}

// One per RenderExecutor, reused by every draw instead of being value-initialised (tens of KB)
// each time. Reset() returns it to the value-initialised state for every field a draw reads
// before writing: the fields below `programs` are bookkeeping for that.
struct DrawRenderState {
	RenderDepthInfo       depth_info;
	RenderColorInfo       color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t              color_count                              = 0;
	bool                  ps_active                                = true;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo  ps_input_info;
	PipelineCache::GraphicsPrograms programs;
	// Per-draw program preparation. The vertex_info/ps_input_info stage runtimes point into it;
	// it is overwritten by the next preparation, and its vectors keep their capacity.
	PipelineCache::GraphicsStagePreps stage_preps;
	// color_info entries and vertex_info stages the current draw may have written.
	uint32_t color_slots_written   = 0;
	uint32_t vertex_stages_written = 0;

	void Reset() {
		// KYTY_RENDER_STATE_FAST reset: a draw reads color_info[0, color_count) and depth_info, and
		// its target resolution assigns every field of those (ResolveRenderColorTarget,
		// ResolveRenderDepthTarget). Debug dumps also log color_info[0] and depth_info of draws
		// without such targets, so they still get value-initialised entries.
		if (!RenderStateFastEnabled(RenderStatePart::Reset) || graphics_debug_dump_enabled()) {
			depth_info = {};
			for (uint32_t slot = 0; slot < color_slots_written; slot++) {
				color_info[slot] = {};
			}
		} else {
			Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateMinimalResets);
		}
		color_slots_written = 0;
		color_count         = 0;
		ps_active           = true;
		// PrepareProgram/PrepareTessellationPrograms value-initialise every stage they prepare,
		// and a draw always prepares vertex_info[0]. Only the tessellation stages of an earlier
		// draw can be left behind.
		for (uint32_t stage = 1; stage < vertex_stages_written; stage++) {
			vertex_info[stage] = {};
		}
		vertex_stages_written = 0;
		// RefreshShaders resets ps_input_info and programs itself.
	}
};

RenderExecutor::RenderExecutor(RenderContext& context)
    : m_context(context), m_draw_state(std::make_unique<DrawRenderState>()) {}

RenderExecutor::~RenderExecutor() = default;

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages,
                                                 const vk::Rect2D* written) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects       = {};
	m_depth_feedback.valid = false;
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	CommitStats::SubScope color_scope(CommitStats::Sub::AcquireColor);
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto owner = cache.m_slot_images.try_get(target.image_id);
		if (owner == nullptr || (!owner->registered && !owner->info.data.Empty()) ||
		    owner->Binding().needs_rebind) {
			EXIT("color target changed after render-state discovery\n");
		}
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc,
		                                               target.guest_mip_level == 0 ? written
		                                                                           : nullptr);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.Binding().is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.Binding().attachment_layout = layout;
		image.Binding().attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.Binding().attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Identity(), true);
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	color_scope.Stop();
	CommitStats::SubScope depth_scope(CommitStats::Sub::AcquireDepth);
	if (depth.image_id) {
		const auto owner = cache.m_slot_images.try_get(depth.image_id);
		if (owner == nullptr || !owner->registered || owner->Binding().needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
		// FindDepthTarget treats the binding as a write. Restored below for draws that write
		// neither aspect, so an alias synchronized from this image stays provably identical.
		const auto  content_mark = cache.MarkContent(depth.image_id);
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		if (metadata.kind == ImageMetadataKind::Htile && depth.depth_clear_enable &&
		    !cache.ClearMeta(metadata.range.address)) {
			EXIT("failed to acquire HTile metadata for a depth clear\n");
		}
		const bool meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		depth.load_clear_aspects = {};
		if (depth.depth_clear_enable || meta_clear)
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eDepth;
		// Apply pending clears to compressed stencil as well as depth (upstream 5bf37f52a).
		if (depth.stencil_clear_enable || (meta_clear && metadata.stencil_compressed))
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eStencil;
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		if (!draw_writes) {
			// Load-op LOAD, store-op STORE, no depth/stencil writes or clears: contents unchanged.
			cache.RestoreContentIfUnwritten(depth.image_id, content_mark);
		}
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			// Without VK_EXT_attachment_feedback_loop_layout/dynamic_state (AMD's Windows driver)
			// the draw samples the depth target it writes in the GENERAL layout, chosen below: not
			// defined by Vulkan, but what the hardware does, and better than ending the emulator.
			static std::atomic_bool warned {false};
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog("Warning: depth attachment feedback loop without host "
				                          "support; using the GENERAL layout\n");
			}
		}
		image.CpParTouch(); // Astro Bot PC (KYTY_CP_PAR=1b3): the overlay state read below
		auto layout = depth_attachment_layout(depth);
		if (!sampled_aspects && DepthLayoutStableEnabled()) {
			// Nothing in this draw samples the image, so its attachment layout is not observable:
			// keep the current one while it allows the draw's writes (no transition, no new
			// rendering instance).
			const auto stable = depth_stable_attachment_layout(
			    image.backing.state.layout, image.backing.subresource_states.empty(),
			    ImageViewOps::DepthAspectMask(depth.desc.view_info.format), draw_writes);
			if (stable != layout && stable == image.backing.state.layout) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DepthLayoutTransitionsAvoided);
			}
			layout = stable;
		}
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		auto        access = vk::AccessFlags2 {vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                                vk::AccessFlagBits2::eDepthStencilAttachmentWrite};
		const auto& view   = depth.desc.view_info;
		// KYTY_DEPTH_FEEDBACK_KEEP: inside the active rendering instance, with no attachment
		// write (draw writes, load clears) since it began - the content serial recorded then is
		// unchanged and this draw writes nothing - every access to the image in the instance is a
		// read (depth/stencil tests, sampling). Reads need no ordering among themselves, so the
		// attachment <-> attachment+shader-read toggles of sampling draws, a barrier and a new
		// instance per draw, are left out: the image keeps the union of both scopes.
		const auto sampled = vk::AccessFlags2 {access | vk::AccessFlagBits2::eShaderRead};
		const bool whole   = view.base_level == 0 && view.base_layer == 0 &&
		                   view.level_count == image.info.resources.levels &&
		                   view.layer_count == image.info.resources.layers;
		const auto& tracked = image.backing.state;
		const bool  proof_instance = image.feedback_instance != 0 &&
		                            image.feedback_instance == buffer.ActiveRenderingSerial();
		const bool  proof_serial   = image.feedback_serial == image.ContentSerial();
		const bool  state_matches  = whole && image.backing.subresource_states.empty() &&
		                            tracked.layout == layout &&
		                            (tracked.access_mask == access || tracked.access_mask == sampled);
		if (DepthFeedbackKeepEnabled() && sampled_aspects &&
		    (draw_writes || !proof_instance || !proof_serial || !state_matches)) {
			// Aggregate profile: why a sampled depth attachment keeps its per-draw toggles.
			Profiler::CountFrameEvent(draw_writes      ? Profiler::FrameEvent::DepthFeedbackKeepMissWrite
			                          : !proof_instance ? Profiler::FrameEvent::DepthFeedbackKeepMissInstance
			                          : !proof_serial   ? Profiler::FrameEvent::DepthFeedbackKeepMissSerial
			                                            : Profiler::FrameEvent::DepthFeedbackKeepMissState);
		}
		const bool keep = DepthFeedbackKeepEnabled() && !draw_writes && proof_instance &&
		                  proof_serial && state_matches;
		if (keep && !DepthFeedbackAdoptsUnion(true, DepthFeedbackLazyEnabled(),
		                                      static_cast<bool>(sampled_aspects),
		                                      tracked.access_mask == access)) {
			// KYTY_DEPTH_FEEDBACK_LAZY: nothing samples the image yet; it keeps the attachment access.
			DrawRun::GetTotals().depth_promotions_deferred.fetch_add(1, std::memory_order_relaxed);
		} else if (keep) {
			// Without the keep: a barrier back to the attachment scope when the previous draw
			// sampled, and one to the sampled scope when this draw samples.
			const uint64_t avoided = (tracked.access_mask == sampled ? 1u : 0u) +
			                         (sampled_aspects ? 1u : 0u);
			if (avoided != 0) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DepthFeedbackBarriersAvoided,
				                          avoided);
			}
			vk::ImageMemoryBarrier2 ordering {};
			ordering.srcStageMask        = tracked.pl_stage;
			ordering.srcAccessMask       = sampled;
			ordering.dstStageMask        = tracked.pl_stage;
			ordering.dstAccessMask       = sampled;
			ordering.oldLayout           = layout;
			ordering.newLayout           = layout;
			ordering.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			ordering.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			ordering.image               = image.backing.image;
			ordering.subresourceRange    = {ImageViewOps::DepthAspectMask(image.backing.format), 0,
			                                VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
			buffer.NoteFeedbackKeep(ordering);
			image.AdoptState(tracked.pl_stage, sampled, layout);
			access = sampled;
		}
		m_depth_feedback = {depth.image_id, static_cast<bool>(draw_writes), true};
		image.Binding().attachment_layout = layout;
		image.Binding().attachment_access = access;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Identity(), true);
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eStencil);
	}
	depth_scope.Stop();
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

void RenderExecutor::NoteDepthFeedback(const CommandBuffer& buffer) {
	m_depth_feedback.valid = false;
	auto* image = m_context.GetTextureCache().m_slot_images.try_get(m_depth_feedback.id);
	if (image == nullptr) {
		return;
	}
	const auto instance      = buffer.ActiveRenderingSerial();
	const bool first_draw    = image->feedback_attached != instance;
	image->feedback_attached = instance;
	if (m_depth_feedback.writes || instance == 0) {
		// This draw writes the attachment (or clears it on load): the instance is not read-only.
		image->feedback_instance = 0;
		image->feedback_serial   = 0;
		return;
	}
	if (first_draw) {
		// The first draw of this instance attaching the image, and it writes nothing: later draws
		// of the instance compare the content serial against this one. An instance whose earlier
		// draw wrote the image keeps feedback_instance 0 until it ends.
		image->feedback_instance = instance;
		image->feedback_serial   = image->ContentSerial();
	}
}

static bool DrawHasActivePixelShader(const CommandBuffer& buffer) {
	const auto& ctx     = buffer.GetRegisters();
	const auto& sh_regs = ctx.GetShaderRegisters();
	// KYTY_SKIP_INACTIVE_PS also requires a nonzero export format (DrawColorOutputFilter).
	const bool has_color_output = SkipInactivePixelShadersEnabled()
	                                  ? DrawColorOutputFilter(ctx) != 0
	                                  : (ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask) != 0;
	return buffer.GetShaders().GetPs().ps_regs.data_addr != 0 &&
	       (has_color_output || PixelShaderHasDepthOrCoverageSideEffects(sh_regs));
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

// These special modes run color-buffer metadata or decompression operations. The shader is a
// vehicle for that operation, and its exported color must not be applied as a normal draw.
// Kyty stores expanded Vulkan images rather than compressed guest surfaces. DCC and CMASK
// clear state must be materialized here even when no ordinary draw ever binds the attachment;
// a later sampled image or buffer reader need not carry its metadata. FMask is not implemented.
static bool IsMetadataColorMode(uint8_t mode) {
	return mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) ||
	       mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

// KYTY_CB_METADATA_MATERIALIZE=0 restores the old diagnostic behavior of dropping the operation.
// By default a fast-clear-eliminate or DCC-decompress draw first resolves each supported
// render-target-tiled colour target it names (FindImage), which materializes a pending uniform
// DCC or CMASK fast clear into the image and consumes the
// clear key, exactly as a later attachment bind would. On hardware these draws exist to write the
// cleared values into the surface memory for readers without DCC metadata (a T# without META,
// buffer reads); dropping them left such readers with the surface's previous contents whenever
// no draw bound the target in between. Astro Bot's water input uses exactly this CMASK route:
// dropping the eliminate exposes stale scene texels at normal and top-down camera angles.
static bool MetadataColorMaterializeEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CB_METADATA_MATERIALIZE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

bool RenderExecutor::ConsumeMetadataColorOperation(CommandBuffer& buffer,
                                                   uint32_t      render_target_slice_offset) {
	const auto& hw   = buffer.GetRegisters();
	const auto  mode = hw.GetColorControl().mode;
	if (!IsMetadataColorMode(mode)) {
		return false;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::MetadataColorOps);
	const bool materialize =
	    MetadataColorMaterializeEnabled() &&
	    (mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) ||
	     mode == static_cast<uint8_t>(CbColorMode::DccDecompress));
	const auto target_mask = hw.GetRenderTargetMask();
	for (uint32_t slot = 0; slot < 8; slot++) {
		const auto& rt = hw.GetRenderTarget(slot);
		if (!render_target_bound(rt) || (slot != 0 && render_target_mask_slot(target_mask, slot) == 0)) {
			continue;
		}
		const bool dcc = rt.info.dcc_compression_enable && rt.dcc_addr.addr != 0;
		// A pending CMASK fast clear is written by the eliminate (TextureCache::MaterializeCmaskClear).
		const bool cmask = mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) &&
		                   rt.info.cmask_fast_clear_enable && rt.cmask.addr != 0 &&
		                   CmaskFastClearEnabled();
		// Only the layouts ResolveRenderColorTarget supports with DCC: render-target tiled,
		// single sample.
		const bool eligible = materialize && (dcc || cmask) &&
		                      rt.attrib3.tile_mode == Prospero::TileMode::kRenderTarget &&
		                      ((rt.attrib.num_samples == 0 && rt.attrib.num_fragments == 0) ||
		                       (cmask && !dcc && CmaskMsaaEnabled() &&
		                        rt.attrib.num_samples == rt.attrib.num_fragments));
		uint64_t image_size = 0;
		if (eligible) {
			RenderColorInfo target {};
			ResolveRenderColorTarget(buffer, target, render_target_slice_offset, slot, true, false);
			if (target.image_id) {
				image_size = target.desc.info.data.size;
				Profiler::CountFrameEvent(Profiler::FrameEvent::MetadataColorOpMaterializations);
			}
		}
		if (HangTrace::CpTraceEnabled()) {
			// cp.csv: address = colour base, value = CB_COLOR_CONTROL.MODE, ref = DCC base,
			// mask = CMASK base, aux = slot | DCC enable << 8 | tile mode << 16 | resolved << 24,
			// size = the resolved image's guest size (0 when not resolved).
			HangTrace::CpEvent event;
			event.event   = "cb-meta-op";
			event.address = rt.base.addr;
			event.value   = mode;
			event.ref     = rt.dcc_addr.addr;
			event.mask    = rt.cmask.addr;
			event.aux     = static_cast<int64_t>(slot) | (static_cast<int64_t>(dcc ? 1 : 0) << 8) |
			            (static_cast<int64_t>(rt.attrib3.tile_mode) << 16) |
			            (static_cast<int64_t>(image_size != 0 ? 1 : 0) << 24);
			event.size = image_size;
			HangTrace::RecordCp(event);
		}
	}
	return true;
}

struct DrawEmitInfo {
	int32_t  vertex_offset = 0;
	uint32_t first_vertex  = 0;
	uint32_t first_instance = 0;
	// Non-null: counts and offsets are GPU data in these records, emitted with
	// vkCmdDraw*Indirect*; the fields above are unused.
	const DrawIndirectSource* indirect = nullptr;
	// KYTY_NATIVE_INDIRECT_MESH: the indirect record feeds a mesh draw through a GPU conversion
	// (meshIndirect.h) with these inputs.
	bool                 mesh_indirect = false;
	MeshIndirect::Inputs mesh_inputs;
};

struct DrawIndexBufferSource {
	uint64_t      address   = 0;
	const void*   host_data = nullptr;
	uint64_t      size      = 0;
	vk::IndexType type      = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

struct PreparedIndirectBuffers {
	vk::Buffer     args         = nullptr;
	vk::DeviceSize args_offset  = 0;
	vk::Buffer     count        = nullptr;
	vk::DeviceSize count_offset = 0;
};

// Pure superset of the ConsumeMetadataColorOperation, DepthStencilCopy and ResolveColorTargets
// conditions. Those draws run an operation instead of drawing, but only when their CPU-visible
// counts are nonzero, so indirect draws in these modes keep the CPU-read arguments.
static bool MayRunTargetOperation(const HW::Context& hw) {
	const auto mode = hw.GetColorControl().mode;
	if (IsMetadataColorMode(mode) || mode == static_cast<uint8_t>(CbColorMode::Resolve) ||
	    hw.GetRenderControl().copy_depth_to_color) { // RenderExecutor::CopyDepthToColor
		return true;
	}
	const auto& override = hw.GetDepthRenderOverride();
	return mode == static_cast<uint8_t>(CbColorMode::Disable) &&
	       ((override.force_z_dirty && override.force_z_valid) ||
	        (override.force_stencil_dirty && override.force_stencil_valid));
}

static bool DrawMayRunTargetOperation(const CommandBuffer& buffer) {
	return MayRunTargetOperation(buffer.GetRegisters());
}

// ResolvePrimitiveRestart without its index scan. nullopt when only a scan of CPU-visible
// indices could decide (a custom reset value), or when the CPU path would reject the control.
static std::optional<bool> ResolveNativePrimitiveRestart(const CommandBuffer& buffer,
                                                         uint32_t element_size) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	if ((control & ~0x3u) != 0) {
		return std::nullopt;
	}
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}
	const auto index_mask  = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	if ((reset_index & index_mask) == index_mask) {
		return true;
	}
	return std::nullopt;
}

// Argument and count records stay in GPU memory. The stream-buffer path of ObtainBuffer only
// applies to CPU-written ranges, whose snapshot is then equally current.
static PreparedIndirectBuffers ObtainIndirectBuffers(CommandBuffer&            buffer,
                                                     const DrawIndirectSource& source) {
	auto&                   cache = buffer.GetContext().GetBufferCache();
	PreparedIndirectBuffers prepared;
	const auto [args, args_offset] = cache.ObtainBuffer(source.args_addr, source.ArgsSize(), false);
	EXIT_IF(args == nullptr || (args_offset & 3u) != 0);
	prepared.args        = args->Handle();
	prepared.args_offset = args_offset;
	if (source.count_addr != 0) {
		const auto [count, count_offset] =
		    cache.ObtainBuffer(source.count_addr, sizeof(uint32_t), false);
		EXIT_IF(count == nullptr || (count_offset & 3u) != 0);
		prepared.count        = count->Handle();
		prepared.count_offset = count_offset;
	}
	return prepared;
}

// Shader and transfer writes (compute culling, DMA, cache uploads) -> indirect command fetch.
// Must be recorded outside a rendering instance.
static void IndirectArgumentsBarrier(const CommandBuffer& buffer, const CommandSink& vk_buffer) {
	if (BarrierBatchEnabled()) {
		// Recorded (outside rendering) by the BeginRendering() that precedes the draw.
		buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eAllGraphics |
		                                vk::PipelineStageFlagBits2::eComputeShader |
		                                vk::PipelineStageFlagBits2::eTransfer,
		                            vk::AccessFlagBits2::eShaderWrite |
		                                vk::AccessFlagBits2::eTransferWrite,
		                            vk::PipelineStageFlagBits2::eDrawIndirect,
		                            vk::AccessFlagBits2::eIndirectCommandRead,
		                            BarrierOrigin::IndirectArgs);
		return;
	}
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
	                              vk::PipelineStageFlagBits::eComputeShader |
	                              vk::PipelineStageFlagBits::eTransfer,
	                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier, 0,
	                          nullptr, 0, nullptr);
}

static void EmitIndirectDraw(const CommandSink& vk_buffer, const DrawIndirectSource& source,
                             const PreparedIndirectBuffers& prepared) {
	if (source.count_addr != 0) {
		if (source.indexed) {
			vk_buffer.drawIndexedIndirectCount(prepared.args, prepared.args_offset, prepared.count,
			                                   prepared.count_offset, source.max_count,
			                                   source.stride);
		} else {
			vk_buffer.drawIndirectCount(prepared.args, prepared.args_offset, prepared.count,
			                            prepared.count_offset, source.max_count, source.stride);
		}
	} else if (source.indexed) {
		vk_buffer.drawIndexedIndirect(prepared.args, prepared.args_offset, source.max_count,
		                              source.stride);
	} else {
		vk_buffer.drawIndirect(prepared.args, prepared.args_offset, source.max_count,
		                       source.stride);
	}
}

static uint64_t VertexBufferDescriptorSize(const ShaderVertexInputBuffer& buffer,
                                           const ShaderVertexInputInfo& info) {
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < buffer.attr_num; i++) {
		const auto& resource = info.resources[buffer.attr_indices[i]];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? static_cast<uint64_t>(buffer.attr_offsets[i]) +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

bool DrawPrep::CollectVertexRanges(const ShaderVertexInputInfo& vs_input_info,
                                   VertexRangePlan& ranges, uint32_t& invalid) {
	// Collect the non-empty guest vertex ranges.
	using Merged = VertexRangePlan::Merged;
	std::array<Merged, VertexRangePlan::MaxBuffers> collected {};
	uint32_t                                        range_count = 0;
	ranges.buffer_count = static_cast<uint32_t>(vs_input_info.buffers_num);
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(vertex, vs_input_info);
		ranges.sizes[i]    = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			invalid = static_cast<uint32_t>(i);
			return false;
		}
		collected[range_count++] = {vertex.addr, vertex.addr + size, 0};
	}

	std::sort(collected.begin(), collected.begin() + range_count,
	          [](const Merged& left, const Merged& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	ranges.merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = collected[i];
		if (ranges.merged_count != 0 &&
		    ranges.merged[ranges.merged_count - 1].requested_end >= range.base_address) {
			ranges.merged[ranges.merged_count - 1].requested_end =
			    std::max(ranges.merged[ranges.merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		ranges.merged[ranges.merged_count++] = {range.base_address, range.requested_end, 0};
	}
	return true;
}

bool DrawPrep::AssignVertexRanges(const ShaderVertexInputInfo& vs_input_info,
                                  VertexRangePlan& ranges, uint32_t& unassigned) {
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		if (ranges.sizes[i] == 0) {
			continue;
		}
		const auto address = vs_input_info.buffers[i].addr;
		uint32_t   index   = 0;
		while (index < ranges.merged_count && !(address >= ranges.merged[index].base_address &&
		                                        address < ranges.merged[index].acquired_end)) {
			index++;
		}
		if (index == ranges.merged_count) {
			unassigned = static_cast<uint32_t>(i);
			return false;
		}
		ranges.merged_index[i] = static_cast<uint8_t>(index);
	}
	return true;
}

// KYTY_DRAW_PREP_BINDINGS_VERIFY: a plan's vertex ranges against the serial ones.
static bool SameVertexRanges(const DrawPrep::VertexRangePlan& a, const DrawPrep::VertexRangePlan& b) {
	if (a.buffer_count != b.buffer_count || a.merged_count != b.merged_count) {
		return false;
	}
	for (uint32_t i = 0; i < a.buffer_count; i++) {
		if (a.sizes[i] != b.sizes[i] || (a.sizes[i] != 0 && a.merged_index[i] != b.merged_index[i])) {
			return false;
		}
	}
	for (uint32_t i = 0; i < a.merged_count; i++) {
		if (a.merged[i].base_address != b.merged[i].base_address ||
		    a.merged[i].requested_end != b.merged[i].requested_end ||
		    a.merged[i].acquired_end != b.merged[i].acquired_end) {
			return false;
		}
	}
	return true;
}

// planned: the draw's guest ranges from its binding plan (KYTY_DRAW_PREP_BINDINGS, clamped by the
// preparing thread while the guest virtual ranges were as they are now), or null.
static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&                   buffer,
                                                  const ShaderVertexInputInfo&     vs_input_info,
                                                  const DrawPrep::VertexRangePlan* planned) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	const bool verify = planned != nullptr && DrawPrep::BindingsVerifyMode() != 0;
	const DrawPrep::VertexRangePlan* ranges = planned;
	DrawPrep::VertexRangePlan        collected;
	if (planned == nullptr || verify) {
		uint32_t invalid = 0;
		if (!DrawPrep::CollectVertexRanges(vs_input_info, collected, invalid)) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vs_input_info.buffers[invalid].addr, collected.sizes[invalid]);
		}
		for (uint32_t i = 0; i < collected.merged_count; i++) {
			auto& range = collected.merged[i];
			// PPSA20298
			const auto size = Libs::LibKernel::Memory::ClampRangeSize(
			    range.base_address, range.requested_end - range.base_address);
			range.acquired_end = range.base_address + size;
		}
		uint32_t unassigned = 0;
		if (!DrawPrep::AssignVertexRanges(vs_input_info, collected, unassigned)) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vs_input_info.buffers[unassigned].addr);
		}
		if (verify) {
			DrawPrep::CountBindingVerifyCheck();
			if (!SameVertexRanges(*planned, collected)) {
				DrawPrep::ReportBindingMismatch("vertex ranges", collected.merged_count);
			}
		}
		ranges = &collected;
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	std::array<std::pair<Buffer*, uint64_t>, ShaderVertexInputInfo::RES_MAX> acquired {};
	for (uint32_t i = 0; i < ranges->merged_count; i++) {
		const auto& range = ranges->merged[i];
		const auto  size  = range.acquired_end - range.base_address;
		acquired[i]       = cache.ObtainBuffer(range.base_address, size, false);
		SetVulkanObjectNameF(
		    buffer.GetContext().GetGraphics().device, acquired[i].first->Handle(),
		    "Kyty.VertexBufferRange[guest=0x{:016x} size=0x{:x}]", range.base_address, size);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = ranges->sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto  index = ranges->merged_index[i];
		const auto& range = ranges->merged[index];
		prepared.buffers[i] = acquired[index].first->Handle();
		prepared.offsets[i] = acquired[index].second + vertex.addr - range.base_address;
		prepared.sizes[i]   = std::min(size, range.acquired_end - vertex.addr);
		SetVulkanObjectNameF(
		    buffer.GetContext().GetGraphics().device, prepared.buffers[i],
		    "Kyty.VertexBuffer[slot={} guest=0x{:016x} size=0x{:x} stride={} records={}]", i,
		    vertex.addr, size, vertex.stride, vertex.num_records);
	}

	return prepared;
}

// KYTY_CP_COMMIT=draws: AcquireVertexBuffers into the draw's value-initialised bindings, without
// its dead work: the collected range plan (about 1 KiB value-initialised) exists only when the
// draw has no binding plan (or verifies it), the acquired ranges are not value-initialised (only
// the merged ones are written and read), and no second PreparedVertexBuffers is built and copied.
// Every value it computes and every call it makes, in order, are AcquireVertexBuffers'.
static void AcquireVertexBuffersInto(CommandBuffer&                   buffer,
                                     const ShaderVertexInputInfo&     vs_input_info,
                                     const DrawPrep::VertexRangePlan* planned,
                                     PreparedVertexBuffers&           prepared) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);
	EXIT_IF(prepared.count != 0); // the draw's bindings, value-initialised

	const bool verify = planned != nullptr && DrawPrep::BindingsVerifyMode() != 0;
	const DrawPrep::VertexRangePlan*          ranges = planned;
	std::optional<DrawPrep::VertexRangePlan> collected_storage;
	if (planned == nullptr || verify) {
		auto&    collected = collected_storage.emplace();
		uint32_t invalid   = 0;
		if (!DrawPrep::CollectVertexRanges(vs_input_info, collected, invalid)) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vs_input_info.buffers[invalid].addr, collected.sizes[invalid]);
		}
		for (uint32_t i = 0; i < collected.merged_count; i++) {
			auto& range = collected.merged[i];
			// PPSA20298
			const auto size = Libs::LibKernel::Memory::ClampRangeSize(
			    range.base_address, range.requested_end - range.base_address);
			range.acquired_end = range.base_address + size;
		}
		uint32_t unassigned = 0;
		if (!DrawPrep::AssignVertexRanges(vs_input_info, collected, unassigned)) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vs_input_info.buffers[unassigned].addr);
		}
		if (verify) {
			DrawPrep::CountBindingVerifyCheck();
			if (!SameVertexRanges(*planned, collected)) {
				DrawPrep::ReportBindingMismatch("vertex ranges", collected.merged_count);
			}
		}
		ranges = &collected;
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	struct Acquired {
		Buffer*  buffer;
		uint64_t offset;
	};
	std::array<Acquired, ShaderVertexInputInfo::RES_MAX> acquired; // [0, merged_count) written
	for (uint32_t i = 0; i < ranges->merged_count; i++) {
		const auto& range           = ranges->merged[i];
		const auto  size            = range.acquired_end - range.base_address;
		const auto [handle, offset] = cache.ObtainBuffer(range.base_address, size, false);
		acquired[i]                 = {handle, offset};
		SetVulkanObjectNameF(
		    buffer.GetContext().GetGraphics().device, handle->Handle(),
		    "Kyty.VertexBufferRange[guest=0x{:016x} size=0x{:x}]", range.base_address, size);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = ranges->sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto  index = ranges->merged_index[i];
		const auto& range = ranges->merged[index];
		prepared.buffers[i] = acquired[index].buffer->Handle();
		prepared.offsets[i] = acquired[index].offset + vertex.addr - range.base_address;
		prepared.sizes[i]   = std::min(size, range.acquired_end - vertex.addr);
		SetVulkanObjectNameF(
		    buffer.GetContext().GetGraphics().device, prepared.buffers[i],
		    "Kyty.VertexBuffer[slot={} guest=0x{:016x} size=0x{:x} stride={} records={}]", i,
		    vertex.addr, size, vertex.stride, vertex.num_records);
	}
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              const DrawEmitInfo& emit, uint32_t phase) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count,
	                    emit.indirect != nullptr ? emit.indirect->args_addr : draw.first_instance);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kPolygon:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

// Draw-prep (drawPrep.h): the early returns of DrawIndex/DrawAuto before PrepareDrawRenderState,
// evaluated on a register snapshot (target operations as their pure superset).
bool DrawPrep::DrawReachesPrograms(const HW::Context& context, const HW::UserConfig& user_config,
                                   const HW::Shader& shaders, uint32_t count,
                                   uint32_t instance_count) {
	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	return count != 0 && instance_count != 0 && !MayRunTargetOperation(context) &&
	       DrawHasValidVertexShader(shaders) && GetDrawTopology(user_config, topology);
}

bool DrawPrep::DrawTopology(const HW::UserConfig& user_config, vk::PrimitiveTopology& topology) {
	return GetDrawTopology(user_config, topology);
}

bool DrawPrep::MeshPrimitiveRestartEnabled() {
	return MeshRestartEnabled();
}

DrawPrep::RestartDecision DrawPrep::DecidePrimitiveRestart(const HW::UserConfig& user_config,
                                                           const HW::Context&    registers,
                                                           uint32_t element_size, bool allow_custom) {
	const auto control = user_config.GetPrimitiveResetControl();
	if ((control & ~0x3u) != 0) {
		return RestartDecision::Unsupported;
	}
	if ((control & 0x1u) == 0) {
		return RestartDecision::Disabled;
	}
	switch (user_config.GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return RestartDecision::Disabled;
	}

	const auto index_mask  = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index = registers.GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return RestartDecision::Disabled;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask || allow_custom) {
		// Native assembly uses the maximum marker; the mesh path also accepts custom values.
		return RestartDecision::Enabled;
	}
	return RestartDecision::Scan;
}

uint32_t DrawPrep::PrimitiveRestartIndex(const HW::Context& registers, uint32_t element_size) {
	return registers.GetPrimitiveResetIndex() & (UINT32_MAX >> ((4 - element_size) * 8));
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source, bool allow_custom = false) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	const auto element_size = source.guest_element_size;
	switch (DrawPrep::DecidePrimitiveRestart(buffer.GetUserConfig(), buffer.GetRegisters(),
	                                         element_size, allow_custom)) {
		case DrawPrep::RestartDecision::Disabled:
		case DrawPrep::RestartDecision::Unsupported: return false;
		case DrawPrep::RestartDecision::Enabled: return true;
		case DrawPrep::RestartDecision::Scan: break;
	}
	FastCheck::Fail(FastCheck::Cause::RestartScan); // Astro Bot PC (KYTY_CP_PAR_FASTCHECK)
	const auto restart_index = DrawPrep::PrimitiveRestartIndex(buffer.GetRegisters(), element_size);

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before final binding preparation: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

// KYTY_RENDER_STATE_FAST vertex (renderDraw.h): copies a vertex input as the used prefixes of its
// arrays (and of each buffer's attribute list) and the fields after the arrays, and resets the
// entries the destination used past the source's counts. Entries past the counts hold their
// default values in every vertex input this copies from or into: the program preparations
// value-initialise the whole structure and fill only these prefixes
// (ShaderGetStaticVertexInputInfo; a failed preparation is not used), the draw state starts
// value-initialised, and this copy keeps it so. The destination then holds the bytes of a full
// copy (KYTY_RENDER_STATE_VERIFY compares them).
bool CopyVertexInputPrefixes(ShaderVertexInputInfo& dst, const ShaderVertexInputInfo& src) {
	static_assert(std::is_trivially_copyable_v<ShaderVertexInputInfo>);
	constexpr int MaxEntries    = ShaderVertexInputInfo::RES_MAX;
	constexpr int MaxAttributes = ShaderVertexInputBuffer::ATTR_MAX;
	const auto counts_valid = [](const ShaderVertexInputInfo& info) {
		if (info.resources_num < 0 || info.resources_num > MaxEntries || info.buffers_num < 0 ||
		    info.buffers_num > MaxEntries) {
			return false;
		}
		for (int i = 0; i < info.buffers_num; i++) {
			if (info.buffers[i].attr_num < 0 || info.buffers[i].attr_num > MaxAttributes) {
				return false;
			}
		}
		return true;
	};
	if (!counts_valid(src) || !counts_valid(dst)) {
		return false;
	}
	// The three arrays lead the structure; everything from `stage` on is copied as one block.
	const auto* base = reinterpret_cast<const std::byte*>(&src);
	const auto  tail = sizeof(src.resources) + sizeof(src.resources_dst) + sizeof(src.buffers);
	EXIT_IF(reinterpret_cast<const std::byte*>(&src.resources) != base ||
	        reinterpret_cast<const std::byte*>(&src.resources_dst) != base + sizeof(src.resources) ||
	        reinterpret_cast<const std::byte*>(&src.buffers) !=
	            base + sizeof(src.resources) + sizeof(src.resources_dst) ||
	        reinterpret_cast<const std::byte*>(&src.stage) != base + tail);

	const int resources     = src.resources_num;
	const int old_resources = dst.resources_num;
	std::copy_n(src.resources, resources, dst.resources);
	std::copy_n(src.resources_dst, resources, dst.resources_dst);
	for (int i = resources; i < old_resources; i++) {
		dst.resources[i]     = {};
		dst.resources_dst[i] = {};
	}
	const int buffers     = src.buffers_num;
	const int old_buffers = dst.buffers_num;
	for (int i = 0; i < std::max(buffers, old_buffers); i++) {
		auto&     to        = dst.buffers[i];
		const int old_attrs = i < old_buffers ? to.attr_num : 0;
		int       new_attrs = 0;
		if (i < buffers) {
			const auto& from = src.buffers[i];
			new_attrs        = from.attr_num;
			to.addr          = from.addr;
			to.stride        = from.stride;
			to.num_records   = from.num_records;
			to.fetch_index   = from.fetch_index;
			to.attr_num      = from.attr_num;
			std::copy_n(from.attr_indices, new_attrs, to.attr_indices);
			std::copy_n(from.attr_offsets, new_attrs, to.attr_offsets);
		} else {
			to.addr        = 0;
			to.stride      = 0;
			to.num_records = 0;
			to.fetch_index = 0;
			to.attr_num    = 0;
		}
		if (old_attrs > new_attrs) {
			std::fill(to.attr_indices + new_attrs, to.attr_indices + old_attrs, 0);
			std::fill(to.attr_offsets + new_attrs, to.attr_offsets + old_attrs, 0u);
		}
	}
	std::memcpy(reinterpret_cast<std::byte*>(&dst) + tail, base + tail,
	            sizeof(ShaderVertexInputInfo) - tail);
	return true;
}

static void CopyPreparedVertexInfo(ShaderVertexInputInfo& dst, const ShaderVertexInputInfo& src) {
	if (!RenderStateFastEnabled(RenderStatePart::VertexCopy)) {
		dst = src;
		return;
	}
	if (CopyVertexInputPrefixes(dst, src)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateVertexPartialCopies);
	} else {
		dst = src;
		Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateVertexFullCopies);
	}
	if (RenderStateVerifyMode() != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateVerifyChecks);
		if (std::memcmp(&dst, &src, sizeof(dst)) != 0) {
			ReportRenderStateMismatch("a partial vertex input copy");
			dst = src;
		}
	}
}

// KYTY_RENDER_STATE_FAST swap: exchanges two stage preparations member by member. The bindings name
// every member in declaration order, so a new member stops this compiling until it is swapped.
static void SwapStagePrepMembers(PipelineCache::StagePrep& a, PipelineCache::StagePrep& b) noexcept {
	auto& [a_resources, a_specialization, a_permutation] = a;
	auto& [b_resources, b_specialization, b_permutation] = b;
	auto& [a_buffers, a_images, a_samplers, a_srt, a_user_data, a_fill] = a_resources;
	auto& [b_buffers, b_images, b_samplers, b_srt, b_user_data, b_fill] = b_resources;
	auto& [a_spec_buffers, a_spec_images] = a_specialization;
	auto& [b_spec_buffers, b_spec_images] = b_specialization;
	a_buffers.swap(b_buffers);
	a_images.swap(b_images);
	a_samplers.swap(b_samplers);
	a_srt.swap(b_srt);
	a_user_data.swap(b_user_data);
	std::swap(a_fill, b_fill);
	a_spec_buffers.swap(b_spec_buffers);
	a_spec_images.swap(b_spec_images);
	std::swap(a_permutation, b_permutation);
}

static bool SameStagePrep(const PipelineCache::StagePrep& a, const PipelineCache::StagePrep& b) {
	const auto& x = a.resources;
	const auto& y = b.resources;
	return x.buffers == y.buffers && x.images == y.images && x.samplers == y.samplers &&
	       x.flattened_srt == y.flattened_srt && x.user_data == y.user_data &&
	       x.uniform_fill == y.uniform_fill && a.specialization == b.specialization &&
	       a.permutation == b.permutation;
}

static void SwapPreparedStage(PipelineCache::StagePrep& state, PipelineCache::StagePrep& prepared) {
	if (!RenderStateFastEnabled(RenderStatePart::PrepSwap)) {
		std::swap(state, prepared);
		return;
	}
	if (RenderStateVerifyMode() == 0) {
		SwapStagePrepMembers(state, prepared);
		Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateMemberSwaps);
		return;
	}
	const auto state_before    = state;
	const auto prepared_before = prepared;
	SwapStagePrepMembers(state, prepared);
	Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateMemberSwaps);
	Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateVerifyChecks);
	if (!SameStagePrep(state, prepared_before) || !SameStagePrep(prepared, state_before)) {
		ReportRenderStateMismatch("a member-wise stage preparation swap");
		state    = prepared_before;
		prepared = state_before;
	}
}

// Draw-prep: installs a validated preparation as if GetGraphicsPrograms had produced it. The
// stage preps are swapped (the preparation keeps the old vectors' capacity) and the stage
// runtimes re-pointed at the draw state's copies. Like GetGraphicsPrograms, a draw without an
// active pixel shader leaves the pixel prep untouched.
static void ApplyPreparedDraw(DrawPrep::PreparedDraw& prepared, DrawRenderState& state) {
	state.programs = prepared.programs;
	CopyPreparedVertexInfo(state.vertex_info[0], prepared.vertex_info);
	state.ps_input_info = prepared.pixel_info;
	SwapPreparedStage(state.stage_preps.vertex[0], prepared.vertex_prep);
	state.vertex_info[0].stage.resources = &state.stage_preps.vertex[0].resources;
	if (prepared.pixel_active) {
		SwapPreparedStage(state.stage_preps.pixel, prepared.pixel_prep);
		state.ps_input_info.stage.resources = &state.stage_preps.pixel.resources;
	}
}

static void RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw,
                           DrawRenderState& state) {
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	const auto& vertex_shader_info = sh_ctx.GetVs();
	const auto& pixel_shader_info  = sh_ctx.GetPs();
	const auto& shader_regs        = ctx.GetShaderRegisters();

	// KYTY_RENDER_STATE_FAST reset: a validated preparation assigns both in full
	// (ApplyPreparedDraw), and the serial path below assigns the programs; it only needs the pixel
	// interface value-initialised (it leaves it alone without an active pixel shader).
	const bool reset_fast = RenderStateFastEnabled(RenderStatePart::Reset);
	if (!reset_fast) {
		state.programs      = {};
		state.ps_input_info = {};
	}
	// Draw-prep: the draw's speculative preparation, used below when its certificate holds now.
	auto&       executor = buffer.GetContext().GetRenderExecutor();
	auto* const prepared = executor.TakePreparedDraw();
	// KYTY_DRAW_PREP_BINDINGS statics: the preparation computed the export mapping from these same
	// registers (the draw's snapshot, bound for its commit).
	const auto* plan            = executor.PendingBindingPlan();
	const bool  planned_mapping = prepared != nullptr && plan != nullptr && plan->statics_valid;
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
	    target_export_mapping {};
	if (planned_mapping && DrawPrep::BindingsVerifyMode() == 0) {
		std::copy(prepared->target_export_mapping.begin(), prepared->target_export_mapping.end(),
		          target_export_mapping.begin());
	} else {
		const auto output_filter = DrawColorOutputFilter(ctx);
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = ctx.GetRenderTarget(slot);
			if (render_target_bound(rt) && render_target_mask_slot(ctx.GetRenderTargetMask(), slot) != 0 &&
			    (output_filter & (1u << slot)) != 0) {
				target_export_mapping[slot] =
				    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
				                                 rt.info.channel_order)
				        .export_mapping;
			}
		}
		if (planned_mapping) {
			DrawPrep::CountBindingVerifyCheck();
			for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
				if (target_export_mapping[slot].packed != prepared->target_export_mapping[slot].packed) {
					DrawPrep::ReportBindingMismatch("target export mapping", slot);
					break;
				}
			}
		}
	}
	auto& pipeline_cache = buffer.GetContext().GetPipelineCache();
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "GetGraphicsPrograms");
	}
	state.vertex_stages_written = PipelineCache::TessellationActive(buffer.GetUserConfig()) ? 3u : 1u;
	// Draw-prep: use the draw's speculative preparation when its certificate holds now.
	CommitStats::StepMark(CommitStats::Step::TakePrepared);
	const bool prepared_valid =
	    prepared != nullptr && DrawPrep::Validate(*prepared, state.ps_active, target_export_mapping);
	CommitStats::StepMark(CommitStats::Step::Validate);
	if (!prepared_valid) {
		// Astro Bot PC (KYTY_CP_PAR_FASTCHECK): the serial program preparation below.
		FastCheck::Fail(prepared == nullptr ? FastCheck::Cause::PrepMissing : FastCheck::Cause::PrepRefused);
	}
	if (prepared_valid) {
		ApplyPreparedDraw(*prepared, state);
		// The binding plan was computed from this preparation (KYTY_DRAW_PREP_BINDINGS).
		executor.ActivateBindingPlan();
		executor.NotePreparedValidated();
		CommitStats::StepMark(CommitStats::Step::Apply);
		if (DrawPrep::VerifyMode() != 0) {
			// The serial preparation on copies, from the same (snapshot) registers.
			auto vertex_copy = std::make_unique<std::array<ShaderVertexInputInfo, 3>>();
			auto pixel_copy  = std::make_unique<ShaderPixelInputInfo>();
			auto prep_copy   = std::make_unique<PipelineCache::GraphicsStagePreps>();
			const auto serial = pipeline_cache.GetGraphicsPrograms(
			    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
			    target_export_mapping, state.ps_active, *vertex_copy, *pixel_copy, *prep_copy);
			(void)DrawPrep::VerifyCommitted(state.ps_active, state.programs, state.vertex_info[0],
			                                state.ps_input_info, state.stage_preps, serial,
			                                (*vertex_copy)[0], *pixel_copy, *prep_copy);
		}
		return;
	}
	if (reset_fast) {
		state.ps_input_info = {};
	}
	state.programs = pipeline_cache.GetGraphicsPrograms(
	    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
	    target_export_mapping, state.ps_active, state.vertex_info, state.ps_input_info,
	    state.stage_preps);
}

// Astro Bot PC - KYTY_MRT_SPLIT_REUSE (read once at start-up; default 0 = off; 1 on; verify). The
// second pass of an MRT split draw (KYTY_MRT_SPLIT, DrawIndex/DrawAuto) is the same guest draw: the
// same registers, programs and resource words, nothing else ran on the command processor in between.
// Its RefreshShaders found no preparation left (the first pass took it) and prepared both programs
// again on the serial path (GetGraphicsPrograms: program keys, SRT materialization, permutations),
// a function of those registers and of guest memory the first pass already read for this draw. With
// the switch the second pass keeps the first pass's programs, stage runtimes and resource snapshots
// (still in the draw state: the first pass's execution only reads them) when the first pass prepared
// a single vertex stage and the pixel shader's activity is the same; everything else of the second
// pass (targets of its slots, bindings, uploads, pipeline, acquisition, descriptors, the draw) is
// unchanged. Its preparation stays "not validated" (no run record), as before. verify: the serial
// preparation runs anyway and its programs, permutations and resource words must be the kept ones
// (differences counted and the first 16 logged; the serial result is used).
static int MrtSplitReuseMode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_MRT_SPLIT_REUSE");
		if (value == nullptr) {
			return 0;
		}
		if (std::strcmp(value, "1") == 0) {
			return 1;
		}
		return std::strcmp(value, "verify") == 0 ? 2 : 0;
	}();
	return mode;
}

static std::atomic<uint64_t> g_mrt_reuse_kept {0};       // second passes that kept the programs
static std::atomic<uint64_t> g_mrt_reuse_checks {0};     // verify: comparisons
static std::atomic<uint64_t> g_mrt_reuse_mismatches {0}; // verify: differences

// KYTY_MRT_SPLIT_REUSE=verify: the kept preparation against the serial one just computed.
struct MrtReuseSnapshot {
	const void*                                       vertex_program = nullptr;
	const void*                                       pixel_program  = nullptr;
	const void*                                       vertex_perm    = nullptr;
	const void*                                       pixel_perm     = nullptr;
	std::array<ShaderRecompiler::IR::ResourceSnapshot, 2> resources;
};

static void MrtReuseSave(const DrawRenderState& state, MrtReuseSnapshot& saved) {
	saved.vertex_program = state.vertex_info[0].stage.program;
	saved.pixel_program  = state.ps_active ? state.ps_input_info.stage.program : nullptr;
	saved.vertex_perm    = state.stage_preps.vertex[0].permutation;
	saved.pixel_perm     = state.ps_active ? state.stage_preps.pixel.permutation : nullptr;
	if (state.vertex_info[0].stage.resources != nullptr) {
		saved.resources[0] = *state.vertex_info[0].stage.resources;
	}
	if (state.ps_active && state.ps_input_info.stage.resources != nullptr) {
		saved.resources[1] = *state.ps_input_info.stage.resources;
	}
}

static void MrtReuseCompare(const DrawRenderState& state, const MrtReuseSnapshot& saved) {
	g_mrt_reuse_checks.fetch_add(1, std::memory_order_relaxed);
	const auto same_snapshot = [](const ShaderRecompiler::IR::ResourceSnapshot* a,
	                              const ShaderRecompiler::IR::ResourceSnapshot& b) {
		if (a == nullptr) {
			return b.buffers.empty() && b.images.empty() && b.samplers.empty() &&
			       b.flattened_srt.empty() && b.user_data.empty();
		}
		return std::ranges::equal(a->buffers, b.buffers) && std::ranges::equal(a->images, b.images) &&
		       std::ranges::equal(a->samplers, b.samplers) && a->flattened_srt == b.flattened_srt &&
		       a->user_data == b.user_data;
	};
	const char* what = nullptr;
	if (state.vertex_info[0].stage.program != saved.vertex_program ||
	    (state.ps_active ? state.ps_input_info.stage.program : nullptr) != saved.pixel_program) {
		what = "program";
	} else if (state.stage_preps.vertex[0].permutation != saved.vertex_perm ||
	           (state.ps_active ? state.stage_preps.pixel.permutation : nullptr) != saved.pixel_perm) {
		what = "permutation";
	} else if (state.vertex_stages_written != 1u ||
	           !same_snapshot(state.vertex_info[0].stage.resources, saved.resources[0]) ||
	           (state.ps_active && !same_snapshot(state.ps_input_info.stage.resources, saved.resources[1]))) {
		what = "resource words";
	}
	if (what != nullptr) {
		const auto n = g_mrt_reuse_mismatches.fetch_add(1, std::memory_order_relaxed);
		if (n < 16) {
			std::printf("MrtSplitReuseVerify: the second pass's serial preparation differs from the kept "
			            "one (%s; %" PRIu64 " so far)\n",
			            what, n + 1);
			std::fflush(stdout);
		}
	}
}

// KYTY_MRT_SPLIT_REUSE: whether the first pass left a preparation the second pass may keep (a single
// vertex stage with its runtime, and the pixel stage's when it was active).
static bool MrtSplitReuseCandidate(const DrawRenderState& state) {
	return MrtSplitReuseMode() != 0 && state.vertex_stages_written == 1u && state.programs.vertex[0] &&
	       !state.programs.vertex[1] && static_cast<bool>(state.vertex_info[0].stage) &&
	       (!state.ps_active || (state.programs.pixel && static_cast<bool>(state.ps_input_info.stage)));
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	state.ps_active = DrawHasActivePixelShader(buffer);
	CommitStats::StepMark(CommitStats::Step::PsActive);
	// KYTY_MRT_SPLIT_REUSE: an MRT split's second pass with the first pass's preparation in `state`.
	const bool reuse_programs = m_mrt_reuse_programs && state.ps_active == m_mrt_reuse_ps;
	if (reuse_programs && MrtSplitReuseMode() == 1) {
		g_mrt_reuse_kept.fetch_add(1, std::memory_order_relaxed);
	} else if (reuse_programs) {
		auto saved = std::make_unique<MrtReuseSnapshot>();
		MrtReuseSave(state, *saved);
		{
			KYTY_PROFILER_DETAIL_BLOCK("Draw::RefreshShaders");
			RefreshShaders(buffer, draw, state);
		}
		MrtReuseCompare(state, *saved);
	} else {
		KYTY_PROFILER_DETAIL_BLOCK("Draw::RefreshShaders");
		RefreshShaders(buffer, draw, state);
	}
	CommitStats::Mark(CommitStats::Phase::Programs);
	// A stage whose shader the recompiler skipped (an unresolvable runtime descriptor,
	// KYTY_SRT_VARIANT_READS) leaves its program empty: drop the draw. A tessellated draw needs
	// all three vertex stages (one without a runtime made PrepareBindings exit: Wolverine
	// HS 0x3ff475220c3b4e7b + TES 0xb39fc591165dbc5f).
	const bool tess_missing =
	    state.vertex_stages_written == 3u &&
	    (!state.programs.vertex[1] || !state.programs.vertex[2] || !state.vertex_info[0].stage ||
	     !state.vertex_info[1].stage || !state.vertex_info[2].stage);
	if (!state.programs.vertex[0] || (state.ps_active && !state.programs.pixel) || tess_missing) {
		// Counted and reported (the first few, then every 2048th) so a dropped draw is never silent.
		static std::atomic_uint64_t dropped {0};
		const auto                  count = dropped.fetch_add(1, std::memory_order_relaxed) + 1u;
		if (count <= 4u || (count & 2047u) == 0u) {
			std::printf("Warning: draw #%" PRIu64 " dropped, %s program missing (%s)\n", count,
			            !state.programs.vertex[0] ? "vertex"
			            : tess_missing           ? "tessellation"
			                                     : "pixel",
			            draw.Name());
		}
		return false;
	}
	// KYTY_DRAW_RUN (drawPrep/drawRun.h): a continuation keeps the previous draw's targets (the
	// entries are still in `state`: the fast reset leaves them, DrawRunCandidate requires it).
	m_run_slice_offset = render_target_slice_offset;
	const bool run_candidate =
	    DrawRun::Enabled() &&
	    (DrawRunCandidate(buffer, state, render_target_slice_offset) ||
	     (DrawRun::RecallEnabled() &&
	      DrawRunRecallCandidate(buffer, state, render_target_slice_offset)));
	CommitStats::StepMark(CommitStats::Step::RunCandidate);
	if (run_candidate) {
		if (DrawRun::GetMode() == DrawRun::Mode::On) {
			state.color_count         = m_run.color_count;
			state.color_slots_written = m_run.color_slots;
			m_run_active              = true;
			DrawRunMarkBindings();
			CommitStats::StepMark(CommitStats::Step::MarkBindings);
			CommitStats::Mark(CommitStats::Phase::Targets);
			return true;
		}
		m_run_verify = true;
	}
	KYTY_PROFILER_DETAIL_BLOCK("Draw::ResolveTargets");
	// KYTY_DRAW_RUN_TARGETS (drawRun.h): the record's targets, when their inputs are the record's.
	const auto targets_mode = DrawRun::TargetsKeepMode();
	const bool keep_targets = targets_mode != 0 && !m_run_verify &&
	                          DrawRunTargetsCandidate(buffer, state, render_target_slice_offset);
	if (keep_targets) {
		OwnedCounters::Add(DrawRun::GetTotals().targets_kept, 1);
	}
	if (keep_targets && targets_mode == 1) {
		const CommitStats::SubScope color_scope(CommitStats::Sub::ColorTargets);
		state.color_count         = m_run.color_count;
		state.color_slots_written = m_run.color_slots;
		// As the resolution does: only a resolved target is marked (a draw without a depth target,
		// or a colour entry without an image, has an invalid id).
		for (uint32_t i = 0; i < m_run.color_count; i++) {
			state.color_info[i] = m_run.colors[i];
			if (state.color_info[i].image_id) {
				BindRenderTarget(state.color_info[i].image_id);
			}
		}
		state.depth_info = m_run.depth[0];
		if (state.depth_info.image_id) {
			BindRenderTarget(state.depth_info.image_id);
		}
		m_run_targets_kept = true;
	} else {
		DrawRunTargets(buffer, draw, render_target_slice_offset, state);
		if (keep_targets) {
			// Verify: the resolution must have given the record's targets.
			DrawRun::CountVerifyCheck();
			bool same = state.color_count == m_run.color_count &&
			            state.color_slots_written == m_run.color_slots &&
			            SameRenderDepthInfo(state.depth_info, m_run.depth[0]);
			for (uint32_t i = 0; same && i < state.color_count; i++) {
				same = SameRenderColorInfo(state.color_info[i], m_run.colors[i]);
			}
			if (!same) {
				DrawRun::ReportMismatch("kept targets");
				if (targets_mode == 3) {
					EXIT("DrawRunVerify: kept targets differ from the resolved ones\n");
				}
			}
		}
	}

	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		return false;
	}

	CommitStats::StepMark(CommitStats::Step::MarkBindings);
	CommitStats::Mark(CommitStats::Phase::Targets);
	return true;
}

// KYTY_MRT_SPLIT (default on; YOTEI 44ea334d4): slots of mask that write the same surface as a
// lower slot of it, each with channels of its own. Astro Bot's foliage-interaction stamps write RG
// (blended) through MRT0 and BA through MRT1 of one 1024x1024 surface; one image bound as two
// Vulkan colour attachments is undefined, and NVIDIA lost the stamps (grass ignored the hover laser
// or bent ahead of Astro). Such a draw (DrawIndex, DrawAuto) runs twice: once without these slots,
// once with only them, each pass keeping its slots' masks and blends. KYTY_MRT_SPLIT=0 binds them
// together as before.
static Live::Switch g_mrt_split("KYTY_MRT_SPLIT", Live::ParseDefaultOn); // live

static bool MrtSplitEnabled() {
	return g_mrt_split.Get() != 0;
}

static uint32_t AliasedColorSlots(const HW::Context& ctx, uint32_t mask) {
	uint32_t aliased = 0;
	for (uint32_t later = 1; later < RENDER_COLOR_ATTACHMENTS_MAX; later++) {
		if ((mask & (1u << later)) == 0) {
			continue;
		}
		const auto later_addr = ctx.GetRenderTarget(later).base.addr;
		const auto later_mask = render_target_mask_slot(ctx.GetRenderTargetMask(), later);
		if (later_addr == 0 || later_mask == 0) {
			continue;
		}
		for (uint32_t first = 0; first < later; first++) {
			if ((mask & (1u << first)) != 0 && (aliased & (1u << first)) == 0 &&
			    ctx.GetRenderTarget(first).base.addr == later_addr &&
			    (render_target_mask_slot(ctx.GetRenderTargetMask(), first) & later_mask) == 0) {
				aliased |= 1u << later;
				break;
			}
		}
	}
	return aliased;
}

// KYTY_MRT_SPLIT accounting: the first splits are logged, then a count every 4096.
static void NoteMrtSplit(uint32_t slots) {
	static std::atomic<uint64_t> count {0};
	const auto                   n = count.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (n <= 4u || (n & 4095u) == 0u) {
		std::printf("MRT split: draw #%llu runs a second pass for colour slots 0x%x (one surface bound "
		            "to several slots)\n",
		            static_cast<unsigned long long>(n), slots);
		if (MrtSplitReuseMode() != 0) {
			// Astro Bot PC (KYTY_MRT_SPLIT_REUSE): counts so far (before this draw's second pass).
			std::printf("MRT split reuse: %llu second passes kept the programs, verify %llu checks, "
			            "%llu mismatches\n",
			            static_cast<unsigned long long>(g_mrt_reuse_kept.load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(g_mrt_reuse_checks.load(std::memory_order_relaxed)),
			            static_cast<unsigned long long>(
			                g_mrt_reuse_mismatches.load(std::memory_order_relaxed)));
		}
	}
}
// The draw's colour and depth target resolution (PrepareDrawRenderState; a KYTY_DRAW_RUN
// continuation whose images changed resolves them late, from the same registers).
void RenderExecutor::DrawRunTargets(CommandBuffer& buffer, const DrawCallInfo& draw,
                                    uint32_t render_target_slice_offset, DrawRenderState& state) {
	state.color_count         = 0;
	state.color_slots_written = 0;
	uint32_t mrt_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt) {
				mrt_mask |= 1u << output.location; // the colour target (KYTY_CB_SHADER_MASK_EXPORTS)
			}
		}
	}
	if (SkipInactivePixelShadersEnabled()) {
		mrt_mask &= DrawColorOutputFilter(buffer.GetRegisters());
	}
	// KYTY_MRT_SPLIT: the first pass of a draw that binds one surface to several slots leaves the
	// later ones for a second pass (m_mrt_split_slots); the second pass binds only those.
	mrt_mask &= m_mrt_filter;
	if (m_mrt_split_allowed && m_mrt_filter == ~0u && (mrt_mask & (mrt_mask - 1u)) != 0 &&
	    MrtSplitEnabled()) {
		m_mrt_split_slots = AliasedColorSlots(buffer.GetRegisters(), mrt_mask);
		mrt_mask &= ~m_mrt_split_slots;
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderColorTarget");
	}
	// KYTY_RENDER_STATE_VERIFY with the reset part: each entry resolved over the previous draw's is
	// compared with the same resolution into a value-initialised entry (run after it, so the
	// draw's own resolution is the one without verification), which then provides the result.
	const bool verify_entries = RenderStateVerifyMode() != 0 &&
	                            RenderStateFastEnabled(RenderStatePart::Reset) &&
	                            !graphics_debug_dump_enabled();
	CommitStats::SubScope color_scope(CommitStats::Sub::ColorTargets);
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((mrt_mask & (1u << slot)) != 0) {
			state.color_slots_written = std::max(state.color_slots_written, state.color_count + 1u);
			auto& entry = state.color_info[state.color_count];
			ResolveRenderColorTarget(buffer, entry, render_target_slice_offset, slot);
			if (verify_entries) {
				RenderColorInfo reference {};
				ResolveRenderColorTarget(buffer, reference, render_target_slice_offset, slot);
				Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateVerifyChecks);
				if (!SameRenderColorInfo(entry, reference)) {
					ReportRenderStateMismatch("a colour target entry resolved over the last draw's");
					entry = reference;
				}
			}
			if (entry.image_id) {
				state.color_count++;
			}
		}
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderDepthTarget");
	}
	color_scope.Stop();
	const CommitStats::SubScope depth_scope(CommitStats::Sub::DepthTarget);
	ResolveRenderDepthTarget(buffer, state.depth_info);
	if (verify_entries) {
		RenderDepthInfo reference {};
		ResolveRenderDepthTarget(buffer, reference);
		Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateVerifyChecks);
		if (!SameRenderDepthInfo(state.depth_info, reference)) {
			ReportRenderStateMismatch("the depth target entry resolved over the last draw's");
			state.depth_info = reference;
		}
	}
	// Upstream 7b7294aa0 (TODO PPSA09243): Vulkan passes take one sample count. Where PS5
	// attachments mix counts, keep those matching the raster count instead of resolving
	// multisampled shadows. Draws with one sample count (all Astro Bot draws seen) skip this.
	if (state.color_count != 0 || state.depth_info.image_id) {
		const auto colors       = std::span {state.color_info, state.color_count};
		const auto pass_samples = state.depth_info.image_id ? state.depth_info.desc.info.samples
		                                                    : state.color_info[0].desc.info.samples;
		if (std::ranges::any_of(colors, [&](const RenderColorInfo& color) {
			    return color.desc.info.samples != pass_samples;
		    })) {
			const auto raster =
			    render_sample_count(buffer.GetRegisters().GetAaConfig().msaa_num_samples);
			const auto dropped = std::ranges::remove_if(colors, [&](const RenderColorInfo& color) {
				return color.desc.info.samples != raster;
			});
			state.color_count -= static_cast<uint32_t>(dropped.size());
			if (state.depth_info.image_id && state.depth_info.desc.info.samples != raster) {
				state.depth_info = {};
			}
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(
				    "Warning: dropping attachments whose sample count differs from the raster.\n");
			}
		}
	}
}

// ------------------------------------------------------------------------------------------------
// KYTY_DRAW_RUN (drawPrep/drawRun.h): run-level commit of structure-sharing draws.

// A program whose draws can form a run: it writes no guest memory and no image (buffer or image
// writes and atomics, address writes, GDS, the fault buffer), so a draw of it changes nothing a
// later draw's structure depends on and needs no barrier against the next one. Its mip-statistics
// counters (an emulator buffer, unordered atomics) are allowed.
// buffer_writes (KYTY_DRAW_RUN_WRITES): V# buffer writes and atomics are allowed as well.
static bool DrawRunProgramEligible(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                   bool buffer_writes = false) {
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	if (program.has_address_writes) {
		return false;
	}
	for (const auto& resource: program.info.buffers) {
		if (!buffer_writes && (resource.written || resource.atomic)) {
			return false;
		}
	}
	for (const auto& resource: program.info.images) {
		if (resource.written || resource.atomic) {
			return false;
		}
	}
	for (const auto& binding: program.bindings.descriptors) {
		if (binding.kind == Kind::Gds || binding.kind == Kind::FaultBuffer) {
			return false;
		}
	}
	return true;
}

// KYTY_PIPELINE_SKIP: whether a draw may be skipped while the driver builds its new pipeline
// because the picture is all it changes: no stage writes guest memory or images (DrawRunProgramEligible),
// no guest occlusion query counts its samples, it clears no depth or stencil, it is no mesh draw,
// and it renders to tiled images only (the CPU reads linear ones back).
// KYTY_PIPELINE_SKIP_RECURRENT (default on): a skipped draw is lost for good when its target is
// not drawn again, e.g. a sky or probe texture rendered once at level start (Sky Garden's sky
// stayed black after a shader-cache invalidation). Only draws whose every target was already a
// render target in at least two earlier presented frames may be skipped; the others wait.
static bool SkipRecurrentTargetsOnly() {
	static const bool on = [] {
		const auto* value = std::getenv("KYTY_PIPELINE_SKIP_RECURRENT");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return on;
}

// Notes the draw's render targets for the rule above (once per image and frame).
static void NoteSkipTargets(RenderContext& context, const DrawRenderState& state) {
	auto&      cache = context.GetTextureCache();
	const auto frame = cache.Frame() + 1u; // 0 = never
	const auto note  = [&](ImageId id) {
		if (!id) {
			return;
		}
		auto* image = cache.TryGetImage(id);
		if (image != nullptr && image->skip_target_frame != frame) {
			image->skip_target_frame  = frame;
			image->skip_target_frames = std::min<uint32_t>(image->skip_target_frames + 1u, 1000u);
		}
	};
	for (uint32_t i = 0; i < state.color_count; i++) {
		note(state.color_info[i].image_id);
	}
	note(state.depth_info.image_id);
}

static bool TargetsRecurrent(RenderContext& context, const DrawRenderState& state) {
	auto&      cache     = context.GetTextureCache();
	const auto frame     = cache.Frame() + 1u;
	const auto recurrent = [&](ImageId id) {
		if (!id) {
			return true;
		}
		const auto* image = cache.TryGetImage(id);
		if (image == nullptr) {
			return false;
		}
		// Earlier frames only: the current frame was noted before this check.
		const auto earlier = image->skip_target_frames - (image->skip_target_frame == frame ? 1u : 0u);
		return earlier >= 2u;
	};
	for (uint32_t i = 0; i < state.color_count; i++) {
		if (!recurrent(state.color_info[i].image_id)) {
			return false;
		}
	}
	return recurrent(state.depth_info.image_id);
}

static bool DrawOnlyChangesPicture(RenderContext& context, const DrawRenderState& state,
                                   std::span<const ShaderVertexInputInfo> vertex_stages, bool mesh_active) {
	if (mesh_active || vertex_stages.size() != 1 || context.GetOcclusionCounter().Active()) {
		return false;
	}
	if (SkipRecurrentTargetsOnly() && !TargetsRecurrent(context, state)) {
		return false;
	}
	for (const auto& stage: vertex_stages) {
		if (stage.stage.program == nullptr || !DrawRunProgramEligible(*stage.stage.program)) {
			return false;
		}
	}
	if (state.ps_active && (state.ps_input_info.stage.program == nullptr ||
	                        !DrawRunProgramEligible(*state.ps_input_info.stage.program))) {
		return false;
	}
	for (uint32_t i = 0; i < state.color_count; i++) {
		if (!state.color_info[i].desc.info.IsTiled()) {
			return false;
		}
	}
	const auto& depth = state.depth_info;
	if (static_cast<bool>(depth.image_id) &&
	    (!depth.desc.info.IsTiled() || depth.depth_clear_enable || depth.load_clear_aspects ||
	     depth.stencil_clear_enable)) {
		return false;
	}
	return true;
}

void RenderExecutor::BeginDrawRun() {
	m_run_prev_valid     = m_run.valid;
	m_run_keepable       = m_run.valid;
	m_run.valid          = false;
	m_run_active         = false;
	m_run_recall         = false;
	m_run_verify         = false;
	m_run_targets_kept   = false;
	m_run_pipeswap       = false;
	m_prepared_validated = false;
	if (!m_in_engine_commit && DrawRun::Enabled()) {
		DrawRun::NoteForeignActivity("draw outside an engine commit");
	}
	if (DrawRun::KeyDiagEnabled()) [[unlikely]] {
		m_key_diag.prev_reason = m_key_diag.reason;
		m_key_diag.reason      = m_in_engine_commit ? DrawRun::KeyDiag::Inelig::NotReached
		                                            : DrawRun::KeyDiag::Inelig::NotEngine;
		m_key_diag.combo       = -2;
	}
}

// ---- Astro Bot PC: KYTY_NATIVE_MODE=n3b, lean native continuations (loader/gameCapture.h "n3b") ----
//
// A native draw op committed from its own published slot (CommandProcessor::NativeLeanScope sets
// m_native_op) may take the lean path when it continues the run record that the previous native
// draw of its engine thread slot made or continued (m_native_record; the same segment: not a
// segment start), with KYTY_DRAW_RUN=1. The lean path skips:
//  - the target operations (metadata colour operation, depth/stencil copy, colour resolve, depth to
//    colour copy): MayRunTargetOperation is false, under which each of them returns false at once;
//  - the hardware register checks (RunDrawChecks: validation only, same registers as the record;
//    run late when the draw turns out not to be a lean continuation);
//  - the barrier-safety evaluation (DrawIsBarrierSafe): the record's value, its inputs unchanged;
//  - the run-record eligibility (the record's, see ExecutePreparedDraw).
// Everything per draw runs as on the normal path, in the same order, so the commands recorded are
// the normal path's: certificate check and preparation, the run certificate (DrawRunCandidate)
// and its late image check, shader data, buffer bindings (BufferCache::ObtainBuffer), vertex and
// index buffers, pipeline selection, descriptors and push constants, rendering begin, pipeline
// bind, the draw, the run record update.
// KYTY_NATIVE_VERIFY=1: one would-be lean draw in 64 takes the normal path instead and every lean
// prediction is compared with what it computed (no target operation, the barrier safety, an
// eligible record without writes); a difference is logged and stops the emulator.
bool RenderExecutor::NativeLeanPrecheck(const CommandBuffer& buffer) {
	namespace GC   = Loader::GameCapture;
	const auto op  = m_native_op;
	const auto slot = static_cast<int32_t>((op >> GC::kNativeOpSlotShift) & GC::kNativeSlotMask);
	auto       why = GC::kNlLean;
	if ((op & GC::kNativeOpSegStart) != 0) {
		why = GC::kNlSegStart;
	} else if (!m_run_prev_valid || m_native_record.slot != slot || m_native_record.key != m_run.key ||
	           m_native_record.activity != m_run.activity ||
	           m_native_record.rendering_serial != m_run.rendering_serial ||
	           m_native_record.push_epoch != m_run.push_epoch) {
		why = GC::kNlChain;
	} else if (!m_in_engine_commit || DrawRun::GetMode() != DrawRun::Mode::On) {
		why = GC::kNlOff;
	} else if (graphics_debug_dump_enabled() || m_mrt_filter != ~0u ||
	           MayRunTargetOperation(buffer.GetRegisters())) {
		why = GC::kNlRegs;
	} else if (GC::NativeVerifyEnabled() && (++m_native_samples & 63u) == 0) {
		m_native_lean_sample = true;
		m_native_lean_reason = GC::kNlSample;
		return false;
	}
	m_native_lean_reason = why;
	m_native_lean_pre    = why == GC::kNlLean;
	return m_native_lean_pre;
}

void RenderExecutor::NativeLeanDecide() {
	namespace GC = Loader::GameCapture;
	if (!m_native_lean_pre && !m_native_lean_sample) {
		return;
	}
	const auto why = !m_run_active   ? GC::kNlNotCont
	                 : m_run_recall   ? GC::kNlRecall
	                 : m_run_pipeswap ? GC::kNlSwap
	                 : m_run.writes   ? GC::kNlWrites
	                                  : GC::kNlLean;
	if (why != GC::kNlLean) {
		NativeLeanDrop(why);
		return;
	}
	m_native_lean = m_native_lean_pre;
}

void RenderExecutor::NativeLeanDrop(uint32_t reason) {
	if (m_native_lean || m_native_lean_sample) {
		m_native_lean_reason = reason;
	}
	m_native_lean        = false;
	m_native_lean_sample = false;
}

void RenderExecutor::NativeLeanMismatch(const char* what, uint64_t detail) {
	m_native_lean_mismatch = true;
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16u) {
		std::printf("NativeLeanVerify: mismatch '%s' (detail 0x%" PRIx64 ", native op 0x%x)\n", what, detail,
		            m_native_op);
		std::fflush(stdout);
	}
	EXIT("NativeLeanVerify: a lean prediction differs from the normal path (%s)\n", what);
}

bool RenderExecutor::DrawRunCommandUnchanged(const CommandBuffer& buffer) const {
	return buffer.Identity() == m_run.command &&
	       m_context.GetCommandScheduler().CurrentTick() == m_run.tick &&
	       buffer.ActiveRenderingSerial() != 0 &&
	       buffer.ActiveRenderingSerial() == m_run.rendering_serial;
}

RenderExecutor::DrawRunImage RenderExecutor::MakeDrawRunImage(ImageId id, bool texture) const {
	DrawRunImage mark;
	mark.id           = id;
	mark.texture      = texture;
	const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
	if (image == nullptr) {
		return mark;
	}
	image->CpParTouch(); // Astro Bot PC (KYTY_CP_PAR=1b3)
	mark.image          = image->backing.image;
	mark.stage          = image->backing.state.pl_stage;
	mark.access         = image->backing.state.access_mask;
	mark.layout         = image->backing.state.layout;
	mark.serial         = image->ContentSerial();
	mark.address        = image->info.data.address;
	mark.size           = image->info.data.size;
	mark.resident_first = image->resident_first;
	mark.registered     = image->registered;
	mark.single_state   = image->backing.subresource_states.empty();
	return mark;
}

// Whether a guest range lies over one of the record's attachments (either may be a null image).
bool RenderExecutor::DrawRunOverAttachment(std::span<const DrawRunImage> marks, uint64_t address,
                                           uint64_t size) {
	if (size == 0) {
		return false;
	}
	for (const auto& mark: marks) {
		if (!mark.texture && mark.size != 0 && address < mark.address + mark.size &&
		    mark.address < address + size) {
			return true;
		}
	}
	return false;
}

// A sampled image whose view the normal path would take as it is: FindTexture's rediscovery checks
// and a no-op RefreshImage (TextureBindingMemo::ViewImageReady and RefreshIsNoOp; any new refresh
// trigger added there must be added here). Null images are never refreshed.
static bool DrawRunTextureReady(const Image& image) {
	if (image.info.data.Empty()) {
		return true;
	}
	if (image.depth_id || image.info.HasStencil() || !image.IsTracked()) {
		return false;
	}
	if (image.ChunkTracked()) {
		return image.chunks.untracked_count == 0;
	}
	return image.track_addr == image.live.address && image.track_addr_end == image.live.End();
}

// Whether every image the recorded structure refers to still has the identity and state the
// previous draw left it in, with nothing the normal path's resolution would refresh (CPU-dirty or
// buffer-modified contents, a released tracking range, a rebind request) (GPU thread; images change
// only on it or, for CPU dirtiness, under the texture-cache lock on a faulting thread).
// 0 when unchanged; otherwise which mark differs and how (verify-mode detail):
// (mark index + 1) << 8 | 0x80 for a texture | the field (1 slot freed, 2 native image,
// 3 registration, 4 rebind request, 5 residency, 6 layout, 7 access, 8 stage, 9 per-subresource
// states, 10 content serial, 11 unregistered, 12 CPU-dirty, 13 buffer-modified, 14 texture refresh).
uint32_t RenderExecutor::DrawRunImagesChange(bool compare_serials, bool attachments_only,
                                           bool log_change, bool attachment_serials) const {
	auto&            cache = m_context.GetTextureCache();
	std::scoped_lock lock {cache.m_lock};
	const auto&      images = cache.m_slot_images;
	for (uint32_t index = 0; index < m_run.images.size(); index++) {
		const auto& mark = m_run.images[index];
		if (attachments_only && mark.texture) {
			continue;
		}
		const auto* image = images.try_get(mark.id);
		if (image != nullptr) {
			image->CpParTouch(); // Astro Bot PC (KYTY_CP_PAR=1b3)
		}
		uint32_t    field = 0;
		if (image == nullptr) {
			field = 1;
		} else if (image->backing.image != mark.image) {
			field = 2;
		} else if (image->registered != mark.registered) {
			field = 3;
		} else if (image->Binding().needs_rebind) {
			field = 4;
		} else if (image->resident_first != mark.resident_first) {
			field = 5;
		} else if (image->backing.state.layout != mark.layout) {
			field = 6;
		} else if (image->backing.state.access_mask != mark.access) {
			field = 7;
		} else if (image->backing.state.pl_stage != mark.stage) {
			field = 8;
		} else if (image->backing.subresource_states.empty() != mark.single_state) {
			field = 9;
		} else if (compare_serials && (mark.texture || attachment_serials) &&
		           image->ContentSerial() != mark.serial) {
			field = 10;
		} else if (!image->info.data.Empty() && !image->registered) {
			field = 11;
		} else if (!image->info.data.Empty() && image->IsCpuDirty()) {
			field = 12;
		} else if (!image->info.data.Empty() && image->IsBufferModified()) {
			field = 13;
		} else if (mark.texture && !DrawRunTextureReady(*image)) {
			field = 14;
		}
		if (field != 0) {
			static uint32_t logged = 0;
			if (log_change && logged++ < DrawRun::MismatchLogLimit()) {
				static const char* names[] = {"none", "slot freed", "native image", "registration",
				    "rebind", "residency", "layout", "access", "stage", "subresource state",
				    "content serial", "unregistered", "CPU dirty", "buffer modified", "texture refresh"};
				std::printf("DrawRunVerifyImage: mark=%u role=%s field=%s guest=0x%llx size=0x%llx "
				            "layout=%u->%u access=0x%llx->0x%llx stage=0x%llx->0x%llx "
				            "serial=%llu->%llu subresources=%zu feedback=%llu/%llu attachment=%llu run=%llu active=%llu\n",
				            index, mark.texture ? "texture" : "attachment", names[field],
				            (unsigned long long)mark.address, (unsigned long long)mark.size,
				            (unsigned)mark.layout, image ? (unsigned)image->backing.state.layout : 0u,
				            (unsigned long long)(VkAccessFlags2)mark.access,
				            image ? (unsigned long long)(VkAccessFlags2)image->backing.state.access_mask : 0ull,
				            (unsigned long long)(VkPipelineStageFlags2)mark.stage,
				            image ? (unsigned long long)(VkPipelineStageFlags2)image->backing.state.pl_stage : 0ull,
				            (unsigned long long)mark.serial, image ? (unsigned long long)image->ContentSerial() : 0ull,
				            image ? image->backing.subresource_states.size() : 0u,
				            image ? (unsigned long long)image->feedback_instance : 0ull,
				            image ? (unsigned long long)image->feedback_serial : 0ull,
				            image ? (unsigned long long)image->feedback_attached : 0ull,
				            (unsigned long long)m_run.rendering_serial,
				            (unsigned long long)m_context.GetCommandScheduler().Current().ActiveRenderingSerial());
				std::fflush(stdout);
			}
			return ((index + 1u) << 8u) | (mark.texture ? 0x80u : 0u) | field;
		}
	}
	return 0;
}

// KYTY_DRAW_RUN_IMAGE_EPOCH (drawRun.h): DrawRunImagesChange is skipped while no image property it
// reads changed since the record's marks were proven current.
bool RenderExecutor::DrawRunImagesUnchanged(bool compare_serials, bool attachments_only) const {
	const auto mode  = DrawRun::ImageEpochMode();
	const auto epoch = ImageStateEpoch();
	if (mode != 0 && m_run.image_epoch != 0 && epoch == m_run.image_epoch) {
		auto& totals = DrawRun::GetTotals();
		if (mode == 1) {
			OwnedCounters::Add(totals.epoch_skips, 1);
			return true;
		}
		totals.epoch_checks.fetch_add(1, std::memory_order_relaxed);
		const auto change =
		    DrawRunImagesChange(compare_serials, attachments_only, false, !m_run_recall);
		if (change != 0) {
			totals.epoch_mismatches.fetch_add(1, std::memory_order_relaxed);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
				std::printf("DrawRunImageEpochVerify: an image changed without an epoch bump "
				            "(change 0x%x)\n",
				            change);
				std::fflush(stdout);
			}
			if (mode == 3) {
				EXIT("DrawRunImageEpochVerify: an image changed without an epoch bump\n");
			}
		}
		return change == 0;
	}
	const bool unchanged =
	    DrawRunImagesChange(compare_serials, attachments_only, false, !m_run_recall) == 0;
	// A full check that passed proves the marks current as of `epoch` (read before it).
	if (unchanged && compare_serials && !attachments_only && !m_run_recall) {
		m_run.image_epoch = epoch;
	}
	return unchanged;
}

void RenderExecutor::DrawRunMarkBindings() {
	auto& images = m_context.GetTextureCache().m_slot_images;
	for (const auto& mark: m_run.images) {
		auto* image = images.try_get(mark.id);
		if (image == nullptr) {
			continue; // the late check falls back
		}
		if (mark.texture) {
			// BindImage (no storage bindings in a run).
			if (image->info.data.Empty()) {
				continue;
			}
			if (image->Binding().is_bound) {
				image->Binding().force_general |= image->Binding().shader_write;
			}
			if (!Common::RendererBatchEnabled() ||
			    (!image->Binding().is_bound && !image->Binding().is_target)) {
				m_bound_images.push_back(mark.id);
			}
			image->Binding().is_bound = true;
		} else {
			// BindRenderTarget.
			if (!Common::RendererBatchEnabled() ||
			    (!image->Binding().is_bound && !image->Binding().is_target)) {
				m_bound_images.push_back(mark.id);
			}
			image->Binding().is_target = true;
		}
	}
}

// KYTY_DRAW_RUN_TARGETS (drawPrep/drawRun.h): the record's targets are what this draw's resolution
// would give (the same inputs, the run certificate and unchanged attachments).
bool RenderExecutor::DrawRunTargetsCandidate(const CommandBuffer& buffer, const DrawRenderState& state,
                                             uint32_t render_target_slice_offset) const {
	const auto& run   = m_run;
	const void* pixel = state.ps_active ? state.ps_input_info.stage.program : nullptr;
	return m_run_prev_valid && !m_run_recall && m_in_engine_commit && run.acquire_valid &&
	       run.depth.size() == 1 && run.colors.size() == run.color_count &&
	       m_run_targets_key != 0 && m_run_targets_key == run.targets_key &&
	       state.ps_active == run.ps_active && pixel == run.pixel_program &&
	       render_target_slice_offset == run.slice_offset &&
	       DrawRun::ActivityEpoch() == run.activity && DrawRunCommandUnchanged(buffer) &&
	       DrawRunImagesUnchanged(true, true);
}

// KYTY_DRAW_RUN_KEY_DIAG: a draw whose key differed from the record's compares its resolved
// pipeline and structure with the record's; a draw that recorded becomes the record.
void RenderExecutor::DrawRunKeyDiagResolved(const CommandBuffer& buffer,
                                            std::span<PreparedBindings* const> stages,
                                            const PipelineCache::Pipeline&     pipeline,
                                            const RenderState& rendering, bool recorded) {
	auto&       kd     = m_key_diag;
	const auto  handle = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(static_cast<VkPipeline>(pipeline.pipeline)));
	std::vector<uint64_t> words;
	words.reserve(64);
	const auto add_handle = [&words](auto native) {
		using Native = typename decltype(native)::NativeType;
		words.push_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<Native>(native))));
	};
	for (const auto* stage: stages) {
		words.push_back(reinterpret_cast<uintptr_t>(stage->runtime->program));
		for (const auto& image: stage->images) {
			add_handle(image.image_view);
			words.push_back(static_cast<uint64_t>(image.layout));
			for (const auto view: image.mip_views) {
				add_handle(view);
			}
		}
		for (const auto sampler: stage->samplers) {
			add_handle(sampler);
		}
		words.push_back(0xffffffffull);
	}
	for (uint32_t i = 0; i < rendering.num_color_attachments; i++) {
		add_handle(rendering.color_attachments[i].image_view);
		words.push_back(static_cast<uint64_t>(rendering.color_attachments[i].image_layout));
	}
	add_handle(rendering.depth_stencil_attachment.image_view);
	words.push_back(static_cast<uint64_t>(rendering.depth_stencil_attachment.image_layout));
	const uint64_t structure = XXH3_64bits(words.data(), words.size() * sizeof(uint64_t));
	if (kd.combo != -2 && kd.record_resolved) {
		DrawRun::KeyDiag::CountResolved(kd.combo, handle == kd.record_pipeline,
		                                structure == kd.record_structure);
	}
	if (recorded && m_run_parts != nullptr) {
		kd.record_parts       = *m_run_parts;
		kd.record_parts_valid = true;
		kd.record_pipeline    = handle;
		kd.record_structure   = structure;
		kd.record_resolved    = true;
		auto& last = kd.recent[(kd.recent_next + kd.recent.size() - 1) % kd.recent.size()];
		if (last.key != m_run.key || last.command != m_run.command || last.serial != m_run.rendering_serial) {
			auto& entry   = kd.recent[kd.recent_next];
			kd.recent_next = (kd.recent_next + 1) % static_cast<uint32_t>(kd.recent.size());
			entry         = {m_run.key, m_run.activity, m_run.command, m_run.tick, m_run.rendering_serial};
		} else {
			last.activity = m_run.activity;
			last.tick     = m_run.tick;
		}
	} else if (!recorded) {
		kd.record_parts_valid = false;
		kd.record_resolved    = false;
	}
}

bool RenderExecutor::DrawRunCandidate(const CommandBuffer& buffer, const DrawRenderState& state,
                                      uint32_t render_target_slice_offset) {
	using DrawRun::Miss;
	auto& totals = DrawRun::GetTotals();
	OwnedCounters::Add(totals.draws, 1);
	if (m_run_parts != nullptr) [[unlikely]] {
		// KYTY_DRAW_RUN_KEY_DIAG: why the key check fails.
		auto& kd = m_key_diag;
		if (!m_run_prev_valid) {
			DrawRun::KeyDiag::CountPrevInvalid(kd.prev_reason, m_run_key != 0 && m_run_key == kd.last_key);
		} else if (m_run_key != 0 && m_run_key != m_run.key && kd.record_parts_valid) {
			uint32_t mask = 0;
			for (uint32_t i = 0; i < DrawRun::KpCount; i++) {
				mask |= (*m_run_parts)[i] != kd.record_parts[i] ? (1u << i) : 0u;
			}
			bool recallable = false;
			bool any        = false;
			for (const auto& recent: kd.recent) {
				if (recent.key == m_run_key && recent.command == buffer.Identity() &&
				    recent.tick == m_context.GetCommandScheduler().CurrentTick() &&
				    recent.serial != 0 && recent.serial == buffer.ActiveRenderingSerial()) {
					any        = true;
					recallable = recallable || recent.activity == DrawRun::ActivityEpoch();
				}
			}
			kd.combo = DrawRun::KeyDiag::CountKeyMismatch(mask, recallable, any);
		}
		kd.last_key = m_run_key;
	}
	bool pipeswap = false;
	if (!m_run_prev_valid || m_run_key == 0 || m_run_key != m_run.key) {
		// KYTY_DRAW_RUN_PIPESWAP: the key differs in the vertex program alone.
		if (!m_run_prev_valid || m_run_key == 0 || m_run_swap_key == 0 ||
		    m_run_swap_key != m_run.swap_key || !DrawRun::PipeSwapEnabled()) {
			return false;
		}
		pipeswap = true;
		OwnedCounters::Add(totals.swap_matches, 1);
	} else {
		OwnedCounters::Add(totals.key_matches, 1);
	}
	if (DrawRun::ActivityEpoch() != m_run.activity) {
		DrawRun::CountMiss(Miss::Activity);
		return false;
	}
	if (buffer.Identity() != m_run.command ||
	    m_context.GetCommandScheduler().CurrentTick() != m_run.tick) {
		DrawRun::CountMiss(Miss::Command);
		return false;
	}
	if (buffer.ActiveRenderingSerial() == 0 ||
	    buffer.ActiveRenderingSerial() != m_run.rendering_serial) {
		DrawRun::CountMiss(Miss::Instance);
		return false;
	}
	if (!m_prepared_validated) {
		DrawRun::CountMiss(Miss::Validation);
		return false;
	}
	const void* pixel = state.ps_active ? state.ps_input_info.stage.program : nullptr;
	if (state.ps_active != m_run.ps_active || render_target_slice_offset != m_run.slice_offset ||
	    state.vertex_stages_written != 1 ||
	    (!pipeswap && state.vertex_info[0].stage.program != m_run.vertex_program) ||
	    pixel != m_run.pixel_program) {
		DrawRun::CountMiss(Miss::Programs);
		return false;
	}
	// KYTY_DRAW_RUN_MESH: a mesh record is continued only by a mesh draw, a vertex record only by a
	// vertex draw (a pipeline swap could otherwise cross the two kinds).
	if (DrawRun::MeshRunsEnabled() &&
	    (state.vertex_info[0].stage.program->stage == ShaderType::Mesh) != m_run.mesh) {
		OwnedCounters::Add(totals.mesh_kind_refused, 1);
		return false;
	}
	bool swap_writes = false;
	if (pipeswap) {
		const auto* recorded = static_cast<const ShaderRecompiler::IR::CompiledShaderInfo*>(
		    m_run.vertex_program);
		const auto& program = *state.vertex_info[0].stage.program;
		if (recorded == nullptr || recorded == &program || !DrawRunPipeSwapCompatible(*recorded, program)) {
			OwnedCounters::Add(totals.swap_incompatible, 1);
			return false;
		}
		// The new vertex program: run-eligible, and its buffer writes (if allowed) off the run's images.
		if (!DrawRunProgramEligible(program)) {
			if (!m_run.record_writes_ok || !DrawRunProgramEligible(program, true)) {
				DrawRun::CountMiss(Miss::Writes);
				return false;
			}
			switch (DrawRunWritesOverRun(state.vertex_info[0].stage, m_run.images)) {
				case 0: break;
				case 1: swap_writes = true; break;
				default: DrawRun::CountMiss(Miss::Writes); return false;
			}
		}
	}
	// KYTY_DRAW_RUN_WRITES: a record of a writing program exists only while the switch is on (a
	// change ends the run). The candidate's own writes must not lie over an image of the run.
	bool writes = false;
	if (m_run.writes) {
		const ShaderStageRuntime* runtimes[] = {&state.vertex_info[0].stage,
		                                        state.ps_active ? &state.ps_input_info.stage : nullptr};
		for (const auto* runtime: runtimes) {
			if (runtime == nullptr) {
				continue;
			}
			switch (DrawRunWritesOverRun(*runtime, m_run.images)) {
				case 0: break;
				case 1: writes = true; break;
				default: DrawRun::CountMiss(Miss::Writes); return false;
			}
		}
	}
	OwnedCounters::Add(totals.continued, 1);
	if (writes || swap_writes) {
		OwnedCounters::Add(totals.write_continued, 1);
	}
	if (pipeswap) {
		OwnedCounters::Add(totals.swapped, 1);
		m_run_pipeswap = true;
	}
	return true;
}

bool RenderExecutor::DrawRunPipeSwapCompatible(const ShaderRecompiler::IR::CompiledShaderInfo& recorded,
                                               const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	const auto& a = recorded.info;
	const auto& b = program.info;
	if (a.images.size() != b.images.size() || a.samplers.size() != b.samplers.size()) {
		return false;
	}
	for (size_t i = 0; i < a.images.size(); i++) {
		auto x = a.images[i];
		auto y = b.images[i];
		x.source = y.source = 0;
		x.first_use_pc = y.first_use_pc = 0;
		if (!(x == y)) {
			return false;
		}
	}
	for (size_t i = 0; i < a.samplers.size(); i++) {
		auto x = a.samplers[i];
		auto y = b.samplers[i];
		x.source = y.source = 0;
		x.first_use_pc = y.first_use_pc = 0;
		if (!(x == y)) {
			return false;
		}
	}
	const auto viewport_index = [](const auto& outputs) {
		return std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
			return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
		});
	};
	return viewport_index(a.outputs) == viewport_index(b.outputs);
}

// KYTY_DRAW_RUN_RECALL: a hash of a binding's mip views (RecallTexture::mips).
static uint64_t DrawRunMipViewsHash(const std::vector<vk::ImageView>& views) {
	return views.empty() ? 0 : XXH3_64bits(views.data(), views.size() * sizeof(vk::ImageView));
}

// KYTY_DRAW_RUN_RECALL: the stage's texture set of these words (-1: the current one, 0..2 its
// history; -2: none) when it holds exactly the bindings the record committed.
template <typename RecallTextures>
static int DrawRunRecallSet(const PreparedBindings& prepared, const ShaderStageRuntime& runtime,
                            const RecallTextures& textures) {
	const auto& program = *runtime.program;
	const auto& words   = runtime.resources->images;
	if (program.info.images.size() != textures.size()) {
		return -2;
	}
	const auto same = [&](const std::vector<TextureBinding>& images) {
		if (images.size() != textures.size()) {
			return false;
		}
		for (size_t i = 0; i < images.size(); i++) {
			const auto& binding = images[i];
			const auto& kept    = textures[i];
			if (binding.image_id != kept.id || binding.image_view != kept.view ||
			    binding.layout != kept.layout || DrawRunMipViewsHash(binding.mip_views) != kept.mips) {
				return false;
			}
		}
		return true;
	};
	if (textures.empty()) {
		return -1; // nothing to take back: the stage's images are cleared
	}
	if (prepared.texture_program == &program && std::ranges::equal(prepared.texture_words, words)) {
		return same(prepared.images) ? -1 : -2;
	}
	for (size_t i = 0; i < prepared.texture_history.size(); i++) {
		const auto& set = prepared.texture_history[i];
		if (set.program == &program && std::ranges::equal(set.words, words)) {
			return same(set.images) ? static_cast<int>(i) : -2;
		}
	}
	return -2;
}

// KYTY_DRAW_RUN_RECALL (drawPrep/drawRun.h): whether the committed draw continues a kept record of
// its rendering instance (PrepareDrawRenderState, after DrawRunCandidate declined a plain
// continuation). On success the record is m_run (the previous one kept in its place) and, unless in
// verify mode, the stages hold the record's texture sets and samplers and `state` its targets.
bool RenderExecutor::DrawRunRecallCandidate(const CommandBuffer& buffer, DrawRenderState& state,
                                            uint32_t render_target_slice_offset) {
	if (m_run_key == 0 || (m_run_prev_valid && m_run_key == m_run.key)) {
		return false;
	}
	DrawRunRecord* kept = nullptr;
	for (auto& record: m_run_kept) {
		if (record.valid && record.key == m_run_key) {
			kept = &record;
			break;
		}
	}
	if (kept == nullptr) {
		return false;
	}
	auto& totals = DrawRun::GetTotals();
	totals.recall_matches.fetch_add(1, std::memory_order_relaxed);
	auto& record = *kept;
	if (DrawRun::ActivityEpoch() != record.activity || buffer.Identity() != record.command ||
	    m_context.GetCommandScheduler().CurrentTick() != record.tick ||
	    buffer.ActiveRenderingSerial() == 0 ||
	    buffer.ActiveRenderingSerial() != record.rendering_serial) {
		record.valid = false; // it can never be recalled again
		return false;
	}
	const void* pixel = state.ps_active ? state.ps_input_info.stage.program : nullptr;
	if (!record.recall_valid || !record.acquire_valid || record.depth.size() != 1 ||
	    record.colors.size() != record.color_count || !m_prepared_validated ||
	    state.ps_active != record.ps_active || render_target_slice_offset != record.slice_offset ||
	    state.vertex_stages_written != 1 ||
	    state.vertex_info[0].stage.program != record.vertex_program || pixel != record.pixel_program ||
	    (state.ps_active && !m_graphics_bindings.pixel.has_value())) {
		return false;
	}
	// KYTY_DRAW_RUN_MESH: only a record of the draw's own kind (same program, so this is a safeguard).
	if (DrawRun::MeshRunsEnabled() &&
	    (state.vertex_info[0].stage.program->stage == ShaderType::Mesh) != record.mesh) {
		OwnedCounters::Add(totals.mesh_kind_refused, 1);
		return false;
	}
	const auto vertex_textures = std::span(record.recall_textures).first(record.recall_vertex_textures);
	const auto pixel_textures  = std::span(record.recall_textures).subspan(record.recall_vertex_textures);
	auto&      vertex          = m_graphics_bindings.vertex[0];
	const auto vertex_set      = DrawRunRecallSet(vertex, state.vertex_info[0].stage, vertex_textures);
	const auto pixel_set       = state.ps_active ? DrawRunRecallSet(*m_graphics_bindings.pixel,
	                                                                state.ps_input_info.stage, pixel_textures)
	                                             : (pixel_textures.empty() ? -1 : -2);
	const auto pixel_samplers  = record.recall_samplers.size() - record.recall_vertex_samplers;
	if (vertex_set == -2 || pixel_set == -2 ||
	    state.vertex_info[0].stage.program->info.samplers.size() != record.recall_vertex_samplers ||
	    (state.ps_active ? state.ps_input_info.stage.program->info.samplers.size() : 0) !=
	        pixel_samplers) {
		totals.recall_sets_missed.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	// KYTY_DRAW_RUN_WRITES: the candidate's own writes must not lie over an image of the record.
	if (record.writes) {
		const ShaderStageRuntime* runtimes[] = {&state.vertex_info[0].stage,
		                                        state.ps_active ? &state.ps_input_info.stage : nullptr};
		for (const auto* runtime: runtimes) {
			if (runtime != nullptr && DrawRunWritesOverRun(*runtime, record.images) > 1) {
				DrawRun::CountMiss(DrawRun::Miss::Writes);
				return false;
			}
		}
	}
	// The record becomes the run; the previous one (complete or not) takes its place.
	std::swap(m_run, record);
	record.valid     = m_run_keepable && record.key != m_run.key;
	m_run.valid      = false; // the draw's own record (DrawRunRecordDraw) validates it again
	m_run_prev_valid = true;
	m_run_keepable   = true;
	m_run_recall     = true;
	totals.recalled.fetch_add(1, std::memory_order_relaxed);
	if (DrawRun::GetMode() != DrawRun::Mode::On) {
		return true; // verify: the normal path runs and is compared with the record
	}
	// The record's bindings: each stage's texture set back from its history, its samplers, and the
	// record's targets.
	const auto restore = [](PreparedBindings& prepared, const ShaderStageRuntime& runtime, int set,
	                        std::span<const vk::Sampler> samplers) {
		if (runtime.program->info.images.empty()) {
			prepared.images.clear();
			prepared.texture_program = nullptr;
		} else if (set >= 0) {
			auto& found = prepared.texture_history[static_cast<size_t>(set)];
			std::swap(found.program, prepared.texture_program);
			std::swap(found.words, prepared.texture_words);
			std::swap(found.images, prepared.images);
		}
		prepared.samplers.assign(samplers.begin(), samplers.end());
	};
	const auto samplers = std::span<const vk::Sampler>(m_run.recall_samplers);
	restore(vertex, state.vertex_info[0].stage, vertex_set,
	        samplers.first(m_run.recall_vertex_samplers));
	if (state.ps_active) {
		restore(*m_graphics_bindings.pixel, state.ps_input_info.stage, pixel_set,
		        samplers.subspan(m_run.recall_vertex_samplers));
	}
	for (uint32_t i = 0; i < m_run.color_count; i++) {
		state.color_info[i] = m_run.colors[i];
	}
	state.depth_info = m_run.depth[0];
	return true;
}

// KYTY_DRAW_RUN_RECALL: before a draw records a new run, the record it replaces is kept (in the
// entry of the same key, else the next one in turn).
void RenderExecutor::DrawRunKeep() {
	if (!m_run_keepable || !m_run.recall_valid || m_run.key == m_run_key || !DrawRun::RecallEnabled()) {
		return;
	}
	m_run_keepable = false;
	auto slot      = m_run_kept_next;
	for (uint32_t i = 0; i < m_run_kept.size(); i++) {
		if (m_run_kept[i].key == m_run.key) {
			slot = i;
			break;
		}
	}
	if (slot == m_run_kept_next) {
		m_run_kept_next = (m_run_kept_next + 1) % static_cast<uint32_t>(m_run_kept.size());
	}
	std::swap(m_run, m_run_kept[slot]);
	m_run_kept[slot].valid = true;
}

// KYTY_DRAW_RUN_WRITES: 0 when the stage writes no buffer, 1 when it writes only outside the run's
// images, 2 when a written V# range (as the descriptor states it, before any clamp) lies over an
// image of the run (a texture or an attachment) or the run has no image marks to compare.
uint32_t RenderExecutor::DrawRunWritesOverRun(const ShaderStageRuntime&     runtime,
                                              std::span<const DrawRunImage> images) {
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	uint32_t    result   = 0;
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource = program.info.buffers[i];
		if (!resource.written && !resource.atomic) {
			continue;
		}
		if (i >= snapshot.buffers.size() || snapshot.buffers[i].dword_count < 4) {
			return 2;
		}
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
		const auto address    = descriptor.Base48();
		const auto size       = descriptor.GetSize();
		if (address == 0 || size == 0) {
			continue; // a null V#: nothing is written (HasShaderBufferWrites)
		}
		result = 1;
		const auto end = address + size < address ? UINT64_MAX : address + size;
		for (const auto& mark: images) {
			if (mark.size != 0 && address < mark.address + mark.size && mark.address < end) {
				return 2;
			}
		}
	}
	return result;
}

bool RenderExecutor::DrawRunAcquireCandidate(const CommandBuffer&               buffer,
                                             const DrawRenderState&             state,
                                             std::span<PreparedBindings* const> stages,
                                             const vk::Rect2D*                  written) const {
	const auto& run = m_run;
	if (!m_run_prev_valid || !m_in_engine_commit || !run.acquire_valid || run.depth.size() != 1 ||
	    run.colors.size() != run.color_count || state.color_count != run.color_count ||
	    DrawRun::ActivityEpoch() != run.activity || buffer.Identity() != run.command ||
	    m_context.GetCommandScheduler().CurrentTick() != run.tick ||
	    buffer.ActiveRenderingSerial() == 0 ||
	    buffer.ActiveRenderingSerial() != run.rendering_serial ||
	    (written != nullptr) != run.bounded || (written != nullptr && *written != run.written)) {
		return false;
	}
	for (uint32_t i = 0; i < state.color_count; i++) {
		if (!SameRenderColorInfo(state.color_info[i], run.colors[i])) {
			return false;
		}
	}
	if (!SameRenderDepthInfo(state.depth_info, run.depth[0])) {
		return false;
	}
	// No texture of the draw is an attachment or lies over one (AcquireRenderTargets would detect a
	// feedback loop or choose other layouts; an alias is synchronized from the attachment).
	{
		auto&            cache = m_context.GetTextureCache();
		std::scoped_lock lock {cache.m_lock};
		// The record's attachments follow its textures.
		const auto attachments = std::span<const DrawRunImage>(run.images).subspan(
		    std::min<size_t>(run.attachments_begin, run.images.size()));
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				const auto* image = cache.m_slot_images.try_get(binding.image_id);
				if (image == nullptr) {
					return false;
				}
				for (const auto& mark: attachments) {
					if (!mark.texture && mark.id == binding.image_id) {
						return false;
					}
				}
				if (DrawRunOverAttachment(attachments, image->info.data.address,
				                          image->info.data.size)) {
					return false;
				}
			}
		}
	}
	return DrawRunImagesUnchanged(true, true);
}

bool RenderExecutor::DrawRunPartialPush(const CommandBuffer&           buffer,
                                        const PipelineCache::Pipeline& pipeline) const {
	return DrawRun::PushPartialEnabled() && !m_run_recall && !m_run_pipeswap &&
	       pipeline.uses_push_descriptors &&
	       m_run.push_valid &&
	       m_run.push_layout == pipeline.pipeline_layout &&
	       buffer.DescriptorEpoch(vk::PipelineBindPoint::eGraphics) == m_run.push_epoch;
}

// After an eligible draw was recorded: what a continuation of it may reuse, and its certificate.
void RenderExecutor::DrawRunRecordDraw(const CommandBuffer& buffer, const DrawRenderState& state,
                                       uint32_t render_target_slice_offset,
                                       const RenderState& rendering,
                                       std::span<PreparedBindings* const> stages,
                                       const vk::Rect2D*                  written,
                                       const PipelineCache::Pipeline&     pipeline) {
	if (!m_run_active) {
		DrawRunKeep();
	}
	auto& run            = m_run;
	run.key              = m_run_key;
	run.targets_key      = m_run_targets_key;
	run.swap_key         = m_run_swap_key;
	run.activity         = DrawRun::ActivityEpoch();
	run.command          = buffer.Identity();
	run.tick             = m_context.GetCommandScheduler().CurrentTick();
	run.rendering_serial = buffer.ActiveRenderingSerial();
	// The draw's push (whole or partial) is the last descriptor command at the bind point.
	run.push_epoch = buffer.DescriptorEpoch(vk::PipelineBindPoint::eGraphics);
	if (m_run_active) {
		// A continuation changed none of the recorded images, targets or bindings; a pipeline swap
		// changed the vertex program and the pipeline (layout) the next draw is compared with.
		if (m_run_pipeswap) {
			run.vertex_program = state.vertex_info[0].stage.program;
			run.push_valid     = pipeline.uses_push_descriptors;
			run.push_layout    = pipeline.pipeline_layout;
			run.mesh           = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
		}
		run.valid = true;
		return;
	}
	run.push_valid  = pipeline.uses_push_descriptors;
	run.push_layout = pipeline.pipeline_layout;
	run.slice_offset   = render_target_slice_offset;
	run.vertex_program = state.vertex_info[0].stage.program;
	run.pixel_program  = state.ps_active ? state.ps_input_info.stage.program : nullptr;
	run.ps_active      = state.ps_active;
	run.mesh           = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	run.color_count    = state.color_count;
	run.color_slots    = state.color_slots_written;
	run.rendering      = rendering;
	run.images.clear();
	{
		std::scoped_lock lock {m_context.GetTextureCache().m_lock};
		// KYTY_DEPTH_FEEDBACK_LAZY: an acquisition that does not sample the image (as a
		// continuation's or a reused acquisition's, whose textures are never an attachment) keeps
		// the attachment access, so nothing is excluded.
		if (state.depth_info.image_id && DepthFeedbackKeepEnabled() && !DepthFeedbackLazyEnabled() &&
		    !state.depth_info.AttachmentWriteAspects()) {
			const auto* depth = m_context.GetTextureCache().m_slot_images.try_get(state.depth_info.image_id);
			const auto attachment_access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
			                               vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
			// NoteDepthFeedback just established an unwritten-instance proof. The next acquisition
			// can now adopt attachment_access | ShaderRead even for an unsampled attachment
			// (AcquireRenderTargets). It must run normally to publish that union and its deferred
			// ordering; this first draw cannot seed either a continuation or acquisition reuse.
			if (depth != nullptr) {
				depth->CpParTouch(); // Astro Bot PC (KYTY_CP_PAR=1b3)
			}
			if (depth != nullptr && depth->backing.state.access_mask == attachment_access &&
			    depth->feedback_instance == run.rendering_serial &&
			    depth->feedback_serial == depth->ContentSerial()) {
				DrawRun::GetTotals().depth_promotions_excluded.fetch_add(1, std::memory_order_relaxed);
				m_key_diag.reason = DrawRun::KeyDiag::Inelig::DepthPromotion;
				return;
			}
		}
		// KYTY_DRAW_RUN_IMAGE_EPOCH: the epoch before the marks, valid when every marked image also
		// passes the check's flag tests now (what the marks do not record).
		const auto epoch = ImageStateEpoch();
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				run.images.push_back(MakeDrawRunImage(binding.image_id, true));
			}
		}
		run.attachments_begin = static_cast<uint32_t>(run.images.size());
		for (uint32_t i = 0; i < state.color_count; i++) {
			run.images.push_back(MakeDrawRunImage(state.color_info[i].image_id, false));
		}
		if (state.depth_info.image_id) {
			run.images.push_back(MakeDrawRunImage(state.depth_info.image_id, false));
		}
		bool clean = true;
		for (const auto& mark: run.images) {
			const auto* image = m_context.GetTextureCache().m_slot_images.try_get(mark.id);
			clean = clean && image != nullptr && !image->Binding().needs_rebind &&
			        (image->info.data.Empty() ||
			         (image->registered && !image->IsCpuDirty() && !image->IsBufferModified())) &&
			        (!mark.texture || DrawRunTextureReady(*image));
		}
		run.image_epoch = clean ? epoch : 0;
	}
	// A texture over an attachment's memory (an alias of it) is synchronized from the attachment by
	// each draw's texture resolution (SyncAliasFromOwner), which a continuation skips: no run.
	const auto attachments = std::span<const DrawRunImage>(run.images).subspan(run.attachments_begin);
	for (const auto& mark: run.images) {
		if (mark.texture && DrawRunOverAttachment(attachments, mark.address, mark.size)) {
			DrawRun::GetTotals().alias_excluded.fetch_add(1, std::memory_order_relaxed);
			m_key_diag.reason = DrawRun::KeyDiag::Inelig::Alias;
			return;
		}
	}
	const bool verify = DrawRun::GetMode() == DrawRun::Mode::Verify;
	run.acquire_valid = verify || DrawRun::AcquireReuseEnabled();
	run.verify_valid  = verify;
	if (run.acquire_valid) {
		run.colors.assign(state.color_info, state.color_info + state.color_count);
		run.depth.assign(1, state.depth_info);
		run.bounded = written != nullptr;
		run.written = written != nullptr ? *written : vk::Rect2D {};
	}
	run.recall_valid = DrawRun::RecallEnabled();
	if (run.recall_valid) {
		run.recall_textures.clear();
		run.recall_samplers.clear();
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				run.recall_textures.push_back({binding.image_id, binding.image_view, binding.layout,
				                               DrawRunMipViewsHash(binding.mip_views)});
			}
			run.recall_samplers.insert(run.recall_samplers.end(), stage->samplers.begin(),
			                           stage->samplers.end());
			if (stage == stages.front()) {
				run.recall_vertex_textures = static_cast<uint32_t>(run.recall_textures.size());
				run.recall_vertex_samplers = static_cast<uint32_t>(run.recall_samplers.size());
			}
		}
	}
	if (verify) {
		run.pushed_images.assign(m_descriptor_images.begin(), m_descriptor_images.end());
		run.textures.clear();
		run.samplers.clear();
		for (const auto* stage: stages) {
			run.textures.insert(run.textures.end(), stage->images.begin(), stage->images.end());
			run.samplers.insert(run.samplers.end(), stage->samplers.begin(), stage->samplers.end());
		}
	}
	run.valid = true;
}

// KYTY_DRAW_RUN=verify: a draw that would have continued the run took the normal path; what the
// continuation would have reused must be what it computed.
void RenderExecutor::DrawRunVerify(const DrawRenderState& state, const RenderState& rendering,
                                   vk::ImageAspectFlags               feedback_aspects,
                                   std::span<PreparedBindings* const> stages) {
	const auto& run = m_run;
	if (!run.verify_valid) {
		return; // recorded before verify mode was switched on
	}
	DrawRun::CountVerifyCheck();
	if (state.color_count != run.color_count || state.color_slots_written != run.color_slots ||
	    run.depth.size() != 1) {
		DrawRun::ReportMismatch("colour target count", state.color_count);
		return;
	}
	for (uint32_t i = 0; i < state.color_count; i++) {
		if (!SameRenderColorInfo(state.color_info[i], run.colors[i])) {
			DrawRun::ReportMismatch("colour target", i);
		}
	}
	if (!SameRenderDepthInfo(state.depth_info, run.depth[0])) {
		DrawRun::ReportMismatch("depth target");
	}
	size_t texture = 0;
	size_t sampler = 0;
	for (const auto* stage: stages) {
		for (const auto& binding: stage->images) {
			if (texture >= run.textures.size()) {
				DrawRun::ReportMismatch("texture count", texture);
				return;
			}
			const auto& old = run.textures[texture++];
			if (binding.image_id != old.image_id || binding.image_view != old.image_view ||
			    binding.layout != old.layout || binding.desc.type != old.desc.type ||
			    binding.mip_views != old.mip_views) {
				DrawRun::ReportMismatch(binding.image_view != old.image_view || binding.mip_views != old.mip_views
				                            ? "texture views" : "texture binding", texture - 1);
			}
		}
		for (const auto handle: stage->samplers) {
			if (sampler >= run.samplers.size() || handle != run.samplers[sampler++]) {
				DrawRun::ReportMismatch("sampler", sampler);
			}
		}
	}
	if (texture != run.textures.size() || sampler != run.samplers.size()) {
		DrawRun::ReportMismatch("binding count", texture);
	}
	if (!(rendering == run.rendering)) {
		DrawRun::ReportMismatch("rendering state");
	}
	if (feedback_aspects) {
		DrawRun::ReportMismatch("attachment feedback", static_cast<uint32_t>(feedback_aspects));
	}
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type = source.type;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
		SetVulkanObjectNameF(buffer.GetContext().GetGraphics().device, prepared.buffer,
		                     "Kyty.IndexBuffer[guest=transient size=0x{:x} type={}]", source.size,
		                     static_cast<uint32_t>(source.type));
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
		SetVulkanObjectNameF(buffer.GetContext().GetGraphics().device, prepared.buffer,
		                     "Kyty.IndexBuffer[guest=0x{:016x} size=0x{:x} type={}]",
		                     source.address, source.size, static_cast<uint32_t>(source.type));
	}
	return prepared;
}

static void CommitVertexBuffers(const CommandSink&           vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(const CommandSink& vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
	                             const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, const CommandSink& vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kPolygon:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.IsIndexed()) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count, emit.first_vertex,
				               emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count, i + emit.first_vertex,
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

// CommandBuffer::DrawScope: the draw's only memory writes are its own color/depth attachments,
// and it samples none of them. Conservative: any shader write path (storage buffers/images,
// atomics, address writes, GDS, fault/LOD counters), indirect arguments or a feedback loop makes
// the draw unsafe for sinking a pending barrier past it.
// plain_pixel: the pixel stage runs its feedback-free variant (KYTY_LOD_STATS_PLAIN_VARIANT), so
// its mip-statistics binding is not written.
bool DrawPrep::ProgramBarrierSafe(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                  bool                                            plain_pixel) {
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	if (program.has_address_writes) {
		return false;
	}
	for (const auto& resource: program.info.buffers) {
		if (resource.written || resource.atomic) {
			return false;
		}
	}
	for (const auto& resource: program.info.images) {
		if (resource.written || resource.atomic) {
			return false;
		}
	}
	for (const auto& binding: program.bindings.descriptors) {
		if (binding.kind == Kind::Gds || binding.kind == Kind::FaultBuffer ||
		    (binding.kind == Kind::MipStats && !(plain_pixel && program.stage == ShaderType::Pixel))) {
			return false;
		}
	}
	return true;
}

// DB_RENDER_CONTROL.COPY_DEPTH_TO_COLOR (upstream 1029511bb): a rect-list draw that copies the
// depth target into the R32F colour target (VK_KHR_maintenance8 depth/colour image copy).
bool RenderExecutor::CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
                                       const DrawEmitInfo& emit, DrawRenderState& state) {
	if (!buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
		return false;
	}
	auto& ucfg = buffer.GetUserConfig();
	const auto& hw = buffer.GetRegisters();
	auto& depth = state.depth_info;
	auto& color = state.color_info[0];
	EXIT_NOT_IMPLEMENTED(!m_context.GetGraphics().maintenance8_enabled ||
	                     !Prospero::IsRectList(ucfg.GetPrimType()) ||
	                     draw.IsIndexed() || draw.index_count != 3 || draw.instance_count != 1 ||
	                     emit.first_vertex != 0 || emit.first_instance != 0 ||
	                     state.color_count != 1 || color.target_slot != 0 || !depth.image_id ||
	                     color.desc.info.samples != 1 ||
	                     depth.desc.info.extent.width != color.Extent().width ||
	                     depth.desc.info.extent.height != color.Extent().height ||
	                     depth.desc.view_info.layer_count != 1 ||
	                     color.desc.view_info.layer_count != 1 ||
	                     hw.GetBlendControl(0).enable || state.ps_input_info.ps_pixel_kill_enable ||
	                     state.ps_input_info.ps_depth_export_enable ||
	                     state.ps_input_info.ps_sample_mask_export_enable);
	vk::ImageAspectFlags feedback;
	const auto rendering = AcquireRenderTargets(buffer, &color, 1, depth, feedback);
	if (depth.load_clear_aspects) {
		m_context.GetCommandScheduler().BeginRendering(rendering);
	}
	auto& cache = m_context.GetTextureCache();
	auto& source = cache.GetImage(depth.image_id);
	auto& destination = cache.GetImage(color.image_id);
	EXIT_NOT_IMPLEMENTED((source.backing.format != vk::Format::eD32Sfloat &&
	                      source.backing.format != vk::Format::eD32SfloatS8Uint) ||
	                     destination.backing.format != vk::Format::eR32Sfloat);
	const auto rect = calc_final_scissor(hw.GetScreenViewport(), hw.GetScanModeControl(),
	                                    {rendering.width, rendering.height}, 0);
	if (rect.right > rect.left && rect.bottom > rect.top) {
		const vk::ImageCopy region {
		    {vk::ImageAspectFlagBits::eDepth, depth.desc.view_info.base_level,
		     depth.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {vk::ImageAspectFlagBits::eColor, color.desc.view_info.base_level,
		     color.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {static_cast<uint32_t>(rect.right - rect.left),
		     static_cast<uint32_t>(rect.bottom - rect.top), 1}};
		m_context.GetCommandScheduler().EndRendering();
		const auto command = buffer.Handle();
		source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
		               {}, command);
		destination.Transit(vk::ImageLayout::eTransferDstOptimal,
		                    vk::AccessFlagBits2::eTransferWrite, {}, command);
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                  1, &region);
	}
	return true;
}

// programs_safe: DrawPrep::ProgramBarrierSafe of every stage's program (a binding plan's), or null
// to check them here.
static bool DrawIsBarrierSafe(std::span<PreparedBindings* const> stages,
                              const RenderColorInfo* colors, uint32_t color_count,
                              const RenderDepthInfo& depth, vk::ImageAspectFlags feedback_aspects,
                              bool indirect, bool plain_pixel, const bool* programs_safe = nullptr) {
	if (indirect || feedback_aspects || !BarrierSinkEnabled()) {
		return false;
	}
	if (programs_safe != nullptr && !*programs_safe) {
		return false;
	}
	for (const auto* stage: stages) {
		if (stage == nullptr || stage->runtime == nullptr || !*stage->runtime) {
			return false;
		}
		if (stage->gds.buffer != nullptr ||
		    (programs_safe == nullptr &&
		     !DrawPrep::ProgramBarrierSafe(*stage->runtime->program, plain_pixel))) {
			return false;
		}
		for (const auto& image: stage->images) {
			if (depth.image_id && image.image_id == depth.image_id) {
				return false;
			}
			for (uint32_t i = 0; i < color_count; i++) {
				if (image.image_id == colors[i].image_id) {
					return false;
				}
			}
		}
	}
	return true;
}

// KYTY_DRAW_BINDING_REPEAT_STATS=1 (opt-in diagnostics, counted with aggregates only; it copies
// every draw's descriptor words): how often a draw binds the same programs and descriptors as the
// draw before it, the room a draw-level binding reuse would have for "stamps"
// (FrameEvent.DrawBindingRepeat*: same programs; also the same images and samplers; also the same
// buffers; also the same user data and flattened SRT words).
static bool BindingRepeatStatsEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DRAW_BINDING_REPEAT_STATS");
		return Profiler::AggregateEnabled() && value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

static void CountBindingRepeats(std::span<const ShaderStageRuntime* const> stages) {
	using ShaderRecompiler::IR::DescriptorValue;
	struct Previous {
		const void*                  program = nullptr;
		std::vector<DescriptorValue> buffers;
		std::vector<DescriptorValue> images;
		std::vector<DescriptorValue> samplers;
		std::vector<uint32_t>        user_data;
		std::vector<uint32_t>        flattened_srt;
	};
	static thread_local std::vector<Previous> previous;
	bool same_programs  = previous.size() == stages.size();
	bool same_textures  = same_programs;
	bool same_buffers   = same_programs;
	bool same_constants = same_programs;
	previous.resize(stages.size());
	for (size_t i = 0; i < stages.size(); i++) {
		const auto& stage     = *stages[i];
		const auto& resources = *stage.resources;
		auto&       old       = previous[i];
		same_programs &= old.program == stage.program;
		same_textures &= old.images == resources.images && old.samplers == resources.samplers;
		same_buffers &= old.buffers == resources.buffers;
		same_constants &=
		    old.user_data == resources.user_data && old.flattened_srt == resources.flattened_srt;
		old.program = stage.program;
		old.buffers.assign(resources.buffers.begin(), resources.buffers.end());
		old.images.assign(resources.images.begin(), resources.images.end());
		old.samplers.assign(resources.samplers.begin(), resources.samplers.end());
		old.user_data.assign(resources.user_data.begin(), resources.user_data.end());
		old.flattened_srt.assign(resources.flattened_srt.begin(), resources.flattened_srt.end());
	}
	if (!same_programs) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawBindingRepeatPrograms);
	if (!same_textures) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawBindingRepeatTextures);
	if (!same_buffers) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawBindingRepeatResources);
	if (same_constants) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawBindingRepeatAll);
	}
}

// KYTY_CP_COMMIT_STATS (commitStats.h): what a recorded draw bound, as hashes.
namespace {

class ShapeWords {
public:
	void Add(uint64_t word) { m_words.push_back(word); }
	void AddFloat(float value) { m_words.push_back(std::bit_cast<uint32_t>(value)); }
	template <typename Handle>
	void AddHandle(Handle handle) {
		m_words.push_back(static_cast<uint64_t>(
		    reinterpret_cast<uintptr_t>(static_cast<typename Handle::CType>(handle))));
	}
	void AddBufferInfo(const vk::DescriptorBufferInfo& info) {
		AddHandle(info.buffer);
		Add(info.offset);
		Add(info.range);
	}
	void AddValue(const ShaderRecompiler::IR::DescriptorValue& value, bool mask_address) {
		Add(value.dword_count);
		for (uint32_t i = 0; i < value.dword_count && i < value.dwords.size(); i++) {
			auto word = value.dwords[i];
			if (mask_address && i == 0) {
				word = 0;
			} else if (mask_address && i == 1) {
				word &= 0xffff0000u;
			}
			Add(word);
		}
	}
	[[nodiscard]] uint64_t Finish() {
		const auto hash = CommitStats::Hash(m_words.data(), m_words.size() * sizeof(uint64_t));
		m_words.clear();
		return hash;
	}

private:
	std::vector<uint64_t> m_words;
};

} // namespace

static CommitStats::DrawShape MakeCommitShape(
    const CommandBuffer& buffer, std::span<PreparedBindings* const> stages,
    const PipelineCache::Pipeline& pipeline, const RenderState& rendering,
    const GraphicsDynamicStateShadow& dynamic, const PreparedVertexBuffers& vertex,
    const PreparedIndexBuffer& index, const DrawCallInfo& draw, const DrawEmitInfo& emit,
    const DrawIndexBufferSource& index_source, vk::PipelineStageFlags shader_write_stages) {
	static thread_local ShapeWords words;
	CommitStats::DrawShape shape;
	shape.rendering_serial = buffer.ActiveRenderingSerial();
	shape.command = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
	    static_cast<VkCommandBuffer>(buffer.Identity())));
	shape.pipeline = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
	    static_cast<VkPipeline>(pipeline.pipeline)));
	shape.shader_writes = static_cast<bool>(shader_write_stages);
	shape.pass_width    = rendering.width;
	shape.pass_height   = rendering.height;
	shape.pass_layers   = rendering.num_layers;
	shape.pass_colors   = rendering.num_color_attachments;
	shape.pass_depth    = static_cast<bool>(rendering.depth_stencil_attachment.image_view);
	// Structure: programs, texture views and layouts, samplers, attachments.
	for (const auto* stage: stages) {
		words.Add(reinterpret_cast<uintptr_t>(stage->runtime->program));
		for (const auto& image: stage->images) {
			words.AddHandle(image.image_view);
			words.Add(static_cast<uint64_t>(image.layout));
			for (const auto view: image.mip_views) {
				words.AddHandle(view);
			}
		}
		for (const auto sampler: stage->samplers) {
			words.AddHandle(sampler);
		}
	}
	for (uint32_t i = 0; i < rendering.num_color_attachments; i++) {
		words.AddHandle(rendering.color_attachments[i].image_view);
		words.Add(static_cast<uint64_t>(rendering.color_attachments[i].image_layout));
	}
	words.AddHandle(rendering.depth_stencil_attachment.image_view);
	words.Add(static_cast<uint64_t>(rendering.depth_stencil_attachment.image_layout));
	shape.structure = words.Finish();
	// Dynamic state.
	words.Add(dynamic.viewport_count);
	for (uint32_t i = 0; i < dynamic.viewport_count && i < dynamic.viewports.size(); i++) {
		const auto& v = dynamic.viewports[i];
		words.AddFloat(v.x);
		words.AddFloat(v.y);
		words.AddFloat(v.width);
		words.AddFloat(v.height);
		words.AddFloat(v.minDepth);
		words.AddFloat(v.maxDepth);
		const auto& s = dynamic.scissors[i];
		words.Add(static_cast<uint32_t>(s.offset.x));
		words.Add(static_cast<uint32_t>(s.offset.y));
		words.Add(s.extent.width);
		words.Add(s.extent.height);
	}
	words.AddFloat(dynamic.line_width);
	for (const auto value: dynamic.blend_constants) {
		words.AddFloat(value);
	}
	words.Add(dynamic.depth_test_enable);
	words.Add(dynamic.depth_write_enable);
	words.Add(static_cast<uint64_t>(dynamic.depth_compare_op));
	words.Add(dynamic.depth_bias_enable);
	for (const auto value: dynamic.depth_bias) {
		words.AddFloat(value);
	}
	words.Add(dynamic.stencil_test_enable);
	for (const auto* op: {&dynamic.stencil_front, &dynamic.stencil_back}) {
		words.Add(static_cast<uint64_t>(op->failOp));
		words.Add(static_cast<uint64_t>(op->passOp));
		words.Add(static_cast<uint64_t>(op->depthFailOp));
		words.Add(static_cast<uint64_t>(op->compareOp));
		words.Add(op->compareMask);
		words.Add(op->writeMask);
		words.Add(op->reference);
	}
	words.Add(dynamic.color_write_count);
	for (uint32_t i = 0; i < dynamic.color_write_count && i < dynamic.color_write.size(); i++) {
		words.Add(dynamic.color_write[i]);
	}
	words.Add(static_cast<uint32_t>(dynamic.feedback));
	words.Add(static_cast<uint32_t>(dynamic.cull_mode));
	words.Add(static_cast<uint64_t>(dynamic.front_face));
	words.Add(dynamic.depth_bounds_test_enable);
	words.AddFloat(dynamic.depth_bounds[0]);
	words.AddFloat(dynamic.depth_bounds[1]);
	shape.dynamic = words.Finish();
	// Per-draw data.
	for (const auto* stage: stages) {
		for (const auto& info: stage->buffers) {
			words.AddBufferInfo(info);
		}
	}
	shape.buffers = words.Finish();
	for (const auto* stage: stages) {
		for (const auto dword: stage->shader_data) {
			words.Add(dword);
		}
	}
	shape.push = words.Finish();
	for (const auto* stage: stages) {
		words.AddBufferInfo(stage->flattened_srt);
		words.AddBufferInfo(stage->shader_data_buffer);
	}
	shape.tables = words.Finish();
	words.AddHandle(index.buffer);
	words.Add(index.offset);
	words.Add(static_cast<uint64_t>(index.type));
	shape.index = words.Finish();
	for (uint32_t i = 0; i < vertex.count; i++) {
		words.AddHandle(vertex.buffers[i]);
		words.Add(vertex.offsets[i]);
		words.Add(vertex.sizes[i]);
	}
	shape.vertex = words.Finish();
	// Cross-frame keys: the complete binding inputs, and the structure alone.
	for (const auto* stage: stages) {
		const auto& resources = *stage->runtime->resources;
		words.Add(reinterpret_cast<uintptr_t>(stage->runtime->program));
		for (const auto& value: resources.buffers) {
			words.AddValue(value, false);
		}
		for (const auto& value: resources.images) {
			words.AddValue(value, false);
		}
		for (const auto& value: resources.samplers) {
			words.AddValue(value, false);
		}
		for (const auto dword: resources.flattened_srt) {
			words.Add(dword);
		}
		for (const auto dword: resources.user_data) {
			words.Add(dword);
		}
	}
	words.Add(draw.index_count);
	words.Add(draw.instance_count);
	words.Add(static_cast<uint32_t>(emit.vertex_offset));
	words.Add(emit.first_vertex);
	words.Add(emit.first_instance);
	words.Add(index_source.address);
	shape.bind_input = words.Finish();
	words.Add(shape.pipeline);
	for (uint32_t i = 0; i < rendering.num_color_attachments; i++) {
		words.AddHandle(rendering.color_attachments[i].image_view);
	}
	words.AddHandle(rendering.depth_stencil_attachment.image_view);
	for (const auto* stage: stages) {
		const auto& resources = *stage->runtime->resources;
		words.Add(reinterpret_cast<uintptr_t>(stage->runtime->program));
		for (const auto& value: resources.buffers) {
			words.AddValue(value, true);
		}
		for (const auto& value: resources.images) {
			words.AddValue(value, false);
		}
		for (const auto& value: resources.samplers) {
			words.AddValue(value, false);
		}
	}
	words.Add(draw.index_count);
	words.Add(draw.instance_count);
	shape.struct_input = words.Finish();
	return shape;
}

// Astro Bot PC (KYTY_NATIVE_MODE=s0, nativeLearn.h): the learning hook, right after the pipeline
// selection of a draw executed under a native op. The op is consumed by its own draw only (an op may
// first commit earlier pending draws, which the record check tells apart).
// Astro Bot PC (S0b, nativeLearn.h): DrawRun::StructureKey's register part (the "targets key": every
// context register, the user-config registers but the index offset and object id), computed the
// same way, for draws whose engine commit did not bring one.
static uint64_t NativeTargetsKey(const HW::Context& context, const HW::UserConfig& user_config) {
	uint64_t hash   = XXH3_64bits_withSeed(&context, sizeof(context), 0x9e3779b97f4a7c15ull);
	auto     config = user_config;
	config.SetIndexOffset(0);
	config.SetObjectId(0);
	hash = XXH3_64bits_withSeed(&config, sizeof(config), hash);
	return hash != 0 ? hash : 1;
}

// S1b (nativeLearn.h "S1b"): the context part of the NativeKey's register signature, the registers
// program and pipeline selection read that the record's state hash does not cover: the shader
// registers (PA_CL_VS_OUT_CNTL, GE/VGT group sizes, SPI_PS_*, BARYC, DB_SHADER_CONTROL,
// CB_SHADER_MASK, target output modes...), the clip control and, with CLIP_DISABLE, viewport 0's
// scale and offset (baked into the program, ApplyClipSpace), VGT_SHADER_STAGES_EN (wave size), mode
// control (provoking vertex, polygon mode, cull), AA / EQAA / scan mode, the depth bounds while
// tested, blend 0 (dual source / alpha remap), and the user-config GE registers, primitive type and
// restart control. Not the per-object registers (stencil reference, blend colour, scissors,
// viewports while clipping, clear values): those made one key per draw. One XXH3 over ~250 bytes.
static uint64_t NativeContextSelectionHash(const HW::Context& c, const HW::UserConfig& u) {
	struct Packed {
		HW::ShaderRegisters sh;
		HW::ClipControl     clip;
		float               viewport[4];
		uint32_t            shader_stages;
		HW::ModeControl     mode;
		HW::AaConfig        aa_config;
		HW::AaSampleControl aa_sample;
		HW::EqaaControl     eqaa;
		HW::ScanModeControl scan;
		float               depth_bounds[2];
		HW::BlendControl    blend0;
		HW::GeControl       ge;
		HW::GeUserVgprEn    ge_vgpr;
		uint32_t            prim_type;
		uint32_t            prim_reset;
	};
	Packed p;
	std::memset(&p, 0, sizeof(p)); // padding: a stable hash
	p.sh   = c.GetShaderRegisters();
	p.clip = c.GetClipControl();
	if (p.clip.clip_disable) {
		const auto& vp = c.GetScreenViewport().viewports[0];
		p.viewport[0]  = vp.xscale;
		p.viewport[1]  = vp.yscale;
		p.viewport[2]  = vp.xoffset;
		p.viewport[3]  = vp.yoffset;
	}
	p.shader_stages = c.GetShaderStages();
	p.mode          = c.GetModeControl();
	p.aa_config     = c.GetAaConfig();
	p.aa_sample     = c.GetAaSampleControl();
	p.eqaa          = c.GetEqaaControl();
	p.scan          = c.GetScanModeControl();
	if (c.GetDepthControl().depth_bounds_enable) {
		p.depth_bounds[0] = c.GetDepthBoundsMin();
		p.depth_bounds[1] = c.GetDepthBoundsMax();
	}
	p.blend0     = c.GetBlendControl(0);
	p.ge         = u.GetGeControl();
	p.ge_vgpr    = u.GetGeUserVgprEn();
	p.prim_type  = static_cast<uint32_t>(u.GetPrimType());
	p.prim_reset = u.GetPrimitiveResetControl();
	return XXH3_64bits_withSeed(&p, sizeof(p), 0x9e3779b97f4a7c15ull);
}

// S1b: the pixel stage's target export mapping (0 without a pixel stage).
static uint64_t NativeExportHash(const ShaderPixelInputInfo* pixel) {
	return pixel != nullptr ? XXH3_64bits(pixel->target_export_mapping.data(), sizeof(pixel->target_export_mapping)) : 0u;
}

// S0b/S1b: the NativeKey's register signature: the context part (above), the stage program
// registers (HW::Shader) and, with a pixel stage, its target export mapping.
static uint64_t NativeRegsHash(const CommandBuffer& buffer, uint64_t context_hash, uint64_t export_hash) {
	const auto& vs = buffer.GetShaders().GetVs();
	const auto& ps = buffer.GetShaders().GetPs().ps_regs;
	struct Packed {
		uint64_t                context_hash, export_hash, es, gs, ls, hs, ps;
		HW::GsShaderResource1   gs_rsrc1;
		HW::GsShaderResource2   gs_rsrc2;
		HW::PsShaderResource1   ps_rsrc1;
		HW::PsShaderResource2   ps_rsrc2;
		int32_t                 start_instance;
	};
	Packed p;
	std::memset(&p, 0, sizeof(p));
	p.context_hash   = context_hash;
	p.export_hash    = export_hash;
	p.es             = vs.es_regs.data_addr;
	p.gs             = vs.gs_regs.data_addr;
	p.ls             = vs.ls_regs.data_addr;
	p.hs             = vs.hs_regs.data_addr;
	p.ps             = ps.data_addr;
	p.gs_rsrc1       = vs.gs_regs.rsrc1;
	p.gs_rsrc2       = vs.gs_regs.rsrc2;
	p.ps_rsrc1       = ps.rsrc1;
	p.ps_rsrc2       = ps.rsrc2;
	p.start_instance = vs.start_instance_user_sgpr;
	return XXH3_64bits(&p, sizeof(p));
}

// S1b: the embedded fetch inputs of the vertex program key and of the pipeline's vertex input state
// (the S0b "vs-static" conflicts on static words 13, 16, 22, 25, 28, 31): NativeLearn::FetchHash
// (S1c: one XXH3, shared with the prediction's rebuilt fetch).
static uint64_t NativeFetchHash(const ShaderVertexInputInfo& info) {
	return NativeLearn::FetchHash(info);
}

// S1b: KYTY_NATIVE_S1_PIXEL (default 1): native emission of continuations with a pixel stage; 0
// keeps them on the normal path (reason "pixel"), the first subset's behaviour.
static bool NativeS1PixelEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_NATIVE_S1_PIXEL");
		return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// S1b: the pixel stage inputs of a prediction (the bound PS program and user SGPRs).
static void NativeS1PixelInputs(const CommandBuffer& buffer, bool pixel, NativeLearn::S1Inputs& in) {
	in.pixel = pixel;
	if (!pixel) return;
	const auto& ps        = buffer.GetShaders().GetPs();
	in.ps_addr            = ps.ps_regs.data_addr;
	in.ps_user_sgpr       = ps.ps_user_sgpr.value;
	in.ps_user_sgpr_count = ps.ps_regs.rsrc2.user_sgpr;
	in.lod_plain          = static_cast<uint32_t>(LodStatsCounter::PlainVariant());
}

void RenderExecutor::NativeS0Learn(const DrawRenderState& state, const PipelineCache::Pipeline& pipeline,
                                   const PipelineCache::GraphicsPrograms& programs, bool plain_pixel, bool mesh_active,
                                   uint32_t vertex_stage_count, bool indexed, uint32_t index_count, const CommandBuffer& buffer) {
	if (m_native_s0 == nullptr) {
		m_native_s0 = new NativeLearn::Session; // never freed (one executor)
	}
	const uint64_t cp_es_addr = buffer.GetShaders().GetVs().es_regs.data_addr;
	NativeLearn::DrawView view;
	view.op_flags           = m_native_s0_op;
	view.vertex_stage_count = vertex_stage_count;
	view.mesh_active        = mesh_active;
	view.plain_pixel        = plain_pixel;
	view.indexed            = indexed;
	view.pipeline           = &pipeline;
	view.mesh               = &state.vertex_info[0].mesh;
	view.cp_es_addr         = cp_es_addr;
	view.index_count        = index_count;
	const auto stage = [&](NativeLearn::StageView& out, const ShaderStageRuntime& runtime, const PipelineCache::StagePrep& prep,
	                       const PreparedBindings* bindings, uint64_t id) {
		out.program     = runtime.program;
		out.snapshot    = runtime.resources;
		out.plan        = PipelineCache::PermutationResourcePlan(prep);
		out.spec        = &prep.specialization;
		out.shader_data = bindings != nullptr ? &bindings->shader_data : nullptr;
		out.program_id  = id;
	};
	stage(view.stages[0], state.vertex_info[0].stage, state.stage_preps.vertex[0], &m_graphics_bindings.vertex[0], programs.vertex[0].id);
	view.stage_count = 1;
	if (state.ps_active) {
		stage(view.stages[1], state.ps_input_info.stage, state.stage_preps.pixel,
		      m_graphics_bindings.pixel.has_value() ? &*m_graphics_bindings.pixel : nullptr, programs.pixel.id);
		view.stage_count = 2;
	}
	uint64_t formats = 0xcbf29ce484222325ull;
	const auto mix   = [&](uint64_t v) { formats = (formats ^ v) * 0x100000001b3ull; };
	mix(state.color_count);
	for (uint32_t i = 0; i < state.color_count; i++) {
		mix(static_cast<uint64_t>(state.color_info[i].desc.info.guest_format));
		mix(state.color_info[i].target_slot);
	}
	mix(static_cast<bool>(state.depth_info.image_id) ? 1u : 0u);
	if (state.depth_info.image_id) mix(static_cast<uint64_t>(state.depth_info.desc.info.guest_format));
	view.formats_hash = formats;
	view.pipeline_generation = m_native_pipeline_generation;
	// S0b/S1b: the register signature (selection registers only: nativeLearn.h "S1b"), the embedded
	// fetch inputs and the permutations' identities.
	const uint64_t context_hash = NativeContextSelectionHash(buffer.GetRegisters(), buffer.GetUserConfig());
	const uint64_t export_hash  = NativeExportHash(state.ps_active ? &state.ps_input_info : nullptr);
	view.regs_hash         = NativeRegsHash(buffer, context_hash, export_hash);
	view.fetch_hash        = NativeFetchHash(state.vertex_info[0]);
	m_native_context_hash  = context_hash; // S1: the segment state of this draw (NativeS1SegmentDraw)
	m_native_export_hash   = export_hash;
	view.identity[0]       = PipelineCache::NativePermutationIdentity(state.stage_preps.vertex[0]);
	if (state.ps_active) {
		view.identity[1]     = PipelineCache::NativePermutationIdentity(state.stage_preps.pixel);
		view.plain_available = static_cast<bool>(PipelineCache::PlainPixelProgram(state.stage_preps.pixel));
		view.ps_mip_stats    = m_graphics_bindings.pixel.has_value() && m_graphics_bindings.pixel->mip_stats_active;
	}
	const auto& vs_regs    = buffer.GetShaders().GetVs();
	view.gs_user_data_addr = vs_regs.gs_regs.user_data_addr;
	view.fetch_embedded    = state.vertex_info[0].fetch_embedded;
	view.vertex_buffers    = static_cast<uint32_t>(std::max(state.vertex_info[0].buffers_num, 0));
	view.user_clip_planes  = state.vertex_info[0].user_clip_planes;
	view.vertex_info       = &state.vertex_info[0];
	// S1d: the bound user SGPR counts (the user-data prefix is the rest of the snapshot's user data)
	// and the stages' buffer writes (verify of the native post-draw barrier).
	view.user_sgpr_count[0] = vs_regs.gs_regs.rsrc2.user_sgpr;
	view.buffer_writes[0]   = HasShaderBufferWrites(state.vertex_info[0].stage);
	if (state.ps_active) {
		view.user_sgpr_count[1] = buffer.GetShaders().GetPs().ps_regs.rsrc2.user_sgpr;
		view.buffer_writes[1]   = HasShaderBufferWrites(state.ps_input_info.stage);
	}
	// S1 verify: the native prediction of this normal-path native draw (a segment head, or a draw
	// that fell back), compared by Begin with what the normal path prepared.
	// S1f: 1 in 64 such draws, and none whose DrawIndex already predicted (S1e predicted every
	// normal-path draw here: the "learn" cost of the profile, ~7 us).
	const bool s1_predicted = std::exchange(m_native_s1_predicted, false);
	if (Loader::GameCapture::NativeS1() && Loader::GameCapture::NativeVerifyEnabled() && !m_native_s0->S1Armed() &&
	    !s1_predicted && (!state.ps_active || NativeS1PixelEnabled()) &&
	    Loader::GameCapture::NativeS1SlotEnabled(m_native_s0_op) && m_native_s0->S1Sample()) {
		NativeLearn::S1Inputs in;
		in.op_flags          = m_native_s0_op;
		in.regs_hash         = view.regs_hash;
		in.formats_hash      = formats;
		in.index_count       = index_count;
		in.es_addr           = cp_es_addr;
		in.user_sgpr         = vs_regs.gs_user_sgpr.value;
		in.user_sgpr_count   = vs_regs.gs_regs.rsrc2.user_sgpr;
		in.user_sgpr_max     = HW::UserSgprInfo::SGPRS_MAX;
		in.gs_user_data_addr = vs_regs.gs_regs.user_data_addr;
		NativeS1PixelInputs(buffer, state.ps_active, in);
		if (m_native_s0->PredictS1(in) == Loader::GameCapture::kS1Emitted) {
			m_native_s0->ArmS1Check();
		}
	}
	if (m_native_s0->Begin(view)) {
		m_native_s0_op = 0; // consumed: the op's later executor calls (an MRT split pass) are not its draw
	}
}

void RenderExecutor::NativeS0Finish() {
	m_native_s0->Finish();
}

void RenderExecutor::NativeS0EndOp(uint32_t op_flags) {
	if (m_native_s0 == nullptr) {
		m_native_s0 = new NativeLearn::Session;
	}
	m_native_s0->EndOp(op_flags);
	m_native_s0_op = 0;
}

// ---- Astro Bot PC: KYTY_NATIVE_MODE=s1, native emission of continuations (nativeLearn.h "S1") -----
//
// First subset. A native op of an S1 slot (KYTY_NATIVE_S1, default the Shadow thread) that is not
// a segment head is emitted natively, from DrawIndex right after the render lock, when:
//  - segment state (NativeS1State::Segment): the previous draw executed was a draw of this native
//    segment (its slot) that completed (normal path or native), in this command buffer, tick and
//    rendering instance; nothing else ran on the command processor since (DrawRun::ActivityEpoch,
//    which needs KYTY_DRAW_RUN on) and no image state changed (ImageStateEpoch);
//  - every context and user-config register but the index offset and object id equals that draw's
//    (the DrawRun targets key): the target operations, the register checks, the targets, the
//    rendering state, the depth state and the dynamic state are that draw's;
//  - an indexed draw (16/32-bit), no primitive restart, no pixel stage, the record is the draw's;
//  - NativeLearn::Session::PredictS1 rebuilt the vertex stage (learned, not conflicted, covered
//    entry: see nativeLearn.h).
// Emission, in the normal path's order: the stage's bindings through the emulator's own binding
// functions on the rebuilt snapshot (PrepareBindings, PrepareGraphicsBindings with the targets
// kept: FindBuffers / BDA preparation / RebindBuffers through the BufferCache, i.e. the cold
// cache-buffer answers and the hot CPU-dirty copies exactly as the normal path makes them; the
// mesh index range registered for BDA reads, or the index buffer), a late check (command buffer,
// tick, rendering instance, image epoch, unregister count: any change falls back to the normal
// path, which redoes the binding work), then DrawScope, CommitBindings (push descriptors / set,
// push constants), the index buffer bind, the dynamic state when the bound pipeline is not the
// shadow's (SetGraphicsDynamicParams with the segment's depth and rendering state),
// BeginRendering with the segment's rendering state (the same instance continues; pending
// barriers are sunk or split as on the normal path; a segment whose rendering state has a clear
// falls back instead when barriers are pending), the pipeline bind (the newest object the normal
// path selected for the key) and the draw (mesh dispatches split as the normal path splits them,
// or the indexed draw). Skipped (pure recomputation for such a draw): program preparation and its
// certificate (Validate), target resolution and acquisition, the pipeline lookup, the barrier
// safety (false: pending barriers are flushed), the DrawRun bookkeeping and statistics.
// After it the DrawRun record and kept records are dropped (BeginDrawRun), so the next normal draw
// resolves its structure in full.
struct RenderExecutor::NativeS1State {
	struct Segment {
		bool              valid   = false;
		bool              pending = false; // set at the draw command, completed at the end of DrawIndex
		uint32_t          slot    = 0;
		vk::CommandBuffer command = nullptr;
		uint64_t          tick = 0, rendering_serial = 0, activity = 0, image_epoch = 0, draw_counter = 0;
		uint64_t          targets_key = 0, formats_hash = 0;
		uint64_t          context_hash = 0, export_hash = 0; // S1b: the selection signature's parts
		RenderState       rendering;
		RenderDepthInfo   depth;
		uint32_t          color_count    = 0;
		std::array<ImageId, RENDER_COLOR_ATTACHMENTS_MAX> color_ids {}; // S1b: attachments (feedback check)
		bool              has_clear      = false;
		bool              viewport_index = false;
		uint64_t          relaxed_key    = 0; // S1d: NativeS1RelaxedKey of the segment's registers (0: not computed)
	};
	Segment               seg;
	PreparedBindings      bindings;
	ShaderStageRuntime    runtime;
	PreparedBindings      ps_bindings; // S1b: the pixel stage (its texture sets persist, RepeatStageTextures)
	ShaderStageRuntime    ps_runtime;
	ShaderVertexInputInfo vs_info;
	// S1c (KYTY_NATIVE_S1_REGS_DIAG=1): the registers of the segment's last normal draw.
	std::unique_ptr<HW::Context>    diag_context;
	std::unique_ptr<HW::UserConfig> diag_config;
};

// S1c: KYTY_NATIVE_S1_REGS_DIAG=1 (diagnostic): the segment keeps a copy of its last normal draw's
// registers, and a "regs" fallback reports the register groups that differ ("s1 regs parts").
static bool NativeS1RegsDiag() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_NATIVE_S1_REGS_DIAG");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

// S1c: the register groups (Loader::GameCapture::NativeS1RegPart bits) in which two register states
// differ, the index offset and object id excluded (as the targets key).
static uint32_t NativeS1RegParts(const HW::Context& a, const HW::UserConfig& ua, const HW::Context& b, const HW::UserConfig& ub) {
	namespace GC   = Loader::GameCapture;
	uint32_t   parts = 0;
	const auto base  = reinterpret_cast<const char*>(&a);
	const auto other = reinterpret_cast<const char*>(&b);
	a.VisitKeyDiagParts([&](uint32_t group, const void* data, size_t size) {
		const auto offset = static_cast<const char*>(data) - base;
		if (std::memcmp(data, other + offset, size) != 0) parts |= 1u << group;
	});
	if (a.GetUserClipPlanes() != b.GetUserClipPlanes()) parts |= 1u << GC::kRpUserClip;
	auto ca = ua;
	auto cb = ub;
	ca.SetIndexOffset(0);
	ca.SetObjectId(0);
	cb.SetIndexOffset(0);
	cb.SetObjectId(0);
	if (std::memcmp(&ca, &cb, sizeof(ca)) != 0) parts |= 1u << GC::kRpUserConfig;
	if (parts == 0 && std::memcmp(&a, &b, sizeof(a)) != 0) parts |= 1u << GC::kRpOther;
	return parts;
}

// S1d: KYTY_NATIVE_S1_REGS_RELAX (default 1). A continuation whose registers differ from the
// segment's last draw only in
//  - the shader registers (m_sh_regs) and VGT_SHADER_STAGES_EN, but the target output modes and
//    CB_SHADER_MASK (the export mapping, DrawColorOutputFilter): every one of them is part of the
//    selection signature (NativeContextSelectionHash), recomputed for the draw, so the NativeKey,
//    i.e. the program permutations and the pipeline, follows them;
//  - the line width, the primitive reset index (ResolvePrimitiveRestart reads the draw's), the
//    mode control (cull mode, front face, polygon offset enables: dynamic state; the rest is in the
//    selection signature) and the polygon offset (dynamic depth bias), and the blend colour
//    (dynamic blend constants): SetGraphicsDynamicParams records them from the draw's registers,
//    the dynamic state is then always set (its shadow skips equal values),
// is emitted natively. Every other register (targets, depth / stencil state: RenderDepthInfo and
// the rendering state; viewports and scan mode: the written rectangle; AA / EQAA; user config;
// user clip planes) must still equal the segment's.
static bool NativeS1RegsRelax() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_NATIVE_S1_REGS_RELAX");
		return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// S1e: KYTY_NATIVE_S1_KEEP_RUN (default 1). A native emission leaves the DrawRun records as they
// are (S1d dropped them, so the next normal draw resolved its structure in full: DrawRun
// "started"). What a continuation of the record reuses is not changed by a successful emission:
//  - targets and rendering instance: the emission continues the segment's (= the record's)
//    rendering state; a change of instance, image state or command buffer fails the record's own
//    checks (ActiveRenderingSerial, ImageStateEpoch through DrawRunImagesUnchanged, Identity, tick);
//  - textures and samplers: the emission binds through its own PreparedBindings (NativeS1State),
//    m_graphics_bindings keep the record's sets;
//  - descriptors: its push / set bind moves the command buffer's DescriptorEpoch, so a partial push
//    (DrawRunPartialPush) is refused and the continuation pushes in full;
//  - pipeline, vertex / index buffers, push constants: always bound again by the normal path
//    (through their shadows);
//  - dynamic state: the emission may record its own (relaxed registers, another vertex program's
//    viewport-index output...) and binds another pipeline: m_native_s1_dynamic_dirty, the next
//    continuation records its own (SetGraphicsDynamicParams through the shadow: only differences
//    are emitted) instead of skipping it.
// KYTY_DRAW_RUN=verify checks continuations against the normal path. 0: S1d's behaviour.
static bool NativeS1KeepRun() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_NATIVE_S1_KEEP_RUN");
		return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// S1d: the context byte ranges NativeS1RelaxedKey leaves out, found once through
// VisitKeyDiagParts (raster members by their order: line width, reset index, scan mode, AA sample,
// AA config, mode control, polygon offset, EQAA; shader group: stages, shader registers); each
// size is checked against its type (a mismatch disables the relaxation).
struct NativeS1Excluded {
	bool                                       valid = false;
	std::vector<std::pair<size_t, size_t>>     ranges; // sorted [offset, offset + size)
	std::array<std::pair<size_t, size_t>, 8>   raster {}; // S1d diag: the raster members
	size_t                                     sh_offset = 0;
};
static const NativeS1Excluded& NativeS1ExcludedRanges() {
	static const NativeS1Excluded excluded = [] {
		NativeS1Excluded e;
		const auto       context = std::make_unique<HW::Context>();
		const auto       base    = reinterpret_cast<const char*>(context.get());
		uint32_t         raster = 0, sh = 0;
		bool             ok     = true;
		enum : uint32_t { BlendColor = 4, Raster = 9, ShRegs = 10 };
		context->VisitKeyDiagParts([&](uint32_t group, const void* data, size_t size) {
			const size_t offset = static_cast<const char*>(data) - base;
			if (group == BlendColor) {
				ok = ok && size == sizeof(HW::BlendColor);
				e.ranges.emplace_back(offset, offset + size);
			} else if (group == Raster) {
				if (raster < e.raster.size()) e.raster[raster] = {offset, size};
				static constexpr size_t sizes[8] = {sizeof(float), sizeof(uint32_t), sizeof(HW::ScanModeControl), sizeof(HW::AaSampleControl),
				                                    sizeof(HW::AaConfig), sizeof(HW::ModeControl), sizeof(HW::PolyOffset), sizeof(HW::EqaaControl)};
				ok = ok && raster < 8 && size == sizes[raster];
				if (raster == 0 || raster == 1 || raster == 5 || raster == 6) e.ranges.emplace_back(offset, offset + size);
				raster++;
			} else if (group == ShRegs) {
				ok = ok && sh < 2 && size == (sh == 0 ? sizeof(uint32_t) : sizeof(HW::ShaderRegisters));
				if (sh == 1) e.sh_offset = offset;
				e.ranges.emplace_back(offset, offset + size);
				sh++;
			}
		});
		ok = ok && raster == 8 && sh == 2;
		std::sort(e.ranges.begin(), e.ranges.end());
		for (size_t i = 1; ok && i < e.ranges.size(); i++) ok = e.ranges[i - 1].second <= e.ranges[i].first;
		e.valid = ok;
		return e;
	}();
	return excluded;
}

// S1d: the targets key (NativeTargetsKey) without the relaxed registers, with the export part of
// the shader registers (target output modes, CB_SHADER_MASK) kept. 0: relaxation unavailable.
static uint64_t NativeS1RelaxedKey(const HW::Context& context, const HW::UserConfig& user_config) {
	const auto& ex = NativeS1ExcludedRanges();
	if (!ex.valid) return 0;
	const auto base = reinterpret_cast<const char*>(&context);
	uint64_t   hash = 0x9e3779b97f4a7c15ull;
	size_t     at   = 0;
	for (const auto& [begin, end]: ex.ranges) {
		if (begin > at) hash = XXH3_64bits_withSeed(base + at, begin - at, hash);
		at = end;
	}
	if (sizeof(context) > at) hash = XXH3_64bits_withSeed(base + at, sizeof(context) - at, hash);
	const auto& sh = context.GetShaderRegisters();
	hash           = XXH3_64bits_withSeed(sh.target_output_mode, sizeof(sh.target_output_mode), hash);
	hash           = XXH3_64bits_withSeed(&sh.m_cbShaderMask, sizeof(sh.m_cbShaderMask), hash);
	auto config    = user_config;
	config.SetIndexOffset(0);
	config.SetObjectId(0);
	hash = XXH3_64bits_withSeed(&config, sizeof(config), hash);
	return hash != 0 ? hash : 1;
}

// S1d: the finer register parts of a "regs" fallback (KYTY_NATIVE_S1_REGS_DIAG=1): which raster
// member, and whether the shader stages, the export part or the rest of the shader registers differ.
static uint32_t NativeS1RegSubParts(const HW::Context& a, const HW::Context& b) {
	namespace GC   = Loader::GameCapture;
	const auto& ex = NativeS1ExcludedRanges();
	if (!ex.valid) return 0;
	const auto x     = reinterpret_cast<const char*>(&a);
	const auto y     = reinterpret_cast<const char*>(&b);
	uint32_t   parts = 0;
	for (uint32_t i = 0; i < 8; i++) {
		const auto [offset, size] = ex.raster[i];
		if (std::memcmp(x + offset, y + offset, size) != 0) parts |= 1u << (GC::kRpRasterLine + i);
	}
	if (a.GetShaderStages() != b.GetShaderStages()) parts |= 1u << GC::kRpShStages;
	const auto& sa = a.GetShaderRegisters();
	const auto& sb = b.GetShaderRegisters();
	if (std::memcmp(sa.target_output_mode, sb.target_output_mode, sizeof(sa.target_output_mode)) != 0 ||
	    sa.m_cbShaderMask != sb.m_cbShaderMask) {
		parts |= 1u << GC::kRpShExport;
	}
	auto ca = sa;
	auto cb = sb;
	std::memset(ca.target_output_mode, 0, sizeof(ca.target_output_mode));
	std::memset(cb.target_output_mode, 0, sizeof(cb.target_output_mode));
	ca.m_cbShaderMask = cb.m_cbShaderMask = 0;
	if (std::memcmp(&ca, &cb, sizeof(ca)) != 0) parts |= 1u << GC::kRpShOther;
	return parts;
}

void RenderExecutor::NativeS1BreakSegment() {
	if (m_native_s1 != nullptr) {
		m_native_s1->seg.valid   = false;
		m_native_s1->seg.pending = false;
	}
}

void RenderExecutor::NativeS1SegmentDraw(const CommandBuffer& buffer, const DrawRenderState& state,
                                         const RenderState& rendering) {
	namespace GC = Loader::GameCapture;
	if (m_native_s1 == nullptr) {
		m_native_s1 = new NativeS1State; // never freed (one executor)
	}
	auto& seg            = m_native_s1->seg;
	seg.valid            = false;
	seg.pending          = true;
	seg.slot             = (m_native_s0->ActiveOp() >> GC::kNativeOpSlotShift) & GC::kNativeSlotMask;
	seg.command          = buffer.Identity();
	seg.tick             = m_context.GetCommandScheduler().CurrentTick();
	seg.rendering_serial = buffer.ActiveRenderingSerial();
	seg.rendering        = rendering;
	seg.depth            = state.depth_info;
	seg.color_count      = state.color_count;
	seg.color_ids        = {};
	for (uint32_t i = 0; i < state.color_count && i < seg.color_ids.size(); i++) {
		seg.color_ids[i] = state.color_info[i].image_id;
	}
	seg.context_hash     = m_native_context_hash;
	seg.export_hash      = m_native_export_hash;
	seg.has_clear        = rendering.depth_stencil_attachment.depth_clear || rendering.depth_stencil_attachment.stencil_clear;
	for (uint32_t i = 0; i < rendering.num_color_attachments; i++) {
		seg.has_clear = seg.has_clear || rendering.color_attachments[i].is_clear;
	}
	seg.targets_key  = m_run_targets_key != 0 ? m_run_targets_key : NativeTargetsKey(buffer.GetRegisters(), buffer.GetUserConfig());
	seg.formats_hash = m_native_s0->ActiveFormats();
	seg.relaxed_key  = NativeS1RegsRelax() ? NativeS1RelaxedKey(buffer.GetRegisters(), buffer.GetUserConfig()) : 0;
	if (NativeS1RegsDiag()) [[unlikely]] {
		if (m_native_s1->diag_context == nullptr) {
			m_native_s1->diag_context = std::make_unique<HW::Context>();
			m_native_s1->diag_config  = std::make_unique<HW::UserConfig>();
		}
		*m_native_s1->diag_context = buffer.GetRegisters();
		*m_native_s1->diag_config  = buffer.GetUserConfig();
	}
	const auto& outputs = state.vertex_info[0].stage.program->info.outputs;
	seg.viewport_index  = std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
        return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
    });
}

void RenderExecutor::NativeS1SegmentEnd(const CommandBuffer& buffer, bool valid) {
	auto& seg = m_native_s1->seg;
	if (!seg.pending) {
		return;
	}
	seg.pending = false;
	// The draw's own later work (write barriers, run record) is part of what the next one sees.
	seg.valid = valid && !buffer.IsInvalid() && buffer.Identity() == seg.command &&
	            m_context.GetCommandScheduler().CurrentTick() == seg.tick && buffer.ActiveRenderingSerial() != 0;
	seg.rendering_serial = buffer.ActiveRenderingSerial();
	seg.activity         = DrawRun::ActivityEpoch();
	seg.image_epoch      = ImageStateEpoch();
	seg.draw_counter     = m_native_draw_counter;
}

// S1f: see render.h. The emission's binding work (PrepareBindings, PrepareGraphicsBindings, vertex
// and index acquisition) records no state command, but it marks images bound (BindImage), may
// record uploads and barriers, and leaves its own PreparedBindings; nothing of it may be taken by
// the normal path for its own. Rare (feedback, late, plain-choice mismatch): the cost of a full
// resolution of the normal draw is accepted.
void RenderExecutor::NativeS1Abandon(CommandBuffer& buffer) {
	ResetBindings();
	buffer.InvalidatePushConstants();
	buffer.InvalidateDescriptors(vk::PipelineBindPoint::eGraphics);
	m_dynamic_state.valid     = false;
	m_native_s1_dynamic_dirty = true;
	m_run.valid               = false; // BeginDrawRun: no continuation of the record
	m_run_prev_valid          = false;
	m_run_keepable            = false;
	for (auto& record: m_run_kept) {
		record.valid = false;
	}
}

// ---- S1g: KYTY_NATIVE_S1_SAFE=1 (diagnostic safety net, default off) -------------------------------
// Before a native emission records anything, every descriptor and binding it is about to write is
// validated: the pipeline and its layout (the learned entry's), each stage's program (the entry's),
// buffer descriptors (non-null handle, non-zero range, the cache buffer the lookup named still live
// and the range inside it), textures (live image, non-null views), samplers, the shader-data /
// flattened-SRT tables, the mip-statistics counters, the BDA page table (and, for a mesh draw, the
// index range registered for BDA reads), the vertex buffers and the index buffer. Any failure falls
// back to the normal path (reason "unsafe", NativeS1Abandon: nothing was recorded) and is counted by
// cause in "GameNative s1 safe" every 10 s; the first 32 are logged with their stage and index.
static bool NativeS1SafeEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_NATIVE_S1_SAFE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

enum NativeS1SafeReason : uint32_t {
	kSafeOk,
	kSafePipeline,    // null pipeline / layout, or a layout other than the learned entry's
	kSafeProgram,     // a stage's prepared program is not the entry's (or no runtime)
	kSafeCounts,      // binding vectors do not match the program's resources
	kSafeBufferNull,  // a buffer descriptor with a null handle or a zero range
	kSafeBufferDead,  // the cache buffer a binding resolved to was unregistered
	kSafeBufferRange, // offset + range past the end of the cache buffer
	kSafeImage,       // a texture's image is not in the texture cache
	kSafeView,        // a null image view (or mip view)
	kSafeSampler,     // a null sampler
	kSafeTable,       // shader data / flattened SRT / mip-statistics buffer missing, or shader data size
	kSafeBda,         // BDA page table missing, or the mesh index range not registered
	kSafeVertex,      // a vertex buffer binding with a null handle
	kSafeIndex,       // no index buffer for an indexed draw
	kSafeCount
};
static constexpr const char* kNativeS1SafeNames[kSafeCount] = {
    "ok", "pipeline", "program", "counts", "buffer-null", "buffer-dead", "buffer-range",
    "image", "view", "sampler", "table", "bda", "vertex", "index"};

struct NativeS1SafeTotals {
	uint64_t                              checks = 0;
	std::array<uint64_t, kSafeCount>      fails {};
	std::chrono::steady_clock::time_point since {};
	uint32_t                              logged = 0;
};

// GPU thread only (under the render lock): plain counters.
static NativeS1SafeTotals& NativeS1Safe() {
	static NativeS1SafeTotals totals;
	return totals;
}

static void NativeS1SafeReport(NativeS1SafeTotals& t) {
	const auto now = std::chrono::steady_clock::now();
	if (t.since == std::chrono::steady_clock::time_point {}) {
		t.since = now;
		return;
	}
	const double elapsed = std::chrono::duration<double>(now - t.since).count();
	if (elapsed < 10.0) {
		return;
	}
	uint64_t    failed = 0;
	std::string causes;
	for (uint32_t k = 1; k < kSafeCount; k++) {
		failed += t.fails[k];
		if (t.fails[k] != 0) causes += fmt::format(" {} {}", kNativeS1SafeNames[k], t.fails[k]);
	}
	std::printf("GameNative s1 safe (%.1f s, KYTY_NATIVE_S1_SAFE=1): %" PRIu64 " emissions checked, %" PRIu64
	            " passed, %" PRIu64 " fell back:%s\n",
	            elapsed, t.checks, t.checks - failed, failed, causes.empty() ? " none" : causes.c_str());
	std::fflush(stdout);
	t.checks = 0;
	t.fails  = {};
	t.since  = now;
}

struct NativeS1SafeDetail {
	uint32_t stage = 0;
	uint32_t index = 0;
	uint64_t value = 0;
};

static uint32_t NativeS1Validate(RenderContext& context, const PipelineCache::Pipeline& pipeline,
                                 const NativeLearn::Entry& entry, std::span<PreparedBindings* const> stages, bool mesh,
                                 const PreparedVertexBuffers& vertex, const PreparedIndexBuffer& index,
                                 const DrawIndexBufferSource& index_source, NativeS1SafeDetail& detail) {
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	if (pipeline.pipeline == nullptr || pipeline.pipeline_layout == nullptr ||
	    (entry.layout != nullptr && entry.layout != pipeline.pipeline_layout)) {
		return kSafePipeline;
	}
	auto&      cache          = context.GetBufferCache();
	auto&      textures       = context.GetTextureCache();
	const auto valid_view     = [](const vk::DescriptorBufferInfo& view) { return view.buffer != nullptr && view.range != 0; };
	bool       has_descriptor = false;
	bool       uses_dma       = false;
	for (uint32_t s = 0; s < stages.size(); s++) {
		detail.stage         = s;
		const auto* prepared = stages[s];
		const auto* expected = s == 0 ? entry.vs : entry.ps;
		if (prepared == nullptr || prepared->runtime == nullptr || prepared->runtime->program == nullptr ||
		    prepared->runtime->program != expected || prepared->runtime->resources == nullptr) {
			return kSafeProgram;
		}
		const auto& program = *prepared->runtime->program;
		uses_dma |= program.info.uses_dma;
		has_descriptor |= !program.bindings.descriptors.empty();
		if (prepared->buffers.size() != program.info.buffers.size() ||
		    prepared->buffer_sources.size() != program.info.buffers.size() ||
		    prepared->images.size() != program.info.images.size() ||
		    prepared->samplers.size() != program.info.samplers.size()) {
			return kSafeCounts;
		}
		if (prepared->shader_data.size() != program.bindings.ShaderDataDwords()) {
			return kSafeTable;
		}
		for (uint32_t i = 0; i < prepared->buffers.size(); i++) {
			detail.index     = i;
			const auto& view = prepared->buffers[i];
			const auto& src  = prepared->buffer_sources[i];
			detail.value     = src.address;
			if (!valid_view(view)) return kSafeBufferNull;
			if (src.address == 0 || src.size == 0) continue; // the null buffer (NativeStorageBuffer)
			// The cache buffer the lookup named: when the descriptor is that buffer, it must be live
			// and the range inside it (a merged or stream answer has another handle).
			if (const auto* named = cache.TryGetBuffer(src.id); named != nullptr && named->Handle() == view.buffer) {
				if (named->is_deleted) return kSafeBufferDead;
				if (view.range != VK_WHOLE_SIZE && (view.offset > named->Size() || view.range > named->Size() - view.offset)) {
					detail.value = view.offset + view.range;
					return kSafeBufferRange;
				}
			}
		}
		for (uint32_t i = 0; i < prepared->images.size(); i++) {
			detail.index        = i;
			const auto& binding = prepared->images[i];
			if (!binding.image_id || textures.TryGetImage(binding.image_id) == nullptr) return kSafeImage;
			if (binding.mip_views.empty() ? binding.image_view == nullptr
			                              : std::any_of(binding.mip_views.begin(), binding.mip_views.end(),
			                                            [](vk::ImageView v) { return v == nullptr; })) {
				return kSafeView;
			}
		}
		for (uint32_t i = 0; i < prepared->samplers.size(); i++) {
			detail.index = i;
			if (prepared->samplers[i] == nullptr) return kSafeSampler;
		}
		for (const auto& binding: program.bindings.descriptors) {
			detail.index = static_cast<uint32_t>(binding.kind);
			switch (binding.kind) {
				case Kind::FlattenedSrt:
					if (!valid_view(prepared->flattened_srt)) return kSafeTable;
					break;
				case Kind::ShaderData:
					if (!valid_view(prepared->shader_data_buffer)) return kSafeTable;
					break;
				case Kind::MipStats: {
					auto& lod = context.GetLodStats();
					if ((prepared->mip_stats_canary ? lod.CanaryBuffer() : lod.CounterBuffer()).Handle() == nullptr) return kSafeTable;
					break;
				}
				case Kind::BdaPagetable: {
					const auto* table = cache.GetBdaPageTableBuffer();
					if (table == nullptr || table->Handle() == nullptr) return kSafeBda;
					break;
				}
				case Kind::FaultBuffer: {
					const auto* fault = cache.GetFaultBuffer();
					if (fault == nullptr || fault->Handle() == nullptr) return kSafeBda;
					break;
				}
				default: break;
			}
		}
	}
	detail.stage = 0;
	detail.index = 0;
	if (has_descriptor && pipeline.descriptor_set_layout == nullptr) return kSafePipeline;
	detail.value = index_source.address;
	if (mesh) {
		// The mesh program reads the guest indices by address: their pages must be registered.
		if (uses_dma && index_source.size != 0 && !cache.IsRegionRegistered(index_source.address, index_source.size)) {
			return kSafeBda;
		}
	} else {
		for (uint32_t i = 0; i < vertex.count; i++) {
			detail.index = i;
			if (vertex.buffers[i] == nullptr) return kSafeVertex;
		}
		if (index.buffer == nullptr) return kSafeIndex;
	}
	return kSafeOk;
}

bool RenderExecutor::NativeS1Emit(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args) {
	namespace GC  = Loader::GameCapture;
	namespace NP  = NativeLearn::Prof;
	const auto op = m_native_s0_op;
	GC::NativeS1Draw note;
	bool             predicted = false; // S1d: PredictS1 ran (its census applies)
	NP::Marker       mark;              // S1e (KYTY_NATIVE_S1_PROFILE)
	bool             predicted_ok = false; // S1f: PredictS1 returned kS1Emitted (Prediction() is this draw's)
	const auto out = [&](uint32_t reason) {
		note.reason = reason;
		// S1f (verify): a prediction made for this draw and not emitted is compared by the normal
		// path's Begin (no second prediction there); not for a plain-choice mismatch (the normal path
		// runs the other pixel program: counted as "plain", not compared).
		if (predicted_ok && reason != GC::kS1Sample && reason != GC::kS1Plain && GC::NativeVerifyEnabled()) {
			m_native_s0->ArmS1Check();
		}
		if (reason == GC::kS1Shape && m_native_s0 != nullptr) {
			note.shape       = m_native_s0->S1ShapeCode();
			note.shape_stage = m_native_s0->S1ShapeStage();
		}
		if (predicted) {
			note.census      = m_native_s0->S1Census();
			note.shape_mask  = m_native_s0->S1Mask();
			note.write_kinds = m_native_s0->S1WriteKinds();
		}
		GC::NoteNativeS1(op, note);
		return false;
	};
	// S1f: a fallback once the binding work began.
	const auto abandon = [&](uint32_t reason) {
		NativeS1Abandon(buffer);
		return out(reason);
	};
	if (!GC::NativeS1SlotEnabled(op)) return out(GC::kS1Slot);
	if ((op & GC::kNativeOpSegStart) != 0) return out(GC::kS1SegStart);
	if (m_native_s0 == nullptr) {
		m_native_s0 = new NativeLearn::Session;
	}
	if (m_native_s1 == nullptr) {
		m_native_s1 = new NativeS1State;
	}
	auto&          st        = *m_native_s1;
	auto&          seg       = st.seg;
	auto&          scheduler = m_context.GetCommandScheduler();
	const uint32_t slot      = (op >> GC::kNativeOpSlotShift) & GC::kNativeSlotMask;
	if (!DrawRun::Enabled()) return out(GC::kS1Activity);
	if (!seg.valid || seg.pending || seg.slot != slot) return out(GC::kS1NoSeg);
	if (seg.draw_counter + 1u != m_native_draw_counter || DrawRun::ActivityEpoch() != seg.activity ||
	    ImageStateEpoch() != seg.image_epoch) {
		return out(GC::kS1Activity);
	}
	if (buffer.Identity() != seg.command || scheduler.CurrentTick() != seg.tick || buffer.ActiveRenderingSerial() == 0 ||
	    buffer.ActiveRenderingSerial() != seg.rendering_serial) {
		return out(GC::kS1Command);
	}
	if (args.index_count == 0 || args.instance_count == 0 || args.offset_source != DrawOffsetSource::DrawState ||
	    args.index_addr == nullptr) {
		return out(GC::kS1Kind);
	}
	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_source.type               = vk::IndexType::eUint16;
			index_source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			index_source.type               = vk::IndexType::eUint32;
			index_source.guest_element_size = 4;
			break;
		default: return out(GC::kS1Kind);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	auto& ucfg = buffer.GetUserConfig();
	if (!DrawHasValidVertexShader(buffer.GetShaders())) return out(GC::kS1Kind);
	const bool pixel = DrawHasActivePixelShader(buffer);
	note.pixel       = pixel;
	if (pixel && !NativeS1PixelEnabled()) return out(GC::kS1Pixel);
	const uint64_t targets_key = m_run_targets_key != 0 ? m_run_targets_key : NativeTargetsKey(buffer.GetRegisters(), ucfg);
	// S1d: registers that differ in the relaxed groups only (NativeS1RegsRelax).
	bool relaxed = false;
	if (targets_key != seg.targets_key) {
		relaxed = seg.relaxed_key != 0 && NativeS1RelaxedKey(buffer.GetRegisters(), ucfg) == seg.relaxed_key;
		if (!relaxed) {
			if (NativeS1RegsDiag() && st.diag_context != nullptr) [[unlikely]] {
				note.regs_parts = NativeS1RegParts(buffer.GetRegisters(), ucfg, *st.diag_context, *st.diag_config) |
				                  NativeS1RegSubParts(buffer.GetRegisters(), *st.diag_context);
			}
			return out(GC::kS1Regs);
		}
		note.relaxed = true;
	}
	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) return out(GC::kS1Kind);
	// S1f: known before the prediction (checked again after the binding work).
	if (seg.has_clear && buffer.HasPendingBarriers()) return out(GC::kS1Barriers);
	// The vertex stage, rebuilt.
	const auto&          vs_regs = buffer.GetShaders().GetVs();
	NativeLearn::S1Inputs in;
	in.op_flags          = op;
	// S1b: the registers equal the segment's last normal draw's (targets key): its selection
	// signature's context part and export mapping are this draw's. S1d: relaxed, the context part is
	// recomputed from the draw's registers (the export mapping's registers are equal).
	const uint64_t context_hash = relaxed ? NativeContextSelectionHash(buffer.GetRegisters(), ucfg) : seg.context_hash;
	in.regs_hash         = NativeRegsHash(buffer, context_hash, pixel ? seg.export_hash : 0u);
	in.formats_hash      = seg.formats_hash;
	in.index_count       = args.index_count;
	in.es_addr           = vs_regs.es_regs.data_addr;
	in.user_sgpr         = vs_regs.gs_user_sgpr.value;
	in.user_sgpr_count   = vs_regs.gs_regs.rsrc2.user_sgpr;
	in.user_sgpr_max     = HW::UserSgprInfo::SGPRS_MAX;
	in.gs_user_data_addr = vs_regs.gs_regs.user_data_addr;
	NativeS1PixelInputs(buffer, pixel, in);
	predicted = true;
	mark(NP::kChecks);
	const auto predict_reason = m_native_s0->PredictS1(in);
	mark(NP::kPredict);
	m_native_s1_predicted = true;
	if (predict_reason != GC::kS1Emitted) return out(predict_reason);
	predicted_ok = true;
	const auto& pred  = m_native_s0->Prediction();
	const auto& entry = *pred.entry;
	if (GC::NativeVerifyEnabled() && m_native_s0->S1Sample()) {
		m_native_s0->ArmS1Check(); // the normal path's Begin compares the prediction
		return out(GC::kS1Sample);
	}
	const bool mesh = entry.mesh_active;
	if (mesh != (entry.vs->stage == ShaderType::Mesh) || entry.current == nullptr) {
		note.reason      = GC::kS1Shape;
		note.shape       = GC::kShPipeline;
		note.shape_stage = 0;
		note.census      = true;
		note.shape_mask  = GC::S1ShapeBit(GC::kShPipeline, 0);
		note.write_kinds = m_native_s0->S1WriteKinds();
		GC::NoteNativeS1(op, note);
		return false;
	}
	// A pipeline replaced since the normal path last selected the key's (its old handle may be
	// destroyed): the normal path selects it again.
	if (entry.current_generation != m_context.GetPipelineCache().PipelineGeneration()) return out(GC::kS1Miss);
	if (ResolvePrimitiveRestart(buffer, index_source, mesh && MeshRestartEnabled())) return out(GC::kS1Kind);
	const auto& limits      = m_context.GetGraphics().mesh_shader_properties;
	uint32_t    mesh_groups = 0;
	if (mesh) {
		const auto primitives = entry.mesh.InputPrimitiveCount(args.index_count);
		if (primitives == 0 || entry.mesh.primitives_per_group == 0) return out(GC::kS1Kind);
		mesh_groups = (primitives - 1u) / entry.mesh.primitives_per_group + 1u;
		if (MeshInstancesPerDispatch(MeshGroupsPerDispatch(mesh_groups, entry.mesh, limits), limits) == 0) {
			return out(GC::kS1Kind);
		}
	}
	if (seg.has_clear && buffer.HasPendingBarriers()) return out(GC::kS1Barriers);
	const auto& pipeline = *entry.current;
	mark(NP::kEPre);
	// Bindings: the emulator's own binding work on the rebuilt snapshots (textures and samplers:
	// RepeatStageTextures / TextureBindingMemo / ResolveTexture and NativeSampler, as the normal
	// path resolves them).
	st.runtime = {entry.vs, &pred.snapshot[0]};
	PrepareBindings(st.runtime, st.bindings, nullptr, false);
	mark(NP::kEBindVs);
	std::array<PreparedBindings*, 2> stage_array {&st.bindings, &st.ps_bindings};
	const uint32_t                   stage_count = pixel ? 2u : 1u;
	if (pixel) {
		st.ps_runtime = {entry.ps, &pred.snapshot[1]};
		PrepareBindings(st.ps_runtime, st.ps_bindings, nullptr, false);
		// S1f: the plain-variant check needs mip_stats_active, which RebindBuffers (in
		// PrepareGraphicsBindings, below) computes: S1e read the previous emission's value here.
	}
	bool has_images = false;
	for (uint32_t s = 0; s < stage_count; s++) {
		for (const auto& binding: stage_array[s]->images) {
			has_images = true;
			// A texture that is one of the segment's attachments: the normal path's feedback and
			// attachment layout handling (AcquireRenderTargets) decides that draw.
			const bool attachment = (seg.depth.image_id && binding.image_id == seg.depth.image_id) ||
			                        std::any_of(seg.color_ids.begin(), seg.color_ids.begin() + std::min<uint32_t>(seg.color_count, RENDER_COLOR_ATTACHMENTS_MAX),
			                                    [&](const ImageId& id) { return id && id == binding.image_id; });
			if (attachment) return abandon(GC::kS1Feedback);
		}
	}
	if (mesh) {
		// The original guest indices, registered for shader reads (synchronized by the BDA
		// preparation below), as the normal path does for a mesh draw.
		(void)m_context.GetBufferCache().FindBuffer(index_source.address, index_source.size);
	}
	const auto stages = std::span<PreparedBindings* const> {stage_array.data(), stage_count};
	mark(NP::kEBindPs);
	// Texture views (RebindImages) when a stage samples; the targets are the segment's.
	PrepareGraphicsBindings(stages, {}, !has_images);
	mark(NP::kEGfx);
	// S1f: the plain pixel variant is the normal path's choice exactly when it is on, the program
	// has one and no pixel T# records mip statistics (ExecutePreparedDraw); the prediction decided it
	// from the rebuilt T#s (BuildS1), RebindBuffers has just computed it from the same snapshot.
	if (pixel) {
		const bool plain_expected = in.lod_plain == 1u && entry.plain_available && !st.ps_bindings.mip_stats_active;
		if (plain_expected != pred.plain || st.ps_bindings.mip_stats_canary) return abandon(GC::kS1Plain);
	}
	// S1c: the embedded fetch's vertex buffers (AcquireVertexBuffers on the rebuilt input info, as
	// the normal path acquires them after the stage bindings), then the index buffer.
	auto&                 pred_mut = m_native_s0->MutablePrediction();
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	if (!mesh) {
		if (pred_mut.fetch) {
			pred_mut.vertex.stage = st.runtime;
			if (CpCommit::Enabled(CpCommit::Part::Draws)) {
				AcquireVertexBuffersInto(buffer, pred_mut.vertex, nullptr, vertex_bindings);
			} else {
				vertex_bindings = AcquireVertexBuffers(buffer, pred_mut.vertex, nullptr);
			}
		}
		index_binding = PrepareIndexBuffer(buffer, index_source);
	}
	// S1d: the stages' buffer writes, as the normal path derives them (HasShaderBufferWrites on the
	// stage's program and snapshot), for the post-draw ShaderWriteBarrier.
	vk::PipelineStageFlags shader_write_stages = {};
	if (HasShaderBufferWrites(st.runtime)) shader_write_stages |= ShaderPipelineStages(NativeShaderStage(entry.vs->stage));
	if (pixel && HasShaderBufferWrites(st.ps_runtime)) shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	// Late check: the binding work may submit, end rendering, change images or retire buffers.
	if (buffer.Identity() != seg.command || scheduler.CurrentTick() != seg.tick ||
	    buffer.ActiveRenderingSerial() != seg.rendering_serial || ImageStateEpoch() != seg.image_epoch ||
	    m_context.GetBufferCache().UnregisterEpoch() != m_rebind_epoch ||
	    (seg.has_clear && buffer.HasPendingBarriers())) {
		return abandon(GC::kS1Late);
	}
	// S1g (KYTY_NATIVE_S1_SAFE=1): every descriptor and binding the emission writes, validated before
	// anything is recorded; a failure falls back (reason "unsafe", counted by cause).
	if (NativeS1SafeEnabled()) [[unlikely]] {
		auto&              safe = NativeS1Safe();
		NativeS1SafeDetail detail;
		const auto         safe_reason = NativeS1Validate(m_context, pipeline, entry, stages, mesh, vertex_bindings,
		                                                  index_binding, index_source, detail);
		safe.checks++;
		if (safe_reason != kSafeOk) {
			safe.fails[safe_reason]++;
			if (safe.logged < 32) {
				safe.logged++;
				std::printf("GameNative s1 safe: fallback %s (slot %u%s, %s, stage %u index %u value 0x%016" PRIx64
				            ", count %u)\n",
				            kNativeS1SafeNames[safe_reason], slot, pixel ? ", pixel" : "", mesh ? "mesh" : "indexed",
				            detail.stage, detail.index, detail.value, args.index_count);
				std::fflush(stdout);
			}
		}
		NativeS1SafeReport(safe);
		if (safe_reason != kSafeOk) return abandon(GC::kS1Unsafe);
	}
	mark(NP::kEVertex);
	bool dynamic_recorded = false; // S1e: the emission recorded dynamic state (KYTY_NATIVE_S1_KEEP_RUN)
	// S1g: the native draw's own checkpoint (NV checkpoints with KYTY_DEVICE_FAULT_DIAGNOSTICS=1,
	// VK_AMD_buffer_marker breadcrumbs with KYTY_GPU_BREADCRUMBS): after a device loss the dump shows
	// "op=NativeS1Draw" for a natively emitted draw (DrawIndex alone: a normal-path draw). Recorded
	// only when one of them is on; with NV checkpoints it is recorded through Handle(), which records
	// pending barriers first (no pending barrier with a clearing rendering state: checked above).
	buffer.SetDebugInfo(GpuBreadcrumbs::OpNativeS1Draw, submit_id, args.index_count,
	                    0x51000000u | (slot << 8u) | (pixel ? 1u : 0u), 1, args.instance_count, index_source.address);
	bool clear_split = false; // S1g: CommitBindings queued barriers under a clearing rendering state
	// Emission (S1g: a do-while(false) block, left early by clear_split).
	do {
		const CommandBuffer::DrawScope draw_scope(buffer, false);
		const auto                     vk_buffer = buffer.EmissionSink();
		if (!mesh) CommitVertexBuffers(vk_buffer, vertex_bindings); // S1c (count 0 without fetch)
		CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages, false);
		// S1g: CommitBindings' texture transitions are queued barriers. BeginRendering cannot sink them
		// past this draw (DrawScope not safe): it would end the instance and begin the segment's
		// rendering state again, whose clear load ops would erase what the instance drew. Only state
		// commands were recorded (their shadows are invalidated by NativeS1Abandon): fall back.
		if (seg.has_clear && buffer.HasPendingBarriers()) {
			clear_split = true;
			break;
		}
		if (!mesh) CommitIndexBuffer(vk_buffer, index_binding);
		// The dynamic state is a function of the segment's registers, targets and depth state and
		// of the vertex program's viewport-index output: kept when the shadow describes it.
		const bool feedback_ok = !m_context.GetGraphics().attachment_feedback_loop_enabled ||
		                         (m_dynamic_state.feedback_valid && !m_dynamic_state.feedback);
		// S1d: relaxed registers: always recorded from the draw's registers (the shadow skips equal values).
		const bool dynamic_kept = !relaxed && DynamicStateShadowEnabled() && m_dynamic_state.valid &&
		                          m_dynamic_state.command == vk_buffer.Identity() && m_dynamic_state.pipeline != nullptr &&
		                          buffer.BoundPipeline(vk::PipelineBindPoint::eGraphics) == m_dynamic_state.pipeline &&
		                          feedback_ok && entry.viewport_index == seg.viewport_index &&
		                          // S1g: and the pipeline about to be bound is the bound one, as the normal
		                          // path keeps a continuation's dynamic state only without a pipeline swap
		                          // (!m_run_pipeswap); otherwise it is recorded again (the shadow emits only
		                          // what differs).
		                          m_dynamic_state.pipeline == pipeline.pipeline;
		dynamic_recorded = !dynamic_kept;
		if (!dynamic_kept) {
			st.vs_info.stage = st.runtime;
			SetGraphicsDynamicParams(buffer, vk_buffer, st.vs_info, seg.depth, seg.rendering, m_dynamic_state, nullptr);
			if (m_context.GetGraphics().attachment_feedback_loop_enabled) {
				auto& shadow = m_dynamic_state;
				if (!(DynamicStateShadowEnabled() && shadow.feedback_valid && !shadow.feedback)) {
					vk_buffer.setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags {});
					shadow.feedback       = vk::ImageAspectFlags {};
					shadow.feedback_valid = true;
					g_dynamic_state_emitted++;
				}
			}
		}
		scheduler.BeginRendering(seg.rendering);
		buffer.BindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
		m_dynamic_state.command  = vk_buffer.Identity();
		m_dynamic_state.pipeline = pipeline.pipeline;
		if (seg.color_count == 0) {
			m_dynamic_state.color_write_valid = false;
		}
		const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count, args.instance_count, args.first_instance};
		DrawEmitInfo       emit {};
		// ResolveDrawOffsets (S1c: an embedded fetch takes its offsets from the user SGPRs it names).
		if (!mesh && pred_mut.fetch) {
			const auto [vertex_offset, instance_offset] = ResolveDrawOffsets(ucfg.GetIndexOffset(), pred_mut.vertex);
			emit.vertex_offset  = vertex_offset + args.base_vertex;
			emit.first_instance = instance_offset;
		} else {
			emit.vertex_offset  = static_cast<int32_t>(ucfg.GetIndexOffset()) + args.base_vertex;
			emit.first_instance = 0;
		}
		if (mesh) {
			const uint32_t mesh_draw_dwords    = ShaderRecompiler::IR::PushData::MeshDrawDwords(entry.mesh.split_groups != 0);
			const auto     groups_per_dispatch = MeshGroupsPerDispatch(mesh_groups, entry.mesh, limits);
			const auto     address             = index_source.address;
			for (uint32_t first_group = 0; first_group < mesh_groups; first_group += groups_per_dispatch) {
				const auto groups                 = std::min(groups_per_dispatch, mesh_groups - first_group);
				const auto instances_per_dispatch = MeshInstancesPerDispatch(groups, limits);
				for (uint32_t base = 0; base < draw.instance_count; base += instances_per_dispatch) {
					const uint32_t draw_data[] {args.index_count,
					                            static_cast<uint32_t>(emit.vertex_offset),
					                            emit.first_instance + base,
					                            index_source.guest_element_size,
					                            static_cast<uint32_t>(address),
					                            static_cast<uint32_t>(address >> 32u),
					                            first_group};
					static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwords(true));
					vk_buffer.pushConstants(pipeline.pipeline_layout,
					                        vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment, 0,
					                        mesh_draw_dwords * sizeof(uint32_t), draw_data);
					vk_buffer.drawMeshTasksEXT(groups, std::min(instances_per_dispatch, draw.instance_count - base), 1);
				}
			}
		} else {
			EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
		}
		// S1d: the normal path's post-draw write barrier (ExecutePreparedDraw): queued inside the
		// rendering instance with KYTY_DRAW_WRITE_SINK, else after ending it.
		if (shader_write_stages) {
			if (DrawWriteSinkEnabled()) {
				ShaderWriteBarrier(buffer, shader_write_stages);
				m_indirect_barrier_rendering = 0;
			} else {
				{
					KYTY_GPU_OP_SITE("draw.write_barrier");
					scheduler.EndRendering();
				}
				ShaderWriteBarrier(buffer, shader_write_stages);
			}
		}
	} while (false);
	if (clear_split) return abandon(GC::kS1Barriers); // S1g
	ResetBindings();
	mark(NP::kECommit);
	if (NativeS1KeepRun()) {
		// S1e: the DrawRun records stay (NativeS1KeepRun). The next continuation records its dynamic
		// state through the shadow (only differences are emitted) instead of skipping it: the emission
		// may have recorded its own (dynamic_recorded) and bound another pipeline.
		(void)dynamic_recorded;
		m_native_s1_dynamic_dirty = true;
	} else {
		// S1d: the DrawRun records no longer describe the command buffer's descriptors and pipeline.
		BeginDrawRun();
		m_run_prev_valid = false;
		m_run_keepable   = false;
		for (auto& record: m_run_kept) {
			record.valid = false;
		}
	}
	seg.rendering_serial = buffer.ActiveRenderingSerial();
	seg.activity         = DrawRun::ActivityEpoch();
	seg.image_epoch      = ImageStateEpoch();
	seg.draw_counter     = m_native_draw_counter;
	seg.viewport_index   = entry.viewport_index;
	seg.valid            = seg.rendering_serial != 0;
	if (relaxed) {
		// S1d: the next continuation compares with this draw's registers (equal to the last
		// normal draw's outside the relaxed groups).
		seg.targets_key  = targets_key;
		seg.context_hash = context_hash;
	}
	m_native_s0->NoteS1Emitted(op);
	m_native_s0_op   = 0;
	m_native_s1_done = true;
	note.reason      = GC::kS1Emitted;
	note.census      = m_native_s0->S1Census();
	note.shape_mask  = 0;
	note.write_kinds = m_native_s0->S1WriteKinds();
	GC::NoteNativeS1(op, note);
	mark(NP::kETail);
	return true;
}

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
	                                     bool primitive_restart_enable) {
	KYTY_GPU_OP_SITE("draw.execute");
	KYTY_PROFILER_DETAIL_FUNCTION();
	if (PassStats::Enabled()) {
		// Astro Bot PC : cible de la passe (KYTY_PASS_STATS).
		const auto& c0 = state.color_info[0];
		const auto& z  = state.depth_info;
		const bool  has_color = state.color_count != 0;
		{
			uint64_t a[8] {};
			uint32_t f[8] {}, w[8] {}, h[8] {};
			const uint32_t n = std::min<uint32_t>(state.color_count, 8u);
			for (uint32_t i = 0; i < n; i++) {
				const auto& ci = state.color_info[i].desc.info;
				a[i] = ci.data.address;
				f[i] = static_cast<uint32_t>(ci.guest_format);
				w[i] = ci.extent.width;
				h[i] = ci.extent.height;
			}
			PassStats::NoteColors(a, f, w, h, n);
		}
		const auto& info = has_color ? c0.desc.info : z.desc.info;
		PassStats::SetTarget(info.data.address, info.extent.width, info.extent.height,
		                     static_cast<uint32_t>(info.guest_format), state.color_count, static_cast<bool>(z.image_id),
		                     z.image_id ? z.desc.info.data.address : 0u,
		                     state.vertex_info[0].stage.program != nullptr ? state.vertex_info[0].stage.program->shader_hash : 0u,
		                     state.ps_active && state.ps_input_info.stage.program != nullptr ? state.ps_input_info.stage.program->shader_hash : 0u);
	}
	auto& ucfg = buffer.GetUserConfig();
	if (m_context.GetOcclusionCounter().DebugLogActive()) {
		OcclusionCounter::DebugLog("e n=%u inst=%u vs=0x%" PRIx64 "\n", draw.index_count, draw.instance_count,
		                           buffer.GetShaders().GetVs().es_regs.data_addr);
	}
	// Astro Bot PC (KYTY_PASS_SKIP, diagnostic) : dessin non traduit selon sa cible ; image volontairement fausse.
	if (PassStats::SkipDraw(state.color_count, state.color_count != 0 ? state.color_info[0].desc.info.extent.width
	                                                              : state.depth_info.desc.info.extent.width,
	                        static_cast<bool>(state.depth_info.image_id))) {
		m_run_active = false;
		return;
	}
	if (CopyDepthToColor(buffer, draw, emit, state)) {
		FastCheck::Fail(FastCheck::Cause::TargetOp); // Astro Bot PC (KYTY_CP_PAR_FASTCHECK)
		FastCheck::Recorded();
		m_run_active = false; // the copy is not a draw of the run
		return;
	}
	const auto vertex_stages =
	    std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	// CommitStats: image transitions recorded while this draw is committed.
	const auto stats_transitions = CommitStats::Enabled() ? Image::RecordedTransitions() : uint64_t {0};
	if (m_context.GetPipelineCache().PipelinePrefetchEnabled() &&
	    (m_binding_plan == nullptr || m_binding_plan->pipeline == nullptr)) {
		const CommitStats::SubScope prefetch_scope(CommitStats::Sub::Prefetch);
		m_context.GetPipelineCache().PrefetchGraphicsPipeline(
		    {state.color_info, state.color_count}, state.depth_info, vertex_stages, buffer,
		    state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable,
		    state.programs);
	}
	const bool mesh_active = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	uint32_t   mesh_groups = 0;
	// Reused per thread (draws run on the GPU thread under the render mutex).
	static thread_local std::vector<MeshDrawSegment> mesh_segments;
	mesh_segments.clear();
	// KYTY_NATIVE_INDIRECT_MESH: the counts are GPU data, converted into the dispatches right
	// before the draw (meshIndirect.h); nothing below may read them.
	const bool mesh_indirect = mesh_active && emit.mesh_indirect;
	EXIT_IF(emit.mesh_indirect && (!mesh_active || emit.indirect == nullptr));
	if (mesh_active) {
		const auto& mesh = state.vertex_info[0].mesh;
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !MeshRestartEnabled() && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed());
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(), primitive_restart_enable);
		}
	}
	if (mesh_active && !mesh_indirect) {
		const auto& mesh = state.vertex_info[0].mesh;
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
		if (primitive_restart_enable && draw.IsIndexed() && MeshRestartEnabled()) {
			const auto mask = UINT32_MAX >> ((4u - index_source.guest_element_size) * 8u);
			SplitMeshRestartIndices(index_source.address, draw.index_count,
			    index_source.guest_element_size, buffer.GetRegisters().GetPrimitiveResetIndex() & mask, mesh,
			    mesh_segments);
			if (mesh_segments.empty()) return;
		} else {
			mesh_segments.push_back({0, draw.index_count, mesh_groups});
		}
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		uint64_t total_groups = 0;
		for (const auto& segment: mesh_segments) {
			// More instances than one dispatch carries are split when the draw is recorded, and
			// with split_groups (MeshGroupsPerDispatch) so are more groups than X allows.
			if (MeshInstancesPerDispatch(MeshGroupsPerDispatch(segment.groups, mesh, limits),
			                             limits) == 0) {
				EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", segment.groups, draw.instance_count);
			}
			total_groups += segment.groups;
		}
		if (Profiler::AggregateEnabled()) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::MeshDraws);
			if (primitive_restart_enable) {
				// Enabled state only; this does not inspect indices for actual restart markers.
				Profiler::CountFrameEvent(Profiler::FrameEvent::MeshRestartEnabledDraws);
			}
			if (draw.IsIndexed()) {
				Profiler::CountFrameEvent(
				    Profiler::FrameEvent::MeshInputIndices,
				    static_cast<uint64_t>(draw.index_count) * draw.instance_count);
			}
			Profiler::CountFrameEvent(Profiler::FrameEvent::MeshWorkgroups,
			                          total_groups * draw.instance_count);
		}
	}

	if (mesh_active && draw.IsIndexed()) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed. A native
		// indirect draw registers the bound range [INDEX_BASE, +INDEX_BUFFER_SIZE) (its count and
		// start index are GPU data).
		(void)m_context.GetBufferCache().FindBuffer(
		    index_source.address, mesh_indirect ? index_source.size
		                                        : static_cast<uint64_t>(draw.index_count) *
		                                              index_source.guest_element_size);
	}
	const auto* indirect = emit.indirect;
	EXIT_IF(indirect != nullptr && mesh_active && !mesh_indirect);
	if (indirect != nullptr) {
		// ObtainBuffer can merge cache buffers, which must not retire buffers that the shader
		// bindings below already resolved. Acquire the argument, count and whole index ranges
		// first; acquiring them again after the bindings then finds the same or a merged,
		// still covering, buffer and merges nothing. Mesh shaders read indices by address (the
		// range registered above), not through an index buffer binding.
		(void)ObtainIndirectBuffers(buffer, *indirect);
		if (index_source.size != 0 && !mesh_active) {
			(void)m_context.GetBufferCache().ObtainBuffer(index_source.address, index_source.size,
			                                              false);
		}
	}
	if (BindingRepeatStatsEnabled()) [[unlikely]] {
		std::array<const ShaderStageRuntime*, 4> runtimes {};
		uint32_t                                 runtime_count = 0;
		for (uint32_t i = 0; i < vertex_stages.size(); i++) {
			runtimes[runtime_count++] = &state.vertex_info[i].stage;
		}
		if (state.ps_active) {
			runtimes[runtime_count++] = &state.ps_input_info.stage;
		}
		CountBindingRepeats(std::span {runtimes.data(), runtime_count});
	}
	// KYTY_DRAW_PREP_BINDINGS: the committed draw's binding plan (RefreshShaders activated it once
	// DrawPrep::Validate accepted the preparation it was computed from). Its stage plans cover a
	// single vertex stage and the pixel stage.
	auto* const plan        = m_binding_plan_active ? m_binding_plan : nullptr;
	const bool  plan_verify = plan != nullptr && DrawPrep::BindingsVerifyMode() != 0;
	const bool  plan_stages = plan != nullptr && vertex_stages.size() == 1;
	CommitStats::StepMark(CommitStats::Step::ExecHead);
	LogDrawPhase(draw.Name(), "PrepareBindings");
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::PrepareBindings");
		// KYTY_DRAW_RUN continuation: the stages keep the previous draw's textures and samplers.
		for (uint32_t i = 0; i < vertex_stages.size(); i++) {
			PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i],
			                plan_stages ? &plan->vertex : nullptr, m_run_active);
			descriptor_stages[stage_count++] = &bindings.vertex[i];
		}
		if (state.ps_active) {
			if (!bindings.pixel) bindings.pixel.emplace();
			PrepareBindings(state.ps_input_info.stage, *bindings.pixel,
			                plan_stages && plan->pixel_active ? &plan->pixel : nullptr,
			                m_run_active);
			descriptor_stages[stage_count++] = &*bindings.pixel;
		}
	}
	CommitStats::StepMark(CommitStats::Step::PrepStages);
	CommitStats::Mark(CommitStats::Phase::PrepareBindings);
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::PrepareGraphicsBindings");
		PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count},
		                        m_run_active);
	}
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	if (!mesh_active) {
		LogDrawPhase(draw.Name(), "PrepareVertexBuffers");
		// KYTY_DRAW_PREP_BINDINGS buffers: the plan's ranges while the guest virtual ranges are the
		// ones its clamps saw.
		const DrawPrep::VertexRangePlan* vertex_ranges = nullptr;
		if (plan_stages && plan->vertex_ranges.valid &&
		    plan->vertex_ranges.buffer_count ==
		        static_cast<uint32_t>(state.vertex_info[0].buffers_num)) {
			if (plan->vm_generation == LibKernel::Memory::VirtualRangesGeneration()) {
				vertex_ranges = &plan->vertex_ranges;
				OwnedCounters::Add(DrawPrep::GetBindingTotals().vertex_ranges_used, 1);
			} else {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingFallbackVm);
			}
		}
		if (CpCommit::Enabled(CpCommit::Part::Draws)) {
			AcquireVertexBuffersInto(buffer, state.vertex_info[0], vertex_ranges, vertex_bindings);
		} else {
			vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0], vertex_ranges);
		}
		CommitStats::StepMark(CommitStats::Step::VertexAcquire);
		index_binding = PrepareIndexBuffer(buffer, index_source);
	}
	PreparedIndirectBuffers indirect_buffers;
	MeshIndirect::Converter::Slot mesh_slot;
	if (mesh_indirect) {
		// Still preparation: a ring wrap may submit the recording here.
		if (m_mesh_indirect == nullptr) {
			m_mesh_indirect = std::make_unique<MeshIndirect::Converter>(m_context);
		}
		mesh_slot = m_mesh_indirect->Reserve();
	}
	if (indirect != nullptr) {
		indirect_buffers = ObtainIndirectBuffers(buffer, *indirect);
	}
	CommitStats::StepMark(CommitStats::Step::IndexPrepare);
	CommitStats::Mark(CommitStats::Phase::VertexIndex);
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "CreatePipeline");
	}
	// All per-draw buffer work is now complete, including vertex/index/indirect acquisition. It
	// can end rendering without changing a command buffer, tick or image state (buffer uploads /
	// barriers). The rendering instance is part of the certificate, not just image identity.
	// Unless every kept image is as the previous draw left it, the structure is resolved now, in the normal order relative to the buffer
	// bindings, which are made again after it (their reservations follow the image identities).
	// KYTY_DRAW_RUN_INDIRECT: an indirect continuation also needs the argument barrier of this
	// rendering instance (recording one ends rendering, which a continuation never does).
	// KYTY_DRAW_RUN_MESH: a mesh indirect draw's argument conversion (below) ends the rendering
	// instance, which a continuation never does: it takes the normal path (as for the barrier).
	const bool run_mesh_indirect = mesh_indirect && DrawRun::MeshRunsEnabled();
	if (run_mesh_indirect && m_run_active) {
		OwnedCounters::Add(DrawRun::GetTotals().mesh_indirect_fallbacks, 1);
	}
	const bool run_indirect_barrier =
	    (indirect != nullptr && buffer.ActiveRenderingSerial() != m_indirect_barrier_rendering) ||
	    run_mesh_indirect;
	// KYTY_DRAW_RUN_TARGETS: kept targets whose attachments changed during the buffer work are
	// resolved now, with the same repeated binding work as a continuation's fallback.
	const bool targets_late = m_run_targets_kept &&
	                          (!DrawRunCommandUnchanged(buffer) || !DrawRunImagesUnchanged(true, true));
	if (targets_late) {
		m_run_targets_kept = false;
		OwnedCounters::Add(DrawRun::GetTotals().targets_late, 1);
	}
	if (targets_late || (m_run_active && (!DrawRunCommandUnchanged(buffer) || !DrawRunImagesUnchanged(true) ||
	                                      run_indirect_barrier))) {
		const CommitStats::SubScope late_scope(CommitStats::Sub::LateFallback);
		FastCheck::Fail(FastCheck::Cause::RunLateFallback); // Astro Bot PC (KYTY_CP_PAR_FASTCHECK)
		if (!targets_late) {
			DrawRun::GetTotals().late_fallbacks.fetch_add(1, std::memory_order_relaxed);
			DrawRun::CountMiss(DrawRun::Miss::Images);
		}
		m_run_active = false;
		DrawRunRecallFallback();
		DrawRunTargets(buffer, draw, m_run_slice_offset, state);
		// No plan: its shader data was taken by the first preparation above.
		for (uint32_t i = 0; i < vertex_stages.size(); i++) {
			PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i], nullptr);
		}
		if (state.ps_active) {
			PrepareBindings(state.ps_input_info.stage, *bindings.pixel, nullptr);
		}
		PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count});
		// The repeated binding work may merge/retire cache buffers or restart the command buffer.
		// Reacquire the per-draw vertex/index reservations after it, just as on the normal path,
		// into fresh bindings: the first acquisition above filled them (AcquireVertexBuffersInto
		// requires value-initialised bindings and stopped the emulator for a draw with vertex
		// buffers).
		vertex_bindings = PreparedVertexBuffers {};
		if (CpCommit::Enabled(CpCommit::Part::Draws)) {
			AcquireVertexBuffersInto(buffer, state.vertex_info[0], nullptr, vertex_bindings);
		} else {
			vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0], nullptr);
		}
		index_binding = PrepareIndexBuffer(buffer, index_source);
		if (indirect != nullptr) {
			indirect_buffers = ObtainIndirectBuffers(buffer, *indirect);
		}
	}
	// KYTY_REBIND_AFTER_JOIN: a buffer the bindings resolved was retired since (a later lookup of
	// this draw - another binding, a vertex, index or argument range - joined it into a new
	// buffer): the recorded draw would read and write a buffer nothing uses again, its writes lost
	// while the range stays GPU-owned. Resolve the buffers and the per-draw ranges again while the
	// count moves (at most 4 times).
	if (RebindAfterJoinEnabled()) {
		auto& cache = m_context.GetBufferCache();
		for (uint32_t pass = 0; pass < 4 && cache.UnregisterEpoch() != m_rebind_epoch; pass++) {
			NoteRebindAfterJoin("draw", draw.Name());
			FastCheck::Fail(FastCheck::Cause::BufferRejoin); // Astro Bot PC (KYTY_CP_PAR_FASTCHECK)
			m_rebind_epoch = cache.UnregisterEpoch();
			{
				const BufferCache::UploadBatch upload_batch(cache);
				for (auto* stage: stages) {
					RebindBuffers(*stage);
				}
			}
			if (!mesh_active) {
				vertex_bindings = PreparedVertexBuffers {};
				if (CpCommit::Enabled(CpCommit::Part::Draws)) {
					AcquireVertexBuffersInto(buffer, state.vertex_info[0], nullptr, vertex_bindings);
				} else {
					vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0], nullptr);
				}
				index_binding = PrepareIndexBuffer(buffer, index_source);
			}
			if (indirect != nullptr) {
				indirect_buffers = ObtainIndirectBuffers(buffer, *indirect);
			}
		}
	}
	// KYTY_DRAW_RUN=verify: whether the kept images would have passed the check above (the normal
	// path's own texture resolution ran before it here). A draw that would have fallen back is
	// not compared.
	if (m_run_verify && (!DrawRunCommandUnchanged(buffer) || !DrawRunImagesUnchanged(true) ||
	                     run_indirect_barrier)) {
		DrawRun::GetTotals().late_fallbacks.fetch_add(1, std::memory_order_relaxed);
		DrawRun::CountMiss(DrawRun::Miss::Images);
		m_run_verify = false;
		DrawRunRecallFallback();
	}
	// N3b: a continuation that fell back late is not lean; an indirect or mesh draw never is.
	if ((m_native_lean || m_native_lean_sample) && (!m_run_active || indirect != nullptr || mesh_active)) {
		NativeLeanDrop(!m_run_active ? Loader::GameCapture::kNlLate : Loader::GameCapture::kNlShape);
	}
	CommitStats::StepMark(CommitStats::Step::LateCheck);

	// KYTY_LOD_STATS_PLAIN_VARIANT: without a mip-statistics counter on any image the pixel
	// program's feedback records nothing; its feedback-free variant then gives the same results
	// without forcing the depth/stencil tests after a shader that can discard.
	const auto* programs    = &state.programs;
	bool        plain_pixel = false;
	PipelineCache::GraphicsPrograms plain_programs;
	if (state.ps_active && bindings.pixel.has_value()) {
		const auto plain_mode = LodStatsCounter::PlainVariant();
		const auto plain      = plain_mode != LodStatsCounter::Plain::Off
		                            ? PipelineCache::PlainPixelProgram(state.stage_preps.pixel)
		                            : ShaderProgram {};
		if (plain) {
			if (bindings.pixel->mip_stats_active) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::LodStatsInstrumentedDraws);
			} else if (plain_mode == LodStatsCounter::Plain::On) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::LodStatsPlainDraws);
				plain_programs       = state.programs;
				plain_programs.pixel = plain;
				programs             = &plain_programs;
				plain_pixel          = true;
			} else if (m_context.GetLodStats().CanaryReady()) {
				// Verify: the instrumented program records into the canary, which must stay reset.
				Profiler::CountFrameEvent(Profiler::FrameEvent::LodStatsPlainDraws);
				bindings.pixel->mip_stats_canary = true;
			}
		}
	}
	// KYTY_DRAW_PREP_BINDINGS pipeline: the object the preparing thread found for this key, when
	// the draw's targets, topology, restart flag and plain-pixel choice are the ones the key was
	// built for and no pipeline was replaced since (DrawPrep::PlannedPipeline).
	auto&       pipeline_cache = m_context.GetPipelineCache();
	const auto  colors_span    = std::span<const RenderColorInfo> {state.color_info, state.color_count};
	if (pipeline_cache.PipelineSkipEnabled() && SkipRecurrentTargetsOnly()) {
		NoteSkipTargets(m_context, state);
	}
	const auto* ps_input       = state.ps_active ? &state.ps_input_info : nullptr;
	// S1 (nativeLearn.h): the pipeline generation before the selection; a pipeline replaced after
	// it makes the learned object stale for native emission (fast-first handles are destroyed later).
	m_native_pipeline_generation = m_native_s0_op != 0 ? pipeline_cache.PipelineGeneration() : 0;
	const auto* planned_pipeline =
	    plan_stages && plan->pipeline != nullptr
	        ? DrawPrep::PlannedPipeline(pipeline_cache, *plan, colors_span, state.depth_info,
	                                    topology, primitive_restart_enable, plain_pixel)
	        : nullptr;
	const PipelineCache::Pipeline* pipeline_object = planned_pipeline;
	if (planned_pipeline != nullptr && !plan_verify) {
		pipeline_cache.NotePlannedPipeline(state.depth_info, ps_input);
	} else {
		FastCheck::Fail(FastCheck::Cause::PipelineNotPlanned); // Astro Bot PC (KYTY_CP_PAR_FASTCHECK)
		const bool skippable = planned_pipeline == nullptr && pipeline_cache.PipelineSkipEnabled() &&
		                       DrawOnlyChangesPicture(m_context, state, vertex_stages, mesh_active);
		pipeline_object = pipeline_cache.TryGetGraphicsPipeline(
		    colors_span, state.depth_info, vertex_stages, buffer, ps_input, topology,
		    primitive_restart_enable, *programs, skippable);
		if (pipeline_object == nullptr) {
			// KYTY_PIPELINE_SKIP: skipped until its pipeline is built. Nothing was recorded; no
			// draw-run record may be continued or recalled across it (its bindings replaced the
			// record's).
			m_run.valid        = false;
			m_run_active       = false;
			m_run_targets_kept = false;
			for (auto& record: m_run_kept) {
				record.valid = false;
			}
			return;
		}
		if (planned_pipeline != nullptr) {
			DrawPrep::CountBindingVerifyCheck();
			// A pipeline replaced after the certificate check is a race, not a difference.
			if (pipeline_object != planned_pipeline &&
			    pipeline_cache.PipelineGeneration() == plan->pipeline_generation) {
				DrawPrep::ReportBindingMismatch("pipeline");
			}
		}
	}
	const auto& pipeline = *pipeline_object;
	FastCheck::Recorded(); // Astro Bot PC (KYTY_CP_PAR_FASTCHECK): the draw is recorded below
	CommitStats::StepMark(CommitStats::Step::PipelineSel);
	CommitStats::Mark(CommitStats::Phase::Pipeline);
	// Astro Bot PC (KYTY_NATIVE_MODE=s0, nativeLearn.h): the learning hook of a native op's draw.
	if (m_native_s0_op != 0) [[unlikely]] {
		NativeLearn::Prof::Marker learn_mark; // S1e (KYTY_NATIVE_S1_PROFILE)
		NativeS0Learn(state, pipeline, *programs, plain_pixel, mesh_active, static_cast<uint32_t>(vertex_stages.size()),
		              draw.IsIndexed(), draw.index_count, buffer);
		learn_mark(NativeLearn::Prof::kNLearn);
	}
	if (mesh_indirect) {
		// The conversion of the argument record into this draw's dispatches and draw dwords:
		// outside rendering (it ends the active instance, before the targets are acquired for the
		// new one), after every earlier write, before the draw (MeshIndirect::Converter::Record).
		m_mesh_indirect->Record(buffer, mesh_slot, emit.mesh_inputs, indirect_buffers.args,
		                        indirect_buffers.args_offset);
	}
	vk::ImageAspectFlags feedback_aspects;
	// KYTY_DRAW_PREP_BINDINGS statics: the plan's program flags and scissor union.
	const bool plan_statics = plan_stages && plan->statics_valid;
	RenderState rendering;
	// KYTY_DRAW_RUN=verify: the continuation's skipped acquisition and transitions must be no-ops.
	const bool run_verify_images   = m_run_verify && DrawRunImagesUnchanged(false);
	const auto run_transitions     = m_run_verify ? Image::RecordedTransitions() : uint64_t {0};
	// Where this draw can write its colour targets (KYTY_ALIAS_BYTES ownership claims); a
	// continuation's are the recorded draw's.
	vk::Rect2D written = m_run_active ? m_run.written : vk::Rect2D {};
	const bool bounded = TextureCache::AliasBytesEnabled();
	if (bounded && !m_run_active) {
		if (plan_statics && plan->written_valid && !plan_verify) {
			written = plan->written;
		} else {
			const auto& outputs = vertex_stages.back().stage.program->info.outputs;
			written             = DrawScissorUnion(
	            buffer.GetRegisters(), std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
	                return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	            }));
			if (plan_statics && plan->written_valid) {
				DrawPrep::CountBindingVerifyCheck();
				if (written != plan->written) {
					DrawPrep::ReportBindingMismatch("scissor union");
				}
			}
		}
	}
	if (m_run_active) {
		// KYTY_DRAW_RUN continuation: the attachments were acquired, claimed and transitioned by the
		// previous draw of the run, for the same targets, scissors and textures (no feedback), and
		// are still in that state (DrawRunImagesUnchanged).
		rendering              = m_run.rendering;
		m_depth_feedback.valid = false;
	} else {
		// KYTY_DRAW_RUN_ACQUIRE: the same targets and scissor union right after the recorded draw, in
		// its rendering instance, with the attachments as it left them: its acquisition, whose
		// transitions, claims, refreshes and depth-feedback bookkeeping would all be repeats.
		// (Verify mode checks a would-be continuation's acquisition as the continuation's.)
		CommitStats::SubScope candidate_scope(CommitStats::Sub::AcqCandidate);
		const bool acquire_reuse =
		    DrawRun::Enabled() && DrawRun::AcquireReuseEnabled() && !m_run_verify &&
		    DrawRunAcquireCandidate(buffer, state, stages, bounded ? &written : nullptr);
		candidate_scope.Stop();
		if (acquire_reuse) {
			OwnedCounters::Add(DrawRun::GetTotals().acquire_reused, 1);
		}
		if (acquire_reuse && DrawRun::GetMode() == DrawRun::Mode::On) {
			rendering              = m_run.rendering;
			m_depth_feedback.valid = false;
		} else {
			const auto acquire_transitions =
			    acquire_reuse ? Image::RecordedTransitions() : uint64_t {0};
			FastCheck::Fail(FastCheck::Cause::TargetsAcquired); // Astro Bot PC (KYTY_CP_PAR_FASTCHECK)
			rendering = AcquireRenderTargets(buffer, state.color_info, state.color_count,
			                                 state.depth_info, feedback_aspects, stages,
			                                 bounded ? &written : nullptr);
			if (acquire_reuse) {
				// Verify mode: the acquisition a reuse skips must have been a repeat.
				DrawRun::CountVerifyCheck();
				if (!(rendering == m_run.rendering) || feedback_aspects) {
					DrawRun::ReportMismatch("reused acquisition's rendering state");
				}
				if (Image::RecordedTransitions() != acquire_transitions) {
					DrawRun::ReportMismatch("reused acquisition's transitions",
					                        DrawRunImagesChange(false, true, true));
				}
				if (const auto change = DrawRunImagesChange(false, true, true); change != 0) {
					DrawRun::ReportMismatch("reused acquisition's attachment state", change);
				}
			}
			if (run_verify_images) {
				if (const auto change = DrawRunImagesChange(false, false, true); change != 0) {
					DrawRun::CountVerifyCheck();
					DrawRun::ReportMismatch("attachment state after acquisition", change);
				}
			}
		}
	}
	CommitStats::StepMark(CommitStats::Step::Acquire);
	CommitStats::Mark(CommitStats::Phase::AcquireTargets);

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory. Until BeginRendering() only state commands are recorded through vk_buffer; image
	// transitions and dependencies go through the barrier batcher, which BeginRendering() flushes
	// (or, for a safe draw continuing the same rendering instance, may sink past the draw).
	const bool* programs_safe = plan_statics && plan->plain_pixel == plain_pixel
	                                ? &plan->programs_barrier_safe
	                                : nullptr;
	// N3b: a lean continuation's barrier safety is its record's: the same programs (so the same
	// GDS bindings and program verdicts), textures and attachments, no feedback (a continuation
	// acquires nothing), not indirect, and the same plain-pixel choice and barrier-sink switch.
	const bool lean_barrier = (m_native_lean || m_native_lean_sample) && !feedback_aspects &&
	                          plain_pixel == m_native_record.plain_pixel &&
	                          BarrierSinkEnabled() == m_native_record.sink;
	if ((m_native_lean || m_native_lean_sample) && !lean_barrier) {
		NativeLeanDrop(Loader::GameCapture::kNlShape);
	}
	const bool barrier_safe =
	    m_native_lean ? m_native_record.barrier_safe
	                  : DrawIsBarrierSafe(stages, state.color_info, state.color_count, state.depth_info,
	                                      feedback_aspects, indirect != nullptr, plain_pixel,
	                                      plan_verify ? nullptr : programs_safe);
	if (m_native_lean_sample) {
		m_native_lean_checked = true;
		if (barrier_safe != m_native_record.barrier_safe) {
			NativeLeanMismatch("barrier safety", barrier_safe ? 1u : 0u);
		}
	}
	if (!m_native_lean && plan_verify && programs_safe != nullptr) {
		DrawPrep::CountBindingVerifyCheck();
		if (DrawIsBarrierSafe(stages, state.color_info, state.color_count, state.depth_info,
		                      feedback_aspects, indirect != nullptr, plain_pixel,
		                      programs_safe) != barrier_safe) {
			DrawPrep::ReportBindingMismatch("barrier safety");
		}
	}
	CommitStats::StepMark(CommitStats::Step::BarrierSafe);
	const CommandBuffer::DrawScope draw_scope(buffer, barrier_safe);
	// Emission safe point (KYTY_CP_RECORDER): no native handle obtained during the preparation above
	// is alive; from here the draw's commands go to the recorder (render.h, CommandBuffer::Sink).
	const auto vk_buffer = buffer.EmissionSink();
	SetDrawDebugPhase(buffer, submit_id, draw, emit, draw.IsIndexed() ? 0x100u : 0x200u);
	CommitStats::StepMark(CommitStats::Step::DrawScope);
	if (!mesh_active) {
		const CommitStats::SubScope vertex_scope(CommitStats::Sub::VertexCommit);
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	CommitStats::StepMark(CommitStats::Step::VertexCommit);
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, emit, 0x300u);
	}
	// KYTY_DRAW_RUN_PUSH verify: a would-be continuation that would push only its per-draw
	// descriptors must push the previous draw's image and sampler descriptors on the normal path.
	const bool run_verify_push =
	    run_verify_images && m_run.verify_valid && DrawRunPartialPush(buffer, pipeline);
	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::CommitBindings");
		const CommitStats::SubScope push_scope(CommitStats::Sub::Push);
		CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages, m_run_active);
	}
	if (run_verify_push) {
		OwnedCounters::Add(DrawRun::GetTotals().partial_pushes, 1);
		DrawRun::CountVerifyCheck();
		if (m_descriptor_images != m_run.pushed_images) {
			DrawRun::ReportMismatch("pushed image or sampler descriptors",
			                        m_descriptor_images.size());
		}
	}
	if (run_verify_images && Image::RecordedTransitions() != run_transitions) {
		DrawRun::CountVerifyCheck();
		// Detail: which recorded image's state the transition changed (0: another image's).
		DrawRun::ReportMismatch("image transition", DrawRunImagesChange(false, false, true));
	}
	CommitStats::StepMark(CommitStats::Step::CbTail);
	CommitStats::Mark(CommitStats::Phase::CommitBindings);
	if (!mesh_active) CommitIndexBuffer(vk_buffer, index_binding);
	CommitStats::StepMark(CommitStats::Step::IndexCommit);

	// KYTY_DRAW_RUN continuation: the dynamic state is a function of the same registers, targets and
	// vertex program as the previous draw's, which recorded it into this command buffer; the
	// shadow shows nothing replaced it since (another pipeline bind or command buffer).
	// A recall records it again: the draws in between left their own.
	// S1e: a native emission since the record recorded the dynamic state of its own registers and
	// vertex program (m_native_s1_dynamic_dirty): the continuation records its own again.
	const bool run_dynamic = (m_run_active || m_run_verify) && !m_run_recall && !m_run_pipeswap &&
	                         !m_native_s1_dynamic_dirty && DynamicStateShadowEnabled() &&
	                         m_dynamic_state.valid && m_dynamic_state.command == vk_buffer.Identity() &&
	                         m_dynamic_state.pipeline != nullptr &&
	                         buffer.BoundPipeline(vk::PipelineBindPoint::eGraphics) ==
	                             m_dynamic_state.pipeline &&
	                         (!m_context.GetGraphics().attachment_feedback_loop_enabled ||
	                          (m_dynamic_state.feedback_valid && !m_dynamic_state.feedback));
	if (m_run_active && !run_dynamic) {
		DrawRun::GetTotals().dynamic_emitted.fetch_add(1, std::memory_order_relaxed);
	}
	const auto run_dynamic_emitted = g_dynamic_state_emitted;
	if (!(m_run_active && run_dynamic)) {
		KYTY_PROFILER_DETAIL_BLOCK("Draw::DynamicState");
		const CommitStats::SubScope dynamic_scope(CommitStats::Sub::Dynamic);
		SetGraphicsDynamicParams(buffer, vk_buffer, vertex_stages.back(), state.depth_info,
		                         rendering, m_dynamic_state,
		                         plan_stages && plan->viewports.valid ? &plan->viewports : nullptr);
		if (m_context.GetGraphics().attachment_feedback_loop_enabled) {
			// Declared dynamic by every renderer pipeline; the shadow (valid for this command
			// buffer and bound pipeline, established just above) covers it too.
			auto& shadow = m_dynamic_state;
			if (DynamicStateShadowEnabled() && shadow.feedback_valid &&
			    shadow.feedback == feedback_aspects) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DynamicStateCommandsAvoided);
			} else {
				vk_buffer.setAttachmentFeedbackLoopEnableEXT(feedback_aspects);
				shadow.feedback       = feedback_aspects;
				shadow.feedback_valid = true;
				Profiler::CountFrameEvent(Profiler::FrameEvent::DynamicStateCommandsEmitted);
				g_dynamic_state_emitted++;
			}
		}
	}
	m_native_s1_dynamic_dirty = false; // S1e: the draw's dynamic state is recorded (or the record's still holds)
	if (m_run_verify && run_verify_images && run_dynamic &&
	    g_dynamic_state_emitted != run_dynamic_emitted) {
		DrawRun::CountVerifyCheck();
		DrawRun::ReportMismatch("dynamic state", g_dynamic_state_emitted - run_dynamic_emitted);
	}

	CommitStats::StepMark(CommitStats::Step::DynamicState);
	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, emit, 0x400u);
	}
	if (indirect != nullptr) {
		// Pipeline barriers cannot be recorded inside dynamic rendering. Every buffer write is
		// recorded outside rendering or ends it (ShaderWriteBarrier), so the rendering instance
		// begun right after the previous barrier needs no new one while it remains active.
		const auto active = buffer.ActiveRenderingSerial();
		if (active == 0 || active != m_indirect_barrier_rendering) {
			const CommitStats::SubScope barrier_scope(CommitStats::Sub::Barriers);
			{
				KYTY_GPU_OP_SITE("draw.indirect_args");
				m_context.GetCommandScheduler().EndRendering();
			}
			IndirectArgumentsBarrier(buffer, vk_buffer);
		}
	}
	{
		const CommitStats::SubScope begin_scope(CommitStats::Sub::Begin);
		m_context.GetCommandScheduler().BeginRendering(rendering);
	}
	if (m_depth_feedback.valid) {
		NoteDepthFeedback(buffer);
	}
	if (indirect != nullptr) {
		// A state change above only switched instances; no buffer write came in between.
		m_indirect_barrier_rendering = buffer.ActiveRenderingSerial();
	}
	CommitStats::StepMark(CommitStats::Step::Begin);
	{
		const CommitStats::SubScope bind_scope(CommitStats::Sub::BindPipeline);
		if (CommitStats::Enabled() &&
		    buffer.BoundPipeline(vk::PipelineBindPoint::eGraphics) != pipeline.pipeline) [[unlikely]] {
			CommitStats::AddItems(CommitStats::Item::PipelineNew, 1);
		}
		buffer.BindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
		// The shadow now describes this command buffer with this pipeline bound. A pipeline
		// without color attachments declares no dynamic color-write enable, so binding it
		// invalidated that state for later pipelines.
		m_dynamic_state.command  = vk_buffer.Identity();
		m_dynamic_state.pipeline = pipeline.pipeline;
		if (state.color_count == 0) {
			m_dynamic_state.color_write_valid = false;
		}
	}
	CommitStats::StepMark(CommitStats::Step::BindPipeline);
	CommitStats::SubScope draw_cmd_scope(CommitStats::Sub::DrawCmd);
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, emit, 0x500u);
	}
	// Mesh programs with split_groups read a seventh draw dword (MeshFirstGroupDword).
	const uint32_t mesh_draw_dwords =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwords(
	                      state.vertex_info[0].mesh.split_groups != 0)
	                : 0u;
	// PA_CL_UCP planes (upstream feed075a9): four float dwords per plane up to the highest enabled.
	if (const auto planes = state.vertex_info[0].user_clip_planes; planes != 0) {
		uint32_t count = 0;
		for (uint32_t mask = planes; mask != 0; mask >>= 1u) {
			count++;
		}
		vk_buffer.pushConstants(
		    pipeline.pipeline_layout,
		    vk::ShaderStageFlagBits::eFragment |
		        (mesh_active ? vk::ShaderStageFlagBits::eMeshEXT : vk::ShaderStageFlagBits::eVertex),
		    ShaderRecompiler::IR::PushData::UserClipPlaneDword * sizeof(uint32_t),
		    count * 4u * sizeof(uint32_t), buffer.GetRegisters().GetUserClipPlanes().data());
	}
	if (mesh_indirect) {
		// One indirect dispatch per conversion record, each reading its draw dwords from its own
		// parameter block (a record the draw does not need has no workgroups). Indirect draws are
		// never split: the block's dword 6 is 0.
		for (uint32_t record = 0; record < MeshIndirect::Records; record++) {
			const auto     address = mesh_slot.ParamsAddress(record);
			const uint32_t draw_data[] {static_cast<uint32_t>(address),
			                            static_cast<uint32_t>(address >> 32u), 0u,
			                            ShaderRecompiler::IR::PushData::MeshIndirectSentinel, 0u, 0u,
			                            0u};
			static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwords(true));
			vk_buffer.pushConstants(pipeline.pipeline_layout,
			    vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment, 0,
			    mesh_draw_dwords * sizeof(uint32_t), draw_data);
			vk_buffer.drawMeshTasksIndirectEXT(mesh_slot.buffer, mesh_slot.CommandOffset(record), 1,
			                                   MeshIndirect::CommandDwords * 4u);
		}
	} else if (mesh_active) {
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		const auto& mesh   = state.vertex_info[0].mesh;
		for (const auto& segment: mesh_segments) {
			const auto address = index_source.address +
			    static_cast<uint64_t>(segment.first) * index_source.guest_element_size;
			// A segment with more groups than X allows is dispatched in parts (split_groups only),
			// each pushing its first group. The shader's instance index is the pushed first instance
			// plus WorkgroupId.y, so a part with more instances than one dispatch carries is split
			// into instance ranges.
			const auto groups_per_dispatch = MeshGroupsPerDispatch(segment.groups, mesh, limits);
			for (uint32_t first_group = 0; first_group < segment.groups;
			     first_group += groups_per_dispatch) {
				const auto groups = std::min(groups_per_dispatch, segment.groups - first_group);
				const auto instances_per_dispatch = MeshInstancesPerDispatch(groups, limits);
				for (uint32_t base = 0; base < draw.instance_count; base += instances_per_dispatch) {
					const uint32_t draw_data[] {
					    segment.count,
					    draw.IsIndexed() ? static_cast<uint32_t>(emit.vertex_offset) : emit.first_vertex,
					    emit.first_instance + base, index_source.guest_element_size,
					    static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u),
					    first_group};
					static_assert(std::size(draw_data) ==
					              ShaderRecompiler::IR::PushData::MeshDrawDwords(true));
					if (m_native_s0 != nullptr && m_native_s0->Capturing()) [[unlikely]] {
						m_native_s0->CaptureMesh(draw_data, mesh_draw_dwords);
					}
					vk_buffer.pushConstants(pipeline.pipeline_layout,
					    vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment,
					    0, mesh_draw_dwords * sizeof(uint32_t), draw_data);
					vk_buffer.drawMeshTasksEXT(groups,
					                           std::min(instances_per_dispatch, draw.instance_count - base),
					                           1);
				}
			}
		}
	} else if (indirect != nullptr) {
		EmitIndirectDraw(vk_buffer, *indirect, indirect_buffers);
	} else {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	}
	draw_cmd_scope.Stop();
	if (m_native_s0 != nullptr) [[unlikely]] {
		// S1: the segment state this native op's normal-path draw leaves (its own draw only).
		// S1b: only for the slots S1 emits for (the segment state costs a targets key without one).
		NativeLearn::Prof::Marker segment_mark; // S1e (KYTY_NATIVE_S1_PROFILE)
		if (Loader::GameCapture::NativeS1() && m_native_s0->Active() && indirect == nullptr && !mesh_indirect &&
		    Loader::GameCapture::NativeS1SlotEnabled(m_native_s0->ActiveOp())) {
			NativeS1SegmentDraw(buffer, state, rendering);
			segment_mark(NativeLearn::Prof::kNSegment);
		}
		NativeS0Finish(); // S0: compares a verify sample's captures, reports the draw
		segment_mark(NativeLearn::Prof::kNFinish);
	}
	CommitStats::StepMark(CommitStats::Step::DrawCmd);
	static const bool occlusion_draw_rows = [] {
		const char* value = std::getenv("KYTY_HANG_TRACE_OCCLUSION_DRAWS");
		return value != nullptr && value[0] == '1';
	}();
	const bool occlusion_debug_log = m_context.GetOcclusionCounter().DebugLogActive();
	if ((occlusion_draw_rows && HangTrace::Enabled() && m_context.GetOcclusionCounter().Active()) ||
	    occlusion_debug_log) {
		// Occlusion diagnostics: the state deciding whether this draw's samples pass.
		const auto&               regs = buffer.GetRegisters();
		const auto&               vp   = regs.GetScreenViewport().viewports[0];
		const auto&               mc   = regs.GetModeControl();
		const auto&               di   = state.depth_info;
		HangTrace::OcclusionEvent event;
		event.event  = "draw";
		event.value  = draw.index_count;
		event.scopes = draw.instance_count;
		event.colors = state.color_count;
		event.has_depth = static_cast<bool>(di.image_id);
		event.detail = fmt::format(
		    "prim={} indexed={} mesh={} indirect={} ps={} zt={} zw={} zop={} db={}[{:g},{:g}] "
		    "st={} cull={}{} face={} vp=[{:g},{:g}] zs={:g} zo={:g} vpxy={:g}x{:g}",
		    static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed() ? 1 : 0,
		    mesh_active ? 1 : 0, indirect != nullptr ? 1 : 0, state.ps_active ? 1 : 0,
		    di.depth_test_enable ? 1 : 0, di.depth_write_enable ? 1 : 0,
		    static_cast<uint32_t>(di.depth_compare_op), di.depth_bounds_test_enable ? 1 : 0,
		    di.depth_min_bounds, di.depth_max_bounds, di.stencil_test_enable ? 1 : 0,
		    mc.cull_front ? "F" : "", mc.cull_back ? "B" : "", mc.face ? 1 : 0, vp.zmin,
		    vp.zmax, vp.zscale, vp.zoffset, vp.xscale, vp.yscale);
		if (occlusion_debug_log) {
			const auto& clip = regs.GetClipControl();
			OcclusionCounter::DebugLog(
			    "d counted=%u n=%u inst=%u vs=0x%" PRIx64 " ps=0x%" PRIx64 " colors=%u zclip=%u/%u %s\n",
			    m_context.GetOcclusionCounter().Active() ? 1u : 0u, draw.index_count, draw.instance_count,
			    buffer.GetShaders().GetVs().es_regs.data_addr, buffer.GetShaders().GetPs().ps_regs.data_addr,
			    state.color_count, clip.min_z_clip_disable ? 1u : 0u, clip.max_z_clip_disable ? 1u : 0u,
			    event.detail.c_str());
		}
		if (occlusion_draw_rows && HangTrace::Enabled() && m_context.GetOcclusionCounter().Active()) {
			HangTrace::RecordOcclusion(event);
		}
	}

	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, emit, 0x600u);
	}
	vk::PipelineStageFlags shader_write_stages = {};
	if (plan_statics && !plan_verify) {
		shader_write_stages = plan->shader_write_stages;
		OwnedCounters::Add(DrawPrep::GetBindingTotals().statics_used, 1);
	} else {
		for (const auto& stage: vertex_stages) {
			if (HasShaderBufferWrites(stage.stage)) {
				shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
			}
		}
		if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
			shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
		}
		if (plan_statics) {
			DrawPrep::CountBindingVerifyCheck();
			if (shader_write_stages != plan->shader_write_stages) {
				DrawPrep::ReportBindingMismatch("shader write stages");
			}
		}
	}
	if (shader_write_stages) {
		const CommitStats::SubScope barrier_scope(CommitStats::Sub::Barriers);
		if (DrawWriteSinkEnabled()) {
			// Queued without ending the instance (render.h, KYTY_DRAW_WRITE_SINK): a later draw of
			// this instance may pass it, anything else records it first.
			ShaderWriteBarrier(buffer, shader_write_stages);
			// An indirect draw can no longer rely on "no buffer write since the argument barrier
			// while this instance is active": it must end the instance and record a new one.
			m_indirect_barrier_rendering = 0;
		} else {
			{
				KYTY_GPU_OP_SITE("draw.write_barrier");
				m_context.GetCommandScheduler().EndRendering();
			}
			ShaderWriteBarrier(buffer, shader_write_stages);
		}
	}
	LogDrawPhase(draw.Name(), "DrawComplete");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, emit, 0x700u);
	}
	CommitStats::StepMark(CommitStats::Step::Occlusion);
	// KYTY_DRAW_RUN: whether the next draw may continue this one (drawPrep/drawRun.h), and in verify
	// mode what this draw's normal path computed against what a continuation would have reused.
	// N3b: a lean continuation is eligible as its record was (DrawRun's own
	// KYTY_DRAW_RUN_CONT_ELIGIBLE argument for the per-stage part: the same programs, textures and
	// attachments) once the per-draw conditions hold: no shader writes, a rendering instance, the
	// fast state reset, no debug dump, no key diagnostic; the record's rendering state has no clear
	// (it was eligible) and a continuation's feedback is none. Only the continuation's record update
	// runs.
	if (m_native_lean && (!DrawRun::Enabled() || m_run_verify || shader_write_stages ||
	                      buffer.ActiveRenderingSerial() == 0 ||
	                      !RenderStateFastEnabled(RenderStatePart::Reset) ||
	                      graphics_debug_dump_enabled() || DrawRun::KeyDiagEnabled() ||
	                      DrawRun::ContEligibleMode() >= 2 || m_run.record_writes_ok != DrawRun::WritesEnabled())) {
		NativeLeanDrop(Loader::GameCapture::kNlShape);
	}
	if (m_native_lean) {
		const CommitStats::SubScope record_scope(CommitStats::Sub::RunRecord);
		OwnedCounters::Add(DrawRun::GetTotals().eligible, 1);
		DrawRunRecordDraw(buffer, state, m_run_slice_offset, rendering, stages,
		                  bounded ? &written : nullptr, pipeline);
		m_run.writes = false;
		m_native_lean_done   = true;
		m_native_lean_reason = Loader::GameCapture::kNlLean;
	} else if (DrawRun::Enabled()) {
		const CommitStats::SubScope record_scope(CommitStats::Sub::RunRecord);
		if (m_run_verify) {
			DrawRunVerify(state, rendering, feedback_aspects, stages);
		}
		const auto is_attachment = [&state](ImageId id) {
			for (uint32_t i = 0; i < state.color_count; i++) {
				if (state.color_info[i].image_id == id) {
					return true;
				}
			}
			return static_cast<bool>(state.depth_info.image_id) && state.depth_info.image_id == id;
		};
		// KYTY_DRAW_RUN_WRITES: buffer writes (shader_write_stages only ever holds those) may seed a run.
		const bool writes_ok = DrawRun::WritesEnabled();
		// KYTY_DRAW_RUN_MESH: a direct mesh draw (no indirect arguments, no argument conversion).
		const bool mesh_ok = !mesh_active || (DrawRun::MeshRunsEnabled() && indirect == nullptr &&
		                                      !mesh_indirect);
		bool eligible = m_in_engine_commit && m_prepared_validated && m_run_key != 0 &&
		                mesh_ok &&
		                (indirect == nullptr || DrawRun::IndirectRunsEnabled()) &&
		                vertex_stages.size() == 1 &&
		                (!shader_write_stages || writes_ok) && !feedback_aspects &&
		                buffer.ActiveRenderingSerial() != 0 &&
		                RenderStateFastEnabled(RenderStatePart::Reset) &&
		                !graphics_debug_dump_enabled() &&
		                !rendering.depth_stencil_attachment.depth_clear &&
		                !rendering.depth_stencil_attachment.stencil_clear;
		for (uint32_t i = 0; eligible && i < rendering.num_color_attachments; i++) {
			eligible = !rendering.color_attachments[i].is_clear;
		}
		// KYTY_DRAW_RUN_CONT_ELIGIBLE (drawRun.h): a continuation's per-stage part is its record's.
		const auto cont_mode  = DrawRun::ContEligibleMode();
		const bool cont_known = cont_mode != 0 && m_run_active && !m_run_pipeswap &&
		                        m_run.record_writes_ok == writes_ok;
		const auto stages_eligible = [&] {
			bool ok = true;
			for (const auto* stage: stages) {
				ok = ok && stage->gds.buffer == nullptr &&
				     DrawRunProgramEligible(*stage->runtime->program, writes_ok);
				for (const auto& binding: stage->images) {
					// Storage bindings are transitioned with write access (a barrier every draw).
					ok = ok && !is_attachment(binding.image_id) &&
					     binding.desc.type != TextureCache::BindingType::Storage;
				}
			}
			return ok;
		};
		const auto stages_write = [&] {
			return writes_ok && std::any_of(stages.begin(), stages.end(), [](const auto* stage) {
				       return !DrawRunProgramEligible(*stage->runtime->program);
			       });
		};
		if (cont_known && cont_mode == 1) {
			// The record's: eligible (it exists), and its writes flag below.
		} else {
			eligible = eligible && stages_eligible();
		}
		if (cont_known && cont_mode >= 2 && eligible) {
			DrawRun::CountVerifyCheck();
			if (!stages_eligible() || stages_write() != m_run.writes) {
				DrawRun::ReportMismatch("continuation's record eligibility");
				if (cont_mode == 3) {
					EXIT("DrawRunVerify: a continuation's record eligibility differs\n");
				}
			}
		}
		if (DrawRun::KeyDiagEnabled()) [[unlikely]] {
			using DrawRun::KeyDiag::Inelig;
			auto reason = Inelig::None;
			if (!eligible) {
				bool color_clear = false;
				for (uint32_t i = 0; i < rendering.num_color_attachments; i++) {
					color_clear = color_clear || rendering.color_attachments[i].is_clear;
				}
				reason = !m_in_engine_commit                                          ? Inelig::NotEngine
				         : !m_prepared_validated                                      ? Inelig::NotValidated
				         : m_run_key == 0                                             ? Inelig::NoKey
				         : !mesh_ok                                                   ? Inelig::Mesh
				         : (indirect != nullptr && !DrawRun::IndirectRunsEnabled())   ? Inelig::Indirect
				         : vertex_stages.size() != 1                                  ? Inelig::VertexStages
				         : (shader_write_stages && !writes_ok)                        ? Inelig::Writes
				         : feedback_aspects                                           ? Inelig::Feedback
				         : buffer.ActiveRenderingSerial() == 0                        ? Inelig::NoRendering
				         : !RenderStateFastEnabled(RenderStatePart::Reset)            ? Inelig::RenderState
				         : graphics_debug_dump_enabled()                              ? Inelig::DebugDump
				         : rendering.depth_stencil_attachment.depth_clear             ? Inelig::DepthClear
				         : rendering.depth_stencil_attachment.stencil_clear           ? Inelig::StencilClear
				         : color_clear                                                ? Inelig::ColorClear
				                                                                      : Inelig::Count;
				if (reason == Inelig::Count) {
					for (const auto* stage: stages) {
						if (reason != Inelig::Count) {
							break;
						}
						if (stage->gds.buffer != nullptr) {
							reason = Inelig::Gds;
						} else if (!DrawRunProgramEligible(*stage->runtime->program, writes_ok)) {
							reason = Inelig::Program;
						} else {
							for (const auto& binding: stage->images) {
								if (is_attachment(binding.image_id)) {
									reason = Inelig::TexAttachment;
									break;
								}
								if (binding.desc.type == TextureCache::BindingType::Storage) {
									reason = Inelig::Storage;
									break;
								}
							}
						}
					}
					if (reason == Inelig::Count) {
						reason = Inelig::NotReached; // not found: shown as "notreached"
					}
				}
				DrawRun::KeyDiag::CountIneligible(reason);
			}
			m_key_diag.reason = reason;
		}
		if (m_native_lean_sample) {
			// N3b verify: the lean path records the continuation as eligible, without writes.
			m_native_lean_checked = true;
			if (!eligible || stages_write()) {
				NativeLeanMismatch(!eligible ? "record eligibility" : "record writes", 0);
			}
		}
		if (eligible) {
			OwnedCounters::Add(DrawRun::GetTotals().eligible, 1);
			const bool program_writes =
			    cont_known && cont_mode == 1 ? m_run.writes : stages_write();
			if (program_writes) {
				OwnedCounters::Add(DrawRun::GetTotals().write_records, 1);
			}
			const bool continued = m_run_active;
			if (mesh_active) {
				// KYTY_DRAW_RUN_MESH (only then does a mesh draw get here).
				OwnedCounters::Add(DrawRun::GetTotals().mesh_records, 1);
				if (continued || m_run_verify) {
					OwnedCounters::Add(DrawRun::GetTotals().mesh_continued, 1);
				}
			}
			DrawRunRecordDraw(buffer, state, m_run_slice_offset, rendering, stages,
			                  bounded ? &written : nullptr, pipeline);
			// A continuation keeps its record's programs (DrawRunCandidate), so the flag holds.
			m_run.writes = program_writes;
			if (!continued) {
				m_run.record_writes_ok = writes_ok;
			}
		}
		if (DrawRun::KeyDiagEnabled()) [[unlikely]] {
			DrawRunKeyDiagResolved(buffer, stages, pipeline, rendering, eligible && m_run.valid);
		}
	}
	// N3b: what a lean continuation of the record now in m_run reuses (every draw that reaches this
	// point sets it; a draw that returned earlier left m_run invalid, which ends the chain).
	m_native_record.slot = m_run.valid && m_native_op != 0 && indirect == nullptr && !mesh_active
	                           ? static_cast<int32_t>((m_native_op >> Loader::GameCapture::kNativeOpSlotShift) &
	                                                  Loader::GameCapture::kNativeSlotMask)
	                           : -1;
	m_native_record.barrier_safe = barrier_safe;
	m_native_record.plain_pixel  = plain_pixel;
	m_native_record.sink         = BarrierSinkEnabled();
	m_native_record.key              = m_run.key;
	m_native_record.activity         = m_run.activity;
	m_native_record.rendering_serial = m_run.rendering_serial;
	m_native_record.push_epoch       = m_run.push_epoch;
	CommitStats::StepMark(CommitStats::Step::RunRecord);
	if (CommitStats::Enabled()) [[unlikely]] {
		CommitStats::Mark(CommitStats::Phase::Emit);
		uint64_t textures = 0;
		uint64_t samplers = 0;
		for (const auto* stage: stages) {
			textures += stage->images.size();
			samplers += stage->samplers.size();
		}
		CommitStats::AddItems(CommitStats::Item::Textures, textures);
		CommitStats::AddItems(CommitStats::Item::Samplers, samplers);
		CommitStats::AddItems(CommitStats::Item::Transitions,
		                      Image::RecordedTransitions() - stats_transitions);
		auto shape = MakeCommitShape(buffer, stages, pipeline, rendering, m_dynamic_state,
		                             vertex_bindings, index_binding, draw, emit, index_source,
		                             shader_write_stages);
		shape.run_continued = m_run_active;
		CommitStats::NoteRecorded(shape);
		// The hashing above is the diagnostic's own cost: not counted in any phase.
		CommitStats::Skip();
	}
}

void RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	KYTY_GPU_OP_SITE("draw.index");
	KYTY_PROFILER_FUNCTION();
	const PassStats::Scope pass_stats(PassStats::Kind::Draw, PassStats::Enabled() ? m_context.GetTextureCache().Frame() : 0u);
	Profiler::CountFrameWork(Profiler::FrameWork::DrawIndex);

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	// S1e (KYTY_NATIVE_S1_PROFILE, nativeLearn.h Prof): the DrawIndex ticks of a native op of an S1
	// slot, charged to its class (native, or normal path) when it returns.
	struct NativeS1ProfileGuard {
		RenderExecutor& ex;
		uint64_t        start = 0;
		bool            on    = false;
		~NativeS1ProfileGuard() {
			if (on) [[unlikely]] {
				const bool native = ex.m_native_s1_done;
				NativeLearn::Prof::Draw(native, ex.m_native_s1_attempted, !native && ex.m_run_active,
				                        NativeLearn::Prof::Marker::Now() - start, ex.m_native_s1_attempt_ticks);
			}
		}
	} native_s1_profile {*this};
	m_native_s1_attempted = false;
	if (m_native_s0_op != 0 && Loader::GameCapture::NativeS1() && NativeLearn::Prof::Enabled() &&
	    Loader::GameCapture::NativeS1SlotEnabled(m_native_s0_op)) [[unlikely]] {
		native_s1_profile.on    = true;
		native_s1_profile.start = NativeLearn::Prof::Marker::Now();
		NativeLearn::Prof::SetSlot(m_native_s0_op);
	}
	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::DrainPendingOperations");
		m_context.GetCommandScheduler().PopReadyOperations();
	}
	CommitStats::StepMark(CommitStats::Step::PopReady);
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	// Self time here includes renderer-lock acquisition and setup outside the
	// separately timed preparation/execution phases.
	KYTY_PROFILER_DETAIL_BLOCK("Draw::SetupAndExecution");
	CommitStats::StepMark(CommitStats::Step::DebugInfo);
	RenderLockGuard lock(m_context.GetMutex());
	CommitStats::StepMark(CommitStats::Step::Lock);
	// Astro Bot PC (KYTY_NATIVE_MODE=s1, nativeLearn.h "S1"): a native op's continuation emitted
	// natively skips everything below. Any other draw ends the segment state it could rely on.
	m_native_draw_counter++;
	m_native_s1_done      = false;
	m_native_s1_predicted = false;
	if (m_native_s0_op != 0 && Loader::GameCapture::NativeS1()) [[unlikely]] {
		const uint64_t attempt_start = native_s1_profile.on ? NativeLearn::Prof::Marker::Now() : 0u;
		if (NativeS1Emit(submit_id, buffer, args)) {
			return;
		}
		if (native_s1_profile.on) [[unlikely]] {
			m_native_s1_attempted     = true;
			m_native_s1_attempt_ticks = NativeLearn::Prof::Marker::Now() - attempt_start;
		}
	}
	// KYTY_DRAW_RUN: only a draw that records through the eligible path seeds the next continuation.
	BeginDrawRun();
	CommitStats::StepMark(CommitStats::Step::BeginRun);
	// S1: the segment state of this draw is completed (or dropped) when it returns.
	struct NativeS1SegmentGuard {
		RenderExecutor&      ex;
		const CommandBuffer& buffer;
		bool                 valid = true;
		~NativeS1SegmentGuard() {
			if (ex.m_native_s1 != nullptr) [[unlikely]] {
				ex.NativeS1SegmentEnd(buffer, valid);
			}
		}
	} native_s1_segment {*this, buffer};
	// KYTY_NATIVE_MODE=n3b: the per-draw lean state (reset now and at every return).
	const NativeLeanDrawGuard native_lean(*this);
	if (args.index_count == 0 || args.instance_count == 0) {
		return;
	}

	// N3b: with the register test that decides them false, the target operations are not tried.
	const bool lean_pre = m_native_op != 0 && NativeLeanPrecheck(buffer);
	if (!lean_pre &&
	    (ConsumeMetadataColorOperation(buffer, args.render_target_slice_offset) ||
	     DepthStencilCopy(buffer) ||
	     ResolveColorTargets(buffer, args.render_target_slice_offset))) {
		if (m_native_lean_sample) {
			NativeLeanMismatch("target operation ran", 0);
		}
		FastCheck::Fail(FastCheck::Cause::TargetOp); // Astro Bot PC (KYTY_CP_PAR_FASTCHECK)
		FastCheck::Recorded();                       // a target operation records work
		ResetBindings();
		return;
	}
	CommitStats::StepMark(CommitStats::Step::TargetOps);

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	CommitStats::StepMark(CommitStats::Step::VsCheck);
	// N3b: the register checks passed for the previous draw, whose registers a lean continuation
	// has (its structure key covers every context and user-config register but the index offset
	// and object id); a draw that turns out not to be one runs them after its program refresh.
	if (!lean_pre) {
		RunDrawChecks(buffer, ucfg, m_binding_plan);
	}
	CommitStats::StepMark(CommitStats::Step::DrawChecks);

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		return;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_source.type               = vk::IndexType::eUint16;
			index_source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			index_source.type               = vk::IndexType::eUint32;
			index_source.guest_element_size = 4;
			break;
		case Prospero::IndexType::kIndex8:
			index_source.guest_element_size = 1;
			break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count,
	                        args.instance_count, args.first_instance};
	// The member state is reused; the render mutex held above makes it exclusive to this draw.
	auto& state = *m_draw_state;
	state.Reset();
	CommitStats::StepMark(CommitStats::Step::StateReset);
	CommitStats::Mark(CommitStats::Phase::Setup);
	m_mrt_split_slots   = 0;
	m_mrt_split_allowed = true;
	const bool prepared = PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state);
	m_mrt_split_allowed = false;
	if (!prepared) {
		if (lean_pre) {
			RunDrawChecks(buffer, ucfg, m_binding_plan); // N3b: not a continuation after all
		}
		m_mrt_split_slots = 0;
		ResetBindings();
		return;
	}
	NativeLeanDecide();
	if (lean_pre && !m_native_lean) {
		// N3b: not a lean continuation: the checks it skipped run now (they only read the same
		// registers and record nothing).
		RunDrawChecks(buffer, ucfg, m_binding_plan);
	}

	const bool primitive_restart = ResolvePrimitiveRestart(buffer, index_source,
	    MeshRestartEnabled() && state.vertex_info[0].stage.program->stage == ShaderType::Mesh);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		FastCheck::Fail(FastCheck::Cause::Index8); // Astro Bot PC (KYTY_CP_PAR_FASTCHECK)
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] = primitive_restart && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     args.index_addr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);

	DrawEmitInfo emit {};
	emit.vertex_offset  = vertex_offset + args.base_vertex;
	emit.first_instance = instance_offset;

	CommitStats::StepMark(CommitStats::Step::Restart);
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart);
	ResetBindings();
	CommitStats::StepMark(CommitStats::Step::ResetBindings);
	if (const auto split = std::exchange(m_mrt_split_slots, 0u); split != 0u) {
		// KYTY_MRT_SPLIT: the second pass, with only the slots the first left out.
		FastCheck::Fail(FastCheck::Cause::MrtSplit); // Astro Bot PC (KYTY_CP_PAR_FASTCHECK)
		native_s1_segment.valid = false; // S1: not a single draw of the segment's state
		NoteMrtSplit(split);
		m_run.valid = false;
		BeginDrawRun();
		// KYTY_MRT_SPLIT_REUSE: the first pass's preparation stays in `state` (Reset keeps it).
		const bool reuse    = MrtSplitReuseCandidate(state);
		const bool reuse_ps = state.ps_active;
		state.Reset();
		if (reuse) {
			state.vertex_stages_written = 1u; // RefreshShaders' value for a single vertex stage
			m_mrt_reuse_programs        = true;
			m_mrt_reuse_ps              = reuse_ps;
		}
		m_mrt_filter = split;
		if (PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
			ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
			                    primitive_restart);
		}
		m_mrt_reuse_programs = false;
		m_mrt_filter = ~0u;
		ResetBindings();
		m_run.valid = false;
		BeginDrawRun();
	}
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_GPU_OP_SITE("draw.auto");
	KYTY_PROFILER_FUNCTION();
	const PassStats::Scope pass_stats(PassStats::Kind::Draw, PassStats::Enabled() ? m_context.GetTextureCache().Frame() : 0u);
	Profiler::CountFrameWork(Profiler::FrameWork::DrawAuto);

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::DrainPendingOperations");
		m_context.GetCommandScheduler().PopReadyOperations();
	}
	CommitStats::StepMark(CommitStats::Step::PopReady);
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	KYTY_PROFILER_DETAIL_BLOCK("Draw::SetupAndExecution");
	CommitStats::StepMark(CommitStats::Step::DebugInfo);
	RenderLockGuard lock(m_context.GetMutex());
	CommitStats::StepMark(CommitStats::Step::Lock);
	m_native_draw_counter++; // S1: another draw ran
	m_native_s1_done = false;
	NativeS1BreakSegment();
	// KYTY_DRAW_RUN: only a draw that records through the eligible path seeds the next continuation.
	BeginDrawRun();
	CommitStats::StepMark(CommitStats::Step::BeginRun);
	// KYTY_NATIVE_MODE=n3b: the per-draw lean state (reset now and at every return).
	const NativeLeanDrawGuard native_lean(*this);
	if (args.vertex_count == 0 || args.instance_count == 0) {
		return;
	}

	// N3b: with the register test that decides them false, the target operations are not tried.
	const bool lean_pre = m_native_op != 0 && NativeLeanPrecheck(buffer);
	if (!lean_pre &&
	    (ConsumeMetadataColorOperation(buffer, args.render_target_slice_offset) ||
	     DepthStencilCopy(buffer) ||
	     ResolveColorTargets(buffer, args.render_target_slice_offset))) {
		if (m_native_lean_sample) {
			NativeLeanMismatch("target operation ran", 0);
		}
		FastCheck::Fail(FastCheck::Cause::TargetOp); // Astro Bot PC (KYTY_CP_PAR_FASTCHECK)
		FastCheck::Recorded();                       // a target operation records work
		ResetBindings();
		return;
	}
	CommitStats::StepMark(CommitStats::Step::TargetOps);

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	CommitStats::StepMark(CommitStats::Step::VsCheck);
	// N3b: the register checks passed for the previous draw, whose registers a lean continuation
	// has (its structure key covers every context and user-config register but the index offset
	// and object id); a draw that turns out not to be one runs them after its program refresh.
	if (!lean_pre) {
		RunDrawChecks(buffer, ucfg, m_binding_plan);
	}
	CommitStats::StepMark(CommitStats::Step::DrawChecks);

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto,
	                         args.vertex_count, args.instance_count, args.first_instance};

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		ResetBindings();
		return;
	}
	// The member state is reused; the render mutex held above makes it exclusive to this draw.
	auto& state = *m_draw_state;
	state.Reset();
	CommitStats::StepMark(CommitStats::Step::StateReset);
	CommitStats::Mark(CommitStats::Phase::Setup);
	m_mrt_split_slots   = 0;
	m_mrt_split_allowed = true;
	const bool prepared = PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state);
	m_mrt_split_allowed = false;
	if (!prepared) {
		if (lean_pre) {
			RunDrawChecks(buffer, ucfg, m_binding_plan); // N3b: not a continuation after all
		}
		m_mrt_split_slots = 0;
		ResetBindings();
		return;
	}
	NativeLeanDecide();
	if (lean_pre && !m_native_lean) {
		// N3b: not a lean continuation: the checks it skipped run now (they only read the same
		// registers and record nothing).
		RunDrawChecks(buffer, ucfg, m_binding_plan);
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		m_mrt_split_slots = 0;
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
	DrawEmitInfo emit {};
	emit.first_vertex = static_cast<uint32_t>(vertex_offset + static_cast<int32_t>(args.first_vertex));
	emit.first_instance = instance_offset;

	DrawIndexBufferSource index_source {};
	CommitStats::StepMark(CommitStats::Step::Restart);
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, false);
	ResetBindings();
	CommitStats::StepMark(CommitStats::Step::ResetBindings);
	if (const auto split = std::exchange(m_mrt_split_slots, 0u); split != 0u) {
		// KYTY_MRT_SPLIT: the second pass, with only the slots the first left out.
		FastCheck::Fail(FastCheck::Cause::MrtSplit); // Astro Bot PC (KYTY_CP_PAR_FASTCHECK)
		NoteMrtSplit(split);
		m_run.valid = false;
		BeginDrawRun();
		// KYTY_MRT_SPLIT_REUSE: the first pass's preparation stays in `state` (Reset keeps it).
		const bool reuse    = MrtSplitReuseCandidate(state);
		const bool reuse_ps = state.ps_active;
		state.Reset();
		if (reuse) {
			state.vertex_stages_written = 1u; // RefreshShaders' value for a single vertex stage
			m_mrt_reuse_programs        = true;
			m_mrt_reuse_ps              = reuse_ps;
		}
		m_mrt_filter = split;
		if (PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
			ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, false);
		}
		m_mrt_reuse_programs = false;
		m_mrt_filter = ~0u;
		ResetBindings();
		m_run.valid = false;
		BeginDrawRun();
	}
}

// Mirrors DrawIndex/DrawAuto with DrawOffsetSource::IndirectArgs, except that the counts, first
// index/vertex, vertex offset and first instance stay GPU data. Every check that needs a
// CPU-visible count returns false before a command is recorded.
// KYTY_CP_OP_STATS=1 (diagnostic): where a native indirect draw's time goes, one console line
// every 10 s ("IndirectDraws 10s: ...").
namespace IndirectStats {
enum class Part : uint8_t { Prepare, Execute, Count };
static bool Enabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CP_OP_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
static uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}
struct Totals {
	std::array<uint64_t, 2> ns {};
	std::array<uint64_t, 2> count {};
	uint64_t                last_print = 0;
};
static Totals g_totals;
class Timer {
public:
	explicit Timer(Part part): m_part(part), m_start(Enabled() ? NowNs() : 0) {}
	void Stop() {
		if (m_start == 0) {
			return;
		}
		const auto now = NowNs();
		g_totals.ns[static_cast<size_t>(m_part)] += now - m_start;
		g_totals.count[static_cast<size_t>(m_part)]++;
		m_start = 0;
		if (g_totals.last_print == 0) {
			g_totals.last_print = now;
		} else if (now - g_totals.last_print >= 10'000'000'000ull) {
			std::printf("IndirectDraws 10s: prepare %" PRIu64 " in %.1f ms, execute %" PRIu64 " in %.1f ms\n",
			            g_totals.count[0], static_cast<double>(g_totals.ns[0]) / 1e6, g_totals.count[1],
			            static_cast<double>(g_totals.ns[1]) / 1e6);
			std::fflush(stdout);
			g_totals = {};
			g_totals.last_print = now;
		}
	}

private:
	Part     m_part;
	uint64_t m_start;
};
} // namespace IndirectStats

bool RenderExecutor::DrawIndirectNative(uint64_t submit_id, CommandBuffer& buffer,
                                        const DrawIndirectSource& source) {
	KYTY_PROFILER_FUNCTION();
	m_native_draw_counter++; // S1: another draw ran
	NativeS1BreakSegment();
	EXIT_IF(buffer.IsInvalid() || source.args_addr == 0 || source.max_count == 0);
	const auto& graphics = m_context.GetGraphics();
	const auto& limits   = graphics.GetPhysicalDeviceProperties().limits;
	// Host command forms. firstInstance is GPU data, so its feature is always required.
	if (!graphics.draw_indirect_first_instance_enabled || (source.args_addr & 3u) != 0 ||
	    source.stride < source.RecordSize() || (source.stride & 3u) != 0 ||
	    source.max_count > limits.maxDrawIndirectCount ||
	    (source.max_count > 1 && !graphics.multi_draw_indirect_enabled) ||
	    (source.count_addr != 0 &&
	     (!graphics.draw_indirect_count_enabled || (source.count_addr & 3u) != 0)) ||
	    !m_context.IsMapped(source.args_addr, source.ArgsSize()) ||
	    (source.count_addr != 0 && !m_context.IsMapped(source.count_addr, sizeof(uint32_t)))) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallbackHost);
		return false;
	}

	DrawIndexBufferSource index_source {};
	if (source.indexed) {
		switch (static_cast<Prospero::IndexType>(source.index_type_and_size)) {
			case Prospero::IndexType::kIndex16:
				index_source.type               = vk::IndexType::eUint16;
				index_source.guest_element_size = 2;
				break;
			case Prospero::IndexType::kIndex32:
				index_source.type               = vk::IndexType::eUint32;
				index_source.guest_element_size = 4;
				break;
			case Prospero::IndexType::kIndex8:
				// The CPU path widens 8-bit indices to 16 bits on the host.
				if (!graphics.index_type_uint8_enabled) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallbackIndexBuffer);
					return false;
				}
				index_source.type               = vk::IndexType::eUint8;
				index_source.guest_element_size = 1;
				break;
			default: EXIT("unknown index_type_and_size: %u\n", source.index_type_and_size);
		}
		// The bound range is [INDEX_BASE, +INDEX_BUFFER_SIZE); firstIndex comes from each record.
		// The CPU path clamps the index count to INDEX_BUFFER_SIZE instead. Past that range the
		// host reads the neighbouring cached guest bytes, or robustBufferAccess values past the
		// host buffer (KYTY_INDIRECT_VALIDATE=1 reports such records). Without a size there is
		// no range to bind.
		if (source.index_base_addr == 0 || source.index_buffer_size == 0 ||
		    !m_context.IsMapped(source.index_base_addr, index_source.guest_element_size)) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallbackIndexBuffer);
			return false;
		}
		index_source.address = source.index_base_addr;
		index_source.size    = Libs::LibKernel::Memory::ClampRangeSize(
		    source.index_base_addr,
		    static_cast<uint64_t>(source.index_buffer_size) * index_source.guest_element_size);
	}

	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::DrainPendingOperations");
		m_context.GetCommandScheduler().PopReadyOperations();
	}
	CommitStats::StepMark(CommitStats::Step::PopReady);
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	const auto debug_op =
	    source.indexed ? CommandBufferDebugOp::DrawIndex : CommandBufferDebugOp::DrawIndexAuto;
	// Counts are unknown on the CPU: 0xFFFFFFFF, with the argument address in place of the index
	// or first-instance detail.
	buffer.SetDebugInfo(static_cast<uint32_t>(debug_op), submit_id, UINT32_MAX, 0, 1, UINT32_MAX,
	                    source.args_addr);

	KYTY_PROFILER_DETAIL_BLOCK("Draw::SetupAndExecution");
	RenderLockGuard lock(m_context.GetMutex());
	// KYTY_DRAW_RUN: an engine commit with KYTY_DRAW_RUN_INDIRECT and a prepared slot, otherwise
	// other command-processor work for the run certificate.
	BeginDrawRun();
	if (DrawMayRunTargetOperation(buffer)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallbackTargetOp);
		return false;
	}
	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndirectNative():Shader:\n");
		uc_print("GraphicsRenderDrawIndirectNative():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndirectNative():Parameters:\n"
		     "\t args_addr           = 0x%016" PRIx64 "\n"
		     "\t stride              = 0x%08" PRIx32 "\n"
		     "\t max_count           = 0x%08" PRIx32 "\n"
		     "\t count_addr          = 0x%016" PRIx64 "\n"
		     "\t indexed             = %u\n"
		     "\t index_base_addr     = 0x%016" PRIx64 "\n"
		     "\t index_buffer_size   = 0x%08" PRIx32 "\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n",
		     source.args_addr, source.stride, source.max_count, source.count_addr,
		     source.indexed ? 1u : 0u, source.index_base_addr, source.index_buffer_size,
		     source.index_type_and_size);
	}

	uc_check(ucfg);

	hw_check(buffer);

	// Legacy quads are split into per-quad host draws from the vertex count.
	if (ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallbackQuadList);
		return false;
	}
	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		return true;
	}
	bool primitive_restart = false;
	if (source.indexed) {
		const auto restart =
		    ResolveNativePrimitiveRestart(buffer, index_source.guest_element_size);
		if (!restart) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallbackRestart);
			return false;
		}
		primitive_restart = *restart;
	}

	const DrawCallInfo draw {debug_op, UINT32_MAX, UINT32_MAX, 0};
	// The member state is reused (as in DrawIndex/DrawAuto) instead of value-initialising about
	// 37 KB and allocating fresh preparation vectors; the render mutex makes it exclusive.
	auto& state = *m_draw_state;
	IndirectStats::Timer prep_timer(IndirectStats::Part::Prepare);
	state.Reset();
	if (!PrepareDrawRenderState(buffer, draw, 0, state)) {
		ResetBindings();
		return true;
	}
	prep_timer.Stop();
	DrawEmitInfo emit {};
	emit.indirect = &source;
	// Mesh draws derive group counts, restart segments and push data from the counts: on the GPU
	// (KYTY_NATIVE_INDIRECT_MESH, meshIndirect.h), or on the CPU path.
	if (state.vertex_info[0].stage.program->stage == ShaderType::Mesh) {
		const auto decline = [&](std::optional<Profiler::FrameEvent> reason) {
			if (reason) {
				Profiler::CountFrameEvent(*reason);
			}
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallbackMesh);
			ResetBindings();
			return false;
		};
		if (MeshIndirect::GetMode() == MeshIndirect::Mode::Off) {
			return decline(std::nullopt);
		}
		const auto& program         = *state.vertex_info[0].stage.program;
		const auto& mesh            = state.vertex_info[0].mesh;
		const auto& limits          = graphics.mesh_shader_properties;
		auto&       inputs          = emit.mesh_inputs;
		inputs.indexed              = source.indexed;
		inputs.index_base           = index_source.address;
		inputs.index_size           = index_source.guest_element_size;
		inputs.index_buffer_size    = source.indexed ? source.index_buffer_size : 0u;
		inputs.primitive_size       = mesh.InputPrimitiveSize();
		inputs.primitive_step       = mesh.InputPrimitiveStep();
		inputs.primitives_per_group = mesh.primitives_per_group;
		inputs.max_groups_x         = limits.maxMeshWorkGroupCount[0];
		inputs.max_groups_y         = limits.maxMeshWorkGroupCount[1];
		inputs.max_groups_total     = limits.maxMeshWorkGroupTotalCount;
		if (MeshIndirect::AlwaysEmpty(inputs)) {
			// No record can reach a primitive (every record's count is clamped below one input
			// primitive): the CPU path would read the records and draw nothing, returning before
			// the render targets are acquired (ExecutePreparedDraw) or, for zero counts, before
			// the render state is resolved (DrawIndex). Target operations and restart values only
			// an index scan decides took the CPU path above.
			MeshIndirect::GetTotals().always_empty.fetch_add(1, std::memory_order_relaxed);
			Profiler::CountFrameEvent(Profiler::FrameEvent::MeshIndirectAlwaysEmpty);
			ResetBindings();
			return true;
		}
		if (!MeshIndirect::ConversionEnabled()) {
			return decline(std::nullopt);
		}
		// One record without a count: the CPU path draws a multi-draw record by record.
		if (source.max_count != 1 || source.count_addr != 0) {
			return decline(Profiler::FrameEvent::MeshIndirectDeclinedMulti);
		}
		// Restart segments come from a CPU scan of the indices (KYTY_MESH_RESTART=1).
		if (primitive_restart && MeshRestartEnabled()) {
			return decline(Profiler::FrameEvent::MeshIndirectDeclinedRestart);
		}
		// A clear-enable draw clears its depth/stencil attachment when its rendering instance
		// begins, which a draw with zero counts on the CPU path never does.
		if (state.depth_info.image_id &&
		    (state.depth_info.depth_clear_enable || state.depth_info.stencil_clear_enable)) {
			return decline(Profiler::FrameEvent::MeshIndirectDeclinedClear);
		}
		// Programs generated with the parameter-block loads (the codegen option follows the
		// mode); the CPU path reports an unusable primitive group itself.
		if (!ShaderRecompiler::GetCodegenOptions().mesh_indirect_params ||
		    program.stage != ShaderType::Mesh || mesh.primitives_per_group == 0) {
			return decline(Profiler::FrameEvent::MeshIndirectDeclinedProgram);
		}
		emit.mesh_indirect = true;
	}
	if (!source.indexed && Prospero::IsRectList(ucfg.GetPrimType()) &&
	    state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, source.index_type_and_size,
	                     reinterpret_cast<const void*>(source.index_base_addr));
	IndirectStats::Timer exec_timer(IndirectStats::Part::Execute);
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart);
	exec_timer.Stop();
	ResetBindings();
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectNative);
	return true;
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer, uint32_t render_target_slice_offset) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, render_target_slice_offset, 1, true, true);
	if (!src.image_id || !dst.image_id) {
		return false;
	}
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source, {src.guest_mip_level, 1, src.guest_array_layer, 1},
	                    {dst.guest_mip_level, 1, dst.guest_array_layer, 1});
	return true;
}

} // namespace Libs::Graphics
