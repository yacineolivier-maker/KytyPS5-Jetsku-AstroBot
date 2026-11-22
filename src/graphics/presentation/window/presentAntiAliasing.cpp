#include "graphics/presentation/window/presentAntiAliasing.h"

#include "common/assert.h"
#include "gpu_blit_shaders/gpu_present_smaa_blend_frag_spv.h"
#include "gpu_blit_shaders/gpu_present_smaa_blend_vert_spv.h"
#include "gpu_blit_shaders/gpu_present_smaa_edges_high_frag_spv.h"
#include "gpu_blit_shaders/gpu_present_smaa_edges_high_vert_spv.h"
#include "gpu_blit_shaders/gpu_present_smaa_edges_ultra_frag_spv.h"
#include "gpu_blit_shaders/gpu_present_smaa_edges_ultra_vert_spv.h"
#include "gpu_blit_shaders/gpu_present_smaa_weights_high_frag_spv.h"
#include "gpu_blit_shaders/gpu_present_smaa_weights_high_vert_spv.h"
#include "gpu_blit_shaders/gpu_present_smaa_weights_ultra_frag_spv.h"
#include "gpu_blit_shaders/gpu_present_smaa_weights_ultra_vert_spv.h"

// Tables precalculees de SMAA (sources officielles, 3rdparty/smaa, licence MIT).
#include "AreaTex.h"
#include "SearchTex.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <span>
#include <string>

namespace Libs::Graphics {

namespace {

constexpr vk::Format kEdgesFormat   = vk::Format::eR8G8Unorm;
constexpr vk::Format kWeightsFormat = vk::Format::eR8G8B8A8Unorm;
// Une cible remplacee sert encore aux images en vol : la chaine en a au plus quelques-unes
// (AcquireNextImage attend la fin de l'image de meme indice avant chaque Record).
constexpr uint32_t kRetireFrames = 16;

struct SmaaConstants {
	float metrics[4];
};
static_assert(sizeof(SmaaConstants) == 16);

uint64_t NowMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

bool TimingRequested() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PRESENT_AA_TIMING");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

const char* ModeName(PresentAaMode mode) {
	switch (mode) {
		case PresentAaMode::SmaaHigh: return "SMAA 1x (preset High)";
		case PresentAaMode::SmaaUltra: return "SMAA 1x (preset Ultra)";
		default: return "aucun";
	}
}

void ImageBarrier(vk::CommandBuffer command, vk::Image image, vk::PipelineStageFlags2 src_stage,
                  vk::AccessFlags2 src_access, vk::ImageLayout old_layout,
                  vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
                  vk::ImageLayout new_layout) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = src_stage;
	barrier.srcAccessMask       = src_access;
	barrier.dstStageMask        = dst_stage;
	barrier.dstAccessMask       = dst_access;
	barrier.oldLayout           = old_layout;
	barrier.newLayout           = new_layout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image;
	barrier.subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

void CreateImage2D(GraphicContext& graphics, VulkanImage& image, vk::ImageView& view,
                   vk::Format format, vk::Extent2D extent, vk::ImageUsageFlags usage,
                   const char* what) {
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = usage;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, image)) {
		EXIT("KYTY_PRESENT_AA: allocation impossible (%s, %ux%u, format %d)\n", what, extent.width,
		     extent.height, static_cast<int>(format));
	}
	vk::ImageViewCreateInfo info {};
	info.image            = image.image;
	info.viewType         = vk::ImageViewType::e2D;
	info.format           = format;
	info.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	RequireVulkanSuccess(graphics.device.createImageView(&info, nullptr, &view), what);
}

void DestroyImage2D(GraphicContext& graphics, VulkanImage& image, vk::ImageView& view) {
	if (view != nullptr) {
		graphics.device.destroyImageView(view, nullptr);
		view = nullptr;
	}
	if (image.image != nullptr) {
		graphics.DeleteImage(image);
	}
}

} // namespace

PresentAaMode PresentAaRequested() noexcept {
	static const PresentAaMode mode = [] {
		const auto* value = std::getenv("KYTY_PRESENT_AA");
		if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
			return PresentAaMode::Off;
		}
		if (std::strcmp(value, "smaa") == 0 || std::strcmp(value, "1") == 0) {
			return PresentAaMode::SmaaHigh;
		}
		if (std::strcmp(value, "smaa_ultra") == 0) {
			return PresentAaMode::SmaaUltra;
		}
		::printf("PresentAA: valeur KYTY_PRESENT_AA=%s inconnue (attendu : 0, smaa, smaa_ultra), "
		         "antialiasing desactive\n",
		         value);
		std::fflush(stdout);
		return PresentAaMode::Off;
	}();
	return mode;
}

// Swapchain::Destroy appelle Release (apres queue.waitIdle) ; le destructeur ne touche pas au GPU.
PresentAntiAliasing::~PresentAntiAliasing() = default;

bool PresentAntiAliasing::Supports(GraphicContext& graphics, vk::Format format) const {
	switch (format) {
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eA2B10G10R10UnormPack32: break;
		default: return false; // flottants (HDR), sRGB, formats exotiques : image laissee telle quelle
	}
	const auto features = graphics.GetFormatProperties(format).optimalTilingFeatures;
	const auto required = vk::FormatFeatureFlagBits::eColorAttachment |
	                      vk::FormatFeatureFlagBits::eSampledImage |
	                      vk::FormatFeatureFlagBits::eSampledImageFilterLinear |
	                      vk::FormatFeatureFlagBits::eBlitSrc;
	return (features & required) == required;
}

vk::Pipeline PresentAntiAliasing::CreatePipeline(vk::Device device, uint32_t pass,
                                                 vk::Format format) const {
	const bool ultra = m_mode == PresentAaMode::SmaaUltra;
	std::span<const uint32_t> vertex_code;
	std::span<const uint32_t> fragment_code;
	switch (pass) {
		case 0:
			vertex_code   = ultra ? std::span<const uint32_t>(GPU_PRESENT_SMAA_EDGES_ULTRA_VERT_SPV)
			                      : std::span<const uint32_t>(GPU_PRESENT_SMAA_EDGES_HIGH_VERT_SPV);
			fragment_code = ultra ? std::span<const uint32_t>(GPU_PRESENT_SMAA_EDGES_ULTRA_FRAG_SPV)
			                      : std::span<const uint32_t>(GPU_PRESENT_SMAA_EDGES_HIGH_FRAG_SPV);
			break;
		case 1:
			vertex_code = ultra ? std::span<const uint32_t>(GPU_PRESENT_SMAA_WEIGHTS_ULTRA_VERT_SPV)
			                    : std::span<const uint32_t>(GPU_PRESENT_SMAA_WEIGHTS_HIGH_VERT_SPV);
			fragment_code =
			    ultra ? std::span<const uint32_t>(GPU_PRESENT_SMAA_WEIGHTS_ULTRA_FRAG_SPV)
			          : std::span<const uint32_t>(GPU_PRESENT_SMAA_WEIGHTS_HIGH_FRAG_SPV);
			break;
		default:
			vertex_code   = GPU_PRESENT_SMAA_BLEND_VERT_SPV;
			fragment_code = GPU_PRESENT_SMAA_BLEND_FRAG_SPV;
			break;
	}
	const auto vertex   = CompileSPV(vertex_code, device);
	const auto fragment = CompileSPV(fragment_code, device);
	std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
	stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
	stages[0].module = vertex;
	stages[0].pName  = "main";
	stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
	stages[1].module = fragment;
	stages[1].pName  = "main";
	vk::PipelineVertexInputStateCreateInfo   vertex_input {};
	vk::PipelineInputAssemblyStateCreateInfo assembly {};
	assembly.topology = vk::PrimitiveTopology::eTriangleList;
	vk::PipelineViewportStateCreateInfo viewport {};
	viewport.viewportCount = 1;
	viewport.scissorCount  = 1;
	vk::PipelineRasterizationStateCreateInfo rasterization {};
	rasterization.lineWidth = 1.0f;
	vk::PipelineMultisampleStateCreateInfo multisample {};
	multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
	vk::PipelineColorBlendAttachmentState attachment {};
	attachment.blendEnable    = VK_FALSE;
	attachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
	                            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
	vk::PipelineColorBlendStateCreateInfo blend {};
	blend.attachmentCount = 1;
	blend.pAttachments    = &attachment;
	const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynamic {};
	dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic.pDynamicStates    = dynamic_states.data();
	vk::PipelineRenderingCreateInfo rendering {};
	rendering.colorAttachmentCount    = 1;
	rendering.pColorAttachmentFormats = &format;
	vk::GraphicsPipelineCreateInfo create {};
	create.pNext               = &rendering;
	create.stageCount          = static_cast<uint32_t>(stages.size());
	create.pStages             = stages.data();
	create.pVertexInputState   = &vertex_input;
	create.pInputAssemblyState = &assembly;
	create.pViewportState      = &viewport;
	create.pRasterizationState = &rasterization;
	create.pMultisampleState   = &multisample;
	create.pColorBlendState    = &blend;
	create.pDynamicState       = &dynamic;
	create.layout              = m_layout;
	vk::Pipeline pipeline      = nullptr;
	RequireVulkanSuccess(device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &pipeline),
	                     "KYTY_PRESENT_AA: create SMAA pipeline");
	device.destroyShaderModule(fragment, nullptr);
	device.destroyShaderModule(vertex, nullptr);
	return pipeline;
}

void PresentAntiAliasing::CreateShared(GraphicContext& graphics, vk::CommandBuffer command) {
	auto device = graphics.device;
	m_mode      = PresentAaRequested();

	vk::SamplerCreateInfo sampler {};
	sampler.magFilter    = vk::Filter::eLinear;
	sampler.minFilter    = vk::Filter::eLinear;
	sampler.mipmapMode   = vk::SamplerMipmapMode::eNearest;
	sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	sampler.maxLod       = 0.0f;
	RequireVulkanSuccess(device.createSampler(&sampler, nullptr, &m_sampler),
	                     "KYTY_PRESENT_AA: create sampler");

	std::array<vk::DescriptorSetLayoutBinding, 3> bindings {};
	for (uint32_t i = 0; i < bindings.size(); i++) {
		bindings[i].binding         = i;
		bindings[i].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags      = vk::ShaderStageFlagBits::eFragment;
	}
	vk::DescriptorSetLayoutCreateInfo descriptors {};
	descriptors.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptors.bindingCount = static_cast<uint32_t>(bindings.size());
	descriptors.pBindings    = bindings.data();
	RequireVulkanSuccess(device.createDescriptorSetLayout(&descriptors, nullptr, &m_descriptors),
	                     "KYTY_PRESENT_AA: create descriptor layout");
	const vk::PushConstantRange constants {
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
	    sizeof(SmaaConstants)};
	vk::PipelineLayoutCreateInfo layout {};
	layout.setLayoutCount         = 1;
	layout.pSetLayouts            = &m_descriptors;
	layout.pushConstantRangeCount = 1;
	layout.pPushConstantRanges    = &constants;
	RequireVulkanSuccess(device.createPipelineLayout(&layout, nullptr, &m_layout),
	                     "KYTY_PRESENT_AA: create pipeline layout");
	m_pipelines[0] = CreatePipeline(device, 0, kEdgesFormat);
	m_pipelines[1] = CreatePipeline(device, 1, kWeightsFormat);

	// Tables AreaTex (R8G8, 160x560) et SearchTex (R8, 64x16), envoyees une fois.
	static_assert(sizeof(areaTexBytes) == AREATEX_SIZE && sizeof(searchTexBytes) == SEARCHTEX_SIZE);
	const auto           lut_usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	CreateImage2D(graphics, m_area, m_area_view, vk::Format::eR8G8Unorm,
	              {AREATEX_WIDTH, AREATEX_HEIGHT}, lut_usage, "KYTY_PRESENT_AA: AreaTex");
	CreateImage2D(graphics, m_search, m_search_view, vk::Format::eR8Unorm,
	              {SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT}, lut_usage, "KYTY_PRESENT_AA: SearchTex");
	vk::BufferCreateInfo buffer {};
	buffer.size  = AREATEX_SIZE + SEARCHTEX_SIZE;
	buffer.usage = vk::BufferUsageFlagBits::eTransferSrc;
	VmaAllocationCreateInfo allocation {};
	allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	allocation.flags =
	    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	VmaAllocationInfo mapped {};
	if (vmaCreateBuffer(graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer),
	                    &allocation, &m_staging, &m_staging_allocation, &mapped) != VK_SUCCESS) {
		EXIT("KYTY_PRESENT_AA: allocation impossible du tampon d'envoi des tables SMAA\n");
	}
	auto* bytes = static_cast<uint8_t*>(mapped.pMappedData);
	std::memcpy(bytes, areaTexBytes, AREATEX_SIZE);
	std::memcpy(bytes + AREATEX_SIZE, searchTexBytes, SEARCHTEX_SIZE);
	RequireVulkanSuccess(static_cast<vk::Result>(vmaFlushAllocation(
	                         graphics.allocator, m_staging_allocation, 0, buffer.size)),
	                     "KYTY_PRESENT_AA: flush SMAA tables");
	for (auto* image: {&m_area, &m_search}) {
		ImageBarrier(command, image->image, vk::PipelineStageFlagBits2::eNone,
		             vk::AccessFlagBits2::eNone, vk::ImageLayout::eUndefined,
		             vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		             vk::ImageLayout::eTransferDstOptimal);
	}
	vk::BufferImageCopy copy {};
	copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.bufferOffset     = 0;
	copy.imageExtent      = vk::Extent3D {AREATEX_WIDTH, AREATEX_HEIGHT, 1};
	command.copyBufferToImage(m_staging, m_area.image, vk::ImageLayout::eTransferDstOptimal, 1,
	                          &copy);
	copy.bufferOffset = AREATEX_SIZE;
	copy.imageExtent  = vk::Extent3D {SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, 1};
	command.copyBufferToImage(m_staging, m_search.image, vk::ImageLayout::eTransferDstOptimal, 1,
	                          &copy);
	for (auto* image: {&m_area, &m_search}) {
		ImageBarrier(command, image->image, vk::PipelineStageFlagBits2::eTransfer,
		             vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eTransferDstOptimal,
		             vk::PipelineStageFlagBits2::eFragmentShader,
		             vk::AccessFlagBits2::eShaderSampledRead,
		             vk::ImageLayout::eShaderReadOnlyOptimal);
	}

	if (TimingRequested()) {
		const auto& limits = graphics.GetPhysicalDeviceProperties().limits;
		if (limits.timestampComputeAndGraphics == VK_FALSE || limits.timestampPeriod <= 0.0f) {
			::printf("PresentAA: chronometrage GPU indisponible sur cette carte\n");
			std::fflush(stdout);
		} else {
			vk::QueryPoolCreateInfo queries {};
			queries.queryType  = vk::QueryType::eTimestamp;
			queries.queryCount = 2 * kTimingSlots;
			RequireVulkanSuccess(device.createQueryPool(&queries, nullptr, &m_queries),
			                     "KYTY_PRESENT_AA: create timestamp query pool");
			m_slot_pending.fill(false);
			m_report_due_ms = NowMs() + 10000;
		}
	}
}

std::unique_ptr<PresentAntiAliasing::Targets>
PresentAntiAliasing::CreateTargets(GraphicContext& graphics, vk::Format format,
                                   vk::Extent2D extent) const {
	auto       targets = std::make_unique<Targets>();
	const auto usage   = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
	CreateImage2D(graphics, targets->edges, targets->edges_view, kEdgesFormat, extent, usage,
	              "KYTY_PRESENT_AA: edges");
	CreateImage2D(graphics, targets->weights, targets->weights_view, kWeightsFormat, extent, usage,
	              "KYTY_PRESENT_AA: blend weights");
	CreateImage2D(graphics, targets->output, targets->output_view, format, extent,
	              usage | vk::ImageUsageFlagBits::eTransferSrc, "KYTY_PRESENT_AA: output");
	targets->blend  = CreatePipeline(graphics.device, 2, format);
	targets->format = format;
	targets->extent = extent;
	return targets;
}

void PresentAntiAliasing::DestroyTargets(GraphicContext& graphics, Targets& targets) {
	DestroyImage2D(graphics, targets.edges, targets.edges_view);
	DestroyImage2D(graphics, targets.weights, targets.weights_view);
	DestroyImage2D(graphics, targets.output, targets.output_view);
	graphics.device.destroyPipeline(targets.blend, nullptr);
	targets.blend = nullptr;
}

void PresentAntiAliasing::CollectRetired(GraphicContext& graphics, bool all) {
	for (size_t i = 0; i < m_retired.size();) {
		auto& targets = *m_retired[i];
		if (all || targets.frames_left == 0) {
			DestroyTargets(graphics, targets);
			m_retired.erase(m_retired.begin() + static_cast<std::ptrdiff_t>(i));
			continue;
		}
		targets.frames_left--;
		i++;
	}
}

void PresentAntiAliasing::ReadTimings(GraphicContext& graphics, uint32_t slot) {
	if (!m_slot_pending[slot]) {
		return;
	}
	m_slot_pending[slot] = false;
	std::array<uint64_t, 2> ticks {};
	const auto result = graphics.device.getQueryPoolResults(
	    m_queries, slot * 2, 2, sizeof(ticks), ticks.data(), sizeof(uint64_t),
	    vk::QueryResultFlagBits::e64);
	if (result != vk::Result::eSuccess || ticks[1] < ticks[0]) {
		return; // pas encore pret (rare) : echantillon ignore
	}
	const double ms = static_cast<double>(ticks[1] - ticks[0]) *
	                  static_cast<double>(graphics.GetPhysicalDeviceProperties().limits.timestampPeriod) /
	                  1.0e6;
	m_sum_ms += ms;
	m_max_ms = std::max(m_max_ms, ms);
	m_samples++;
}

bool PresentAntiAliasing::Record(GraphicContext& graphics, vk::CommandBuffer command,
                                 vk::ImageView source_view, vk::Format format,
                                 vk::Extent2D extent) {
	if (PresentAaRequested() == PresentAaMode::Off || source_view == nullptr || extent.width == 0 ||
	    extent.height == 0) {
		return false;
	}
	if (!Supports(graphics, format)) {
		if (!m_reported_unsupported) {
			m_reported_unsupported = true;
			::printf("PresentAA: image du jeu au format %s non lissee (format HDR/flottant ou non "
			         "pris en charge), presentation inchangee\n",
			         vk::to_string(format).c_str());
			std::fflush(stdout);
		}
		return false;
	}
	if (m_layout == nullptr) {
		CreateShared(graphics, command);
	}
	CollectRetired(graphics, false);
	if (m_targets == nullptr || m_targets->format != format ||
	    m_targets->extent.width != extent.width || m_targets->extent.height != extent.height) {
		if (m_targets != nullptr) {
			m_targets->frames_left = kRetireFrames;
			m_retired.push_back(std::move(m_targets));
		}
		m_targets = CreateTargets(graphics, format, extent);
		::printf("PresentAA: %s actif sur l'image du jeu %ux%u (%s)%s\n", ModeName(m_mode),
		         extent.width, extent.height, vk::to_string(format).c_str(),
		         m_queries != nullptr ? ", cout GPU ecrit toutes les 10 s" : "");
		std::fflush(stdout);
	}
	auto& targets = *m_targets;

	const uint32_t slot = m_slot;
	if (m_queries != nullptr) {
		ReadTimings(graphics, slot);
		command.resetQueryPool(m_queries, slot * 2, 2);
	}

	// Les trois cibles sont reecrites entierement : contenu precedent abandonne (eUndefined).
	// L'image precedente a pu les lire (nuanceurs, blit) ou les ecrire.
	const auto previous_stages = vk::PipelineStageFlagBits2::eColorAttachmentOutput |
	                             vk::PipelineStageFlagBits2::eFragmentShader |
	                             vk::PipelineStageFlagBits2::eTransfer;
	for (auto* image: {&targets.edges, &targets.weights, &targets.output}) {
		ImageBarrier(command, image->image, previous_stages,
		             vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined,
		             vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		             vk::AccessFlagBits2::eColorAttachmentWrite,
		             vk::ImageLayout::eColorAttachmentOptimal);
	}
	if (m_queries != nullptr) {
		// eAllCommands : l'horodatage attend la fin du travail precedent, la mesure ne couvre
		// que les trois passes.
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, m_queries, slot * 2);
	}

	const SmaaConstants constants {{1.0f / static_cast<float>(extent.width),
	                                1.0f / static_cast<float>(extent.height),
	                                static_cast<float>(extent.width),
	                                static_cast<float>(extent.height)}};
	const vk::Viewport viewport {0.0f, 0.0f, static_cast<float>(extent.width),
	                             static_cast<float>(extent.height), 0.0f, 1.0f};
	const vk::Rect2D   scissor {{0, 0}, extent};

	const auto pass = [&](vk::Pipeline pipeline, vk::ImageView target, bool clear,
	                      std::initializer_list<vk::ImageView> inputs) {
		std::array<vk::DescriptorImageInfo, 3> images {};
		std::array<vk::WriteDescriptorSet, 3>  writes {};
		uint32_t                               count = 0;
		for (const auto view: inputs) {
			images[count] = {m_sampler, view, vk::ImageLayout::eShaderReadOnlyOptimal};
			writes[count].dstBinding      = count;
			writes[count].descriptorCount = 1;
			writes[count].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
			writes[count].pImageInfo      = &images[count];
			count++;
		}
		command.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_layout, 0, count,
		                             writes.data());
		command.pushConstants(m_layout,
		                      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
		                      0, sizeof(constants), &constants);
		vk::RenderingAttachmentInfo attachment {};
		attachment.imageView   = target;
		attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		attachment.loadOp      = clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eDontCare;
		attachment.storeOp     = vk::AttachmentStoreOp::eStore;
		// clearValue reste a zero (initialisation de valeur).
		vk::RenderingInfo rendering {};
		rendering.renderArea.extent    = extent;
		rendering.layerCount           = 1;
		rendering.colorAttachmentCount = 1;
		rendering.pColorAttachments    = &attachment;
		command.beginRendering(&rendering);
		command.setViewport(0, 1, &viewport);
		command.setScissor(0, 1, &scissor);
		command.draw(3, 1, 0, 0);
		command.endRendering();
	};
	const auto to_read = [&](VulkanImage& image) {
		ImageBarrier(command, image.image, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		             vk::AccessFlagBits2::eColorAttachmentWrite,
		             vk::ImageLayout::eColorAttachmentOptimal,
		             vk::PipelineStageFlagBits2::eFragmentShader,
		             vk::AccessFlagBits2::eShaderSampledRead,
		             vk::ImageLayout::eShaderReadOnlyOptimal);
	};

	// 1. Bords (luminance) ; les pixels sans bord sont abandonnes, d'ou l'effacement a zero.
	pass(m_pipelines[0], targets.edges_view, true, {source_view});
	to_read(targets.edges);
	// 2. Poids de melange (recherches le long des bords, tables AreaTex / SearchTex).
	pass(m_pipelines[1], targets.weights_view, false,
	     {targets.edges_view, m_area_view, m_search_view});
	to_read(targets.weights);
	// 3. Melange du voisinage vers l'image de sortie.
	pass(targets.blend, targets.output_view, false, {source_view, targets.weights_view});
	targets.output.state = {vk::PipelineStageFlagBits2::eColorAttachmentOutput,
	                        vk::AccessFlagBits2::eColorAttachmentWrite,
	                        vk::ImageLayout::eColorAttachmentOptimal};

	if (m_queries != nullptr) {
		command.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, m_queries, slot * 2 + 1);
		m_slot_pending[slot] = true;
		m_slot               = (slot + 1) % kTimingSlots;
		const auto now       = NowMs();
		if (now >= m_report_due_ms) {
			if (m_samples > 0) {
				::printf("PresentAA: %s %ux%u : cout GPU moyen %.3f ms par image (max %.3f ms, "
				         "%llu images mesurees en 10 s)\n",
				         ModeName(m_mode), extent.width, extent.height,
				         m_sum_ms / static_cast<double>(m_samples), m_max_ms,
				         static_cast<unsigned long long>(m_samples));
				std::fflush(stdout);
			}
			m_sum_ms        = 0.0;
			m_max_ms        = 0.0;
			m_samples       = 0;
			m_report_due_ms = now + 10000;
		}
	}
	return true;
}

void PresentAntiAliasing::TransitOutput(vk::CommandBuffer command, vk::ImageLayout layout,
                                        vk::AccessFlags2 access, vk::PipelineStageFlags2 stage) {
	auto& image = m_targets->output;
	ImageBarrier(command, image.image, image.state.pl_stage, image.state.access_mask,
	             image.state.layout, stage, access, layout);
	image.state = {stage, access, layout};
}

void PresentAntiAliasing::Release(GraphicContext& graphics) {
	auto device = graphics.device;
	CollectRetired(graphics, true);
	if (m_targets != nullptr) {
		DestroyTargets(graphics, *m_targets);
		m_targets.reset();
	}
	for (auto& pipeline: m_pipelines) {
		device.destroyPipeline(pipeline, nullptr);
		pipeline = nullptr;
	}
	DestroyImage2D(graphics, m_area, m_area_view);
	DestroyImage2D(graphics, m_search, m_search_view);
	if (m_staging != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics.allocator, m_staging, m_staging_allocation);
		m_staging            = VK_NULL_HANDLE;
		m_staging_allocation = nullptr;
	}
	device.destroyQueryPool(m_queries, nullptr);
	device.destroyPipelineLayout(m_layout, nullptr);
	device.destroyDescriptorSetLayout(m_descriptors, nullptr);
	device.destroySampler(m_sampler, nullptr);
	m_queries     = nullptr;
	m_layout      = nullptr;
	m_descriptors = nullptr;
	m_sampler     = nullptr;
	m_slot_pending.fill(false);
	m_slot    = 0;
	m_sum_ms  = 0.0;
	m_max_ms  = 0.0;
	m_samples = 0;
}

} // namespace Libs::Graphics
