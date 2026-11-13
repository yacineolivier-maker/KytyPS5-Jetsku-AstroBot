#include "graphics/presentation/window/fsr3FrameGen.h"

#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "3rdparty_local/ffx_api/ffx_api.h"
#include "3rdparty_local/ffx_api/ffx_api_loader.h"
#include "3rdparty_local/ffx_api/ffx_framegeneration.h"
#include "3rdparty_local/ffx_api/vk/ffx_api_vk.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Libs::Graphics::Fsr3 {

namespace {

struct Neutral {
	VkImage           image  = VK_NULL_HANDLE;
	VkDeviceMemory    memory = VK_NULL_HANDLE;
	VkImageCreateInfo info {};
};

struct State {
	bool          dll_tried = false;
	HMODULE       dll       = nullptr;
	ffxFunctions  f {};
	bool          active    = false;
	ffxContext    swapchain_ctx = nullptr;
	ffxContext    fg_ctx        = nullptr;
	ffxQueryDescSwapchainReplacementFunctionsVK repl {};
	VkDevice         device    = VK_NULL_HANDLE;
	VkPhysicalDevice physical  = VK_NULL_HANDLE;
	VkSwapchainKHR   swapchain = VK_NULL_HANDLE;
	VkExtent2D       extent {};
	Neutral          depth;
	Neutral          motion;
	bool             neutral_ready = false;
	uint64_t         frame_id      = 0;
	std::chrono::steady_clock::time_point last;
	GraphicContext*  graphics = nullptr;
	uint32_t         logged_errors = 0;
};

State& S() {
	static State s;
	return s;
}

// Diagnostic des gels : etape FSR 3 en cours et instant de la derniere presentation aboutie. Un fil de
// surveillance ecrit l'etape bloquee si aucune presentation n'aboutit pendant 10 s.
enum class Stage : int { Aucune, CreationChaine, CreationContexte, Acquisition, Preparation, Presentation,
	                     DestructionContexte, DestructionChaine };
const char* StageName(int s) {
	static const char* names[] = {"aucune", "creation de la chaine", "creation du contexte", "acquisition d'image",
	                              "preparation", "presentation", "destruction du contexte", "destruction de la chaine"};
	return s >= 0 && s < 8 ? names[s] : "?";
}
std::atomic<int>     g_stage {0};
std::atomic<int64_t> g_stage_since {0};
std::atomic<int64_t> g_last_present {0};
int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}
struct StageScope {
	int previous;
	explicit StageScope(Stage s): previous(g_stage.exchange(static_cast<int>(s))) { g_stage_since = NowMs(); }
	~StageScope() {
		g_stage = previous;
		g_stage_since = NowMs();
	}
};
void StartWatchdog() {
	static std::atomic<bool> started {false};
	if (started.exchange(true)) {
		return;
	}
	g_last_present = NowMs();
	CreateThread(
	    nullptr, 0,
	    [](void*) -> DWORD {
		    SetThreadDescription(GetCurrentThread(), L"FSR3 surveillance");
		    int64_t reported_for = -1;
		    for (;;) {
			    Sleep(2000);
			    if (!S().active) {
				    continue;
			    }
			    const int64_t now  = NowMs();
			    const int64_t last = g_last_present.load();
			    if (now - last >= 10000 && reported_for != last) {
				    reported_for = last;
				    std::printf("FSR 3 : aucune presentation depuis %lld ms ; etape en cours : %s depuis %lld ms\n",
				                static_cast<long long>(now - last), StageName(g_stage.load()),
				                static_cast<long long>(now - g_stage_since.load()));
				    std::fflush(stdout);
			    }
		    }
	    },
	    nullptr, 0, nullptr);
}

void Log(const char* what, ffxReturnCode_t code) {
	auto& s = S();
	if (code != FFX_API_RETURN_OK && s.logged_errors++ < 16) {
		std::printf("FSR 3 : %s a echoue (code %u)\n", what, static_cast<unsigned>(code));
		std::fflush(stdout);
	}
}

bool LoadDll() {
	auto& s = S();
	if (s.dll_tried) {
		return s.dll != nullptr;
	}
	s.dll_tried = true;
	wchar_t path[MAX_PATH] {};
	GetModuleFileNameW(nullptr, path, MAX_PATH);
	std::wstring dir(path);
	dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
	s.dll = LoadLibraryW((dir + L"amd_fidelityfx_vk.dll").c_str());
	if (s.dll == nullptr) {
		std::printf("FSR 3 : amd_fidelityfx_vk.dll introuvable a cote de l'executable (erreur %lu)\n", GetLastError());
		return false;
	}
	ffxLoadFunctions(&s.f, s.dll);
	if (s.f.CreateContext == nullptr || s.f.Configure == nullptr || s.f.Dispatch == nullptr || s.f.Query == nullptr ||
	    s.f.DestroyContext == nullptr) {
		std::printf("FSR 3 : fonctions de l'API introuvables dans amd_fidelityfx_vk.dll\n");
		FreeLibrary(s.dll);
		s.dll = nullptr;
		return false;
	}
	return true;
}

uint32_t FindMemoryType(const GraphicContext& g, uint32_t bits, VkMemoryPropertyFlags flags) {
	const auto& props = g.physical_device_memory_properties;
	for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
		if ((bits & (1u << i)) != 0 &&
		    (static_cast<VkMemoryPropertyFlags>(props.memoryTypes[i].propertyFlags) & flags) == flags) {
			return i;
		}
	}
	return UINT32_MAX;
}

bool CreateNeutral(GraphicContext& g, Neutral& n, VkFormat format, VkExtent2D extent) {
	const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	n.info                   = {};
	n.info.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	n.info.imageType         = VK_IMAGE_TYPE_2D;
	n.info.format            = format;
	n.info.extent            = {extent.width, extent.height, 1};
	n.info.mipLevels         = 1;
	n.info.arrayLayers       = 1;
	n.info.samples           = VK_SAMPLE_COUNT_1_BIT;
	n.info.tiling            = VK_IMAGE_TILING_OPTIMAL;
	n.info.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	n.info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
	n.info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
	const VkDevice device    = static_cast<VkDevice>(g.device);
	if (d.vkCreateImage(device, &n.info, nullptr, &n.image) != VK_SUCCESS) {
		return false;
	}
	VkMemoryRequirements req {};
	d.vkGetImageMemoryRequirements(device, n.image, &req);
	VkMemoryAllocateInfo alloc {};
	alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc.allocationSize  = req.size;
	alloc.memoryTypeIndex = FindMemoryType(g, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (alloc.memoryTypeIndex == UINT32_MAX || d.vkAllocateMemory(device, &alloc, nullptr, &n.memory) != VK_SUCCESS) {
		return false;
	}
	return d.vkBindImageMemory(device, n.image, n.memory, 0) == VK_SUCCESS;
}

void DestroyNeutral(VkDevice device, Neutral& n) {
	const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (n.image != VK_NULL_HANDLE) {
		d.vkDestroyImage(device, n.image, nullptr);
	}
	if (n.memory != VK_NULL_HANDLE) {
		d.vkFreeMemory(device, n.memory, nullptr);
	}
	n = {};
}

// Premier passage : les images neutres sont effacees (profondeur 1, mouvement 0) puis laissees en lecture.
void ClearNeutral(VkCommandBuffer cmd, Neutral& n, float value) {
	const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	VkImageMemoryBarrier b {};
	b.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	b.srcAccessMask               = 0;
	b.dstAccessMask               = VK_ACCESS_TRANSFER_WRITE_BIT;
	b.oldLayout                   = VK_IMAGE_LAYOUT_UNDEFINED;
	b.newLayout                   = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	b.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
	b.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
	b.image                       = n.image;
	b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	b.subresourceRange.levelCount = 1;
	b.subresourceRange.layerCount = 1;
	d.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
	                       nullptr, 1, &b);
	VkClearColorValue c {};
	c.float32[0] = value;
	d.vkCmdClearColorImage(cmd, n.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &b.subresourceRange);
	b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	d.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
	                       nullptr, 1, &b);
}

// Le moteur Vulkan de FidelityFX charge toutes ses fonctions par vkGetDeviceProcAddr et en appelle certaines sans
// verifier qu'elles existent (vkGetBufferMemoryRequirements2KHR, ffx_vk.cpp:1663). Sur un peripherique ou
// l'extension n'est pas activee, le nom suffixe KHR revient nul alors que la fonction de base (Vulkan 1.1) existe :
// on la renvoie a sa place. Les fonctions de debogage facultatives absentes deviennent des fonctions vides.
VKAPI_ATTR void VKAPI_CALL NoopSetName(VkDevice, const VkDebugUtilsObjectNameInfoEXT*) {}
VKAPI_ATTR void VKAPI_CALL NoopLabel(VkCommandBuffer, const VkDebugUtilsLabelEXT*) {}
VKAPI_ATTR void VKAPI_CALL NoopEndLabel(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL NoopMarker2(VkCommandBuffer, VkPipelineStageFlags2, VkBuffer, VkDeviceSize, uint32_t) {}
VKAPI_ATTR void VKAPI_CALL NoopMarker(VkCommandBuffer, VkPipelineStageFlagBits, VkBuffer, VkDeviceSize, uint32_t) {}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL ProcAddr(VkDevice device, const char* name) {
	const auto real = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
	if (auto f = real(device, name); f != nullptr) {
		return f;
	}
	const std::string n(name);
	for (const char* suffix: {"KHR", "EXT"}) {
		const size_t len = std::strlen(suffix);
		if (n.size() > len && n.compare(n.size() - len, len, suffix) == 0) {
			if (auto f = real(device, n.substr(0, n.size() - len).c_str()); f != nullptr) {
				return f;
			}
		}
	}
	if (n == "vkSetDebugUtilsObjectNameEXT") return reinterpret_cast<PFN_vkVoidFunction>(&NoopSetName);
	if (n == "vkCmdBeginDebugUtilsLabelEXT") return reinterpret_cast<PFN_vkVoidFunction>(&NoopLabel);
	if (n == "vkCmdEndDebugUtilsLabelEXT") return reinterpret_cast<PFN_vkVoidFunction>(&NoopEndLabel);
	if (n == "vkCmdWriteBufferMarker2AMD") return reinterpret_cast<PFN_vkVoidFunction>(&NoopMarker2);
	if (n == "vkCmdWriteBufferMarkerAMD") return reinterpret_cast<PFN_vkVoidFunction>(&NoopMarker);
	static int logged = 0;
	if (logged++ < 32) {
		std::printf("FSR 3 : fonction Vulkan absente : %s\n", name);
		std::fflush(stdout);
	}
	return nullptr;
}

// Soumissions de FSR 3 sur la file graphique de l'emulateur : depuis ses propres fils comme depuis la presentation.
// Vulkan exige qu'une file ne recoive jamais deux soumissions a la fois ; l'emulateur soumet sous queue_mutex
// (section critique, reentrante), FSR 3 doit donc le prendre aussi. Sans cela (submitFunc nul), FSR 3 soumettait
// en concurrence avec l'emulateur.
VkResult GameQueueSubmit(uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
	auto*             g = S().graphics;
	Common::LockGuard lock(g->queue_mutex);
	return VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit(static_cast<VkQueue>(g->queue), count, submits, fence);
}

// Sonde : une acquisition ou une presentation FSR 3 de plus de 500 ms est notee (gel ou attente anormale).
struct SlowProbe {
	const char*                           what;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	~SlowProbe() {
		const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		if (ms > 500.0) {
			static int logged = 0;
			if (logged++ < 64) {
				std::printf("FSR 3 : %s a dure %.0f ms\n", what, ms);
				std::fflush(stdout);
			}
		}
	}
};

ffxReturnCode_t DispatchCallback(ffxDispatchDescFrameGeneration* params, void* user) {
	return S().f.Dispatch(static_cast<ffxContext*>(user), &params->header);
}

bool Flag(const char* name) {
	const char* v = std::getenv(name);
	return v != nullptr && std::strcmp(v, "1") == 0;
}

} // namespace

bool Requested() {
	static const bool requested = Flag("KYTY_FSR3_FG");
	return requested;
}

bool Active() {
	return S().active;
}

bool CreateSwapchain(GraphicContext& graphics, const VkSwapchainCreateInfoKHR& info, VkSwapchainKHR* swapchain) {
	auto& s = S();
	if (!Requested() || graphics.fsr_queue_base == 0 || !LoadDll()) {
		return false;
	}
	s.graphics = &graphics;
	s.device   = static_cast<VkDevice>(graphics.device);
	s.physical = static_cast<VkPhysicalDevice>(graphics.physical_device);

	ffxCreateContextDescFrameGenerationSwapChainVK desc {};
	desc.header.type        = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
	desc.physicalDevice     = s.physical;
	desc.device             = s.device;
	VkSwapchainKHR handle   = VK_NULL_HANDLE;
	desc.swapchain          = &handle;
	desc.allocator          = nullptr;
	desc.createInfo         = info;
	desc.createInfo.oldSwapchain = VK_NULL_HANDLE;
	desc.gameQueue          = {static_cast<VkQueue>(graphics.queue), graphics.queue_family, &GameQueueSubmit};
	desc.asyncComputeQueue  = {static_cast<VkQueue>(graphics.fsr_queues[0]), graphics.queue_family, nullptr};
	desc.presentQueue       = {static_cast<VkQueue>(graphics.fsr_queues[1]), graphics.queue_family, nullptr};
	desc.imageAcquireQueue  = {static_cast<VkQueue>(graphics.fsr_queues[2]), graphics.queue_family, nullptr};
	std::printf("FSR 3 : etape 1, chaine de presentation (files %p %p %p %p, getDeviceProcAddr %p)\n", static_cast<void*>(desc.gameQueue.queue), static_cast<void*>(desc.asyncComputeQueue.queue), static_cast<void*>(desc.presentQueue.queue), static_cast<void*>(desc.imageAcquireQueue.queue), reinterpret_cast<void*>(VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr));
	std::fflush(stdout);
	StartWatchdog();
	ffxReturnCode_t code;
	{
		const StageScope stage(Stage::CreationChaine);
		code = s.f.CreateContext(&s.swapchain_ctx, &desc.header, nullptr);
	}
	std::printf("FSR 3 : etape 1 terminee (code %u)\n", static_cast<unsigned>(code));
	std::fflush(stdout);
	if (code != FFX_API_RETURN_OK || handle == VK_NULL_HANDLE) {
		Log("la creation de la chaine de presentation", code);
		s.swapchain_ctx = nullptr;
		return false;
	}
	s.repl             = {};
	s.repl.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
	code               = s.f.Query(&s.swapchain_ctx, &s.repl.header);
	if (code != FFX_API_RETURN_OK || s.repl.pOutQueuePresentKHR == nullptr || s.repl.pOutAcquireNextImageKHR == nullptr ||
	    s.repl.pOutGetSwapchainImagesKHR == nullptr) {
		Log("la lecture des fonctions de remplacement", code);
		s.f.DestroyContext(&s.swapchain_ctx, nullptr);
		s.swapchain_ctx = nullptr;
		return false;
	}

	ffxCreateBackendVKDesc backend {};
	backend.header.type      = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
	backend.vkDevice         = s.device;
	backend.vkPhysicalDevice = s.physical;
	backend.vkDeviceProcAddr = &ProcAddr;
	ffxCreateContextDescFrameGeneration fg {};
	fg.header.type      = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
	fg.header.pNext     = &backend.header;
	fg.flags            = 0;
	fg.displaySize      = {info.imageExtent.width, info.imageExtent.height};
	fg.maxRenderSize    = {info.imageExtent.width, info.imageExtent.height};
	fg.backBufferFormat = ffxApiGetSurfaceFormatVK(info.imageFormat);
	std::printf("FSR 3 : etape 2, contexte de generation\n");
	std::fflush(stdout);
	{
		const StageScope stage(Stage::CreationContexte);
		code = s.f.CreateContext(&s.fg_ctx, &fg.header, nullptr);
	}
	std::printf("FSR 3 : etape 2 terminee (code %u)\n", static_cast<unsigned>(code));
	std::fflush(stdout);
	if (code != FFX_API_RETURN_OK) {
		Log("la creation du contexte de generation", code);
		s.fg_ctx = nullptr;
		s.f.DestroyContext(&s.swapchain_ctx, nullptr);
		s.swapchain_ctx = nullptr;
		return false;
	}

	std::printf("FSR 3 : etape 3, images neutres\n");
	std::fflush(stdout);
	s.extent = info.imageExtent;
	if (!CreateNeutral(graphics, s.depth, VK_FORMAT_R32_SFLOAT, s.extent) ||
	    !CreateNeutral(graphics, s.motion, VK_FORMAT_R16G16_SFLOAT, s.extent)) {
		std::printf("FSR 3 : images neutres non creees\n");
		DestroySwapchain(graphics);
		return false;
	}
	s.neutral_ready = false;
	s.swapchain     = handle;
	s.frame_id      = 0;
	s.last          = std::chrono::steady_clock::now();
	s.active        = true;
	*swapchain      = handle;
	std::printf("FSR 3 : generation d'images active (%ux%u), prototype sans vecteurs de mouvement du jeu\n",
	            info.imageExtent.width, info.imageExtent.height);
	std::fflush(stdout);
	return true;
}

void DestroySwapchain(GraphicContext& graphics) {
	auto& s = S();
	std::printf("FSR 3 : destruction de la chaine (recreation ou fermeture)\n");
	std::fflush(stdout);
	if (s.fg_ctx != nullptr) {
		const StageScope stage(Stage::DestructionContexte);
		s.f.DestroyContext(&s.fg_ctx, nullptr);
		s.fg_ctx = nullptr;
	}
	if (s.swapchain_ctx != nullptr) {
		// Detruit aussi la chaine de presentation qu'il a creee.
		const StageScope stage(Stage::DestructionChaine);
		s.f.DestroyContext(&s.swapchain_ctx, nullptr);
		s.swapchain_ctx = nullptr;
	}
	const VkDevice device = static_cast<VkDevice>(graphics.device);
	DestroyNeutral(device, s.depth);
	DestroyNeutral(device, s.motion);
	s.swapchain = VK_NULL_HANDLE;
	s.active    = false;
}

VkResult GetSwapchainImages(VkDevice device, VkSwapchainKHR swapchain, uint32_t* count, VkImage* images) {
	return S().repl.pOutGetSwapchainImagesKHR(device, swapchain, count, images);
}

VkResult AcquireNextImage(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore,
                          VkFence fence, uint32_t* index) {
	const SlowProbe  probe {"l'acquisition d'image"};
	const StageScope stage(Stage::Acquisition);
	return S().repl.pOutAcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, index);
}

static std::atomic<double> g_displayed {0.0};
static std::atomic<double> g_game {0.0};

double DisplayedFps() {
	return g_displayed.load(std::memory_order_relaxed);
}

static std::atomic<int>     g_presents_done {0};
static std::atomic<int64_t> g_present_started {0}; // 0 : aucune presentation en cours

bool HoldWindowPump() {
	if (g_presents_done.load(std::memory_order_relaxed) >= 3) {
		return false;
	}
	const int64_t started = g_present_started.load(std::memory_order_relaxed);
	return started != 0 && NowMs() - started < 3000;
}

VkResult QueuePresent(VkQueue queue, const VkPresentInfoKHR* info) {
	auto& s = S();
	VkResult result;
	{
		const SlowProbe  probe {"la presentation"};
		const StageScope stage(Stage::Presentation);
		g_present_started = NowMs();
		result         = s.repl.pOutQueuePresentKHR(queue, info);
		g_present_started = 0;
		g_presents_done.fetch_add(1, std::memory_order_relaxed);
		g_last_present = NowMs();
	}
	// Toutes les 5 s : images affichees (presentations reelles de FSR 3) et images du jeu, par seconde.
	static auto     window_start = std::chrono::steady_clock::now();
	static uint64_t presents0 = 0, frames0 = 0;
	const auto      now = std::chrono::steady_clock::now();
	const double    sec = std::chrono::duration<double>(now - window_start).count();
	if (sec >= 1.0 && s.repl.pOutGetLastPresentCountFFXAPI != nullptr) {
		const uint64_t presents = s.repl.pOutGetLastPresentCountFFXAPI(s.swapchain);
		g_displayed = static_cast<double>(presents - presents0) / sec;
		g_game      = static_cast<double>(s.frame_id - frames0) / sec;
		static int ticks = 0;
		if (++ticks % 5 == 0) {
			std::printf("FSR 3 : %.1f images affichees/s, %.1f images du jeu/s\n", g_displayed.load(), g_game.load());
			std::fflush(stdout);
		}
		presents0    = presents;
		frames0      = s.frame_id;
		window_start = now;
	}
	return result;
}

void PrepareFrame(VkCommandBuffer command) {
	auto& s = S();
	if (!s.active) {
		return;
	}
	const StageScope stage(Stage::Preparation);
	if (!s.neutral_ready) {
		ClearNeutral(command, s.depth, 1.0f);
		ClearNeutral(command, s.motion, 0.0f);
		s.neutral_ready = true;
	}
	const auto now   = std::chrono::steady_clock::now();
	const float ms   = std::chrono::duration<float, std::milli>(now - s.last).count();
	s.last           = now;
	static const bool debug = Flag("KYTY_FSR3_FG_DEBUG");

	ffxConfigureDescFrameGeneration cfg {};
	cfg.header.type                        = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
	cfg.swapChain                          = s.swapchain;
	cfg.presentCallback                    = nullptr;
	cfg.frameGenerationCallback            = DispatchCallback;
	cfg.frameGenerationCallbackUserContext = &s.fg_ctx;
	cfg.frameGenerationEnabled             = true;
	cfg.allowAsyncWorkloads                = false;
	cfg.HUDLessColor                       = {};
	cfg.flags = debug ? (FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_TEAR_LINES | FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_RESET_INDICATORS)
	                  : 0u;
	cfg.onlyPresentGenerated   = false;
	cfg.generationRect         = {0, 0, static_cast<int32_t>(s.extent.width), static_cast<int32_t>(s.extent.height)};
	cfg.frameID                = s.frame_id;
	Log("la configuration", s.f.Configure(&s.fg_ctx, &cfg.header));

	ffxDispatchDescFrameGenerationPrepare prep {};
	prep.header.type             = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
	prep.frameID                 = s.frame_id;
	prep.flags                   = cfg.flags;
	prep.commandList             = command;
	prep.renderSize              = {s.extent.width, s.extent.height};
	prep.jitterOffset            = {0.0f, 0.0f};
	prep.motionVectorScale       = {1.0f, 1.0f};
	prep.frameTimeDelta          = ms;
	prep.cameraNear              = 0.1f;
	prep.cameraFar               = 1000.0f;
	prep.cameraFovAngleVertical  = 1.0f;
	prep.viewSpaceToMetersFactor = 1.0f;
	prep.depth         = ffxApiGetResourceVK(reinterpret_cast<void*>(s.depth.image),
                                     ffxApiGetImageResourceDescriptionVK(s.depth.image, s.depth.info, 0),
                                     FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
	prep.motionVectors = ffxApiGetResourceVK(reinterpret_cast<void*>(s.motion.image),
	                                         ffxApiGetImageResourceDescriptionVK(s.motion.image, s.motion.info, 0),
	                                         FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
	Log("la preparation", s.f.Dispatch(&s.fg_ctx, &prep.header));
	s.frame_id++;
}

} // namespace Libs::Graphics::Fsr3
