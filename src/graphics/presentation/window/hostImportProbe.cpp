#include "graphics/presentation/window/hostImportProbe.h"

#include "common/common.h"
#include "graphics/host_gpu/graphicContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#endif

// Astro Bot PC: KYTY_HOST_IMPORT_PROBE (hostImportProbe.h). Diagnostic only.
namespace Libs::Graphics::HostImportProbe {

namespace {

constexpr uint64_t ProbeSize = uint64_t {64} << 20u;

struct State {
	VkDevice         device          = VK_NULL_HANDLE;
	VkPhysicalDevice physical_device = VK_NULL_HANDLE;
	bool             extension       = false; // VK_EXT_external_memory_host enabled
	uint64_t         alignment       = 0x1000;
	VkPhysicalDeviceMemoryProperties memory {};
	std::thread             worker;
	std::mutex              mutex;
	std::condition_variable wake;
	bool                    stop = false;
};

State& GetState() {
	static State state;
	return state;
}

void Line(const std::string& text) {
	std::printf("HostImport probe: %s\n", text.c_str());
	std::fflush(stdout);
}

std::string Format(const char* format, ...) {
	char    text[512];
	va_list args;
	va_start(args, format);
	std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	return text;
}

double MicrosSince(std::chrono::steady_clock::time_point start) {
	return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                               std::chrono::steady_clock::now() - start)
	                               .count()) *
	       1e-3;
}

std::string MemoryFlags(VkMemoryPropertyFlags flags) {
	std::string out;
	auto add = [&](VkMemoryPropertyFlags bit, const char* name) {
		if ((flags & bit) != 0) {
			out += out.empty() ? "" : "|";
			out += name;
		}
	};
	add(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "DEVICE_LOCAL");
	add(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "HOST_VISIBLE");
	add(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT");
	add(VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "HOST_CACHED");
	add(VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, "LAZY");
	return out.empty() ? std::string("none") : out;
}

std::string HostPageInfo(const void* pointer) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(pointer, &info, sizeof(info)) == 0) {
		return "VirtualQuery failed";
	}
	return Format("host page: state=0x%lx type=0x%lx protect=0x%lx region=0x%" PRIx64,
	              static_cast<unsigned long>(info.State), static_cast<unsigned long>(info.Type),
	              static_cast<unsigned long>(info.Protect),
	              static_cast<uint64_t>(info.RegionSize));
#else
	(void)pointer;
	return "host page: n/a";
#endif
}

// The driver calls that take the foreign pointer run under a structured exception guard: an
// access fault inside them is reported, not fatal. No C++ object lives in these frames.
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS && defined(_MSC_VER)
VkResult GuardedHostPointerProperties(VkDevice device, const void* pointer,
                                      VkMemoryHostPointerPropertiesEXT* props, bool* faulted) {
	*faulted = false;
	__try {
		return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetMemoryHostPointerPropertiesEXT(
		    device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, pointer, props);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		*faulted = true;
		return VK_ERROR_UNKNOWN;
	}
}
VkResult GuardedAllocate(VkDevice device, const VkMemoryAllocateInfo* info, VkDeviceMemory* memory,
                         bool* faulted) {
	*faulted = false;
	__try {
		return VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory(device, info, nullptr, memory);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		*faulted = true;
		return VK_ERROR_UNKNOWN;
	}
}
#else
VkResult GuardedHostPointerProperties(VkDevice device, const void* pointer,
                                      VkMemoryHostPointerPropertiesEXT* props, bool* faulted) {
	*faulted = false;
	return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetMemoryHostPointerPropertiesEXT(
	    device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, pointer, props);
}
VkResult GuardedAllocate(VkDevice device, const VkMemoryAllocateInfo* info, VkDeviceMemory* memory,
                         bool* faulted) {
	*faulted = false;
	return VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory(device, info, nullptr, memory);
}
#endif

std::string ResultName(VkResult result) {
	return vk::to_string(static_cast<vk::Result>(result));
}

void ProbeOne(const char* label, uint64_t address, uint64_t size) {
	auto&      st        = GetState();
	const auto alignment = st.alignment;
	const auto aligned   = (address + alignment - 1) & ~(alignment - 1);
	if (aligned < address || size < aligned - address) {
		Line(Format("%s: range 0x%016" PRIx64 "+0x%" PRIx64 " too small", label, address, size));
		return;
	}
	size = (size - (aligned - address)) & ~(alignment - 1);
	if (size == 0) {
		Line(Format("%s: range 0x%016" PRIx64 " too small after alignment", label, address));
		return;
	}
	auto* pointer = reinterpret_cast<void*>(aligned);
	Line(Format("%s: pointer=0x%016" PRIx64 " size=0x%" PRIx64 " (%" PRIu64 " KiB), %s", label,
	            aligned, size, size >> 10u, HostPageInfo(pointer).c_str()));

	const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (d.vkGetMemoryHostPointerPropertiesEXT == nullptr || d.vkAllocateMemory == nullptr ||
	    d.vkCreateBuffer == nullptr ||
	    (d.vkGetBufferDeviceAddress == nullptr && d.vkGetBufferDeviceAddressKHR == nullptr)) {
		Line(Format("%s: entry points missing", label));
		return;
	}

	VkMemoryHostPointerPropertiesEXT props {};
	props.sType     = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
	bool       faulted = false;
	auto       t0      = std::chrono::steady_clock::now();
	const auto pr      = GuardedHostPointerProperties(st.device, pointer, &props, &faulted);
	Line(Format("%s: vkGetMemoryHostPointerPropertiesEXT = %s%s, memoryTypeBits=0x%x (%.1f us)",
	            label, ResultName(pr).c_str(), faulted ? " (FAULT caught)" : "", props.memoryTypeBits,
	            MicrosSince(t0)));
	if (pr != VK_SUCCESS || faulted) {
		return;
	}

	VkExternalMemoryBufferCreateInfo external {};
	external.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	VkBufferCreateInfo buffer_info {};
	buffer_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.pNext       = &external;
	buffer_info.size        = size;
	buffer_info.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
	                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VkBuffer   buffer       = VK_NULL_HANDLE;
	const auto br           = d.vkCreateBuffer(st.device, &buffer_info, nullptr, &buffer);
	if (br != VK_SUCCESS) {
		Line(Format("%s: vkCreateBuffer = %s", label, ResultName(br).c_str()));
		return;
	}
	VkMemoryRequirements reqs {};
	d.vkGetBufferMemoryRequirements(st.device, buffer, &reqs);
	const uint32_t bits = props.memoryTypeBits & reqs.memoryTypeBits;
	uint32_t       type = UINT32_MAX;
	for (uint32_t pass = 0; pass < 2 && type == UINT32_MAX; pass++) {
		for (uint32_t i = 0; i < st.memory.memoryTypeCount; i++) {
			const auto flags = st.memory.memoryTypes[i].propertyFlags;
			const bool coherent =
			    (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
			    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
			if ((bits & (1u << i)) != 0 && (pass == 1 || coherent)) {
				type = i;
				break;
			}
		}
	}
	Line(Format("%s: buffer requirements size=0x%" PRIx64 " alignment=0x%" PRIx64
	            " memoryTypeBits=0x%x, usable=0x%x",
	            label, static_cast<uint64_t>(reqs.size), static_cast<uint64_t>(reqs.alignment),
	            reqs.memoryTypeBits, bits));
	if (type == UINT32_MAX || reqs.size > size) {
		Line(Format("%s: no usable memory type (or buffer larger than the range): not allocated",
		            label));
		d.vkDestroyBuffer(st.device, buffer, nullptr);
		return;
	}
	const auto heap = st.memory.memoryTypes[type].heapIndex;
	Line(Format("%s: memory type %u [%s], heap %u (0x%" PRIx64 " bytes%s)", label, type,
	            MemoryFlags(st.memory.memoryTypes[type].propertyFlags).c_str(), heap,
	            static_cast<uint64_t>(st.memory.memoryHeaps[heap].size),
	            (st.memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0
	                ? ", device local"
	                : ""));

	VkMemoryAllocateFlagsInfo flags_info {};
	flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
	flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	VkImportMemoryHostPointerInfoEXT import_info {};
	import_info.sType        = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
	import_info.pNext        = &flags_info;
	import_info.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	import_info.pHostPointer = pointer;
	VkMemoryAllocateInfo allocate {};
	allocate.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocate.pNext           = &import_info;
	allocate.allocationSize  = size;
	allocate.memoryTypeIndex = type;
	VkDeviceMemory memory    = VK_NULL_HANDLE;
	t0                       = std::chrono::steady_clock::now();
	const auto ar            = GuardedAllocate(st.device, &allocate, &memory, &faulted);
	const auto alloc_us      = MicrosSince(t0);
	Line(Format("%s: vkAllocateMemory(import) = %s%s (%.1f us)", label, ResultName(ar).c_str(),
	            faulted ? " (FAULT caught)" : "", alloc_us));
	if (ar != VK_SUCCESS || faulted || memory == VK_NULL_HANDLE) {
		d.vkDestroyBuffer(st.device, buffer, nullptr);
		return;
	}
	const auto bind = d.vkBindBufferMemory(st.device, buffer, memory, 0);
	if (bind != VK_SUCCESS) {
		Line(Format("%s: vkBindBufferMemory = %s", label, ResultName(bind).c_str()));
	} else {
		VkBufferDeviceAddressInfo address_info {};
		address_info.sType     = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		address_info.buffer    = buffer;
		const auto get_address    = d.vkGetBufferDeviceAddress != nullptr
		                                ? d.vkGetBufferDeviceAddress
		                                : d.vkGetBufferDeviceAddressKHR;
		const auto device_address = get_address(st.device, &address_info);
		Line(Format("%s: OK, buffer bound, device address 0x%016" PRIx64
		            " (host 0x%016" PRIx64 ", %s)",
		            label, static_cast<uint64_t>(device_address), aligned,
		            device_address == aligned ? "same value" : "different value"));
	}
	d.vkDestroyBuffer(st.device, buffer, nullptr);
	t0 = std::chrono::steady_clock::now();
	d.vkFreeMemory(st.device, memory, nullptr);
	Line(Format("%s: freed (%.1f us)", label, MicrosSince(t0)));
}

void ProbeBacked(bool flexible, uint64_t min_size) {
	LibKernel::Memory::VirtualQueryInfo info {};
	if (!LibKernel::Memory::DiagFindBackedRange(flexible, min_size, &info)) {
		return;
	}
	char       label[96];
	const auto range_size = static_cast<uint64_t>(info.end - info.start);
	std::snprintf(label, sizeof(label), "(b) guest %s view '%.32s'", flexible ? "flexible" : "direct",
	              info.name);
	Line(Format("%s: range 0x%016" PRIx64 "-0x%016" PRIx64 " (%" PRIu64 " KiB) prot=0x%x type=%d",
	            label, static_cast<uint64_t>(info.start), static_cast<uint64_t>(info.end),
	            range_size >> 10u, info.protection, info.memory_type));
	ProbeOne(label, info.start, std::min(range_size, ProbeSize));
}

void Worker() {
	auto&      st      = GetState();
	const auto start   = std::chrono::steady_clock::now();
	bool       direct  = false;
	bool       flex    = false;
	for (;;) {
		{
			std::unique_lock lock(st.mutex);
			if (st.wake.wait_for(lock, std::chrono::seconds(2), [&] { return st.stop; })) {
				return;
			}
		}
		const auto elapsed = std::chrono::steady_clock::now() - start;
		// A 64 MiB mapping first; after 30 s, any mapping of at least 64 KiB.
		const uint64_t min_size =
		    elapsed < std::chrono::seconds(30) ? ProbeSize : uint64_t {64} << 10u;
		if (!direct) {
			LibKernel::Memory::VirtualQueryInfo info {};
			if (LibKernel::Memory::DiagFindBackedRange(false, min_size, &info)) {
				ProbeBacked(false, min_size);
				direct = true;
			}
		}
		if (!flex) {
			LibKernel::Memory::VirtualQueryInfo info {};
			if (LibKernel::Memory::DiagFindBackedRange(true, min_size, &info)) {
				ProbeBacked(true, min_size);
				flex = true;
			}
		}
		if ((direct && flex) || elapsed > std::chrono::minutes(3)) {
			Line(Format("(b) done (direct %s, flexible %s)", direct ? "probed" : "none mapped",
			            flex ? "probed" : "none mapped"));
			return;
		}
	}
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_HOST_IMPORT_PROBE");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

void AppendDeviceExtension(std::vector<const char*>&                 device_extensions,
                           const std::vector<vk::ExtensionProperties>& available) {
	if (!Enabled()) {
		return;
	}
	bool found = false;
	for (const auto& e: available) {
		if (std::strcmp(e.extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0) {
			found = true;
			break;
		}
	}
	bool already = false;
	for (const char* e: device_extensions) {
		already = already || std::strcmp(e, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0;
	}
	if (found && !already) {
		device_extensions.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
	}
	GetState().extension = found;
	Line(found ? "VK_EXT_external_memory_host enabled for the probe"
	           : "VK_EXT_external_memory_host not available: probe skipped");
}

void Start(GraphicContext& ctx) {
	if (!Enabled()) {
		return;
	}
	auto& st = GetState();
	if (!st.extension) {
		return;
	}
	st.device          = static_cast<VkDevice>(ctx.device);
	st.physical_device = static_cast<VkPhysicalDevice>(ctx.physical_device);
	const auto& d      = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (d.vkGetPhysicalDeviceProperties2 != nullptr) {
		VkPhysicalDeviceExternalMemoryHostPropertiesEXT host {};
		host.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
		VkPhysicalDeviceProperties2 props {};
		props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		props.pNext = &host;
		d.vkGetPhysicalDeviceProperties2(st.physical_device, &props);
		if (host.minImportedHostPointerAlignment != 0) {
			st.alignment = host.minImportedHostPointerAlignment;
		}
	}
	d.vkGetPhysicalDeviceMemoryProperties(st.physical_device, &st.memory);
	Line(Format("minImportedHostPointerAlignment=0x%" PRIx64 ", %u memory types",
	            st.alignment, st.memory.memoryTypeCount));
	for (uint32_t i = 0; i < st.memory.memoryTypeCount; i++) {
		Line(Format("  type %u: heap %u [%s]", i, st.memory.memoryTypes[i].heapIndex,
		            MemoryFlags(st.memory.memoryTypes[i].propertyFlags).c_str()));
	}
	if ((st.alignment & (st.alignment - 1)) != 0) {
		Line("alignment is not a power of two: probe skipped");
		return;
	}

	// (a) the alias view of the backing section: 64 MiB in its middle (offset aligned to 64 MiB).
	uint64_t backing = 0;
	uint64_t backing_size = 0;
	if (LibKernel::Memory::DiagGuestBacking(&backing, &backing_size) && backing_size >= ProbeSize) {
		const auto offset = ((backing_size / 2) & ~(ProbeSize - 1));
		const auto at     = std::min(offset, backing_size - ProbeSize);
		Line(Format("(a) backing alias view base=0x%016" PRIx64 " size=0x%" PRIx64
		            ", probing offset 0x%" PRIx64,
		            backing, backing_size, at));
		ProbeOne("(a) backing alias", backing + at, ProbeSize);
	} else {
		Line("(a) guest backing section not available");
	}

	// (c) control: a plain private allocation.
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (void* control = VirtualAlloc(nullptr, ProbeSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	    control != nullptr) {
		ProbeOne("(c) VirtualAlloc control", reinterpret_cast<uint64_t>(control), ProbeSize);
		VirtualFree(control, 0, MEM_RELEASE);
	} else {
		Line("(c) VirtualAlloc failed");
	}
#else
	Line("(c) control skipped (Windows only)");
#endif

	// (b) guest views: once the game has mapped direct / flexible memory.
	st.stop   = false;
	st.worker = std::thread(Worker);
}

void Stop() {
	auto& st = GetState();
	if (!st.worker.joinable()) {
		return;
	}
	{
		std::lock_guard lock(st.mutex);
		st.stop = true;
	}
	st.wake.notify_all();
	st.worker.join();
}

} // namespace Libs::Graphics::HostImportProbe
