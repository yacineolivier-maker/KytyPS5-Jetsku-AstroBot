#include "graphics/presentation/window/hostImportProbe.h"

#include "common/common.h"
#include "graphics/host_gpu/graphicContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <deque>
#include <memory>
#include <xxhash.h>
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

bool ExtensionEnabled() {
	return GetState().extension;
}

void AppendDeviceExtension(std::vector<const char*>&                 device_extensions,
                           const std::vector<vk::ExtensionProperties>& available) {
	if (!Enabled() && !HostImport::Enabled()) {
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

// Astro Bot PC: KYTY_HOST_IMPORT (hostImportProbe.h, step B0 of docs\PLAN-PARALLELE.md).
namespace Libs::Graphics::HostImport {

namespace {

using HostImportProbe::Format;
using HostImportProbe::GuardedAllocate;
using HostImportProbe::GuardedHostPointerProperties;
using HostImportProbe::MicrosSince;

constexpr uint32_t ChunkBits = 26; // 64 MiB of the backing per import
constexpr uint64_t ChunkSize = uint64_t {1} << ChunkBits;

enum : uint8_t { ChunkNone = 0, ChunkRequested = 1, ChunkReady = 2, ChunkFailed = 3 };

struct Chunk {
	std::atomic<uint64_t> address {0};
	std::atomic<uint8_t>  state {ChunkNone};
	VkBuffer              buffer = VK_NULL_HANDLE;
	VkDeviceMemory        memory = VK_NULL_HANDLE;
};

struct Manager {
	VkDevice                          device = VK_NULL_HANDLE;
	VkPhysicalDeviceMemoryProperties  memory {};
	uint64_t                          alignment    = 0x1000;
	uint64_t                          backing      = 0;
	uint64_t                          backing_size = 0;
	uint32_t                          chunk_count  = 0;
	std::unique_ptr<Chunk[]>          chunks;
	std::atomic<bool>                 ready {false};
	std::mutex                        mutex;
	std::condition_variable           wake;
	std::vector<uint32_t>             queue;
	bool                              stop = false;
	std::thread                       worker;
	std::atomic<uint64_t>             lookups {0};
	std::atomic<uint64_t>             hits {0};
	std::atomic<uint64_t>             unmapped {0};
	std::atomic<uint64_t>             cross {0};
	std::atomic<uint64_t>             pending {0};
	std::atomic<uint64_t>             failed_lookups {0};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(BindFallback::Count)> bind {};
	struct RaceEntry {
		uint64_t offset = 0;
		uint64_t size   = 0;
		uint64_t hash   = 0;
		uint64_t tick   = 0;
	};
	std::deque<RaceEntry>             race; // GPU thread only
	uint64_t                          race_calls = 0, race_recorded = 0, race_dropped = 0;
	std::atomic<uint64_t>             race_checked {0}, race_changed {0}, race_changed_bytes {0};
	uint64_t                          imported  = 0; // worker only
	uint64_t                          reserved_found = 0; // worker only: reserved bytes committed before import
	uint64_t                          failures  = 0; // worker only
	double                            import_us = 0; // worker only
};

Manager& GetManager() {
	static Manager manager;
	return manager;
}

int64_t Mode() {
	static const int64_t mode = [] {
		const auto* value = std::getenv("KYTY_HOST_IMPORT");
		if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) return int64_t {0};
		if (std::strcmp(value, "census") == 0) return int64_t {2};
		if (std::strcmp(value, "race") == 0) return int64_t {4};
		return std::strcmp(value, "bind") == 0 ? int64_t {3} : int64_t {1};
	}();
	return mode;
}

void Log(const std::string& text) {
	std::printf("HostImport: %s\n", text.c_str());
	std::fflush(stdout);
}

uint32_t FindMemoryType(const Manager& m, uint32_t bits) {
	for (uint32_t pass = 0; pass < 2; pass++) {
		for (uint32_t i = 0; i < m.memory.memoryTypeCount; i++) {
			const auto flags    = m.memory.memoryTypes[i].propertyFlags;
			const auto wanted   = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			const bool coherent = (flags & wanted) == wanted;
			if ((bits & (1u << i)) != 0 && (pass == 1 || coherent)) return i;
		}
	}
	return UINT32_MAX;
}

// Worker: imports one 64 MiB chunk of the backing alias view (the backing outlives every guest
// view of it, so a guest unmap never frees imported pages).
bool ImportChunk(Manager& m, uint32_t index, std::string& error) {
	const auto& d     = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto  start = static_cast<uint64_t>(index) << ChunkBits;
	const auto  size  = std::min(ChunkSize, m.backing_size - start) & ~(m.alignment - 1);
	if (size == 0) {
		error = "empty chunk";
		return false;
	}
	auto* pointer = reinterpret_cast<void*>(m.backing + start);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// B2 fix (device loss of 14/11 10:05): the backing section commits its pages lazily
	// (GuestBackingStore::EnsureCommitted). A page still reserved when the chunk is imported is not
	// part of what the driver maps for the GPU; once the guest commits and uses it, a GPU read of
	// it faults. Every page of the chunk is committed first (pages never touched stay demand-zero;
	// a later ZeroRange clears committed pages itself), and the reserved bytes found are counted.
	{
		uint64_t reserved = 0;
		for (uint64_t at = 0; at < size;) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(static_cast<uint8_t*>(pointer) + at, &info, sizeof(info)) != sizeof(info)) {
				error = "VirtualQuery";
				return false;
			}
			const auto region_end = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
			const auto bytes      = std::min(size - at, region_end - (reinterpret_cast<uint64_t>(pointer) + at));
			if (bytes == 0) break;
			if (info.State == MEM_RESERVE) reserved += bytes;
			at += bytes;
		}
		m.reserved_found += reserved;
		if (reserved != 0 && VirtualAlloc(pointer, size, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
			error = "commit of the chunk's reserved pages";
			return false;
		}
	}
#endif
	VkMemoryHostPointerPropertiesEXT props {};
	props.sType  = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
	bool faulted = false;
	if (GuardedHostPointerProperties(m.device, pointer, &props, &faulted) != VK_SUCCESS || faulted) {
		error = faulted ? "host pointer fault" : "host pointer properties";
		return false;
	}
	VkExternalMemoryBufferCreateInfo external {};
	external.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	VkBufferCreateInfo buffer_info {};
	buffer_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.pNext       = &external;
	buffer_info.size        = size;
	buffer_info.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
	                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
	                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VkBuffer buffer         = VK_NULL_HANDLE;
	if (d.vkCreateBuffer(m.device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
		error = "vkCreateBuffer";
		return false;
	}
	VkMemoryRequirements reqs {};
	d.vkGetBufferMemoryRequirements(m.device, buffer, &reqs);
	const auto type = FindMemoryType(m, props.memoryTypeBits & reqs.memoryTypeBits);
	if (type == UINT32_MAX || reqs.size > size) {
		d.vkDestroyBuffer(m.device, buffer, nullptr);
		error = "no memory type";
		return false;
	}
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
	if (GuardedAllocate(m.device, &allocate, &memory, &faulted) != VK_SUCCESS || faulted ||
	    memory == VK_NULL_HANDLE) {
		d.vkDestroyBuffer(m.device, buffer, nullptr);
		error = faulted ? "allocate fault" : "vkAllocateMemory(import)";
		return false;
	}
	if (d.vkBindBufferMemory(m.device, buffer, memory, 0) != VK_SUCCESS) {
		d.vkDestroyBuffer(m.device, buffer, nullptr);
		d.vkFreeMemory(m.device, memory, nullptr);
		error = "vkBindBufferMemory";
		return false;
	}
	VkBufferDeviceAddressInfo address_info {};
	address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	address_info.buffer = buffer;
	const auto get_address = d.vkGetBufferDeviceAddress != nullptr ? d.vkGetBufferDeviceAddress
	                                                               : d.vkGetBufferDeviceAddressKHR;
	const auto address     = static_cast<uint64_t>(get_address(m.device, &address_info));
	auto&      chunk       = m.chunks[index];
	chunk.buffer           = buffer;
	chunk.memory           = memory;
	chunk.address.store(address, std::memory_order_release);
	return address != 0;
}

void Report(Manager& m) {
	const auto lookups = m.lookups.exchange(0, std::memory_order_relaxed);
	const auto hits    = m.hits.exchange(0, std::memory_order_relaxed);
	if (lookups == 0 && m.imported == 0 && Mode() != 4) return;
	if (Mode() == 4) {
		const auto checked = m.race_checked.exchange(0, std::memory_order_relaxed);
		const auto changed = m.race_changed.exchange(0, std::memory_order_relaxed);
		Log(Format("10s race: recorded %" PRIu64 " (dropped %" PRIu64 "), checked after GPU completion %" PRIu64
		           ", changed %" PRIu64 " (%.2f%%, %" PRIu64 " KiB)",
		           m.race_recorded, m.race_dropped, checked, changed,
		           checked != 0 ? 100.0 * static_cast<double>(changed) / static_cast<double>(checked) : 0.0,
		           m.race_changed_bytes.exchange(0, std::memory_order_relaxed) >> 10u));
	}
	if (Mode() == 3) {
		static constexpr const char* names[] = {"served", "written", "large", "gpu-modified", "image-modified",
		                                        "not-imported"};
		std::string line = "10s bind:";
		for (size_t i = 0; i < m.bind.size(); i++) {
			line += Format(" %s %" PRIu64, names[i], m.bind[i].exchange(0, std::memory_order_relaxed));
		}
		Log(line);
	}
	char text[512];
	std::snprintf(text, sizeof(text),
	              "10s: lookups %" PRIu64 " served %" PRIu64 " (%.1f%%), not one guest view %" PRIu64
	              ", across chunks %" PRIu64 ", chunk pending %" PRIu64 ", chunk failed %" PRIu64
	              " | chunks imported %" PRIu64 " (%" PRIu64 " MiB of %" PRIu64 " MiB), failures %" PRIu64
	              ", import %.1f ms, reserved pages committed first %" PRIu64 " MiB",
	              lookups, hits, lookups != 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(lookups) : 0.0,
	              m.unmapped.exchange(0, std::memory_order_relaxed), m.cross.exchange(0, std::memory_order_relaxed),
	              m.pending.exchange(0, std::memory_order_relaxed),
	              m.failed_lookups.exchange(0, std::memory_order_relaxed), m.imported,
	              (m.imported * ChunkSize) >> 20u, m.backing_size >> 20u, m.failures, m.import_us * 1e-3,
	              m.reserved_found >> 20u);
	Log(text);
}

void WorkerLoop() {
	auto& m           = GetManager();
	auto  last_report = std::chrono::steady_clock::now();
	for (;;) {
		std::vector<uint32_t> batch;
		{
			std::unique_lock lock(m.mutex);
			m.wake.wait_for(lock, std::chrono::seconds(1), [&] { return m.stop || !m.queue.empty(); });
			if (m.stop) return;
			batch.swap(m.queue);
		}
		for (const auto index: batch) {
			auto&       chunk = m.chunks[index];
			std::string error;
			const auto  t0 = std::chrono::steady_clock::now();
			const bool  ok = ImportChunk(m, index, error);
			m.import_us += MicrosSince(t0);
			if (ok) {
				m.imported++;
				chunk.state.store(ChunkReady, std::memory_order_release);
			} else {
				m.failures++;
				chunk.state.store(ChunkFailed, std::memory_order_release);
				if (m.failures <= 8) {
					Log(Format("chunk %u (backing 0x%" PRIx64 ") not imported: %s", index,
					           static_cast<uint64_t>(index) << ChunkBits, error.c_str()));
				}
			}
		}
		if (std::chrono::steady_clock::now() - last_report >= std::chrono::seconds(10)) {
			last_report = std::chrono::steady_clock::now();
			Report(m);
		}
	}
}

} // namespace

bool Enabled() {
	return Mode() != 0;
}

bool CensusEnabled() {
	return Mode() == 2;
}

bool BindEnabled() {
	return Mode() == 3;
}

bool RaceEnabled() {
	return Mode() == 4;
}

void RaceRecord(uint64_t vaddr, uint64_t size, uint64_t tick) {
	auto& m = GetManager();
	if (!m.ready.load(std::memory_order_acquire) || (m.race_calls++ & 15u) != 0) return;
	uint64_t offset = 0;
	if (!LibKernel::Memory::GuestToBacking(vaddr, size, &offset) || offset >= m.backing_size ||
	    m.backing_size - offset < size) {
		return;
	}
	if (m.race.size() >= 65536) {
		m.race_dropped++;
		return;
	}
	const auto hash = XXH3_64bits(reinterpret_cast<const void*>(m.backing + offset), static_cast<size_t>(size));
	m.race.push_back({offset, size, hash, tick});
	m.race_recorded++;
}

uint64_t RaceFrontTick() {
	auto& m = GetManager();
	return m.race.empty() ? 0 : m.race.front().tick;
}

void RaceCheckFront() {
	auto& m = GetManager();
	if (m.race.empty()) return;
	const auto entry = m.race.front();
	m.race.pop_front();
	const auto hash = XXH3_64bits(reinterpret_cast<const void*>(m.backing + entry.offset), static_cast<size_t>(entry.size));
	m.race_checked.fetch_add(1, std::memory_order_relaxed);
	if (hash != entry.hash) {
		m.race_changed.fetch_add(1, std::memory_order_relaxed);
		m.race_changed_bytes.fetch_add(entry.size, std::memory_order_relaxed);
	}
}

uint64_t BindMaxBytes() {
	static const uint64_t bytes = [] {
		const auto* value = std::getenv("KYTY_HOST_IMPORT_BIND_MAX");
		return value != nullptr && value[0] != 0 ? std::strtoull(value, nullptr, 0) : uint64_t {65536};
	}();
	return bytes;
}

void NoteBind(BindFallback outcome) {
	GetManager().bind[static_cast<size_t>(outcome)].fetch_add(1, std::memory_order_relaxed);
}

bool LookupBinding(uint64_t vaddr, uint64_t size, VkBuffer* buffer, uint64_t* offset) {
	const auto address = Lookup(vaddr, size);
	if (address == 0) return false;
	auto&          m = GetManager();
	uint64_t       backing_offset = 0;
	// Lookup succeeded: the same translation (mappings do not change on the GPU thread's draw).
	if (!LibKernel::Memory::GuestToBacking(vaddr, size, &backing_offset)) return false;
	const auto& chunk = m.chunks[static_cast<uint32_t>(backing_offset >> ChunkBits)];
	*buffer           = chunk.buffer;
	*offset           = backing_offset & (ChunkSize - 1);
	return *buffer != VK_NULL_HANDLE;
}

uint64_t Lookup(uint64_t vaddr, uint64_t size) {
	auto& m = GetManager();
	if (!m.ready.load(std::memory_order_acquire) || size == 0) return 0;
	m.lookups.fetch_add(1, std::memory_order_relaxed);
	uint64_t offset = 0;
	if (!LibKernel::Memory::GuestToBacking(vaddr, size, &offset) || offset >= m.backing_size ||
	    m.backing_size - offset < size) {
		m.unmapped.fetch_add(1, std::memory_order_relaxed);
		return 0;
	}
	const auto index = static_cast<uint32_t>(offset >> ChunkBits);
	if (((offset + size - 1) >> ChunkBits) != index) {
		m.cross.fetch_add(1, std::memory_order_relaxed);
		return 0;
	}
	auto&      chunk = m.chunks[index];
	const auto state = chunk.state.load(std::memory_order_acquire);
	if (state == ChunkReady) {
		m.hits.fetch_add(1, std::memory_order_relaxed);
		return chunk.address.load(std::memory_order_relaxed) + (offset & (ChunkSize - 1));
	}
	if (state == ChunkFailed) {
		m.failed_lookups.fetch_add(1, std::memory_order_relaxed);
		return 0;
	}
	m.pending.fetch_add(1, std::memory_order_relaxed);
	uint8_t expected = ChunkNone;
	if (chunk.state.compare_exchange_strong(expected, ChunkRequested, std::memory_order_acq_rel)) {
		{
			std::lock_guard lock(m.mutex);
			m.queue.push_back(index);
		}
		m.wake.notify_one();
	}
	return 0;
}

void Start(GraphicContext& ctx) {
	if (!Enabled()) return;
	auto& m = GetManager();
	if (!HostImportProbe::ExtensionEnabled()) {
		Log("VK_EXT_external_memory_host not available: off");
		return;
	}
	if (!LibKernel::Memory::DiagGuestBacking(&m.backing, &m.backing_size)) {
		Log("guest backing section not available: off");
		return;
	}
	m.device       = static_cast<VkDevice>(ctx.device);
	const auto& d  = VULKAN_HPP_DEFAULT_DISPATCHER;
	auto physical  = static_cast<VkPhysicalDevice>(ctx.physical_device);
	if (d.vkGetPhysicalDeviceProperties2 != nullptr) {
		VkPhysicalDeviceExternalMemoryHostPropertiesEXT host {};
		host.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
		VkPhysicalDeviceProperties2 props {};
		props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		props.pNext = &host;
		d.vkGetPhysicalDeviceProperties2(physical, &props);
		if (host.minImportedHostPointerAlignment != 0) m.alignment = host.minImportedHostPointerAlignment;
	}
	d.vkGetPhysicalDeviceMemoryProperties(physical, &m.memory);
	if ((m.alignment & (m.alignment - 1)) != 0 || (m.backing & (m.alignment - 1)) != 0 ||
	    m.alignment > ChunkSize) {
		Log("alignment unusable: off");
		return;
	}
	m.chunk_count = static_cast<uint32_t>((m.backing_size + ChunkSize - 1) >> ChunkBits);
	m.chunks      = std::make_unique<Chunk[]>(m.chunk_count);
	m.stop        = false;
	m.worker      = std::thread(WorkerLoop);
	m.ready.store(true, std::memory_order_release);
	Log(Format("on (%s): backing 0x%016" PRIx64 " size 0x%" PRIx64 ", %u chunks of 64 MiB imported on demand",
	           RaceEnabled() ? "race: guest rewrites before GPU completion measured" : BindEnabled() ? "bind: small read-only buffer bindings read in place" : CensusEnabled() ? "census of the buffer read bindings" : "no consumer yet", m.backing,
	           m.backing_size, m.chunk_count));
}

void Stop() {
	auto& m = GetManager();
	if (!m.worker.joinable()) return;
	m.ready.store(false, std::memory_order_release);
	{
		std::lock_guard lock(m.mutex);
		m.stop = true;
	}
	m.wake.notify_all();
	m.worker.join();
	Report(m);
	const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	for (uint32_t i = 0; i < m.chunk_count; i++) {
		auto& chunk = m.chunks[i];
		if (chunk.buffer != VK_NULL_HANDLE) d.vkDestroyBuffer(m.device, chunk.buffer, nullptr);
		if (chunk.memory != VK_NULL_HANDLE) d.vkFreeMemory(m.device, chunk.memory, nullptr);
		chunk.buffer = VK_NULL_HANDLE;
		chunk.memory = VK_NULL_HANDLE;
	}
}

} // namespace Libs::Graphics::HostImport
