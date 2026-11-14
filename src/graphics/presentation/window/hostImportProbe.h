#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_HOSTIMPORTPROBE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_HOSTIMPORTPROBE_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <vector>

// Astro Bot PC (KYTY_HOST_IMPORT_PROBE=1, diagnostic, default off; step S0 of the "unified memory"
// track, docs\PLAN-PARALLELE.md): can guest memory be imported as Vulkan device memory
// (VK_EXT_external_memory_host, HOST_ALLOCATION) and read through a buffer device address?
// Probes, each logged as "HostImport probe: ..." lines and freed at once:
//  (a) a 64 MiB, 4 KiB-aligned sub-range of the writable alias view of the guest direct/flexible
//      backing section (GuestBackingStore::m_backing_base);
//  (b) a sub-range of a guest view of that section (the largest direct mapping, and the largest
//      flexible one), taken by a background thread once the game has mapped one (up to 3 min);
//  (c) a plain VirtualAlloc buffer (control).
// For each: vkGetMemoryHostPointerPropertiesEXT, vkAllocateMemory with the import, the memory
// type, then a VkBuffer (SHADER_DEVICE_ADDRESS) bound to it and its device address. Nothing the
// emulator uses is touched. Off: the extension is not enabled and nothing runs.
namespace Libs::Graphics {
struct GraphicContext;
} // namespace Libs::Graphics

namespace Libs::Graphics::HostImportProbe {

[[nodiscard]] bool Enabled();
// Device creation: appends VK_EXT_external_memory_host when the probe is on and it is available.
void AppendDeviceExtension(std::vector<const char*>&                 device_extensions,
                           const std::vector<vk::ExtensionProperties>& available);
// After the device exists (and the guest address space): probes (a) and (c), starts (b).
void Start(GraphicContext& ctx);
// Before the device is destroyed: stops and joins the (b) thread.
void Stop();
// VK_EXT_external_memory_host was enabled on the device (by the probe or by KYTY_HOST_IMPORT).
[[nodiscard]] bool ExtensionEnabled();

} // namespace Libs::Graphics::HostImportProbe

// Astro Bot PC: KYTY_HOST_IMPORT (default off; step B0 of the "buffers by address" track,
// docs\PLAN-PARALLELE.md). The guest's direct/flexible backing section is imported as Vulkan
// memory (VK_EXT_external_memory_host, host-coherent system memory), 64 MiB of its alias view at
// a time, on demand, by a background thread: a chunk stays imported until shutdown (the backing
// outlives every guest view of it). Lookup gives the device address of a guest range held whole
// by one guest view of the backing and one imported chunk (0 otherwise, and the chunk is then
// requested). 1: on, no consumer; census: ObtainReadBinding looks every buffer read binding up
// (diagnostic). One "HostImport 10s: ..." line every 10 s.
namespace Libs::Graphics::HostImport {

[[nodiscard]] bool Enabled();
[[nodiscard]] bool CensusEnabled();
// Any thread. 0: not available now.
[[nodiscard]] uint64_t Lookup(uint64_t vaddr, uint64_t size);
void Start(GraphicContext& ctx);
void Stop();

} // namespace Libs::Graphics::HostImport

#endif /* EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_HOSTIMPORTPROBE_H_ */
