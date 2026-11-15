#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFER_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

VK_DEFINE_HANDLE(VmaAllocation)

namespace Libs::Graphics {

class CommandBuffer;
class CommandScheduler;
struct StreamBufferTestAccess;
struct GraphicContext;

enum class MemoryUsage : uint8_t {
	DeviceLocal,
	Upload,
	Download,
	Stream,
};

inline constexpr vk::BufferUsageFlags ReadFlags =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

inline constexpr vk::BufferUsageFlags AllFlags =
    ReadFlags | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

class Buffer {
public:
	// transfer_shared: concurrent sharing with the upload DMA transfer queue family
	// (GraphicContext::transfer_queue_family), when the device has that queue.
	// host_cached: mapped cached host memory (CPU writes stay in the CPU caches, the GPU reads
	// them over the bus) instead of what `usage` picks (KYTY_STREAM_RING_HOST, BufferCache).
	// sparse_residency: a sparse residency buffer without memory (DeviceLocal only, needs
	// GraphicContext::sparse_residency_buffer_enabled); its owner binds memory to the ranges it
	// uses, the others read as zero (KYTY_BDA_PAGETABLE_SPARSE, BufferCache).
	Buffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
	       uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size,
	       bool transfer_shared = false, bool host_cached = false, bool sparse_residency = false);
	~Buffer();
	KYTY_CLASS_NO_COPY(Buffer);

	[[nodiscard]] vk::Buffer         Handle() const noexcept { return m_buffer; }
	[[nodiscard]] uint64_t           Size() const noexcept { return m_size; }
	[[nodiscard]] std::span<uint8_t> Mapped() const noexcept { return m_mapped; }
	[[nodiscard]] bool               IsCoherent() const noexcept { return m_coherent; }
	[[nodiscard]] bool               IsSparse() const noexcept { return m_sparse; }
	[[nodiscard]] MemoryUsage        Usage() const noexcept { return m_usage; }
	[[nodiscard]] uint64_t           CpuAddress() const noexcept { return m_cpu_address; }
	// GPU-thread command-recording revision, including writes that have not executed yet.
	[[nodiscard]] uint64_t ContentRevision() const noexcept { return m_content_revision; }
	void MarkContentWritten();
	[[nodiscard]] vk::DeviceAddress BufferDeviceAddress() const noexcept;
	[[nodiscard]] uint64_t           Offset(uint64_t address) const noexcept {
		return address - m_cpu_address;
	}
	[[nodiscard]] bool IsInBounds(uint64_t address, uint64_t size) const noexcept;
	void               IncreaseStreamScore(int score) noexcept { stream_score += score; }
	[[nodiscard]] int  StreamScore() const noexcept { return stream_score; }
	void               Flush(uint64_t offset, uint64_t size);
	void               Invalidate(uint64_t offset, uint64_t size);
	void CopyFrom(CommandBuffer& command, const Buffer& source, uint64_t source_offset,
	              uint64_t destination_offset, uint64_t size,
	              vk::AccessFlags source_before      = vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlags destination_before = vk::AccessFlagBits::eMemoryRead |
	                                                   vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlags source_after       = vk::AccessFlagBits::eMemoryRead |
	                                                   vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlags destination_after  = vk::AccessFlagBits::eMemoryRead |
	                                                   vk::AccessFlagBits::eMemoryWrite);
	void Fill(uint64_t offset, uint64_t size, uint32_t value);

	// BufferCache state lives directly on the resource.
	bool   is_deleted   = false;
	int    stream_score = 0;
	size_t lru_id       = 0;
	// KYTY_BUFFER_LRU_SKIP=1 (BufferCache::TouchBuffer): the tick the LRU item last received.
	mutable uint64_t lru_tick = 0;

protected:
	[[nodiscard]] GraphicContext&   Graphics() const noexcept { return *m_graphics; }
	[[nodiscard]] CommandScheduler& Scheduler() const noexcept { return *m_scheduler; }

private:
	[[nodiscard]] vk::BufferMemoryBarrier Barrier(uint64_t offset, uint64_t size,
	                                              vk::AccessFlags source,
	                                              vk::AccessFlags destination) const;

	GraphicContext*               m_graphics    = nullptr;
	CommandScheduler*             m_scheduler   = nullptr;
	MemoryUsage                   m_usage       = MemoryUsage::DeviceLocal;
	uint64_t                      m_cpu_address = 0;
	uint64_t                      m_content_revision = 1;
	vk::DeviceAddress             m_device_address = 0;
	vk::Buffer                    m_buffer     = nullptr;
	VmaAllocation                 m_allocation = nullptr;
	uint64_t                      m_size;
	bool                          m_coherent = false;
	bool                          m_sparse   = false;
	std::span<uint8_t>            m_mapped;
	// KYTY_VRAM_STATS (vramStats.h): what this buffer was counted as (0 bytes: not counted).
	uint64_t                      m_vram_bytes        = 0;
	uint8_t                       m_vram_kind         = 0;
	bool                          m_vram_device_local = false;
};

class StreamBuffer final: public Buffer {
public:
	// extra_flags: usage beyond AllFlags (e.g. eShaderDeviceAddress for a ring whose ranges
	// shaders reach by device address).
	StreamBuffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
	             uint64_t size, bool transfer_shared = false,
	             vk::BufferUsageFlags extra_flags = {}, bool host_cached = false);

	[[nodiscard]] std::pair<uint8_t*, uint64_t> Map(uint64_t size, uint64_t alignment = 0,
	                                                bool allow_wait = true);
	void                                        Commit();
	[[nodiscard]] uint64_t Copy(const void* source, uint64_t size, uint64_t alignment = 0);

	// Astro Bot PC (KYTY_CP_PAR_LANE_STREAM=1, drawPrep/laneShare.h, C1): this ring is lane A's
	// (and every thread without a lane); lanes B.. get their own ring, of the same size and memory,
	// created at the lane's first use. Lane(): the ring of the calling thread's recorder stream
	// (CommandScheduler::ThreadContext); this ring unless EnableLaneRings ran. Start-up only.
	void EnableLaneRings() noexcept { m_lane_split = true; }
	[[nodiscard]] StreamBuffer& Lane() {
		if (!m_lane_split) [[likely]] {
			return *this;
		}
		return LaneSlow();
	}
	// One of the lane rings (or this ring).
	[[nodiscard]] bool IsRingOf(const Buffer* buffer) const noexcept;

private:
	friend struct StreamBufferTestAccess;

	[[nodiscard]] StreamBuffer& LaneSlow();
	// KYTY_CP_PAR_VERIFY with the lane rings: the ranges committed and not yet known free.
	void VerifyMap(uint64_t begin, uint64_t end, bool wrapped);
	void VerifyCommit(uint64_t begin, uint64_t end, uint64_t tick);
	struct LiveRange {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
	};
	bool                                              m_transfer_shared = false;
	vk::BufferUsageFlags                              m_extra_flags {};
	bool                                              m_host_cached = false;
	bool                                              m_lane_split  = false;
	uint32_t                                          m_ring_lane   = 0;
	std::array<std::atomic<StreamBuffer*>, 4>         m_lane_rings {};
	std::array<std::unique_ptr<StreamBuffer>, 4>      m_lane_owned;
	std::mutex                                        m_lane_mutex;
	std::deque<LiveRange>                             m_live;

	struct Watch {
		uint64_t tick        = 0;
		uint64_t upper_bound = 0;
	};

	[[nodiscard]] static bool NormalizeReservation(bool coherent, uint64_t atom, uint64_t& size,
	                                               uint64_t& alignment);
	[[nodiscard]] bool        WaitPendingOperations(const std::vector<Watch>& watches,
	                                                std::optional<size_t>     invalidation_mark,
	                                                uint64_t requested_upper_bound, bool allow_wait,
	                                                size_t& wait_cursor, uint64_t& wait_bound);

	uint64_t              m_offset      = 0;
	uint64_t              m_mapped_size = 0;
	std::vector<Watch>    m_current_watches;
	size_t                m_current_watch_cursor = 0;
	std::optional<size_t> m_invalidation_mark;
	std::vector<Watch>    m_previous_watches;
	size_t                m_wait_cursor = 0;
	uint64_t              m_wait_bound  = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFER_H_
