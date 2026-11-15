#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace Libs::Graphics {

struct GraphicContext;
class MasterSemaphore;

class DescriptorHeap {
public:
	DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore);
	~DescriptorHeap();
	KYTY_CLASS_NO_COPY(DescriptorHeap);

	[[nodiscard]] vk::DescriptorSet Commit(vk::DescriptorSetLayout layout);

	// Astro Bot PC (KYTY_CP_PAR_LANE_DESCRIPTORS=1, drawPrep/laneShare.h, C2): this heap is lane A's
	// (and every thread without a lane's); lanes B.. get their own heap (pools, batches, retired at
	// the lane's own ticks), created at the lane's first set. Start-up only.
	void EnableLaneHeaps() noexcept { m_lane_split = true; }

private:
	[[nodiscard]] vk::DescriptorSet CommitOwn(vk::DescriptorSetLayout layout);
	[[nodiscard]] DescriptorHeap&   LaneHeap(uint32_t lane);
	// KYTY_CP_PAR_VERIFY with the lane heaps: the set handed out now (double handing out check).
	vk::DescriptorSet Handed(vk::DescriptorSet set);

	static constexpr uint32_t DescriptorSetBatch = 32;

	struct Batch {
		std::array<vk::DescriptorSet, DescriptorSetBatch> sets {};
		uint32_t                                          size       = 0;
		uint32_t                                          allocation = DescriptorSetBatch;
	};

	[[nodiscard]] bool Allocate(vk::DescriptorSetLayout layout, Batch& batch);
	void               CreateDescriptorPool();

	GraphicContext&                                     m_graphics;
	MasterSemaphore&                                    m_master_semaphore;
	vk::DescriptorPool                                  m_current_pool = nullptr;
	std::deque<std::pair<vk::DescriptorPool, uint64_t>> m_pending_pools;
	// Astro Bot PC (KYTY_CP_PAR=2b, tick ranges): the largest tick any set of the current pool was
	// handed out at; the pool retires at it (a lane's tick may be below another lane's).
	uint64_t                                            m_current_pool_tick = 0;
	std::unordered_map<vk::DescriptorSetLayout, Batch>  m_sets;
	// Astro Bot PC (KYTY_CP_PAR_LANE_DESCRIPTORS): the lane heaps (index = lane; [0] unused) and
	// this heap's lane; the largest tick a set of the current pool was handed out at (verify).
	bool                                                m_lane_split = false;
	uint32_t                                            m_heap_lane  = 0;
	std::array<std::atomic<DescriptorHeap*>, 4>         m_lane_heaps {};
	std::array<std::unique_ptr<DescriptorHeap>, 4>      m_lane_owned;
	std::mutex                                          m_lane_mutex;
	uint64_t                                            m_pool_handed_tick = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
