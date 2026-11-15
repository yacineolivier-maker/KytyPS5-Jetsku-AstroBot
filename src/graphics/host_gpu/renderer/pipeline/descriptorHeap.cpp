#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include "graphics/guest_gpu/command_processor/cpParallel.h" // Astro Bot PC (KYTY_CP_PAR=2b)

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/drawPrep/laneShare.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {
namespace {

constexpr uint32_t   DescriptorHeapCount = 1024;
constexpr std::array DescriptorPoolSizes = {
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eUniformBuffer, 4096},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 1024},
};

// KYTY_DESCRIPTOR_POOL_SCALE (default 4, 1..64): pools hold this many times the sets and
// descriptors above. A full pool retires with its unused batches and the next one is created (or
// an idle one reset) on the command processor. At the Sky Garden start about 1,000 sets a flip are
// written, about a pool's worth, and the driver calls under Commit (batch allocations, pool
// creation and resets) were about 1% of the command processor's samples. A larger pool only
// changes how often a pool turns over.
uint32_t DescriptorPoolScale() {
	static const uint32_t scale = [] {
		const char* value = std::getenv("KYTY_DESCRIPTOR_POOL_SCALE");
		if (value == nullptr || value[0] == '\0') {
			return 4u;
		}
		return std::clamp(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 1u, 64u);
	}();
	return scale;
}

} // namespace

DescriptorHeap::DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore)
    : m_graphics(graphics), m_master_semaphore(master_semaphore) {
	CreateDescriptorPool();
}

DescriptorHeap::~DescriptorHeap() {
	m_graphics.device.destroyDescriptorPool(m_current_pool, nullptr);
	for (const auto& [pool, tick]: m_pending_pools) {
		m_master_semaphore.Wait(tick);
		m_graphics.device.destroyDescriptorPool(pool, nullptr);
	}
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout layout) {
	if (m_lane_split) [[unlikely]] {
		// Astro Bot PC (KYTY_CP_PAR_LANE_DESCRIPTORS): the recording thread's lane's heap.
		const auto lane = CommandScheduler::ThreadContext();
		if (lane != 0 && lane < m_lane_heaps.size()) {
			return LaneHeap(lane).CommitOwn(layout);
		}
	}
	return CommitOwn(layout);
}

DescriptorHeap& DescriptorHeap::LaneHeap(uint32_t lane) {
	if (auto* heap = m_lane_heaps[lane].load(std::memory_order_acquire); heap != nullptr) {
		return *heap;
	}
	const std::lock_guard lock(m_lane_mutex);
	if (auto* heap = m_lane_heaps[lane].load(std::memory_order_acquire); heap != nullptr) {
		return *heap;
	}
	m_lane_owned[lane]              = std::make_unique<DescriptorHeap>(m_graphics, m_master_semaphore);
	m_lane_owned[lane]->m_heap_lane = lane;
	m_lane_heaps[lane].store(m_lane_owned[lane].get(), std::memory_order_release);
	LaneShare::Note(lane, LaneShare::Stat::DescriptorHeaps);
	return *m_lane_owned[lane];
}

namespace {
// KYTY_CP_PAR_VERIFY with the lane heaps: the tick each set was last handed out at, all heaps.
std::mutex                                       g_handed_mutex;
std::unordered_map<VkDescriptorSet, uint64_t>    g_handed;
} // namespace

vk::DescriptorSet DescriptorHeap::Handed(vk::DescriptorSet set) {
	const auto tick = m_master_semaphore.CurrentTick();
	m_pool_handed_tick = std::max(m_pool_handed_tick, tick);
	if (m_heap_lane != 0) {
		LaneShare::Note(m_heap_lane, LaneShare::Stat::DescriptorSets);
	}
	if (!LaneShare::g_verify) {
		return set;
	}
	const std::lock_guard lock(g_handed_mutex);
	auto [it, fresh] = g_handed.try_emplace(static_cast<VkDescriptorSet>(set), tick);
	if (!fresh) {
		const auto previous = it->second;
		bool       free     = m_master_semaphore.IsFree(previous);
		if (!free) {
			m_master_semaphore.Refresh();
			free = m_master_semaphore.IsFree(previous);
		}
		if (!free) {
			LaneShare::Note(m_heap_lane, LaneShare::Stat::DescriptorDouble);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 8u) {
				std::printf("Kyty CpPar lane share: descriptor set handed out again by lane %c's heap at tick "
				            "%" PRIu64 " while its previous tick %" PRIu64 " is not free\n",
				            static_cast<char>('A' + m_heap_lane), tick, previous);
			}
		}
		it->second = tick;
	}
	return set;
}

vk::DescriptorSet DescriptorHeap::CommitOwn(vk::DescriptorSetLayout layout) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(layout == nullptr);
	const bool lane_heaps = LaneShare::g_descriptors && (m_lane_split || m_heap_lane != 0);

	if (m_master_semaphore.Ranges()) {
		// Astro Bot PC (KYTY_CP_PAR=2b): the pool retires at the largest tick of its users.
		m_current_pool_tick = std::max(m_current_pool_tick, m_master_semaphore.CurrentTick());
	}
	auto& batch = m_sets[layout];
	if (batch.size != 0) {
		const auto set = batch.sets[--batch.size];
		return lane_heaps ? Handed(set) : set;
	}
	if (Allocate(layout, batch)) {
		const auto set = batch.sets[--batch.size];
		return lane_heaps ? Handed(set) : set;
	}
	if (lane_heaps) {
		// Astro Bot PC (KYTY_CP_PAR_LANE_DESCRIPTORS): the pool retires below.
		const auto retire = m_master_semaphore.Ranges() ? m_current_pool_tick : m_master_semaphore.CurrentTick();
		if (m_heap_lane != 0) {
			LaneShare::Note(m_heap_lane, LaneShare::Stat::DescriptorPools);
		}
		if (LaneShare::g_verify && retire < m_pool_handed_tick) {
			LaneShare::Note(m_heap_lane, LaneShare::Stat::DescriptorEarly);
		}
		m_pool_handed_tick = 0;
	}

	if (m_master_semaphore.Ranges() && m_master_semaphore.CurrentTick() < m_current_pool_tick) {
		// Astro Bot PC (KYTY_CP_PAR=2b): retiring at the caller's tick would have been too early.
		CpPar::NoteLaneStat(std::min<uint32_t>(MasterSemaphore::ThreadTickLane(), 3u),
		                    CpPar::LaneStat::EarlyRetires, 1);
	}
	m_pending_pools.emplace_back(m_current_pool, m_master_semaphore.Ranges()
	                                                 ? m_current_pool_tick
	                                                 : m_master_semaphore.CurrentTick());
	m_current_pool_tick = 0;
	if (const auto& [pool, tick] = m_pending_pools.front(); m_master_semaphore.IsFree(tick)) {
		m_current_pool = pool;
		m_pending_pools.pop_front();
		EXIT_IF(m_graphics.device.resetDescriptorPool(m_current_pool, {}) != vk::Result::eSuccess);
	} else {
		CreateDescriptorPool();
	}

	m_sets.clear();
	auto& fresh_batch = m_sets[layout];
	EXIT_IF(!Allocate(layout, fresh_batch));
	const auto set = fresh_batch.sets[--fresh_batch.size];
	return lane_heaps ? Handed(set) : set;
}

bool DescriptorHeap::Allocate(vk::DescriptorSetLayout layout, Batch& batch) {
	std::array<vk::DescriptorSetLayout, DescriptorSetBatch> layouts;
	layouts.fill(layout);

	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool = m_current_pool;
	allocate.pSetLayouts    = layouts.data();

	for (;;) {
		allocate.descriptorSetCount = batch.allocation;
		const auto result = m_graphics.device.allocateDescriptorSets(&allocate, batch.sets.data());
		if (result == vk::Result::eSuccess) {
			batch.size = batch.allocation;
			return true;
		}
		EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory &&
		        result != vk::Result::eErrorFragmentedPool);
		if (batch.allocation == 1) {
			return false;
		}
		batch.allocation /= 2;
	}
}

void DescriptorHeap::CreateDescriptorPool() {
	const auto scale = DescriptorPoolScale();
	auto       sizes = DescriptorPoolSizes;
	for (auto& size: sizes) {
		size.descriptorCount *= scale;
	}
	vk::DescriptorPoolCreateInfo create {};
	create.maxSets       = DescriptorHeapCount * scale;
	create.poolSizeCount = static_cast<uint32_t>(sizes.size());
	create.pPoolSizes    = sizes.data();
	EXIT_IF(m_graphics.device.createDescriptorPool(&create, nullptr, &m_current_pool) !=
	        vk::Result::eSuccess);
}

} // namespace Libs::Graphics
