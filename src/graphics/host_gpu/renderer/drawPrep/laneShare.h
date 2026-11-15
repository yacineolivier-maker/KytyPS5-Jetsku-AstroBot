#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_LANESHARE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_LANESHARE_H_

#include <cstdint>

// Astro Bot PC (KYTY_CP_PAR=2b, P3 prerequisites, docs-p3-drawindex.md §5): the structures a lane's
// draw writes that are shared between lanes today, made per lane, each behind its own switch
// (default 0: nothing changes). Set once at start-up (Configure, after 2b is enabled), before any
// lane records; read on the recording paths without a call. Every switch needs 2b (refused
// otherwise). With KYTY_CP_PAR_VERIFY=1 each step also checks itself (counters below, 0 expected).
//  - KYTY_CP_PAR_LANE_STREAM=1 (C1): each lane B.. has its own stream ring (StreamBuffer::Lane),
//    the per-draw shader data, small CPU-written buffers and 8-bit indices go to the ring of the
//    recording thread's lane; ticks in a lane's ring are that lane's (increasing). Verify: a range
//    a ring hands out never overlaps a range still in flight (a tick not free), and no lane thread
//    uses another lane's ring.
//  - KYTY_CP_PAR_LANE_DESCRIPTORS=1 (C2): each lane B.. has its own descriptor heap (pools and
//    batches), retired at its own ticks. Verify: a set is never handed out again while the tick of
//    its previous handing out is not free (a "double handing out"), across all heaps.
//  - KYTY_CP_PAR_LANE_PREREQS=1 (small prerequisites): image binding flags per lane (C3), the upload
//    batch depth per lane (W5), CommandScheduler::Active() on the calling thread's wrapper (W13),
//    LRU touches of images and buffers deferred per lane and applied at the op's end (W1/W2), the
//    eager readback flush request per lane (W12). (W8 and R8 need no switch: thread-local and
//    atomic, same values.)
// Console line every 10 s: "CpPar 10s: 2b lane share".
namespace Libs::Graphics::LaneShare {

inline bool g_stream      = false;
inline bool g_descriptors = false;
inline bool g_prereqs     = false;
inline bool g_verify      = false; // with KYTY_CP_PAR_VERIFY=1 and at least one switch
inline bool g_any         = false;

// Start-up, once (graphicsRun.cpp, after 2b is enabled; lanes_2b false: every switch stays off).
void Configure(bool lanes_2b);

enum class Stat : uint8_t {
	StreamRings,       // lane rings created
	StreamMaps,        // ranges handed out by a lane ring
	StreamWraps,       // wraps of a lane ring
	StreamOverlaps,    // verify: a range handed out over one still in flight (0 expected)
	StreamForeign,     // verify: a ring used by a thread of another lane (0 expected)
	DescriptorHeaps,   // lane heaps created
	DescriptorSets,    // sets handed out by a lane heap
	DescriptorPools,   // pools retired by a lane heap
	DescriptorDouble,  // verify: a set handed out again before its previous tick is free (0)
	DescriptorEarly,   // verify: a pool retired below the largest tick of its sets (0)
	TouchDeferred,     // LRU touches deferred (W1/W2)
	TouchFlushed,      // deferred touches applied
	TouchStale,        // deferred touches of an entry freed (or reused) since: dropped
	Count
};

void Note(uint32_t lane, Stat stat, uint64_t value = 1) noexcept;
// Any thread; prints the line every 10 s (cheap when no switch is on).
void MaybePrint() noexcept;

} // namespace Libs::Graphics::LaneShare

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_LANESHARE_H_
