#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_

// CP recorder (P2): Vulkan emission off the command-processor thread.
// Design and invariants: Profiling/analysis/CP-RECORDER-P2.md.
//
// KYTY_CP_RECORDER=0 (default) | 1 | inline
//   1:      the guest CommandScheduler encodes its native commands (vkBegin/EndCommandBuffer, every
//           vkCmd* of the converted paths and the queue hand-off) into a ring (commandStream.h);
//           one recorder thread replays them in order and submits exactly as the CP did.
//   inline: the same encoding, replayed immediately on the CP (no thread) - isolates the
//           encode/decode path from threading.
// KYTY_CP_RECORDER_VERIFY=1 | exit
//   Per packet: the CP's hash of its original call arguments against the recorder's hash of the
//   arguments passed to the driver, and sequence numbers; per command buffer: digest and count at
//   Submit; ownership: dispatcher hooks reject native recording into the guest command buffer
//   outside the recorder and outside a CP direct window. `exit` stops at the first fault.
// KYTY_CP_RECORDER_RING_MB (16), KYTY_CP_RECORDER_SPIN_US (30: recorder idle spin and CP ring-full
// spin), KYTY_CP_RECORDER_DRAIN_SPIN_US (2000: CP spin in a drain before blocking),
// KYTY_CP_RECORDER_DRAIN_LOG=1 (per-site drain histogram at shutdown),
// KYTY_CP_RECORDER_IDEAL_CPU=<n> (recorder thread ideal processor hint).

#include "common/common.h"
#include "graphics/host_gpu/renderer/commandStream.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace Libs::Graphics {

struct GraphicContext;
class MasterSemaphore;
class GpuTimestampRing;

class CommandRecorder {
public:
	enum class Mode : uint8_t { Off, Thread, Inline };

	// KYTY_CP_RECORDER (evaluated once). Off when the barrier batcher is disabled: the batcher's
	// flush points are the recorder's only way to order batched barriers against native paths.
	[[nodiscard]] static Mode ConfiguredMode();
	[[nodiscard]] static bool VerifyEnabled();
	[[nodiscard]] static bool VerifyExits();

	CommandRecorder(GraphicContext& graphics, MasterSemaphore& master, GpuTimestampRing* timing,
	                bool gpu_ops, Mode mode);
	~CommandRecorder();
	KYTY_CLASS_NO_COPY(CommandRecorder);

	[[nodiscard]] Mode GetMode() const noexcept { return m_mode; }

	// ---- Producer (the CP thread) ----

	[[nodiscard]] CommandStream::Encoder& Encoder() noexcept { return m_encoder; }

	// Astro Bot PC (KYTY_CP_PAR=1b2, cpParallel.h): a second producer stream (its own ring and
	// encoder) for the second recording context. The calling thread's stream (ThreadStream, 0 by
	// default) picks the ring that Begin, Submit, Drain and WaitRecorded's wake use. Each stream's
	// segments end with EndStreamSegment (an Op::Switch packet); the replay follows them from ring
	// to ring, starting in stream 0. Thread mode without verify only; call before any packet of
	// stream 1 (returns false, nothing changed, otherwise).
	bool EnableSecondStream();
	[[nodiscard]] bool DualStreams() const noexcept { return m_stream_count > 1; }
	[[nodiscard]] CommandStream::Encoder& EncoderB() noexcept { return *m_encoders_x[1]; }
	// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): streams 2..count-1 (rings and encoders) after
	// EnableSecondStream, at most MaxStreams in all, before any packet of them. Returns false
	// (nothing changed) without stream 1.
	static constexpr uint32_t MaxStreams = 4;
	bool EnableStreams(uint32_t count);
	[[nodiscard]] uint32_t StreamCount() const noexcept { return m_stream_count; }
	[[nodiscard]] CommandStream::Encoder& StreamEncoder(uint32_t stream) noexcept {
		return stream != 0 ? *m_encoders_x[stream] : m_encoder;
	}
	static void                   SetThreadStream(uint32_t stream) noexcept;
	[[nodiscard]] static uint32_t ThreadStream() noexcept;
	// The calling thread's stream: ends its segment (the replay continues in the other ring).
	// Astro Bot PC (KYTY_CP_PAR=2a): `target` names the ring the replay continues in (the lane
	// taking the token); without it (1b2, 1b3: two streams) the replay goes to the other ring.
	void EndStreamSegment();
	void EndStreamSegment(uint32_t target);
	// Astro Bot PC (KYTY_CP_PAR=1b3): the calling thread's ring, bytes written but not released by
	// the replay yet, and its capacity (its producer only).
	[[nodiscard]] uint64_t StreamBacklog() noexcept {
		auto& ring = ProducerRing();
		return ring.WritePosition() - ring.Consumed();
	}
	[[nodiscard]] uint64_t StreamCapacity() noexcept { return ProducerRing().Capacity(); }
	// Packets encoded so far by stream stream (its producer thread).
	[[nodiscard]] uint64_t StreamPackets(uint32_t stream) const noexcept {
		return stream != 0 && stream < m_stream_count ? m_encoders_x[stream]->Packets() : m_encoder.Packets();
	}
	// Begin of a guest command buffer: the native buffer the CP picked from its pool.
	void Begin(vk::CommandBuffer command, uint64_t tick, uint64_t record_ns);
	// Astro Bot PC (KYTY_CP_PAR=2a, lazy begin): the calling thread's stream keeps the Begin
	// packet pending until its next packet (CommandStream::Encoder::DeferBegin); a Drain encodes it
	// first. BeginPending: nothing was encoded since; CancelPendingBegin drops it.
	void DeferBegin(vk::CommandBuffer command, uint64_t tick, uint64_t record_ns) {
		ProducerEncoder().DeferBegin(command, tick, record_ns);
	}
	[[nodiscard]] bool BeginPending() noexcept { return ProducerEncoder().BeginPending(); }
	void               CancelPendingBegin() noexcept { ProducerEncoder().CancelPendingBegin(); }
	// Hands the current command buffer to the recorder for vkEndCommandBuffer and submission.
	void Submit(const CommandStream::SubmitPacket& submit);
	// Waits until the recorder has executed every packet encoded so far (a direct window may then
	// be opened: the recorder is idle and the native buffer holds everything in order).
	// `site_key` attributes the drain (KYTY_CP_RECORDER_DRAIN_LOG).
	void Drain(const void* site_key, bool is_site);
	// Direct-window bookkeeping of the CP (verify ownership hooks). The guest command buffer is
	// recorded by the replayer only while no window is open, and by the producer side (any
	// thread driving the scheduler; tests hand it between threads) only while one is open.
	void SetWindowOpen(bool open) noexcept { m_window_open.store(open, std::memory_order_release); }
	// Waits until the Submit packet of `tick` has been executed (handed to the broker or the
	// queue). From the producer it first wakes the recorder; any thread may call it.
	void WaitRecorded(uint64_t tick, bool from_producer);
	// Ring positions (CP thread): bytes encoded so far, and executed by the recorder. A packet
	// encoded before WritePosition() returned P has executed once Consumed() >= P.
	[[nodiscard]] uint64_t RingWritePosition() const noexcept { return m_ring.WritePosition(); }
	[[nodiscard]] uint64_t RingConsumed() const noexcept { return m_ring.Consumed(); }
	// The tick of the last Submit packet executed.
	[[nodiscard]] uint64_t RecordedTick() const noexcept {
		return m_recorded_tick.load(std::memory_order_acquire);
	}
	// Drains and stops the recorder thread (scheduler shutdown). Idempotent.
	void Stop();

	// Verify hooks: installed once after the dispatcher is initialized (and after the GpuOpProfiler
	// hooks, which they wrap). No-op unless verify mode is on.
	static void InstallVerifyHooks();

	// KYTY_CMD_GUARD=1 (diagnostic, default off; "exit" stops at the first violation): a small
	// command-buffer state machine on the dispatcher for every pool (guest, presenter, side,
	// upload DMA, BDA table): recording outside Begin/End, two threads in one pool at once, a
	// submit of a buffer that is not executable, and a reset/begin/free of a buffer whose
	// timeline signal has not been reached. Installed after the verify hooks.
	static void InstallGuardHooks(vk::Device device);
	[[nodiscard]] static uint64_t GuardViolations() noexcept;

	// Called by hooks and diagnostics: whether the calling thread may record into `command`.
	[[nodiscard]] static bool MayRecord(VkCommandBuffer command) noexcept;

	// ---- Diagnostics and tests ----
	// Ownership faults reported by the verify hooks (process-wide).
	[[nodiscard]] static uint64_t OwnershipFaults() noexcept;
	// Tests: holds the recorder thread before its next batch (packets stay in the ring).
	void SetTestStall(bool stall) noexcept { m_test_stall.store(stall, std::memory_order_release); }
	[[nodiscard]] const CommandStream::WaitStats& ProducerStats() const noexcept {
		return m_encoder.Stats();
	}
	[[nodiscard]] uint64_t Drains() const noexcept { return m_drains; }
	[[nodiscard]] uint64_t IdleDrains() const noexcept { return m_idle_drains; }
	[[nodiscard]] uint64_t DrainNs() const noexcept { return m_drain_ns; }
	// Recorder-thread time replaying packets (read after a drain or Stop).
	[[nodiscard]] uint64_t BusyNs() const noexcept {
		return m_busy_ns.load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t VerifyMismatches() const noexcept { return m_replay.mismatches; }

private:
	struct NativeExecutor;
	friend struct NativeExecutor;

	void Run(std::stop_token stop);
	// Replays every published packet (recorder thread, or the CP in inline mode).
	void        ConsumeAvailable();
	void        AfterCommit();
	static void AfterCommitThunk(void* self);
	static void OnMismatch(void* self, uint32_t kind, CommandStream::Op op, uint64_t sequence,
	                       uint64_t expected, uint64_t actual);
	void        PublishCounters();
	void        PublishConsumerCounters();
	void        SamplePlacement();
	void        PrintDrainLog();

	GraphicContext&                 m_graphics;
	MasterSemaphore&                m_master;
	GpuTimestampRing*               m_timing  = nullptr;
	bool                            m_gpu_ops = false;
	Mode                            m_mode    = Mode::Off;
	CommandStream::Ring             m_ring;
	CommandStream::Encoder          m_encoder;
	CommandStream::ReplayState      m_replay;
	CommandStream::WaitPolicy       m_drain_policy;
	CommandStream::WaitPolicy       m_idle_policy;
	std::unique_ptr<NativeExecutor> m_exec;
	// Astro Bot PC (KYTY_CP_PAR=1b2): stream 1 (null unless EnableSecondStream), the ring the
	// replay reads now (consumer-local), the Switch packets it followed and the serial of the last
	// Switch encoded (producers, in turn).
	// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): streams 1..m_stream_count-1, index = stream ([0]
	// unused: stream 0 is m_ring / m_encoder).
	std::array<std::unique_ptr<CommandStream::Ring>, MaxStreams>    m_rings_x;
	std::array<std::unique_ptr<CommandStream::Encoder>, MaxStreams> m_encoders_x;
	uint32_t                                m_stream_count  = 1;
	CommandStream::Ring*                    m_consume_ring  = nullptr;
	uint64_t                                m_switch_serial = 0;
	[[nodiscard]] CommandStream::Ring& ProducerRing() noexcept {
		const auto stream = ThreadStream();
		return stream != 0 && stream < m_stream_count ? *m_rings_x[stream] : m_ring;
	}
	[[nodiscard]] CommandStream::Encoder& ProducerEncoder() noexcept {
		const auto stream = ThreadStream();
		return stream != 0 && stream < m_stream_count ? *m_encoders_x[stream] : m_encoder;
	}
	// Every packet of both streams executed (both rings consumed up to what was published).
	[[nodiscard]] bool StreamsIdle() const noexcept;

	// Verify-hook registry slot (MaxRecorders when not registered).
	size_t m_registry_slot = SIZE_MAX;
	// Consumer-side stats (recorder thread; read by the producer only after a drain).
	CommandStream::WaitStats m_consumer_stats;
	uint64_t                 m_published_parks  = 0;
	uint64_t                 m_published_checks = 0;
	// Producer waits in drains (kept apart from ring-full waits).
	CommandStream::WaitStats m_drain_stats;
	std::atomic<uint64_t>    m_recorded_tick {0};
	std::atomic<uint32_t>    m_recorded_waiters {0};
	std::atomic<bool>        m_window_open {false};
	std::atomic<bool>        m_stop {false};
	std::atomic<bool>        m_test_stall {false};
	std::atomic<uint64_t>    m_busy_ns {0};
	bool                     m_stopped      = false;
	uint64_t                 m_drain_serial = 0;

	// Producer-side counters published at every Submit (FrameEvents / FrameWaits).
	uint64_t m_published_packets  = 0;
	uint64_t m_published_bytes    = 0;
	uint64_t m_published_spins    = 0;
	uint64_t m_published_wait_ns  = 0;
	uint64_t m_published_wakes    = 0;
	uint64_t m_drains             = 0;
	uint64_t m_drain_ns           = 0;
	uint64_t m_drains_published   = 0;
	uint64_t m_drain_ns_published = 0;
	// Drains that found the recorder idle (no marker, no wake).
	uint64_t m_idle_drains           = 0;
	uint64_t m_idle_drains_published = 0;

	// KYTY_CP_RECORDER_DRAIN_LOG: drain count per site (GpuOpProfiler::Site*) or caller address.
	struct DrainKey {
		const void* key                               = nullptr;
		bool        is_site                           = false;
		bool        operator==(const DrainKey&) const = default;
	};
	struct DrainKeyHash {
		size_t operator()(const DrainKey& key) const noexcept {
			return std::hash<const void*> {}(key.key) ^ (key.is_site ? 1u : 0u);
		}
	};
	struct DrainStats {
		uint64_t count     = 0;
		uint64_t idle      = 0;
		uint64_t ns        = 0;
		uint64_t waited_ns = 0; // non-idle drains only
	};
	std::unordered_map<DrainKey, DrainStats, DrainKeyHash> m_drain_log;
	uint64_t                                               m_drain_log_printed_ns = 0;

	// Placement sampling (recorder thread): the CP's last processor, published at Submit.
	std::atomic<uint32_t> m_cp_processor {UINT32_MAX};
	uint64_t              m_placement_batches = 0;

	std::jthread m_thread; // last: started after every member above
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
