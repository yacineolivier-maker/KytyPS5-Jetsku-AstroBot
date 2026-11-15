#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/listCensus.h" // Astro Bot PC (KYTY_CP_PAR_CENSUS2)
#include "graphics/host_gpu/renderer/cache/bindingCensus.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/assert.h"
#include "common/prepPhases.h"
#include "common/paceLog.h"
#include "common/cpuPlacement.h"
#include "common/emulatorConfig.h"
#include "common/hangTrace.h"
#include "common/hangWatchdog.h"
#include "common/liveSwitch.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/cpOps.h"
#include "graphics/guest_gpu/command_processor/cpParallel.h" // Astro Bot PC (KYTY_CP_PAR)
#include "graphics/guest_gpu/command_processor/cpSequencer.h"
#include "graphics/guest_gpu/command_processor/parCensus.h"
#include "graphics/guest_gpu/command_processor/stateTrace.h"

// Astro Bot PC (KYTY_CP_PAR=1b4) : KYTY_CP_PAR_CLOCK=local (defaut) laisse le second fil ecrire lui-meme les horodatages
// (COPY_DATA de l'horloge, fins de travail avec horodatage) au lieu de repasser en serie : chaque liste commence par un
// horodatage, et ces valeurs ne servent qu'aux mesures de temps du jeu (decalees de quelques ms au plus).
// KYTY_CP_PAR_CLOCK=serial : comportement precedent (promotion).
static bool ClockLocal() {
	static const bool local = [] {
		const char* v = std::getenv("KYTY_CP_PAR_CLOCK");
		return v == nullptr || std::strcmp(v, "serial") != 0;
	}();
	return local;
}
#include "graphics/guest_gpu/command_processor/cpVerify.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/gpuTimestampScale.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/coherenceLog.h"
#include "graphics/host_gpu/gpuTouchedPages.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/readFaultDiag.h"
#include "graphics/host_gpu/renderer/commandRecorder.h" // Astro Bot PC (KYTY_CP_PAR=1b2)
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/cpCommit.h"
#include "graphics/host_gpu/renderer/drawPrep/commitStats.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"
#include "graphics/host_gpu/renderer/drawPrep/drawPrep.h"
#include "graphics/host_gpu/renderer/drawPrep/drawRun.h"
#include "graphics/host_gpu/renderer/drawPrep/packetClass.h"
#include "graphics/host_gpu/renderer/drawPrep/repeatTrace.h"
#include "graphics/host_gpu/renderer/eopTimestamps.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/image/image.h" // Astro Bot PC (KYTY_CP_PAR=1b3)
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/renderer/threadSampler.h"
#include "graphics/host_gpu/syncEpoch.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"
#include "loader/gameCapture.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <fmt/format.h>
#include <mutex>
#include <optional>
#include <semaphore>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility> // Astro Bot PC (n3b: std::exchange)
#include <xxhash.h>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_mutex_owned   = false;
static thread_local bool              g_gpu_thread        = false;
static thread_local GuestGpu*         g_gpu_state         = nullptr;
// KYTY_CP_SEQ=1: the sequencer thread (the graphics front); only it emits thread-mode ops.
static thread_local bool g_sequencer_thread = false;

// Astro Bot PC (KYTY_CP_PAR=1b3, cpParallel.h): the run-ahead's state. Every field is read and
// written under the CP op lock, except the offer's constant fields (target*, submit_id: set by
// Thread_Gpu before OfferRunAhead) and oldest_done (atomic: the run-ahead polls it unlocked).
namespace {
struct CpParRun {
	// The ring the replay is in, which is Thread_Gpu's recording context, and the rings whose
	// recorder segment is open (a ring that is not open has ended its segment with a Switch).
	uint32_t            open  = 0;
	std::array<bool, 2> begun = {true, false};
	// The offer: Thread_Gpu resolves `oldest`, the run-ahead may run the leading ops of `target`.
	bool              offered = false;
	uint64_t          oldest  = 0;
	std::atomic<bool> oldest_done {false};
	std::atomic<bool> ended {false}; // the run-ahead task returned (Thread_Gpu stops interleaving)
	uint64_t          target     = 0;
	uint32_t          target_ctx = 0;
	void*             target_sub = nullptr; // GuestGpu::Submission, in m_queues[0]
	uint64_t          submit_id  = 0;       // the submit id Thread_Gpu will give the target
	// The run-ahead's recorder segment is open and not reached by the replay (CpPar dirty).
	bool     began   = false;
	bool     started = false;
	uint64_t next_op = 0; // sequence + 1 of its last executed op
	// An insertion inside a run-ahead op: the replay returns to `back_ring` at the op's end.
	bool     switch_back = false;
	uint32_t back_ring   = 0;
	// The CP fields its resolver loop keeps across ops, between its ops; Thread_Gpu adopts them
	// when it continues the target (adopt_pending, set at the hand-over).
	uint64_t resolver_packets = 0;
	uint64_t retry_op         = 0;
	bool     adopt_pending    = false;
	// Astro Bot PC (KYTY_CP_PAR=1b5) statistics: the run-ahead's ops and draws (non-transport),
	// the offer's time (window to the merge).
	uint64_t ahead_ops   = 0;
	uint64_t ahead_draws = 0;
	uint64_t offer_ns    = 0;
};
CpParRun g_cp_par_run;
// The resolver context of the calling thread's ops (CommandProcessor::ResolvePrep), 1b3 only.
thread_local bool t_cp_par_resolve_b = false;
} // namespace

// Astro Bot PC (KYTY_CP_PAR=1b4, cpParallel.h): the CP address book. Read and written under the CP
// op lock only (Thread_Gpu's ops of queue-0 submissions and the run-ahead's ops). Per guest dword
// written by a CP op: the last submission that wrote it and the value (when the op's data says
// it); the dwords learned as CP-internal labels (awaited by a WAIT_REG_MEM of the submission that
// last wrote them) and those awaited by another submission (never CP-internal); during an overlap
// (offer to merge or hand-over), the dwords each side wrote and those the run-ahead read.
namespace {
struct CpParBook {
	struct Write {
		uint64_t submission = 0;
		uint32_t value      = 0;
		bool     known      = false;
		uint16_t kind       = static_cast<uint16_t>(CpSeq::OpKind::Count); // Astro Bot PC (1b5): the op
	};
	std::unordered_map<uint64_t, Write> writes;
	std::unordered_set<uint64_t>        labels;
	std::unordered_set<uint64_t>        shared;
	bool                                overlap = false;
	std::unordered_set<uint64_t>        ahead_written;
	std::unordered_set<uint64_t>        ahead_read;
	std::unordered_set<uint64_t>        oldest_written;
	void EndOverlap() {
		overlap = false;
		ahead_written.clear();
		ahead_read.clear();
		oldest_written.clear();
	}
};
CpParBook g_cp_par_book;

// The guest dwords an op writes (at most MaxDwords tracked; the values when known).
struct CpParRange {
	static constexpr uint32_t MaxDwords = 64;
	uint64_t                  address   = 0;
	uint32_t                  dwords    = 0;
	bool                      known     = false;
	uint32_t                  values[2] = {};      // known values (labels)
	const uint32_t*           source    = nullptr; // known values (WRITE_DATA inline data)
	[[nodiscard]] uint32_t Value(uint32_t i) const {
		return source != nullptr ? source[i] : values[i];
	}
};

// Astro Bot PC (KYTY_CP_PAR=1b5, cpParallel.h): the run-ahead's deferred interrupts and label
// stores, in execution order, delivered at the merge (CpParDeliverDeferred). Written by the
// run-ahead inside its ops, read and cleared by Thread_Gpu: both under the CP op lock.
struct CpParDeferredEntry {
	uint64_t                     sequence = 0; // the run-ahead op that raised it
	bool                         store    = false;
	uint16_t                     kind     = 0; // its op kind (stores: for the address book)
	uint64_t                     address  = 0; // stores
	uint64_t                     value    = 0;
	uint32_t                     size     = 0;
	Common::UniqueFunction<void> operation; // interrupts: the completion runner's operation
};
struct CpParDeferredList {
	std::vector<CpParDeferredEntry> entries;
	uint64_t                        submission    = 0; // N + 1
	uint64_t                        op_sequence   = 0; // the run-ahead op being executed
	uint16_t                        op_kind       = 0;
	bool                            capture_store = false; // its label store waits for the merge
};
CpParDeferredList g_cp_par_deferred;

// CommandScheduler::PriorityCapture of the run-ahead's ops: every EopInterrupt operation.
bool CpParCaptureInterrupt(Common::UniqueFunction<void>& operation) {
	auto&              list = g_cp_par_deferred;
	CpParDeferredEntry entry;
	entry.sequence  = list.op_sequence;
	entry.kind      = list.op_kind;
	entry.operation = std::move(operation);
	list.entries.push_back(std::move(entry));
	return true;
}

// Astro Bot PC (KYTY_CP_PAR=1b5): the rank of each queue-0 submission among those of its frame
// fence (-1: no fence), for the stop logs. Thread_Gpu, op lock held.
struct CpParRank {
	uint64_t sequence = 0;
	int64_t  rank     = -1;
};
std::array<CpParRank, 64> g_cp_par_ranks;
uint64_t                  g_cp_par_rank_fence = 0;
int64_t                   g_cp_par_rank_next  = 0;
uint64_t                  g_cp_par_rank_last  = 0;

void CpParNoteRank(uint64_t sequence, uint64_t fence) {
	if (sequence <= g_cp_par_rank_last) {
		return; // already ranked (another slice, or ranked as a run-ahead target)
	}
	g_cp_par_rank_last = sequence;
	int64_t rank       = -1;
	if (fence != 0) {
		if (fence != g_cp_par_rank_fence) {
			g_cp_par_rank_fence = fence;
			g_cp_par_rank_next  = 0;
		}
		rank = g_cp_par_rank_next++;
	}
	g_cp_par_ranks[sequence % g_cp_par_ranks.size()] = {sequence, rank};
}

[[nodiscard]] int64_t CpParRankOf(uint64_t sequence) {
	const auto& slot = g_cp_par_ranks[sequence % g_cp_par_ranks.size()];
	return slot.sequence == sequence ? slot.rank : -1;
}

// Astro Bot PC (KYTY_CP_PAR=1b5): the outcome of the last run-ahead target, counted while
// Thread_Gpu resolves it after the hand-over (Thread_Gpu only).
struct CpParTally {
	bool     active      = false;
	bool     started     = false;
	uint64_t submission  = 0;
	uint64_t ahead_ops   = 0;
	uint64_t ahead_draws = 0;
	uint64_t rest_ops    = 0;
	uint64_t rest_draws  = 0;
};
CpParTally g_cp_par_tally;

[[nodiscard]] constexpr bool CpParIsDraw(CpSeq::OpKind kind) noexcept {
	switch (kind) {
		case CpSeq::OpKind::DrawIndex:
		case CpSeq::OpKind::DrawAuto:
		case CpSeq::OpKind::DrawIndirect:
		case CpSeq::OpKind::DrawIndirectMulti:
		case CpSeq::OpKind::DispatchDirect:
		case CpSeq::OpKind::DispatchIndirect: return true;
		default: return false;
	}
}

// An op that raises an end-of-pipe interrupt (EOP INT_SEL 1/2, RELEASE_MEM INT_SEL 1/2/4).
[[nodiscard]] bool CpParOpRaisesInterrupt(CpSeq::OpKind kind, const void* payload) {
	if (kind == CpSeq::OpKind::EndOfPipe) {
		const auto selector = static_cast<const CpSeq::EndOfPipeOp*>(payload)->interrupt_selector;
		return selector != 0x00u && selector != 0x03u;
	}
	if (kind == CpSeq::OpKind::ReleaseMem) {
		const auto selector = (static_cast<const CpSeq::ReleaseMemOp*>(payload)->body[1] >> 24u) & 0x7u;
		return selector != 0x00u && selector != 0x03u;
	}
	return false;
}
} // namespace

// Astro Bot PC (KYTY_CP_PAR=1b5, cpParallel.h): the record-time store of a run-ahead interrupt
// op's label that the book does not know as CP-internal waits for the merge (queued before the
// op's interrupt, which follows it in the list). True when queued: the caller skips the store.
static bool CpParDeferLabelStore(const void* dst, uint64_t value, uint32_t size) {
	auto& list = g_cp_par_deferred;
	if (!CpPar::Deferred() || !list.capture_store || !CpPar::OnResolver2() || dst == nullptr) {
		return false;
	}
	CpParDeferredEntry entry;
	entry.sequence = list.op_sequence;
	entry.store    = true;
	entry.kind     = list.op_kind;
	entry.address  = reinterpret_cast<uint64_t>(dst);
	entry.value    = value;
	entry.size     = size;
	list.entries.push_back(std::move(entry));
	return true;
}

static bool CpParOpWrites(CpSeq::OpKind kind, const void* payload, const void* data,
                          CpParRange& range) {
	using CpSeq::OpKind;
	range = {};
	switch (kind) {
		case OpKind::WriteData: {
			const auto& op = *static_cast<const CpSeq::WriteDataOp*>(payload);
			if (op.dw_num == 0 || op.dst == 0) {
				return false;
			}
			const bool  one = ((op.write_control >> 16u) & 0x1u) != 0;
			const auto* src = static_cast<const uint32_t*>(data);
			range.address   = op.dst;
			range.dwords    = one ? 1u : op.dw_num;
			range.known     = src != nullptr;
			range.source    = src != nullptr ? (one ? src + (op.dw_num - 1u) : src) : nullptr;
			break;
		}
		case OpKind::EndOfPipe: {
			const auto& op = *static_cast<const CpSeq::EndOfPipeOp*>(payload);
			if (op.dst == 0) {
				return false;
			}
			range.address = op.dst;
			if (op.size == sizeof(uint32_t) && op.event_write_source == 0x01u) {
				range.dwords = static_cast<uint32_t>(op.value >> 16u) & 0xffffu; // GDS snapshot
			} else if (op.event_write_source == 0x02u) {
				range.dwords    = op.size == sizeof(uint64_t) ? 2u : 1u;
				range.known     = true;
				range.values[0] = static_cast<uint32_t>(op.value);
				range.values[1] = static_cast<uint32_t>(op.value >> 32u);
			} else if (op.event_write_source == 0x01u) {
				range.dwords    = 1; // 64-bit form with 32-bit data
				range.known     = true;
				range.values[0] = static_cast<uint32_t>(op.value);
			} else {
				range.dwords = 2; // clock
			}
			break;
		}
		case OpKind::ReleaseMem: {
			const auto* body     = static_cast<const CpSeq::ReleaseMemOp*>(payload)->body;
			const auto  data_sel = (body[1] >> 29u) & 0x7u;
			range.address        = body[2] | (static_cast<uint64_t>(body[3]) << 32u);
			if (data_sel == 0 || range.address == 0) {
				return false;
			}
			if (data_sel == 1 || data_sel == 2) {
				range.dwords    = data_sel == 1 ? 1u : 2u;
				range.known     = true;
				range.values[0] = body[4];
				range.values[1] = body[5];
			} else if (data_sel == 5) {
				range.dwords = (body[4] >> 16u) & 0xffffu; // GDS snapshot
			} else {
				range.dwords = 2; // clock
			}
			break;
		}
		case OpKind::ReferenceClock: {
			const auto& op = *static_cast<const CpSeq::ReferenceClockOp*>(payload);
			range.address  = op.dst;
			range.dwords   = op.num_bytes / 4u;
			break;
		}
		case OpKind::DmaData: {
			const auto& op = *static_cast<const CpSeq::DmaDataOp*>(payload);
			if ((op.dst_sel != 0 && op.dst_sel != 3) || static_cast<uint32_t>(op.dst) == 0x3022cu) {
				return false; // GDS, nowhere, the ignored register
			}
			range.address = op.dst;
			range.dwords  = op.num_bytes / 4u;
			break;
		}
		case OpKind::EventWrite: {
			const auto& op = *static_cast<const CpSeq::EventWriteOp*>(payload);
			if (op.event_type != 0x39u || op.address == 0) {
				return false;
			}
			range.address = op.address; // occlusion results
			range.dwords  = 62;
			break;
		}
		default: return false;
	}
	range.address &= ~uint64_t {3};
	range.dwords = std::min(range.dwords, CpParRange::MaxDwords);
	return range.address != 0 && range.dwords != 0;
}

// The guest dwords an op reads (WAIT_REG_MEM, DMA_DATA source in memory): false for none.
static bool CpParOpReads(CpSeq::OpKind kind, const void* payload, uint64_t& address,
                         uint32_t& dwords) {
	if (kind == CpSeq::OpKind::WaitRegMem) {
		const auto& op = *static_cast<const CpSeq::WaitRegMemOp*>(payload);
		address        = op.addr & ~uint64_t {3};
		dwords         = op.size == sizeof(uint64_t) ? 2u : 1u;
		return op.addr != 0;
	}
	if (kind == CpSeq::OpKind::DmaData) {
		const auto& op = *static_cast<const CpSeq::DmaDataOp*>(payload);
		if ((op.src_sel != 0 && op.src_sel != 3) || op.src == 0 || op.num_bytes == 0) {
			return false;
		}
		address = op.src & ~uint64_t {3};
		dwords  = std::min(op.num_bytes / 4u, CpParRange::MaxDwords);
		return dwords != 0;
	}
	return false;
}

// Thread_Gpu, before executing an op of a queue-0 submission: its reads of the run-ahead's writes.
static void CpParBookBeforeOldestOp(CpSeq::OpKind kind, const void* payload) {
	auto& book = g_cp_par_book;
	if (!book.overlap || book.ahead_written.empty()) {
		return;
	}
	uint64_t address = 0;
	uint32_t dwords  = 0;
	if (CpParOpReads(kind, payload, address, dwords)) {
		for (uint32_t i = 0; i < dwords; i++) {
			if (book.ahead_written.count(address + 4u * i) != 0) {
				CpPar::NoteCross(CpPar::Cross::OldestReadAhead, address + 4u * i);
				break;
			}
		}
	}
}

// Either thread, after an op of `submission` executed (`ahead`: by the run-ahead).
static void CpParBookAfterOp(CpSeq::OpKind kind, const void* payload, const void* data,
                             uint64_t submission, bool ahead, bool skip_writes = false) {
	auto&      book = g_cp_par_book;
	CpParRange range;
	// Astro Bot PC (KYTY_CP_PAR=1b5): a captured label store is booked at its delivery.
	if (!skip_writes && CpParOpWrites(kind, payload, data, range)) {
		for (uint32_t i = 0; i < range.dwords; i++) {
			const auto address = range.address + 4u * i;
			if (!ahead && book.overlap &&
			    (book.ahead_written.count(address) != 0 || book.ahead_read.count(address) != 0)) {
				CpPar::NoteCross(CpPar::Cross::OldestWriteAhead, address);
			}
			auto& write      = book.writes[address];
			write.submission = submission;
			write.known      = range.known && (range.source != nullptr || i < 2u);
			write.value      = write.known ? range.Value(i) : 0u;
			write.kind       = static_cast<uint16_t>(kind); // Astro Bot PC (KYTY_CP_PAR=1b5)
			if (book.overlap) {
				(ahead ? book.ahead_written : book.oldest_written).insert(address);
			}
		}
	}
	// Reads: a WAIT_REG_MEM teaches the labels (CP-internal when the waiting submission wrote the
	// dword last, shared otherwise); the run-ahead's reads of the overlap are kept.
	uint64_t read_address = 0;
	uint32_t read_dwords  = 0;
	if (CpParOpReads(kind, payload, read_address, read_dwords)) {
		for (uint32_t i = 0; i < read_dwords; i++) {
			const auto dword = read_address + 4u * i;
			if (kind == CpSeq::OpKind::WaitRegMem) {
				if (const auto found = book.writes.find(dword); found != book.writes.end()) {
					if (found->second.submission == submission) {
						if (book.shared.count(dword) == 0) {
							book.labels.insert(dword);
						}
					} else {
						book.shared.insert(dword);
						book.labels.erase(dword);
					}
				}
			}
			if (ahead && book.overlap) {
				book.ahead_read.insert(dword);
			}
		}
	}
	// Bounded: forgetting only makes later run-ahead ops promote.
	if (book.writes.size() > (1u << 18u)) {
		book.writes.clear();
	}
	if (book.labels.size() > (1u << 16u) || book.shared.size() > (1u << 16u)) {
		book.labels.clear();
		book.shared.clear();
	}
	CpPar::NoteBook(book.writes.size(), book.labels.size());
}

// Astro Bot PC (KYTY_CP_PAR=1b4, cpParallel.h): an op of the oldest submission no interleave point
// may precede while the run-ahead could still run (it may submit on purpose, flip, wait on a tick
// or need a service command; KYTY_CP_PAR_OLDEST_WRITES=hard: also its memory writes the
// run-ahead's draws may read at record time). Sequencer thread (emission).
static bool CpParHardForOldest(CpSeq::OpKind kind, const void* payload) {
	using CpSeq::OpKind;
	switch (kind) {
		case OpKind::Flip:
		case OpKind::WaitFlipDone:
		case OpKind::Predication:
		case OpKind::CondExec:
		case OpKind::Branch:
		case OpKind::LockstepRead:
		case OpKind::DrawIndirect:
		case OpKind::DrawIndirectMulti:
		case OpKind::DispatchIndirect:
		case OpKind::LodStats:
		case OpKind::DumpConstRam:
		case OpKind::Handoff: return true;
		case OpKind::EventWrite:
			return static_cast<const CpSeq::EventWriteOp*>(payload)->event_type == 0x39u; // occlusion
		case OpKind::EndOfPipe: {
			// A GDS read (SynchronizeGpu, or a deferred snapshot): 32-bit, GDS source.
			const auto& op = *static_cast<const CpSeq::EndOfPipeOp*>(payload);
			return op.size == sizeof(uint32_t) && op.event_write_source == 0x01u;
		}
		case OpKind::ReleaseMem: {
			const auto* body = static_cast<const CpSeq::ReleaseMemOp*>(payload)->body;
			return ((body[1] >> 29u) & 0x7u) == 5u; // DATA_SEL GDS
		}
		case OpKind::DmaData: {
			const auto& op = *static_cast<const CpSeq::DmaDataOp*>(payload);
			return op.dst_sel == 1u || op.src_sel == 1u || CpPar::OldestWritesHard();
		}
		case OpKind::WriteData:
		case OpKind::ReferenceClock: return CpPar::OldestWritesHard();
		default: return false;
	}
}

// Astro Bot PC (KYTY_CP_PAR=1b3): the op lock changed threads. The RenderExecutor's shadows of
// the command buffer describe the other context's stream; a draw-run continuation must not cross.
static void CpParNoteHolderChange(RenderContext& renderer) {
	renderer.GetRenderExecutor().CpParForgetCommandShadows();
	if (DrawRun::Enabled()) {
		DrawRun::NoteForeignActivity("cp-par context switch");
	}
}

struct DrawIndirectArgs {
	uint32_t vertex_count_per_instance;
	uint32_t instance_count;
	uint32_t start_vertex_location;
	uint32_t start_instance_location;
};

struct DrawIndexedIndirectArgs {
	uint32_t index_count_per_instance;
	uint32_t instance_count;
	uint32_t start_index_location;
	uint32_t base_vertex_location;
	uint32_t start_instance_location;
};

// Native indirect draws hand these records to the host GPU unchanged.
static_assert(sizeof(DrawIndirectArgs) == sizeof(vk::DrawIndirectCommand) &&
              offsetof(DrawIndirectArgs, instance_count) ==
                  offsetof(vk::DrawIndirectCommand, instanceCount) &&
              offsetof(DrawIndirectArgs, start_instance_location) ==
                  offsetof(vk::DrawIndirectCommand, firstInstance));
static_assert(sizeof(DrawIndexedIndirectArgs) == sizeof(vk::DrawIndexedIndirectCommand) &&
              offsetof(DrawIndexedIndirectArgs, instance_count) ==
                  offsetof(vk::DrawIndexedIndirectCommand, instanceCount) &&
              offsetof(DrawIndexedIndirectArgs, base_vertex_location) ==
                  offsetof(vk::DrawIndexedIndirectCommand, vertexOffset) &&
              offsetof(DrawIndexedIndirectArgs, start_instance_location) ==
                  offsetof(vk::DrawIndexedIndirectCommand, firstInstance));
constexpr uint64_t IndirectInstanceCountOffset = offsetof(DrawIndirectArgs, instance_count);
static_assert(offsetof(DrawIndexedIndirectArgs, instance_count) == IndirectInstanceCountOffset);

// KYTY_NATIVE_INDIRECT=0 keeps every indirect draw on CPU-read arguments.
static bool NativeIndirectEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_NATIVE_INDIRECT");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_INDIRECT_DRAW_INDEX (default on; =0 off): DRAW_(INDEX_)INDIRECT_MULTI with DRAW_INDEX_ENABLE
// has the CP write each record's index into the user SGPR at DRAW_INDEX_LOC before that record
// draws. Kyty ignored it, so every record saw the SGPR's previous value: Wolverine's title mesh
// draws (one record per object, the shader indexes a per-draw table with that SGPR) all drew the
// first table entry with the other records' meshlet counts, i.e. garbage triangles in front of
// the camera. The host indirect path has no per-record SGPR, so such multi-draws of more than one
// record take the CPU path (which reads the records).
static bool IndirectDrawIndexEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_INDIRECT_DRAW_INDEX");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_INDIRECT_VALIDATE=1 (diagnostic): also read native arguments on the CPU, which drains
// the GPU, and log where the native and CPU-read draws would differ.
static bool IndirectValidateEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_INDIRECT_VALIDATE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

static uint64_t IndexElementSize(uint32_t index_type_and_size) {
	switch (index_type_and_size) {
		case 0: return 2;
		case 1: return 4;
		case 2: return 1;
		default: EXIT("unknown index_type_and_size: %u\n", index_type_and_size);
	}
	return 0;
}

void ReadGuestForCp(uint64_t vaddr, uint64_t size, void* dst) {
	EXIT_IF(vaddr == 0 || dst == nullptr);
	if (LibKernel::Memory::TryReadGpuCleanBacking(vaddr, dst, size)) {
		return;
	}
	if (LibKernel::Memory::SynchronizeGpuBackingForRead(vaddr, size) &&
	    LibKernel::Memory::TryReadGpuCleanBacking(vaddr, dst, size)) {
		return;
	}
	std::memcpy(dst, reinterpret_cast<const void*>(vaddr), size);
}

class GpuMutexLock final {
public:
	explicit GpuMutexLock(Common::Mutex& mutex): m_mutex(mutex) {
		if (g_gpu_mutex_owned) {
			EXIT("recursive GPU mutex acquisition\n");
		}
		g_gpu_mutex_owned = true;
		m_mutex.Lock();
	}
	~GpuMutexLock() {
		if (!g_gpu_mutex_owned) {
			EXIT("invalid GPU mutex release\n");
		}
		m_mutex.Unlock();
		g_gpu_mutex_owned = false;
	}

private:
	Common::Mutex& m_mutex;
};

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::LogDirection::Silent;
}

// KYTY_CP_WAKEUPS=0 restores the old blocked-queue handling: sleep 100 us (about 1 ms with the
// Windows condition variable) and retry every blocked queue only after that timeout or when a
// queue completes. By default completed GPU work (every completion-runner operation), deferred
// label writes and flip completions wake the scheduler and unblock its queues at once, and
// before sleeping it spins for KYTY_CP_BLOCKED_SPIN_US (default 50) retrying blocked queues,
// which catches guest CPU writes (not observable otherwise) that follow shortly.
static bool CpWakeupsEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CP_WAKEUPS");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// Live switch (common/liveSwitch.h): read where the spin starts.
static Live::Switch g_cp_blocked_spin_us("KYTY_CP_BLOCKED_SPIN_US", [](const char* value) -> int64_t {
	const auto parsed = value != nullptr ? std::strtoul(value, nullptr, 10) : 50ul;
	return static_cast<int64_t>(std::min(parsed, 10000ul));
});

static uint64_t CpBlockedSpinNs() {
	return static_cast<uint64_t>(g_cp_blocked_spin_us.Get()) * 1000u;
}

static uint64_t CpNowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// cp.csv label rows (KYTY_HANG_TRACE_CP).
static void TraceCpLabel(const char* kind, const void* dst, uint64_t value, uint64_t size) {
	// KYTY_READ_FAULT_DIAG: label writes (not deferral notes) the guest may already see.
	if (ReadFaultDiag::Enabled() && std::strncmp(kind, "label-defer", 11) != 0) {
		ReadFaultDiag::NoteLabel(kind, reinterpret_cast<uint64_t>(dst), value);
	}
	if (HangTrace::CpTraceEnabled()) {
		HangTrace::CpEvent event;
		event.event   = kind;
		event.address = reinterpret_cast<uint64_t>(dst);
		event.value   = value;
		event.size    = size;
		HangTrace::RecordCp(event);
	}
}

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GraphicsInitJmpTables();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	if (CpSeq::ConfiguredMode() == CpSeq::Mode::Thread) {
		if (RepeatTrace::Enabled()) {
			std::printf("Kyty CP sequencer: off (KYTY_CP_REPEAT_TRACE needs the direct CP)\n");
		} else {
			// P3b (cpSequencer.h): the graphics queue's front runs on its own thread.
			m_sequencer.reset(new CpSeq::Sequencer(*m_gfx_cp));
			m_gfx_cp->AttachSequencer(m_sequencer.get());
			m_sequencer->Start();
			std::printf("Kyty CP sequencer: on (KYTY_CP_SEQ=1, verify=%d, %u snapshots of %zu "
			            "bytes)\n",
			            CpSeq::VerifyMode(), CpSeq::SnapshotCount(), sizeof(CpSeq::RegisterState));
		}
	}
	// Astro Bot PC (KYTY_CP_PAR=1a, cpParallel.h): CP resolver #2, which resolves context B's
	// submissions in turn with Thread_Gpu. The token's holder is the GPU thread: it owns the
	// renderer lock (RenderMutex, KYTY_RENDER_LOCK_ASYMMETRIC), and resolver #2 borrows
	// g_gpu_thread (GuestGpu::IsGpuThread) and g_gpu_state while it holds the token.
	if (CpPar::Enabled()) {
		if (m_sequencer == nullptr) {
			std::printf("Kyty CpPar: needs KYTY_CP_SEQ=1; every submission stays on Thread_Gpu\n");
		} else if (CpSeq::VerifyMode() != 0) {
			std::printf("Kyty CpPar: KYTY_CP_SEQ_VERIFY is on; every submission stays on Thread_Gpu\n");
		} else {
			if (CpPar::SeparateStreams()) {
				// Astro Bot PC (KYTY_CP_PAR=1b2, cpParallel.h): context B's own recording target.
				if (m_renderer.GetCommandScheduler().EnableContextB()) {
					CpPar::g_streams_active = true;
					// Astro Bot PC (KYTY_CP_PAR=1b3): the run-ahead instead of the token; the image
					// state hash bookkeeping is 1b2's only.
					CpPar::g_concurrent = CpPar::GetMode() == CpPar::Mode::Concurrent1b3 ||
					                      CpPar::GetMode() == CpPar::Mode::Concurrent1b4;
					// Astro Bot PC (KYTY_CP_PAR=1b4): the large run-ahead window (cpParallel.h).
					CpPar::g_concurrent = CpPar::g_concurrent ||
					                      CpPar::GetMode() == CpPar::Mode::Concurrent1b5;
					CpPar::g_wide      = CpPar::GetMode() == CpPar::Mode::Concurrent1b4 ||
					                CpPar::GetMode() == CpPar::Mode::Concurrent1b5;
					// Astro Bot PC (KYTY_CP_PAR=1b5): deferred interrupts (cpParallel.h).
					CpPar::g_deferred  = CpPar::GetMode() == CpPar::Mode::Concurrent1b5;
					CpPar::g_hash_book = !CpPar::g_concurrent;
					std::printf("Kyty CpPar: context B records into its own command buffer wrapper "
					            "and recorder ring (%s)\n",
					            CpPar::g_concurrent ? "segments per submission, replayed in admission order"
					                                : "segments at every token hand-off");
					if (CpPar::LanesMode()) { // Astro Bot PC (KYTY_CP_PAR=2a, 2b)
						// Astro Bot PC (KYTY_CP_PAR=2a, cpParallel.h): each lane's own native command
						// buffers (pool) and RenderExecutor, submitted at every hand-off.
						// N lanes (KYTY_CP_PAR_LANES): lane k = ranks k + 1, k + 1 + N, ... of a frame.
						const uint32_t lanes = CpPar::RequestedLanes();
						static_assert(CpPar::MaxLanes == CommandScheduler::MaxLanes &&
						              CpPar::MaxLanes == CommandRecorder::MaxStreams);
						if (m_renderer.LaneExecutors() + 1 >= lanes &&
						    m_renderer.GetCommandScheduler().EnableLanes(lanes)) {
							CpPar::g_lane_count = lanes;
							CpPar::g_lanes      = true;
							std::printf("Kyty CpPar: 2a lanes (%u): each lane has its own command pool, "
							            "native command buffers and RenderExecutor; a lane's buffer is "
							            "submitted at every hand-off (list order) unless nothing was "
							            "encoded in it (lazy begin)\n",
							            lanes);
							if (CpPar::GetMode() == CpPar::Mode::Lanes2b) {
								// Astro Bot PC (KYTY_CP_PAR=2b, cpParallel.h): each lane's own replay
								// thread; buffers handed to the queue in list order.
								if (m_renderer.GetCommandScheduler().EnableLanes2b()) {
									CpPar::g_lanes_2b = true;
									(void)CpPar::LaneLockEnabled(); // step 3b-2: read once, printed
									std::printf("Kyty CpPar: 2b: each lane has its own replay thread; "
									            "native buffers are handed to the queue in list "
									            "order\n");
								} else {
									std::printf("Kyty CpPar: 2b unavailable (KYTY_GPU_TIMING, GPU op "
									            "profiler on); running as "
									            "2a\n");
								}
							}
						} else {
							std::printf("Kyty CpPar: 2a lanes unavailable; running as 1b2\n");
						}
					}
				} else {
					std::printf("Kyty CpPar: KYTY_CP_PAR=%s needs KYTY_CP_RECORDER=1 without "
					            "KYTY_CP_RECORDER_VERIFY; running as 1b1\n",
					            CpPar::GetMode() == CpPar::Mode::Concurrent1b3   ? "1b3"
					            : CpPar::GetMode() == CpPar::Mode::Concurrent1b4 ? "1b4" // Astro Bot PC
					            : CpPar::GetMode() == CpPar::Mode::Concurrent1b5 ? "1b5"
					            : CpPar::GetMode() == CpPar::Mode::Lanes2a       ? "2a" // Astro Bot PC
					            : CpPar::GetMode() == CpPar::Mode::Lanes2b       ? "2b"
					                                                             : "1b2");
				}
			}
			if (CpPar::Concurrent()) {
				// Astro Bot PC (KYTY_CP_PAR=1b3, cpParallel.h).
				(void)CpPar::VerifyEnabled();
				if (CpPar::Wide()) {
					(void)CpPar::WritesAll(); // Astro Bot PC (KYTY_CP_PAR=1b4): read once, printed
					(void)CpPar::OldestWritesHard();
					(void)CpPar::InterleaveNext();
				}
				CpPar::SetInsertHook(
				    [](void* context, uint32_t trigger) {
					    static_cast<GuestGpu*>(context)->CpParInsert(trigger);
				    },
				    this);
				CpPar::StartRunAheadThread();
			}
		}
	}
	if (CpPar::Enabled() && m_sequencer != nullptr && CpSeq::VerifyMode() == 0 &&
	    !CpPar::Concurrent()) {
		{
			CpPar::Hooks hooks;
			hooks.context = this;
			hooks.take    = [](void* context) {
				auto* gpu = static_cast<GuestGpu*>(context);
				if (CpPar::OnResolver2()) {
					g_gpu_thread = true;
					g_gpu_state  = gpu;
				}
				gpu->m_renderer.GetMutex().BecomeOwner();
				if (CpPar::StreamsActive()) {
					// Astro Bot PC (KYTY_CP_PAR=1b2): this thread's context begins a segment (2a: in
					// a new native buffer of its own pool; GetRenderExecutor is then the lane's).
					const uint32_t stream = CpPar::ThreadLane(); // Astro Bot PC (2a: N lanes)
					CommandRecorder::SetThreadStream(stream);
					if (CpPar::Lanes2b()) {
						CommandScheduler::SetThreadTickLane(stream); // Astro Bot PC (2b: tick ranges)
					}
					gpu->m_renderer.GetCommandScheduler().BeginContextSegment();
					gpu->m_renderer.GetRenderExecutor().CpParForgetCommandShadows();
					CpPar::BeginSegment(stream);
				}
			};
			hooks.give = [](void* context) {
				auto* gpu = static_cast<GuestGpu*>(context);
				if (CpPar::StreamsActive()) {
					// Astro Bot PC (KYTY_CP_PAR=1b2): this thread's context ends its segment (still
					// owning the renderer), the merge point of its image states.
					const uint32_t stream = CpPar::ThreadLane(); // Astro Bot PC (2a: N lanes)
					CommandRecorder::SetThreadStream(stream);
					const auto counts = gpu->m_renderer.GetCommandScheduler().EndContextSegment();
					CpPar::EndSegment(stream, counts.packets, counts.submits,
					                  counts.new_command_buffer);
					if (counts.lane_submitted || counts.lane_skipped) {
						// Astro Bot PC (KYTY_CP_PAR=2a): the lane's buffer went to the queue, or back
						// to its pool (lazy begin: nothing encoded).
						CpPar::NoteLaneHandoff(stream, counts.lane_empty, counts.lane_skipped);
					}
				}
				gpu->m_renderer.GetMutex().ResignOwner();
				if (CpPar::OnResolver2()) {
					g_gpu_thread = false;
					g_gpu_state  = nullptr;
				}
			};
			CpPar::Start(hooks);
		}
	}
	if (CpWakeupsEnabled()) {
		m_renderer.GetCommandScheduler().SetProgressHook(
		    [](void* context) { static_cast<GuestGpu*>(context)->NotifyProgress(); }, this);
	}
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	// Astro Bot PC (KYTY_CP_PAR): Thread_Gpu returned, resolver #2 is idle.
	CpPar::Stop();
	if (m_sequencer != nullptr) {
		// The GPU thread executed every admitted submission's ops; the sequencer is idle.
		m_sequencer->Stop();
	}
	// No completion-runner call may reach this object once it is destroyed.
	m_renderer.GetCommandScheduler().SetProgressHook(nullptr, nullptr);
	m_renderer.GetCommandScheduler().DrainPriorityOperations();
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread());
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		{
			const CpGaps::Scope gap(CpGaps::Cat::Service);
			command();
		}
		// Service commands (mapping changes, readbacks, deferred label writes) change guest
		// memory outside the command stream (syncEpoch.h).
		SyncEpoch::Advance();
		// KYTY_DRAW_RUN: other command-processor work (drawPrep/drawRun.h).
		if (DrawRun::Enabled()) {
			DrawRun::NoteForeignActivity("service command");
		}
	}
}

// KYTY_SYNC_COMMAND_SPIN_US (default 0: sleep at once, as before; live): GuestGpu::SendCommandSync.
static Live::Switch g_sync_command_spin_us("KYTY_SYNC_COMMAND_SPIN_US", [](const char* value) -> int64_t {
	const auto parsed = value != nullptr ? std::strtoul(value, nullptr, 10) : 0ul;
	return static_cast<int64_t>(std::min(parsed, 10000ul));
});

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	const auto spin_us = static_cast<uint64_t>(g_sync_command_spin_us.Get());
	if (spin_us != 0) {
		// KYTY_SYNC_COMMAND_SPIN_US: the caller spins that long (about what such a command waits
		// for its packet boundary and runs) before it sleeps, and the GPU thread wakes it only when
		// it sleeps: a guest readback's begin was a sleep, then a wake-up system call on the GPU
		// thread (chenxiao07/KytyPS5 59aa6dc79). States: 0 running, 1 done, 2 the caller sleeps.
		std::atomic<uint32_t> state {0};
		SendCommand([operation = std::move(command), &state]() mutable {
			operation();
			if (state.exchange(1, std::memory_order_acq_rel) == 2) {
				// WakeByAddress only names the address: harmless if the caller already returned.
				state.notify_one();
			}
		});
		HangWatchdog::Scope wait("cp-service-command", reinterpret_cast<uint64_t>(this),
		                         reinterpret_cast<uint64_t>(&state));
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(spin_us);
		uint32_t   spins    = 0;
		while (state.load(std::memory_order_acquire) != 1) {
			if ((++spins & 63u) == 0 && std::chrono::steady_clock::now() >= deadline) {
				uint32_t expected = 0;
				if (state.compare_exchange_strong(expected, 2, std::memory_order_acq_rel)) {
					while (state.load(std::memory_order_acquire) == 2) {
						state.wait(2, std::memory_order_acquire);
					}
				}
				break;
			}
			_mm_pause();
		}
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	HangWatchdog::Scope wait("cp-service-command", reinterpret_cast<uint64_t>(this),
	                         reinterpret_cast<uint64_t>(&done));
	done.acquire();
}

bool GuestGpu::TrySendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	Common::LockGuard lock(m_queue_mutex);
	if (!m_accepting) {
		return false;
	}
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
	return true;
}

void GuestGpu::NotifyProgress() {
	// Cheap when nothing is blocked (called after every completion-runner operation).
	if (!m_has_blocked.exchange(false, std::memory_order_acq_rel)) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpProgressWakeups);
	Common::LockGuard lock(m_queue_mutex);
	for (auto& queue: m_queues) {
		if (!queue.empty()) {
			queue.front().blocked = false;
		}
	}
	m_work_available.Signal();
}

bool GuestGpu::HasRunnableComputeWork() {
	Common::LockGuard lock(m_queue_mutex);
	for (uint32_t id = 1; id < QueueCount; id++) {
		if (!m_queues[id].empty() && !m_queues[id].front().blocked &&
		    FrameFencePassed(m_queues[id].front())) {
			return true;
		}
	}
	return false;
}

// KYTY_FRAME_FENCE=0 lets a submission admitted after sceAgcSuspendPoint (bounded Done) start
// while submissions admitted before that Done are still pending on other queues. By default the
// CP scheduler holds it until they have completed, which keeps the cross-queue frame order the
// idle Done gave (only the scheduler waits, the guest thread does not). Without it, Astro Bot's
// async-compute copy into memory that the previous frame's water draw still samples ran between
// that frame's refraction copy and water draw once graphics slices could yield
// (KYTY_GFX_SLICE_DRAWS): white or opaque water.
static bool FrameFenceEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_FRAME_FENCE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// A front held this long by the fence runs anyway (logged once): a hang would be worse.
constexpr uint64_t FrameFenceTimeoutNs = 2'000'000'000;

bool GuestGpu::FrameFencePassed(const Submission& submission) const {
	// m_in_flight holds the submission itself (a larger sequence), so its first element exceeds
	// the fence exactly when every submission admitted before that Done has completed. A started
	// submission (blocked or yielded since) passed the fence already.
	return submission.frame_fence == 0 || submission.started || m_in_flight.empty() ||
	       *m_in_flight.begin() > submission.frame_fence ||
	       (submission.fence_hold_ns != 0 &&
	        CpNowNs() - submission.fence_hold_ns > FrameFenceTimeoutNs);
}

void GuestGpu::AddDeferredLabel(uint64_t address, uint32_t size, uint64_t tick, uint64_t value,
                                bool cp_visible) {
	Common::LockGuard lock(m_queue_mutex);
	m_deferred_labels.push_back({address, size, tick, value, cp_visible && size <= sizeof(value)});
	if (cp_visible) {
		m_cp_visible_labels.store(true, std::memory_order_relaxed);
	}
	HangWatchdog::NoteEvent(address, "deferred-label", tick, INT16_MIN, false, 0, size);
	m_deferred_label_count.store(static_cast<uint32_t>(m_deferred_labels.size()),
	                             std::memory_order_release);
}

void GuestGpu::RemoveDeferredLabel(uint64_t address, uint64_t tick) {
	Common::LockGuard lock(m_queue_mutex);
	const auto found = std::find_if(m_deferred_labels.begin(), m_deferred_labels.end(),
	                                [address, tick](const DeferredLabel& label) {
		                                return label.address == address && label.tick == tick;
	                                });
	EXIT_IF(found == m_deferred_labels.end());
	m_deferred_labels.erase(found);
	HangWatchdog::NoteEvent(address, "deferred-label", tick, INT16_MIN, false, 0, 0, true);
	m_deferred_label_count.store(static_cast<uint32_t>(m_deferred_labels.size()),
	                             std::memory_order_release);
	// A queue suspended on this label (WAIT_REG_MEM) can make progress now.
	for (auto& queue: m_queues) {
		if (!queue.empty()) {
			queue.front().blocked = false;
		}
	}
	m_work_available.Signal();
}

uint64_t GuestGpu::DeferredLabelTick(uint64_t address, uint64_t size) {
	if (!HasDeferredLabels()) {
		return 0;
	}
	Common::LockGuard lock(m_queue_mutex);
	uint64_t          tick = 0;
	for (const auto& label: m_deferred_labels) {
		if (label.address < address + size && address < label.address + label.size) {
			tick = std::max(tick, label.tick);
		}
	}
	return tick;
}

bool GuestGpu::DeferredLabelSuperseded(uint64_t address, uint64_t tick) {
	if (!m_cp_visible_labels.load(std::memory_order_relaxed)) {
		return false;
	}
	Common::LockGuard lock(m_queue_mutex);
	for (const auto& label: m_deferred_labels) {
		if (label.address == address && label.tick == tick) {
			return label.superseded;
		}
	}
	return false;
}

void GuestGpu::DropCpVisibleLabels(uint64_t address, uint64_t size) {
	if (!m_cp_visible_labels.load(std::memory_order_relaxed) || !HasDeferredLabels()) {
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	for (auto& label: m_deferred_labels) {
		if (label.cp_visible && label.address < address + size &&
		    address < label.address + label.size) {
			label.cp_visible = false;
			// Hardware wrote the label before the CP passed the wait that preceded this write.
			label.superseded = label.cp_waited;
			static std::atomic<uint64_t> dropped {0};
			const auto count = dropped.fetch_add(1, std::memory_order_relaxed) + 1;
			if (count <= 16 || (count & 0xfffu) == 0) {
				std::printf("Kyty end-of-pipe labels: %" PRIu64 " pending labels replaced by a later CP "
				            "write (0x%" PRIx64 ", %s)\n",
				            count, label.address, label.superseded ? "skipped" : "kept");
			}
		}
	}
}

bool GuestGpu::CpVisibleLabelOverlaps(uint64_t address, uint64_t size) {
	if (!m_cp_visible_labels.load(std::memory_order_relaxed) || !HasDeferredLabels()) {
		return false;
	}
	Common::LockGuard lock(m_queue_mutex);
	for (const auto& label: m_deferred_labels) {
		if (label.cp_visible && label.address < address + size &&
		    address < label.address + label.size) {
			return true;
		}
	}
	return false;
}

void GuestGpu::NoteCpVisibleLabelsWaited(uint64_t address, uint64_t size) {
	Common::LockGuard lock(m_queue_mutex);
	for (auto& label: m_deferred_labels) {
		if (label.cp_visible && label.address < address + size &&
		    address < label.address + label.size) {
			label.cp_waited = true;
		}
	}
}

bool GuestGpu::CpVisibleLabelValue(uint64_t address, uint32_t size, void* value) {
	if (!m_cp_visible_labels.load(std::memory_order_relaxed) || !HasDeferredLabels()) {
		return false;
	}
	Common::LockGuard lock(m_queue_mutex);
	bool              found = false;
	// Oldest first: a newer label's bytes replace an older one's.
	for (const auto& label: m_deferred_labels) {
		if (!label.cp_visible || label.address >= address + size ||
		    address >= label.address + label.size) {
			continue;
		}
		const auto begin = std::max(address, label.address);
		const auto end   = std::min(address + size, label.address + label.size);
		std::memcpy(static_cast<uint8_t*>(value) + (begin - address),
		            reinterpret_cast<const uint8_t*>(&label.value) + (begin - label.address),
		            end - begin);
		found = true;
	}
	return found;
}

void GuestGpu::Submit(std::span<const uint32_t> draw_commands,
                      std::span<const uint32_t> constant_commands) {
	if (draw_commands.empty()) {
		return;
	}
	GpuMutexLock lock(m_submission_mutex);
	Submission   submission;
	submission.type              = SubmissionType::Graphics;
	submission.queue_id          = 0;
	submission.commands          = draw_commands;
	submission.constant_commands = constant_commands;
	submission.reset_processor   = m_graphics_done;
	m_graphics_done              = false;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());
	GpuMutexLock lock(m_submission_mutex);

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type     = SubmissionType::Compute;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	GpuMutexLock lock(m_submission_mutex);
	Submission   submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.reset_processor = m_graphics_done;
	submission.flip_request_id = request_id;
	m_graphics_done            = false;
	Enqueue(std::move(submission));
}

// KYTY_AGC_DONE_MODE: "idle" makes sceAgcSuspendPoint (GuestGpu::Done) hold the submission lock
// and wait until the CP has consumed everything and has nothing queued (the previous behaviour;
// 105-145 ms per frame in Sky Garden, serializing guest frame N+1 building with CP frame N and
// blocking other threads' submissions). "bounded" (default) records the frame boundary (the
// processor-reset flag for the next graphics submission and the frame number) under the lock,
// releases it, and waits only until the CP has completed every submission admitted before the
// previous Done: at most two guest frames in flight on the CP.
static bool DoneBounded() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_AGC_DONE_MODE");
		const bool  on    = value == nullptr || std::strcmp(value, "idle") != 0;
		std::printf("Kyty AgcSuspendPoint: %s (KYTY_AGC_DONE_MODE)\n",
		            on ? "bounded (two frames in flight)" : "wait for idle");
		return on;
	}();
	return enabled;
}

void GuestGpu::Done() {
	if (!DoneBounded()) {
		GpuMutexLock lock(m_submission_mutex);
		if (!IsGpuThread()) {
			const auto wait_start = HangTrace::Enabled() ? HangTrace::NowNs() : 0;
			Profiler::ScopedFrameWait frame_wait(Profiler::FrameWait::AgcDoneWait);
			WaitForIdle();
			if (HangTrace::Enabled()) {
				HangTrace::RecordDoneWait(HangTrace::NowNs() - wait_start);
			}
		}
		m_graphics_done = true;
		m_done_num++;
		return;
	}
	uint64_t target = 0;
	{
		GpuMutexLock lock(m_submission_mutex);
		// Admission order is what defines the frame boundary: every later graphics submission
		// resets the processor first, exactly as after an idle wait.
		m_graphics_done = true;
		m_done_num++;
		Common::LockGuard queue_lock(m_queue_mutex);
		target              = m_done_boundary;
		m_done_boundary     = m_next_submission_sequence - 1;
	}
	if (IsGpuThread() || target == 0) {
		return;
	}
	HangWatchdog::Scope blocked("agc-done-prefix", reinterpret_cast<uint64_t>(this), target);
	const auto wait_start = HangTrace::Enabled() ? HangTrace::NowNs() : 0;
	const auto pace_start = PaceLog::Enabled() ? PaceLog::NowNs() : 0;
	{
		Profiler::ScopedFrameWait frame_wait(Profiler::FrameWait::AgcDoneWait);
		Common::LockGuard         lock(m_queue_mutex);
		auto                      prefix_done = [this, target] {
			return m_stopping || m_in_flight.empty() || *m_in_flight.begin() > target;
		};
		if (!prefix_done()) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::AgcDoneBoundedWaits);
			++m_done_waiters;
			while (!prefix_done()) {
				blocked.Observed(m_in_flight.empty() ? 0 : *m_in_flight.begin());
				m_done_progress.Wait(&m_queue_mutex);
			}
			--m_done_waiters;
		}
	}
	if (HangTrace::Enabled()) {
		HangTrace::RecordDoneWait(HangTrace::NowNs() - wait_start);
	}
	if (pace_start != 0) {
		PaceLog::NoteDone(pace_start, PaceLog::NowNs() - pace_start);
	}
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

void GuestGpu::SequencerDeleter::operator()(CpSeq::Sequencer* sequencer) const noexcept {
	delete sequencer;
}

CommandProcessor::CommandProcessor(RenderContext& renderer, int interrupt_event_id)
    : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id),
      m_front_mode(CpSeq::ConfiguredMode() == CpSeq::Mode::Inline ? FrontMode::Inline
                                                                  : FrontMode::Direct) {}

CommandProcessor::~CommandProcessor() = default;

void CommandProcessor::OpStreamDeleter::operator()(CpSeq::OpStream* stream) const noexcept {
	delete stream;
}

void CommandProcessor::VerifierDeleter::operator()(CpSeq::Verifier* verifier) const noexcept {
	delete verifier;
}

void CommandProcessor::DrawPrepDeleter::operator()(DrawPrep::Engine* engine) const noexcept {
	delete engine;
}

DrawPrep::Engine* CommandProcessor::DrawPrepEngine() {
	// Only the graphics processor draws; compute queues would only dilute the S0 histogram.
	if (!DrawPrep::PacketHookActive() || IsAsyncComputeQueue()) {
		return nullptr;
	}
	if (m_draw_prep == nullptr) {
		m_draw_prep.reset(new DrawPrep::Engine(
		    m_renderer,
		    [] {
			    // Astro Bot PC (KYTY_CP_PAR=1b3): not while the run-ahead is dirty (cpParallel.h).
			    if (g_gpu_state != nullptr && !(CpPar::Concurrent() && CpPar::RunAheadDirty())) {
				    g_gpu_state->ProcessCommands();
			    }
		    },
		    // After each committed (recorded) draw, with the live registers bound again: the
		    // idle-GPU early submit counts recorded draws, exactly as after a serial draw.
		    [this] { MaybeFlushIdleGpu(); }));
	}
	return m_draw_prep.get();
}

// Astro Bot PC (KYTY_CP_PAR=1b1, cpParallel.h): each resolver context's draw-prep engine and
// snapshot ring. m_emit_b / m_resolve_b are only ever set in 1b1 mode, and the B objects only
// exist then: otherwise these are m_draw_prep and the sequencer's one snapshot ring.
DrawPrep::Engine* CommandProcessor::EmitPrep() const {
	// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): the context's engine (B, C or D).
	auto* engine = m_emit_lane != 0 ? LanePrep(m_emit_lane) : nullptr;
	return engine != nullptr ? engine : m_draw_prep.get();
}

DrawPrep::Engine* CommandProcessor::ResolvePrep() const {
	// Astro Bot PC (KYTY_CP_PAR=1b3): the calling thread's context (two resolvers at once).
	const uint32_t lane   = CpPar::Concurrent() ? (t_cp_par_resolve_b ? 1u : 0u) : ResolveLaneIndex();
	auto*          engine = lane != 0 ? LanePrep(lane) : nullptr;
	return engine != nullptr ? engine : m_draw_prep.get();
}

CpSeq::SnapshotRing& CommandProcessor::EmitSnapshots() const {
	return m_sequencer->Snapshots(m_emit_lane);
}

CpSeq::SnapshotRing& CommandProcessor::ResolveSnapshots() const {
	// Astro Bot PC (1b3: the calling thread's context; 2a: N lanes).
	const uint32_t lane = CpPar::Concurrent() ? (t_cp_par_resolve_b ? 1u : 0u) : ResolveLaneIndex();
	return m_sequencer->Snapshots(lane);
}

void CommandProcessor::CpParWindowCounts(uint32_t context, uint64_t& published,
                                         uint64_t& retired) const {
	const auto* engine = context != 0 ? LanePrep(context) : m_draw_prep.get();
	published          = 0;
	retired            = 0;
	if (engine != nullptr) {
		engine->WindowCounts(published, retired);
	}
}

bool CommandProcessor::TrySubmitPreparedDraw(const DrawIndexArgs* index_args,
                                             const DrawAutoArgs*  auto_args) {
	if (DrawPrep::GetMode() == DrawPrep::Mode::Off) {
		return false;
	}
	auto* engine = DrawPrepEngine();
	if (engine == nullptr) {
		return false;
	}
	if (m_front_mode == FrontMode::Thread) {
		// The resolver: a draw the sequencer did not publish (indirect draws' CPU path, or draw
		// prep inline). The parallel window's producer is the sequencer: such draws are drawn
		// serially. Inline preparation uses the op's registers, which the command buffer binds.
		if (engine->Parallel()) {
			return false;
		}
		auto& command = CurrentBuffer();
		return engine->Submit(LaneCp().submit_id, index_args, auto_args, command.GetRegisters(),
		                      command.GetUserConfig(), command.GetShaders());
	}
	return engine->Submit(LaneCp().submit_id, index_args, auto_args, m_ctx, m_ucfg, m_sh_ctx);
}

void CommandProcessor::DrainPreparedDraws() {
	// KYTY_CP_SEQ=1: every published draw belongs to an op the resolver has not reached yet (the
	// earlier ones are committed); each is committed by its own op.
	if (m_front_mode == FrontMode::Thread) {
		return;
	}
	if (m_draw_prep != nullptr) {
		m_draw_prep->Drain();
	}
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_user_data_marker                 = HW::UserSgprType::Unknown;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;

	std::memset(m_const_ram, 0, sizeof(m_const_ram));
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_context_state_pushed);
			m_ctx                  = m_saved_ctx;
			m_saved_ctx            = {};
			m_context_state_pushed = false;
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			m_ctx.Reset();
			break;
		default: EXIT("unknown context state operation: %u\n", static_cast<uint32_t>(operation));
	}
}

void CommandProcessor::BufferInit() {
	if (m_front_mode == FrontMode::Thread) {
		// The resolver: the live registers belong to the sequencer. Ops that read registers bind
		// their snapshot (or draw-prep slot); the others read none (P3-SEQUENCER.md 9.3).
		GetScheduler().Begin(m_back_ctx, m_back_ucfg, m_back_sh);
		return;
	}
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
}

// KYTY_SUBMIT_MIN_INTERVAL_US (default 0 = off): the optional submits of the command processors
// (idle flushes, MaybeFlushIdleGpu; end-of-pipe interrupt batches, BufferFlushForEop, and their
// packet bound) happen only once this long has passed since the last submit of any of them; until
// then the recording continues, and the next submit takes the work along (a later request past
// the interval, a slice end, a wait, a flip, a readback: every other flush is unchanged). Fewer
// command buffers cost the command processor fewer ends, begins and state re-emissions; the GPU
// gets the same work in larger pieces. BryanKAdams' fork batches label submits the same way (2 ms).
static uint64_t SubmitMinIntervalNs() {
	static const uint64_t ns = [] {
		const char* value  = std::getenv("KYTY_SUBMIT_MIN_INTERVAL_US");
		const auto  parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
		return std::min<uint64_t>(parsed, 1'000'000ull) * 1000u;
	}();
	return ns;
}
// GPU thread (every command processor runs there): the last submit, with the interval on.
static uint64_t g_last_submit_ns = 0;

static bool OptionalSubmitAllowed() {
	if (CpPar::Concurrent() && CpPar::RunAheadDirty()) {
		return false; // Astro Bot PC (KYTY_CP_PAR=1b3): deferred while the run-ahead is dirty
	}
	const auto interval = SubmitMinIntervalNs();
	if (interval == 0 || CpNowNs() - g_last_submit_ns >= interval) {
		return true;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::SubmitIntervalDeferrals);
	return false;
}

void CommandProcessor::BufferFlush() {
	KYTY_PROFILER_DETAIL_FUNCTION();
	LaneCp().deferred_eop_flushes     = 0;
	LaneCp().packets_since_eop_request = 0;
	if (SubmitMinIntervalNs() != 0) {
		g_last_submit_ns = CpNowNs();
	}
	// Between packets: append eager readback copies of read-hot pages to this recording
	// (KYTY_READBACK_EAGER), published when it completes.
	m_renderer.GetBufferCache().IssueEagerReadbacks();
	GetScheduler().Flush();
}

// KYTY_EOP_FLUSH_PACKETS: packets processed after the first deferred interrupt before the command
// buffer is flushed anyway (default 256). Live switch (common/liveSwitch.h), as are
// KYTY_EOP_FLUSH_BATCH, KYTY_IDLE_FLUSH_DRAWS and KYTY_GFX_SLICE_DRAWS below: thresholds the
// command processor compares its counters with at every use.
static Live::Switch g_eop_flush_packets("KYTY_EOP_FLUSH_PACKETS", [](const char* value) -> int64_t {
	const auto parsed = value != nullptr ? std::strtoul(value, nullptr, 10) : 256ul;
	return static_cast<int64_t>(std::clamp(parsed, 1ul, 1ul << 20u));
});

// KYTY_GPU_TIMESTAMP_SCALE (gpuTimestampScale.h): guest GPU clock values stretched since the last
// guest flip. The anchor is the reference clock when the CP executed the last flip op.
static Live::Switch          g_gpu_timestamp_scale("KYTY_GPU_TIMESTAMP_SCALE",
                                                   GpuTimestampScale::ParsePercent);
static std::atomic<uint64_t> g_gpu_timestamp_anchor {0};

static uint64_t GuestGpuTimestamp() {
	const auto now     = Sync::ReadReferenceClock();
	const auto percent = g_gpu_timestamp_scale.Get();
	if (percent == GpuTimestampScale::Off) {
		return now;
	}
	static std::atomic_bool reported {false};
	if (!reported.exchange(true, std::memory_order_relaxed)) {
		std::printf("Kyty GPU timestamps: stretched to %lld%% since each flip "
		            "(KYTY_GPU_TIMESTAMP_SCALE)\n",
		            static_cast<long long>(percent));
	}
	return GpuTimestampScale::Stretch(now, g_gpu_timestamp_anchor.load(std::memory_order_relaxed),
	                                  percent);
}

static Live::Switch g_eop_flush_batch("KYTY_EOP_FLUSH_BATCH", [](const char* value) -> int64_t {
	const auto parsed = value != nullptr ? std::strtoul(value, nullptr, 10) : 8ul;
	return static_cast<int64_t>(std::clamp(parsed, 1ul, 1024ul));
});

uint32_t CommandProcessor::EopFlushPacketLimit() {
	return static_cast<uint32_t>(g_eop_flush_packets.Get());
}

void CommandProcessor::BufferFlushForEop() {
	if (++LaneCp().deferred_eop_flushes >= static_cast<uint32_t>(g_eop_flush_batch.Get()) && OptionalSubmitAllowed()) {
		BufferFlush();
	}
}

// KYTY_IDLE_FLUSH_DRAWS (default 8, 0 disables): after at least that many draws/dispatches in
// the current recording, submit it early when the GPU has finished everything already submitted
// (KnownGpuTick >= CurrentTick - 1), e.g. after a drain, instead of leaving the GPU idle until
// the next natural boundary. Inside an active rendering instance it waits for 4x the draws (or
// the instance's end) to avoid splitting render passes.
static Live::Switch g_idle_flush_draws("KYTY_IDLE_FLUSH_DRAWS", [](const char* value) -> int64_t {
	const auto parsed = value != nullptr ? std::strtoul(value, nullptr, 10) : 8ul;
	return static_cast<int64_t>(std::min(parsed, 65536ul));
});

static uint32_t IdleFlushMinDraws() {
	return static_cast<uint32_t>(g_idle_flush_draws.Get());
}

void CommandProcessor::MaybeFlushIdleGpu() {
	if (CpPar::Concurrent() && (CpPar::OnResolver2() || CpPar::RunAheadDirty())) {
		// Astro Bot PC (KYTY_CP_PAR=1b3): no optional submit by the run-ahead, nor while it is
		// dirty (cpParallel.h); the request stays pending.
		return;
	}
	// A draw or dispatch just recorded writes a page the command processor reads back later (e.g.
	// indirect arguments): submit it now, outside a rendering instance, so that read waits for
	// this producer at most, not for everything recorded until then (KYTY_READBACK_EAGER).
	if (m_renderer.GetBufferCache().TakeEagerFlushRequest(CurrentBuffer().ActiveRenderingSerial() !=
	                                                     0)) {
		BufferFlush();
		return;
	}
	const auto min_draws = IdleFlushMinDraws();
	if (min_draws == 0) {
		return;
	}
	auto&      scheduler = GetScheduler();
	const auto current   = scheduler.CurrentTick();
	if (current != LaneCp().idle_flush_tick) {
		// Something else submitted since the last count: restart the bound.
		LaneCp().idle_flush_tick  = current;
		LaneCp().idle_flush_draws = 0;
	}
	if (++LaneCp().idle_flush_draws < min_draws) {
		return;
	}
	const bool in_pass = CurrentBuffer().ActiveRenderingSerial() != 0;
	if (in_pass && LaneCp().idle_flush_draws < min_draws * 4u) {
		return;
	}
	auto& master = scheduler.GetMasterSemaphore();
	if (master.KnownGpuTick() + 1u < current) {
		// Refresh the timeline value only every few draws (a driver query).
		if ((LaneCp().idle_flush_draws % 4u) != 0) {
			return;
		}
		master.Refresh();
		if (master.KnownGpuTick() + 1u < current) {
			return;
		}
	}
	if (!OptionalSubmitAllowed()) {
		return;
	}
	Profiler::CountFrameEvent(in_pass ? Profiler::FrameEvent::IdleFlushesInPass
	                                  : Profiler::FrameEvent::IdleFlushes);
	BufferFlush();
}

// KYTY_GFX_SLICE_DRAWS (default 128, 0 disables): one GPU thread runs the graphics queue and all
// async compute queues round-robin, and a graphics slice used to run until it completed or
// blocked (up to ~100 ms of CP time per frame), so compute submissions waited that long. Every
// that many draws the graphics CP checks whether another queue has runnable (not suspended)
// work and, only then, ends its slice after the current packet (a slice end flushes).
static Live::Switch g_gfx_slice_draws("KYTY_GFX_SLICE_DRAWS", [](const char* value) -> int64_t {
	const auto parsed = value != nullptr ? std::strtoul(value, nullptr, 10) : 128ul;
	return static_cast<int64_t>(std::min(parsed, 1000000ul));
});

static uint32_t GfxSliceDraws() {
	return static_cast<uint32_t>(g_gfx_slice_draws.Get());
}

void CommandProcessor::MaybeYieldSlice() {
	if (CpPar::Concurrent() && CpPar::RunAheadDirty()) {
		return; // Astro Bot PC (KYTY_CP_PAR=1b3): no slice end while the run-ahead is dirty
	}
	const auto limit = GfxSliceDraws();
	if (limit == 0 || IsAsyncComputeQueue() || g_current_execution == nullptr ||
	    g_gpu_state == nullptr || g_current_processor != this) {
		return;
	}
	if (++LaneCp().slice_draws < limit || (LaneCp().slice_draws % limit) != 0) {
		return;
	}
	if (!g_gpu_state->HasRunnableComputeWork()) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::GfxSliceYields);
	g_current_execution->m_yield = true;
}

void CommandProcessor::BufferFlushAndWait() {
	KYTY_PROFILER_DETAIL_FUNCTION();
	GetScheduler().FlushAndWait();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish();
}

void CommandProcessor::ResetDeCe() {
	m_de_count    = 0;
	m_ce_count    = 0;
	m_ce_complete = false;
}

void CommandProcessor::WaitCe() {
	if (m_ce_count <= m_de_count && !m_ce_complete) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitDeDiff(uint32_t diff) {
	EXIT_IF(m_de_count > m_ce_count);
	if (m_ce_count - m_de_count >= diff) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4();
	}
}

void CommandProcessor::IncrementDe() {
	m_de_count++;
}

void CommandProcessor::IncrementCe() {
	m_ce_count++;
}

void CommandProcessor::WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num) {
	memcpy(m_const_ram + offset / 4, src, static_cast<size_t>(dw_num) * 4);
}

void CommandProcessor::DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num) {
	CpSeq::DumpConstRamOp op;
	op.dst    = reinterpret_cast<uint64_t>(dst);
	op.offset = offset;
	op.dw_num = dw_num;
	// The front's constant RAM travels with the op (the direct path reads it in place).
	(void)Submit(op, m_const_ram + offset / 4, dw_num * static_cast<uint32_t>(sizeof(uint32_t)));
}

void CommandProcessor::ExecDumpConstRam(const CpSeq::DumpConstRamOp& op, const uint32_t* src) {
	StreamCopyDefer::NoteGpuOnlyWrite(op.dst, uint64_t {op.dw_num} * 4u);
	memcpy(reinterpret_cast<uint32_t*>(op.dst), src, static_cast<size_t>(op.dw_num) * 4);
	NoteCpWrite(op.dst, uint64_t {op.dw_num} * 4u);
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}
	CpSeq::WaitRegMemOp op;
	op.addr    = reinterpret_cast<uint64_t>(addr);
	op.ref     = static_cast<uint64_t>(ref);
	op.mask    = static_cast<uint64_t>(mask);
	op.func    = func;
	op.poll    = poll;
	op.wait_op = wait_op;
	op.size    = static_cast<uint32_t>(sizeof(T));
	if (Submit(op).suspended) {
		SuspendPm4();
	}
}

// Diagnostic (astrocrash): KYTY_CP_WRITE_LOG=1 prints each new (kind, address) once, at most 600:
// command-processor writes to guest memory done by the CPU at record time (write-data, dma) and
// WAIT_REG_MEM addresses (passed through a split-mode label, or suspended).
static void LogCpAddress(const char* kind, uint64_t address, uint64_t value, uint64_t size,
                         uint32_t queue) {
	static const bool enabled = [] {
		const auto* v = std::getenv("KYTY_CP_WRITE_LOG");
		return v != nullptr && std::strcmp(v, "1") == 0;
	}();
	if (!enabled) {
		return;
	}
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> seen;
	const uint64_t key = address ^ (static_cast<uint64_t>(kind[0]) << 56u) ^
	                     (static_cast<uint64_t>(kind[1]) << 48u);
	std::scoped_lock lock(mutex);
	if (seen.size() < 600 && seen.insert(key).second) {
		std::printf("Kyty cp %s 0x%" PRIx64 " value 0x%" PRIx64 " size %" PRIu64 " queue %u\n", kind,
		            address, value, size, queue);
	}
}

CpSeq::Result CommandProcessor::ExecWaitRegMem(const CpSeq::WaitRegMemOp& op) {
	return op.size == sizeof(uint64_t) ? ExecWaitRegMemSized<uint64_t>(op)
	                                   : ExecWaitRegMemSized<uint32_t>(op);
}

template <typename T>
CpSeq::Result CommandProcessor::ExecWaitRegMemSized(const CpSeq::WaitRegMemOp& op) {
	const auto queue =
	    m_interrupt_event_id == 0 ? 0u : static_cast<uint32_t>(m_interrupt_event_id - 0x20 + 1);
	HangWatchdog::Scope blocked("wait-reg-mem-read", op.addr, op.ref, 0, op.mask, op.func);
	const auto* addr  = reinterpret_cast<const T*>(op.addr);
	const auto  ref   = static_cast<T>(op.ref);
	const auto  mask  = static_cast<T>(op.mask);
	const auto  func  = op.func;
	auto        value  = ReadGuestForCp<T>(reinterpret_cast<uint64_t>(addr));
	bool        passed = TestWaitRegMemValue(value, ref, mask, func);
	// KYTY_LABEL_MODE=split: the guest sees end-of-pipe labels at completion, the command
	// processor's own waits at record time, as in record mode (Kyty orders the GPU work itself).
	if (!passed && g_gpu_state != nullptr &&
	    g_gpu_state->CpVisibleLabelValue(reinterpret_cast<uint64_t>(addr), sizeof(T), &value)) {
		passed = TestWaitRegMemValue(value, ref, mask, func);
		if (passed) {
			g_gpu_state->NoteCpVisibleLabelsWaited(reinterpret_cast<uint64_t>(addr), sizeof(T));
		}
		LogCpAddress(passed ? "wait-overlay" : "wait-overlay-fail", op.addr,
		             static_cast<uint64_t>(value), sizeof(T), queue);
	} else if (!passed) {
		LogCpAddress("wait-suspend", op.addr, static_cast<uint64_t>(value), sizeof(T), queue);
	}
	blocked.Observed(static_cast<uint64_t>(value));
	if (passed)
		HangWatchdog::ClearQueueWait(queue);
	else
		HangWatchdog::NoteQueueWait(queue, "WAIT_REG_MEM", op.addr, op.ref, value, op.mask, op.func,
		                            op.size);
	if (HangTrace::CpTraceEnabled()) {
		// One row per wait: the first failed evaluation, and the pass (aux = failed retries).
		struct WaitTrace {
			uint64_t address = 0;
			uint64_t ref     = 0;
			int64_t  retries = -1; // -1: no failed evaluation pending
		};
		static std::unordered_map<const CommandProcessor*, WaitTrace> traces; // GPU thread
		auto&      trace   = traces[this];
		const auto address = reinterpret_cast<uint64_t>(addr);
		const bool same    = trace.retries >= 0 && trace.address == address &&
		                  trace.ref == static_cast<uint64_t>(ref);
		if (!passed && same) {
			trace.retries++;
		} else {
			HangTrace::CpEvent event;
			event.event   = passed ? "wait-pass" : "wait-fail";
			event.address = address;
			event.value   = static_cast<uint64_t>(value);
			event.ref     = static_cast<uint64_t>(ref);
			event.mask    = static_cast<uint64_t>(mask);
			event.aux     = passed ? (same ? trace.retries : 0) : static_cast<int64_t>(func);
			event.size    = sizeof(T);
			HangTrace::RecordCp(event);
			trace = passed ? WaitTrace {} : WaitTrace {address, static_cast<uint64_t>(ref), 0};
		}
	}
	if (!passed) {
		// Waiting on a deferred label: it is written only after its tick completes, so that
		// tick must be submitted before this queue suspends (the slice-end flush would do it
		// too; flushing here also covers a label recorded earlier in this slice).
		if (g_gpu_state != nullptr) {
			if (const auto tick =
			        g_gpu_state->DeferredLabelTick(reinterpret_cast<uint64_t>(addr), sizeof(T));
			    tick != 0) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::WaitRegMemDeferredLabel);
				if (tick >= GetScheduler().CurrentTick()) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::WaitRegMemDeferredLabelFlushes);
					BufferFlush();
				}
			}
		}
		return {true, 0};
	}
	return {};
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	CpSeq::WriteDataOp op;
	op.dst           = reinterpret_cast<uint64_t>(dst);
	op.dw_num        = dw_num;
	op.write_control = write_control;
	(void)Submit(op, src, dw_num * static_cast<uint32_t>(sizeof(uint32_t)));
}

static bool LabelSplitMode();
static bool LabelCompletionMode();

// Diagnostic (Bug C): KYTY_CP_AHEAD_LOG=1 counts command-processor writes the guest CPU can see at
// once (CPU writes at record time) made while deferred labels are pending, i.e. while the CP may be
// ahead of the GPU in split mode; prints the first 32 addresses per kind and totals every 1000.
enum class CpAheadKind : uint32_t { WriteData, WriteDataDeferred, FlipLabel, FlipLabelDeferred, Count };
static void NoteCpAhead(CpAheadKind kind, uint64_t address, uint64_t value) {
	static const bool enabled = [] {
		const auto* v = std::getenv("KYTY_CP_AHEAD_LOG");
		return v != nullptr && std::strcmp(v, "1") == 0;
	}();
	if (!enabled) {
		return;
	}
	static const char*           names[] = {"write-data", "write-data-deferred", "flip-label",
	                                        "flip-label-deferred"};
	static std::atomic<uint64_t> counts[static_cast<uint32_t>(CpAheadKind::Count)] {};
	const auto                   k = static_cast<uint32_t>(kind);
	const auto                   n = counts[k].fetch_add(1, std::memory_order_relaxed) + 1;
	if (n <= 32 || (n % 1000) == 0) {
		std::printf("Kyty cp-ahead %s #%" PRIu64 " 0x%" PRIx64 " value 0x%" PRIx64 "\n", names[k], n,
		            address, value);
	}
}

// KYTY_LABEL_MODE=split: a CPU write-data to a label that is still pending (its guest write waits
// for completion) is deferred behind it, so the guest sees the label, then this write, in
// hardware order and not before the GPU got there. KYTY_LABEL_SPLIT_WRITE_DATA=0: written at once
// (the pending label's write is then skipped if a CP wait passed on it).
static bool SplitWriteDataDeferred() {
	static const bool enabled = [] {
		const auto* v = std::getenv("KYTY_LABEL_SPLIT_WRITE_DATA");
		return LabelSplitMode() && (v == nullptr || std::strcmp(v, "0") != 0);
	}();
	return enabled;
}

void CommandProcessor::ExecWriteData(const CpSeq::WriteDataOp& op, const uint32_t* src) {
	auto*          dst           = reinterpret_cast<uint32_t*>(op.dst);
	const uint32_t dw_num        = op.dw_num;
	const uint32_t write_control = op.write_control;
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}

	// KYTY_WRITE_DATA_GPU (default on): a destination owned by recorded-but-unexecuted GPU work is
	// written on the GPU timeline, after that work, instead of by the CPU now (which drained the
	// GPU through a fault, or was later overwritten by the GPU data's readback).
	static const bool gpu_writes = [] {
		const auto* value = std::getenv("KYTY_WRITE_DATA_GPU");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	if (gpu_writes) {
		const auto address = reinterpret_cast<uint64_t>(dst);
		auto&      cache   = m_renderer.GetBufferCache();
		const bool written =
		    write_one_address
		        ? cache.TryWriteDataGpu(address, src + (dw_num - 1u), sizeof(uint32_t))
		        : cache.TryWriteDataGpu(address, src, uint64_t {dw_num} * sizeof(uint32_t));
		if (written) {
			if (g_gpu_state != nullptr) {
				g_gpu_state->DropCpVisibleLabels(
				    address, write_one_address ? sizeof(uint32_t) : uint64_t {dw_num} * sizeof(uint32_t));
			}
			// KYTY_DRAW_RUN_QUIET_OPS: GPU-timeline writes still end a run (ExecuteOp).
			if (DrawRun::Enabled() && DrawRun::QuietOpsEnabled()) {
				DrawRun::NoteForeignActivity("WriteData gpu");
			}
			Profiler::CountFrameEvent(Profiler::FrameEvent::WriteDataGpu);
			TraceCpLabel("wd-gpu", dst, src[write_one_address ? dw_num - 1u : 0u],
			             uint64_t {dw_num} * 4u);
			return;
		}
	}
	const uint64_t cpu_bytes = write_one_address ? sizeof(uint32_t) : uint64_t {dw_num} * 4u;
	if (g_gpu_state != nullptr && g_gpu_state->HasDeferredLabels()) {
		if (cpu_bytes <= sizeof(uint64_t) && SplitWriteDataDeferred() &&
		    g_gpu_state->CpVisibleLabelOverlaps(op.dst, cpu_bytes)) {
			uint64_t value = 0;
			std::memcpy(&value, write_one_address ? src + (dw_num - 1u) : src, cpu_bytes);
			NoteCpAhead(CpAheadKind::WriteDataDeferred, op.dst, value);
			const bool proxy   = m_defer_next_label;
			m_defer_next_label = false;
			const bool deferred =
			    TryDeferLabel(dst, value, static_cast<uint32_t>(cpu_bytes), false, 0);
			m_defer_next_label = proxy;
			if (deferred) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::WriteDataCpu);
				TraceCpLabel("wd-deferred", dst, value, cpu_bytes);
				return;
			}
		}
		NoteCpAhead(CpAheadKind::WriteData, op.dst, src[write_one_address ? dw_num - 1u : 0u]);
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::WriteDataCpu);
	TraceCpLabel("wd-cpu", dst, src[write_one_address ? dw_num - 1u : 0u], uint64_t {dw_num} * 4u);
	LogCpAddress("write-data", op.dst, src[write_one_address ? dw_num - 1u : 0u],
	             uint64_t {dw_num} * 4u,
	             m_interrupt_event_id == 0 ? 0u : static_cast<uint32_t>(m_interrupt_event_id - 0x20 + 1));

	StreamCopyDefer::NoteGuestWrite(op.dst, write_one_address ? sizeof(uint32_t) : uint64_t {dw_num} * 4u);
	// Completion labels can be reused immediately; memcpy may write a destination twice.
	auto* output = static_cast<volatile uint32_t*>(dst);
	for (uint32_t i = 0; i < dw_num; i++) {
		*output = src[i];
		output += !write_one_address;
	}
	NoteCpWrite(op.dst, write_one_address ? sizeof(uint32_t) : uint64_t {dw_num} * 4u);
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	CpSeq::ReferenceClockOp op;
	op.dst       = dst_address;
	op.num_bytes = num_bytes;
	(void)Submit(op);
}

void CommandProcessor::ExecReferenceClock(const CpSeq::ReferenceClockOp& op) {
	const auto dst_address = op.dst;
	const auto num_bytes   = op.num_bytes;
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = GuestGpuTimestamp();
	StreamCopyDefer::NoteGpuOnlyWrite(dst_address, num_bytes);
	std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	NoteCpWrite(dst_address, num_bytes);
	static std::atomic<uint32_t> clock_log_count {0};
	if (clock_log_count.fetch_add(1) < 64) {
		LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
		     " size=%u\n",
		     dst_address, value, num_bytes);
	}
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	CpSeq::DmaDataOp op;
	op.dst               = dst_address_or_offset;
	op.src               = src_address_or_offset_or_immediate;
	op.num_bytes         = num_bytes;
	op.engine            = engine;
	op.dst_sel           = dst_sel;
	op.dst_cache_policy  = dst_cache_policy;
	op.src_sel           = src_sel;
	op.src_cache_policy  = src_cache_policy;
	op.wait_for_previous = wait_for_previous;
	op.write_confirm     = write_confirm;
	op.block_engine      = block_engine;
	(void)Submit(op);
}

void CommandProcessor::ExecDmaData(const CpSeq::DmaDataOp& op) {
	const uint8_t  engine                             = op.engine;
	const uint8_t  dst_sel                            = op.dst_sel;
	const uint8_t  dst_cache_policy                   = op.dst_cache_policy;
	const uint64_t dst_address_or_offset              = op.dst;
	const uint8_t  src_sel                            = op.src_sel;
	const uint8_t  src_cache_policy                   = op.src_cache_policy;
	const uint64_t src_address_or_offset_or_immediate = op.src;
	const uint32_t num_bytes                          = op.num_bytes;
	const uint8_t  wait_for_previous                  = op.wait_for_previous;
	const uint8_t  write_confirm                      = op.write_confirm;
	const uint8_t  block_engine                       = op.block_engine;
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	// Hang-trace CP rows for GDS transfers (the append/consume counter resets and readbacks of
	// passes such as Astro Bot's GI ray-bundle linked lists): address = GDS byte offset, value =
	// fill value or memory address, ref = 0 fill, 1 memory to GDS, 2 GDS to memory.
	const auto trace_gds = [&](uint64_t gds_offset, uint64_t value, uint64_t direction) {
		if (HangTrace::CpTraceEnabled()) {
			HangTrace::CpEvent event;
			event.event   = "gds-dma";
			event.address = gds_offset;
			event.value   = value;
			event.ref     = direction;
			event.size    = num_bytes;
			HangTrace::RecordCp(event);
		}
	};
	if (src_sel == 2) {
		if (dst_gds) {
			trace_gds(dst_address_or_offset, src_address_or_offset_or_immediate & 0xffffffffu, 0);
		}
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}
	if (dst_gds) {
		trace_gds(dst_address_or_offset, src_address_or_offset_or_immediate, 1);
	} else if (src_gds) {
		trace_gds(src_address_or_offset_or_immediate, dst_address_or_offset, 2);
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	if (HangTrace::Enabled()) {
		submission.enqueue_ns = HangTrace::NowNs();
	}
	if (PaceLog::Enabled()) {
		submission.pace_enqueue_ns = PaceLog::NowNs();
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	submission.sequence    = m_next_submission_sequence++;
	submission.frame_fence = FrameFenceEnabled() ? m_done_boundary : 0;
	if (HangTrace::CpTraceEnabled()) {
		HangTrace::CpEvent event;
		event.event   = submission.type == SubmissionType::Compute   ? "admit-compute"
		                : submission.type == SubmissionType::Graphics ? "admit-graphics"
		                                                               : "admit-flip";
		event.address = reinterpret_cast<uint64_t>(submission.commands.data());
		event.value   = submission.frame_fence;
		event.ref     = static_cast<uint64_t>(m_done_num.load());
		event.size    = submission.commands.size() * sizeof(uint32_t);
		event.queue   = submission.queue_id;
		event.seq     = submission.sequence;
		HangTrace::RecordCp(event);
	}
	m_in_flight.insert(submission.sequence);
	// Astro Bot PC (KYTY_CP_PAR=1a): queue-0 graphics submissions of even rank in the frame (rank 1:
	// the reset_processor one) go to resolver context B, unless 1a mode cannot run them there.
	if (CpPar::Enabled() && submission.queue_id == 0 &&
	    submission.type == SubmissionType::Graphics) {
		m_cp_par_rank = submission.reset_processor ? 1u : m_cp_par_rank + 1u;
		// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): rank r -> context (r - 1) % N (N = 2: even -> B).
		const uint32_t context = (m_cp_par_rank - 1u) % CpPar::ContextCount();
		if (context != 0) {
			if (m_sequencer == nullptr) {
				CpPar::NoteFallback(CpPar::Fallback::NoSequencer);
			} else if (CpSeq::VerifyMode() != 0 || !CpPar::Started()) {
				CpPar::NoteFallback(CpPar::Fallback::Verify);
			} else if (!submission.constant_commands.empty()) {
				CpPar::NoteFallback(CpPar::Fallback::Handoff);
			} else {
				submission.cp_par_context = context;
			}
		}
	}
	if (m_sequencer != nullptr && submission.queue_id == 0) {
		// Admission order is the sequencer's parse order (and the resolver's queue-0 order).
		CpSeq::Intake intake;
		intake.sequence          = submission.sequence;
		intake.frame_fence       = submission.frame_fence;
		if (CpPar::Concurrent() && submission.frame_fence != 0) {
			// Astro Bot PC (KYTY_CP_PAR=1b3) : sous m_queue_mutex, dans l'ordre d'admission.
			static uint64_t last_fence = 0, first_seq = 0;
			if (submission.frame_fence != last_fence) {
				last_fence = submission.frame_fence;
				first_seq  = submission.sequence;
			}
			intake.fence_first = first_seq;
		}
		intake.commands          = submission.commands;
		intake.constant_commands = submission.constant_commands;
		intake.graphics          = submission.type == SubmissionType::Graphics;
		intake.reset_processor   = submission.reset_processor;
		intake.context           = submission.cp_par_context; // Astro Bot PC (KYTY_CP_PAR)
		if (intake.graphics) {
			ParCensus::NoteAdmit(intake.sequence, intake.reset_processor); // Astro Bot PC : KYTY_CP_PAR_CENSUS
		}
		m_sequencer->Admit(intake);
	}
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	HangWatchdog::Scope wait("cp-idle", reinterpret_cast<uint64_t>(this));
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

/// Body of the GPU thread. Raises the thread's host priority on Windows, then takes queued commands
/// and submissions one at a time and runs them until the GPU is told to stop.
/// @param data the GuestGpu that owns the queues
void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	// The command processor is the frame-rate limit; keep it ahead of guest spin loops.
	Common::RaiseCurrentThreadPriority();
	// KYTY_CPU_RESERVE: its own physical core.
	Common::PlaceCurrentThread(Common::ThreadRole::Cp);
	// KYTY_CP_GAP_STATS (drawPrep/cpGaps.h): this thread's time by cause.
	CpGaps::AttachThread();
	g_gpu_thread = true;
	g_gpu_state  = gpu;
	// KYTY_LIVE_FILE (common/liveSwitch.h): live switches, applied at this thread's flips.
	Live::Start();
	// KYTY_CP_SAMPLER (diagnostics): stack samples of this thread (threadSampler.h).
	StartThreadSampler("cp");
	// KYTY_CP_COMMIT: parsed (and its process-wide switches set) before the first command.
	(void)CpCommit::Parts();
	// KYTY_RENDER_LOCK_ASYMMETRIC (RenderMutex): this thread is the renderer lock's owner.
	gpu->m_renderer.GetMutex().BecomeOwner();
	if (CpPar::Lanes2b()) {
		// Astro Bot PC (KYTY_CP_PAR=2b): lane A's tick range (MasterSemaphore::EnableRanges).
		CommandScheduler::SetThreadTickLane(0);
	}
	if (CpPar::Concurrent()) {
		// Astro Bot PC (KYTY_CP_PAR=1b3, cpParallel.h): this thread holds the CP op lock except at
		// its interleave points and while it waits for the run-ahead.
		(void)CpPar::OpLockAcquire(0);
	}

	const bool wakeups       = CpWakeupsEnabled();
	uint64_t   spin_deadline = 0; // 0: not spinning on blocked queues
	for (;;) {
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				HangWatchdog::Scope idle("cp-no-work", reinterpret_cast<uint64_t>(gpu));
				const CpGaps::Scope gap(CpGaps::Cat::Idle);
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				gpu->m_done_progress.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				int  selected_queue = -1;
				bool fence_held     = false;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id    = (gpu->m_next_queue + offset) % QueueCount;
					auto&      queue = gpu->m_queues[id];
					if (queue.empty() || queue.front().blocked) {
						continue;
					}
					if (!gpu->FrameFencePassed(queue.front())) {
						HangWatchdog::NoteQueue(id, queue.front().sequence, "frame-fence-held",
						                        queue.front().frame_fence);
						if (queue.front().fence_hold_ns == 0) {
							queue.front().fence_hold_ns = CpNowNs();
						}
						fence_held = true;
						continue;
					}
					if (queue.front().fence_hold_ns != 0 &&
					    CpNowNs() - queue.front().fence_hold_ns > FrameFenceTimeoutNs) {
						static std::atomic_bool logged {false};
						if (!logged.exchange(true)) {
							LOGF("CP frame fence: queue %u waited over 2 s for the previous frame; "
							     "running it anyway (KYTY_FRAME_FENCE)\n",
							     id);
						}
					}
					selected_queue = static_cast<int>(id);
					break;
				}
				if (fence_held && selected_queue >= 0) {
					// The fence changed which queue runs next.
					Profiler::CountFrameEvent(Profiler::FrameEvent::FrameFenceHolds);
				}
				if (selected_queue < 0) {
					HangWatchdog::Scope suspended("cp-all-queues-suspended",
					                              reinterpret_cast<uint64_t>(gpu),
					                              gpu->m_submission_count);
					gpu->m_processing = false;
					const CpGaps::Scope gap(CpGaps::Cat::Blocked);
					if (!wakeups) {
						gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					} else {
						const auto now = CpNowNs();
						if (spin_deadline == 0) {
							spin_deadline = now + CpBlockedSpinNs();
						}
						if (now < spin_deadline) {
							// Every queue is suspended: retry shortly without sleeping.
							Profiler::CountFrameEvent(Profiler::FrameEvent::CpBlockedSpins);
							gpu->m_queue_mutex.Unlock();
							std::this_thread::yield();
							gpu->m_queue_mutex.Lock();
						} else {
							// Completions, deferred labels and flips signal this condition.
							Profiler::CountFrameEvent(Profiler::FrameEvent::CpBlockedSleeps);
							gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 1000);
						}
					}
					gpu->m_has_blocked.store(false, std::memory_order_release);
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				if (CpPar::Concurrent() && selected_queue == 0 && g_cp_par_run.offered &&
				    g_cp_par_run.target == queue.front().sequence) {
					// Astro Bot PC (KYTY_CP_PAR=1b3): the run-ahead's submission: it stops first
					// (it may need the op lock to get there, never this mutex).
					const auto front_sequence = queue.front().sequence;
					gpu->m_queue_mutex.Unlock();
					(void)gpu->CpParHandOver(front_sequence);
					gpu->m_queue_mutex.Lock();
				}
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
				has_submission    = true;
			}
		}
		if (should_stop) {
			if (CpPar::Concurrent()) {
				// Astro Bot PC (KYTY_CP_PAR=1b3): resolver #2 idle, the op lock released.
				if (g_cp_par_run.offered) {
					(void)gpu->CpParHandOver(g_cp_par_run.target);
				}
				CpPar::OpLockRelease(0);
			}
			gpu->m_gfx_cp->BufferWait();
			// Deferred label writes still queued on the completion runner write the backing
			// directly once commands are refused; finish them while this GuestGpu exists.
			gpu->m_renderer.GetCommandScheduler().DrainPriorityOperations();
			gpu->m_renderer.GetMutex().ResignOwner();
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			HangWatchdog::Scope service("cp-service-execute", reinterpret_cast<uint64_t>(gpu));
			const CpGaps::Scope gap(CpGaps::Cat::Service);
			command();
			SyncEpoch::Advance();
			// KYTY_DRAW_RUN: a service command is other command-processor work.
			if (DrawRun::Enabled()) {
				DrawRun::NoteForeignActivity("service command");
			}
			spin_deadline = 0;

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		uint64_t slice_start = 0;
		if (HangTrace::Enabled()) {
			slice_start = HangTrace::NowNs();
			if (!submission.started && submission.enqueue_ns != 0) {
				HangTrace::RecordQueueWait(submission.queue_id, slice_start - submission.enqueue_ns);
			}
		}
		HangTrace::SetCpContext(submission.queue_id, submission.sequence);
		HangWatchdog::SetCpContext(submission.queue_id, submission.sequence);
		HangWatchdog::NoteQueue(submission.queue_id, submission.sequence, "executing",
		                        submission.frame_fence);
		HangWatchdog::DebugDelay("queue", submission.queue_id);
		Live::CpSliceBegin(); // CP busy time in the "Live:" lines (KYTY_LIVE_FILE only)
		bool complete = false;
		{
			const CpGaps::Scope gap(CpGaps::Cat::Slice);
			complete = gpu->Process(submission);
		}
		HangWatchdog::NoteQueue(
		    submission.queue_id, submission.sequence,
		    complete ? "complete"
		             : (submission.command_execution.Yielded() ? "yielded" : "blocked"),
		    submission.frame_fence);
		Live::CpSliceEnd();
		if (HangTrace::Enabled()) {
			HangTrace::RecordQueueBusy(submission.queue_id, HangTrace::NowNs() - slice_start,
			                           complete);
		}
		if (HangTrace::CpTraceEnabled()) {
			HangTrace::CpEvent event;
			event.event = "slice";
			// 0 complete, 1 suspended (blocked), 2 yielded
			event.aux = complete ? 0 : (submission.command_execution.Yielded() ? 2 : 1);
			event.value = submission.slice_progress ? 1 : 0;
			HangTrace::RecordCp(event);
		}
		HangTrace::SetCpContext(UINT32_MAX, 0);
		HangWatchdog::SetCpContext(UINT32_MAX, 0);

		if (complete || submission.slice_progress) {
			spin_deadline = 0;
		}
		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete && submission.command_execution.Yielded()) {
			// Yielded to other queues: runnable again in round-robin order.
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else if (!complete) {
			submission.blocked = true;
			gpu->m_has_blocked.store(true, std::memory_order_release);
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
			gpu->m_in_flight.erase(submission.sequence);
			if (gpu->m_done_waiters != 0) {
				gpu->m_done_progress.SignalAll();
			}
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

bool GuestGpu::Process(Submission& submission) {
	if (CpPar::Enabled()) {
		CpPar::MaybeReport(); // Astro Bot PC (KYTY_CP_PAR): "CpPar 10s: ..."
	}
	if (m_sequencer != nullptr && submission.queue_id == 0) {
		return ProcessSequenced(submission);
	}
	const bool first_slice = !submission.started;
	auto& cp = GetProcessor(submission.queue_id);
	// A new submission, or a slice after other queues ran (syncEpoch.h).
	SyncEpoch::Advance();
	if (first_slice) {
		SyncEpoch::AdvanceSubmission();
	}
	if (DrawRun::Enabled()) {
		DrawRun::NoteForeignActivity(first_slice ? "submission start" : "submission resume");
	}

	if (first_slice && submission.reset_processor) {
		cp.Reset();
	}
	if (first_slice && submission.reset_processor && submission.type != SubmissionType::Compute) {
		CommitStats::OnFrameBoundary();
		CpGaps::NoteFrame();
		Common::PrepPhases::NoteFrame();
	}
	if (first_slice && RepeatTrace::Enabled()) {
		// KYTY_CP_REPEAT_TRACE: a guest frame starts with the processor reset after
		// sceAgcSuspendPoint; every submission's content is hashed.
		if (submission.reset_processor && submission.type != SubmissionType::Compute) {
			RepeatTrace::OnFrameBoundary();
		}
		RepeatTrace::OnSubmission(submission.queue_id, submission.sequence, submission.commands,
		                          submission.constant_commands);
	}

	if (first_slice) {
		submission.started = true;
		if (submission.type == SubmissionType::Graphics) {
			PaceLog::NoteCpSubmission(submission.pace_enqueue_ns, submission.reset_processor);
		}
		cp.SetSubmitId(++m_submit_id);
		cp.ResetDeCe();
		cp.SetFlip({});
		if (!submission.constant_commands.empty()) {
			// KYTY_CP_SEQ_VERIFY: the constant engine's stream changes the front state (constant
			// RAM, CE counter) between slices of the draw engine's, which a reference front
			// following one stream cannot mirror; neither stream is compared.
			submission.command_execution.DisableSequencerVerify();
			submission.constant_execution.DisableSequencerVerify();
		}
	}

	cp.BufferInit();
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Graphics: complete = ProcessGraphicsDirect(submission, cp); break;
		case SubmissionType::Compute: {
			const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
			const auto*     buffer = submission.commands.data();
			static uint32_t compute_batch_log_count = 0;
			if (first_slice && num_dw <= 128 && compute_batch_log_count++ < 32) {
				LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(buffer), num_dw);
				for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
					LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
				}
			}
			if (first_slice) {
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
			}
			complete = cp.Process(submission.command_execution, submission.commands) ==
			           Pm4ProcessResult::Complete;
			submission.slice_progress = submission.command_execution.MadeProgress();
			if (submission.command_execution.MadeProgress()) {
				if (complete) {
					m_renderer.RunGarbageCollector();
				}
				cp.BufferFlush();
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::FlipPreparation:
			m_renderer.RunGarbageCollector();
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
	}

	return complete;
}

bool GuestGpu::ProcessGraphicsDirect(Submission& submission, CommandProcessor& cp) {
	bool complete   = true;
	bool progressed = false;
	submission.constant_complete |= submission.constant_commands.empty();
	for (;;) {
		bool round_progress = false;
		if (!submission.constant_complete) {
			submission.constant_complete =
			    cp.Process(submission.constant_execution, submission.constant_commands) ==
			    Pm4ProcessResult::Complete;
			round_progress |= submission.constant_execution.MadeProgress();
		}
		cp.SetCeComplete(submission.constant_complete);
		if (!submission.command_complete) {
			submission.command_complete =
			    cp.Process(submission.command_execution, submission.commands) ==
			    Pm4ProcessResult::Complete;
			round_progress |= submission.command_execution.MadeProgress();
		}
		progressed |= round_progress;
		complete = submission.command_complete && submission.constant_complete;
		if (complete || !round_progress || submission.command_execution.Yielded()) {
			break;
		}
	}
	submission.slice_progress = progressed;
	if (progressed) {
		if (complete) {
			m_renderer.RunGarbageCollector();
		}
		cp.BufferFlush();
	} else if (complete) {
		m_renderer.RunGarbageCollector();
	}
	return complete;
}

// KYTY_CP_SEQ=1 (cpSequencer.h): a queue-0 submission. The sequencer has applied its front setup
// (processor reset, DE/CE counters, flip info) and emits its ops; this thread executes them.
bool GuestGpu::ProcessSequenced(Submission& submission) {
	auto&      cp          = *m_gfx_cp;
	const bool first_slice = !submission.started;
	// A new submission, or a slice after other queues ran (syncEpoch.h).
	SyncEpoch::Advance();
	if (DrawRun::Enabled()) {
		DrawRun::NoteForeignActivity(first_slice ? "submission start" : "submission resume");
	}
	if (first_slice) {
		SyncEpoch::AdvanceSubmission();
		submission.started = true;
		PaceLog::NoteCpSubmission(submission.pace_enqueue_ns, submission.reset_processor);
		cp.SetSubmitId(++m_submit_id, submission.cp_par_context); // Astro Bot PC (2b: its lane)
		// The frame fence's ordering point: the sequencer reads a fenced submission's command
		// bytes only from here on.
		m_sequencer->NoteStarted(submission.sequence);
		// KYTY_CP_COMMIT_STATS: a guest frame starts with the processor reset after
		// sceAgcSuspendPoint.
		if (submission.reset_processor && submission.type != SubmissionType::Compute) {
			CommitStats::OnFrameBoundary();
			CpGaps::NoteFrame();
			Common::PrepPhases::NoteFrame();
		}
	}
	cp.BufferInit();
	if (submission.handoff) {
		const bool complete = ProcessGraphicsDirect(submission, cp);
		if (complete) {
			cp.EndHandoff(submission.sequence);
		}
		return complete;
	}
	bool             handoff = false;
	Pm4ProcessResult result  = Pm4ProcessResult::Complete;
	// Astro Bot PC (KYTY_CP_PAR=1b1): the draw-prep decisions of this slice, for its context's
	// line in "CpPar 10s" (the contexts run in turn: every decision meanwhile is this slice's).
	const bool prep_stats      = CpPar::SeparateWindows();
	uint64_t   prep_committed  = 0;
	uint64_t   prep_fell_back  = 0;
	auto       prep_decisions  = [](uint64_t& committed, uint64_t& fell_back) {
		auto& totals = DrawPrep::GetTotals();
		committed    = totals.committed.load(std::memory_order_relaxed) +
		            totals.dispatch_committed.load(std::memory_order_relaxed);
		fell_back = totals.fallbacks.load(std::memory_order_relaxed) +
		            totals.dispatch_fallbacks.load(std::memory_order_relaxed);
	};
	if (prep_stats) {
		prep_decisions(prep_committed, prep_fell_back);
	}
	if (!CpPar::Enabled()) {
		result = cp.ResolveSubmission(submission.command_execution, submission.sequence,
		                              submission.commands, handoff);
	} else if (CpPar::Concurrent()) {
		// Astro Bot PC (KYTY_CP_PAR=1b3, cpParallel.h): every submission on this thread (the
		// oldest context), in its context's ring and recording target; the next one may run ahead.
		CpParEnterSlice(submission);
		CpParMaybeOffer(submission);
		const auto start = CpPar::NowNs();
		result = cp.ResolveSubmission(submission.command_execution, submission.sequence,
		                              submission.commands, handoff, submission.cp_par_context);
		CpPar::NoteResolveTime(0, CpPar::NowNs() - start); // Thread_Gpu's time
	} else if (submission.cp_par_context == 0) {
		// Astro Bot PC (KYTY_CP_PAR=1a): context A, on this thread, timed.
		const auto start = CpPar::NowNs();
		result = cp.ResolveSubmission(submission.command_execution, submission.sequence,
		                              submission.commands, handoff, 0);
		CpPar::NoteResolveTime(0, CpPar::NowNs() - start);
	} else {
		// Astro Bot PC (KYTY_CP_PAR=1a): context B: this slice runs on CP resolver #2 with the
		// same code, while this thread waits (serving its requests: service commands, flips).
		// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): on the resolver of the submission's lane.
		struct Slice {
			CommandProcessor* cp;
			Submission*       submission;
			bool              handoff;
			Pm4ProcessResult  result;
			uint32_t          lane;
		} slice {&cp, &submission, false, Pm4ProcessResult::Complete, submission.cp_par_context};
		CpPar::RunSlice(
		    slice.lane,
		    [](void* argument) {
			    auto&      s     = *static_cast<Slice*>(argument);
			    const auto start = CpPar::NowNs();
			    s.result = s.cp->ResolveSubmission(s.submission->command_execution,
			                                       s.submission->sequence, s.submission->commands,
			                                       s.handoff, s.lane);
			    CpPar::NoteResolveTime(s.lane, CpPar::NowNs() - start);
		    },
		    &slice);
		result  = slice.result;
		handoff = slice.handoff;
	}
	if (CpPar::Enabled() && !handoff && result == Pm4ProcessResult::Complete &&
	    submission.type == SubmissionType::Graphics) {
		CpPar::NoteCompleted(submission.cp_par_context); // Astro Bot PC (KYTY_CP_PAR)
	}
	if (prep_stats) {
		// Astro Bot PC (KYTY_CP_PAR=1b1).
		uint64_t committed = 0;
		uint64_t fell_back = 0;
		prep_decisions(committed, fell_back);
		uint64_t published = 0;
		uint64_t retired   = 0;
		cp.CpParWindowCounts(submission.cp_par_context, published, retired);
		CpPar::NotePrep(submission.cp_par_context, committed - prep_committed,
		                fell_back - prep_fell_back, published, retired);
	}
	if (handoff) {
		// A constant-engine submission: parsed and executed here, in Direct mode, while the
		// sequencer waits (it applied the front setup already).
		submission.handoff = true;
		submission.command_execution.DisableSequencerVerify();
		submission.constant_execution.DisableSequencerVerify();
		cp.BeginHandoff();
		cp.BufferInit();
		const bool complete = ProcessGraphicsDirect(submission, cp);
		if (complete) {
			cp.EndHandoff(submission.sequence);
		}
		return complete;
	}
	const bool complete   = result == Pm4ProcessResult::Complete;
	const bool progressed = submission.command_execution.MadeProgress();
	if (submission.type == SubmissionType::FlipPreparation) {
		submission.slice_progress = complete;
		if (complete) {
			m_renderer.RunGarbageCollector();
			cp.PrepareCpuFlip(submission.flip_request_id);
		}
		return complete;
	}
	submission.slice_progress = progressed;
	if (progressed) {
		if (complete) {
			m_renderer.RunGarbageCollector();
		}
		cp.BufferFlush();
	} else if (complete) {
		m_renderer.RunGarbageCollector();
	}
	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	EXIT_IF(m_front_mode == FrontMode::Reference);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
		// A new stream (KYTY_CP_SEQ): ops refer to it by id and packet count.
		execution.m_stream_id    = ++m_next_stream_id;
		execution.m_packets      = 0;
		execution.m_packets_hash = 0;
		if (m_front_mode == FrontMode::Inline && CpSeq::VerifyMode() != 0) {
			if (m_verifier == nullptr) {
				m_verifier.reset(new CpSeq::Verifier(m_renderer, m_interrupt_event_id));
			}
			if (execution.m_verify) {
				m_verifier->Attach(*this, execution.m_stream_id, commands);
			} else {
				m_verifier->Detach();
			}
		}
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;
	execution.m_yield         = false;
	execution.m_yielded       = false;
	LaneCp().slice_draws             = 0;
	// The draw-prep packet hook, decided once per slice instead of per packet: with draw prep on
	// it is on for the process (PacketHookEnabled); only the Tracy-gated fence histogram of draw
	// prep off can change, and it now follows a connection at the next slice.
	m_packet_hook = DrawPrep::PacketHookActive() && DrawPrepEngine() != nullptr;

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~ExecutionScope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} execution_scope(*this, execution);

	ProcessPm4(execution);
	// Draw-prep: every draw parsed in this slice is committed before the slice ends.
	DrainPreparedDraws();
	const bool complete = execution.m_buffer_stack.empty();
	if (complete && m_verifier != nullptr && m_verifier->Follows(execution.m_stream_id)) {
		// KYTY_CP_SEQ_VERIFY: the reference front must end its stream here too.
		m_verifier->Finish();
	}
	return complete ? Pm4ProcessResult::Complete : Pm4ProcessResult::Blocked;
}

// KYTY_CP_REPEAT_TRACE: the guest address of the packet whose handler is running (a draw packet
// when called from DrawIndex/DrawIndexAuto).
void CommandProcessor::NoteRepeatTraceDrawPacket() {
	const uint32_t* packet = nullptr;
	if (g_current_execution != nullptr && !g_current_execution->m_buffer_stack.empty()) {
		const auto& cursor = g_current_execution->m_buffer_stack.back();
		packet             = cursor.commands.data() + cursor.offset_dw;
	}
	RepeatTrace::NoteDrawPacket(packet);
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	if (RepeatTrace::Enabled() && m_front_mode != FrontMode::Reference) {
		RepeatTrace::OnIndirectBuffer(commands, chain);
	}
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	while (!execution.m_buffer_stack.empty()) {
		if (!ProcessPacket(execution)) {
			return;
		}
	}
}

// KYTY_CP_SEQ: a completed (or skipped) packet of the stream is counted and, in verify mode,
// hashed with its address: what the fronts parsed is compared through it.
static void CountPacket(Pm4Execution& execution, const uint32_t* packet, uint32_t packet_dw,
                        uint64_t& packets, uint64_t& packets_hash, uint64_t guest_address) {
	(void)execution;
	packets++;
	if (CpSeq::PacketHashing()) {
		packets_hash = XXH3_64bits_withSeed(packet, uint64_t {packet_dw} * sizeof(uint32_t),
		                                    packets_hash ^ guest_address);
	}
}

bool CommandProcessor::ProcessPacket(Pm4Execution& execution) {
	// A reference front (KYTY_CP_SEQ_VERIFY) only parses: no services, sync epochs, draw-prep
	// window or flush batching, and none of the parse's counters. The sequencer (thread mode)
	// parses too: the resolver runs services and flush batching between ops, and the sync epoch
	// advance travels with the next op.
	const bool reference = m_front_mode == FrontMode::Reference;
	// P3c: the speculative front parses like the sequencer, without waiting for anything.
	const bool prefetch  = m_front_mode == FrontMode::Prefetch;
	const bool sequencer = m_front_mode == FrontMode::Thread || prefetch;
	if (g_gpu_state != nullptr && !reference && !sequencer) {
		if (m_draw_prep != nullptr && m_draw_prep->Pending()) {
			// Draw-prep: service commands (readbacks, unmaps) observe every parsed draw, so
			// they only run with an empty window.
			if (g_gpu_state->HasPendingCommands()) {
				m_draw_prep->Drain();
				g_gpu_state->ProcessCommands();
			}
		} else {
			g_gpu_state->ProcessCommands();
		}
	}
	auto& cursor = execution.m_buffer_stack.back();
	EXIT_IF(cursor.offset_dw > cursor.commands.size());
	if (cursor.offset_dw == cursor.commands.size()) {
		execution.m_buffer_stack.pop_back();
		return true;
	}
	if (sequencer && cursor.checked_epoch != m_barrier_epoch &&
	    !(prefetch ? CheckCommandBytesPrefetch(cursor) : CheckCommandBytes(cursor))) {
		execution.m_suspended = true; // stopping (the speculative front: bytes it cannot read)
		return false;
	}

	// Placement samples (common/cpuPlacement.h), every 256th packet, on the thread that
	// executes it: not the sequencer (the resolver samples per op) or a reference front.
	if (!reference && !sequencer && (++m_placement_packets & 255u) == 0u) {
		Common::SamplePlacement(Common::ThreadRole::Cp);
	}
	const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
	// The packet's guest address (a lockstep copy of a command buffer is parsed elsewhere).
	const uint64_t guest_packet =
	    cursor.copy != nullptr ? cursor.guest_address + uint64_t {cursor.offset_dw} * 4u
	                           : reinterpret_cast<uint64_t>(packet);
	const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
	const auto        remaining_dw  = total_dw - cursor.offset_dw;
	const auto        packet_header = packet[0];
	const auto        opcode        = (packet_header >> 8u) & 0xffu;
	EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);

	// Type-2 padding and one-word NOP reservations (Pm4::PacketSizeDw, upstream 4c8fc53a8).
	if (packet_header == 0x80000000u || Pm4::PacketSizeDw(packet_header) == 1u) {
		cursor.offset_dw++;
		execution.m_made_progress = true;
		CountPacket(execution, packet, 1, execution.m_packets, execution.m_packets_hash,
		            guest_packet);
		TrackAdoptPacket(packet, 1, guest_packet);
		return true;
	}

	EXIT_NOT_IMPLEMENTED(remaining_dw < 2);
	if (HangWatchdog::Enabled() && !reference && !sequencer) {
		const auto q =
		    m_interrupt_event_id == 0 ? 0u : static_cast<uint32_t>(m_interrupt_event_id - 0x20 + 1);
		const auto word = [&](uint32_t i) {
			return i < std::min<uint32_t>(remaining_dw, KYTY_PM4_LEN(packet_header) + 2u)
			           ? packet[i]
			           : 0u;
		};
		HangWatchdog::NotePacket(q, LaneCp().submit_id, guest_packet, packet_header, word(1), word(2),
		                         word(3), word(4));
	}

	if (GraphicsRunDebugDumpEnabled() && !reference && !prefetch) {
		LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
		     " len=%" PRIu32 "\n",
		     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
	}

	if ((packet_header & 1u) != 0 && !reference && !prefetch) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::PredicatedPackets);
	}
	if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
		auto packet_dw = KYTY_PM4_LEN(packet_header);
		EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
		if (!reference && !prefetch) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::PredicatedPacketsSkipped);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.fetch_add(1) < 128) {
					const auto dst = packet[3] | (static_cast<uint64_t>(packet[4]) << 32u);
					const auto val = packet[5] | (static_cast<uint64_t>(packet[6]) << 32u);
					LOGF("\t predicated skip: R_RELEASE_MEM dst=0x%016" PRIx64
					     ", value=0x%016" PRIx64 ", action=0x%08" PRIx32
					     ", gcr/data/int=0x%08" PRIx32 "\n",
					     dst, val, packet[1], packet[2]);
				}
			}
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		CountPacket(execution, packet, packet_dw, execution.m_packets, execution.m_packets_hash,
		            guest_packet);
		TrackAdoptPacket(packet, packet_dw, guest_packet);
		return true;
	}

	auto handler = g_cp_op_func[opcode];

	if (handler == nullptr) {
		const auto offset = total_dw - remaining_dw;
		LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
		     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
		     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
		     reinterpret_cast<uint64_t>(packet));
		const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
		const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
		auto* const base       = packet - offset;
		for (uint32_t i = dump_begin; i < dump_end; i++) {
			LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
		}
		EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
		     total_dw - remaining_dw, packet_header);
	}

	if (sequencer) {
		// The resolver advances the epoch before the next op, i.e. before the next GPU work.
		if (SyncEpoch::Enabled() &&
		    DrawPrep::AdvancesSyncEpoch(packet_header & ~1u, packet + 1, remaining_dw)) {
			m_epoch_pending = true;
		}
	} else if (!reference) {
		// KYTY_SYNC_EPOCH (syncEpoch.h): every fence but a register load from memory is a point
		// where guest CPU writes must become visible to the GPU work that follows it.
		if (SyncEpoch::Enabled() &&
		    DrawPrep::AdvancesSyncEpoch(packet_header & ~1u, packet + 1, remaining_dw)) {
			SyncEpoch::Advance();
		}
		if (m_packet_hook) [[unlikely]] {
			// Window fences commit every pending draw before their handler runs.
			if (auto* engine = m_draw_prep.get(); engine != nullptr) {
				const auto header       = packet_header & ~1u;
				auto       packet_class = DrawPrep::ClassifyPacket(header, packet + 1, remaining_dw);
				auto       fence_kind   = DrawPrep::FenceKind::Other;
				if (packet_class == DrawPrep::PacketClass::Fence) {
					fence_kind = DrawPrep::ClassifyFence(header);
					// Register loads from clean guest memory keep the window open (drawPrep.h).
					DrawPrep::RegisterIndirectRange pairs;
					if (fence_kind == DrawPrep::FenceKind::RegIndirect &&
					    DrawPrep::RegisterIndirectWindowEnabled() &&
					    DrawPrep::RegisterIndirectPairs(header, packet + 1, remaining_dw, pairs) &&
					    (pairs.size == 0 ||
					     LibKernel::Memory::IsGpuCleanForRead(pairs.address, pairs.size))) {
						packet_class = DrawPrep::PacketClass::WindowSafe;
						Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepRegIndirectKept);
						DrawPrep::GetTotals().register_indirect_kept.fetch_add(
						    1, std::memory_order_relaxed);
					}
				}
				engine->OnPacket(packet_class, fence_kind);
			}
		}
	}
	if (Loader::GameCapture::Active() && !reference &&
	    (opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 || opcode == Pm4::IT_DRAW_INDEX_AUTO ||
	     opcode == Pm4::IT_DRAW_INDEX_INDIRECT || opcode == Pm4::IT_DRAW_INDEX_2)) [[unlikely]] {
		// Astro Bot PC (KYTY_GAME_HOOKS=capture): the CP's view of this draw, compared with the
		// game-side record sampled under the same packet address (loader/gameCapture.h).
		Loader::GameCapture::CpDrawView view;
		view.packet      = guest_packet;
		view.opcode      = opcode;
		view.speculative = prefetch;
		const auto* body = packet + 1;
		if (opcode == Pm4::IT_DRAW_INDEX_OFFSET_2) {
			view.index_offset = body[1];
			view.index_count  = body[2];
		} else if (opcode == Pm4::IT_DRAW_INDEX_AUTO) {
			view.index_count = body[0];
		} else if (opcode == Pm4::IT_DRAW_INDEX_2) {
			view.index_count = body[3];
		}
		if (m_front_mode != FrontMode::Direct) {
			view.instances_known = m_front_instances_known;
			view.instances       = m_front_num_instances;
		} else {
			view.instances_known = m_pending_num_instances.empty();
			view.instances       = m_num_instances;
		}
		view.ib_base  = m_index_base_addr;
		view.ib_type  = m_index_type_and_size;
		view.rt0_base = m_ctx.GetRenderTarget(0).base.addr;
		view.ps_addr  = m_sh_ctx.GetPs().ps_regs.data_addr;
		view.gs_addr  = m_sh_ctx.GetVs().gs_regs.data_addr;
		view.es_addr  = m_sh_ctx.GetVs().es_regs.data_addr;
		view.hs_addr  = m_sh_ctx.GetVs().hs_regs.data_addr;
		Loader::GameCapture::CheckCpDraw(view);
	}
	const auto packet_dw =
	    handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
	EXIT_IF(packet_dw > remaining_dw);
	if (execution.m_suspended) {
		// The packet runs again when the stream resumes. A reference front's lockstep packet may
		// have chosen a buffer from its placeholder result before suspending (branches): drop it.
		execution.m_next_buffer = {};
		execution.m_chain       = false;
		return false;
	}
	cursor.offset_dw += packet_dw;
	execution.m_made_progress = true;
	CountPacket(execution, packet, packet_dw, execution.m_packets, execution.m_packets_hash,
	            guest_packet);
	TrackAdoptPacket(packet, packet_dw, guest_packet);
	if (!reference && !sequencer && LaneCp().deferred_eop_flushes != 0 &&
	    ++LaneCp().packets_since_eop_request >= EopFlushPacketLimit() && OptionalSubmitAllowed()) {
		// Bound how long a batched end-of-pipe interrupt can wait inside a long slice.
		BufferFlush();
	}
	if (!execution.m_next_buffer.empty()) {
		// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
		if (execution.m_chain) {
			cursor = {execution.m_next_buffer};
		} else {
			execution.m_buffer_stack.push_back({execution.m_next_buffer});
		}
		execution.m_next_buffer = {};
	}
	if (execution.m_yield) {
		execution.m_yield   = false;
		execution.m_yielded = true;
		return false;
	}
	return true;
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	if (m_front_mode != FrontMode::Direct) {
		// Front state; the back applies it with the next indirect draw (m_front_num_instances).
		m_front_num_instances   = num_instances;
		m_front_instances_known = true;
		return;
	}
	m_num_instances = num_instances;
	m_pending_num_instances.clear();
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	if (op == 0x00) {
		m_predicate_skip = false;
		return;
	}
	CpSeq::PredicationOp payload;
	payload.address         = reinterpret_cast<uint64_t>(address);
	payload.condition       = condition;
	payload.op              = op;
	payload.wait_op         = wait_op;
	payload.count_in_dwords = count_in_dwords;
	const auto result       = Submit(payload);
	if (result.suspended) {
		SuspendPm4();
		return;
	}
	m_predicate_skip = result.value != 0;
}

// KYTY_PREDICATION_NO_DRAIN=1 (default off; upstream KytyPS5 840b9f575): SET_PREDICATION's wait
// selector applies only to Z-pass query readiness, so a boolean (memory) predicate is read
// without draining the GPU first. ReadGuestForCp still synchronizes GPU-written bytes, but labels
// deferred to completion are not run before the read.
static bool PredicationNoDrain() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PREDICATION_NO_DRAIN");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

CpSeq::Result CommandProcessor::ExecPredication(const CpSeq::PredicationOp& payload) {
	const auto  condition = payload.condition;
	const auto  op        = payload.op;
	const auto  wait_op   = payload.wait_op;
	const auto* address   = reinterpret_cast<const volatile void*>(payload.address);
	uint64_t    value     = 0;

	switch (op) {
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			Profiler::CountFrameEvent(Profiler::FrameEvent::OcclusionPredicates);
			// Native dumps are published to guest memory when their GPU work completes. Make every
			// recorded dump visible before this CPU-side read.
			if (OcclusionCounter::Enabled() &&
			    m_renderer.GetOcclusionCounter().HasUnpublishedDumps()) {
				Profiler::ScopedGpuWaitReason wait_reason(Profiler::FrameWait::GpuWaitOcclusion);
				BufferWait();
				if (OcclusionCounter::PriorityPublication()) {
					// Publications run on the completion runner: wait for them too.
					GetScheduler().WaitPriorityOperations(GetScheduler().CurrentTick());
				}
			}
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t ready_bit = 1ull << 63u;
			const auto* results = reinterpret_cast<const volatile uint64_t*>(address);
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::OcclusionPredicatesPending);
					// Wait: the packet is retried. No wait: the predicate is cleared.
					return {wait_op == 0, 0};
				}
				value += end - begin;
			}
		} break;
		case 0x03:
			if (wait_op != 0 && !PredicationNoDrain()) {
				Profiler::ScopedGpuWaitReason wait_reason(Profiler::FrameWait::GpuWaitPredication);
				BufferFlushAndWait();
				// Labels deferred to completion (defer-label / KYTY_LABEL_MODE=completion) are
				// written by commands the completion runner posts: run them before reading.
				if (g_gpu_state != nullptr &&
				    g_gpu_state->DeferredLabelTick(reinterpret_cast<uint64_t>(address),
				                                   sizeof(uint64_t)) != 0) {
					GetScheduler().WaitPriorityOperations(GetScheduler().CurrentTick());
					g_gpu_state->ProcessCommands();
				}
			}
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			value = ReadGuestForCp<uint64_t>(reinterpret_cast<uint64_t>(address));
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	bool skip = false;
	switch (condition) {
		case 0x00: skip = (value != 0); break;
		case 0x01: skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
	if (op == 0x01 && HangTrace::Enabled()) {
		HangTrace::OcclusionEvent event;
		event.event     = "predicate";
		event.address   = reinterpret_cast<uint64_t>(address);
		event.value     = value;
		event.condition = condition;
		event.skip      = skip;
		HangTrace::RecordOcclusion(event);
	}
	if (op == 0x03) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 128) {
			LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
			     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
			     reinterpret_cast<uint64_t>(address), value, condition, skip ? 1u : 0u,
			     wait_op);
		}
	}
	return {false, skip ? 1u : 0u};
}

// The DrawIndexArgs / DrawAutoArgs of a draw op (what the direct path passes to the executor).
static DrawIndexArgs DrawArgsOf(const CpSeq::DrawIndexOp& op) {
	DrawIndexArgs args;
	args.index_count                = op.index_count;
	args.index_addr                 = reinterpret_cast<const void*>(op.index_addr);
	args.instance_count             = op.instance_count;
	args.index_type_and_size        = op.index_type_and_size;
	args.base_vertex                = op.base_vertex;
	args.first_instance             = op.first_instance;
	args.offset_source              = static_cast<DrawOffsetSource>(op.offset_source);
	args.render_target_slice_offset = op.render_target_slice_offset;
	return args;
}

static DrawAutoArgs DrawArgsOf(const CpSeq::DrawAutoOp& op) {
	DrawAutoArgs args;
	args.vertex_count               = op.vertex_count;
	args.instance_count             = op.instance_count;
	args.first_vertex               = op.first_vertex;
	args.first_instance             = op.first_instance;
	args.offset_source              = static_cast<DrawOffsetSource>(op.offset_source);
	args.render_target_slice_offset = op.render_target_slice_offset;
	return args;
}

// Thread mode: a direct draw's registers reach the resolver as a draw-prep window slot (parallel
// draw prep: the slot's snapshot, prepared by the workers meanwhile) or as a register snapshot.
template <typename Op, typename Args>
static void AttachDrawRegisters(Op& op, Args args, DrawPrep::Engine* engine,
                                const HW::Context& context, const HW::UserConfig& user_config,
                                const HW::Shader& shaders, const std::function<bool()>& wait,
                                const std::function<uint32_t()>& take_snapshot) {
	StateTrace::NoteDraw(); // Astro Bot PC : KYTY_STATE_TRACE
	if (engine != nullptr && engine->Parallel()) {
		if ((op.flags & CpSeq::DrawFlagInheritInstances) != 0) {
			// Resolved at commit; the preparation assumes the draw draws (an unused one is
			// dropped, DrawPrepUnused).
			args.instance_count = 1;
		}
		uint64_t position = UINT64_MAX;
		if constexpr (std::is_same_v<Args, DrawIndexArgs>) {
			position = engine->Publish(&args, nullptr, context, user_config, shaders, wait);
		} else {
			position = engine->Publish(nullptr, &args, context, user_config, shaders, wait);
		}
		if (position != UINT64_MAX) {
			op.flags |= CpSeq::DrawFlagPublished;
			op.window = position;
		}
		return;
	}
	op.flags |= CpSeq::DrawFlagSnapshot;
	op.snapshot = take_snapshot();
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	CpSeq::DrawIndexOp op;
	op.flags                      = m_native_op_tag; // Astro Bot PC (KYTY_NATIVE n2), 0 otherwise
	op.index_addr                 = reinterpret_cast<uint64_t>(args.index_addr);
	op.index_count                = args.index_count;
	op.instance_count             = args.instance_count;
	op.index_type_and_size        = m_index_type_and_size;
	op.base_vertex                = args.base_vertex;
	op.first_instance             = args.first_instance;
	op.render_target_slice_offset = args.render_target_slice_offset;
	op.offset_source              = static_cast<uint32_t>(args.offset_source);
	if (args.instance_count == 0) {
		if (m_front_mode != FrontMode::Direct && m_front_instances_known) {
			op.instance_count = m_front_num_instances;
		} else {
			op.flags |= CpSeq::DrawFlagInheritInstances;
		}
	}
	if (m_front_mode == FrontMode::Prefetch) {
		PrefetchDraw(op);
		return;
	}
	if (m_front_mode == FrontMode::Thread) {
		// P3c: the next speculative slot, or its drop (a SkipSlots op) before this draw's op.
		const bool adopted = AdoptSpeculativeDraw(op);
		const auto draw_op = m_emit_ops->Emitted(); // Astro Bot PC (KYTY_CP_PAR)
		if (!adopted) {
			AttachDrawRegisters(
			    op, DrawArgsOf(op), EmitPrep(), m_ctx, m_ucfg, m_sh_ctx,
			    [this] { return WaitForWindowSpace(); }, [this] { return TakeSnapshot(); });
		}
		if ((op.flags & CpSeq::DrawFlagPublished) != 0) {
			m_published_ops.push_back(draw_op);
			if (m_published_ops.size() > 256u) {
				(void)OldestPendingOp(m_published_ops);
			}
		}
	}
	(void)Submit(op);
}

// Astro Bot PC (KYTY_NATIVE_MODE=n2): a spliced native draw's op is timed on the resolver, with
// how the emulator's draw run treated it (read from the executor after the commit).
template <typename F>
struct NativeResolveTimer {
	uint32_t flags = 0;
	uint64_t start = 0;
	F        done;
	NativeResolveTimer(uint32_t op_flags, F f): flags(op_flags & CpSeq::DrawFlagNativeMask), done(f) {
		if (flags != 0) [[unlikely]] {
			start = __rdtsc();
		}
	}
	~NativeResolveTimer() {
		if (flags != 0) [[unlikely]] {
			done(flags, start);
		}
	}
	NativeResolveTimer(const NativeResolveTimer&)            = delete;
	NativeResolveTimer& operator=(const NativeResolveTimer&) = delete;
};

void CommandProcessor::NoteNativeResolvedOp(uint32_t flags, uint64_t start) {
	const uint64_t end = __rdtsc();
	auto&          ex  = m_renderer.GetRenderExecutor();
	namespace GC       = Loader::GameCapture;
	// S1: the op's draw was emitted natively (read once: a later op may not reach DrawIndex).
	const bool     s1  = std::exchange(ex.m_native_s1_done, false);
	const uint32_t run = s1 ? GC::kNrNative : ex.m_run_active ? (ex.m_run_recall     ? GC::kNrRecalled
	                                        : ex.m_run_pipeswap ? GC::kNrSwapped
	                                                            : GC::kNrContinued)
	                     : ex.m_run_targets_kept ? GC::kNrTargets
	                                             : GC::kNrStarted;
	// N3b: a continuation committed on the lean path is reported as its own class.
	const bool lean_op = std::exchange(m_native_lean_op, false);
	GC::NoteNativeResolved(flags, start, end, lean_op && ex.m_native_lean_done ? GC::kNrLean : run);
	if (GC::NativeLean()) {
		// A draw that passed every lean test but returned before its record step did not complete.
		const uint32_t reason = !lean_op                                ? GC::kNlOff
		                        : ex.m_native_lean_done                 ? GC::kNlLean
		                        : ex.m_native_lean_reason == GC::kNlLean ? GC::kNlEarly
		                                                                : ex.m_native_lean_reason;
		GC::NoteNativeLean(flags, reason, lean_op && ex.m_native_lean_checked, lean_op && ex.m_native_lean_mismatch);
	}
	if (m_native_tier_valid) {
		m_native_tier_valid = false;
		GC::NoteNativeN3(flags, start, end, m_native_tier_draw);
	}
}

// Astro Bot PC (KYTY_NATIVE_MODE=n3b, stage N3b): the native op flags handed to the executor for
// the commit of the op's own published slot (exactly this draw), so that it may take the lean path
// (RenderExecutor::NativeLeanPrecheck); 0 for every other draw it commits.
struct NativeLeanScope {
	RenderExecutor* ex = nullptr;
	NativeLeanScope(RenderExecutor& e, uint32_t op_flags, bool& lean_op) {
		namespace GC = Loader::GameCapture;
		lean_op      = false;
		if ((op_flags & CpSeq::DrawFlagNativeMask) != 0 && GC::NativeLean()) [[unlikely]] {
			ex            = &e;
			lean_op       = true;
			e.m_native_op = op_flags & GC::kNativeOpMask;
		}
	}
	~NativeLeanScope() {
		if (ex != nullptr) [[unlikely]] {
			ex->m_native_op = 0;
		}
	}
	NativeLeanScope(const NativeLeanScope&)            = delete;
	NativeLeanScope& operator=(const NativeLeanScope&) = delete;
};

// Astro Bot PC (KYTY_NATIVE_MODE=n3, stage N3a): the buffer cache's segment resolution windows
// (BufferCache::NativeTierBegin) for the duration of a native draw op. Declared after the
// NativeResolveTimer, so it ends first and the timer's report sees what the windows did.
struct NativeTierScope {
	BufferCache*                       cache = nullptr;
	Loader::GameCapture::NativeN3Draw* out   = nullptr;
	bool*                              valid = nullptr;
	NativeTierScope(BufferCache& c, uint32_t op_flags, Loader::GameCapture::NativeN3Draw& draw, bool& draw_valid) {
		namespace GC = Loader::GameCapture;
		if ((op_flags & CpSeq::DrawFlagNativeMask) != 0 && GC::GetNativeMode() == GC::NativeMode::N3) [[unlikely]] {
			cache = &c;
			out   = &draw;
			valid = &draw_valid;
			c.NativeTierBegin((op_flags >> GC::kNativeOpSlotShift) & GC::kNativeSlotMask,
			                  (op_flags & GC::kNativeOpSegStart) != 0, GC::NativeVerifyEnabled());
		}
	}
	~NativeTierScope() {
		if (cache != nullptr) [[unlikely]] {
			*out   = cache->NativeTierEnd();
			*valid = true;
		}
	}
	NativeTierScope(const NativeTierScope&)            = delete;
	NativeTierScope& operator=(const NativeTierScope&) = delete;
};

// Astro Bot PC (KYTY_NATIVE_MODE=s0, renderer/nativeLearn.h): the native op (slot, serial) handed to
// the executor for the whole op, so that the learning hook of the op's own draw finds its record;
// the op's end is reported (a draw that never reached the hook is counted as skipped).
struct NativeS0Scope {
	RenderExecutor* ex    = nullptr;
	uint32_t        flags = 0;
	NativeS0Scope(RenderExecutor& e, uint32_t op_flags) {
		namespace GC = Loader::GameCapture;
		if ((op_flags & CpSeq::DrawFlagNativeMask) != 0 && GC::NativeS0()) [[unlikely]] {
			ex              = &e;
			flags           = op_flags & (GC::kNativeOpMask | (0xffffu << GC::kNativeOpSerialShift));
			e.m_native_s0_op = flags;
		}
	}
	~NativeS0Scope() {
		if (ex != nullptr) [[unlikely]] {
			ex->NativeS0EndOp(flags);
		}
	}
	NativeS0Scope(const NativeS0Scope&)            = delete;
	NativeS0Scope& operator=(const NativeS0Scope&) = delete;
};

// Astro Bot PC (KYTY_BINDING_CENSUS, bindingCensus.h): the engine thread slot of the native draw
// op being executed, for the census's per-slot binding time (bindings resolved later, outside the
// op, count as non-native).
struct CensusSlotScope {
	bool on = false;
	explicit CensusSlotScope(uint32_t op_flags) {
		if (BindingCensus::Enabled()) [[unlikely]] {
			on = true;
			namespace GC = Loader::GameCapture;
			BindingCensus::SetNativeSlot(
			    (op_flags & CpSeq::DrawFlagNativeMask) != 0
			        ? static_cast<uint8_t>(((op_flags >> GC::kNativeOpSlotShift) & GC::kNativeSlotMask) + 1u)
			        : uint8_t {0});
		}
	}
	~CensusSlotScope() {
		if (on) [[unlikely]] {
			BindingCensus::SetNativeSlot(0);
		}
	}
	CensusSlotScope(const CensusSlotScope&)            = delete;
	CensusSlotScope& operator=(const CensusSlotScope&) = delete;
};

void CommandProcessor::ExecDrawIndex(const CpSeq::DrawIndexOp& op) {
	const CensusSlotScope census_slot(op.flags);
	const NativeResolveTimer native_timer(op.flags, [this](uint32_t flags, uint64_t start) {
		NoteNativeResolvedOp(flags, start);
	});
	const NativeTierScope native_tier(m_renderer.GetBufferCache(), op.flags, m_native_tier_draw,
	                                  m_native_tier_valid);
	const NativeS0Scope native_s0(m_renderer.GetRenderExecutor(), op.flags);
	auto args = DrawArgsOf(op);
	if ((op.flags & CpSeq::DrawFlagInheritInstances) != 0) {
		if (!m_pending_num_instances.empty()) {
			// The inherited count may be GPU data an earlier (pending) draw writes.
			DrainPreparedDraws();
		}
		args.instance_count = NumInstances();
	}
	if ((op.flags & CpSeq::DrawFlagPublished) != 0) {
		// Thread mode: the sequencer published the draw; commit its prepared slot now, in order.
		const NativeLeanScope native_lean(m_renderer.GetRenderExecutor(), op.flags, m_native_lean_op);
		ResolvePrep()->CommitPublished(
		    op.window, LaneCp().submit_id,
		    (op.flags & CpSeq::DrawFlagInheritInstances) != 0 ? args.instance_count : UINT32_MAX);
		MaybeYieldSlice();
		CommitStats::StepMark(CommitStats::Step::GapRetire);
		return;
	}
	if (GraphicsRunDebugDumpEnabled() && (args.base_vertex != 0 || args.first_instance != 0)) {
		LOGF("\t draw indexed offsets: base_vertex = %" PRId32 ", first_instance = %" PRIu32 "\n",
		     args.base_vertex, args.first_instance);
	}
	if (RepeatTrace::Enabled()) {
		NoteRepeatTraceDrawPacket();
	}
	if (TrySubmitPreparedDraw(&args, nullptr)) {
		// Draw-prep: the engine runs MaybeFlushIdleGpu after it records (commits) each draw. The
		// slice may count the draw now: a yield ends the slice after this packet, and the slice
		// end commits the whole window before anything else runs.
		MaybeYieldSlice();
		return;
	}
	if (RepeatTrace::Enabled()) {
		(void)RepeatTrace::TakeDrawPacket();
		RepeatTrace::OnUnpreparedDraw();
	}
	m_renderer.GetRenderExecutor().DrawIndex(LaneCp().submit_id, CurrentBuffer(), args);
	MaybeFlushIdleGpu();
	MaybeYieldSlice();
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(index_offset) * index_size);

	DrawIndex({.index_count = index_count, .index_addr = index_addr});
}

uint32_t CommandProcessor::NumInstances() {
	if (m_pending_num_instances.empty()) {
		return m_num_instances;
	}
	// A draw without its own count inherits instance_count of the last record a native indirect
	// draw drew. Reading it synchronizes (drains) as the CPU-read path would have at the
	// indirect draw, but reads the bytes now: a record rewritten in between yields its newer
	// value (KYTY_INDIRECT_VALIDATE=1 reports that case).
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectInstanceReads);
	for (auto it = m_pending_num_instances.rbegin(); it != m_pending_num_instances.rend(); ++it) {
		const auto& pending = *it;
		uint32_t    count   = pending.max_count;
		if (pending.count_addr != 0) {
			if (!m_renderer.IsMapped(pending.count_addr, sizeof(uint32_t))) {
				break; // Unmapped since: keep the last known count.
			}
			count = std::min(ReadGuestForCp<uint32_t>(pending.count_addr), pending.max_count);
		}
		if (count == 0) {
			// That draw drew nothing and left the count unchanged: an older source decides.
			continue;
		}
		const auto address = pending.args_addr +
		                     static_cast<uint64_t>(count - 1u) * pending.stride +
		                     IndirectInstanceCountOffset;
		if (!m_renderer.IsMapped(address, sizeof(uint32_t))) {
			break;
		}
		const auto instances = ReadGuestForCp<uint32_t>(address);
		if (pending.has_expected && pending.expected != instances) {
			LOGF("IndirectValidate: inherited instance count changed after the native draw: "
			     "args=0x%016" PRIx64 " at_draw=%u now=%u\n",
			     pending.args_addr, pending.expected, instances);
		}
		m_num_instances = instances;
		break;
	}
	m_pending_num_instances.clear();
	return m_num_instances;
}

void CommandProcessor::ValidateIndirectSource(const DrawIndirectSource& source) {
	static std::atomic<uint32_t> log_count {0};
	const auto log_enabled = [] { return log_count.fetch_add(1, std::memory_order_relaxed) < 256; };
	uint32_t   count       = source.max_count;
	if (source.count_addr != 0) {
		count = std::min(ReadGuestForCp<uint32_t>(source.count_addr), source.max_count);
	}
	if (log_enabled()) {
		LOGF("IndirectValidate: native %s draw args=0x%016" PRIx64 " stride=%u count=%u/%u\n",
		     source.indexed ? "indexed" : "auto", source.args_addr, source.stride, count,
		     source.max_count);
	}
	if (!source.indexed) {
		return;
	}
	for (uint32_t i = 0; i < count; i++) {
		const auto record = source.args_addr + static_cast<uint64_t>(i) * source.stride;
		const auto args   = ReadGuestForCp<DrawIndexedIndirectArgs>(record);
		// The CPU path clamps the index count to INDEX_BUFFER_SIZE (ignoring firstIndex); the
		// native draw reads past the bound range instead.
		const auto end =
		    static_cast<uint64_t>(args.start_index_location) + args.index_count_per_instance;
		if (args.instance_count != 0 && args.index_count_per_instance != 0 &&
		    end > source.index_buffer_size && log_enabled()) {
			LOGF("IndirectValidate: record %u/%u args=0x%016" PRIx64 " indices [%u, +%u) exceed "
			     "INDEX_BUFFER_SIZE=%u (the CPU-read draw uses %u indices)\n",
			     i, count, record, args.start_index_location, args.index_count_per_instance,
			     source.index_buffer_size,
			     std::min(args.index_count_per_instance, source.index_buffer_size));
		}
	}
}

// Takes the host indirect path when the argument (or count) bytes are GPU-owned, i.e. when a CPU
// read would page-fault and drain the GPU. CPU-clean arguments stay on the CPU path, which then
// reads them without synchronization.
bool CommandProcessor::TryDrawIndirectNative(DrawIndirectSource source) {
	if (!NativeIndirectEnabled() || !GuestRange {source.args_addr, source.ArgsSize()}.Valid() ||
	    (source.count_addr != 0 && !GuestRange {source.count_addr, sizeof(uint32_t)}.Valid())) {
		return false;
	}
	auto&      cache     = m_renderer.GetBufferCache();
	const auto gpu_owned = [&cache](uint64_t address, uint64_t size) {
		return cache.HasGpuDirtyBytes(address, size) ||
		       cache.HasPendingBackingPublication(address, size);
	};
	if (!gpu_owned(source.args_addr, source.ArgsSize()) &&
	    (source.count_addr == 0 || !gpu_owned(source.count_addr, sizeof(uint32_t)))) {
		return false;
	}
	if (!source.indexed) {
		source.index_base_addr     = 0;
		source.index_buffer_size   = 0;
		source.index_type_and_size = 0;
	}

	PendingNumInstances pending {.args_addr  = source.args_addr,
	                             .stride     = source.stride,
	                             .max_count  = source.max_count,
	                             .count_addr = source.count_addr};
	if (IndirectValidateEnabled()) {
		ValidateIndirectSource(source);
		uint32_t count = source.max_count;
		if (source.count_addr != 0) {
			count = std::min(ReadGuestForCp<uint32_t>(source.count_addr), source.max_count);
		}
		if (count != 0) {
			pending.has_expected = true;
			pending.expected     = ReadGuestForCp<uint32_t>(
			    source.args_addr + static_cast<uint64_t>(count - 1u) * source.stride +
			    IndirectInstanceCountOffset);
		}
	}

	if (!m_renderer.GetRenderExecutor().DrawIndirectNative(LaneCp().submit_id, CurrentBuffer(), source)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallback);
		return false;
	}
	if (source.count_addr == 0) {
		// At least one record was drawn: older sources can no longer decide.
		m_pending_num_instances.clear();
	} else if (m_pending_num_instances.size() >= 64) {
		(void)NumInstances();
	}
	m_pending_num_instances.push_back(pending);
	if (RepeatTrace::Enabled()) {
		RepeatTrace::OnUnpreparedDraw();
	}
	MaybeFlushIdleGpu();
	MaybeYieldSlice();
	return true;
}

CpSeq::DrawIndirectOp CommandProcessor::IndirectDrawOp(uint32_t data_offset,
                                                      uint32_t draw_initiator, bool indexed) {
	// Indexed records select the DMA index source (0), auto-index records generated indices (2).
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != (indexed ? 0u : 2u));
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);
	CpSeq::DrawIndirectOp op;
	op.args_base           = m_draw_indirect_args_base_addr;
	op.index_base_addr     = m_index_base_addr;
	op.index_buffer_size   = m_index_buffer_size;
	op.index_type_and_size = m_index_type_and_size;
	op.data_offset         = data_offset;
	op.draw_initiator      = draw_initiator;
	op.flags               = indexed ? CpSeq::IndirectFlagIndexed : 0u;
	if (m_front_mode != FrontMode::Direct) {
		if (m_front_instances_known) {
			// The back's count still predates the last SET_NUM_INSTANCES: apply it first.
			op.flags |= CpSeq::IndirectFlagSetInstances;
			op.num_instances = m_front_num_instances;
		}
		// From here on the count is back state (read from the arguments, maybe GPU data).
		m_front_instances_known = false;
	}
	if (m_front_mode == FrontMode::Thread) {
		op.flags |= CpSeq::IndirectFlagSnapshot;
		op.snapshot = TakeSnapshot();
	}
	return op;
}

// An indirect draw record read on the CPU, drawn as the direct path's DrawIndexAuto/DrawIndex:
// a zero instance count takes the back's instance state (NumInstances), which the draw's own
// record has just set.
static CpSeq::DrawAutoOp CpuIndirectAutoDraw(uint32_t vertex_count, uint32_t instance_count,
                                             uint32_t first_vertex, uint32_t first_instance) {
	CpSeq::DrawAutoOp draw;
	draw.vertex_count   = vertex_count;
	draw.instance_count = instance_count;
	draw.first_vertex   = first_vertex;
	draw.first_instance = first_instance;
	draw.offset_source  = static_cast<uint32_t>(DrawOffsetSource::IndirectArgs);
	draw.flags          = instance_count == 0 ? CpSeq::DrawFlagInheritInstances : 0u;
	return draw;
}

static CpSeq::DrawIndexOp CpuIndirectIndexDraw(uint64_t index_addr, uint32_t index_count,
                                               uint32_t instance_count, int32_t base_vertex,
                                               uint32_t first_instance,
                                               uint32_t index_type_and_size) {
	CpSeq::DrawIndexOp draw;
	draw.index_addr          = index_addr;
	draw.index_count         = index_count;
	draw.instance_count      = instance_count;
	draw.index_type_and_size = index_type_and_size;
	draw.base_vertex         = base_vertex;
	draw.first_instance      = first_instance;
	draw.offset_source       = static_cast<uint32_t>(DrawOffsetSource::IndirectArgs);
	draw.flags               = instance_count == 0 ? CpSeq::DrawFlagInheritInstances : 0u;
	return draw;
}

// `registers` (the SH locations the CP patches with a record's offsets) is not applied: native
// (GPU-read) records never reach the CPU, and the start instance reaches its SGPR in the shader
// (KYTY_START_INSTANCE_SGPR, recorded by the PM4 handler).
void CommandProcessor::DrawIndirect(uint32_t data_offset,
                                    [[maybe_unused]] IndirectDrawRegisters registers,
                                    uint32_t draw_initiator, bool indexed) {
	// KYTY_DRAW_PREP_INDIRECT (drawPrep.h): the draw's programs are prepared in a window slot.
	auto*      engine  = EmitPrep();
	const bool prepare = m_front_mode == FrontMode::Thread && engine != nullptr &&
	                     engine->Parallel() && DrawPrep::Engine::IndirectEnabled();
	if (prepare && engine->SpeculativeSlots() != 0) {
		// The slot must follow the real parse's (a SkipSlots op retires the speculative ones).
		DropSpeculativeSlots();
	}
	auto op = IndirectDrawOp(data_offset, draw_initiator, indexed);
	if (prepare) {
		const auto position = engine->PublishIndirect(indexed, m_ctx, m_ucfg, m_sh_ctx,
		                                              [this] { return WaitForWindowSpace(); });
		if (position != UINT64_MAX) {
			op.flags |= CpSeq::IndirectFlagPrepared;
			op.window = position;
			m_published_ops.push_back(m_emit_ops->Emitted()); // Astro Bot PC (KYTY_CP_PAR)
			if (m_published_ops.size() > 256u) {
				(void)OldestPendingOp(m_published_ops);
			}
		}
	}
	(void)Submit(CpSeq::OpKind::DrawIndirect, &op, sizeof(op));
}

// KYTY_CP_OP_STATS: an Exec* body may name its outcome for the op being timed (ExecuteOp).
static const char* g_cp_op_outcome = nullptr;

void CommandProcessor::ExecDrawIndirect(const CpSeq::DrawIndirectOp& op) {
	if ((op.flags & CpSeq::IndirectFlagPrepared) != 0) {
		// KYTY_DRAW_PREP_INDIRECT: the same execution, with the slot's preparation offered.
		auto plain = op;
		plain.flags &= ~CpSeq::IndirectFlagPrepared;
		ResolvePrep()->ExecuteIndirect(op.window, [this, &plain] { ExecDrawIndirect(plain); });
		return;
	}
	if ((op.flags & CpSeq::IndirectFlagSetInstances) != 0) {
		m_num_instances = op.num_instances;
		m_pending_num_instances.clear();
	}
	const bool indexed     = (op.flags & CpSeq::IndirectFlagIndexed) != 0;
	const auto args_addr   = op.args_base + op.data_offset;
	const auto record_size = static_cast<uint32_t>(indexed ? sizeof(DrawIndexedIndirectArgs)
	                                                       : sizeof(DrawIndirectArgs));
	if (TryDrawIndirectNative({.args_addr           = args_addr,
	                           .stride              = record_size,
	                           .max_count           = 1,
	                           .count_addr          = 0,
	                           .indexed             = indexed,
	                           .index_base_addr     = op.index_base_addr,
	                           .index_buffer_size   = op.index_buffer_size,
	                           .index_type_and_size = op.index_type_and_size})) {
		g_cp_op_outcome = "DrawIndirect native";
		return;
	}
	g_cp_op_outcome = indexed ? "DrawIndirect cpu indexed" : "DrawIndirect cpu auto";
	// The CPU path reads the (GPU-written) arguments: other command-processor work for a run.
	if (DrawRun::Enabled()) {
		DrawRun::NoteForeignActivity("DrawIndirect cpu");
	}

	m_pending_num_instances.clear();
	if (!indexed) {
		const auto args = ReadGuestForCp<DrawIndirectArgs>(args_addr);
		m_num_instances = args.instance_count;
		ExecDrawAuto(CpuIndirectAutoDraw(args.vertex_count_per_instance, args.instance_count,
		                                 args.start_vertex_location,
		                                 args.start_instance_location));
		return;
	}

	const auto args       = ReadGuestForCp<DrawIndexedIndirectArgs>(args_addr);
	const auto index_size = IndexElementSize(op.index_type_and_size);

	const auto index_addr =
	    op.index_base_addr + static_cast<uint64_t>(args.start_index_location) * index_size;

	const uint32_t index_count =
	    (op.index_buffer_size != 0 ? std::min(args.index_count_per_instance, op.index_buffer_size)
	                               : args.index_count_per_instance);
	if (GraphicsRunDebugDumpEnabled() && index_count != args.index_count_per_instance) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
			LOGF("\t DrawIndexIndirect: clamped index_count from %" PRIu32 " to %" PRIu32
			     " using INDEX_BUFFER_SIZE\n",
			     args.index_count_per_instance, index_count);
		}
	}

	m_num_instances = args.instance_count;
	ExecDrawIndex(CpuIndirectIndexDraw(index_addr, index_count, args.instance_count,
	                                   static_cast<int32_t>(args.base_vertex_location),
	                                   args.start_instance_location, op.index_type_and_size));
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t                 stride_in_bytes,
                                         [[maybe_unused]] IndirectDrawRegisters registers,
                                         uint32_t draw_initiator, bool indexed,
                                         int32_t draw_index_sgpr) {
	auto op               = IndirectDrawOp(data_offset, draw_initiator, indexed);
	op.count_addr         = reinterpret_cast<uint64_t>(count_addr);
	op.max_count_or_count = max_count_or_count;
	op.stride             = stride_in_bytes;
	if (draw_index_sgpr >= 0 && IndirectDrawIndexEnabled()) {
		op.draw_index_sgpr = static_cast<uint32_t>(draw_index_sgpr) + 1u;
	}
	(void)Submit(CpSeq::OpKind::DrawIndirectMulti, &op, sizeof(op));
}

// The CP's DRAW_INDEX_LOC write: `index` into the bound first stage's user SGPR (1 + GS user SGPR,
// or 1 + 32 + HS user SGPR). The register keeps the value, as on the GPU.
static void WriteDrawIndexSgpr(HW::Shader* shaders, uint32_t encoded_sgpr, uint32_t index) {
	if (shaders == nullptr || encoded_sgpr == 0) {
		return;
	}
	const auto sgpr = encoded_sgpr - 1u;
	if (sgpr < 32u) {
		shaders->SetGsUserSgpr(sgpr, index, shaders->GetVs().gs_user_sgpr.type[sgpr]);
	} else if (sgpr - 32u < 32u) {
		shaders->SetHsUserSgpr(sgpr - 32u, index, shaders->GetVs().hs_user_sgpr.type[sgpr - 32u]);
	}
}

void CommandProcessor::ExecDrawIndirectMulti(const CpSeq::DrawIndirectOp& op) {
	if ((op.flags & CpSeq::IndirectFlagSetInstances) != 0) {
		m_num_instances = op.num_instances;
		m_pending_num_instances.clear();
	}
	const bool indexed            = (op.flags & CpSeq::IndirectFlagIndexed) != 0;
	const auto max_count_or_count = op.max_count_or_count;
	const auto stride_in_bytes    = op.stride;
	const auto count_addr         = op.count_addr;

	// Zero records draw nothing and leave the instance count unchanged on either path.
	if (max_count_or_count == 0) {
		return;
	}

	// DRAW_INDEX_ENABLE: each record gets its index in that SGPR. The host path draws every record
	// with the registers as they are, so it is right for a single record only.
	HW::Shader* const draw_index_shaders =
	    op.draw_index_sgpr != 0 ? GetScheduler().BoundShaders() : nullptr;
	if (draw_index_shaders != nullptr) {
		WriteDrawIndexSgpr(draw_index_shaders, op.draw_index_sgpr, 0);
		static std::atomic<bool> reported {false};
		if (!reported.exchange(true, std::memory_order_relaxed)) {
			LOGF("Kyty: indirect multi-draw writes the draw index to user SGPR %u "
			     "(KYTY_INDIRECT_DRAW_INDEX=0 turns it off)\n",
			     op.draw_index_sgpr - 1u);
		}
	}

	// The renderer rejects strides shorter than a record; the CPU path below reports them.
	const auto args_base = op.args_base + op.data_offset;
	if ((draw_index_shaders == nullptr || max_count_or_count == 1) &&
	    TryDrawIndirectNative({.args_addr           = args_base,
	                           .stride              = stride_in_bytes,
	                           .max_count           = max_count_or_count,
	                           .count_addr          = count_addr,
	                           .indexed             = indexed,
	                           .index_base_addr     = op.index_base_addr,
	                           .index_buffer_size   = op.index_buffer_size,
	                           .index_type_and_size = op.index_type_and_size})) {
		return;
	}

	uint32_t draw_count = max_count_or_count;
	if (count_addr != 0) {
		draw_count = ReadGuestForCp<uint32_t>(count_addr);
		if (draw_count > max_count_or_count) {
			draw_count = max_count_or_count;
		}
	}

	if (draw_count == 0) {
		return;
	}
	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	EXIT_NOT_IMPLEMENTED(stride_in_bytes < args_size);
	// Every drawn record sets the count, so pending native sources cannot decide any more.
	m_pending_num_instances.clear();

	const uint64_t index_size = indexed ? IndexElementSize(op.index_type_and_size) : 0;

	for (uint32_t i = 0; i < draw_count; i++) {
		const auto args_addr = args_base + static_cast<uint64_t>(i) * stride_in_bytes;
		WriteDrawIndexSgpr(draw_index_shaders, op.draw_index_sgpr, i);

		if (!indexed) {
			const auto args = ReadGuestForCp<DrawIndirectArgs>(args_addr);
			m_num_instances = args.instance_count;
			ExecDrawAuto(CpuIndirectAutoDraw(args.vertex_count_per_instance, args.instance_count,
			                                 args.start_vertex_location,
			                                 args.start_instance_location));
			continue;
		}

		const auto args = ReadGuestForCp<DrawIndexedIndirectArgs>(args_addr);

		const auto index_addr =
		    op.index_base_addr + static_cast<uint64_t>(args.start_index_location) * index_size;

		const uint32_t index_count =
		    (op.index_buffer_size != 0
		         ? std::min(args.index_count_per_instance, op.index_buffer_size)
		         : args.index_count_per_instance);
		if (GraphicsRunDebugDumpEnabled() && index_count != args.index_count_per_instance) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("\t DrawIndexIndirectMulti: clamped index_count from %" PRIu32 " to %" PRIu32
				     " using INDEX_BUFFER_SIZE\n",
				     args.index_count_per_instance, index_count);
			}
		}

		m_num_instances = args.instance_count;
		ExecDrawIndex(CpuIndirectIndexDraw(index_addr, index_count, args.instance_count,
		                                   static_cast<int32_t>(args.base_vertex_location),
		                                   args.start_instance_location, op.index_type_and_size));
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	// The wave size is register state: the front sets it, the op's execution reads it.
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	// KYTY_DRAW_PREP_DISPATCH (drawPrep.h): the dispatch's program is prepared in a window slot.
	auto*      engine  = EmitPrep();
	const bool prepare = m_front_mode == FrontMode::Thread && engine != nullptr &&
	                     engine->Parallel() && DrawPrep::DispatchPrepEnabled();
	if (prepare && engine->SpeculativeSlots() != 0) {
		// The slot must follow the real parse's (a SkipSlots op retires the speculative ones).
		DropSpeculativeSlots();
	}
	CpSeq::DispatchDirectOp op;
	op.x    = thread_group_x;
	op.y    = thread_group_y;
	op.z    = thread_group_z;
	op.mode = mode;
	if (m_front_mode == FrontMode::Thread) {
		op.flags |= CpSeq::DispatchFlagSnapshot;
		op.snapshot = TakeSnapshot();
	}
	if (prepare) {
		const auto position = engine->PublishDispatch(
		    {thread_group_x, thread_group_y, thread_group_z, mode}, m_ctx, m_ucfg, m_sh_ctx,
		    [this] { return WaitForWindowSpace(); });
		if (position != UINT64_MAX) {
			op.flags |= CpSeq::DispatchFlagPrepared;
			op.window = position;
			m_published_ops.push_back(m_emit_ops->Emitted()); // Astro Bot PC (KYTY_CP_PAR)
			if (m_published_ops.size() > 256u) {
				(void)OldestPendingOp(m_published_ops);
			}
		}
	}
	(void)Submit(op);
}

void CommandProcessor::ExecDispatchDirect(const CpSeq::DispatchDirectOp& op) {
	if ((op.flags & CpSeq::DispatchFlagPrepared) != 0) {
		// KYTY_DRAW_PREP_DISPATCH: the same execution, with the slot's preparation offered.
		auto plain = op;
		plain.flags &= ~CpSeq::DispatchFlagPrepared;
		ResolvePrep()->ExecuteDispatch(op.window, [this, &plain] { ExecDispatchDirect(plain); });
		return;
	}
	const auto thread_group_x = op.x;
	const auto thread_group_y = op.y;
	const auto thread_group_z = op.z;
	const auto mode           = op.mode;
	// The registers the command buffer reads (the front's, bound by BufferInit).
	auto&       command = CurrentBuffer();
	const auto& shaders = command.GetShaders();

	uint32_t frame_num = 0;

	{
		frame_num = m_renderer.GetGpu().GetFrameNum();
		if (GraphicsRunDebugDumpEnabled()) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
				const auto& user_config = command.GetUserConfig();
				const auto& cs          = shaders.GetCs().cs_regs;
				const auto& oa =
				    user_config.GetGdsOaCounter(user_config.GetGdsOaState().GetIndex());
				LOGF("QueuePoint DispatchDirect: frame=%u submit=%" PRIu64
				     " groups=%ux%ux%u local=%ux%ux%u mode=0x%08" PRIx32 " wave=%u cs=0x%016" PRIx64
				     " oa_index=%u oa_enabled=%s oa_addr=0x%04" PRIx32 " oa_space=0x%08" PRIx32
				     "\n",
				     frame_num, LaneCp().submit_id, thread_group_x, thread_group_y, thread_group_z,
				     std::max(cs.num_thread_x, 1u), std::max(cs.num_thread_y, 1u),
				     std::max(cs.num_thread_z, 1u), mode, static_cast<uint32_t>(cs.wave_size),
				     cs.data_addr, user_config.GetGdsOaState().GetIndex(),
				     oa.IsCounterEnabled() ? "true" : "false", oa.GetAddressBytes(),
				     oa.GetSpaceAvailable());
			}
		}

		if (RepeatTrace::Enabled()) {
			RepeatTrace::OnDispatch(static_cast<uint32_t>(m_interrupt_event_id), shaders.GetCs(),
			                        thread_group_x, thread_group_y, thread_group_z, mode);
		}
		m_renderer.GetRenderExecutor().DispatchDirect(LaneCp().submit_id, command, thread_group_x,
		                                              thread_group_y, thread_group_z, mode,
		                                              IsAsyncComputeQueue());
		MaybeFlushIdleGpu();
	}
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	// KYTY_DRAW_PREP_DISPATCH: the program of a native indirect dispatch does not depend on its
	// (GPU-written) arguments. One with thread dimensions reads them on the CPU and runs as a direct
	// dispatch, unprepared.
	auto*      engine  = EmitPrep();
	const bool prepare = m_front_mode == FrontMode::Thread && engine != nullptr &&
	                     engine->Parallel() && DrawPrep::DispatchPrepEnabled() &&
	                     (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) == 0;
	if (prepare && engine->SpeculativeSlots() != 0) {
		DropSpeculativeSlots();
	}
	CpSeq::DispatchIndirectOp op;
	op.args_addr = args_addr;
	op.mode      = mode;
	if (m_front_mode == FrontMode::Thread) {
		op.flags |= CpSeq::DispatchFlagSnapshot;
		op.snapshot = TakeSnapshot();
	}
	if (prepare) {
		const auto position = engine->PublishDispatch({0, 0, 0, mode, true}, m_ctx, m_ucfg, m_sh_ctx,
		                                              [this] { return WaitForWindowSpace(); });
		if (position != UINT64_MAX) {
			op.flags |= CpSeq::DispatchFlagPrepared;
			op.window = position;
			m_published_ops.push_back(m_emit_ops->Emitted()); // Astro Bot PC (KYTY_CP_PAR)
			if (m_published_ops.size() > 256u) {
				(void)OldestPendingOp(m_published_ops);
			}
		}
	}
	(void)Submit(op);
}

void CommandProcessor::ExecDispatchIndirect(const CpSeq::DispatchIndirectOp& op) {
	if ((op.flags & CpSeq::DispatchFlagPrepared) != 0) {
		// KYTY_DRAW_PREP_DISPATCH: the same execution, with the slot's preparation offered.
		auto plain = op;
		plain.flags &= ~CpSeq::DispatchFlagPrepared;
		ResolvePrep()->ExecuteDispatch(op.window, [this, &plain] { ExecDispatchIndirect(plain); });
		return;
	}
	if ((op.mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		const auto args = ReadGuestForCp<vk::DispatchIndirectCommand>(op.args_addr);
		CpSeq::DispatchDirectOp direct;
		direct.x    = args.x;
		direct.y    = args.y;
		direct.z    = args.z;
		direct.mode = op.mode; // (thread mode: the op's snapshot is bound)
		ExecDispatchDirect(direct);
		return;
	}
	m_renderer.GetRenderExecutor().DispatchIndirect(LaneCp().submit_id, CurrentBuffer(), op.args_addr,
	                                                op.mode, IsAsyncComputeQueue());
	MaybeFlushIdleGpu();
}

void CommandProcessor::ReportLodStats(uint64_t destination, uint32_t size, uint32_t control) {
	(void)CurrentBuffer(); // the report is recorded at this command position
	RenderLockGuard lock(m_renderer.GetMutex());
	m_renderer.GetLodStats().Report(destination, size, control);
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	CpSeq::DrawAutoOp op;
	op.flags                      = m_native_op_tag; // Astro Bot PC (KYTY_NATIVE n2), 0 otherwise
	op.vertex_count               = args.vertex_count;
	op.instance_count             = args.instance_count;
	op.first_vertex               = args.first_vertex;
	op.first_instance             = args.first_instance;
	op.offset_source              = static_cast<uint32_t>(args.offset_source);
	op.render_target_slice_offset = args.render_target_slice_offset;
	if (args.instance_count == 0) {
		if (m_front_mode != FrontMode::Direct && m_front_instances_known) {
			op.instance_count = m_front_num_instances;
		} else {
			op.flags |= CpSeq::DrawFlagInheritInstances;
		}
	}
	if (m_front_mode == FrontMode::Prefetch) {
		PrefetchDraw(op);
		return;
	}
	if (m_front_mode == FrontMode::Thread) {
		// P3c: see DrawIndex.
		const bool adopted = AdoptSpeculativeDraw(op);
		const auto draw_op = m_emit_ops->Emitted(); // Astro Bot PC (KYTY_CP_PAR)
		if (!adopted) {
			AttachDrawRegisters(
			    op, DrawArgsOf(op), EmitPrep(), m_ctx, m_ucfg, m_sh_ctx,
			    [this] { return WaitForWindowSpace(); }, [this] { return TakeSnapshot(); });
		}
		if ((op.flags & CpSeq::DrawFlagPublished) != 0) {
			m_published_ops.push_back(draw_op);
			if (m_published_ops.size() > 256u) {
				(void)OldestPendingOp(m_published_ops);
			}
		}
	}
	(void)Submit(op);
}

void CommandProcessor::ExecDrawAuto(const CpSeq::DrawAutoOp& op) {
	const CensusSlotScope census_slot(op.flags);
	const NativeResolveTimer native_timer(op.flags, [this](uint32_t flags, uint64_t start) {
		NoteNativeResolvedOp(flags, start);
	});
	const NativeTierScope native_tier(m_renderer.GetBufferCache(), op.flags, m_native_tier_draw,
	                                  m_native_tier_valid);
	const NativeS0Scope native_s0(m_renderer.GetRenderExecutor(), op.flags);
	auto args = DrawArgsOf(op);
	if ((op.flags & CpSeq::DrawFlagInheritInstances) != 0) {
		if (!m_pending_num_instances.empty()) {
			DrainPreparedDraws();
		}
		args.instance_count = NumInstances();
	}
	if ((op.flags & CpSeq::DrawFlagPublished) != 0) {
		const NativeLeanScope native_lean(m_renderer.GetRenderExecutor(), op.flags, m_native_lean_op);
		ResolvePrep()->CommitPublished(
		    op.window, LaneCp().submit_id,
		    (op.flags & CpSeq::DrawFlagInheritInstances) != 0 ? args.instance_count : UINT32_MAX);
		MaybeYieldSlice();
		CommitStats::StepMark(CommitStats::Step::GapRetire);
		return;
	}
	if (RepeatTrace::Enabled()) {
		NoteRepeatTraceDrawPacket();
	}
	if (TrySubmitPreparedDraw(nullptr, &args)) {
		// See DrawIndex: idle flushes follow each commit, the slice counts the draw now.
		MaybeYieldSlice();
		return;
	}
	if (RepeatTrace::Enabled()) {
		(void)RepeatTrace::TakeDrawPacket();
		RepeatTrace::OnUnpreparedDraw();
	}
	m_renderer.GetRenderExecutor().DrawAuto(LaneCp().submit_id, CurrentBuffer(), args);
	MaybeFlushIdleGpu();
	MaybeYieldSlice();
}

// KYTY_FLIP_WAIT_MODE=block restores the blocking WAIT_FLIP_DONE, which stalled every guest
// queue and GPU-thread command (e.g. guest readbacks) until the presenter finished the flip. By
// default ("suspend") only the waiting queue is suspended, like WAIT_REG_MEM, and retried when
// a flip completes (NotifyGpuProgress) or the scheduler's blocked-queue timeout passes.
static bool FlipWaitSuspends() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_FLIP_WAIT_MODE");
		return value == nullptr || std::strcmp(value, "block") != 0;
	}();
	return enabled;
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	CpSeq::WaitFlipDoneOp op;
	op.video_out_handle     = video_out_handle;
	op.display_buffer_index = display_buffer_index;
	if (Submit(op).suspended) {
		SuspendPm4();
	}
}

CpSeq::Result CommandProcessor::ExecWaitFlipDone(const CpSeq::WaitFlipDoneOp& op) {
	const auto handle = static_cast<int>(op.video_out_handle);
	const auto index  = static_cast<int>(op.display_buffer_index);
	if (!FlipWaitSuspends()) {
		BufferFlush();
		Profiler::CountFrameEvent(Profiler::FrameEvent::FlipWaitBlocking);
		HangWatchdog::Scope wait("cp-flip-done", handle, index);
		m_renderer.GetVideoOut().WaitFlipDone(handle, index);
		return {};
	}
	if (!m_flip_wait_suspended) {
		// First evaluation of this packet: submit everything recorded before it (as the
		// blocking form did); retries have nothing new to submit.
		BufferFlush();
	}
	if (m_renderer.GetVideoOut().IsFlipPending(handle, index)) {
		if (!m_flip_wait_suspended) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::FlipWaitSuspends);
		}
		m_flip_wait_suspended = true;
		const auto queue =
		    m_interrupt_event_id == 0 ? 0u : static_cast<uint32_t>(m_interrupt_event_id - 0x20 + 1);
		HangWatchdog::NoteQueueWait(queue, "WAIT_FLIP_DONE", handle, 0, 1, index, 0, 0);
		return {true, 0};
	}
	m_flip_wait_suspended = false;
	const auto queue =
	    m_interrupt_event_id == 0 ? 0u : static_cast<uint32_t>(m_interrupt_event_id - 0x20 + 1);
	HangWatchdog::ClearQueueWait(queue);
	return {};
}

// KYTY_LABEL_MODE=completion writes every end-of-pipe label (RELEASE_MEM / EVENT_WRITE_EOP data
// writes) only after its tick has completed, in order, like hardware; the default ("record")
// writes them when the packet is recorded and defers only visibility-proxy labels. "split" writes
// them to guest memory at completion like "completion", but the command processor's own waits
// (WAIT_REG_MEM) see each value at record time (GuestGpu::CpVisibleLabelValue), as in "record":
// guest CPU threads cannot run ahead of the GPU, and the CP does not stall on GPU round trips.
static bool LabelSplitMode() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_LABEL_MODE");
		return value != nullptr && std::strcmp(value, "split") == 0;
	}();
	return enabled;
}

static bool LabelCompletionMode() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_LABEL_MODE");
		const bool  split = LabelSplitMode();
		const bool  on    = split || (value != nullptr && std::strcmp(value, "completion") == 0);
		std::printf("Kyty end-of-pipe labels: written at %s (KYTY_LABEL_MODE)\n",
		            split ? "completion, CP waits see them at record time"
		                  : (on ? "completion" : "record time"));
		return on;
	}();
	return enabled;
}

// Diagnostic (astrocrash): KYTY_LABEL_COMPLETION_RANGES=<lo>-<hi>[,<lo>-<hi>...] (hex guest
// addresses) defers only the end-of-pipe labels in those ranges to completion, like
// KYTY_LABEL_MODE=completion does for all; KYTY_LABEL_ADDR_LOG=1 prints each new label address once
// (at most 400) with its value and size.
static bool LabelRangeSelected(uint64_t address, uint64_t value, uint32_t size) {
	struct Range {
		uint64_t lo;
		uint64_t hi;
	};
	static const std::vector<Range> ranges = [] {
		std::vector<Range> list;
		const auto*        text = std::getenv("KYTY_LABEL_COMPLETION_RANGES");
		for (const char* p = text; p != nullptr && *p != 0;) {
			char*          end = nullptr;
			const uint64_t lo  = std::strtoull(p, &end, 16);
			uint64_t       hi  = lo;
			if (end != nullptr && *end == '-') {
				hi = std::strtoull(end + 1, &end, 16);
			}
			list.push_back({lo, hi});
			std::printf("Kyty end-of-pipe labels: 0x%" PRIx64 "-0x%" PRIx64
			            " written at completion (KYTY_LABEL_COMPLETION_RANGES)\n",
			            lo, hi);
			p = (end != nullptr && *end == ',') ? end + 1 : nullptr;
		}
		return list;
	}();
	static const bool log = [] {
		const auto* value = std::getenv("KYTY_LABEL_ADDR_LOG");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	if (log) {
		static std::mutex                   mutex;
		static std::unordered_set<uint64_t> seen;
		std::scoped_lock                    lock(mutex);
		if (seen.size() < 400 && seen.insert(address).second) {
			std::printf("Kyty label address 0x%" PRIx64 " value 0x%" PRIx64 " size %u\n", address,
			            value, size);
		}
	}
	for (const auto& r: ranges) {
		if (address >= r.lo && address <= r.hi) {
			return true;
		}
	}
	return false;
}

bool CommandProcessor::TryDeferLabel(void* dst, uint64_t value, uint32_t size, bool interrupt,
                                     uint32_t interrupt_context_id, uint32_t timestamp_slot) {
	const auto address = reinterpret_cast<uint64_t>(dst);
	auto&      gpu     = m_renderer.GetGpu();
	const bool proxy   = m_defer_next_label;
	const bool all     = LabelCompletionMode();
	// End-of-pipe writes become visible in order on hardware: while any older deferred write
	// (label or GDS snapshot) is pending, a later label must not overtake it.
	(void)address;
	// KYTY_LABEL_COMPLETION_INTERRUPTS=1 (diagnostic): labels that also raise an end-of-pipe
	// interrupt are written at completion.
	static const bool interrupts_at_completion = [] {
		const auto* value = std::getenv("KYTY_LABEL_COMPLETION_INTERRUPTS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	const bool selected =
	    LabelRangeSelected(address, value, size) || (interrupts_at_completion && interrupt);
	// KYTY_LABEL_COMPLETION_NOORDER=1 (diagnostic): labels outside the ranges stay at record time
	// even while a selected one is pending, so a range bisect does not defer everything after it.
	static const bool no_order = [] {
		const auto* value = std::getenv("KYTY_LABEL_COMPLETION_NOORDER");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	const bool ordered = !proxy && !all && !selected && !no_order && gpu.HasDeferredLabels();
	if (!proxy && !all && !selected && !ordered) {
		return false;
	}
	m_defer_next_label = false;
	auto&      scheduler = GetScheduler();
	const auto tick      = scheduler.CurrentTick();
	gpu.AddDeferredLabel(address, size, tick, value, LabelSplitMode());
	HangWatchdog::NoteEvent(address, "deferred-label", tick, INT16_MIN, false, value, size);
	TraceCpLabel(proxy ? "label-defer-proxy" : (ordered ? "label-defer-ordered" : "label-defer"),
	             dst, value, size);
	const int64_t trace_queue =
	    m_interrupt_event_id == 0 ? 0 : static_cast<int64_t>(m_interrupt_event_id) - 0x20 + 1;
	Profiler::CountFrameEvent(Profiler::FrameEvent::LabelWritesDeferred);
	if (proxy) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::LabelWritesDeferredProxy);
	} else if (ordered) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::LabelWritesDeferredOrdered);
	}
	auto*     renderer = &m_renderer;
	const int event_id = m_interrupt_event_id;
	// KYTY_EOP_TIMESTAMPS=gpu: a clock value is written with its query's GPU time.
	auto* timestamps =
	    timestamp_slot != EopTimestampRing::NoSlot ? scheduler.GuestTimestamps() : nullptr;
	// Runs on the completion runner once `tick` has completed, after every priority operation
	// registered before it (FIFO), including this tick's occlusion publications. The write itself
	// is handed to the GPU thread: a label page may be protected by resource tracking, and only
	// the GPU thread may take the resulting fault/readback. The interrupt follows the write.
	scheduler.DeferPriorityOperation(
	    [renderer, &gpu, address, value, size, tick, interrupt, event_id, interrupt_context_id,
	     trace_queue, timestamps, timestamp_slot] {
		    // The tick has completed, so a timestamp query's result is final.
		    const uint64_t written =
		        timestamps != nullptr ? timestamps->TakeDeferred(timestamp_slot, value) : value;
		    const bool sent = gpu.TrySendCommand([renderer, &gpu, address, written, size, tick,
		                                          interrupt, event_id, interrupt_context_id,
		                                          trace_queue] {
			    {
				    KYTY_PROFILER_DETAIL_BLOCK("EndOfPipe::WriteDeferredLabel");
				    HangWatchdog::Scope write("deferred-label-write", address, written, tick, size);
				    HangWatchdog::DebugDelay("label", address);
				    // KYTY_LABEL_MODE=split: a later CP write to the label (the CP passed a wait
				    // on it at record time) stays; hardware would have written the label first.
				    if (!gpu.DeferredLabelSuperseded(address, tick)) {
					    StreamCopyDefer::NoteGpuOnlyWrite(address, size); // after its tick completed
					    std::memcpy(reinterpret_cast<void*>(address), &written, size);
				    }
			    }
			    ReadFaultDiag::NoteLabel("label-deferred-write", address, written);
			    if (HangTrace::CpTraceEnabled()) {
				    HangTrace::CpEvent event;
				    event.event   = "label-deferred-write";
				    event.address = address;
				    event.value   = written;
				    event.size    = size;
				    event.aux     = static_cast<int64_t>(tick);
				    event.queue   = trace_queue;
				    HangTrace::RecordCp(event);
			    }
			    // Emulator write outside a fence position (draw-prep log certificate only).
			    Coherence::NoteContentWrite(address, size, Coherence::Source::CpWrite);
			    gpu.RemoveDeferredLabel(address, tick);
			    if (interrupt) {
				    renderer->TriggerInterrupt(event_id, interrupt_context_id);
			    }
		    });
		    if (!sent) {
			    // Shutdown: the GPU thread no longer runs commands. Best-effort direct write.
			    (void)LibKernel::Memory::TryWriteBacking(address, &written, size);
			    gpu.RemoveDeferredLabel(address, tick);
			    if (interrupt) {
				    renderer->TriggerInterrupt(event_id, interrupt_context_id);
			    }
		    }
	    },
	    interrupt ? CommandScheduler::PriorityOperationKind::EopInterrupt
	              : CommandScheduler::PriorityOperationKind::Generic);
	if (proxy) {
		// A guest thread is about to wait for this label: submit its tick now rather than at the
		// next EOP batch or slice boundary, which can be most of a frame away.
		BufferFlush();
	}
	return true;
}

// KYTY_EOP_DROPPED_LABELS: "write" (default) also writes the data of end-of-pipe events whose
// interrupt selector made Kyty skip it (graphics INT_SEL=1, RELEASE_MEM INT_SEL=4 with a data
// selection); "count" only counts them, as before.
static bool DroppedLabelsWritten() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_EOP_DROPPED_LABELS");
		return value == nullptr || std::strcmp(value, "count") != 0;
	}();
	return enabled;
}

bool CommandProcessor::WriteDroppedLabel(void* dst, uint64_t value, uint32_t size,
                                         bool interrupt, uint32_t interrupt_context_id,
                                         Profiler::FrameEvent counter, bool timestamp) {
	Profiler::CountFrameEvent(counter);
	if (!DroppedLabelsWritten() || dst == nullptr || (size != 4 && size != 8)) {
		return false;
	}
	const auto timestamp_slot =
	    timestamp && size == 8 ? RecordEopTimestamp() : EopTimestampRing::NoSlot;
	if (TryDeferLabel(dst, value, size, interrupt, interrupt_context_id, timestamp_slot)) {
		return interrupt;
	}
	KYTY_PROFILER_DETAIL_BLOCK("EndOfPipe::WriteLabel");
	if (timestamp_slot == EopTimestampRing::NoSlot && CpParDeferLabelStore(dst, value, size)) {
		// Astro Bot PC (KYTY_CP_PAR=1b5): written at the merge, before its interrupt.
		TraceCpLabel("label-dropped-cp-par-deferred", dst, value, size);
		return false;
	}
	StreamCopyDefer::NoteGuestWrite(reinterpret_cast<uint64_t>(dst), size);
	std::memcpy(dst, &value, size);
	NoteCpWrite(reinterpret_cast<uint64_t>(dst), size);
	QueueEopTimestamp(timestamp_slot, dst, value);
	TraceCpLabel("label-dropped", dst, value, size);
	return false;
}

uint32_t CommandProcessor::RecordEopTimestamp() {
	auto& scheduler  = GetScheduler();
	auto* timestamps = scheduler.GuestTimestamps();
	if (timestamps == nullptr) {
		return EopTimestampRing::NoSlot;
	}
	// A fence position: publish what completed ticks measured first (never waits).
	timestamps->Publish(scheduler.GetMasterSemaphore().KnownGpuTick());
	// A query write touches no guest resource, so batched barriers may stay pending (and no
	// rendering instance is split for it). With KYTY_CP_RECORDER it is a recorder packet, in
	// stream order, and drains nothing.
	return timestamps->RecordQuery(CurrentBuffer().StateSink());
}

void CommandProcessor::QueueEopTimestamp(uint32_t slot, const void* dst, uint64_t value) {
	if (slot == EopTimestampRing::NoSlot) {
		return;
	}
	auto& scheduler = GetScheduler();
	scheduler.GuestTimestamps()->Queue(slot, scheduler.CurrentTick(),
	                                   reinterpret_cast<uint64_t>(dst), value);
}

bool CommandProcessor::WriteReleaseMemDroppedData(void* dst, uint64_t value, uint32_t data_sel,
                                                  bool interrupt, uint32_t interrupt_context_id) {
	uint32_t size = 0;
	switch (data_sel) {
		case 1: size = 4; break;
		case 2: size = 8; break;
		case 3:
			size  = 8;
			value = GuestGpuTimestamp();
			break;
		default: return false; // 5 (GDS) and others keep the old behaviour
	}
	return WriteDroppedLabel(dst, value, size, interrupt, interrupt_context_id,
	                         Profiler::FrameEvent::ReleaseMemLabelsIntSel4, data_sel == 3);
}

// KYTY_GDS_EOP_MODE=defer snapshots the GDS range with a copy recorded at the packet's position
// and writes it to guest memory once that tick has completed (as a deferred label), instead of
// draining the GPU (SynchronizeGpu) and reading GDS at record time. Default "sync" until the
// counters show this path is frequent enough to matter (FrameEvent.GdsEopReads).
static bool GdsEopDeferEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_GDS_EOP_MODE");
		return value != nullptr && std::strcmp(value, "defer") == 0;
	}();
	return enabled;
}

bool CommandProcessor::TryDeferGdsRead(uint32_t* dst, uint32_t dw_offset, uint32_t dw_size,
                                       bool interrupt, uint32_t interrupt_context_id) {
	if (!GdsEopDeferEnabled() || dst == nullptr || dw_size == 0) {
		return false;
	}
	const auto* gds    = m_renderer.GetBufferCache().GetGdsBuffer();
	const auto  offset = uint64_t {dw_offset} * sizeof(uint32_t);
	const auto  size   = uint64_t {dw_size} * sizeof(uint32_t);
	if (offset > gds->Size() || size > gds->Size() - offset) {
		return false;
	}
	auto& download         = m_renderer.GetBufferCache().GetUtilityBuffer(MemoryUsage::Download);
	const auto [mapped, staged] = download.Map(size, 16);
	if (mapped == nullptr) {
		return false;
	}
	download.Commit();

	// Snapshot at this point of the GPU timeline: later work may change GDS before completion.
	auto& scheduler = GetScheduler();
	auto& buffer    = CurrentBuffer();
	buffer.EndRendering();
	const auto              native = buffer.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = gds->Handle();
	before.offset              = offset;
	before.size                = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	const vk::BufferCopy copy {offset, staged, size};
	native.copyBuffer(gds->Handle(), download.Handle(), 1, &copy);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = staged;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &after, 0,
	                       nullptr);
	// The GDS buffer's own shader accesses after this copy are ordered by their own barriers
	// (it is written only by recorded GPU work), exactly as for any other transfer read.

	const auto address = reinterpret_cast<uint64_t>(dst);
	const auto tick    = scheduler.CurrentTick();
	auto&      gpu     = m_renderer.GetGpu();
	gpu.AddDeferredLabel(address, static_cast<uint32_t>(size), tick);
	Profiler::CountFrameEvent(Profiler::FrameEvent::GdsEopReadsDeferred);
	auto*     renderer = &m_renderer;
	const int event_id = m_interrupt_event_id;
	scheduler.DeferPriorityOperation(
	    [renderer, &gpu, &download, mapped, staged, address, size, tick, interrupt, event_id,
	     interrupt_context_id] {
		    // The download ring slot stays reserved until this tick's priority operations ran.
		    download.Invalidate(staged, size);
		    std::vector<uint8_t> bytes(mapped, mapped + size);
		    auto write = [renderer, &gpu, bytes = std::move(bytes), address, tick, interrupt,
		                  event_id, interrupt_context_id](bool on_gpu_thread) {
			    if (on_gpu_thread) {
				    StreamCopyDefer::NoteGpuOnlyWrite(address, bytes.size()); // after its tick completed
				    std::memcpy(reinterpret_cast<void*>(address), bytes.data(), bytes.size());
			    } else {
				    (void)LibKernel::Memory::TryWriteBacking(address, bytes.data(), bytes.size());
			    }
			    Coherence::NoteContentWrite(address, bytes.size(), Coherence::Source::CpWrite);
			    gpu.RemoveDeferredLabel(address, tick);
			    if (interrupt) {
				    renderer->TriggerInterrupt(event_id, interrupt_context_id);
			    }
		    };
		    auto shared = std::make_shared<decltype(write)>(std::move(write));
		    if (!gpu.TrySendCommand([shared] { (*shared)(true); })) {
			    (*shared)(false);
		    }
	    },
	    interrupt ? CommandScheduler::PriorityOperationKind::EopInterrupt
	              : CommandScheduler::PriorityOperationKind::Generic);
	return true;
}

template <typename T>
void CommandProcessor::WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest,
                                        uint32_t eop_event_type, uint32_t cache_action,
                                        uint32_t event_index, uint32_t event_write_source,
                                        void* dst_gpu_addr, T value, uint32_t interrupt_selector,
                                        uint32_t interrupt_context_id) {
	static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		const auto bits      = static_cast<unsigned>(sizeof(T) * 8u);
		const auto log_width = static_cast<int>(sizeof(T) * 2u);

		LOGF("CommandProcessor::WriteAtEndOfPipe%u()\n"
		     "\t cache_policy        = 0x%08" PRIx32 "\n"
		     "\t event_write_dest    = 0x%08" PRIx32 "\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t event_index         = 0x%08" PRIx32 "\n"
		     "\t event_write_source  = 0x%08" PRIx32 "\n"
		     "\t interrupt_selector  = 0x%08" PRIx32 "\n"
		     "\t interrupt_context   = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%0*" PRIx64 "\n",
		     bits, cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
		     event_write_source, interrupt_selector, interrupt_context_id,
		     reinterpret_cast<uint64_t>(dst_gpu_addr), log_width, static_cast<uint64_t>(value));
	}

	EXIT_NOT_IMPLEMENTED(cache_policy != 0x00000000);
	EXIT_NOT_IMPLEMENTED(event_write_dest != 0x00000000);

	bool with_interrupt = false;
	switch (interrupt_selector) {
		case 0x00:
		case 0x03: with_interrupt = false; break;
		case 0x01:
			if (!IsAsyncComputeQueue()) {
				// INT_SEL=1 on the graphics queue used to raise only the interrupt. On RDNA
				// INT_SEL does not gate DATA_SEL: write the data as well (KYTY_EOP_DROPPED_LABELS).
				// Plain data selections only (32-bit data, 64-bit data, reference clock).
				uint32_t label_size  = 0;
				uint64_t label_value = static_cast<uint64_t>(value);
				bool     label_clock = false;
				if constexpr (sizeof(T) == sizeof(uint32_t)) {
					label_size = event_write_source == 0x02 ? 4u : 0u;
				} else {
					switch (event_write_source) {
						case 0x01: label_size = 4; break;
						case 0x02: label_size = 8; break;
						case 0x04:
							label_size  = 8;
							label_value = GuestGpuTimestamp();
							label_clock = true;
							break;
						default: break;
					}
				}
				if (label_size != 0 && dst_gpu_addr != nullptr &&
				    WriteDroppedLabel(dst_gpu_addr, label_value, label_size, true,
				                      interrupt_context_id, Profiler::FrameEvent::EopLabelsIntSel1,
				                      label_clock)) {
					return; // the deferred label raises the interrupt after its write
				}
				// KYTY_LABEL_MODE=completion/split: deferred label writes run on the GPU thread, so
				// an interrupt-only end-of-pipe event goes there too, after the labels before it
				// (it could wake the guest before they were visible). KYTY_LABEL_ORDERED_INTERRUPTS=0.
				static const bool ordered_interrupts = [] {
					const auto* v = std::getenv("KYTY_LABEL_ORDERED_INTERRUPTS");
					return LabelCompletionMode() && (v == nullptr || std::strcmp(v, "0") != 0);
				}();
				if (ordered_interrupts) {
					auto*     renderer = &m_renderer;
					auto&     gpu      = m_renderer.GetGpu();
					const int event_id = m_interrupt_event_id;
					const auto ctx     = interrupt_context_id;
					GetScheduler().DeferPriorityOperation(
					    [renderer, &gpu, event_id, ctx] {
						    if (!gpu.TrySendCommand([renderer, event_id, ctx] {
							        renderer->TriggerInterrupt(event_id, ctx);
						        })) {
							    renderer->TriggerInterrupt(event_id, ctx);
						    }
					    },
					    CommandScheduler::PriorityOperationKind::EopInterrupt);
					return;
				}
				Sync::TriggerEopEventAtEndOfPipe(command, m_interrupt_event_id,
				                                 interrupt_context_id);
				return;
			}
			with_interrupt = true;
			break;
		case 0x02: with_interrupt = true; break;
		default: EXIT("unknown interrupt_selector\n");
	}

	auto write32 = [&](bool with_writeback) {
		auto* dst  = static_cast<uint32_t*>(dst_gpu_addr);
		auto  data = static_cast<uint32_t>(value);
		if (TryDeferLabel(dst, data, sizeof(data), with_interrupt, interrupt_context_id)) {
			// The deferred write raises the interrupt itself, after the label is visible.
			if (with_writeback) {
				Sync::WriteAtEndOfPipeWithWriteBack32(LaneCp().submit_id, command, dst, data);
			} else {
				Sync::WriteAtEndOfPipe32(LaneCp().submit_id, command, dst, data);
			}
			return;
		}
		if (CpParDeferLabelStore(dst, data, sizeof(data))) {
			// Astro Bot PC (KYTY_CP_PAR=1b5): written at the merge, before its interrupt.
		} else {
			{
				// Guest label pages can be protected by resource tracking. Attribute any
				// resulting fault separately from the end-of-pipe submission/interrupt work.
				KYTY_PROFILER_DETAIL_BLOCK("EndOfPipe::WriteLabel32");
				StreamCopyDefer::NoteGuestWrite(reinterpret_cast<uint64_t>(dst), sizeof(data));
				std::memcpy(dst, &data, sizeof(data));
			}
			NoteCpWrite(reinterpret_cast<uint64_t>(dst), sizeof(data));
		}
		TraceCpLabel("label-eop", dst, data, sizeof(data));

		if (with_interrupt) {
			if (with_writeback) {
				Sync::WriteAtEndOfPipeWithInterruptWriteBack32(LaneCp().submit_id, command, dst,
				                                               data, m_interrupt_event_id,
				                                               interrupt_context_id);
			} else {
				Sync::WriteAtEndOfPipeWithInterrupt32(LaneCp().submit_id, command, dst, data,
				                                      m_interrupt_event_id, interrupt_context_id);
			}
		} else if (with_writeback) {
			Sync::WriteAtEndOfPipeWithWriteBack32(LaneCp().submit_id, command, dst, data);
		} else {
			Sync::WriteAtEndOfPipe32(LaneCp().submit_id, command, dst, data);
		}
	};

	switch (event_write_source) {
		case 0x01:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && cache_action == 0x00 && event_index == 0x06) {
					auto* dst = static_cast<uint32_t*>(dst_gpu_addr);
					Profiler::CountFrameEvent(Profiler::FrameEvent::GdsEopReads);
					if (TryDeferGdsRead(dst, value & 0xffffu, value >> 16u, with_interrupt,
					                    interrupt_context_id)) {
						Sync::WriteAtEndOfPipeGds32(LaneCp().submit_id, command, dst, value & 0xffffu,
						                            value >> 16u);
						return;
					}
					Profiler::ScopedGpuWaitReason wait_reason(Profiler::FrameWait::GpuWaitGds);
					SynchronizeGpu();
					StreamCopyDefer::NoteGuestWrite(reinterpret_cast<uint64_t>(dst), uint64_t {value >> 16u} * 4u);
					Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(), dst, value & 0xffffu,
					              value >> 16u);
					NoteCpWrite(reinterpret_cast<uint64_t>(dst), uint64_t {value >> 16u} * 4u);
					TraceCpLabel("label-gds", dst, *dst, value >> 16u);
					Sync::WriteAtEndOfPipeGds32(LaneCp().submit_id, command, dst, value & 0xffffu,
					                            value >> 16u);
					if (with_interrupt) {
						m_renderer.TriggerInterrupt(m_interrupt_event_id, interrupt_context_id);
					}
					return;
				}
			} else if (eop_event_type == 0x04 && cache_action == 0x00 && event_index == 0x05) {
				write32(false);
				return;
			}
			break;
		case 0x02:
		case 0x04:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (event_write_source == 0x02 && eop_event_type == 0x2f && event_index == 0x06) {
					switch (cache_action) {
						case 0x00: write32(false); return;
						case 0x38: write32(true); return;
						default: break;
					}
				}
			} else {
				const bool clock_write = event_write_source == 0x04;
				if (clock_write) {
					value = GuestGpuTimestamp();
				}
				auto write64 = [&](bool with_writeback) {
					auto* dst = static_cast<uint64_t*>(dst_gpu_addr);
					// KYTY_EOP_TIMESTAMPS=gpu: a query at this point; the slot gets its GPU time
					// once the tick completes (from the deferred write when deferred).
					const auto timestamp_slot =
					    clock_write ? RecordEopTimestamp() : EopTimestampRing::NoSlot;
					if (TryDeferLabel(dst, value, sizeof(value), with_interrupt,
					                  interrupt_context_id, timestamp_slot)) {
						// The deferred write raises the interrupt itself, after the label.
						if (with_writeback) {
							Sync::WriteAtEndOfPipeWithWriteBack64(LaneCp().submit_id, command, dst,
							                                      value);
						} else {
							Sync::WriteAtEndOfPipe64(LaneCp().submit_id, command, dst, value);
						}
						return;
					}
					if (timestamp_slot == EopTimestampRing::NoSlot &&
					    CpParDeferLabelStore(dst, value, sizeof(value))) {
						// Astro Bot PC (KYTY_CP_PAR=1b5): written at the merge, before its interrupt.
					} else {
						{
							KYTY_PROFILER_DETAIL_BLOCK("EndOfPipe::WriteLabel64");
							StreamCopyDefer::NoteGuestWrite(reinterpret_cast<uint64_t>(dst), sizeof(value));
							std::memcpy(dst, &value, sizeof(value));
						}
						NoteCpWrite(reinterpret_cast<uint64_t>(dst), sizeof(value));
					}
					QueueEopTimestamp(timestamp_slot, dst, value);
					TraceCpLabel("label-eop", dst, value, sizeof(value));

					if (with_interrupt) {
						if (with_writeback) {
							Sync::WriteAtEndOfPipeWithInterruptWriteBack64(
							    LaneCp().submit_id, command, dst, value, m_interrupt_event_id,
							    interrupt_context_id);
						} else {
							Sync::WriteAtEndOfPipeWithInterrupt64(LaneCp().submit_id, command, dst,
							                                      value, m_interrupt_event_id,
							                                      interrupt_context_id);
						}
					} else if (with_writeback) {
						Sync::WriteAtEndOfPipeWithWriteBack64(LaneCp().submit_id, command, dst,
						                                      value);
					} else {
						Sync::WriteAtEndOfPipe64(LaneCp().submit_id, command, dst, value);
					}
				};

				switch (cache_action) {
					case 0x00:
						switch (eop_event_type) {
							case 0x04:
								if (event_index == 0x05) {
									write64(false);
									return;
								}
								break;
							case 0x14:
							case 0x28:
							case 0x2f:
								if (event_index == 0x00) {
									write64(false);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
							case 0x30:
								if (event_index == 0x00 && !with_interrupt) {
									write64(false);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x38:
						switch (eop_event_type) {
							case 0x04:
							case 0x14:
							case 0x28:
								if (((eop_event_type == 0x04 || eop_event_type == 0x28) &&
								     event_index == 0x05) ||
								    (event_index == 0x00)) {
									write64(true);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
								if (event_index == 0x00 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							case 0x2f:
								if (event_index == 0x06 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x3b:
						if (eop_event_type == 0x04 && event_index == 0x05 && with_interrupt) {
							write64(true);
							return;
						}
						break;
					default: break;
				}
			}
			break;
		default: break;
	}

	EXIT("unknown event type\n");
}

static CpSeq::EndOfPipeOp EndOfPipePayload(uint32_t cache_policy, uint32_t event_write_dest,
                                           uint32_t eop_event_type, uint32_t cache_action,
                                           uint32_t event_index, uint32_t event_write_source,
                                           void* dst_gpu_addr, uint64_t value,
                                           uint32_t interrupt_selector,
                                           uint32_t interrupt_context_id, uint32_t size) {
	CpSeq::EndOfPipeOp op;
	op.dst                  = reinterpret_cast<uint64_t>(dst_gpu_addr);
	op.value                = value;
	op.cache_policy         = cache_policy;
	op.event_write_dest     = event_write_dest;
	op.eop_event_type       = eop_event_type;
	op.cache_action         = cache_action;
	op.event_index          = event_index;
	op.event_write_source   = event_write_source;
	op.interrupt_selector   = interrupt_selector;
	op.interrupt_context_id = interrupt_context_id;
	op.size                 = size;
	return op;
}

void CommandProcessor::WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint32_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	(void)Submit(EndOfPipePayload(cache_policy, event_write_dest, eop_event_type, cache_action,
	                              event_index, event_write_source, dst_gpu_addr, value,
	                              interrupt_selector, interrupt_context_id, sizeof(uint32_t)));
}

void CommandProcessor::WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint64_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	(void)Submit(EndOfPipePayload(cache_policy, event_write_dest, eop_event_type, cache_action,
	                              event_index, event_write_source, dst_gpu_addr, value,
	                              interrupt_selector, interrupt_context_id, sizeof(uint64_t)));
}

void CommandProcessor::ExecEndOfPipe(const CpSeq::EndOfPipeOp& op) {
	auto* dst = reinterpret_cast<void*>(op.dst);
	if (op.size == sizeof(uint32_t)) {
		WriteAtEndOfPipe(op.cache_policy, op.event_write_dest, op.eop_event_type, op.cache_action,
		                 op.event_index, op.event_write_source, dst,
		                 static_cast<uint32_t>(op.value), op.interrupt_selector,
		                 op.interrupt_context_id);
		return;
	}
	WriteAtEndOfPipe(op.cache_policy, op.event_write_dest, op.eop_event_type, op.cache_action,
	                 op.event_index, op.event_write_source, dst, op.value, op.interrupt_selector,
	                 op.interrupt_context_id);
}

void CommandProcessor::EmitGlobalBarrier() {
	KYTY_GPU_OP_SITE("guest.global_barrier");
	KYTY_PROFILER_DETAIL_FUNCTION();
	RenderLockGuard lock(m_renderer.GetMutex());
	// Keep renderer-lock contention in the parent zone's self time.
	KYTY_PROFILER_DETAIL_BLOCK("CommandProcessor::RecordGlobalBarrier");

	if (BarrierBatchEnabled()) {
		// Queued: merged with adjacent requests, elided when the previous barrier already covers
		// it with nothing recorded since, recorded before the next memory-accessing command.
		// The rendering instance ends only if the barrier is recorded (see render.h).
		CurrentBuffer().RequestMemoryBarrier(
		    vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
		    vk::PipelineStageFlagBits2::eAllCommands,
		    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
		    BarrierOrigin::Guest);
		return;
	}

	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	GetScheduler().EndRendering();
	CurrentBuffer().Handle().pipelineBarrier2(dependency);
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	Sync::TriggerEopEventAtEndOfPipe(CurrentBuffer(), m_interrupt_event_id, interrupt_context_id);
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	CpSeq::EventWriteOp op;
	op.address     = event_address;
	op.event_type  = event_type;
	op.event_index = event_index;
	(void)Submit(op);
}

void CommandProcessor::ExecEventWrite(const CpSeq::EventWriteOp& op) {
	const auto event_type    = op.event_type;
	const auto event_index   = op.event_index;
	const auto event_address = op.address;
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: EmitGlobalBarrier(); break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGlobalBarrier();
			break;
		// DbDataWritebackInvalidate, DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGlobalBarrier();
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			Profiler::CountFrameEvent(Profiler::FrameEvent::OcclusionCounterDumps);
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			if (OcclusionCounter::Enabled()) {
				bool sync = false;
				bool kept = false;
				{
					RenderLockGuard lock(m_renderer.GetMutex());
					auto&             counter = m_renderer.GetOcclusionCounter();
					sync = counter.Dump(event_address);
					kept = counter.LastDumpKeptInstance();
				}
				// KYTY_DRAW_RUN: a dump that kept the instance and does not wait recorded only query
				// commands, which a continuation's skipped work does not depend on (ExecuteOp).
				const bool waits =
				    sync && OcclusionCounter::GetProxyMode() == OcclusionCounter::ProxyMode::Sync;
				if (DrawRun::Enabled() && DrawRun::QuietOpsEnabled() &&
				    OcclusionCounter::SplitEnabled() && (!kept || waits)) {
					DrawRun::NoteForeignActivity("EventWrite occlusion dump");
				}
				if (sync) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::OcclusionProxyDumps);
					if (OcclusionCounter::GetProxyMode() == OcclusionCounter::ProxyMode::Sync) {
						// Publish this visibility-proxy result before the CP processes the label
						// that follows it (labels are written at record time).
						Profiler::ScopedGpuWaitReason wait_reason(
						    Profiler::FrameWait::GpuWaitOcclusion);
						BufferWait();
					} else {
						// defer-label: the next end-of-pipe label is written only after this
						// dump's tick completed and its publication (queued above on the same
						// completion runner) has run. See TryDeferLabel.
						m_defer_next_label = true;
					}
				}
				break;
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Occlusion queries: treated as always visible (performance mode). If objects "
				            "show through walls, turn on accurate occlusion queries in the launcher "
				            "(--gpu-occlusion on, KYTY_GPU_OCCLUSION=1).\n");
			});

			// Performance mode (KYTY_GPU_OCCLUSION=0): publish an always-visible result. The
			// PS5 layout contains one interleaved begin/end pair per DB, and bit 63 marks a result
			// ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			StreamCopyDefer::NoteGpuOnlyWrite(event_address, (16u * 2u - 1u) * sizeof(uint64_t));
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			// The bytes written: 16 slots 16 bytes apart. Not 256: an end dump sits 8 bytes into its
			// pair block, and 256 bytes from it covers the label the game keeps right after the
			// block, which split label mode then took for overwritten and skipped its write.
			NoteCpWrite(event_address, (16u * 2u - 1u) * sizeof(uint64_t));
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

// KYTY_LOOP_GUARD: guarded shaders count exhausted invocations in the last GDS dword. The mapped
// word is read without waiting for the GPU (a diagnostic) and reported when it changes.
static void NoteLoopGuardHits(RenderContext& renderer) {
	const auto& options = Libs::Graphics::ShaderRecompiler::GetCodegenOptions();
	if (options.loop_guard_budget == 0 ||
	    (options.loop_guard_shaders.empty() && !options.loop_guard_all)) {
		return;
	}
	const auto mapped = renderer.GetBufferCache().GetGdsBuffer()->Mapped();
	constexpr auto hash_from_end = Libs::Graphics::ShaderRecompiler::LoopGuardHashGdsFromEnd;
	if (mapped.size() < (hash_from_end + 1u) * sizeof(uint32_t)) {
		return;
	}
	static uint32_t reported = 0;
	uint32_t        hits     = 0;
	std::memcpy(&hits, mapped.data() + mapped.size() - sizeof(uint32_t), sizeof(hits));
	if (hits != reported) {
		reported = hits;
		// The last exhausted shader (KYTY_LOOP_GUARD_SHADERS=all names it; written by every one).
		uint32_t hash_words[2] = {};
		std::memcpy(hash_words, mapped.data() + mapped.size() - hash_from_end * sizeof(uint32_t),
		            sizeof(hash_words));
		const uint64_t hash = hash_words[0] | (uint64_t {hash_words[1]} << 32u);
		Log::WriteToConsoleAndLog(fmt::format(
		    "Loop guard: {} invocations of the guarded shaders exhausted the {}-iteration loop "
		    "budget (KYTY_LOOP_GUARD); last one in shader 0x{:016x}\n",
		    hits, options.loop_guard_budget, hash));
	}
}

// KYTY_RT_NODE_BUDGET / KYTY_RT_NODE_STATS (KYTY_RT_SOFTWARE): the invocations that exhausted the
// BVH node budget and, with the statistics, how many node tests invocations ran, by power of two.
// The mapped GDS words are read without waiting for the GPU (a diagnostic): the budget line when
// its count changes, the histogram at most every 300 flips.
static void NoteRtNodeCounts(RenderContext& renderer) {
	namespace Recompiler = Libs::Graphics::ShaderRecompiler;
	const auto& options  = Recompiler::GetCodegenOptions();
	if (!options.rt_software || (options.rt_node_budget == 0 && !options.rt_node_stats)) {
		return;
	}
	const auto     mapped = renderer.GetBufferCache().GetGdsBuffer()->Mapped();
	constexpr auto Needed = Recompiler::RtNodeStatsGdsFromEnd + Recompiler::RtNodeStatsBins;
	if (mapped.size() < Needed * sizeof(uint32_t)) {
		return;
	}
	const auto word = [&](uint32_t from_end) {
		uint32_t value = 0;
		std::memcpy(&value, mapped.data() + mapped.size() - from_end * sizeof(uint32_t),
		            sizeof(value));
		return value;
	};
	if (options.rt_node_budget != 0) {
		static uint32_t reported = 0;
		const auto      hits     = word(Recompiler::RtNodeBudgetGdsFromEnd);
		if (hits != reported) {
			reported = hits;
			Log::WriteToConsoleAndLog(fmt::format(
			    "RT node budget: {} invocations exhausted the {}-test BVH node budget; their node "
			    "tests missed from then on and they left their loops (KYTY_RT_NODE_BUDGET)\n",
			    hits, options.rt_node_budget));
		}
	}
	if (options.rt_node_stats) {
		static uint32_t flips = 0;
		static std::array<uint32_t, Recompiler::RtNodeStatsBins> reported {};
		if (flips++ % 300u != 0u) {
			return;
		}
		std::array<uint32_t, Recompiler::RtNodeStatsBins> bins {};
		for (uint32_t bin = 0; bin < bins.size(); bin++) {
			bins[bin] = word(Recompiler::RtNodeStatsGdsFromEnd + bin);
		}
		if (bins == reported) {
			return;
		}
		reported = bins;
		std::string text;
		for (uint32_t bin = 0; bin < bins.size(); bin++) {
			if (bins[bin] != 0) {
				text += fmt::format(" [{},{}):{}", 1u << bin, 2ull << bin, bins[bin]);
			}
		}
		Log::WriteToConsoleAndLog(fmt::format(
		    "RT node stats: invocations by BVH node tests per lane, since start{} "
		    "(KYTY_RT_NODE_STATS)\n",
		    text));
	}
}

// The flip markers: the front's flip info (FLIP packet / SetFlip) travels with the op.
static CpSeq::FlipOp FlipPayload(const CommandProcessor::FlipInfo& flip, CpSeq::FlipVariant variant,
                                 void* dst_gpu_addr, uint32_t value, uint32_t eop_event_type,
                                 uint32_t cache_action) {
	CpSeq::FlipOp op;
	op.flip_arg       = flip.flip_arg;
	op.dst            = reinterpret_cast<uint64_t>(dst_gpu_addr);
	op.handle         = flip.handle;
	op.index          = flip.index;
	op.flip_mode      = flip.flip_mode;
	op.variant        = static_cast<uint32_t>(variant);
	op.value          = value;
	op.eop_event_type = eop_event_type;
	op.cache_action   = cache_action;
	return op;
}

void CommandProcessor::Flip() {
	(void)Submit(FlipPayload(m_flip, CpSeq::FlipVariant::Plain, nullptr, 0, 0, 0));
}

void CommandProcessor::Flip(void* dst_gpu_addr, uint32_t value) {
	(void)Submit(FlipPayload(m_flip, CpSeq::FlipVariant::Label, dst_gpu_addr, value, 0, 0));
}

void CommandProcessor::FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action,
                                         void* dst_gpu_addr, uint32_t value) {
	(void)Submit(FlipPayload(m_flip, CpSeq::FlipVariant::LabelInterrupt, dst_gpu_addr, value,
	                         eop_event_type, cache_action));
}

// KYTY_LABEL_MODE=completion/split: a flip's end-of-pipe label is written at completion like any
// other label (it was written at record time). KYTY_LABEL_SPLIT_FLIPS=0 restores that.
bool CommandProcessor::DeferFlipLabel(void* dst, uint32_t value, bool interrupt) {
	static const bool enabled = [] {
		const auto* v = std::getenv("KYTY_LABEL_SPLIT_FLIPS");
		return LabelCompletionMode() && (v == nullptr || std::strcmp(v, "0") != 0);
	}();
	if (!enabled || dst == nullptr) {
		return false;
	}
	static std::once_flag once;
	std::call_once(once, [interrupt] {
		std::printf("Kyty end-of-pipe labels: flip labels written at completion (%s)\n",
		            interrupt ? "with interrupt" : "plain");
	});
	NoteCpAhead(CpAheadKind::FlipLabelDeferred, reinterpret_cast<uint64_t>(dst), value);
	const bool proxy   = m_defer_next_label;
	m_defer_next_label = false;
	const bool deferred = TryDeferLabel(dst, value, sizeof(value), interrupt, 0);
	m_defer_next_label  = proxy;
	return deferred;
}

void CommandProcessor::ExecFlip(const CpSeq::FlipOp& op) {
	g_gpu_timestamp_anchor.store(Sync::ReadReferenceClock(), std::memory_order_relaxed);
	const FlipInfo flip {.handle    = op.handle,
	                     .index     = op.index,
	                     .flip_mode = op.flip_mode,
	                     .flip_arg  = op.flip_arg};
	auto* const    dst_gpu_addr = reinterpret_cast<void*>(op.dst);
	const auto     value        = op.value;
	switch (static_cast<CpSeq::FlipVariant>(op.variant)) {
		case CpSeq::FlipVariant::Plain: {
			NoteLoopGuardHits(m_renderer);
			NoteRtNodeCounts(m_renderer);
			if (GraphicsRunDebugDumpEnabled()) {
				LOGF("CommandProcessor::Flip()\n");
			}

			auto& command = CurrentBuffer();
			auto  request = Sync::PrepareVideoOutFlip(command, flip.handle, flip.index,
			                                          flip.flip_mode, flip.flip_arg);
			Sync::WriteAtEndOfPipeOnlyFlip(LaneCp().submit_id, command, flip.handle, flip.index,
			                               flip.flip_mode, flip.flip_arg, request);
			GetScheduler().Flush();
			Live::OnCpFlip(); Libs::Graphics::PageManager::NoteCpFlip(); // the frame boundary: live switches (KYTY_LIVE_FILE)
			return;
		}
		case CpSeq::FlipVariant::Label: {
			NoteLoopGuardHits(m_renderer);
			NoteRtNodeCounts(m_renderer);
			auto& command = CurrentBuffer();

			if (GraphicsRunDebugDumpEnabled()) {
				LOGF("CommandProcessor::Flip()\n"
				     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
				     "\t value        = 0x%08" PRIx32 "\n",
				     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
			}

			if (!DeferFlipLabel(dst_gpu_addr, value, false)) {
				StreamCopyDefer::NoteGuestWrite(op.dst, sizeof(value));
				std::memcpy(dst_gpu_addr, &value, sizeof(value));
				NoteCpWrite(op.dst, sizeof(value));
			}
			auto request = Sync::PrepareVideoOutFlip(command, flip.handle, flip.index,
			                                         flip.flip_mode, flip.flip_arg);
			Sync::WriteAtEndOfPipeWithFlip32(LaneCp().submit_id, command,
			                                 static_cast<uint32_t*>(dst_gpu_addr), value,
			                                 flip.handle, flip.index, flip.flip_mode,
			                                 flip.flip_arg, request);
			GetScheduler().Flush();
			Live::OnCpFlip(); Libs::Graphics::PageManager::NoteCpFlip();
			return;
		}
		case CpSeq::FlipVariant::LabelInterrupt: {
			const auto eop_event_type = op.eop_event_type;
			const auto cache_action   = op.cache_action;
			NoteLoopGuardHits(m_renderer);
			NoteRtNodeCounts(m_renderer);
			auto& command = CurrentBuffer();

			if (GraphicsRunDebugDumpEnabled()) {
				LOGF("CommandProcessor::FlipWithInterrupt()\n"
				     "\t eop_event_type      = 0x%08" PRIx32 "\n"
				     "\t cache_action        = 0x%08" PRIx32 "\n"
				     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
				     "\t value               = 0x%08" PRIx32 "\n",
				     eop_event_type, cache_action, reinterpret_cast<uint64_t>(dst_gpu_addr),
				     value);
			}

			if (eop_event_type != 0x00000004 || cache_action != 0x00000038) {
				EXIT("unknown event type\n");
			}
			if (DeferFlipLabel(dst_gpu_addr, value, true)) {
				// The deferred label raises the flip's interrupt after its write.
				auto request = Sync::PrepareVideoOutFlip(command, flip.handle, flip.index,
				                                         flip.flip_mode, flip.flip_arg);
				Sync::WriteAtEndOfPipeWithFlip32(LaneCp().submit_id, command,
				                                 static_cast<uint32_t*>(dst_gpu_addr), value,
				                                 flip.handle, flip.index, flip.flip_mode,
				                                 flip.flip_arg, request);
				GetScheduler().Flush();
				Live::OnCpFlip(); Libs::Graphics::PageManager::NoteCpFlip();
				return;
			}
			StreamCopyDefer::NoteGuestWrite(op.dst, sizeof(value));
			std::memcpy(dst_gpu_addr, &value, sizeof(value));
			NoteCpWrite(op.dst, sizeof(value));
			auto request = Sync::PrepareVideoOutFlip(command, flip.handle, flip.index,
			                                         flip.flip_mode, flip.flip_arg);
			Sync::WriteAtEndOfPipeWithInterruptWriteBackFlip32(
			    LaneCp().submit_id, command, static_cast<uint32_t*>(dst_gpu_addr), value, flip.handle,
			    flip.index, flip.flip_mode, flip.flip_arg, request, m_interrupt_event_id);
			GetScheduler().Flush();
			Live::OnCpFlip(); Libs::Graphics::PageManager::NoteCpFlip();
			return;
		}
	}
	EXIT("unknown flip variant %u\n", op.variant);
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	auto& command = CurrentBuffer();
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	struct ProcessorScope {
		explicit ProcessorScope(CommandProcessor& processor) { g_current_processor = &processor; }
		~ProcessorScope() { g_current_processor = nullptr; }
	};
	ProcessorScope processor_scope(*this);

	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	GetScheduler().DeferPriorityOperation(
	    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
	GetScheduler().Flush();
	Live::OnCpFlip(); Libs::Graphics::PageManager::NoteCpFlip(); // a CPU-submitted flip's frame boundary (KYTY_LIVE_FILE)
}

void CommandProcessor::SynchronizeGpu() {
	KYTY_PROFILER_DETAIL_FUNCTION();
	GetScheduler().Finish();
}

// ---- P3: the front's op hand-off (cpOps.h) -----------------------------------------------------

// KYTY_CP_SEQ_RING_KB (default 512): the op ring of inline mode. The largest record (a
// WRITE_DATA of 16383 dwords) needs at least 256 KiB.
static uint64_t OpRingBytes() {
	static const uint64_t bytes = [] {
		const auto* value  = std::getenv("KYTY_CP_SEQ_RING_KB");
		const auto  parsed = value != nullptr ? std::strtoul(value, nullptr, 10) : 512ul;
		uint64_t    size   = 256u << 10u;
		while (size < uint64_t {std::min(parsed, 1ul << 20u)} << 10u) {
			size <<= 1u;
		}
		return size;
	}();
	return bytes;
}

CpSeq::Result CommandProcessor::Submit(CpSeq::OpKind kind, const void* payload,
                                       uint32_t payload_size, const void* data,
                                       uint32_t data_size) {
	switch (m_front_mode) {
		case FrontMode::Direct: return ExecuteOp(kind, payload, data);
		case FrontMode::Inline: return SubmitInline(kind, payload, payload_size, data, data_size);
		case FrontMode::Reference: {
			EXIT_IF(m_capture == nullptr || g_current_execution == nullptr);
			return m_capture->Capture(kind, payload, payload_size, data, data_size,
			                          g_current_execution->m_packets,
			                          g_current_execution->m_packets_hash);
		}
		case FrontMode::Thread: return SubmitThread(kind, payload, payload_size, data, data_size);
		case FrontMode::Prefetch: return SubmitPrefetch(kind, payload);
	}
	EXIT("unknown front mode\n");
	return {};
}

CpSeq::Result CommandProcessor::SubmitInline(CpSeq::OpKind kind, const void* payload,
                                             uint32_t payload_size, const void* data,
                                             uint32_t data_size) {
	// An op's execution never emits another op (Exec* bodies call Exec*, not front methods).
	EXIT_IF(m_executing);
	if (m_ops == nullptr) {
		m_ops.reset(new CpSeq::OpStream(OpRingBytes()));
	}
	const auto* execution = g_current_execution;
	const bool  verify    = m_verifier != nullptr && execution != nullptr &&
	                    m_verifier->Follows(execution->m_stream_id);
	(void)m_ops->Emit(kind, payload, payload_size, data, data_size,
	                  execution != nullptr ? execution->m_packets : 0,
	                  execution != nullptr ? execution->m_packets_hash : 0, verify);
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqOps);
	// The resolver, at once on this thread: decode, compare with the reference front, execute.
	CpSeq::OpView view;
	EXIT_IF(!m_ops->Peek(view));
	if (verify) {
		m_verifier->Before(view);
	}
	m_executing       = true;
	const auto result = ExecuteOp(view.Kind(), view.payload, view.data);
	m_executing       = false;
	if (verify) {
		m_verifier->After(view, result);
	}
	m_ops->Pop();
	return result;
}

// KYTY_DRAW_RUN's "activity by" attribution of an EVENT_WRITE (ExecEventWrite's groups).
static const char* EventWriteActivityName(uint32_t event_type) {
	switch (event_type) {
		case 0x07:
		case 0x0f:
		case 0x10: return "EventWrite partial flush";
		case 0x16:
		case 0x31:
		case 0x2a:
		case 0x2c:
		case 0x2e: return "EventWrite cache writeback";
		case 0x39: return "EventWrite occlusion dump";
		default: return "EventWrite ignored type";
	}
}

// KYTY_CP_OP_STATS=1 (diagnostic, default off): the resolver's time per op kind (EVENT_WRITE by
// group), one console line every 10 s: "CpOps 10s: <kind> <count> <ms> ...", sorted by time.
namespace {
struct CpOpStats {
	struct Entry {
		const char* name  = nullptr;
		uint64_t    count = 0;
		uint64_t    ns    = 0;
	};
	std::array<Entry, 40> entries {};
	uint64_t              last_print_ns = 0;
};
bool CpOpStatsEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CP_OP_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
uint64_t CpOpStatsNowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}
class CpOpTimer {
public:
	explicit CpOpTimer(const char* name): m_name(name), m_start(CpOpStatsNowNs()) {}
	~CpOpTimer() {
		static CpOpStats stats;
		const auto       now = CpOpStatsNowNs();
		if (g_cp_op_outcome != nullptr) {
			m_name          = g_cp_op_outcome;
			g_cp_op_outcome = nullptr;
		}
		for (auto& entry: stats.entries) {
			if (entry.name == m_name || entry.name == nullptr) {
				entry.name = m_name;
				entry.count++;
				entry.ns += now - m_start;
				break;
			}
		}
		if (stats.last_print_ns == 0) {
			stats.last_print_ns = now;
		} else if (now - stats.last_print_ns >= 10'000'000'000ull) {
			auto sorted = stats.entries;
			std::sort(sorted.begin(), sorted.end(),
			          [](const auto& a, const auto& b) { return a.ns > b.ns; });
			std::string line = "CpOps 10s:";
			for (const auto& entry: sorted) {
				if (entry.name != nullptr && entry.count != 0) {
					line += fmt::format(" {} {} {:.1f}ms;", entry.name, entry.count,
					                    static_cast<double>(entry.ns) / 1e6);
				}
			}
			std::printf("%s\n", line.c_str());
			std::fflush(stdout);
			stats.entries       = {};
			stats.last_print_ns = now;
		}
	}
	CpOpTimer(const CpOpTimer&)            = delete;
	CpOpTimer& operator=(const CpOpTimer&) = delete;

private:
	const char* m_name;
	uint64_t    m_start;
};
} // namespace

// KYTY_CP_GAP_STATS: an op's category (drawPrep/cpGaps.h).
static CpGaps::Cat OpGapCat(CpSeq::OpKind kind) {
	using CpSeq::OpKind;
	switch (kind) {
		case OpKind::DrawIndex:
		case OpKind::DrawAuto:
		case OpKind::DrawIndirect:
		case OpKind::DrawIndirectMulti: return CpGaps::Cat::DrawOp;
		case OpKind::DispatchDirect:
		case OpKind::DispatchIndirect: return CpGaps::Cat::DispatchOp;
		case OpKind::EndOfPipe:
		case OpKind::ReleaseMem: return CpGaps::Cat::EndOfPipe;
		case OpKind::EventWrite: return CpGaps::Cat::EventWrite;
		case OpKind::WriteData: return CpGaps::Cat::WriteData;
		case OpKind::DmaData: return CpGaps::Cat::DmaData;
		case OpKind::WaitRegMem:
		case OpKind::WaitFlipDone: return CpGaps::Cat::WaitOp;
		case OpKind::Flip: return CpGaps::Cat::FlipOp;
		default: return CpGaps::Cat::OtherOp;
	}
}

CpSeq::Result CommandProcessor::ExecuteOp(CpSeq::OpKind kind, const void* payload,
                                          const void* data) {
	using CpSeq::OpKind;
	std::optional<CpOpTimer> op_timer;
	const CpGaps::Scope      gap(OpGapCat(kind));
	if (CpOpStatsEnabled()) [[unlikely]] {
		op_timer.emplace(kind == OpKind::EventWrite
		                     ? EventWriteActivityName(
		                           static_cast<const CpSeq::EventWriteOp*>(payload)->event_type)
		                     : CpSeq::OpKindName(kind));
	}
	// KYTY_DRAW_RUN (drawPrep/drawRun.h): every operation but a direct draw (whose commit keeps its
	// own run bookkeeping) and pure control flow is other command-processor work, which ends a run.
	// An occlusion dump that keeps the rendering instance (KYTY_OCCLUSION_SPLIT) records only query
	// commands: ExecEventWrite notes the activity when it ends the instance or waits instead.
	// KYTY_DRAW_RUN_QUIET_OPS: so does ExecWriteData for a WRITE_DATA done on the GPU timeline.
	// KYTY_DRAW_RUN_QUIET_SYNTHETIC: so does a synthetic occlusion dump (KYTY_GPU_OCCLUSION=0), a
	// CPU write of an always-visible result exactly like a CPU WRITE_DATA.
	const bool occlusion_dump =
	    kind == OpKind::EventWrite &&
	    static_cast<const CpSeq::EventWriteOp*>(payload)->event_type == 0x39u;
	const bool quiet =
	    DrawRun::QuietOpsEnabled() &&
	    (kind == OpKind::WriteData ||
	     (occlusion_dump && OcclusionCounter::Enabled() && OcclusionCounter::SplitEnabled()) ||
	     (occlusion_dump && !OcclusionCounter::Enabled() && DrawRun::QuietSyntheticEnabled()));
	// KYTY_DRAW_RUN_INDIRECT: so does a prepared single-record indirect draw (an engine commit).
	const bool indirect_commit =
	    kind == OpKind::DrawIndirect && DrawRun::IndirectRunsEnabled() &&
	    (static_cast<const CpSeq::DrawIndirectOp*>(payload)->flags & CpSeq::IndirectFlagPrepared) != 0;
	if (DrawRun::Enabled() && kind != OpKind::DrawIndex && kind != OpKind::DrawAuto &&
	    kind != OpKind::ReadCheck && kind != OpKind::CondExec && kind != OpKind::Branch &&
	    !quiet && !indirect_commit) {
		DrawRun::NoteForeignActivity(
		    kind == OpKind::EventWrite
		        ? EventWriteActivityName(static_cast<const CpSeq::EventWriteOp*>(payload)->event_type)
		        : CpSeq::OpKindName(kind));
	}
	// KYTY_DISPATCH_OVERLAP (render.h): any operation but a direct dispatch (and the verifier's read
	// checks) may order the dispatches around it: draws, synchronization, labels, waits, flips.
	if (kind != OpKind::DispatchDirect && kind != OpKind::ReadCheck && GetScheduler().Active()) {
		GetScheduler().Current().BreakDispatchOverlap();
	}
	CommitStats::StepMark(CommitStats::Step::GapOpHead);
	switch (kind) {
		case OpKind::DrawIndex: ExecDrawIndex(*static_cast<const CpSeq::DrawIndexOp*>(payload)); break;
		case OpKind::DrawAuto: ExecDrawAuto(*static_cast<const CpSeq::DrawAutoOp*>(payload)); break;
		case OpKind::DrawIndirect:
			ExecDrawIndirect(*static_cast<const CpSeq::DrawIndirectOp*>(payload));
			break;
		case OpKind::DrawIndirectMulti:
			ExecDrawIndirectMulti(*static_cast<const CpSeq::DrawIndirectOp*>(payload));
			break;
		case OpKind::DispatchDirect:
			ExecDispatchDirect(*static_cast<const CpSeq::DispatchDirectOp*>(payload));
			break;
		case OpKind::DispatchIndirect:
			ExecDispatchIndirect(*static_cast<const CpSeq::DispatchIndirectOp*>(payload));
			break;
		case OpKind::EndOfPipe: ExecEndOfPipe(*static_cast<const CpSeq::EndOfPipeOp*>(payload)); break;
		case OpKind::ReleaseMem:
			ExecReleaseMem(*static_cast<const CpSeq::ReleaseMemOp*>(payload));
			break;
		case OpKind::EventWrite:
			ExecEventWrite(*static_cast<const CpSeq::EventWriteOp*>(payload));
			break;
		case OpKind::WriteData:
			ExecWriteData(*static_cast<const CpSeq::WriteDataOp*>(payload),
			              static_cast<const uint32_t*>(data));
			break;
		case OpKind::ReferenceClock:
			ExecReferenceClock(*static_cast<const CpSeq::ReferenceClockOp*>(payload));
			break;
		case OpKind::DmaData: ExecDmaData(*static_cast<const CpSeq::DmaDataOp*>(payload)); break;
		case OpKind::LodStats: ExecLodStats(*static_cast<const CpSeq::LodStatsOp*>(payload)); break;
		case OpKind::Flip: ExecFlip(*static_cast<const CpSeq::FlipOp*>(payload)); break;
		case OpKind::WaitRegMem:
			return ExecWaitRegMem(*static_cast<const CpSeq::WaitRegMemOp*>(payload));
		case OpKind::WaitFlipDone:
			return ExecWaitFlipDone(*static_cast<const CpSeq::WaitFlipDoneOp*>(payload));
		case OpKind::DumpConstRam:
			ExecDumpConstRam(*static_cast<const CpSeq::DumpConstRamOp*>(payload),
			                 static_cast<const uint32_t*>(data));
			break;
		case OpKind::Predication:
			return ExecPredication(*static_cast<const CpSeq::PredicationOp*>(payload));
		case OpKind::CondExec: return ExecCondExec(*static_cast<const CpSeq::CondExecOp*>(payload));
		case OpKind::Branch: return ExecBranch(*static_cast<const CpSeq::BranchOp*>(payload));
		case OpKind::ReadCheck: break; // compared by the verifier only
		case OpKind::LockstepRead:
			return ExecLockstepRead(*static_cast<const CpSeq::LockstepReadOp*>(payload));
		case OpKind::StreamBegin:
		case OpKind::StreamEnd:
		case OpKind::Handoff:
		case OpKind::SkipSlots: EXIT("stream ops are consumed by ResolveSubmission\n"); break;
		case OpKind::Count: EXIT("invalid op kind\n");
	}
	return {};
}

bool CommandProcessor::CondExec(uint64_t address) {
	CpSeq::CondExecOp op;
	op.address        = address;
	const auto result = Submit(op);
	if (result.suspended) {
		// Reference front: the condition arrives with the resolver's answer.
		SuspendPm4();
		return false;
	}
	return result.value != 0;
}

CpSeq::Result CommandProcessor::ExecCondExec(const CpSeq::CondExecOp& op) {
	return {false, *reinterpret_cast<const volatile uint32_t*>(op.address) != 0 ? 1u : 0u};
}

bool CommandProcessor::Branch(uint64_t compare_addr, uint64_t mask, uint64_t reference,
                              uint32_t function) {
	CpSeq::BranchOp op;
	op.compare_addr   = compare_addr;
	op.mask           = mask;
	op.reference      = reference;
	op.function       = function;
	const auto result = Submit(op);
	if (result.suspended) {
		SuspendPm4();
		return false;
	}
	return result.value != 0;
}

CpSeq::Result CommandProcessor::ExecBranch(const CpSeq::BranchOp& op) {
	const bool taken =
	    TestWaitRegMemValue(ReadGuestForCp<uint64_t>(op.compare_addr), op.reference, op.mask,
	                        op.function);
	return {false, taken ? 1u : 0u};
}

const uint32_t* CommandProcessor::ReadRegisterPairs(uint64_t address, uint32_t num_regs) {
	m_register_pairs.resize(static_cast<size_t>(num_regs) * 2u);
	const auto bytes = static_cast<uint64_t>(m_register_pairs.size()) * sizeof(uint32_t);
	if (m_front_mode == FrontMode::Thread) {
		// The sequencer: the backing when the pages were never GPU-touched and no pending CP
		// write covers them, else the resolver reads them in order.
		if (!ReadGuestForFront(address, bytes, m_register_pairs.data())) {
			std::memset(m_register_pairs.data(), 0, bytes); // stopping
		}
	} else if (m_front_mode == FrontMode::Prefetch) {
		// P3c: the same bytes directly, or the speculative parse stops after this packet.
		if (!ReadGuestForPrefetch(address, bytes, m_register_pairs.data())) {
			std::memset(m_register_pairs.data(), 0, bytes);
		}
	} else {
		ReadGuestForCp(address, bytes, m_register_pairs.data());
	}
	// P3c: the bytes a parse after a wait read are part of its draws' adoption key.
	TrackAdoptInputs(m_register_pairs.data(), bytes, address);
	// KYTY_CP_SEQ_VERIFY: the front's read is compared with the reference front's serial read.
	const bool check =
	    m_front_mode == FrontMode::Reference ||
	    (m_front_mode == FrontMode::Thread && CpSeq::VerifyMode() != 0) ||
	    (m_front_mode == FrontMode::Inline && m_verifier != nullptr &&
	     g_current_execution != nullptr && m_verifier->Follows(g_current_execution->m_stream_id));
	if (check) {
		CpSeq::ReadCheckOp op;
		op.address = address;
		op.size    = bytes;
		op.hash    = XXH3_64bits(m_register_pairs.data(), bytes);
		(void)Submit(op);
	}
	return m_register_pairs.data();
}

void CommandProcessor::ReleaseMem(const uint32_t* body) {
	CpSeq::ReleaseMemOp op;
	std::memcpy(op.body, body, 7 * sizeof(uint32_t));
	(void)Submit(op);
}

void CommandProcessor::GetLodStats(const uint32_t* body) {
	CpSeq::LodStatsOp op;
	std::memcpy(op.body, body, sizeof(op.body));
	(void)Submit(op);
}

void CommandProcessor::SetCeComplete(bool complete) {
	m_ce_complete = complete;
	if (m_verifier != nullptr) {
		m_verifier->MirrorCeComplete(complete);
	}
}

bool CommandProcessor::ReferenceStep(Pm4Execution& execution) {
	EXIT_IF(m_front_mode != FrontMode::Reference);
	struct Scope {
		Scope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~Scope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}
		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} scope(*this, execution);
	execution.m_suspended = false;
	(void)ProcessPacket(execution);
	return execution.m_suspended;
}

void CommandProcessor::CopyFrontState(const CommandProcessor& from) {
	m_ctx                              = from.m_ctx;
	m_saved_ctx                        = from.m_saved_ctx;
	m_context_state_pushed             = from.m_context_state_pushed;
	m_ucfg                             = from.m_ucfg;
	m_sh_ctx                           = from.m_sh_ctx;
	m_user_data_marker                 = from.m_user_data_marker;
	m_index_type_and_size              = from.m_index_type_and_size;
	m_index_buffer_size                = from.m_index_buffer_size;
	m_index_base_addr                  = from.m_index_base_addr;
	m_draw_indirect_args_base_addr     = from.m_draw_indirect_args_base_addr;
	m_dispatch_indirect_args_base_addr = from.m_dispatch_indirect_args_base_addr;
	m_front_num_instances              = from.m_front_num_instances;
	m_front_instances_known            = from.m_front_instances_known;
	m_de_count                         = from.m_de_count;
	m_ce_count                         = from.m_ce_count;
	m_ce_complete                      = from.m_ce_complete;
	std::memcpy(m_const_ram, from.m_const_ram, sizeof(m_const_ram));
	m_flip           = from.m_flip;
	m_predicate_skip = from.m_predicate_skip;
}

// ---- P3b: the sequencer thread (KYTY_CP_SEQ=1, cpSequencer.h) --------------------------------

// Every CPU write of guest memory by the command processor is logged while the sequencer runs:
// draw preparations then span the packets between ordering points, and a preparation that read
// the written bytes must fail its certificate at commit (drawPrep.h, P3-SEQUENCER.md 2.6).
void NoteCpWrite(uint64_t address, uint64_t size) {
	if (CpSeq::ConfiguredMode() == CpSeq::Mode::Thread && address != 0 && size != 0) {
		Coherence::NoteContentWrite(address, size, Coherence::Source::CpWrite);
	}
	// KYTY_LABEL_MODE=split: a later CP write replaces a pending label's value in the CP's view.
	if (g_gpu_state != nullptr && address != 0 && size != 0) {
		g_gpu_state->DropCpVisibleLabels(address, size);
	}
}

void CommandProcessor::AttachSequencer(CpSeq::Sequencer* sequencer) {
	EXIT_IF(sequencer == nullptr || m_sequencer != nullptr || IsAsyncComputeQueue());
	m_sequencer  = sequencer;
	m_front_mode = FrontMode::Thread;
	m_ops.reset(new CpSeq::OpStream(OpRingBytes()));
	m_ops->SetSpinNs(CpSeq::SequencerSpinNs());
	m_emit_ops = m_ops.get(); // Astro Bot PC (KYTY_CP_PAR): context A's ring, as before
	if (CpPar::Enabled() && CpSeq::VerifyMode() == 0) {
		// Astro Bot PC (KYTY_CP_PAR=1a): resolver context B's ring (cpParallel.h).
		m_ops_b.reset(new CpSeq::OpStream(OpRingBytes()));
		m_ops_b->SetSpinNs(CpSeq::SequencerSpinNs());
		// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): contexts C, D (KYTY_CP_PAR_LANES).
		for (uint32_t lane = 2; lane < CpPar::RequestedLanes(); lane++) {
			m_ops_more[lane].reset(new CpSeq::OpStream(OpRingBytes()));
			m_ops_more[lane]->SetSpinNs(CpSeq::SequencerSpinNs());
		}
	}
	// The draw-prep engine's producer is the sequencer from now on: create it before either
	// thread can look at it.
	(void)DrawPrepEngine();
	if (m_draw_prep != nullptr && !m_draw_prep->Parallel() &&
	    DrawPrep::GetMode() != DrawPrep::Mode::Off) {
		std::printf("Kyty CP sequencer: KYTY_DRAW_PREP=inline prepares on the resolver\n");
	}
	if (m_ops_b != nullptr && CpPar::SeparateWindows()) {
		// Astro Bot PC (KYTY_CP_PAR=1b1): context B's own snapshot ring and, with parallel draw
		// prep, its own engine (window and workers), created before either thread looks at them.
		sequencer->AddSnapshotRings(CpPar::RequestedLanes()); // Astro Bot PC (2a: N lanes)
		if (m_draw_prep != nullptr && m_draw_prep->Parallel()) {
			auto make_engine = [this](uint32_t lane) {
				return new DrawPrep::Engine(
				    m_renderer,
				    [] {
					    if (g_gpu_state != nullptr && !(CpPar::Concurrent() && CpPar::RunAheadDirty())) { // Astro Bot PC (1b3)
						    g_gpu_state->ProcessCommands();
					    }
				    },
				    [this] { MaybeFlushIdleGpu(); }, lane);
			};
			m_draw_prep_b.reset(make_engine(1));
			for (uint32_t lane = 2; lane < CpPar::RequestedLanes(); lane++) {
				m_draw_prep_more[lane].reset(make_engine(lane)); // Astro Bot PC (2a: N lanes)
			}
		}
		std::printf("Kyty CpPar: context B has its own snapshot ring (%u entries)%s\n",
		            sequencer->Snapshots(1).Capacity(),
		            m_draw_prep_b != nullptr ? " and draw-prep window/workers"
		                                     : " (draw prep not parallel: no window)");
	}
}

void CommandProcessor::DetachSequencer() {
	EXIT_IF(m_front_mode != FrontMode::Thread || m_sequencer == nullptr);
	CpSeq::OpView view;
	EXIT_IF(m_ops->Peek(view)); // every op executed
	EXIT_IF(m_ops_b != nullptr && m_ops_b->Peek(view)); // Astro Bot PC (KYTY_CP_PAR)
	for (const auto& ops: m_ops_more) {
		EXIT_IF(ops != nullptr && ops->Peek(view)); // Astro Bot PC (KYTY_CP_PAR=2a, N lanes)
	}
	m_front_mode = FrontMode::Direct;
	m_sequencer  = nullptr;
	m_pending_writes.clear();
}

void CommandProcessor::BeginHandoff() {
	EXIT_IF(m_front_mode != FrontMode::Thread);
	m_front_mode = FrontMode::Direct;
}

void CommandProcessor::EndHandoff(uint64_t submission) {
	EXIT_IF(m_front_mode != FrontMode::Direct || m_sequencer == nullptr);
	// The direct path drained its draws at the slice's end.
	EXIT_IF(m_draw_prep != nullptr && m_draw_prep->Pending());
	m_front_mode = FrontMode::Thread;
	m_sequencer->NoteHandoffDone(submission);
}

bool CommandProcessor::CpWriteRange(CpSeq::OpKind kind, const void* payload, uint64_t& begin,
                                    uint64_t& end) {
	using CpSeq::OpKind;
	uint64_t address = 0;
	uint64_t size    = 0;
	switch (kind) {
		case OpKind::WriteData: {
			const auto& op = *static_cast<const CpSeq::WriteDataOp*>(payload);
			const bool  one_address = ((op.write_control >> 16u) & 0x1u) != 0;
			address                 = op.dst;
			size = one_address ? sizeof(uint32_t) : uint64_t {op.dw_num} * sizeof(uint32_t);
			break;
		}
		case OpKind::DumpConstRam: {
			const auto& op = *static_cast<const CpSeq::DumpConstRamOp*>(payload);
			address        = op.dst;
			size           = uint64_t {op.dw_num} * sizeof(uint32_t);
			break;
		}
		case OpKind::ReferenceClock: {
			const auto& op = *static_cast<const CpSeq::ReferenceClockOp*>(payload);
			address        = op.dst;
			size           = op.num_bytes;
			break;
		}
		case OpKind::EndOfPipe: {
			const auto& op = *static_cast<const CpSeq::EndOfPipeOp*>(payload);
			address        = op.dst;
			// A GDS read (32-bit form, source 1) writes value >> 16 dwords.
			size = op.size == sizeof(uint32_t) && op.event_write_source == 1u
			           ? uint64_t {static_cast<uint32_t>(op.value) >> 16u} * sizeof(uint32_t)
			           : 8u;
			break;
		}
		case OpKind::ReleaseMem: {
			const auto& op = *static_cast<const CpSeq::ReleaseMemOp*>(payload);
			address        = op.body[2] | (uint64_t {op.body[3]} << 32u);
			const auto data_sel = (op.body[1] >> 29u) & 0x7u;
			size = data_sel == 5u ? uint64_t {op.body[4] >> 16u} * sizeof(uint32_t) : 8u;
			break;
		}
		case OpKind::Flip: {
			const auto& op = *static_cast<const CpSeq::FlipOp*>(payload);
			address        = op.dst;
			size           = sizeof(uint32_t);
			break;
		}
		case OpKind::EventWrite: {
			const auto& op = *static_cast<const CpSeq::EventWriteOp*>(payload);
			if (op.event_type == 0x39u) {
				address = op.address;
				size    = 16u * 2u * sizeof(uint64_t); // one begin/end pair per DB
			}
			break;
		}
		case OpKind::LodStats: {
			const auto& op = *static_cast<const CpSeq::LodStatsOp*>(payload);
			address = (op.body[1] & 0xffffffc0u) | (uint64_t {op.body[2]} << 32u);
			size    = op.body[0];
			break;
		}
		case OpKind::DmaData: {
			const auto& op = *static_cast<const CpSeq::DmaDataOp*>(payload);
			// Memory destinations (0, 3); GDS (1) and nowhere (2) are not guest memory.
			if (op.dst_sel == 0u || op.dst_sel == 3u) {
				address = op.dst;
				size    = op.num_bytes;
			}
			break;
		}
		default: break;
	}
	if (address == 0 || size == 0) {
		return false;
	}
	begin = address;
	end   = size > UINT64_MAX - address ? UINT64_MAX : address + size;
	return true;
}

// KYTY_CP_WAIT_JOURNAL (live switch; thread mode without the verifier). Read by the sequencer at
// each wait: either answer is valid at any wait.
static Live::Switch g_cp_wait_journal("KYTY_CP_WAIT_JOURNAL", Live::ParseDefaultOff);
static bool WaitJournalEnabled() {
	return g_cp_wait_journal.Get() != 0 && CpSeq::VerifyMode() == 0;
}

CpSeq::Result CommandProcessor::SubmitThread(CpSeq::OpKind kind, const void* payload,
                                             uint32_t payload_size, const void* data,
                                             uint32_t data_size) {
	// Only the sequencer emits; an op's execution on the resolver never emits another op.
	EXIT_IF(!g_sequencer_thread || m_sequencer == nullptr);
	auto*      execution = g_current_execution;
	const auto sequence  = m_emit_ops->Emitted(); // Astro Bot PC (KYTY_CP_PAR)
	uint16_t   flags     = 0;
	if (m_epoch_pending) {
		flags |= CpSeq::FlagAdvanceEpoch;
		m_epoch_pending = false;
	}
	uint64_t begin = 0;
	uint64_t end   = 0;
	if (CpWriteRange(kind, payload, begin, end)) {
		if (kind == CpSeq::OpKind::EndOfPipe || kind == CpSeq::OpKind::ReleaseMem) {
			m_last_label_begin = begin;
			m_last_label_end   = end;
			NoteLastLabel(kind, payload);
		} else {
			// Another CP write since the label: a later wait is not the plain idiom.
			m_last_label_known = false;
		}
		// Command bytes under it are parsed only after it executed (CheckCommandBytes).
		if (m_pending_writes.size() >= 64u) {
			const auto executed = m_sequencer->Executed();
			std::erase_if(m_pending_writes,
			              [executed](const PendingWrite& write) { return write.op < executed; });
		}
		m_pending_writes.push_back({begin, end, sequence});
		if (execution != nullptr) {
			for (auto& cursor: execution->m_buffer_stack) {
				const auto rest_begin =
				    cursor.copy != nullptr ? cursor.guest_address + uint64_t {cursor.offset_dw} * 4u
				                           : reinterpret_cast<uint64_t>(cursor.commands.data() +
				                                                        cursor.offset_dw);
				const auto rest_end =
				    rest_begin + (cursor.commands.size() - cursor.offset_dw) * sizeof(uint32_t);
				if (begin < rest_end && rest_begin < end) {
					cursor.checked_epoch = 0;
				}
			}
		}
	}
	if (CpSeq::IsLockstep(kind)) {
		// P3c: speculative slots the real parse has not adopted by an ordering point are stale (a
		// speculation stops at the first one); the resolver retires them before this op.
		DropSpeculativeSlots();
	}
	if (CpPar::Concurrent()) {
		// Astro Bot PC (KYTY_CP_PAR=1b3, cpParallel.h): the quiet tail of this submission for the
		// oldest context: after its last op that may submit on purpose, wait, flip, need a service
		// command or write guest memory a later submission's draws may read at record time.
		using CpSeq::OpKind;
		bool quiet = false;
		if (CpPar::Wide()) {
			// Astro Bot PC (KYTY_CP_PAR=1b4): only the hard ops end the interleaving (cpParallel.h).
			quiet = !CpParHardForOldest(kind, payload);
		} else switch (kind) {
			case OpKind::DrawIndex:
			case OpKind::DrawAuto:
			case OpKind::DispatchDirect:
			case OpKind::EndOfPipe:
			case OpKind::ReleaseMem:
			case OpKind::StreamBegin:
			case OpKind::StreamEnd:
			case OpKind::ReadCheck: quiet = true; break;
			case OpKind::EventWrite:
				// Not an occlusion dump (queries, possible waits).
				quiet = static_cast<const CpSeq::EventWriteOp*>(payload)->event_type != 0x39u;
				break;
			default: break;
		}
		if (kind == OpKind::StreamBegin) {
			m_cp_par_last_sync = 0;
		} else if (!quiet) {
			m_cp_par_last_sync = sequence + 1u;
		}
		if (kind == OpKind::StreamEnd) {
			CpPar::PublishStreamEnd(static_cast<const CpSeq::StreamEndOp*>(payload)->submission,
			                        m_cp_par_last_sync);
		}
	}
	(void)m_emit_ops->Emit(kind, payload, payload_size, data, data_size, // Astro Bot PC (KYTY_CP_PAR)
	                  execution != nullptr ? execution->m_packets : 0,
	                  execution != nullptr ? execution->m_packets_hash : 0,
	                  CpSeq::VerifyMode() != 0, flags);
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqOps);
	if (!CpSeq::IsLockstep(kind)) {
		return {};
	}
	// Diagnostics: the barrier's kind (lockstep reads count CpSeqLockstepReads/Buffers).
	switch (kind) {
		case CpSeq::OpKind::WaitRegMem: {
			const auto address = static_cast<const CpSeq::WaitRegMemOp*>(payload)->addr;
			Profiler::CountFrameEvent(address >= m_last_label_begin && address < m_last_label_end
			                              ? Profiler::FrameEvent::CpSeqBarrierWaitSelfLabel
			                              : Profiler::FrameEvent::CpSeqBarrierWaitOther);
			break;
		}
		case CpSeq::OpKind::WaitFlipDone:
			Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqBarrierFlipWait);
			break;
		case CpSeq::OpKind::Predication:
		case CpSeq::OpKind::CondExec:
		case CpSeq::OpKind::Branch:
			Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqBarrierCondition);
			break;
		default: break;
	}
	// KYTY_CP_WAIT_JOURNAL=1 (default off): a WAIT_REG_MEM on the label this stream just wrote,
	// with no other CP write since, whose written value passes the compare, is answered from that
	// pending write instead of stopping the parse: the resolver executes the wait in order (it
	// suspends there itself if the label is late), and the front keeps parsing, so its draws reach
	// the draw-prep window while the resolver catches up. The front's own guest reads stay ordered
	// (pending CP writes and GPU-touched bytes are read by the resolver in order). Thread mode only.
	if (kind == CpSeq::OpKind::WaitRegMem && WaitJournalEnabled() &&
	    IsSelfLabelWait(*static_cast<const CpSeq::WaitRegMemOp*>(payload))) {
		static std::atomic<uint64_t> passed {0};
		const auto count = passed.fetch_add(1, std::memory_order_relaxed) + 1;
		if (count == 1 || (count & 0xffffu) == 0) {
			std::printf("CP wait journal: %" PRIu64 " self-label waits passed without stopping the parse\n",
			            count);
		}
		m_adopt_tracking = false;
		return {false, 0};
	}
	// P3c (KYTY_CP_SEQ_PREFETCH): at a wait on the label this stream just wrote, parse ahead on a
	// copy of the front while the resolver catches up, publishing the draws it meets.
	const bool prefetch = kind == CpSeq::OpKind::WaitRegMem && CpSeq::PrefetchMode() != 0 &&
	                      IsSelfLabelWait(*static_cast<const CpSeq::WaitRegMemOp*>(payload));
	if (prefetch) {
		RunPrefetch(sequence);
	}
	// An ordering point: the resolver executes everything before it, then this op.
	const auto value = m_sequencer->AwaitAnswer(sequence);
	m_pending_writes.clear();
	m_barrier_epoch++;
	if (EmitPrep() != nullptr) {
		EmitPrep()->NoteStop(1);
	}
	// The real parse after the wait: its draws are matched with the speculative slots by the
	// packets and bytes it consumes from here (the wait packet itself completes first).
	m_adopt_tracking = prefetch && EmitPrep() != nullptr && EmitPrep()->SpeculativeSlots() != 0;
	m_adopt_skip_one = true;
	m_adopt_packets  = 0;
	m_adopt_inputs   = 0;
	return {false, value};
}

bool CommandProcessor::AwaitPendingWrites(uint64_t begin, uint64_t end) {
	uint64_t needed = 0; // op sequence + 1 that must have executed
	for (const auto& write: m_pending_writes) {
		if (write.begin < end && begin < write.end) {
			needed = std::max(needed, write.op + 1u);
		}
	}
	if (needed == 0 || m_sequencer->Executed() >= needed) {
		return true;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqPendingWriteStops);
	if (!m_sequencer->Wait([this, needed] { return m_sequencer->Executed() >= needed; }, needed,
	                       needed)) {
		return false;
	}
	const auto executed = m_sequencer->Executed();
	std::erase_if(m_pending_writes,
	              [executed](const PendingWrite& write) { return write.op < executed; });
	m_barrier_epoch++;
	if (EmitPrep() != nullptr) {
		EmitPrep()->NoteStop(2);
	}
	return true;
}

bool CommandProcessor::ReadGuestForFront(uint64_t address, uint64_t size, void* dst) {
	if (size == 0) {
		return true;
	}
	if (!AwaitPendingWrites(address, address + size)) {
		return false;
	}
	if (!GpuTouched::g_pages.AnyTouched(address, address + size)) {
		// Never GPU-touched: the backing is authoritative and the read cannot fault
		// (gpuTouchedPages.h); the serial CP would read the same bytes (P3-SEQUENCER.md 2.5).
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqDirectReads);
		std::memcpy(dst, reinterpret_cast<const void*>(address), size);
		return true;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqLockstepReads);
	CpSeq::LockstepReadOp op;
	op.address     = address;
	op.size        = size;
	op.destination = reinterpret_cast<uint64_t>(dst);
	(void)SubmitThread(CpSeq::OpKind::LockstepRead, &op, sizeof(op), nullptr, 0);
	return !m_sequencer->Stopping();
}

CpSeq::Result CommandProcessor::ExecLockstepRead(const CpSeq::LockstepReadOp& op) {
	ReadGuestForCp(op.address, op.size, reinterpret_cast<void*>(op.destination));
	return {};
}

bool CommandProcessor::CheckCommandBytes(Pm4Execution::BufferCursor& cursor) {
	const auto rest_dw    = cursor.commands.size() - cursor.offset_dw;
	const auto rest_begin = cursor.copy != nullptr
	                            ? cursor.guest_address + uint64_t {cursor.offset_dw} * 4u
	                            : reinterpret_cast<uint64_t>(cursor.commands.data() +
	                                                         cursor.offset_dw);
	const auto rest_end = rest_begin + rest_dw * sizeof(uint32_t);
	if (!AwaitPendingWrites(rest_begin, rest_end)) {
		return false;
	}
	if (cursor.copy == nullptr && !GpuTouched::g_pages.AnyTouched(rest_begin, rest_end)) {
		cursor.checked_epoch = m_barrier_epoch;
		return true;
	}
	if (cursor.copy != nullptr && cursor.checked_epoch != 0) {
		// A copy taken in order stays what the serial CP reads, unless a CP write since then
		// covered it (checked_epoch 0, below).
		cursor.checked_epoch = m_barrier_epoch;
		return true;
	}
	// GPU-touched command bytes (or a copy a CP write made stale): the resolver copies the rest
	// of the buffer at this point of the stream.
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqLockstepBuffers);
	auto copy = std::make_shared<std::vector<uint32_t>>(rest_dw);
	CpSeq::LockstepReadOp op;
	op.address     = rest_begin;
	op.size        = rest_dw * sizeof(uint32_t);
	op.destination = reinterpret_cast<uint64_t>(copy->data());
	(void)SubmitThread(CpSeq::OpKind::LockstepRead, &op, sizeof(op), nullptr, 0);
	if (m_sequencer->Stopping()) {
		return false;
	}
	cursor.commands      = {copy->data(), rest_dw};
	cursor.offset_dw     = 0;
	cursor.guest_address = rest_begin;
	cursor.copy          = std::move(copy);
	cursor.checked_epoch = m_barrier_epoch;
	return true;
}

uint32_t CommandProcessor::TakeSnapshot() {
	auto& ring = EmitSnapshots();
	if (!ring.HasFree()) {
		const auto wake_at = OldestPendingOp(m_snapshot_ops);
		if (!m_sequencer->Wait([&ring] { return ring.HasFree(); }, wake_at, wake_at)) {
			return 0; // stopping
		}
	}
	// The op this snapshot is for is the next one emitted.
	m_snapshot_ops.push_back(m_emit_ops->Emitted()); // Astro Bot PC (KYTY_CP_PAR)
	if (m_snapshot_ops.size() > 2u * ring.Capacity()) {
		(void)OldestPendingOp(m_snapshot_ops);
	}
	const auto index   = ring.Acquire();
	auto&      entry   = ring.Entry(index);
	entry.context      = m_ctx;
	entry.user_config  = m_ucfg;
	entry.shaders      = m_sh_ctx;
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqSnapshots);
	return index;
}

CpSeq::RegisterState* CommandProcessor::OpSnapshot(CpSeq::OpKind kind, const void* payload) {
	using CpSeq::OpKind;
	uint32_t index = UINT32_MAX;
	switch (kind) {
		case OpKind::DrawIndex: {
			const auto& op = *static_cast<const CpSeq::DrawIndexOp*>(payload);
			index          = (op.flags & CpSeq::DrawFlagSnapshot) != 0 ? op.snapshot : UINT32_MAX;
			break;
		}
		case OpKind::DrawAuto: {
			const auto& op = *static_cast<const CpSeq::DrawAutoOp*>(payload);
			index          = (op.flags & CpSeq::DrawFlagSnapshot) != 0 ? op.snapshot : UINT32_MAX;
			break;
		}
		case OpKind::DrawIndirect:
		case OpKind::DrawIndirectMulti: {
			const auto& op = *static_cast<const CpSeq::DrawIndirectOp*>(payload);
			index = (op.flags & CpSeq::IndirectFlagSnapshot) != 0 ? op.snapshot : UINT32_MAX;
			break;
		}
		case OpKind::DispatchDirect: {
			const auto& op = *static_cast<const CpSeq::DispatchDirectOp*>(payload);
			index = (op.flags & CpSeq::DispatchFlagSnapshot) != 0 ? op.snapshot : UINT32_MAX;
			break;
		}
		case OpKind::DispatchIndirect: {
			const auto& op = *static_cast<const CpSeq::DispatchIndirectOp*>(payload);
			index = (op.flags & CpSeq::DispatchFlagSnapshot) != 0 ? op.snapshot : UINT32_MAX;
			break;
		}
		default: break;
	}
	if (index == UINT32_MAX || m_sequencer == nullptr) {
		return nullptr;
	}
	return &ResolveSnapshots().Entry(index);
}

uint64_t CommandProcessor::OldestPendingOp(std::deque<uint64_t>& ops) const {
	const auto executed = m_sequencer->Executed();
	while (!ops.empty() && ops.front() < executed) {
		ops.pop_front();
	}
	return ops.empty() ? 0u : ops.front() + 1u;
}

bool CommandProcessor::WaitForWindowSpace() {
	auto*      engine  = EmitPrep();
	const auto wake_at = OldestPendingOp(m_published_ops);
	return m_sequencer->Wait([engine] { return engine->WindowHasSpace(); }, wake_at, wake_at);
}

bool CommandProcessor::SequenceSubmission(const CpSeq::Intake& intake) {
	EXIT_IF(m_sequencer == nullptr || m_front_mode != FrontMode::Thread);
	g_sequencer_thread = true;
	// Like a draw-prep worker, the sequencer never reaches the scheduler or the caches:
	// CommandScheduler::CheckActive stops the emulator if a front path does.
	DrawPrep::t_worker_thread = true;
	auto& sequencer = *m_sequencer;
	// Astro Bot PC (KYTY_CP_PAR=1a): this submission's ops go to its resolver context's ring,
	// numbered after the ops of the other ring (one global op order). Off: always m_ops.
	auto* lane_ring = intake.context != 0 ? OpsOf(intake.context) : nullptr; // Astro Bot PC (2a)
	if (auto* ring = lane_ring != nullptr ? lane_ring : m_ops.get(); ring != m_emit_ops) {
		ring->SetNextSequence(m_emit_ops->Emitted());
		m_emit_ops = ring;
		if (CpPar::SeparateWindows()) {
			// Astro Bot PC (KYTY_CP_PAR=1b1): the draws and snapshots of this submission go to its
			// context's window and ring, with that context's pending-op lists (speculative slots
			// never outlive their stream: none is pending here).
			EXIT_IF(EmitPrep() != nullptr && EmitPrep()->SpeculativeSlots() != 0);
			// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): park the lists of the context left, take the
			// ones of the context entered (two contexts: the former single swap).
			const uint32_t entered = ring == m_ops.get() ? 0u : intake.context;
			std::swap(m_published_ops, m_published_ops_parked[m_emit_lane]);
			std::swap(m_snapshot_ops, m_snapshot_ops_parked[m_emit_lane]);
			std::swap(m_published_ops, m_published_ops_parked[entered]);
			std::swap(m_snapshot_ops, m_snapshot_ops_parked[entered]);
			m_emit_lane = entered;
		}
	}
	if (intake.graphics && intake.frame_fence != 0) {
		// Ordering point (the frame fence): the command bytes of a fenced submission are read only
		// once the resolver has started it, i.e. once the work the fence orders has completed.
		// Astro Bot PC (KYTY_CP_PAR=1b3) : une soeur d'une soumission deja demarree avec la meme barriere peut
		// etre lue tout de suite (sinon l'avance du second fil n'aurait jamais d'operations).
		const uint64_t fence_wait = intake.fence_first != 0 ? intake.fence_first : intake.sequence;
		if (!sequencer.Wait([&] { return sequencer.Started() >= fence_wait; })) {
			return false;
		}
		m_pending_writes.clear();
		m_barrier_epoch++;
	}
	// Diagnostics: a submission start is a stop of the sequencer when the resolver may have
	// caught up meanwhile (the next draw starts a burst, FrameEvent DrawPrepCommitWaitsStart).
	if (EmitPrep() != nullptr) {
		EmitPrep()->NoteStop(2);
	}
	m_last_label_known = false;
	// The front's part of GuestGpu::Process's first slice.
	if (intake.reset_processor) {
		Reset();
	}
	if (intake.graphics) {
		StateTrace::BeginSubmission(intake.sequence, intake.reset_processor); // Astro Bot PC : KYTY_STATE_TRACE
	}
	ResetDeCe();
	SetFlip({});
	m_ce_complete = intake.constant_commands.empty();

	const bool handoff = intake.graphics && !intake.constant_commands.empty();
	CpSeq::StreamBeginOp begin;
	begin.submission = intake.sequence;
	if (CpSeq::VerifyMode() != 0 && intake.graphics && !handoff) {
		// The reference front starts from this front state (owned by the op).
		auto* state = new CommandProcessor(m_renderer, m_interrupt_event_id);
		state->CopyFrontState(*this);
		begin.state  = reinterpret_cast<uint64_t>(state);
		begin.verify = 1;
	}
	(void)SubmitThread(CpSeq::OpKind::StreamBegin, &begin, sizeof(begin), nullptr, 0);
	if (!intake.graphics) {
		CpSeq::StreamEndOp end;
		end.submission = intake.sequence;
		(void)SubmitThread(CpSeq::OpKind::StreamEnd, &end, sizeof(end), nullptr, 0);
		return true;
	}
	if (handoff) {
		// Constant engine: the resolver parses and executes this submission itself, with this
		// front state, while this thread waits.
		CpSeq::HandoffOp op;
		op.submission = intake.sequence;
		(void)SubmitThread(CpSeq::OpKind::Handoff, &op, sizeof(op), nullptr, 0);
		if (!sequencer.Wait([&] { return sequencer.HandoffsDone() >= intake.sequence; })) {
			return false;
		}
		m_pending_writes.clear();
		m_barrier_epoch++;
		return true;
	}

	Pm4Execution execution;
	execution.m_buffer_stack.push_back({intake.commands});
	execution.m_stream_id = intake.sequence;
	struct Scope {
		Scope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~Scope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}
		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} scope(*this, execution);
	while (!execution.m_buffer_stack.empty()) {
		if (sequencer.Stopping()) {
			return false;
		}
		execution.m_suspended = false;
		if (ProcessPacket(execution) || !execution.m_suspended) {
			continue;
		}
		if (sequencer.Stopping()) {
			return false;
		}
		// A wait of the front alone (REWIND before its valid bit): the packet is parsed again,
		// from memory, shortly.
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqFrontWaits);
		std::this_thread::sleep_for(std::chrono::microseconds(50));
		m_barrier_epoch++;
	}
	// P3c: speculative slots are confined to their stream.
	DropSpeculativeSlots();
	CpSeq::StreamEndOp end;
	end.submission = intake.sequence;
	end.packets    = execution.m_packets;
	g_current_execution = nullptr; // the end op carries its own packet count
	(void)SubmitThread(CpSeq::OpKind::StreamEnd, &end, sizeof(end), nullptr, 0);
	return true;
}

// ---- P3c: prep prefetch past "wait for idle" waits (KYTY_CP_SEQ_PREFETCH, cpOps.h) -----------
//
// At a WAIT_REG_MEM on the label the stream has just written (end-of-pipe data, no other CP write
// since), the sequencer does not idle until the resolver has executed the wait: a copy of its
// front (FrontMode::Prefetch) parses the packets after the wait and publishes their draws to the
// draw-prep window, so the workers prepare them while the resolver still works through the ops
// before the wait. After the answer the real parse runs as before; each of its draws takes the
// next speculative slot if both parses consumed the same packets and bytes since the wait, else
// every speculative slot is handed to the resolver to retire unused (SkipSlots).
//
// Exactness (P3-SEQUENCER.md 2.5, 5 P3c):
// - The op stream is the real parse's, as without prefetch (the speculative front emits nothing).
// - The front state after a packet is a deterministic function of the state before it and the
//   bytes the packet and its register-indirect loads read. Both parses start from the front state
//   at the wait, so equal packets and bytes (the adoption key: their count and a 64-bit hash with
//   addresses) mean equal registers and draw arguments: the slot's preparation is the one a
//   publish by the real parse would get. Reads the speculative parse cannot prove to be what the
//   real parse will read (never GPU-touched, no pending CP write, its own ops' writes included)
//   stop it instead.
// - The preparation itself is certified at commit like every other: emulator writes between it
//   and the commit (CP writes, GPU dirty marks, publications, including those of the ops before
//   the wait) are in the coherence log. Guest CPU writes are not; none can be ordered before a
//   wait on the stream's own label (the guest can wait for the label but cannot delay the wait),
//   so such a write races the draws on the hardware as well (class E2). No speculation crosses
//   another ordering point: a lockstep op stops it, and it runs only at self-label waits.

void CommandProcessor::PrefetchDeleter::operator()(CommandProcessor* processor) const noexcept {
	delete processor;
}

void CommandProcessor::NoteLastLabel(CpSeq::OpKind kind, const void* payload) {
	m_last_label_known = false;
	if (kind == CpSeq::OpKind::EndOfPipe) {
		const auto& op = *static_cast<const CpSeq::EndOfPipeOp*>(payload);
		// Data labels (WriteAtEndOfPipe): source 2 (32- or 64-bit data), or source 1 in the 32-bit
		// event-type-4 form; not GDS reads or the reference clock.
		const bool data = op.event_write_source == 0x02u ||
		                  (op.event_write_source == 0x01u && op.size == sizeof(uint32_t) &&
		                   op.eop_event_type == 0x04u);
		if (data && (op.size == sizeof(uint32_t) || op.size == sizeof(uint64_t))) {
			m_last_label_value = op.size == sizeof(uint32_t) ? (op.value & 0xffffffffu) : op.value;
			m_last_label_size  = op.size;
			m_last_label_known = true;
		}
		return;
	}
	if (kind == CpSeq::OpKind::ReleaseMem) {
		const auto& op       = *static_cast<const CpSeq::ReleaseMemOp*>(payload);
		const auto  data_sel = (op.body[1] >> 29u) & 0x7u;
		if (data_sel == 1u || data_sel == 2u) {
			m_last_label_value = data_sel == 1u ? uint64_t {op.body[4]}
			                                    : (op.body[4] | (uint64_t {op.body[5]} << 32u));
			m_last_label_size  = data_sel == 1u ? 4u : 8u;
			m_last_label_known = true;
		}
	}
}

bool CommandProcessor::IsSelfLabelWait(const CpSeq::WaitRegMemOp& op) const {
	if (!m_last_label_known || op.addr != m_last_label_begin || op.size > m_last_label_size) {
		return false;
	}
	const auto value =
	    op.size == sizeof(uint32_t) ? (m_last_label_value & 0xffffffffu) : m_last_label_value;
	return TestWaitRegMemValue(value, op.ref, op.mask, op.func);
}

void CommandProcessor::CopyFrontStateForPrefetch(const CommandProcessor& from) {
	m_ctx                              = from.m_ctx;
	m_saved_ctx                        = from.m_saved_ctx;
	m_context_state_pushed             = from.m_context_state_pushed;
	m_ucfg                             = from.m_ucfg;
	m_sh_ctx                           = from.m_sh_ctx;
	m_user_data_marker                 = from.m_user_data_marker;
	m_index_type_and_size              = from.m_index_type_and_size;
	m_index_buffer_size                = from.m_index_buffer_size;
	m_index_base_addr                  = from.m_index_base_addr;
	m_draw_indirect_args_base_addr     = from.m_draw_indirect_args_base_addr;
	m_dispatch_indirect_args_base_addr = from.m_dispatch_indirect_args_base_addr;
	m_front_num_instances              = from.m_front_num_instances;
	m_front_instances_known            = from.m_front_instances_known;
	m_de_count                         = from.m_de_count;
	m_ce_count                         = from.m_ce_count;
	m_ce_complete                      = from.m_ce_complete;
	m_flip                             = from.m_flip;
	m_predicate_skip                   = from.m_predicate_skip;
}

void CommandProcessor::TrackAdoptPacket(const uint32_t* packet, uint32_t packet_dw,
                                        uint64_t guest_address) {
	if (!m_adopt_tracking) [[likely]] {
		return;
	}
	if (m_adopt_skip_one) {
		m_adopt_skip_one = false;
		return;
	}
	m_adopt_packets++;
	m_adopt_inputs = XXH3_64bits_withSeed(packet, uint64_t {packet_dw} * sizeof(uint32_t),
	                                      m_adopt_inputs ^ guest_address);
}

void CommandProcessor::TrackAdoptInputs(const void* data, uint64_t size, uint64_t guest_address) {
	if (!m_adopt_tracking) [[likely]] {
		return;
	}
	m_adopt_inputs =
	    XXH3_64bits_withSeed(data, size, m_adopt_inputs ^ guest_address ^ 0x9e3779b97f4a7c15ull);
}

uint64_t CommandProcessor::CurrentPacketKey() const {
	const auto* execution = g_current_execution;
	EXIT_IF(execution == nullptr || execution->m_buffer_stack.empty());
	const auto& cursor = execution->m_buffer_stack.back();
	const auto* packet = cursor.commands.data() + cursor.offset_dw;
	const auto  rest   = cursor.commands.size() - cursor.offset_dw;
	const auto  dw     = std::min<uint64_t>(KYTY_PM4_LEN(packet[0]), rest);
	const uint64_t guest = cursor.copy != nullptr
	                           ? cursor.guest_address + uint64_t {cursor.offset_dw} * 4u
	                           : reinterpret_cast<uint64_t>(packet);
	return XXH3_64bits_withSeed(packet, dw * sizeof(uint32_t), m_adopt_inputs ^ guest);
}

void CommandProcessor::EmitSkipSlots(uint64_t count) {
	const auto*  execution = g_current_execution;
	const auto   sequence  = m_emit_ops->Emitted(); // Astro Bot PC (KYTY_CP_PAR)
	CpSeq::SkipSlotsOp op;
	op.count = static_cast<uint32_t>(count);
	// Transport: no sync-epoch flag (it travels with the next real op) and no verify hash.
	(void)m_emit_ops->Emit(CpSeq::OpKind::SkipSlots, &op, sizeof(op), nullptr, 0, // Astro Bot PC (KYTY_CP_PAR)
	                  execution != nullptr ? execution->m_packets : 0,
	                  execution != nullptr ? execution->m_packets_hash : 0, false, 0);
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqOps);
	// The op frees these window slots (the sequencer's window-full wake threshold).
	for (uint64_t i = 0; i < count; i++) {
		m_published_ops.push_back(sequence);
	}
	while (m_published_ops.size() > 256u) {
		const auto before = m_published_ops.size();
		(void)OldestPendingOp(m_published_ops);
		if (m_published_ops.size() == before) {
			break;
		}
	}
}

void CommandProcessor::DropSpeculativeSlots() {
	m_adopt_tracking = false;
	if (EmitPrep() == nullptr || !EmitPrep()->Parallel()) {
		return;
	}
	if (const auto dropped = EmitPrep()->DropSpeculative(); dropped != 0) {
		EmitSkipSlots(dropped);
	}
}

template <typename Op>
bool CommandProcessor::AdoptSpeculativeDraw(Op& op) {
	auto* engine = EmitPrep();
	if (engine == nullptr || !engine->Parallel() || engine->SpeculativeSlots() == 0) {
		m_adopt_tracking = false;
		return false;
	}
	if (!m_adopt_tracking) {
		DropSpeculativeSlots();
		return false;
	}
	uint64_t position = 0;
	uint64_t dropped  = 0;
	switch (engine->TryAdopt(m_adopt_packets, CurrentPacketKey(), position, dropped)) {
		case DrawPrep::Engine::Adoption::Adopted:
			op.flags |= CpSeq::DrawFlagPublished;
			op.window = position;
			if (engine->SpeculativeSlots() == 0) {
				m_adopt_tracking = false;
			}
			return true;
		case DrawPrep::Engine::Adoption::Dropped:
			m_adopt_tracking = false;
			EmitSkipSlots(dropped);
			return false;
		case DrawPrep::Engine::Adoption::None: break;
	}
	m_adopt_tracking = false;
	return false;
}

void CommandProcessor::StopPrefetch(Profiler::FrameEvent reason) {
	if (!m_prefetch_stop) {
		m_prefetch_stop = true;
		Profiler::CountFrameEvent(reason);
	}
}

bool CommandProcessor::PrefetchOverlaps(uint64_t begin, uint64_t end) const {
	for (const auto& write: m_prefetch_writes) {
		if (write.begin < end && begin < write.end) {
			return true;
		}
	}
	if (m_prefetch_owner != nullptr) {
		// CP writes of ops before the wait that the resolver may not have executed yet.
		for (const auto& write: m_prefetch_owner->m_pending_writes) {
			if (write.begin < end && begin < write.end) {
				return true;
			}
		}
	}
	return false;
}

CpSeq::Result CommandProcessor::SubmitPrefetch(CpSeq::OpKind kind, const void* payload) {
	if (CpSeq::IsLockstep(kind)) {
		// An ordering point: only the resolver can answer it.
		StopPrefetch(Profiler::FrameEvent::CpSeqPrefetchStopLockstep);
		return {true, 0};
	}
	uint64_t begin = 0;
	uint64_t end   = 0;
	if (CpWriteRange(kind, payload, begin, end)) {
		m_prefetch_writes.push_back({begin, end, 0});
	}
	return {};
}

bool CommandProcessor::CheckCommandBytesPrefetch(Pm4Execution::BufferCursor& cursor) {
	const auto rest_dw    = cursor.commands.size() - cursor.offset_dw;
	const auto rest_begin = cursor.copy != nullptr
	                            ? cursor.guest_address + uint64_t {cursor.offset_dw} * 4u
	                            : reinterpret_cast<uint64_t>(cursor.commands.data() +
	                                                         cursor.offset_dw);
	const auto rest_end   = rest_begin + rest_dw * sizeof(uint32_t);
	const bool readable =
	    !PrefetchOverlaps(rest_begin, rest_end) &&
	    (cursor.copy != nullptr ? cursor.checked_epoch != 0
	                            : !GpuTouched::g_pages.AnyTouched(rest_begin, rest_end));
	if (!readable) {
		// A pending CP write over the rest, GPU-touched bytes (the real parse reads them in
		// lockstep), or a copy a CP write made stale.
		StopPrefetch(Profiler::FrameEvent::CpSeqPrefetchStopRead);
		return false;
	}
	cursor.checked_epoch = m_barrier_epoch;
	return true;
}

bool CommandProcessor::ReadGuestForPrefetch(uint64_t address, uint64_t size, void* dst) {
	if (size == 0) {
		return true;
	}
	if (PrefetchOverlaps(address, address + size) ||
	    GpuTouched::g_pages.AnyTouched(address, address + size)) {
		StopPrefetch(Profiler::FrameEvent::CpSeqPrefetchStopRead);
		return false;
	}
	std::memcpy(dst, reinterpret_cast<const void*>(address), size);
	return true;
}

template <typename Op>
void CommandProcessor::PrefetchDraw(const Op& op) {
	auto* owner  = m_prefetch_owner;
	auto* engine = owner != nullptr ? owner->EmitPrep() : nullptr;
	EXIT_IF(engine == nullptr || !engine->Parallel());
	if (m_prefetch_stop) {
		return;
	}
	if (m_prefetch_draws >= CpSeq::PrefetchDraws()) {
		StopPrefetch(Profiler::FrameEvent::CpSeqPrefetchStopWindow);
		return;
	}
	// The arguments the real parse's publish passes (AttachDrawRegisters).
	auto args = DrawArgsOf(op);
	if ((op.flags & CpSeq::DrawFlagInheritInstances) != 0) {
		args.instance_count = 1;
	}
	const auto key = CurrentPacketKey();
	// The window frees up while the resolver retires the draws before the wait (briefly: a
	// resolver blocked on the wait itself retires nothing).
	auto&    sequencer  = *owner->m_sequencer;
	uint64_t full_since = 0;
	for (uint32_t spins = 0; !engine->WindowHasSpace(); spins++) {
		if (sequencer.Answered(m_prefetch_wait) || sequencer.Stopping()) {
			StopPrefetch(Profiler::FrameEvent::CpSeqPrefetchStopWindow);
			return;
		}
		if ((spins & 63u) == 0u) {
			const auto now = CpNowNs();
			if (full_since == 0) {
				full_since = now;
			} else if (now - full_since > 50'000u) {
				StopPrefetch(Profiler::FrameEvent::CpSeqPrefetchStopWindow);
				return;
			}
		}
#if defined(_M_X64) || defined(__x86_64__)
		_mm_pause();
#endif
	}
	bool published = false;
	if constexpr (std::is_same_v<decltype(args), DrawIndexArgs>) {
		published = engine->PublishSpeculative(&args, nullptr, m_ctx, m_ucfg, m_sh_ctx,
		                                       m_adopt_packets, key);
	} else {
		published = engine->PublishSpeculative(nullptr, &args, m_ctx, m_ucfg, m_sh_ctx,
		                                       m_adopt_packets, key);
	}
	if (!published) {
		StopPrefetch(Profiler::FrameEvent::CpSeqPrefetchStopWindow);
		return;
	}
	m_prefetch_draws++;
}

void CommandProcessor::RunPrefetch(uint64_t wait_op) {
	auto*       engine    = EmitPrep();
	const auto* execution = g_current_execution;
	if (engine == nullptr || !engine->Parallel() || execution == nullptr ||
	    execution->m_buffer_stack.empty() || engine->SpeculativeSlots() != 0) {
		return;
	}
	Profiler::ScopedFrameWait time(Profiler::FrameWait::CpSeqPrefetch);
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqPrefetchRuns);
	if (m_prefetch == nullptr) {
		m_prefetch.reset(new CommandProcessor(m_renderer, m_interrupt_event_id));
		m_prefetch->m_front_mode     = FrontMode::Prefetch;
		m_prefetch->m_prefetch_owner = this;
	}
	auto& shadow = *m_prefetch;
	shadow.CopyFrontStateForPrefetch(*this);
	shadow.m_prefetch_writes.clear();
	shadow.m_prefetch_wait  = wait_op;
	shadow.m_prefetch_stop  = false;
	shadow.m_prefetch_draws = 0;
	shadow.m_adopt_tracking = true;
	shadow.m_adopt_skip_one = false;
	shadow.m_adopt_packets  = 0;
	shadow.m_adopt_inputs   = 0;
	// Every buffer's rest is checked again against this front's pending writes.
	shadow.m_barrier_epoch = m_barrier_epoch + 1u;
	Pm4Execution spec;
	spec.m_buffer_stack = execution->m_buffer_stack;
	spec.m_stream_id    = execution->m_stream_id;
	{
		// From the packet after the wait (whose handler is running on this front).
		auto&      cursor = spec.m_buffer_stack.back();
		const auto length = KYTY_PM4_LEN(cursor.commands[cursor.offset_dw]);
		EXIT_IF(cursor.offset_dw + length > cursor.commands.size());
		cursor.offset_dw += length;
	}
	struct Scope {
		Scope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~Scope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}
		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} scope(shadow, spec);
	auto& sequencer = *m_sequencer;
	for (;;) {
		if (spec.m_buffer_stack.empty()) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqPrefetchStopEnd);
			break;
		}
		if (sequencer.Answered(wait_op) || sequencer.Stopping()) {
			break;
		}
		spec.m_suspended = false;
		(void)shadow.ProcessPacket(spec);
		if (shadow.m_prefetch_stop || spec.m_suspended) {
			break;
		}
	}
}

// Astro Bot PC (KYTY_CP_PAR=1a): on CP resolver #2, runs `body` on Thread_Gpu (which holds the
// token meanwhile, cpParallel.h) with this thread's processor and execution scope.
template <typename Body>
static void RunOnGpuThreadScoped(Body& body) {
	struct Context {
		Body*             body;
		CommandProcessor* processor;
		Pm4Execution*     execution;
	} context {&body, g_current_processor, g_current_execution};
	CpPar::RunOnGpuThread(
	    [](void* argument) {
		    auto&      c                  = *static_cast<Context*>(argument);
		    auto*      previous_processor = g_current_processor;
		    auto*      previous_execution = g_current_execution;
		    g_current_processor           = c.processor;
		    g_current_execution           = c.execution;
		    (*c.body)();
		    g_current_processor = previous_processor;
		    g_current_execution = previous_execution;
	    },
	    &context);
}

// ---- Astro Bot PC (KYTY_CP_PAR=1b3, cpParallel.h): the run-ahead ----------------------------

// The fix-up barriers of the run-ahead's overlay (its states become the global ones) and one
// global memory barrier, recorded in the calling thread's context (the end of the oldest
// context's segment, before its Switch).
static void CpParRecordFixUps(CommandScheduler& scheduler, uint64_t& fixups, uint64_t& images,
                              uint64_t& impossible) {
	Image::Barriers barriers;
	Image::CpParMergeOverlay(barriers, images, impossible);
	fixups = barriers.size();
	if (!scheduler.Active() || !scheduler.CurrentValid()) {
		// No native buffer to record into (never seen): counted as impossible fix-ups.
		impossible += barriers.size();
		fixups = 0;
		return;
	}
	scheduler.EndRendering();
	vk::MemoryBarrier2 memory {};
	memory.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	memory.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	memory.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	memory.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount      = 1;
	dependency.pMemoryBarriers         = &memory;
	dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
	dependency.pImageMemoryBarriers    = barriers.data();
	scheduler.Current().Sink().pipelineBarrier2(dependency);
}

// Astro Bot PC (KYTY_CP_PAR=1b5, cpParallel.h): Thread_Gpu, op lock held, right after the oldest
// submission's last op (the merge; `late`: a hand-over or a new offer found the list pending).
// In list order: each captured label store is written as the record-time write would have been
// (and booked as the run-ahead's), each captured interrupt is queued on the completion runner for
// the current tick, after every priority operation queued before (FIFO).
static void CpParDeliverDeferred(CommandScheduler& scheduler, bool late) {
	auto& list = g_cp_par_deferred;
	if (list.entries.empty()) {
		return;
	}
	EXIT_IF(CpPar::OnResolver2());
	uint64_t interrupts = 0;
	uint64_t stores     = 0;
	uint64_t previous   = 0;
	uint64_t noted      = 0;
	for (auto& entry: list.entries) {
		if (entry.sequence < previous) {
			// The list is in execution order: an earlier op cannot follow a later one.
			if (CpPar::VerifyEnabled()) {
				EXIT("Kyty CpPar: KYTY_CP_PAR_VERIFY=1: deferred list out of order (op %" PRIu64
				     " after op %" PRIu64 ")\n",
				     entry.sequence, previous);
			}
		}
		previous = std::max(previous, entry.sequence);
		if (entry.store) {
			auto* dst = reinterpret_cast<void*>(entry.address);
			StreamCopyDefer::NoteGuestWrite(entry.address, entry.size);
			std::memcpy(dst, &entry.value, entry.size);
			NoteCpWrite(entry.address, entry.size);
			TraceCpLabel("label-cp-par-delivered", dst, entry.value, entry.size);
			for (uint32_t i = 0; i < entry.size / 4u; i++) {
				auto& write      = g_cp_par_book.writes[(entry.address & ~uint64_t {3}) + 4u * i];
				write.submission = list.submission;
				write.known      = true;
				write.value      = static_cast<uint32_t>(entry.value >> (32u * i));
				write.kind       = entry.kind;
			}
			stores++;
		} else {
			if (entry.sequence != noted) {
				CpPar::NoteInterruptOrder(entry.sequence, true);
				noted = entry.sequence;
			}
			scheduler.DeferPriorityOperation(std::move(entry.operation),
			                                 CommandScheduler::PriorityOperationKind::EopInterrupt);
			interrupts++;
		}
	}
	list.entries.clear();
	CpPar::NoteDeferredDelivery(interrupts, stores, late);
}

// Astro Bot PC (KYTY_CP_PAR=1b5, cpParallel.h): a foreign-wait or external promotion, logged
// (first 16 of each): the address, its last CP writer and what memory holds now. Op lock held.
static void CpParLogStop(CpPar::Promotion reason, CpSeq::OpKind kind, uint64_t submission,
                         uint64_t address, const char* why) {
	if (!CpPar::Deferred() || !CpPar::TakeStopLog(reason)) {
		return;
	}
	const auto& book     = g_cp_par_book;
	uint32_t    memory   = 0;
	const bool  readable = address != 0 && (address & 3u) == 0 &&
	                      LibKernel::Memory::TryReadGpuCleanBacking(address, &memory, sizeof(memory));
	char writer[256];
	const char* state = "";
	if (const auto found = book.writes.find(address); found != book.writes.end()) {
		const auto& write = found->second;
		std::snprintf(writer, sizeof(writer),
		              "submission %" PRIu64 " (rank %" PRId64 " in its frame, %+" PRId64
		              " from this one), %s, value %s0x%08" PRIx32,
		              write.submission, CpParRankOf(write.submission),
		              static_cast<int64_t>(write.submission) - static_cast<int64_t>(submission),
		              write.kind < static_cast<uint16_t>(CpSeq::OpKind::Count)
		                  ? CpSeq::OpKindName(static_cast<CpSeq::OpKind>(write.kind))
		                  : "?",
		              write.known ? "" : "unknown, ", write.value);
		state = !readable       ? "memory not readable clean (GPU-dirty)"
		        : !write.known  ? "memory value not comparable"
		        : memory == write.value
		            ? "memory still holds the CP's value"
		            : "memory differs from the CP's value: written outside the CP since (guest CPU or GPU)";
	} else {
		std::snprintf(writer, sizeof(writer), "none on record");
		state = "no CP write booked: written by the guest CPU or the GPU only (or forgotten)";
	}
	std::printf("Kyty CpPar 1b5: %s at %s of submission %" PRIu64 " (rank %" PRId64
	            "): address 0x%016" PRIx64 " (%s); last CP writer: %s; %s; learned label %s, awaited "
	            "by another submission %s; memory 0x%08" PRIx32 "\n",
	            reason == CpPar::Promotion::ForeignWait ? "foreign-wait" : "external",
	            CpSeq::OpKindName(kind), submission, CpParRankOf(submission), address, why, writer,
	            state, book.labels.count(address) != 0 ? "yes" : "no",
	            book.shared.count(address) != 0 ? "yes" : "no", readable ? memory : 0u);
	std::fflush(stdout);
}

void GuestGpu::CpParEnterSlice(const Submission& submission) {
	auto&          run       = g_cp_par_run;
	auto&          scheduler = m_renderer.GetCommandScheduler();
	const uint32_t context   = submission.cp_par_context != 0 ? 1u : 0u;
	// The run-ahead's submission is handed over before its first slice here.
	EXIT_IF(run.offered && run.target == submission.sequence);
	if (context != run.open) {
		// The replay continues in this submission's ring: the open ring's segment ends (Switch).
		EXIT_IF(run.begun[context] || run.began);
		CommandRecorder::SetThreadStream(run.open);
		(void)scheduler.EndContextSegment();
		run.begun[run.open] = false;
		run.open            = context;
		CpParNoteHolderChange(m_renderer); // another wrapper from now on
	}
	CommandRecorder::SetThreadStream(context);
	if (!run.begun[context]) {
		scheduler.BeginContextSegment();
		run.begun[context] = true;
	}
	t_cp_par_resolve_b = context != 0;
}

void GuestGpu::CpParMaybeOffer(const Submission& submission) {
	auto& run = g_cp_par_run;
	if (CpPar::Deferred() && submission.queue_id == 0) {
		CpParNoteRank(submission.sequence, submission.frame_fence); // Astro Bot PC (KYTY_CP_PAR=1b5)
	}
	if (run.offered || submission.type != SubmissionType::Graphics || submission.handoff ||
	    !submission.constant_commands.empty() || submission.queue_id != 0) {
		return;
	}
	Submission* next = nullptr;
	{
		Common::LockGuard lock(m_queue_mutex);
		if (m_queues[0].empty()) {
			return;
		}
		next = &m_queues[0].front();
		// The next queue-0 submission, in the other context, not started, not fenced (its bytes are
		// parsed only once it starts) and not a constant-engine one.
		if (next->type != SubmissionType::Graphics || next->started || next->handoff ||
		    !next->constant_commands.empty() ||
		    (next->frame_fence != 0 && next->frame_fence != submission.frame_fence) || // meme image : lue deja
		    (next->cp_par_context != 0) == (submission.cp_par_context != 0)) {
			return;
		}
		// Admitted while this lock is held; its fields used below do not change until popped.
		run.target     = next->sequence;
		run.target_ctx = next->cp_par_context != 0 ? 1u : 0u;
		run.target_sub = next;
		if (CpPar::Deferred()) {
			CpParNoteRank(next->sequence, next->frame_fence); // Astro Bot PC (KYTY_CP_PAR=1b5)
		}
	}
	run.offered = true;
	run.oldest  = submission.sequence;
	run.oldest_done.store(false, std::memory_order_relaxed);
	run.ended.store(false, std::memory_order_relaxed);
	run.submit_id        = m_submit_id + 1u;
	run.began            = false;
	run.started          = false;
	run.next_op          = 0;
	run.switch_back      = false;
	run.resolver_packets = 0;
	run.retry_op         = 0;
	run.adopt_pending    = false;
	run.ahead_ops        = 0;
	run.ahead_draws      = 0;
	run.offer_ns         = CpPar::NowNs();
	if (CpPar::Deferred()) {
		// Astro Bot PC (KYTY_CP_PAR=1b5): the previous list was delivered at its merge.
		if (!g_cp_par_deferred.entries.empty()) {
			CpPar::NoteDeferredLeft(g_cp_par_deferred.entries.size());
			CpParDeliverDeferred(m_renderer.GetCommandScheduler(), true);
		}
		g_cp_par_deferred.submission = run.target;
	}
	if (CpPar::Wide()) {
		// Astro Bot PC (KYTY_CP_PAR=1b4): the overlap's cross-access sets (op lock held).
		g_cp_par_book.EndOverlap();
		g_cp_par_book.overlap = true;
	}
	CpPar::OfferRunAhead(&GuestGpu::CpParRunAheadTask, this);
}

bool GuestGpu::CpParHandOver(uint64_t front_sequence) {
	auto& run = g_cp_par_run;
	if (!run.offered || run.target != front_sequence) {
		return false;
	}
	if (CpPar::StopRunAhead()) {
		CpParNoteHolderChange(m_renderer);
	}
	// The oldest submission ended first (the queue front is the target): merged if it began.
	EXIT_IF(run.began || run.switch_back);
	g_cp_par_book.EndOverlap(); // Astro Bot PC (KYTY_CP_PAR=1b4)
	run.offered       = false;
	run.adopt_pending = run.started;
	if (CpPar::Deferred()) {
		// Astro Bot PC (KYTY_CP_PAR=1b5): normally delivered at the merge already; the outcome of
		// this target is counted while Thread_Gpu resolves the rest of it.
		CpParDeliverDeferred(m_renderer.GetCommandScheduler(), true);
		g_cp_par_tally             = {};
		g_cp_par_tally.active      = true;
		g_cp_par_tally.started     = run.started;
		g_cp_par_tally.submission  = run.target;
		g_cp_par_tally.ahead_ops   = run.ahead_ops;
		g_cp_par_tally.ahead_draws = run.ahead_draws;
	}
	return true;
}

void GuestGpu::CpParMerge() {
	auto& run = g_cp_par_run;
	run.oldest_done.store(true, std::memory_order_release);
	g_cp_par_book.EndOverlap(); // Astro Bot PC (KYTY_CP_PAR=1b4): the oldest has ended
	if (CpPar::Deferred()) {
		CpPar::NoteWindow(CpPar::NowNs() - run.offer_ns); // Astro Bot PC (KYTY_CP_PAR=1b5)
	}
	if (!run.began) {
		// Astro Bot PC (KYTY_CP_PAR=1b5): an insertion merged the packets already; the deferred
		// interrupts are delivered here all the same (right after the oldest's last op).
		CpParDeliverDeferred(m_renderer.GetCommandScheduler(), false);
		return;
	}
	EXIT_IF(run.switch_back || run.target_ctx == run.open);
	auto&    scheduler  = m_renderer.GetCommandScheduler();
	uint64_t fixups     = 0;
	uint64_t images     = 0;
	uint64_t impossible = 0;
	// The end of the oldest context's segment: fix-ups, then the Switch into the run-ahead's ring,
	// which becomes this thread's recording context (both threads record there from now on).
	CommandRecorder::SetThreadStream(run.open);
	CpParRecordFixUps(scheduler, fixups, images, impossible);
	(void)scheduler.EndContextSegment();
	run.begun[run.open] = false;
	run.open            = run.target_ctx;
	CpParNoteHolderChange(m_renderer); // another wrapper from now on
	CommandRecorder::SetThreadStream(run.open);
	run.began = false;
	CpPar::SetRunAheadDirty(false);
	CpPar::NoteMerge(fixups, impossible, images);
	// Astro Bot PC (KYTY_CP_PAR=1b5, cpParallel.h): the delivery point of the run-ahead's
	// interrupts and captured label stores: the oldest's last op has executed, no later op of the
	// target has run on this thread yet, and the replay reaches the run-ahead's packets in the
	// current tick (the Switch above).
	CpParDeliverDeferred(scheduler, false);
	// Every op before the run-ahead's next one has executed now.
	if (run.next_op > m_sequencer->Executed()) {
		m_sequencer->NoteExecuted(run.next_op);
	}
}

void GuestGpu::CpParInsert(uint32_t trigger) {
	auto& run = g_cp_par_run;
	if (!run.began) {
		return;
	}
	EXIT_IF(run.switch_back || run.target_ctx == run.open);
	auto&      scheduler   = m_renderer.GetCommandScheduler();
	const bool in_op       = CpPar::OnResolver2();
	const auto stream      = CommandRecorder::ThreadStream();
	const auto oldest_ring = run.open;
	if (in_op) {
		// Inside a run-ahead op (a drain or a submit): its overlay states become the global ones
		// below; the rest of the op runs on them.
		Image::CpParEndOp();
		CpPar::t_overlay = false;
	}
	uint64_t fixups     = 0;
	uint64_t images     = 0;
	uint64_t impossible = 0;
	CommandRecorder::SetThreadStream(oldest_ring);
	CpParRecordFixUps(scheduler, fixups, images, impossible);
	(void)scheduler.EndContextSegment();
	run.begun[oldest_ring] = false;
	if (in_op) {
		// The replay continues in the run-ahead's ring until its op ends (RunAheadSubmission
		// switches back then).
		run.open        = run.target_ctx;
		run.switch_back = true;
		run.back_ring   = oldest_ring;
		CommandRecorder::SetThreadStream(run.target_ctx);
	} else {
		// The run-ahead is between ops: its segment ends too, and the oldest context's begins
		// again (the run-ahead's next op begins a new segment).
		CommandRecorder::SetThreadStream(run.target_ctx);
		(void)scheduler.EndContextSegment();
		run.begun[run.target_ctx] = false;
		CommandRecorder::SetThreadStream(oldest_ring);
		scheduler.BeginContextSegment();
		run.begun[oldest_ring] = true;
		CommandRecorder::SetThreadStream(stream);
		m_renderer.GetRenderExecutor().CpParForgetCommandShadows();
	}
	run.began = false;
	CpPar::SetRunAheadDirty(false);
	CpPar::NoteInsertion(static_cast<CpPar::Insertion>(trigger), fixups, impossible, images);
}

void GuestGpu::CpParRunAheadTask(void* argument) {
	auto* gpu = static_cast<GuestGpu*>(argument);
	auto& run = g_cp_par_run;
	auto* sub = static_cast<Submission*>(run.target_sub);
	// The GPU thread's identity (IsGpuThread paths), without its services (g_gpu_state: service
	// commands, slice yields and deferred-label waits stay Thread_Gpu's).
	g_gpu_thread       = true;
	g_gpu_state        = nullptr;
	t_cp_par_resolve_b = run.target_ctx != 0;
	CommandRecorder::SetThreadStream(run.target_ctx);
	const auto reason =
	    gpu->m_gfx_cp->RunAheadSubmission(sub->command_execution, sub->sequence, run.target_ctx);
	CpPar::NotePromotion(static_cast<CpPar::Promotion>(reason));
	run.ended.store(true, std::memory_order_release); // no more interleave points for it
	g_gpu_thread = false;
}

// The op the run-ahead cannot execute ahead of the oldest context, by kind (cpParallel.h), or
// Promotion::Count for a free op.
static CpPar::Promotion RunAheadBlocker(CpSeq::OpKind kind, const void* payload) {
	using CpPar::Promotion;
	using CpSeq::OpKind;
	switch (kind) {
		case OpKind::StreamBegin:
		case OpKind::SkipSlots: return Promotion::Count;
		case OpKind::DrawIndex:
		case OpKind::DrawAuto: {
			const auto flags = kind == OpKind::DrawIndex
			                       ? static_cast<const CpSeq::DrawIndexOp*>(payload)->flags
			                       : static_cast<const CpSeq::DrawAutoOp*>(payload)->flags;
			if ((flags & CpSeq::DrawFlagInheritInstances) != 0) {
				return Promotion::Inherit;
			}
			if ((flags & (CpSeq::DrawFlagPublished | CpSeq::DrawFlagSnapshot)) == 0) {
				return Promotion::NoRegisters;
			}
			return Promotion::Count;
		}
		case OpKind::DispatchDirect:
			return (static_cast<const CpSeq::DispatchDirectOp*>(payload)->flags &
			        CpSeq::DispatchFlagSnapshot) != 0
			           ? Promotion::Count
			           : Promotion::NoRegisters;
		case OpKind::DrawIndirect:
		case OpKind::DrawIndirectMulti:
		case OpKind::DispatchIndirect: return Promotion::Indirect;
		case OpKind::StreamEnd:
		case OpKind::Handoff: return Promotion::StreamEnd;
		case OpKind::Flip: return Promotion::Flip;
		case OpKind::WaitRegMem:
		case OpKind::WaitFlipDone:
		case OpKind::Predication:
		case OpKind::CondExec:
		case OpKind::Branch:
		case OpKind::LockstepRead: return Promotion::Lockstep;
		default: return Promotion::Write;
	}
}

// Astro Bot PC (KYTY_CP_PAR=1b4, cpParallel.h): the kinds whose run-ahead decision depends on the
// op's payload and the CP address book (RunAheadSubmission, under the op lock).
static bool RunAheadWideKind(CpSeq::OpKind kind) {
	using CpSeq::OpKind;
	switch (kind) {
		case OpKind::EndOfPipe:
		case OpKind::ReleaseMem:
		case OpKind::EventWrite:
		case OpKind::WriteData:
		case OpKind::DmaData:
		case OpKind::ReferenceClock:
		case OpKind::WaitRegMem: return true;
		default: return false;
	}
}

// KYTY_LABEL_COMPLETION_RANGES is set (LabelRangeSelected may defer any label).
static bool LabelRangesConfigured() {
	static const bool configured = [] {
		const auto* value = std::getenv("KYTY_LABEL_COMPLETION_RANGES");
		return value != nullptr && *value != 0;
	}();
	return configured;
}

uint32_t CommandProcessor::RunAheadSubmission(Pm4Execution& execution, uint64_t submission,
                                              uint32_t context) {
	using CpPar::Promotion;
	KYTY_PROFILER_BLOCK("CommandProcessor::RunAhead");
	EXIT_IF(!CpPar::Concurrent() || !CpPar::OnResolver2() || m_ops_b == nullptr);
	auto& run       = g_cp_par_run;
	auto& ops       = context != 0 ? *m_ops_b : *m_ops;
	auto& sequencer = *m_sequencer;
	auto& scheduler = GetScheduler();
	struct Scope {
		Scope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~Scope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}
		Scope(const Scope&)            = delete;
		Scope& operator=(const Scope&) = delete;
		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} scope(*this, execution);
	// Astro Bot PC (KYTY_CP_PAR=1b4, cpParallel.h): the decision for a write or a wait of the next
	// submission (op lock held): Promotion::Count executes it. `checked`: a record-time write whose
	// address the book allowed; `local_wait`: a wait on this submission's own write.
	auto&      book           = g_cp_par_book;
	const auto label_deferred = [this] {
		return LabelCompletionMode() || LabelRangesConfigured() || m_defer_next_label ||
		       m_renderer.GetGpu().HasDeferredLabels();
	};
	// Astro Bot PC (KYTY_CP_PAR=1b5): the op's details for the stop logs (address, why), whether it
	// raises an interrupt and whether its label store is captured (cpParallel.h). Reset per op.
	struct Diag {
		uint64_t    address       = 0;
		const char* why           = "";
		bool        interrupt     = false;
		bool        capture_store = false;
	} diag;
	const auto check_write = [&diag](uint64_t address, uint64_t bytes, bool& checked) {
		const auto& book = g_cp_par_book;
		if (address == 0 || bytes == 0) {
			return Promotion::Write;
		}
		if (!LibKernel::Memory::IsGpuCleanForRead(address, bytes)) {
			return Promotion::GpuDirty;
		}
		if (!CpPar::WritesAll()) {
			const auto first  = address & ~uint64_t {3};
			const auto dwords = (address + bytes - first + 3u) / 4u;
			if (dwords > CpParRange::MaxDwords) {
				diag.address = first;
				diag.why     = "wider than the book tracks";
				return Promotion::External;
			}
			for (uint64_t i = 0; i < dwords; i++) {
				const auto dword = first + 4u * i;
				if (book.labels.count(dword) == 0 || book.shared.count(dword) != 0) {
					diag.address = dword;
					diag.why     = book.shared.count(dword) != 0 ? "awaited by another submission"
					                                             : "not a learned CP-internal label";
					return Promotion::External;
				}
			}
		}
		checked = true;
		return Promotion::Count;
	};
	const auto decide = [&](CpSeq::OpKind kind, const void* payload, const void* data,
	                        bool& checked, bool& local_wait) {
		using CpSeq::OpKind;
		switch (kind) {
			case OpKind::EventWrite:
				// Cache flushes and partial flushes: a barrier in its own ring (or ignored).
				return static_cast<const CpSeq::EventWriteOp*>(payload)->event_type == 0x39u
				           ? Promotion::Occlusion
				           : Promotion::Count;
			case OpKind::WriteData: {
				const auto& op = *static_cast<const CpSeq::WriteDataOp*>(payload);
				if (op.dw_num == 0) {
					return Promotion::Count;
				}
				if (LabelCompletionMode() || data == nullptr) {
					return Promotion::LabelDefer; // split mode may defer it behind a pending label
				}
				const bool one = ((op.write_control >> 16u) & 0x1u) != 0;
				return check_write(op.dst, one ? 4u : uint64_t {op.dw_num} * 4u, checked);
			}
			case OpKind::EndOfPipe: {
				const auto& op        = *static_cast<const CpSeq::EndOfPipeOp*>(payload);
				const bool  interrupt = op.interrupt_selector != 0x00u && op.interrupt_selector != 0x03u;
				if (interrupt) {
					if (!CpPar::Deferred()) {
						return Promotion::Interrupt;
					}
					// Astro Bot PC (KYTY_CP_PAR=1b5): its interrupt is captured until the merge,
					// unless a deferred label (TryDeferLabel, ordered interrupts) would raise it.
					diag.interrupt = true;
					if (label_deferred()) {
						return Promotion::LabelDefer;
					}
				}
				if (op.event_write_source == 0x04u) {
					return ClockLocal() ? Promotion::Count : Promotion::Clock; // Astro Bot PC : KYTY_CP_PAR_CLOCK
				}
				if (op.size == sizeof(uint32_t) && op.event_write_source == 0x01u) {
					return Promotion::Gds;
				}
				if (interrupt && (op.dst == 0 || op.event_write_source == 0x00u)) {
					return Promotion::Count; // Astro Bot PC (1b5): the interrupt only
				}
				if (op.event_write_source != 0x01u && op.event_write_source != 0x02u) {
					return Promotion::Write;
				}
				if (label_deferred()) {
					return Promotion::LabelDefer;
				}
				const bool wide   = op.size == sizeof(uint64_t) && op.event_write_source == 0x02u;
				const auto reason = check_write(op.dst, wide ? 8u : 4u, checked);
				if (interrupt && reason == Promotion::External) {
					diag.capture_store = true; // Astro Bot PC (1b5): stored at the merge
					return Promotion::Count;
				}
				return reason;
			}
			case OpKind::ReleaseMem: {
				const auto* body        = static_cast<const CpSeq::ReleaseMemOp*>(payload)->body;
				const auto  data_sel    = (body[1] >> 29u) & 0x7u;
				const auto  int_sel     = (body[1] >> 24u) & 0x7u;
				const auto  release_dst = (body[1] >> 16u) & 0x3u;
				const auto  dst         = body[2] | (static_cast<uint64_t>(body[3]) << 32u);
				const bool  interrupt   = int_sel != 0x00u && int_sel != 0x03u;
				if (interrupt) {
					if (!CpPar::Deferred()) {
						return Promotion::Interrupt;
					}
					// Astro Bot PC (KYTY_CP_PAR=1b5): captured until the merge (see EndOfPipe).
					diag.interrupt = true;
					if (label_deferred()) {
						return Promotion::LabelDefer;
					}
				}
				if (data_sel == 0 || (release_dst == 0 && dst == 0)) {
					return Promotion::Count; // a barrier only (1b5: and its interrupt)
				}
				if (data_sel == 5) {
					return Promotion::Gds;
				}
				if (data_sel == 3) {
					return ClockLocal() ? Promotion::Count : Promotion::Clock; // Astro Bot PC : KYTY_CP_PAR_CLOCK
				}
				if ((data_sel != 1 && data_sel != 2) || dst == 0) {
					return Promotion::Write;
				}
				if (label_deferred()) {
					return Promotion::LabelDefer;
				}
				const auto reason = check_write(dst, data_sel == 1 ? 4u : 8u, checked);
				if (interrupt && reason == Promotion::External) {
					diag.capture_store = true; // Astro Bot PC (1b5): stored at the merge
					return Promotion::Count;
				}
				return reason;
			}
			case OpKind::DmaData: {
				// Buffer cache fills and copies, recorded in its own ring (GPU order after N).
				const auto& op = *static_cast<const CpSeq::DmaDataOp*>(payload);
				return op.dst_sel == 1u || op.src_sel == 1u ? Promotion::Gds : Promotion::Count;
			}
			case OpKind::ReferenceClock:
				return ClockLocal() ? Promotion::Count : Promotion::Clock; // Astro Bot PC : KYTY_CP_PAR_CLOCK
			case OpKind::WaitRegMem: {
				const auto& op   = *static_cast<const CpSeq::WaitRegMemOp*>(payload);
				const auto  size = op.size == sizeof(uint64_t) ? 8u : 4u;
				diag.address    = op.addr;
				if (op.addr == 0 || (op.addr & 3u) != 0) {
					diag.why = "null or misaligned address";
					return Promotion::ForeignWait;
				}
				uint32_t stored[2] = {};
				for (uint32_t i = 0; i < size / 4u; i++) {
					const auto found = book.writes.find(op.addr + 4u * i);
					if (found == book.writes.end() || found->second.submission != submission ||
					    !found->second.known) {
						diag.address = op.addr + 4u * i;
						diag.why     = found == book.writes.end() ? "no CP write booked"
						               : found->second.submission != submission
						                   ? "last written by another submission"
						                   : "own write of unknown value";
						return Promotion::ForeignWait; // not this submission's own write
					}
					stored[i] = found->second.value;
				}
				uint64_t value = 0;
				if (!LibKernel::Memory::IsGpuCleanForRead(op.addr, size) ||
				    !LibKernel::Memory::TryReadGpuCleanBacking(op.addr, &value, size)) {
					return Promotion::GpuDirty; // the read would synchronize
				}
				if (static_cast<uint32_t>(value) != stored[0] ||
				    (size == 8u && static_cast<uint32_t>(value >> 32u) != stored[1])) {
					diag.why = "changed since this submission wrote it";
					return Promotion::ForeignWait; // changed since this submission wrote it
				}
				const bool passes =
				    size == 8u ? TestWaitRegMemValue(value, op.ref, op.mask, op.func)
				               : TestWaitRegMemValue(static_cast<uint32_t>(value),
				                                     static_cast<uint32_t>(op.ref),
				                                     static_cast<uint32_t>(op.mask), op.func);
				if (!passes) {
					diag.why = "own write, but the wait does not pass yet";
					return Promotion::ForeignWait; // it waits for a later write
				}
				local_wait = true;
				return Promotion::Count;
			}
			default: return Promotion::Write;
		}
	};
	// Astro Bot PC (KYTY_CP_PAR=1b5): every stop with the kind of the op it stopped at (none: no
	// op of the target emitted yet) and the time spent waiting for the sequencer (cpParallel.h).
	uint64_t       starve_start = 0;
	constexpr auto no_op        = static_cast<uint32_t>(CpSeq::OpKind::Count);
	const auto     stop_at      = [&starve_start](Promotion reason, uint32_t at_kind) {
		if (starve_start != 0) {
			CpPar::NoteStarved(CpPar::NowNs() - starve_start);
			starve_start = 0;
		}
		if (CpPar::Deferred()) {
			CpPar::NoteStopKind(reason, at_kind);
		}
		return static_cast<uint32_t>(reason);
	};
	for (uint32_t idle = 0;;) {
		if (CpPar::RunAheadStopRequested() || run.oldest_done.load(std::memory_order_acquire)) {
			CpSeq::OpView pending;
			return stop_at(Promotion::Stop,
			               ops.Peek(pending) ? static_cast<uint32_t>(pending.Kind()) : no_op);
		}
		CpSeq::OpView view;
		if (!ops.Peek(view)) {
			// The sequencer has not produced it yet (it may still parse the oldest submission).
			if (starve_start == 0 && CpPar::Deferred()) {
				starve_start = CpPar::NowNs(); // Astro Bot PC (KYTY_CP_PAR=1b5)
			}
			if ((++idle & 63u) != 0u) {
#if defined(_M_X64) || defined(__x86_64__)
				_mm_pause();
#endif
			} else if (idle < 4096u) {
				std::this_thread::yield();
			} else {
				std::this_thread::sleep_for(std::chrono::microseconds(50));
			}
			continue;
		}
		idle = 0;
		if (starve_start != 0) {
			CpPar::NoteStarved(CpPar::NowNs() - starve_start); // Astro Bot PC (KYTY_CP_PAR=1b5)
			starve_start = 0;
		}
		diag               = {};
		const auto kind    = view.Kind();
		const auto blocker = RunAheadBlocker(kind, view.payload);
		// Astro Bot PC (KYTY_CP_PAR=1b4): writes and waits are decided under the op lock below.
		const bool wide_op = CpPar::Wide() && RunAheadWideKind(kind);
		if (blocker != Promotion::Count && !wide_op) {
			return stop_at(blocker, static_cast<uint32_t>(kind));
		}
		if (CpPar::OpLockAcquire(1)) {
			CpParNoteHolderChange(m_renderer);
		}
		// Busy (overlap statistics) while it holds the lock.
		const auto op_start = CpPar::NowNs();
		CpPar::NoteBusy(1, true);
		auto release = [] {
			CpPar::NoteBusy(1, false);
			CpPar::OpLockRelease(1);
		};
		if (CpPar::RunAheadStopRequested() || run.oldest_done.load(std::memory_order_acquire)) {
			release();
			return stop_at(Promotion::Stop, static_cast<uint32_t>(kind));
		}
		// Its recorder ring is replayed only after the merge: keep half of it free.
		if (auto* recorder = scheduler.Recorder();
		    recorder != nullptr && recorder->StreamBacklog() > recorder->StreamCapacity() / 2u) {
			release();
			return stop_at(Promotion::Ring, static_cast<uint32_t>(kind));
		}
		// Astro Bot PC (KYTY_CP_PAR=1b4): its own writes and waits (cpParallel.h).
		bool checked_write = false;
		bool local_wait    = false;
		if (wide_op) {
			const auto reason = decide(kind, view.payload, view.data, checked_write, local_wait);
			if (reason != Promotion::Count) {
				if (reason == Promotion::ForeignWait || reason == Promotion::External) {
					// Astro Bot PC (KYTY_CP_PAR=1b5): the first ones are logged (op lock held).
					CpParLogStop(reason, kind, submission, diag.address, diag.why);
				}
				release();
				return stop_at(reason, static_cast<uint32_t>(kind));
			}
		}
		if (!run.began) {
			EXIT_IF(run.begun[context] || run.open == context);
			scheduler.BeginContextSegment();
			run.begun[context] = true;
			run.began          = true;
			CpPar::SetRunAheadDirty(true);
			if (!run.started) {
				run.started = true;
				CpPar::NoteRunAheadStart();
			}
		}
		// This context's CP resolver fields (Thread_Gpu's are saved, it is between ops).
		bool       suspended       = false; // Astro Bot PC (KYTY_CP_PAR=1b4): a wait that did not pass
		const auto saved_packets   = m_resolver_packets;
		const auto saved_retry     = m_retry_op;
		const auto saved_submit_id = LaneCp().submit_id;
		m_resolver_packets         = run.resolver_packets;
		m_retry_op                 = run.retry_op;
		LaneCp().submit_id                = run.submit_id;
		CpPar::t_overlay           = true;
		const auto  sequence       = view.header->sequence;
		const auto  packet         = view.header->packet;
		const auto& header         = *view.header;
		if ((header.flags & CpSeq::FlagAdvanceEpoch) != 0) {
			SyncEpoch::Advance();
		}
		if (packet > m_resolver_packets) {
			m_resolver_packets = packet; // no end-of-pipe flush here (Thread_Gpu's)
		}
		if (kind == CpSeq::OpKind::StreamBegin) {
			const auto& begin = view.As<CpSeq::StreamBeginOp>();
			EXIT_IF(begin.submission != submission || begin.state != 0);
			ParCensus::NoteBegin(submission); // Astro Bot PC : KYTY_CP_PAR_CENSUS
			m_resolver_packets = 0;
			m_retry_op         = 0;
			ops.Pop();
		} else if (kind == CpSeq::OpKind::SkipSlots) {
			EXIT_IF(ResolvePrep() == nullptr);
			ResolvePrep()->SkipPublished(view.As<CpSeq::SkipSlotsOp>().count);
			ops.Pop();
		} else {
			auto*                             snapshot = OpSnapshot(kind, view.payload);
			CommandScheduler::RegisterBinding previous {};
			if (snapshot != nullptr) {
				previous = scheduler.BindRegisters(snapshot->context, snapshot->user_config,
				                                   snapshot->shaders);
			}
			if (CpPar::Deferred()) {
				// Astro Bot PC (KYTY_CP_PAR=1b5, cpParallel.h): the op's interrupts (and the store of
				// an interrupt label the book does not know) wait for the merge.
				g_cp_par_deferred.op_sequence   = sequence;
				g_cp_par_deferred.op_kind       = static_cast<uint16_t>(kind);
				g_cp_par_deferred.capture_store = diag.capture_store;
				CommandScheduler::SetThreadPriorityCapture(&CpParCaptureInterrupt);
			}
			m_executing       = true;
			const auto result = ExecuteOp(kind, view.payload, view.data);
			m_executing       = false;
			if (CpPar::Deferred()) {
				CommandScheduler::SetThreadPriorityCapture(nullptr);
				g_cp_par_deferred.capture_store = false;
			}
			if (snapshot != nullptr) {
				scheduler.RestoreRegisters(previous);
			}
			if (result.suspended) {
				// Astro Bot PC (KYTY_CP_PAR=1b4): only a wait suspends (draws and dispatches never
				// do); it stays at the ring's head and Thread_Gpu retries it after the merge.
				EXIT_IF(kind != CpSeq::OpKind::WaitRegMem || snapshot != nullptr);
				m_retry_op = sequence + 1u;
				suspended  = true;
			} else {
				m_retry_op = 0;
				ParCensus::NoteOp(kind, view.payload); // Astro Bot PC : KYTY_CP_PAR_CENSUS
				if (snapshot != nullptr) {
					ResolveSnapshots().Release();
				}
				if (local_wait) {
					// Astro Bot PC (KYTY_CP_PAR=1b4): what it read is its own write (verify checks).
					const auto& wait  = view.As<CpSeq::WaitRegMemOp>();
					const auto  size  = wait.size == sizeof(uint64_t) ? 8u : 4u;
					uint64_t    value = 0;
					const bool  read  = !CpPar::VerifyEnabled() ||
					                   LibKernel::Memory::TryReadGpuCleanBacking(wait.addr, &value, size);
					for (uint32_t i = 0; i < size / 4u; i++) {
						const auto found = book.writes.find(wait.addr + 4u * i);
						if (found != book.writes.end() && found->second.submission == run.oldest) {
							CpPar::NoteCross(CpPar::Cross::AheadReadOldest, wait.addr + 4u * i);
						}
						if (CpPar::VerifyEnabled() &&
						    (!read || found == book.writes.end() || found->second.submission != submission ||
						     found->second.value != static_cast<uint32_t>(value >> (32u * i)))) {
							CpPar::NoteCross(CpPar::Cross::AheadForeignValue, wait.addr + 4u * i);
						}
					}
				}
				ops.Pop();
				if (CpSeq::IsLockstep(kind)) {
					sequencer.Answer(sequence, result.value); // Astro Bot PC (KYTY_CP_PAR=1b4)
				}
				if (CpPar::Wide()) {
					CpParBookAfterOp(kind, view.payload, view.data, submission, true,
					                 diag.capture_store); // Astro Bot PC (1b5): booked at delivery
				}
			}
		}
		CpPar::t_overlay = false;
		Image::CpParEndOp();
		if (run.switch_back) {
			// An insertion inside this op (CpParInsert): back to the oldest context's ring.
			(void)scheduler.EndContextSegment();
			run.begun[context] = false;
			CommandRecorder::SetThreadStream(run.back_ring);
			scheduler.BeginContextSegment();
			run.begun[run.back_ring] = true;
			run.open                 = run.back_ring;
			run.switch_back          = false;
			CommandRecorder::SetThreadStream(context);
			m_renderer.GetRenderExecutor().CpParForgetCommandShadows();
		}
		if (!suspended) {
			run.next_op = sequence + 1u;
		}
		run.resolver_packets = m_resolver_packets;
		run.retry_op         = m_retry_op;
		run.submit_id        = LaneCp().submit_id;
		m_resolver_packets   = saved_packets;
		m_retry_op           = saved_retry;
		LaneCp().submit_id          = saved_submit_id;
		if (suspended) {
			CpParLogStop(Promotion::ForeignWait, kind, submission, view.As<CpSeq::WaitRegMemOp>().addr,
			             "executed, did not pass (suspended)"); // Astro Bot PC (KYTY_CP_PAR=1b5)
			release();
			return stop_at(Promotion::ForeignWait, static_cast<uint32_t>(kind));
		}
		CpPar::NoteRunAheadOp(CpPar::NowNs() - op_start);
		if (kind != CpSeq::OpKind::StreamBegin && kind != CpSeq::OpKind::SkipSlots) {
			// Astro Bot PC (KYTY_CP_PAR=1b5): this target's outcome (CpParTally).
			run.ahead_ops++;
			if (CpParIsDraw(kind)) {
				run.ahead_draws++;
			}
		}
		if (CpPar::Deferred() && diag.interrupt) {
			CpPar::NoteDeferredOp(diag.capture_store);
		}
		if (CpPar::Wide()) {
			// Astro Bot PC (KYTY_CP_PAR=1b4) statistics.
			CpPar::NoteRunAheadKind(static_cast<uint32_t>(kind));
			if (local_wait) {
				CpPar::NoteLocalWait();
			}
			if (checked_write) {
				CpPar::NoteLocalWrite();
			}
		}
		release();
		// Its window and snapshot slots are free: a parked sequencer re-checks.
		sequencer.Poke();
	}
}

Pm4ProcessResult CommandProcessor::ResolveSubmission(Pm4Execution& execution, uint64_t submission,
                                                     std::span<const uint32_t> commands,
                                                     bool& handoff, uint32_t context) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Resolve");
	ListCensus::BeginList(submission); // Astro Bot PC (KYTY_CP_PAR_CENSUS2)
	// Astro Bot PC (KYTY_CP_PAR=1a): the resolver context's op ring (always m_ops when off).
	EXIT_IF(OpsOf(context) == nullptr); // Astro Bot PC (KYTY_CP_PAR=2a: contexts 0..N-1)
	auto& ops = *OpsOf(context);
	// Astro Bot PC (KYTY_CP_PAR=1b1): ResolvePrep / ResolveSnapshots follow this context (only
	// read by the op executions below; the next ResolveSubmission sets it again).
	m_resolve_lane = CpPar::SeparateWindows() ? context : 0u;
	const CpGaps::Scope gap(CpGaps::Cat::Resolver);
	EXIT_IF(g_current_execution != nullptr || m_front_mode != FrontMode::Thread);
	handoff                   = false;
	execution.m_suspended     = false;
	execution.m_made_progress = false;
	execution.m_yield         = false;
	execution.m_yielded       = false;
	execution.m_stream_id     = submission;
	LaneCp().slice_draws             = 0;
	struct Scope {
		Scope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~Scope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}
		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} scope(*this, execution);
	auto& sequencer = *m_sequencer;
	// Astro Bot PC (KYTY_CP_PAR=1b3, cpParallel.h): Thread_Gpu is busy here (overlap statistics);
	// a submission the run-ahead started continues from its resolver state.
	const bool concurrent = CpPar::Concurrent();
	struct BusyScope {
		bool on;
		explicit BusyScope(bool on_): on(on_) {
			if (on) {
				CpPar::NoteBusy(0, true);
			}
		}
		~BusyScope() {
			if (on) {
				CpPar::NoteBusy(0, false);
			}
		}
		BusyScope(const BusyScope&)            = delete;
		BusyScope& operator=(const BusyScope&) = delete;
	} busy_scope(concurrent);
	if (concurrent && g_cp_par_run.adopt_pending && g_cp_par_run.target == submission) {
		g_cp_par_run.adopt_pending = false;
		m_resolver_packets         = g_cp_par_run.resolver_packets;
		m_retry_op                 = g_cp_par_run.retry_op;
	}
	// Service commands and the interleave point wait while the run-ahead is dirty / may run.
	auto service_allowed = [concurrent] { return !concurrent || !CpPar::RunAheadDirty(); };
	for (;;) {
		CommitStats::StepMark(CommitStats::Step::GapOpExit);
		// Service commands run between ops: every op before this one has executed and no later
		// one has, the serial point (P3-SEQUENCER.md 2.4).
		if (g_gpu_state != nullptr && g_gpu_state->HasPendingCommands() && service_allowed()) {
			if (CpPar::OnResolver2()) {
				// Astro Bot PC (KYTY_CP_PAR=1a): service commands stay on Thread_Gpu.
				auto* gpu  = g_gpu_state;
				auto  body = [gpu] { gpu->ProcessCommands(); };
				RunOnGpuThreadScoped(body);
				CpPar::NoteBounce(false);
			} else {
				g_gpu_state->ProcessCommands();
			}
		}
		CpSeq::OpView view;
		if (!ops.Peek(view)) {
			// The sequencer has not produced the next op yet: spin briefly, then let other queues
			// and services run while it parses.
			Profiler::ScopedFrameWait starved(Profiler::FrameWait::CpSeqResolverStarved);
			const CpGaps::Scope       starved_gap(CpGaps::Cat::Starved);
			Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqResolverStarved);
			const auto start    = CpNowNs();
			bool       produced = false;
			for (uint32_t spins = 0;; spins++) {
				if (ops.Peek(view)) {
					produced = true;
					break;
				}
				if ((spins & 63u) != 63u) {
#if defined(_M_X64) || defined(__x86_64__)
					_mm_pause();
#endif
					continue;
				}
				const auto elapsed = CpNowNs() - start;
				if (elapsed < CpSeq::ResolverSpinNs()) {
					continue;
				}
				if (g_gpu_state != nullptr && g_gpu_state->HasPendingCommands() && service_allowed()) {
					break;
				}
				if (elapsed > 2'000'000u ||
				    (g_gpu_state != nullptr && g_gpu_state->HasRunnableComputeWork())) {
					break;
				}
				std::this_thread::yield();
			}
			if (!produced) {
				if (g_gpu_state != nullptr && g_gpu_state->HasPendingCommands() && service_allowed()) {
					continue;
				}
				execution.m_yielded = true;
				return Pm4ProcessResult::Blocked;
			}
		}
		const auto  kind     = view.Kind();
		const auto  sequence = view.header->sequence;
		const auto  packet   = view.header->packet;
		const auto& header   = *view.header;
		if (concurrent && g_cp_par_run.offered && g_cp_par_run.oldest == submission &&
		    !g_cp_par_run.oldest_done.load(std::memory_order_relaxed) &&
		    !g_cp_par_run.ended.load(std::memory_order_acquire)) {
			// Astro Bot PC (KYTY_CP_PAR=1b4): KYTY_CP_PAR_INTERLEAVE=next tests N's next op only.
			const bool point = CpPar::Wide() && CpPar::InterleaveNext()
			                       ? !CpParHardForOldest(kind, view.payload)
			                       : CpPar::TailQuiet(submission, sequence);
			if (point) {
				// Astro Bot PC (KYTY_CP_PAR=1b3, cpParallel.h): the interleave point, in the quiet
				// tail (1b4: once no hard op remains): the run-ahead may execute one op now (the
				// lock is FIFO).
				if (CpPar::Wide()) {
					CpPar::NoteInterleave(true);
				}
				CpPar::OpLockRelease(0);
				if (CpPar::OpLockAcquire(0)) {
					CpParNoteHolderChange(m_renderer);
				}
			} else if (CpPar::Wide()) {
				CpPar::NoteInterleave(false); // Astro Bot PC (KYTY_CP_PAR=1b4)
			}
		}
		if (kind == CpSeq::OpKind::StreamBegin) {
			const auto& begin = view.As<CpSeq::StreamBeginOp>();
			EXIT_IF(begin.submission != submission);
			ParCensus::NoteBegin(submission); // Astro Bot PC : KYTY_CP_PAR_CENSUS
			m_resolver_packets = 0;
			m_retry_op         = 0;
			if (begin.state != 0) {
				auto* state = reinterpret_cast<CommandProcessor*>(begin.state);
				if (begin.verify != 0) {
					if (m_verifier == nullptr) {
						m_verifier.reset(new CpSeq::Verifier(m_renderer, m_interrupt_event_id));
					}
					m_verifier->Attach(*state, submission, commands);
				}
				delete state;
			} else if (m_verifier != nullptr) {
				m_verifier->Detach();
			}
			ops.Pop();
			sequencer.NoteExecuted(sequence + 1u);
			continue;
		}
		if (kind == CpSeq::OpKind::StreamEnd) {
			EXIT_IF(view.As<CpSeq::StreamEndOp>().submission != submission);
			ParCensus::NoteEnd(submission); // Astro Bot PC : KYTY_CP_PAR_CENSUS
			if (m_verifier != nullptr && m_verifier->Follows(submission)) {
				m_verifier->Finish();
			}
			ops.Pop();
			sequencer.NoteExecuted(sequence + 1u);
			execution.m_made_progress = true;
			if (concurrent && g_cp_par_run.offered && g_cp_par_run.oldest == submission) {
				// Astro Bot PC (KYTY_CP_PAR=1b3): the merge point of the run-ahead (cpParallel.h).
				EXIT_IF(g_gpu_state == nullptr);
				g_gpu_state->CpParMerge();
			}
			if (concurrent && CpPar::Deferred() && g_cp_par_tally.active &&
			    g_cp_par_tally.submission == submission) {
				// Astro Bot PC (KYTY_CP_PAR=1b5): the run-ahead target is complete.
				const auto& tally = g_cp_par_tally;
				CpPar::NoteOutcome(tally.started, tally.ahead_ops, tally.ahead_draws, tally.rest_ops,
				                   tally.rest_draws);
				g_cp_par_tally.active = false;
			}
			return Pm4ProcessResult::Complete;
		}
		if (kind == CpSeq::OpKind::Handoff) {
			EXIT_IF(view.As<CpSeq::HandoffOp>().submission != submission);
			ops.Pop();
			sequencer.NoteExecuted(sequence + 1u);
			handoff = true;
			return Pm4ProcessResult::Complete;
		}
		if (kind == CpSeq::OpKind::SkipSlots) {
			// P3c: speculative slots the real parse did not adopt (no op draws them).
			EXIT_IF(ResolvePrep() == nullptr);
			ResolvePrep()->SkipPublished(view.As<CpSeq::SkipSlotsOp>().count);
			ops.Pop();
			sequencer.NoteExecuted(sequence + 1u);
			execution.m_made_progress = true;
			continue;
		}
		// Placement samples (common/cpuPlacement.h), every 256th op: this thread executes the
		// graphics queue's packets, and ProcessPacket samples only where packets execute.
		if ((++m_placement_packets & 255u) == 0u) {
			Common::SamplePlacement(Common::ThreadRole::Cp);
		}
		// The back's per-packet work of the packets parsed since the previous op.
		if ((header.flags & CpSeq::FlagAdvanceEpoch) != 0) {
			SyncEpoch::Advance();
		}
		if (packet > m_resolver_packets) {
			if (LaneCp().deferred_eop_flushes != 0) {
				LaneCp().packets_since_eop_request +=
				    static_cast<uint32_t>(std::min<uint64_t>(packet - m_resolver_packets, 1u << 20u));
				if (LaneCp().packets_since_eop_request >= EopFlushPacketLimit() && OptionalSubmitAllowed()) {
					// Bound how long a batched end-of-pipe interrupt waits (at an op, not a packet).
					BufferFlush();
				}
			}
			m_resolver_packets = packet;
		}
		const bool retry     = m_retry_op == sequence + 1u;
		const bool following = m_verifier != nullptr && m_verifier->Follows(submission) &&
		                       !CpSeq::IsTransport(kind);
		if (following && !retry) {
			m_verifier->Before(view);
		}
		auto*                             snapshot = OpSnapshot(kind, view.payload);
		CommandScheduler::RegisterBinding previous {};
		if (snapshot != nullptr) {
			previous = GetScheduler().BindRegisters(snapshot->context, snapshot->user_config,
			                                        snapshot->shaders);
		}
		m_executing       = true;
		uint64_t diagnostic_args[4] {};
		if (HangWatchdog::Enabled()) {
			std::memcpy(diagnostic_args, view.payload,
			            std::min<size_t>(sizeof(diagnostic_args), CpSeq::PayloadSize(kind)));
			HangWatchdog::NotePacket(0, submission, 0, 0x10000u + static_cast<uint32_t>(kind),
			                         diagnostic_args[0], diagnostic_args[1], diagnostic_args[2],
			                         diagnostic_args[3], CpSeq::OpKindName(kind));
		}
		CommitStats::StepMark(CommitStats::Step::GapLoop);
		if (concurrent && CpPar::Wide()) {
			CpParBookBeforeOldestOp(kind, view.payload); // Astro Bot PC (KYTY_CP_PAR=1b4)
		}
		CpSeq::Result result;
		if (CpPar::OnResolver2() &&
		    (kind == CpSeq::OpKind::Flip || kind == CpSeq::OpKind::WaitFlipDone)) {
			// Astro Bot PC (KYTY_CP_PAR=1a): flips stay on Thread_Gpu (presentation, live switches).
			auto body = [&] { result = ExecuteOp(kind, view.payload, view.data); };
			RunOnGpuThreadScoped(body);
			CpPar::NoteBounce(true);
		} else if (CpPar::Lanes2b() && CpPar::LaneLockEnabled()) {
			// Astro Bot PC (KYTY_CP_PAR=2b, step 3b-2): the op under the lanes' global lock.
			CpPar::LaneLockAcquire();
			result = ExecuteOp(kind, view.payload, view.data);
			CpPar::LaneLockRelease();
		} else {
			result = ExecuteOp(kind, view.payload, view.data);
		}
		m_executing = false;
		if (snapshot != nullptr) {
			GetScheduler().RestoreRegisters(previous);
		}
		if (result.suspended) {
			// A wait that has not passed: the op stays at the ring's head and is executed again
			// in the next slice. Only lockstep ops suspend; they carry no snapshot.
			EXIT_IF(snapshot != nullptr);
			m_retry_op            = sequence + 1u;
			execution.m_suspended = true;
			return Pm4ProcessResult::Blocked;
		}
		m_retry_op = 0;
		ParCensus::NoteOp(kind, view.payload); // Astro Bot PC : KYTY_CP_PAR_CENSUS
		if (concurrent && CpPar::Deferred()) {
			// Astro Bot PC (KYTY_CP_PAR=1b5, cpParallel.h): the global interrupt order (deferred ones
			// are noted at their delivery); a run-ahead target's ops left to this thread.
			if (CpParOpRaisesInterrupt(kind, view.payload)) {
				CpPar::NoteInterruptOrder(sequence, false);
			}
			if (g_cp_par_tally.active && g_cp_par_tally.submission == submission) {
				g_cp_par_tally.rest_ops++;
				if (CpParIsDraw(kind)) {
					g_cp_par_tally.rest_draws++;
				}
			}
		}
		if (concurrent && CpPar::Wide()) {
			// Astro Bot PC (KYTY_CP_PAR=1b4): the CP address book (op lock held).
			CpParBookAfterOp(kind, view.payload, view.data, submission, false);
		}
		if (following && m_verifier->Follows(submission)) {
			m_verifier->After(view, result);
		}
		if (snapshot != nullptr) {
			ResolveSnapshots().Release();
		}
		ops.Pop();
		if (CpSeq::IsLockstep(kind)) {
			sequencer.Answer(sequence, result.value);
		}
		sequencer.NoteExecuted(sequence + 1u);
		execution.m_made_progress = true;
		if (execution.m_yield) {
			execution.m_yield   = false;
			execution.m_yielded = true;
			return Pm4ProcessResult::Blocked;
		}
	}
}

bool GuestGpu::IsGpuThread() noexcept {
	// Astro Bot PC (KYTY_CP_PAR=2b, step 3b-2): the holder of the lanes' global lock too (in turn
	// it is the token holder, which already has the identity).
	return g_gpu_thread || (CpPar::Lanes2b() && CpPar::LaneLockHeld());
}

} // namespace Libs::Graphics
