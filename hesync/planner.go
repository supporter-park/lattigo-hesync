package hesync

import (
	"fmt"
	"time"
)

// InstructionType represents the type of a prefetch plan instruction.
type InstructionType int

const (
	// InstrNop indicates no prefetch action at this step.
	InstrNop InstructionType = iota
	// InstrFetch indicates to begin asynchronously loading an EVK from disk.
	InstrFetch
	// InstrSync indicates to wait until the EVK has been loaded.
	InstrSync
	// InstrFree indicates to release an EVK from memory.
	InstrFree
)

// Instruction is a single entry in the prefetch plan.
type Instruction struct {
	Type     InstructionType
	KeyID    string // key identifier (e.g., "gk_12345" or "rlk")
	GaloisEl uint64 // Galois element (for Galois keys)
	KeyType  KeyType
}

// Plan is an ordered sequence of prefetch instructions, one per step
// in the trace. The Prefetcher executes these concurrently with the
// main HE computation to hide EVK load latency.
type Plan struct {
	// Instructions is indexed by step number. Each step may have
	// multiple instructions (e.g., both a Free and a Fetch).
	Instructions [][]Instruction

	// TotalSteps is the number of steps in the plan.
	TotalSteps int
}

// GeneratePlan creates a prefetch plan from a trace and profile using the
// backward-walk algorithm described in the HESync paper.
//
// The algorithm:
//  1. Walk the trace in reverse order
//  2. At each EVK access: place a Sync instruction at that step
//  3. Start a countdown timer initialized to the EVK load latency
//  4. As we walk backward, subtract each operation's profiled execution time
//  5. When the timer reaches zero (or below): place a Fetch instruction
//  6. After identifying last-use of each EVK: place a Free instruction
//
// The profile's level-aware operation latencies determine how far ahead
// each Fetch must be placed to fully hide the load latency.
func GeneratePlan(trace Trace, profile Profile, defaultLevel int) *Plan {
	n := len(trace.Entries)
	if n == 0 {
		return &Plan{TotalSteps: 0}
	}

	// Initialize instruction slices for each step
	instructions := make([][]Instruction, n)
	for i := range instructions {
		instructions[i] = []Instruction{}
	}

	// Phase 1: Find last use of each key to place Free instructions
	lastUse := make(map[string]int) // keyID -> last step where it's used
	for i, entry := range trace.Entries {
		keyID := entry.KeyID()
		lastUse[keyID] = i
	}

	for keyID, step := range lastUse {
		instructions[step] = append(instructions[step], Instruction{
			Type:  InstrFree,
			KeyID: keyID,
		})
	}

	// Phase 2: Backward walk to place Fetch and Sync instructions
	// Track which keys have already been scheduled (to avoid duplicate fetches)
	scheduled := make(map[string]bool)

	// Process trace entries in reverse
	// We use a list of pending fetches that need to be placed
	type pendingFetch struct {
		keyID    string
		galoisEl uint64
		keyType  KeyType
		timer    time.Duration // remaining time budget before fetch must start
	}
	var pending []pendingFetch

	for i := n - 1; i >= 0; i-- {
		entry := trace.Entries[i]
		keyID := entry.KeyID()

		// Get the operation latency at this step's level
		opLatency := profile.GetOpLatency(defaultLevel)

		// Decrease timer for all pending fetches
		for j := range pending {
			pending[j].timer -= opLatency
		}

		// Check if any pending fetch timers have expired — place them here
		remaining := pending[:0]
		for _, pf := range pending {
			if pf.timer <= 0 {
				instructions[i] = append(instructions[i], Instruction{
					Type:     InstrFetch,
					KeyID:    pf.keyID,
					GaloisEl: pf.galoisEl,
					KeyType:  pf.keyType,
				})
			} else {
				remaining = append(remaining, pf)
			}
		}
		pending = remaining

		// If this step accesses an EVK, place a Sync and schedule a Fetch
		if !scheduled[keyID] {
			// Place Sync at this step
			instructions[i] = append(instructions[i], Instruction{
				Type:     InstrSync,
				KeyID:    keyID,
				GaloisEl: entry.GaloisEl,
				KeyType:  entry.Type,
			})

			// Schedule a Fetch with timer = EVK load latency
			pending = append(pending, pendingFetch{
				keyID:    keyID,
				galoisEl: entry.GaloisEl,
				keyType:  entry.Type,
				timer:    profile.EVKLoadLatency,
			})

			scheduled[keyID] = true
		}
	}

	// Place any remaining pending fetches at step 0
	for _, pf := range pending {
		instructions[0] = append(instructions[0], Instruction{
			Type:     InstrFetch,
			KeyID:    pf.keyID,
			GaloisEl: pf.galoisEl,
			KeyType:  pf.keyType,
		})
	}

	return &Plan{
		Instructions: instructions,
		TotalSteps:   n,
	}
}

// String returns a human-readable representation of the plan.
func (p *Plan) String() string {
	var s string
	for i, instrs := range p.Instructions {
		if len(instrs) == 0 {
			continue
		}
		for _, instr := range instrs {
			var typeName string
			switch instr.Type {
			case InstrNop:
				typeName = "NOP"
			case InstrFetch:
				typeName = "FETCH"
			case InstrSync:
				typeName = "SYNC"
			case InstrFree:
				typeName = "FREE"
			}
			s += fmt.Sprintf("step %3d: %-5s %s\n", i, typeName, instr.KeyID)
		}
	}
	return s
}

// Stats returns statistics about the plan.
func (p *Plan) Stats() (fetches, syncs, frees int) {
	for _, instrs := range p.Instructions {
		for _, instr := range instrs {
			switch instr.Type {
			case InstrFetch:
				fetches++
			case InstrSync:
				syncs++
			case InstrFree:
				frees++
			}
		}
	}
	return
}
