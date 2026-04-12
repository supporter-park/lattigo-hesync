package hesync

import (
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// Prefetcher executes the prefetch plan concurrently with the main HE
// computation thread. It manages a pool of loader goroutines that
// asynchronously read EVKs from disk, and a syncer that moves loaded
// keys into the active key container.
type Prefetcher struct {
	plan  *Plan
	index *EVKIndex

	// pending tracks in-flight loads: keyID -> done channel
	pending   map[string]chan struct{}
	pendingMu sync.Mutex

	// loaded holds keys that have been loaded but not yet consumed
	loadedGK  map[uint64]*rlwe.GaloisKey
	loadedRLK *rlwe.RelinearizationKey
	loadedMu  sync.RWMutex

	// workers is the number of concurrent loader goroutines
	workers int

	// sem limits concurrent loads
	sem chan struct{}

	// wg tracks worker goroutine completion
	wg sync.WaitGroup
}

// NewPrefetcher creates a new Prefetcher with the given plan, index, and worker count.
func NewPrefetcher(plan *Plan, index *EVKIndex, workers int) *Prefetcher {
	if workers <= 0 {
		workers = 4
	}

	return &Prefetcher{
		plan:     plan,
		index:    index,
		pending:  make(map[string]chan struct{}),
		loadedGK: make(map[uint64]*rlwe.GaloisKey),
		workers:  workers,
		sem:      make(chan struct{}, workers),
	}
}

// ExecuteStep processes all prefetch instructions for the given step.
// This should be called before the main HE operation at each step.
func (p *Prefetcher) ExecuteStep(step int) {
	if step >= len(p.plan.Instructions) {
		return
	}

	instrs := p.plan.Instructions[step]
	for _, instr := range instrs {
		switch instr.Type {
		case InstrFetch:
			p.startLoad(instr)
		case InstrSync:
			p.waitForKey(instr)
		case InstrFree:
			p.freeKey(instr)
		case InstrNop:
			// nothing
		}
	}
}

// startLoad initiates an asynchronous EVK load from disk.
func (p *Prefetcher) startLoad(instr Instruction) {
	p.pendingMu.Lock()
	if _, exists := p.pending[instr.KeyID]; exists {
		p.pendingMu.Unlock()
		return // already loading
	}
	done := make(chan struct{})
	p.pending[instr.KeyID] = done
	p.pendingMu.Unlock()

	p.wg.Add(1)
	go func() {
		defer p.wg.Done()

		// Acquire semaphore slot
		p.sem <- struct{}{}
		defer func() { <-p.sem }()

		if instr.KeyType == KeyTypeRelin {
			if p.index.RelinKeyPath != "" {
				rlk, err := LoadRelinearizationKey(p.index.RelinKeyPath)
				if err == nil {
					p.loadedMu.Lock()
					p.loadedRLK = rlk
					p.loadedMu.Unlock()
				}
			}
		} else {
			if path, ok := p.index.GaloisKeys[instr.GaloisEl]; ok {
				gk, err := LoadGaloisKey(path)
				if err == nil {
					p.loadedMu.Lock()
					p.loadedGK[instr.GaloisEl] = gk
					p.loadedMu.Unlock()
				}
			}
		}

		// Signal completion
		close(done)
	}()
}

// waitForKey blocks until the specified key has been loaded.
func (p *Prefetcher) waitForKey(instr Instruction) {
	p.pendingMu.Lock()
	done, exists := p.pending[instr.KeyID]
	p.pendingMu.Unlock()

	if exists {
		<-done // wait for load to complete
	}
}

// freeKey removes a loaded key from memory.
func (p *Prefetcher) freeKey(instr Instruction) {
	p.loadedMu.Lock()
	defer p.loadedMu.Unlock()

	if instr.KeyType == KeyTypeRelin {
		p.loadedRLK = nil
	} else {
		delete(p.loadedGK, instr.GaloisEl)
	}

	// Clean up pending tracking
	p.pendingMu.Lock()
	delete(p.pending, instr.KeyID)
	p.pendingMu.Unlock()
}

// GetGaloisKey returns a loaded GaloisKey, or nil if not loaded.
func (p *Prefetcher) GetGaloisKey(galEl uint64) *rlwe.GaloisKey {
	p.loadedMu.RLock()
	defer p.loadedMu.RUnlock()
	return p.loadedGK[galEl]
}

// GetRelinearizationKey returns the loaded RelinearizationKey, or nil if not loaded.
func (p *Prefetcher) GetRelinearizationKey() *rlwe.RelinearizationKey {
	p.loadedMu.RLock()
	defer p.loadedMu.RUnlock()
	return p.loadedRLK
}

// Stop waits for all in-flight loads to complete.
func (p *Prefetcher) Stop() {
	p.wg.Wait()
}
