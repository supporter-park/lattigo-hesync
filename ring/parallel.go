package ring

import (
	"runtime"
	"sync"
)

// parallelThreshold is the minimum number of SubRings required before
// parallel dispatch is used. Below this, sequential iteration is faster.
//
// Set high by default: goroutine-per-operation parallelism has non-trivial
// overhead in tight loops (GadgetProduct calls hundreds of ring operations).
// The persistent worker pool (workerPool) avoids this but requires explicit
// opt-in per Ring instance via EnableParallel().
var parallelThreshold = 999

// workerPool is a persistent goroutine pool attached to a Ring.
// Instead of creating goroutines per ring operation (which causes
// scheduling overhead in tight loops), the pool pre-creates workers
// that wait on a shared channel.
type workerPool struct {
	tasks chan func()
	wg    sync.WaitGroup
	size  int
}

func newWorkerPool(size int) *workerPool {
	p := &workerPool{
		tasks: make(chan func(), size*2),
		size:  size,
	}
	for i := 0; i < size; i++ {
		go func() {
			for f := range p.tasks {
				f()
				p.wg.Done()
			}
		}()
	}
	return p
}

func (p *workerPool) dispatch(n int, f func(int)) {
	p.wg.Add(n)
	for i := 0; i < n; i++ {
		idx := i
		p.tasks <- func() { f(idx) }
	}
	p.wg.Wait()
}

func (p *workerPool) close() {
	close(p.tasks)
}

func init() {
	if runtime.GOMAXPROCS(0) < runtime.NumCPU() {
		runtime.GOMAXPROCS(runtime.NumCPU())
	}
}

// SetParallelThreshold sets the minimum number of SubRings for parallel dispatch.
func SetParallelThreshold(n int) {
	parallelThreshold = n
}

// parallelSubRings dispatches f(i, SubRing) with automatic parallelization.
// Uses a persistent worker pool if available, otherwise goroutines.
func (r Ring) parallelSubRings(f func(int, *SubRing)) {
	n := r.level + 1
	if n < parallelThreshold {
		for i := 0; i < n; i++ {
			f(i, r.SubRings[i])
		}
		return
	}

	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(idx int) {
			f(idx, r.SubRings[idx])
			wg.Done()
		}(i)
	}
	wg.Wait()
}
