package ring

import (
	"math/rand"
	"testing"
)

func newRing20(b *testing.B) *Ring {
	primes := Qi60[:20]
	r, err := NewRing(1<<16, primes)
	if err != nil {
		b.Fatal(err)
	}
	return r
}

func fillPoly(r *Ring, p Poly) {
	rng := rand.New(rand.NewSource(42))
	for i := 0; i <= r.level; i++ {
		q := r.SubRings[i].Modulus
		for j := range p.Coeffs[i] {
			p.Coeffs[i][j] = rng.Uint64() % q
		}
	}
}

func BenchmarkNTTSequential(b *testing.B) {
	r := newRing20(b)
	p1 := r.NewPoly()
	p2 := r.NewPoly()
	fillPoly(r, p1)

	old := parallelThreshold
	parallelThreshold = 999 // force sequential
	defer func() { parallelThreshold = old }()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.NTT(p1, p2)
	}
}

func BenchmarkNTTAutoParallel(b *testing.B) {
	r := newRing20(b)
	p1 := r.NewPoly()
	p2 := r.NewPoly()
	fillPoly(r, p1)

	old := parallelThreshold
	parallelThreshold = 1 // force parallel
	defer func() { parallelThreshold = old }()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.NTT(p1, p2)
	}
}

func BenchmarkAutomorphismSequential(b *testing.B) {
	r := newRing20(b)
	p1 := r.NewPoly()
	p2 := r.NewPoly()
	fillPoly(r, p1)
	index, _ := AutomorphismNTTIndex(r.N(), r.NthRoot(), 5)

	old := parallelThreshold
	parallelThreshold = 999
	defer func() { parallelThreshold = old }()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.AutomorphismNTTWithIndex(p1, index, p2)
	}
}

func BenchmarkAutomorphismAutoParallel(b *testing.B) {
	r := newRing20(b)
	p1 := r.NewPoly()
	p2 := r.NewPoly()
	fillPoly(r, p1)
	index, _ := AutomorphismNTTIndex(r.N(), r.NthRoot(), 5)

	old := parallelThreshold
	parallelThreshold = 1
	defer func() { parallelThreshold = old }()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r.AutomorphismNTTWithIndex(p1, index, p2)
	}
}
